"""Unit tests for the search module (Python side).

These tests verify the criticality detector, config, and search engine
logic without requiring the Rust libriichi library (which needs to be
built separately). Tests that need libriichi are marked and can be
skipped if it's not available.
"""

import numpy as np
import pytest

from .config import SearchConfig
from .criticality import (
    ACTION_SPACE,
    IDX_AGARI,
    IDX_PASS,
    IDX_RIICHI,
    _compute_entropy_factor,
    compute_criticality,
)
from .engine import SearchEngine, SearchStats


# ---- SearchConfig tests ----

class TestSearchConfig:
    def test_defaults(self):
        cfg = SearchConfig()
        assert cfg.no_search_threshold == 0.25
        assert cfg.light_search_threshold == 0.45
        assert cfg.deep_search_threshold == 0.65
        assert cfg.search_trust_weight == 0.7
        assert cfg.enabled is True

    def test_allocate_no_search(self):
        cfg = SearchConfig()
        budget, particles = cfg.allocate(0.10)
        assert budget == 0.0
        assert particles == 0

    def test_allocate_light(self):
        cfg = SearchConfig()
        budget, particles = cfg.allocate(0.30)
        assert budget == cfg.light_budget_ms
        assert particles == cfg.light_particles

    def test_allocate_standard(self):
        cfg = SearchConfig()
        budget, particles = cfg.allocate(0.50)
        assert budget == cfg.standard_budget_ms
        assert particles == cfg.standard_particles

    def test_allocate_deep(self):
        cfg = SearchConfig()
        budget, particles = cfg.allocate(0.80)
        assert budget == cfg.deep_budget_ms
        assert particles == cfg.deep_particles

    def test_allocate_disabled(self):
        cfg = SearchConfig(enabled=False)
        budget, particles = cfg.allocate(0.90)
        assert budget == 0.0
        assert particles == 0

    def test_allocate_boundary_values(self):
        cfg = SearchConfig()
        # Exactly at threshold = still in lower tier
        _, p1 = cfg.allocate(cfg.no_search_threshold)
        assert p1 == cfg.light_particles

        _, p2 = cfg.allocate(cfg.light_search_threshold)
        assert p2 == cfg.standard_particles

        _, p3 = cfg.allocate(cfg.deep_search_threshold)
        assert p3 == cfg.deep_particles


# ---- Entropy factor tests ----

class TestEntropyFactor:
    def test_single_action_zero_entropy(self):
        q = np.full(ACTION_SPACE, -1e30)
        q[0] = 1.0
        mask = np.zeros(ACTION_SPACE, dtype=bool)
        mask[0] = True
        result = _compute_entropy_factor(q, mask)
        assert result == 0.0

    def test_uniform_distribution_max_entropy(self):
        n_legal = 10
        q = np.full(ACTION_SPACE, -1e30)
        q[:n_legal] = 0.0  # Uniform q-values -> uniform softmax
        mask = np.zeros(ACTION_SPACE, dtype=bool)
        mask[:n_legal] = True

        result = _compute_entropy_factor(q, mask)
        # Should be close to 0.15 (max factor * normalized_entropy ~ 1.0)
        assert 0.14 <= result <= 0.15

    def test_peaked_distribution_low_entropy(self):
        q = np.full(ACTION_SPACE, -1e30)
        q[0] = 10.0
        q[1] = 0.0
        q[2] = 0.0
        mask = np.zeros(ACTION_SPACE, dtype=bool)
        mask[:3] = True

        result = _compute_entropy_factor(q, mask)
        # Peaked distribution -> low normalized entropy
        assert result < 0.05

    def test_no_mask_infers_from_q(self):
        q = np.full(ACTION_SPACE, -1e31)
        q[0] = 1.0
        q[1] = 0.5
        # With no explicit mask, should infer legal from finite values
        result = _compute_entropy_factor(q, None)
        assert 0.0 <= result <= 0.15


# ---- Criticality tests (with mock state) ----

class MockPlayerState:
    """Minimal mock of PlayerState for testing criticality."""
    def __init__(
        self,
        riichi_accepted=None,
        shanten=3,
        kyoku=0,
        scores=None,
        last_cans=None,
    ):
        self._riichi = riichi_accepted or [False, False, False, False]
        self._shanten = shanten
        self._kyoku = kyoku
        self._scores = scores or [25000, 25000, 25000, 25000]
        self._last_cans = last_cans or MockCans()

    def riichi_accepted(self):
        return self._riichi

    @property
    def shanten(self):
        return self._shanten

    @property
    def kyoku(self):
        return self._kyoku

    def scores(self):
        return self._scores

    @property
    def last_cans(self):
        return self._last_cans


class MockCans:
    def __init__(self, can_discard=True):
        self.can_discard = can_discard


class TestCriticality:
    def test_baseline_no_threats(self):
        state = MockPlayerState()
        crit = compute_criticality(state)
        # No threats: only shanten > 1 contributes nothing extra
        assert crit < 0.25  # Should be below search threshold

    def test_one_riichi_increases(self):
        state = MockPlayerState(riichi_accepted=[False, True, False, False])
        crit = compute_criticality(state)
        assert crit >= 0.25  # Riichi adds 0.25

    def test_two_riichi_increases_more(self):
        state = MockPlayerState(riichi_accepted=[False, True, True, False])
        crit = compute_criticality(state)
        assert crit >= 0.35  # Double riichi adds 0.35

    def test_tenpai_increases(self):
        state = MockPlayerState(shanten=0)
        crit = compute_criticality(state)
        assert crit >= 0.20

    def test_iishanten_increases(self):
        state = MockPlayerState(shanten=1)
        crit_1 = compute_criticality(state)
        state_far = MockPlayerState(shanten=3)
        crit_far = compute_criticality(state_far)
        assert crit_1 > crit_far

    def test_south_round_increases(self):
        east = MockPlayerState(kyoku=0)
        south = MockPlayerState(kyoku=4)
        assert compute_criticality(south) > compute_criticality(east)

    def test_all_last_increases(self):
        south3 = MockPlayerState(kyoku=6)
        south4 = MockPlayerState(kyoku=7)
        assert compute_criticality(south4) > compute_criticality(south3)

    def test_close_scores_increase(self):
        # 2nd place, close to 1st and 3rd
        close = MockPlayerState(scores=[26000, 27000, 25000, 22000])
        # 1st place by wide margin
        far = MockPlayerState(scores=[40000, 25000, 20000, 15000])
        assert compute_criticality(close) > compute_criticality(far)

    def test_danger_discard_increases(self):
        # Opponent in riichi, we need to discard, not tenpai
        danger = MockPlayerState(
            riichi_accepted=[False, True, False, False],
            shanten=2,
            last_cans=MockCans(can_discard=True),
        )
        # Same but no discard needed
        safe = MockPlayerState(
            riichi_accepted=[False, True, False, False],
            shanten=2,
            last_cans=MockCans(can_discard=False),
        )
        assert compute_criticality(danger) > compute_criticality(safe)

    def test_capped_at_one(self):
        # Stack all factors: multiple riichi + tenpai + south4 + close scores + danger
        state = MockPlayerState(
            riichi_accepted=[False, True, True, True],
            shanten=0,
            kyoku=7,
            scores=[25000, 25500, 24500, 25000],
            last_cans=MockCans(can_discard=True),
        )
        # Add q_values for entropy
        q = np.zeros(ACTION_SPACE)
        q[:13] = 0.0  # Uniform over 13 tiles
        mask = np.zeros(ACTION_SPACE, dtype=bool)
        mask[:13] = True

        crit = compute_criticality(state, q, mask)
        assert crit <= 1.0

    def test_entropy_integration(self):
        state = MockPlayerState()
        # Without q_values
        crit_no_q = compute_criticality(state)
        # With high-entropy q_values (many equally good actions)
        q = np.full(ACTION_SPACE, -1e30)
        q[:10] = 0.0
        mask = np.zeros(ACTION_SPACE, dtype=bool)
        mask[:10] = True
        crit_with_q = compute_criticality(state, q, mask)
        assert crit_with_q > crit_no_q


# ---- SearchEngine tests ----

class TestSearchEngine:
    def test_disabled_returns_none(self):
        cfg = SearchConfig(enabled=False)
        engine = SearchEngine(cfg)
        state = MockPlayerState()
        q = np.zeros(ACTION_SPACE)
        mask = np.ones(ACTION_SPACE, dtype=bool)
        result = engine.maybe_search(state, q, mask, player_id=0)
        assert result is None

    def test_low_criticality_returns_none(self):
        cfg = SearchConfig()
        engine = SearchEngine(cfg)
        # Boring state: no threats, far from tenpai, East round
        state = MockPlayerState(shanten=5, kyoku=0)
        q = np.full(ACTION_SPACE, -1e30)
        q[0] = 10.0  # One clearly best action
        mask = np.zeros(ACTION_SPACE, dtype=bool)
        mask[0] = True
        result = engine.maybe_search(state, q, mask, player_id=0)
        assert result is None

    def test_stats_tracking(self):
        stats = SearchStats()
        stats.record_criticality(0.5)
        stats.record_criticality(0.3)
        stats.record_skip()
        stats.record_search(100.0, 50, 0.5)
        stats.record_error()

        assert stats.total_decisions == 2
        assert stats.searches_skipped == 1
        assert stats.searches_performed == 1
        assert stats.search_errors == 1
        assert abs(stats.avg_criticality - 0.4) < 1e-6
        assert stats.avg_search_ms == 100.0
        assert stats.search_rate == 0.5

    def test_stats_summary(self):
        stats = SearchStats()
        stats.record_criticality(0.5)
        stats.record_search(150.0, 100, 0.5)
        s = stats.summary()
        assert "1 decisions" in s
        assert "1 searches" in s

    def test_get_candidates_includes_special_actions(self):
        cfg = SearchConfig(max_candidates=3)
        engine = SearchEngine(cfg)

        q = np.full(ACTION_SPACE, -10.0)
        # Top-3 by q-value are discard actions
        q[0] = 5.0
        q[1] = 4.0
        q[2] = 3.0
        # But riichi is also legal
        q[IDX_RIICHI] = 1.0
        mask = np.zeros(ACTION_SPACE, dtype=bool)
        mask[0] = True
        mask[1] = True
        mask[2] = True
        mask[IDX_RIICHI] = True

        candidates = engine._get_candidates(q, mask)
        # Should include top-3 discards AND riichi (special action)
        assert 0 in candidates
        assert 1 in candidates
        assert 2 in candidates
        assert IDX_RIICHI in candidates

    def test_get_candidates_respects_max(self):
        cfg = SearchConfig(max_candidates=5, min_prob_coverage=0.0)
        engine = SearchEngine(cfg)

        q = np.full(ACTION_SPACE, -10.0)
        for i in range(13):
            q[i] = float(13 - i)  # 13, 12, ..., 1
        mask = np.zeros(ACTION_SPACE, dtype=bool)
        mask[:13] = True

        candidates = engine._get_candidates(q, mask)
        # Should have at most max_candidates (5) discard actions
        # (no special actions legal, so no must-includes)
        assert len(candidates) <= 5 + 1  # Allow small overshoot from coverage

    def test_select_action_returns_best_candidate(self):
        cfg = SearchConfig()
        engine = SearchEngine(cfg)

        q = np.full(ACTION_SPACE, -10.0)
        q[3] = 5.0
        q[7] = 8.0
        q[1] = 3.0
        mask = np.zeros(ACTION_SPACE, dtype=bool)
        mask[1] = True
        mask[3] = True
        mask[7] = True

        action = engine._select_action(q, mask, [1, 3, 7], 0.0, 0.0)
        assert action == 7  # Highest q-value


# ---- Integration test (requires libriichi) ----

def test_integration_with_libriichi():
    """Full integration test: create a game state, run search.

    Requires libriichi to be built and importable.
    """
    try:
        from libriichi.state import PlayerState
        from libriichi.mjai import Event
    except ImportError:
        pytest.skip("libriichi not available")

    # Set up a basic game state
    state = PlayerState(0)
    # We need to use the Rust Event type, construct via JSON
    import json

    events = [
        json.dumps({
            "type": "start_kyoku",
            "bakaze": "E",
            "dora_marker": "1m",
            "kyoku": 1,
            "honba": 0,
            "kyotaku": 0,
            "oya": 0,
            "scores": [25000, 25000, 25000, 25000],
            "tehais": [
                ["1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "1p", "2p", "3p", "4p"],
                ["?", "?", "?", "?", "?", "?", "?", "?", "?", "?", "?", "?", "?"],
                ["?", "?", "?", "?", "?", "?", "?", "?", "?", "?", "?", "?", "?"],
                ["?", "?", "?", "?", "?", "?", "?", "?", "?", "?", "?", "?", "?"],
            ],
        }),
        json.dumps({"type": "tsumo", "actor": 0, "pai": "5p"}),
    ]

    for ev_json in events:
        ev = Event.from_json(ev_json)
        state.update(ev)

    # Run search
    cfg = SearchConfig(seed=42, light_particles=5, standard_particles=10, deep_particles=20)
    engine = SearchEngine(cfg)

    q = np.zeros(ACTION_SPACE)
    # Make a few discard actions look reasonable
    q[0] = 2.0  # 1m
    q[4] = 3.0  # 5m
    q[13] = 1.0  # 5p (the tsumo)
    mask = np.zeros(ACTION_SPACE, dtype=bool)
    mask[0] = True
    mask[4] = True
    mask[13] = True

    # Force high criticality for testing
    cfg_high = SearchConfig(
        seed=42,
        no_search_threshold=0.0,
        light_particles=5,
    )
    engine_high = SearchEngine(cfg_high)
    result = engine_high.maybe_search(state, q, mask, player_id=0)

    # Should return an action (one of the legal ones)
    if result is not None:
        assert result in [0, 4, 13]

    # Stats should reflect the search
    assert engine_high.stats.total_decisions == 1
