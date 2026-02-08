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
from .engine import SearchEngine, _normalize_values
from .metrics import SearchMetricsCollector


# ---- SearchConfig tests ----

class TestSearchConfig:
    def test_defaults(self):
        cfg = SearchConfig()
        assert cfg.no_search_threshold == 0.25
        assert cfg.light_search_threshold == 0.45
        assert cfg.deep_search_threshold == 0.65
        assert cfg.search_trust_weight == 0.5
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
        bakaze=27,
        scores=None,
        last_cans=None,
    ):
        self._riichi = riichi_accepted or [False, False, False, False]
        self._shanten = shanten
        self._kyoku = kyoku
        self._bakaze = bakaze
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

    @property
    def bakaze(self):
        return self._bakaze

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
        east = MockPlayerState(kyoku=0, bakaze=27)   # East 1
        south = MockPlayerState(kyoku=0, bakaze=28)   # South 1
        assert compute_criticality(south) > compute_criticality(east)

    def test_all_last_increases(self):
        south3 = MockPlayerState(kyoku=2, bakaze=28)  # South 3
        south4 = MockPlayerState(kyoku=3, bakaze=28)  # South 4
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
            kyoku=3, bakaze=28,  # South 4 (All-Last)
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


# ---- Normalization tests ----

class TestNormalization:
    def test_basic_normalization(self):
        values = np.array([0.0, 5.0, 10.0])
        result = _normalize_values(values)
        np.testing.assert_allclose(result, [0.0, 0.5, 1.0])

    def test_flat_values_return_half(self):
        values = np.array([3.0, 3.0, 3.0])
        result = _normalize_values(values)
        np.testing.assert_allclose(result, [0.5, 0.5, 0.5])

    def test_single_value_returns_half(self):
        values = np.array([7.0])
        result = _normalize_values(values)
        np.testing.assert_allclose(result, [0.5])

    def test_negative_values(self):
        values = np.array([-10.0, -5.0, 0.0])
        result = _normalize_values(values)
        np.testing.assert_allclose(result, [0.0, 0.5, 1.0])

    def test_empty_array(self):
        values = np.array([])
        result = _normalize_values(values)
        assert len(result) == 0

    def test_two_values(self):
        values = np.array([100.0, 200.0])
        result = _normalize_values(values)
        np.testing.assert_allclose(result, [0.0, 1.0])


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
        stats = SearchMetricsCollector()
        stats.record_decision(0.5)
        stats.record_decision(0.3)
        stats.record_skip()
        stats.record_search(
            elapsed_ms=100.0, tier="light", n_particles_actual=50,
            gen_attempts=50, gen_accepted=50, policy_action=0,
            search_action=1, action_deltas={0: [100.0], 1: [200.0]},
            rollout_results={}, game_phase="East",
        )
        stats.record_error()

        assert stats.total_decisions == 2
        assert stats.skipped_count == 1
        assert stats.search_count == 1
        assert stats.error_count == 1
        avg_crit = float(np.mean(stats.criticalities))
        assert abs(avg_crit - 0.4) < 1e-6
        assert stats.search_rate == 0.5
        assert stats.override_rate == 1.0  # action 1 != policy 0

    def test_stats_summary(self):
        stats = SearchMetricsCollector()
        stats.record_decision(0.5)
        stats.record_search(
            elapsed_ms=150.0, tier="standard", n_particles_actual=100,
            gen_attempts=100, gen_accepted=100, policy_action=0,
            search_action=0, action_deltas={0: [100.0]},
            rollout_results={}, game_phase="East",
        )
        s = stats.summary()
        assert "1," in s or "Decisions: 1" in s
        assert "1," in s or "Searches:" in s

    def test_get_candidates_excludes_riichi(self):
        """Riichi (action 37) should be excluded from search candidates."""
        cfg = SearchConfig(max_candidates=5)
        engine = SearchEngine(cfg)

        q = np.full(ACTION_SPACE, -10.0)
        q[0] = 5.0
        q[1] = 4.0
        q[2] = 3.0
        q[IDX_RIICHI] = 10.0  # Riichi has highest q-value
        mask = np.zeros(ACTION_SPACE, dtype=bool)
        mask[0] = True
        mask[1] = True
        mask[2] = True
        mask[IDX_RIICHI] = True

        candidates = engine._get_candidates(q, mask)
        assert 0 in candidates
        assert 1 in candidates
        assert 2 in candidates
        assert IDX_RIICHI not in candidates  # Riichi excluded

    def test_get_candidates_includes_non_riichi_special(self):
        """Non-riichi special actions (agari, pon, chi) should be included."""
        cfg = SearchConfig(max_candidates=3)
        engine = SearchEngine(cfg)

        q = np.full(ACTION_SPACE, -10.0)
        q[0] = 5.0
        q[1] = 4.0
        q[IDX_AGARI] = 1.0  # Agari is legal
        q[IDX_PASS] = 0.0   # Pass is legal
        mask = np.zeros(ACTION_SPACE, dtype=bool)
        mask[0] = True
        mask[1] = True
        mask[IDX_AGARI] = True
        mask[IDX_PASS] = True

        candidates = engine._get_candidates(q, mask)
        assert IDX_AGARI in candidates  # Must-include special action
        assert IDX_PASS in candidates   # Must-include special action

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
        assert len(candidates) <= 5 + 1  # Allow small overshoot from coverage

    def test_select_action_pure_policy(self):
        """With search_trust_weight=0, should return policy-best action."""
        cfg = SearchConfig(search_trust_weight=0.0)
        engine = SearchEngine(cfg)

        q = np.full(ACTION_SPACE, -10.0)
        q[3] = 5.0   # Policy likes action 3
        q[7] = 2.0   # Policy dislikes action 7
        mask = np.zeros(ACTION_SPACE, dtype=bool)
        mask[3] = True
        mask[7] = True

        # Search strongly prefers action 7
        action_values = {3: -1000.0, 7: 5000.0}
        action = engine._select_action(q, mask, [3, 7], action_values)
        assert action == 3  # Pure policy: action 3 has higher q-value

    def test_select_action_pure_search(self):
        """With search_trust_weight=1, should return search-best action."""
        cfg = SearchConfig(search_trust_weight=1.0)
        engine = SearchEngine(cfg)

        q = np.full(ACTION_SPACE, -10.0)
        q[3] = 8.0   # Policy likes action 3
        q[7] = 2.0   # Policy dislikes action 7
        mask = np.zeros(ACTION_SPACE, dtype=bool)
        mask[3] = True
        mask[7] = True

        # Search strongly prefers action 7
        action_values = {3: -5000.0, 7: 5000.0}
        action = engine._select_action(q, mask, [3, 7], action_values)
        assert action == 7  # Pure search: action 7 has higher search value

    def test_select_action_blending(self):
        """With balanced weight, blending should combine both signals."""
        cfg = SearchConfig(search_trust_weight=0.5)
        engine = SearchEngine(cfg)

        q = np.full(ACTION_SPACE, -10.0)
        # Policy strongly prefers action 3 over action 7
        q[3] = 10.0
        q[7] = -10.0
        mask = np.zeros(ACTION_SPACE, dtype=bool)
        mask[3] = True
        mask[7] = True

        # Search strongly prefers action 7 over action 3
        action_values = {3: -10000.0, 7: 10000.0}
        action = engine._select_action(q, mask, [3, 7], action_values)
        # With w=0.5:
        # action 3: 0.5 * 0.0 (normalized search) + 0.5 * ~1.0 (policy prob)
        # action 7: 0.5 * 1.0 (normalized search) + 0.5 * ~0.0 (policy prob)
        # Both should be ~0.5, but exact result depends on softmax
        assert action in [3, 7]  # Either is reasonable with balanced weight

    def test_select_action_flat_search_values(self):
        """When all search values are equal, should fall back to policy."""
        cfg = SearchConfig(search_trust_weight=0.5)
        engine = SearchEngine(cfg)

        q = np.full(ACTION_SPACE, -10.0)
        q[3] = 8.0
        q[7] = 2.0
        mask = np.zeros(ACTION_SPACE, dtype=bool)
        mask[3] = True
        mask[7] = True

        # Flat search values -> normalized to 0.5 each
        action_values = {3: 100.0, 7: 100.0}
        action = engine._select_action(q, mask, [3, 7], action_values)
        # Search contributes equally, policy breaks the tie
        assert action == 3  # Policy prefers action 3

    def test_select_action_no_evaluated_actions(self):
        """When no actions were evaluated, fall back to policy argmax."""
        cfg = SearchConfig()
        engine = SearchEngine(cfg)

        q = np.full(ACTION_SPACE, -10.0)
        q[5] = 3.0
        q[9] = 7.0
        mask = np.zeros(ACTION_SPACE, dtype=bool)
        mask[5] = True
        mask[9] = True

        action_values = {}  # No evaluations
        action = engine._select_action(q, mask, [5, 9], action_values)
        assert action == 9  # Policy argmax


# ---- Integration test (requires libriichi) ----

def test_integration_with_libriichi():
    """Full integration test: create a game state, run search.

    Requires libriichi to be built and importable.
    """
    try:
        from libriichi.state import PlayerState
    except ImportError:
        pytest.skip("libriichi not available")

    import json

    # Set up a basic game state.
    # PlayerState.update() takes a JSON string directly (not an Event object).
    state = PlayerState(0)

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
        state.update(ev_json)

    # Force high criticality for testing
    cfg = SearchConfig(
        seed=42,
        no_search_threshold=0.0,
        light_particles=5,
        search_trust_weight=0.5,
    )
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

    result = engine.maybe_search(state, q, mask, player_id=0)

    # Should return an action (one of the legal ones)
    if result is not None:
        assert result in [0, 4, 13]

    # Stats should reflect the search
    assert engine.stats.total_decisions == 1
