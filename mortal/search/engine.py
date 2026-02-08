import time
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple

from .config import SearchConfig
from .criticality import compute_criticality, ACTION_SPACE, IDX_RIICHI
from .metrics import SearchMetricsCollector

log = logging.getLogger(__name__)


class SearchEngine:
    """Orchestrates search-enhanced decision making for Mortal.

    Phase 2 implementation:
    - Per-action rollouts: each candidate action is evaluated separately
    - Value blending: combines search values with policy probabilities
    - Criticality-based conditional activation (skip search for easy decisions)
    - Tsumogiri rollout policy (agents always discard drawn tile after our move)

    Usage:
        search_engine = SearchEngine(config)

        # In the decision loop:
        action = search_engine.maybe_search(
            state=player_state,
            q_values=q_out_for_this_player,
            masks=action_mask,
            player_id=player_id,
        )
        # action is None if search was skipped (use policy action)
        # action is int if search recommends a specific action
    """

    def __init__(self, config: Optional[SearchConfig] = None):
        self.config = config or SearchConfig()
        self._search_module = None
        self._stats = SearchMetricsCollector()

    def _ensure_module(self):
        """Lazily initialize the Rust SearchModule."""
        if self._search_module is not None:
            return

        from libriichi.search import SearchModule
        if self.config.seed is not None:
            self._search_module = SearchModule.with_seed(
                self.config.deep_particles,  # Max particles we might need
                self.config.seed,
            )
        else:
            self._search_module = SearchModule(self.config.deep_particles)

    def maybe_search(
        self,
        state,
        q_values: np.ndarray,
        masks: np.ndarray,
        player_id: int,
    ) -> Optional[int]:
        """Run search if the decision is critical enough.

        Args:
            state: PlayerState from libriichi.
            q_values: Shape (ACTION_SPACE,) q-values from DQN.
            masks: Shape (ACTION_SPACE,) boolean legal action mask.
            player_id: Absolute seat of the player (for score deltas).

        Returns:
            Action index (int) if search was performed and recommends an action.
            None if search was skipped (caller should use pure policy).
        """
        if not self.config.enabled:
            return None

        # Step 1: Compute criticality
        crit = compute_criticality(state, q_values, masks)
        self._stats.record_decision(crit)

        # Step 2: Allocate budget
        budget_ms, n_particles = self.config.allocate(crit)
        if n_particles == 0:
            self._stats.record_skip()
            return None

        # Determine search tier
        if n_particles <= self.config.light_particles:
            tier = "light"
        elif n_particles <= self.config.standard_particles:
            tier = "standard"
        else:
            tier = "deep"

        # Step 3: Run search
        self._ensure_module()
        start = time.monotonic()

        try:
            action, search_data = self._run_search(
                state=state,
                q_values=q_values,
                masks=masks,
                player_id=player_id,
                n_particles=n_particles,
                budget_ms=budget_ms,
            )
        except Exception:
            log.exception("Search failed, falling back to policy")
            self._stats.record_error()
            return None

        elapsed_ms = (time.monotonic() - start) * 1000

        # Determine game phase from bakaze
        try:
            bakaze_u8 = state.bakaze
            phase = {27: "East", 28: "South", 29: "West", 30: "North"}.get(bakaze_u8, "Unknown")
        except Exception:
            phase = "Unknown"

        self._stats.record_search(
            elapsed_ms=elapsed_ms,
            tier=tier,
            n_particles_actual=search_data.get("n_particles", 0),
            gen_attempts=self._search_module.last_gen_attempts,
            gen_accepted=self._search_module.last_gen_accepted,
            policy_action=search_data.get("policy_action", -1),
            search_action=action,
            action_deltas=search_data.get("action_deltas", {}),
            rollout_results=search_data.get("rollout_results", {}),
            game_phase=phase,
        )

        return action

    def _run_search(
        self,
        state,
        q_values: np.ndarray,
        masks: np.ndarray,
        player_id: int,
        n_particles: int,
        budget_ms: float,
    ) -> Tuple[Optional[int], Dict]:
        """Core search logic: generate particles, simulate per-action, blend.

        For each candidate action, runs rollouts across particles to get
        expected score deltas. Then blends with policy probabilities.

        Returns (action_index, search_data) where action_index is None if
        search is inconclusive, and search_data contains intermediate metrics.
        """
        from libriichi.search import ParticleConfig

        # Compute policy action (argmax over legal actions) for override tracking
        q = np.asarray(q_values, dtype=np.float64)
        mask = np.asarray(masks, dtype=bool)
        policy_action = int(np.argmax(np.where(mask, q, -np.inf)))

        # Configure particle count
        self._search_module.config = ParticleConfig(n_particles)

        # Generate particles
        particles = self._search_module.generate_particles(state)
        if len(particles) == 0:
            log.warning("No particles generated, falling back to policy")
            return (None, {})

        n_particles_actual = len(particles)

        # Get candidate actions to evaluate
        candidates = self._get_candidates(q_values, masks)
        if len(candidates) <= 1:
            action = candidates[0] if candidates else None
            return (action, {"n_particles": n_particles_actual, "policy_action": policy_action})

        # Use batch API: replays event history once, derives context once
        start = time.monotonic()

        results = self._search_module.evaluate_actions(state, particles, candidates)

        # Convert results to action_deltas format
        action_deltas: Dict[int, List[float]] = {}
        for action, rollout_results in results.items():
            deltas = []
            for result in rollout_results:
                try:
                    delta = result.player_delta(player_id)
                    deltas.append(float(delta))
                except Exception:
                    continue
            if deltas:
                action_deltas[action] = deltas

        if len(action_deltas) == 0:
            log.warning("All per-action rollouts failed, falling back to policy")
            return (None, {"n_particles": n_particles_actual, "policy_action": policy_action,
                           "rollout_results": results})

        # Compute mean delta per action
        action_values = {
            a: float(np.mean(deltas)) for a, deltas in action_deltas.items()
        }

        selected = self._select_action(q_values, masks, candidates, action_values)
        search_data = {
            "n_particles": n_particles_actual,
            "action_deltas": action_deltas,
            "rollout_results": results,
            "policy_action": policy_action,
        }
        return (selected, search_data)

    def _get_candidates(self, q_values: np.ndarray, masks: np.ndarray) -> List[int]:
        """Get candidate actions to evaluate, pruned by policy.

        Skips riichi (action 37) from search candidates per team lead decision.
        """
        q = np.asarray(q_values, dtype=np.float64)
        mask = np.asarray(masks, dtype=bool)

        # All legal actions
        legal = np.where(mask)[0]
        if len(legal) == 0:
            return []

        # Always include special actions (non-discard), except riichi
        must_include = set()
        for a in legal:
            if a >= 37 and a != IDX_RIICHI:
                must_include.add(int(a))

        # Sort by q-value (descending) and take top-k
        sorted_actions = legal[np.argsort(-q[legal])]
        candidates = set()

        for a in sorted_actions:
            a_int = int(a)
            if a_int == IDX_RIICHI:
                continue  # Skip riichi from search candidates
            candidates.add(a_int)
            if len(candidates) >= self.config.max_candidates:
                break

        # Ensure must-includes
        candidates.update(must_include)

        # Ensure minimum probability coverage
        legal_q = q[legal]
        legal_q = legal_q - legal_q.max()
        probs = np.exp(legal_q) / np.exp(legal_q).sum()

        action_to_prob_idx = {int(a): i for i, a in enumerate(legal)}

        total_prob = sum(probs[action_to_prob_idx[a]]
                         for a in candidates if a in action_to_prob_idx)
        if total_prob < self.config.min_prob_coverage:
            for a in sorted_actions:
                a_int = int(a)
                if a_int == IDX_RIICHI:
                    continue
                if a_int not in candidates:
                    candidates.add(a_int)
                    total_prob += probs[action_to_prob_idx[a_int]]
                    if total_prob >= self.config.min_prob_coverage:
                        break

        return sorted(candidates)

    def _select_action(
        self,
        q_values: np.ndarray,
        masks: np.ndarray,
        candidates: List[int],
        action_values: Dict[int, float],
    ) -> Optional[int]:
        """Blend search values with policy for final action selection.

        Formula:
            blended[a] = w * normalize(V_search(a)) + (1-w) * pi_policy(a)
        where:
            V_search(a) = mean delta for action a across particles
            normalize(x) = (x - min) / (max - min) over evaluated actions
            pi_policy(a) = softmax(q_values)[a] over legal actions
            w = config.search_trust_weight
        """
        q = np.asarray(q_values, dtype=np.float64)
        mask = np.asarray(masks, dtype=bool)
        w = self.config.search_trust_weight

        # Compute policy probabilities via softmax over legal actions
        legal_mask = mask.copy()
        legal_q = np.where(legal_mask, q, -np.inf)
        max_legal_q = np.max(legal_q[legal_mask])
        legal_q_shifted = legal_q - max_legal_q
        probs = np.zeros(ACTION_SPACE, dtype=np.float64)
        probs[legal_mask] = np.exp(legal_q_shifted[legal_mask])
        prob_sum = probs.sum()
        if prob_sum > 0:
            probs /= prob_sum

        # Normalize search values to [0, 1]
        evaluated = [a for a in candidates if a in action_values]
        if len(evaluated) == 0:
            # No actions evaluated, fall back to policy argmax
            best_action = None
            best_q = -np.inf
            for a in candidates:
                if mask[a] and q[a] > best_q:
                    best_q = q[a]
                    best_action = a
            return best_action

        search_vals = np.array([action_values[a] for a in evaluated])
        normalized = _normalize_values(search_vals)

        # Blend: w * normalized_search + (1-w) * policy_prob
        best_action = None
        best_score = -np.inf
        for i, a in enumerate(evaluated):
            if not mask[a]:
                continue
            blended = w * normalized[i] + (1.0 - w) * probs[a]
            if blended > best_score:
                best_score = blended
                best_action = a

        return best_action

    @property
    def stats(self) -> SearchMetricsCollector:
        """Access search performance statistics."""
        return self._stats

    def reset_stats(self):
        """Reset accumulated statistics."""
        self._stats = SearchMetricsCollector()


def _normalize_values(values: np.ndarray) -> np.ndarray:
    """Normalize values to [0, 1] range using min-max scaling.

    If all values are identical, returns uniform 0.5 for all.
    """
    if len(values) == 0:
        return values
    vmin = values.min()
    vmax = values.max()
    spread = vmax - vmin
    if spread < 1e-10:
        return np.full_like(values, 0.5)
    return (values - vmin) / spread
