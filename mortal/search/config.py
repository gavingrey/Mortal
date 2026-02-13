# DEPRECATED: Search configuration is now handled in Rust-side
# SearchIntegration (libriichi/src/agent/mortal.rs). Configure via
# MortalEngine parameters (search_particles, search_weight, etc.).
from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass
class SearchConfig:
    """Configuration for the SMCPG-OA search algorithm.

    Phase 1 simplifications:
    - Tsumogiri rollouts (no policy network evaluation during rollouts)
    - Uniform particle weights
    - Fixed search depth (rollout to completion)
    - No opponent modeling
    """

    # --- Criticality thresholds ---
    # Below no_search_threshold: skip search entirely, use pure policy
    no_search_threshold: float = 0.25
    # Between no_search and light_search: light search budget
    light_search_threshold: float = 0.45
    # Above deep_search_threshold: deep search budget
    deep_search_threshold: float = 0.65

    # --- Particle counts per tier ---
    light_particles: int = 50
    standard_particles: int = 100
    deep_particles: int = 200

    # --- Time budgets (ms) per tier ---
    light_budget_ms: float = 200.0
    standard_budget_ms: float = 500.0
    deep_budget_ms: float = 1500.0

    # --- Blending ---
    # Weight for search values vs policy prior when combining.
    # 0.0 = pure policy, 1.0 = pure search values.
    search_trust_weight: float = 0.5

    # --- Action pruning ---
    # Maximum number of candidate actions to evaluate during search.
    max_candidates: int = 10
    # Minimum cumulative policy probability to cover with candidates.
    min_prob_coverage: float = 0.95

    # --- Rollout settings (Phase 1: fixed) ---
    # Seed for reproducibility (None = random seed each time).
    seed: Optional[int] = None

    # Whether search is enabled at all.
    enabled: bool = True

    def allocate(self, criticality: float) -> Tuple[float, int]:
        """Return (budget_ms, n_particles) based on criticality score."""
        if not self.enabled or criticality < self.no_search_threshold:
            return 0.0, 0
        elif criticality < self.light_search_threshold:
            return self.light_budget_ms, self.light_particles
        elif criticality < self.deep_search_threshold:
            return self.standard_budget_ms, self.standard_particles
        else:
            return self.deep_budget_ms, self.deep_particles
