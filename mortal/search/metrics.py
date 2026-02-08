"""Search metrics collection for monitoring and analysis."""

import numpy as np
from typing import Dict, List, Optional
from collections import defaultdict


class SearchMetricsCollector:
    """Collects comprehensive metrics during search-enhanced evaluation."""

    def __init__(self):
        # 1. Criticality distribution
        self.criticalities: List[float] = []

        # 2-3. Override and agreement tracking
        self.search_count = 0
        self.override_count = 0  # search picked different action than policy argmax
        self.agreement_count = 0  # search agreed with policy top choice

        # 4. Search delta magnitude (how different search values are from policy)
        self.delta_magnitudes: List[float] = []

        # 5. Rollout lengths (steps per rollout)
        self.rollout_steps: List[int] = []

        # 6. Hora rate
        self.rollout_count = 0
        self.hora_count = 0

        # 7. Action variance (std dev of deltas across particles, per search)
        self.action_variances: List[float] = []

        # 8. Time per search by tier
        self.times_by_tier: Dict[str, List[float]] = defaultdict(list)  # tier -> [ms, ms, ...]

        # 9. Particle generation
        self.gen_attempts: List[int] = []
        self.gen_accepted: List[int] = []

        # 10. Particles per search (actual)
        self.particles_per_search: List[int] = []

        # 11. Search impact by game phase
        self.phase_searches: Dict[str, int] = defaultdict(int)  # "East"/"South" -> count
        self.phase_overrides: Dict[str, int] = defaultdict(int)  # "East"/"South" -> override count

        # Total decisions (including skipped)
        self.total_decisions = 0
        self.skipped_count = 0
        self.error_count = 0

    def record_decision(self, criticality: float):
        """Record every decision point (called before search/skip decision)."""
        self.total_decisions += 1
        self.criticalities.append(criticality)

    def record_skip(self):
        """Record a skipped search (criticality too low)."""
        self.skipped_count += 1

    def record_error(self):
        """Record a search error."""
        self.error_count += 1

    def record_search(
        self,
        elapsed_ms: float,
        tier: str,  # "light", "standard", "deep"
        n_particles_actual: int,
        gen_attempts: int,
        gen_accepted: int,
        policy_action: int,  # argmax of policy
        search_action: Optional[int],  # what search returned
        action_deltas: Dict[int, List[float]],  # action -> list of deltas across particles
        rollout_results,  # dict of action -> list of RolloutResult (from Rust)
        game_phase: str,  # "East" or "South"
    ):
        """Record a completed search with all associated metrics."""
        self.search_count += 1

        # 8. Time by tier
        self.times_by_tier[tier].append(elapsed_ms)

        # 9. Particle gen stats
        self.gen_attempts.append(gen_attempts)
        self.gen_accepted.append(gen_accepted)

        # 10. Particles per search
        self.particles_per_search.append(n_particles_actual)

        # 2-3. Override/agreement
        if search_action is not None:
            if search_action != policy_action:
                self.override_count += 1
                self.phase_overrides[game_phase] += 1
            else:
                self.agreement_count += 1

        # 11. Phase tracking
        self.phase_searches[game_phase] += 1

        # 7. Action variance - compute mean variance across evaluated actions
        variances = []
        for action, deltas in action_deltas.items():
            if len(deltas) > 1:
                variances.append(float(np.std(deltas)))
        if variances:
            self.action_variances.append(float(np.mean(variances)))

        # 4. Delta magnitude - max spread between action values
        if action_deltas:
            means = [float(np.mean(d)) for d in action_deltas.values()]
            if len(means) > 1:
                self.delta_magnitudes.append(max(means) - min(means))

        # 5-6. Rollout-level metrics from results
        if rollout_results:
            for action, results in rollout_results.items():
                for r in results:
                    self.rollout_count += 1
                    if hasattr(r, 'steps'):
                        self.rollout_steps.append(r.steps)
                    if hasattr(r, 'has_hora') and r.has_hora:
                        self.hora_count += 1

    @property
    def search_rate(self) -> float:
        return self.search_count / self.total_decisions if self.total_decisions > 0 else 0.0

    @property
    def override_rate(self) -> float:
        return self.override_count / self.search_count if self.search_count > 0 else 0.0

    @property
    def agreement_rate(self) -> float:
        return self.agreement_count / self.search_count if self.search_count > 0 else 0.0

    @property
    def avg_rollout_steps(self) -> float:
        return float(np.mean(self.rollout_steps)) if self.rollout_steps else 0.0

    @property
    def hora_rate(self) -> float:
        return self.hora_count / self.rollout_count if self.rollout_count > 0 else 0.0

    @property
    def avg_action_variance(self) -> float:
        return float(np.mean(self.action_variances)) if self.action_variances else 0.0

    @property
    def particle_gen_success_rate(self) -> float:
        if not self.gen_attempts:
            return 0.0
        total_attempts = sum(self.gen_attempts)
        total_accepted = sum(self.gen_accepted)
        return total_accepted / total_attempts if total_attempts > 0 else 0.0

    def _criticality_histogram(self) -> str:
        """Return a simple text histogram of criticality distribution."""
        if not self.criticalities:
            return "  (no data)"
        bins = [0, 0.25, 0.45, 0.65, 1.01]
        labels = ["<0.25 (skip)", "0.25-0.45 (light)", "0.45-0.65 (std)", "0.65+ (deep)"]
        counts = [0] * 4
        for c in self.criticalities:
            for i in range(4):
                if c < bins[i + 1]:
                    counts[i] += 1
                    break
        total = len(self.criticalities)
        lines = []
        for label, count in zip(labels, counts):
            pct = count / total * 100 if total > 0 else 0
            bar = "#" * int(pct / 2)
            lines.append(f"  {label:25s} {count:5d} ({pct:5.1f}%) {bar}")
        return "\n".join(lines)

    def _time_stats_by_tier(self) -> str:
        """Format timing stats by tier."""
        lines = []
        for tier in ["light", "standard", "deep"]:
            times = self.times_by_tier.get(tier, [])
            if times:
                lines.append(
                    f"  {tier:10s}: n={len(times):4d}, "
                    f"avg={np.mean(times):7.1f}ms, "
                    f"p50={np.median(times):7.1f}ms, "
                    f"p95={np.percentile(times, 95):7.1f}ms, "
                    f"max={max(times):7.1f}ms"
                )
        return "\n".join(lines) if lines else "  (no searches performed)"

    def summary(self) -> str:
        """Return a formatted multi-line summary of all metrics."""
        lines = []
        lines.append("=" * 64)
        lines.append(" SEARCH METRICS SUMMARY")
        lines.append("=" * 64)

        # Overview
        lines.append(f"\nDecisions: {self.total_decisions:,}")
        lines.append(f"  Searches:  {self.search_count:,} ({self.search_rate:.1%})")
        lines.append(f"  Skipped:   {self.skipped_count:,}")
        lines.append(f"  Errors:    {self.error_count:,}")

        # Criticality distribution
        lines.append(f"\nCriticality Distribution:")
        lines.append(self._criticality_histogram())
        if self.criticalities:
            lines.append(f"  avg={np.mean(self.criticalities):.3f}, "
                        f"max={max(self.criticalities):.3f}")

        # Override/agreement
        if self.search_count > 0:
            lines.append(f"\nSearch vs Policy:")
            lines.append(f"  Override rate:  {self.override_rate:.1%} ({self.override_count}/{self.search_count})")
            lines.append(f"  Agreement rate: {self.agreement_rate:.1%} ({self.agreement_count}/{self.search_count})")

        # Delta magnitude
        if self.delta_magnitudes:
            lines.append(f"\nSearch Delta Magnitude (max spread between action values):")
            lines.append(f"  avg={np.mean(self.delta_magnitudes):.0f}, "
                        f"p50={np.median(self.delta_magnitudes):.0f}, "
                        f"max={max(self.delta_magnitudes):.0f}")

        # Rollout metrics
        if self.rollout_count > 0:
            lines.append(f"\nRollout Metrics ({self.rollout_count:,} total rollouts):")
            if self.rollout_steps:
                lines.append(f"  Avg steps:  {self.avg_rollout_steps:.1f}")
            lines.append(f"  Hora rate:  {self.hora_rate:.1%}")

        # Action variance
        if self.action_variances:
            lines.append(f"\nAction Variance (avg std dev of deltas across particles):")
            lines.append(f"  avg={self.avg_action_variance:.0f}, "
                        f"p50={np.median(self.action_variances):.0f}, "
                        f"max={max(self.action_variances):.0f}")

        # Timing
        lines.append(f"\nTiming by Tier:")
        lines.append(self._time_stats_by_tier())

        # Particle generation
        if self.gen_attempts:
            lines.append(f"\nParticle Generation:")
            lines.append(f"  Success rate: {self.particle_gen_success_rate:.1%}")
            lines.append(f"  Avg particles/search: {np.mean(self.particles_per_search):.1f}")

        # Phase breakdown
        if self.phase_searches:
            lines.append(f"\nSearch by Game Phase:")
            for phase in sorted(self.phase_searches.keys()):
                count = self.phase_searches[phase]
                overrides = self.phase_overrides.get(phase, 0)
                override_pct = overrides / count * 100 if count > 0 else 0
                lines.append(f"  {phase}: {count} searches, {overrides} overrides ({override_pct:.1f}%)")

        lines.append("=" * 64)
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Return metrics as a dictionary for programmatic access."""
        return {
            "total_decisions": self.total_decisions,
            "search_count": self.search_count,
            "skipped_count": self.skipped_count,
            "error_count": self.error_count,
            "search_rate": self.search_rate,
            "override_rate": self.override_rate,
            "agreement_rate": self.agreement_rate,
            "avg_criticality": float(np.mean(self.criticalities)) if self.criticalities else 0.0,
            "max_criticality": max(self.criticalities) if self.criticalities else 0.0,
            "avg_rollout_steps": self.avg_rollout_steps,
            "hora_rate": self.hora_rate,
            "avg_action_variance": self.avg_action_variance,
            "avg_delta_magnitude": float(np.mean(self.delta_magnitudes)) if self.delta_magnitudes else 0.0,
            "particle_gen_success_rate": self.particle_gen_success_rate,
            "avg_particles_per_search": float(np.mean(self.particles_per_search)) if self.particles_per_search else 0.0,
            "times_by_tier": {k: {"count": len(v), "avg_ms": float(np.mean(v))} for k, v in self.times_by_tier.items()},
            "phase_searches": dict(self.phase_searches),
            "phase_overrides": dict(self.phase_overrides),
        }
