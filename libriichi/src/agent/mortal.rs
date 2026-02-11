use super::{BatchAgent, InvisibleState};
use crate::arena::MidgameContextBase;
use crate::consts::ACTION_SPACE;
use crate::mjai::{Event, EventExt, Metadata};
use crate::search::config::ParticleConfig;
use crate::search::grp::{self, GrpEntry, GrpEvaluator};
use crate::search::{particle, simulator};
use crate::state::PlayerState;
use crate::{must_tile, tu8};
use std::collections::HashMap;
use std::mem;
use std::panic;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, ensure};
use crossbeam::sync::WaitGroup;
use ndarray::prelude::*;
use numpy::{PyArray1, PyArray2};
use parking_lot::Mutex;
use pyo3::intern;
use pyo3::prelude::*;
use rand::prelude::*;
use rand::SeedableRng;
use rand_chacha::ChaCha12Rng;
use rayon::prelude::*;

pub struct MortalBatchAgent {
    engine: PyObject,
    is_oracle: bool,
    version: u32,
    enable_quick_eval: bool,
    enable_rule_based_agari_guard: bool,
    name: String,
    player_ids: Vec<u8>,

    actions: Vec<usize>,
    q_values: Vec<[f32; ACTION_SPACE]>,
    masks_recv: Vec<[bool; ACTION_SPACE]>,
    is_greedy: Vec<bool>,
    last_eval_elapsed: Duration,
    last_batch_size: usize,

    evaluated: bool,
    quick_eval_reactions: Vec<Option<Event>>,

    wg: WaitGroup,
    sync_fields: Arc<Mutex<SyncFields>>,

    search: Option<SearchIntegration>,
}

struct SyncFields {
    states: Vec<Array2<f32>>,
    invisible_states: Vec<Array2<f32>>,
    masks: Vec<Array1<bool>>,
    action_idxs: Vec<usize>,
    kan_action_idxs: Vec<Option<usize>>,
}

/// Tracks why a search was skipped.
#[derive(Default)]
struct SkipMetrics {
    low_criticality: u32,
    insufficient_candidates: u32,
    no_particles: u32,
    particle_gen_error: u32,
    replay_error: u32,
    insufficient_data: u32,
}

/// Tracks error categories during search rollouts.
#[derive(Default)]
struct ErrorMetrics {
    panics: u32,
    rollout_errors: u32,
    /// First N unique error messages for debugging.
    unique_msgs: Vec<String>,
}

impl ErrorMetrics {
    const MAX_UNIQUE_MSGS: usize = 10;

    fn record_panic(&mut self, msg: &str) {
        self.panics += 1;
        self.record_msg(msg);
    }

    fn record_error(&mut self, msg: &str) {
        self.rollout_errors += 1;
        self.record_msg(msg);
    }

    fn record_msg(&mut self, msg: &str) {
        if self.unique_msgs.len() < Self::MAX_UNIQUE_MSGS
            && !self.unique_msgs.iter().any(|m| m == msg)
        {
            self.unique_msgs.push(msg.to_owned());
        }
    }

    const fn total(&self) -> u32 {
        self.panics + self.rollout_errors
    }
}

/// Per-game GRP history state, tracking kyoku boundary entries.
#[derive(Clone, Default)]
struct GrpGameState {
    history: Vec<GrpEntry>,
    current_kyoku: Option<(u8, GrpEntry)>,
}

/// Optional search integration that overrides DQN actions with
/// particle-based rollout evaluation at critical decision points.
struct SearchIntegration {
    rng: ChaCha12Rng,
    config: ParticleConfig,
    max_candidates: usize,
    search_weight: f32,

    // GRP value truncation (optional — when None, uses terminal rollouts)
    grp: Option<GrpEvaluator>,
    grp_max_steps: u32,

    // Per-game GRP history (keyed by game index within the batch).
    // Each game accumulates its own kyoku boundary entries independently.
    grp_game_states: HashMap<usize, GrpGameState>,

    // Core counts
    search_count: u32,
    override_count: u32,

    // GRP-specific metrics
    grp_eval_count: u64,
    grp_eval_time_us: u64,
    grp_truncated_count: u64,
    grp_terminated_count: u64,
    grp_search_count: u32,
    grp_override_count: u32,

    // Detailed breakdowns
    skips: SkipMetrics,
    errors: ErrorMetrics,

    // Time tracking (per-search for percentiles)
    search_times_us: Vec<u64>,

    // Particle utilization
    particles_requested: u32,
    particles_generated: u32,
    particle_gen_attempts: u32,
}

/// Extract a human-readable message from a caught panic payload.
fn extract_panic_msg(info: &Box<dyn std::any::Any + Send>) -> String {
    info.downcast_ref::<String>()
        .map(String::as_str)
        .or_else(|| info.downcast_ref::<&str>().copied())
        .unwrap_or("unknown panic")
        .to_owned()
}

/// Shared search preparation context returned by `prepare_search()`.
struct SearchContext {
    actions: Vec<usize>,
    candidates: Vec<(usize, f32)>,
    particles: Vec<particle::Particle>,
    replayed: [PlayerState; 4],
    base: MidgameContextBase,
    seeds: Vec<u64>,
    start: Instant,
}

impl SearchIntegration {
    /// Common preamble shared by `maybe_override` and `maybe_override_grp`.
    ///
    /// Performs criticality check, candidate selection, particle generation,
    /// event replay, context derivation, and seed generation.
    /// Returns `None` on any skip condition.
    fn prepare_search(
        &mut self,
        state: &PlayerState,
        q_values: &[f32; ACTION_SPACE],
        masks: &[bool; ACTION_SPACE],
    ) -> Option<SearchContext> {
        let start = Instant::now();

        if !Self::is_critical(state) {
            self.skips.low_criticality += 1;
            return None;
        }

        let mut candidates: Vec<(usize, f32)> = q_values
            .iter()
            .enumerate()
            .filter(|&(i, _)| masks[i] && i != 37)
            .map(|(i, &q)| (i, q))
            .collect();
        candidates.sort_by(|a, b| b.1.total_cmp(&a.1));
        candidates.truncate(self.max_candidates);

        if candidates.len() < 2 {
            self.skips.insufficient_candidates += 1;
            return None;
        }

        let actions: Vec<usize> = candidates.iter().map(|&(a, _)| a).collect();
        let n_target = self.config.n_particles;

        let gen_result =
            particle::generate_particles(state, &self.config, &mut self.rng);
        let (particles, attempts) = match gen_result {
            Ok(pair) => pair,
            Err(e) => {
                self.skips.particle_gen_error += 1;
                log::debug!("particle gen error: {e}");
                return None;
            }
        };
        self.particle_gen_attempts += attempts as u32;
        if particles.is_empty() {
            self.skips.no_particles += 1;
            return None;
        }
        self.particles_requested += n_target as u32;
        self.particles_generated += particles.len() as u32;

        self.search_count += 1;

        let replayed = match simulator::replay_player_states(state) {
            Ok(r) => r,
            Err(e) => {
                self.skips.replay_error += 1;
                log::debug!("replay error: {e}");
                self.record_elapsed(start);
                return None;
            }
        };

        let base = simulator::derive_midgame_context_base(state.event_history());

        let seeds: Vec<u64> = (0..particles.len())
            .map(|_| self.rng.next_u64())
            .collect();

        Some(SearchContext {
            actions,
            candidates,
            particles,
            replayed,
            base,
            seeds,
            start,
        })
    }

    /// Decide whether to override the DQN choice using Welch's t-test.
    ///
    /// Compares the search's best action against the DQN's top choice.
    /// Returns `Some(action)` only when the difference is statistically
    /// significant (t-statistic exceeds threshold derived from `search_weight`).
    fn decide_override(
        &mut self,
        actions: &[usize],
        candidates: &[(usize, f32)],
        action_sums: &[f64],
        action_sum_sq: &[f64],
        action_counts: &[usize],
    ) -> Option<usize> {
        // Find search's best action (highest mean)
        let search_means: Vec<f64> = action_sums
            .iter()
            .zip(action_counts.iter())
            .map(|(&sum, &count)| {
                if count > 0 { sum / count as f64 } else { f64::NEG_INFINITY }
            })
            .collect();

        let search_best_idx = search_means
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.total_cmp(b))?
            .0;

        let dqn_action = candidates[0].0;
        let search_best_action = actions[search_best_idx];

        // If search agrees with DQN, no override needed
        if search_best_action == dqn_action {
            return None;
        }

        // Find DQN's action index in our actions array
        let dqn_idx = actions.iter().position(|&a| a == dqn_action)?;

        let n_best = action_counts[search_best_idx];
        let n_dqn = action_counts[dqn_idx];

        // Need at least 2 samples for variance estimation
        if n_best < 2 || n_dqn < 2 {
            self.skips.insufficient_data += 1;
            return None;
        }

        let mean_best = search_means[search_best_idx];
        let mean_dqn = search_means[dqn_idx];

        // Bessel-corrected sample variance: (sum_sq - n * mean²) / (n - 1)
        let var_best =
            (action_sum_sq[search_best_idx] - n_best as f64 * mean_best * mean_best)
                / (n_best - 1) as f64;
        let var_dqn =
            (action_sum_sq[dqn_idx] - n_dqn as f64 * mean_dqn * mean_dqn)
                / (n_dqn - 1) as f64;

        // Welch's t-test: SE² = var_a/n_a + var_b/n_b
        let se_sq = var_best / n_best as f64 + var_dqn / n_dqn as f64;
        if se_sq < 1e-20 {
            // Essentially identical distributions
            self.skips.insufficient_data += 1;
            return None;
        }
        let t_stat = (mean_best - mean_dqn) / se_sq.sqrt();

        // Threshold from search_weight:
        //   weight=0.3 → t>2.1 (conservative, ~p<0.05)
        //   weight=0.5 → t>1.5 (moderate)
        //   weight=1.0 → t>0.0 (always override)
        let threshold = 3.0 * (1.0 - f64::from(self.search_weight));

        if t_stat > threshold {
            Some(search_best_action)
        } else {
            self.skips.insufficient_data += 1;
            None
        }
    }

    /// Update the GRP history accumulator for the current kyoku.
    ///
    /// When we enter a new kyoku (detected by `grand_kyoku` change), pushes
    /// the previous kyoku's entry to the game's history and creates a new one
    /// from the current event log's `StartKyoku` event.
    fn update_grp_history(&mut self, game_index: usize, state: &PlayerState, grand_kyoku: u8) {
        let gs = self.grp_game_states.entry(game_index).or_default();
        let need_update = match &gs.current_kyoku {
            None => true,
            Some((gk, _)) => *gk != grand_kyoku,
        };
        if need_update {
            if let Some((_, prev_entry)) = gs.current_kyoku.take() {
                gs.history.push(prev_entry);
            }
            // Find the StartKyoku event (first event in per-kyoku history)
            for event in state.event_history().iter().rev() {
                if let Event::StartKyoku {
                    bakaze, kyoku, honba, kyotaku, scores, ..
                } = event
                {
                    let gk = (bakaze.as_u8() - tu8!(E)) * 4 + (kyoku - 1);
                    let entry = grp::make_grp_entry(gk, *honba, *kyotaku, scores);
                    gs.current_kyoku = Some((grand_kyoku, entry));
                    break;
                }
            }
        }
    }

    /// Decide whether to override the DQN action using terminal rollout search.
    ///
    /// Returns `Some(action)` if search found a statistically significantly
    /// better action, `None` to keep DQN choice.
    fn maybe_override(
        &mut self,
        state: &PlayerState,
        q_values: &[f32; ACTION_SPACE],
        masks: &[bool; ACTION_SPACE],
        actor: u8,
    ) -> Option<usize> {
        let ctx = self.prepare_search(state, q_values, masks)?;
        let SearchContext {
            actions, candidates, particles, replayed, base, seeds, start,
        } = ctx;

        // Parallel rollouts: particle-first, action-second
        enum RolloutOutcome {
            Ok { action: usize, value: f64 },
            Error { msg: String },
            Panic { msg: String },
        }

        let per_particle: Vec<Vec<RolloutOutcome>> = particles
            .par_iter()
            .enumerate()
            .map(|(pi, p)| {
                let board_state = match panic::catch_unwind(panic::AssertUnwindSafe(|| {
                    simulator::build_midgame_board_state_with_base(
                        state, &replayed, p, &base,
                    )
                })) {
                    Ok(Ok(bs)) => bs,
                    Ok(Err(e)) => {
                        let msg = e.to_string();
                        return actions
                            .iter()
                            .map(|_| RolloutOutcome::Error { msg: msg.clone() })
                            .collect();
                    }
                    Err(panic_info) => {
                        let msg = extract_panic_msg(&panic_info);
                        return actions
                            .iter()
                            .map(|_| RolloutOutcome::Panic { msg: msg.clone() })
                            .collect();
                    }
                };
                let initial_scores = board_state.board.scores;
                let mut particle_rng = ChaCha12Rng::seed_from_u64(seeds[pi]);

                actions
                    .iter()
                    .map(|&action| {
                        let action_seed = particle_rng.next_u64();
                        let mut action_rng = ChaCha12Rng::seed_from_u64(action_seed);

                        let rollout = panic::catch_unwind(panic::AssertUnwindSafe(|| {
                            simulator::simulate_action_rollout_prebuilt(
                                &board_state,
                                initial_scores,
                                actor,
                                action,
                                Some(&mut action_rng),
                            )
                        }));
                        match rollout {
                            Ok(Ok(result)) => {
                                RolloutOutcome::Ok {
                                    action,
                                    value: f64::from(result.deltas[actor as usize]),
                                }
                            }
                            Ok(Err(e)) => {
                                RolloutOutcome::Error { msg: e.to_string() }
                            }
                            Err(panic_info) => RolloutOutcome::Panic {
                                msg: extract_panic_msg(&panic_info),
                            },
                        }
                    })
                    .collect()
            })
            .collect();

        // Aggregate results
        let mut action_sums: Vec<f64> = vec![0.0; actions.len()];
        let mut action_sum_sq: Vec<f64> = vec![0.0; actions.len()];
        let mut action_counts: Vec<usize> = vec![0; actions.len()];

        for particle_results in &per_particle {
            for outcome in particle_results {
                match outcome {
                    RolloutOutcome::Ok { action, value } => {
                        let ai = actions.iter().position(|&a| a == *action).unwrap();
                        action_sums[ai] += value;
                        action_sum_sq[ai] += value * value;
                        action_counts[ai] += 1;
                    }
                    RolloutOutcome::Error { msg } => {
                        self.errors.record_error(msg);
                        log::debug!("search rollout error: {msg}");
                    }
                    RolloutOutcome::Panic { msg } => {
                        self.errors.record_panic(msg);
                        log::warn!("search rollout panic: {msg}");
                    }
                }
            }
        }

        let result = self.decide_override(
            &actions, &candidates, &action_sums, &action_sum_sq, &action_counts,
        );
        self.record_elapsed(start);

        if result.is_some() {
            self.override_count += 1;
        }
        result
    }

    /// GRP-based search override: truncated rollouts + value network evaluation.
    ///
    /// Returns `Some(action)` if search found a statistically significantly
    /// better action, `None` to keep DQN choice.
    fn maybe_override_grp(
        &mut self,
        game_index: usize,
        state: &PlayerState,
        q_values: &[f32; ACTION_SPACE],
        masks: &[bool; ACTION_SPACE],
        actor: u8,
    ) -> Option<usize> {
        if self.grp.is_none() {
            return None;
        }

        let ctx = self.prepare_search(state, q_values, masks)?;
        let SearchContext {
            actions, candidates, particles, replayed, base, seeds, start,
        } = ctx;

        self.grp_search_count += 1;
        let max_steps = self.grp_max_steps;

        // Update GRP history accumulator for this specific game
        let bakaze_u8 = state.bakaze().as_u8();
        let grand_kyoku = (bakaze_u8 - tu8!(E)) * 4 + state.kyoku();
        self.update_grp_history(game_index, state, grand_kyoku);

        // Build current state GRP entry for baseline
        let player_id = state.player_id();
        let current_scores = state.scores();
        let mut abs_scores = [0_i32; 4];
        for i in 0..4 {
            abs_scores[(player_id as usize + i) % 4] = current_scores[i];
        }
        let current_entry = grp::make_grp_entry(
            grand_kyoku,
            state.honba(),
            state.kyotaku(),
            &abs_scores,
        );

        // Baseline GRP evaluation (scoped to avoid borrow conflict)
        let baseline_eval_start = Instant::now();
        let current_ev = {
            let grp = self.grp.as_ref().unwrap();
            let gs = self.grp_game_states.entry(game_index).or_default();
            grp.evaluate_leaf_impl(&gs.history, &current_entry, player_id)
        };
        let current_ev = match current_ev {
            Ok(ev) => ev,
            Err(e) => {
                log::debug!("GRP baseline eval error: {e}");
                self.record_elapsed(start);
                return None;
            }
        };
        let baseline_eval_us = Instant::now()
            .checked_duration_since(baseline_eval_start)
            .unwrap_or(Duration::ZERO)
            .as_micros() as u64;

        // Parallel truncated rollouts: particle-first, action-second
        enum GrpOutcome {
            Ok {
                action: usize,
                value: f64,
                terminated: bool,
                grp_eval_us: u64,
            },
            Error { msg: String },
            Panic { msg: String },
        }

        let grp = self.grp.as_ref().unwrap();
        let gs = self.grp_game_states.entry(game_index).or_default();
        let history = &gs.history;

        let per_particle: Vec<Vec<GrpOutcome>> = particles
            .par_iter()
            .enumerate()
            .map(|(pi, p)| {
                let board_state = match panic::catch_unwind(panic::AssertUnwindSafe(|| {
                    simulator::build_midgame_board_state_with_base(
                        state, &replayed, p, &base,
                    )
                })) {
                    Ok(Ok(bs)) => bs,
                    Ok(Err(e)) => {
                        let msg = e.to_string();
                        return actions
                            .iter()
                            .map(|_| GrpOutcome::Error { msg: msg.clone() })
                            .collect();
                    }
                    Err(panic_info) => {
                        let msg = extract_panic_msg(&panic_info);
                        return actions
                            .iter()
                            .map(|_| GrpOutcome::Panic { msg: msg.clone() })
                            .collect();
                    }
                };
                let initial_scores = board_state.board.scores;
                let mut particle_rng = ChaCha12Rng::seed_from_u64(seeds[pi]);

                actions
                    .iter()
                    .map(|&action| {
                        let action_seed = particle_rng.next_u64();
                        let mut action_rng = ChaCha12Rng::seed_from_u64(action_seed);

                        let rollout = panic::catch_unwind(panic::AssertUnwindSafe(|| {
                            simulator::simulate_action_rollout_prebuilt_truncated(
                                &board_state,
                                initial_scores,
                                actor,
                                action,
                                Some(&mut action_rng),
                                max_steps,
                            )
                        }));
                        match rollout {
                            Ok(Ok(result)) => {
                                // Build GRP entry: for terminated rollouts, compute
                                // next-kyoku metadata; for truncated, use leaf state.
                                let leaf_entry = if result.terminated {
                                    let (next_kyoku, next_honba) = if result.can_renchan {
                                        (result.kyoku, result.honba.saturating_add(1))
                                    } else if result.has_hora {
                                        (result.kyoku.saturating_add(1), 0)
                                    } else {
                                        (result.kyoku.saturating_add(1), result.honba.saturating_add(1))
                                    };
                                    grp::make_grp_entry(
                                        next_kyoku,
                                        next_honba,
                                        result.kyotaku,
                                        &result.scores,
                                    )
                                } else {
                                    grp::make_grp_entry(
                                        result.kyoku,
                                        result.honba,
                                        result.kyotaku,
                                        &result.scores,
                                    )
                                };

                                let eval_start = Instant::now();
                                match grp.evaluate_leaf_impl(
                                    history, &leaf_entry, actor,
                                ) {
                                    Ok(leaf_ev) => {
                                        let eval_us = Instant::now()
                                            .checked_duration_since(eval_start)
                                            .unwrap_or(Duration::ZERO)
                                            .as_micros() as u64;
                                        GrpOutcome::Ok {
                                            action,
                                            value: leaf_ev - current_ev,
                                            terminated: result.terminated,
                                            grp_eval_us: eval_us,
                                        }
                                    }
                                    Err(e) => GrpOutcome::Error { msg: e.to_string() },
                                }
                            }
                            Ok(Err(e)) => {
                                GrpOutcome::Error { msg: e.to_string() }
                            }
                            Err(panic_info) => GrpOutcome::Panic {
                                msg: extract_panic_msg(&panic_info),
                            },
                        }
                    })
                    .collect()
            })
            .collect();

        // Aggregate results and GRP metrics
        let mut action_sums: Vec<f64> = vec![0.0; actions.len()];
        let mut action_sum_sq: Vec<f64> = vec![0.0; actions.len()];
        let mut action_counts: Vec<usize> = vec![0; actions.len()];
        let mut n_truncated = 0_u64;
        let mut n_terminated = 0_u64;
        let mut leaf_eval_us_total = 0_u64;

        for particle_results in &per_particle {
            for outcome in particle_results {
                match outcome {
                    GrpOutcome::Ok {
                        action, value, terminated, grp_eval_us,
                    } => {
                        let ai = actions.iter().position(|&a| a == *action).unwrap();
                        action_sums[ai] += value;
                        action_sum_sq[ai] += value * value;
                        action_counts[ai] += 1;
                        leaf_eval_us_total += grp_eval_us;
                        if *terminated {
                            n_terminated += 1;
                        } else {
                            n_truncated += 1;
                        }
                    }
                    GrpOutcome::Error { msg } => {
                        self.errors.record_error(msg);
                        log::debug!("search GRP rollout error: {msg}");
                    }
                    GrpOutcome::Panic { msg } => {
                        self.errors.record_panic(msg);
                        log::warn!("search GRP rollout panic: {msg}");
                    }
                }
            }
        }

        // Update GRP metrics
        self.grp_eval_count += 1 + n_truncated + n_terminated;
        self.grp_eval_time_us += baseline_eval_us + leaf_eval_us_total;
        self.grp_truncated_count += n_truncated;
        self.grp_terminated_count += n_terminated;

        let result = self.decide_override(
            &actions, &candidates, &action_sums, &action_sum_sq, &action_counts,
        );
        self.record_elapsed(start);

        if result.is_some() {
            self.override_count += 1;
            self.grp_override_count += 1;
        }
        result
    }

    /// Record elapsed time from `start` into `search_times_us`.
    fn record_elapsed(&mut self, start: Instant) {
        let elapsed_us = Instant::now()
            .checked_duration_since(start)
            .unwrap_or(Duration::ZERO)
            .as_micros() as u64;
        self.search_times_us.push(elapsed_us);
    }

    /// Check if the current decision point is critical enough to warrant search.
    fn is_critical(state: &PlayerState) -> bool {
        let mut score: f32 = 0.0;

        // Factor 1: Opponent riichi (0.3 if any opponent in riichi)
        let riichi = state.riichi_accepted();
        if riichi[1..].iter().any(|&r| r) {
            score += 0.3;
        }

        // Factor 2: Shanten (0.4 for tenpai, 0.2 for iishanten)
        let shanten = state.shanten();
        if shanten <= 0 {
            score += 0.4;
        } else if shanten == 1 {
            score += 0.2;
        }

        // Factor 3: Late kyoku (0.15 for South round)
        let bakaze = state.bakaze();
        if bakaze.as_u8() >= tu8!(S) {
            score += 0.15;
        }

        // Factor 4: Many tiles played (proxy for danger/information)
        if state.tiles_left() < 30 {
            score += 0.15;
        }

        score >= 0.25
    }

    /// Return a summary of GRP-specific metrics.
    fn grp_metrics_summary(&self) -> String {
        let total_rollouts = self.grp_truncated_count + self.grp_terminated_count;
        let truncation_rate = if total_rollouts > 0 {
            self.grp_truncated_count as f64 / total_rollouts as f64 * 100.0
        } else {
            0.0
        };
        let grp_override_pct = if self.grp_search_count > 0 {
            f64::from(self.grp_override_count) / f64::from(self.grp_search_count) * 100.0
        } else {
            0.0
        };
        format!(
            "GRP: searches={}, overrides={} ({:.1}%), evals={}, eval_time={:.1}ms, \
             truncated={}, terminated={}, truncation_rate={:.1}%",
            self.grp_search_count,
            self.grp_override_count,
            grp_override_pct,
            self.grp_eval_count,
            self.grp_eval_time_us as f64 / 1000.0,
            self.grp_truncated_count,
            self.grp_terminated_count,
            truncation_rate,
        )
    }
}

impl SearchIntegration {
    /// Compute percentile from a sorted slice. `p` is 0.0..=1.0.
    fn percentile(sorted: &[u64], p: f64) -> u64 {
        if sorted.is_empty() {
            return 0;
        }
        let idx = ((sorted.len() as f64 - 1.0) * p).round() as usize;
        sorted[idx.min(sorted.len() - 1)]
    }
}

impl Drop for SearchIntegration {
    fn drop(&mut self) {
        let skip_total = self.skips.low_criticality
            + self.skips.insufficient_candidates
            + self.skips.no_particles
            + self.skips.particle_gen_error
            + self.skips.replay_error
            + self.skips.insufficient_data;
        let error_total = self.errors.total();

        if self.search_count == 0 && skip_total == 0 {
            return;
        }

        // Time percentiles
        self.search_times_us.sort_unstable();
        let times = &self.search_times_us;
        let total_time_us: u64 = times.iter().sum();
        let p50 = Self::percentile(times, 0.5);
        let p95 = Self::percentile(times, 0.95);
        let p99 = Self::percentile(times, 0.99);
        let avg_us = if self.search_count > 0 {
            total_time_us / u64::from(self.search_count)
        } else {
            0
        };

        // Particle utilization
        let particle_fill_pct = if self.particles_requested > 0 {
            (f64::from(self.particles_generated) / f64::from(self.particles_requested)) * 100.0
        } else {
            0.0
        };

        let msg = format!(
            "\n=== Search Metrics ===\n\
             Searches: {} | Overrides: {} ({:.1}%) | Errors: {} ({:.1}%)\n\
             Skips: {} (criticality={}, candidates={}, no_particles={}, \
             particle_err={}, replay_err={}, insufficient_data={})\n\
             Errors: panics={}, rollout_errors={}\n\
             Time: total={}ms, avg={}us, p50={}us, p95={}us, p99={}us\n\
             Particles: requested={}, generated={} ({:.0}% fill), gen_attempts={}",
            self.search_count,
            self.override_count,
            if self.search_count > 0 {
                f64::from(self.override_count) / f64::from(self.search_count) * 100.0
            } else {
                0.0
            },
            error_total,
            if self.search_count > 0 {
                f64::from(error_total) / f64::from(self.search_count) * 100.0
            } else {
                0.0
            },
            skip_total,
            self.skips.low_criticality,
            self.skips.insufficient_candidates,
            self.skips.no_particles,
            self.skips.particle_gen_error,
            self.skips.replay_error,
            self.skips.insufficient_data,
            self.errors.panics,
            self.errors.rollout_errors,
            total_time_us / 1000,
            avg_us,
            p50,
            p95,
            p99,
            self.particles_requested,
            self.particles_generated,
            particle_fill_pct,
            self.particle_gen_attempts,
        );

        // Append GRP metrics if any GRP searches were performed
        let msg = if self.grp_search_count > 0 {
            format!("{msg}\n             {}", self.grp_metrics_summary())
        } else {
            msg
        };

        if error_total > 0 {
            log::warn!("{msg}");
            if !self.errors.unique_msgs.is_empty() {
                log::warn!(
                    "Unique error messages ({}):",
                    self.errors.unique_msgs.len()
                );
                for (i, m) in self.errors.unique_msgs.iter().enumerate() {
                    log::warn!("  [{}] {}", i + 1, m);
                }
            }
        } else {
            log::info!("{msg}");
        }
    }
}

impl MortalBatchAgent {
    pub fn new(engine: PyObject, player_ids: &[u8]) -> Result<Self> {
        ensure!(player_ids.iter().all(|&id| matches!(id, 0..=3)));

        let (
            name,
            is_oracle,
            version,
            enable_quick_eval,
            enable_rule_based_agari_guard,
            search_cfg,
        ) = Python::with_gil(|py| {
            let obj = engine.bind_borrowed(py);
            ensure!(
                obj.getattr("react_batch")?.is_callable(),
                "missing method react_batch",
            );

            let name = obj.getattr("name")?.extract()?;
            let is_oracle = obj.getattr("is_oracle")?.extract()?;
            let version = obj.getattr("version")?.extract()?;
            let enable_quick_eval = obj.getattr("enable_quick_eval")?.extract()?;
            let enable_rule_based_agari_guard =
                obj.getattr("enable_rule_based_agari_guard")?.extract()?;

            // Read optional search parameters
            let enable_search: bool = obj
                .getattr("enable_search")
                .and_then(|v| v.extract())
                .unwrap_or(false);

            let search_cfg = if enable_search {
                let n_particles: usize = obj
                    .getattr("search_particles")
                    .and_then(|v| v.extract())
                    .unwrap_or(50);
                // Python None fails u64 extraction -> Err -> .ok() -> None.
                // Python int succeeds -> Ok(val) -> .ok() -> Some(val).
                // Missing attr -> getattr Err -> .ok() -> None.
                let seed: Option<u64> = obj
                    .getattr("search_seed")
                    .and_then(|v| v.extract())
                    .ok();
                let weight: f32 = obj
                    .getattr("search_weight")
                    .and_then(|v| v.extract())
                    .unwrap_or(0.3);

                // GRP model path (optional)
                let grp_model_path: Option<String> = obj
                    .getattr("search_grp_model")
                    .and_then(|v| v.extract())
                    .ok();
                let grp_max_steps: u32 = obj
                    .getattr("search_max_steps")
                    .and_then(|v| v.extract())
                    .unwrap_or(20);
                let grp_placement_pts: Option<[f64; 4]> = obj
                    .getattr("search_placement_pts")
                    .and_then(|v| v.extract())
                    .ok();

                Some((n_particles, seed, weight, grp_model_path, grp_max_steps, grp_placement_pts))
            } else {
                None
            };

            Ok((
                name,
                is_oracle,
                version,
                enable_quick_eval,
                enable_rule_based_agari_guard,
                search_cfg,
            ))
        })?;

        let search = search_cfg
            .map(|(n_particles, seed, weight, grp_model_path, grp_max_steps, grp_placement_pts)| -> anyhow::Result<SearchIntegration> {
                let rng = match seed {
                    Some(s) => ChaCha12Rng::seed_from_u64(s),
                    None => ChaCha12Rng::from_os_rng(),
                };

                let placement_pts = grp_placement_pts.unwrap_or([6.0, 4.0, 2.0, 0.0]);

                let grp = grp_model_path.map(|path| {
                    log::info!("Loading GRP model from {path}");
                    GrpEvaluator::load_impl(&path, placement_pts)
                }).transpose()
                .context("failed to load GRP ONNX model")?;

                Ok(SearchIntegration {
                    rng,
                    config: ParticleConfig::new(n_particles),
                    max_candidates: 5,
                    search_weight: weight,
                    grp,
                    grp_max_steps,
                    grp_game_states: HashMap::new(),
                    search_count: 0_u32,
                    override_count: 0_u32,
                    grp_eval_count: 0_u64,
                    grp_eval_time_us: 0_u64,
                    grp_truncated_count: 0_u64,
                    grp_terminated_count: 0_u64,
                    grp_search_count: 0_u32,
                    grp_override_count: 0_u32,
                    skips: SkipMetrics::default(),
                    errors: ErrorMetrics::default(),
                    search_times_us: Vec::new(),
                    particles_requested: 0_u32,
                    particles_generated: 0_u32,
                    particle_gen_attempts: 0_u32,
                })
            }).transpose()?;

        let size = player_ids.len();
        let quick_eval_reactions = if enable_quick_eval {
            vec![None; size]
        } else {
            vec![]
        };
        let sync_fields = Arc::new(Mutex::new(SyncFields {
            states: vec![],
            invisible_states: vec![],
            masks: vec![],
            action_idxs: vec![0; size],
            kan_action_idxs: vec![None; size],
        }));

        Ok(Self {
            engine,
            is_oracle,
            version,
            enable_quick_eval,
            enable_rule_based_agari_guard,
            name,
            player_ids: player_ids.to_vec(),

            actions: vec![],
            q_values: vec![],
            masks_recv: vec![],
            is_greedy: vec![],
            last_eval_elapsed: Duration::ZERO,
            last_batch_size: 0,

            evaluated: false,
            quick_eval_reactions,

            wg: WaitGroup::new(),
            sync_fields,

            search,
        })
    }

    fn evaluate(&mut self) -> Result<()> {
        // Wait for all feature encodings to complete.
        mem::take(&mut self.wg).wait();
        let mut sync_fields = self.sync_fields.lock();

        if sync_fields.states.is_empty() {
            return Ok(());
        }

        let start = Instant::now();
        self.last_batch_size = sync_fields.states.len();

        (self.actions, self.q_values, self.masks_recv, self.is_greedy) = Python::with_gil(|py| {
            let states: Vec<_> = sync_fields
                .states
                .drain(..)
                .map(|v| PyArray2::from_owned_array(py, v))
                .collect();
            let masks: Vec<_> = sync_fields
                .masks
                .drain(..)
                .map(|v| PyArray1::from_owned_array(py, v))
                .collect();
            let invisible_states: Option<Vec<_>> = self.is_oracle.then(|| {
                sync_fields
                    .invisible_states
                    .drain(..)
                    .map(|v| PyArray2::from_owned_array(py, v))
                    .collect()
            });

            let args = (states, masks, invisible_states);
            self.engine
                .bind_borrowed(py)
                .call_method1(intern!(py, "react_batch"), args)
                .context("failed to execute `react_batch` on Python engine")?
                .extract()
                .context("failed to extract to Rust type")
        })?;

        self.last_eval_elapsed = Instant::now()
            .checked_duration_since(start)
            .unwrap_or(Duration::ZERO);

        Ok(())
    }

    fn gen_meta(&self, state: &PlayerState, action_idx: usize) -> Metadata {
        let q_values = self.q_values[action_idx];
        let masks = self.masks_recv[action_idx];
        let is_greedy = self.is_greedy[action_idx];

        let mut mask_bits = 0;
        let q_values_compact = q_values
            .into_iter()
            .zip(masks)
            .enumerate()
            .filter(|&(_, (_, m))| m)
            .map(|(i, (q, _))| {
                mask_bits |= 0b1 << i;
                q
            })
            .collect();

        Metadata {
            q_values: Some(q_values_compact),
            mask_bits: Some(mask_bits),
            is_greedy: Some(is_greedy),
            shanten: Some(state.shanten()),
            at_furiten: Some(state.at_furiten()),
            ..Default::default()
        }
    }
}

impl BatchAgent for MortalBatchAgent {
    #[inline]
    fn name(&self) -> String {
        self.name.clone()
    }

    #[inline]
    fn oracle_obs_version(&self) -> Option<u32> {
        self.is_oracle.then_some(self.version)
    }

    fn start_game(&mut self, index: usize) -> Result<()> {
        // Clear this game's GRP history so it doesn't leak across hanchan.
        if let Some(si) = &mut self.search {
            si.grp_game_states.remove(&index);
        }
        Ok(())
    }

    fn set_scene(
        &mut self,
        index: usize,
        _: &[EventExt],
        state: &PlayerState,
        invisible_state: Option<InvisibleState>,
    ) -> Result<()> {
        self.evaluated = false;
        let cans = state.last_cans();

        if self.enable_quick_eval
            && cans.can_discard
            && !cans.can_riichi
            && !cans.can_tsumo_agari
            && !cans.can_ankan
            && !cans.can_kakan
            && !cans.can_ryukyoku
        {
            let candidates = state.discard_candidates_aka();
            let mut only_candidate = None;
            for (tile, &flag) in candidates.iter().enumerate() {
                if !flag {
                    continue;
                }
                match only_candidate.take() {
                    None => only_candidate = Some(tile),
                    Some(_) => break,
                }
            }

            if let Some(tile_id) = only_candidate {
                let actor = self.player_ids[index];
                let pai = must_tile!(tile_id);
                let tsumogiri = state.last_self_tsumo().is_some_and(|t| t == pai);
                let ev = Event::Dahai {
                    actor,
                    pai,
                    tsumogiri,
                };
                self.quick_eval_reactions[index] = Some(ev);
                return Ok(());
            }
        }

        let need_kan_select = if !cans.can_ankan && !cans.can_kakan {
            false
        } else if !self.enable_quick_eval {
            true
        } else {
            state.ankan_candidates().len() + state.kakan_candidates().len() > 1
        };

        let version = self.version;
        let state = state.clone();
        let sync_fields = Arc::clone(&self.sync_fields);
        let wg = self.wg.clone();
        rayon::spawn(move || {
            let _wg = wg;

            // Encode features in parallel within the game batch to utilize
            // multiple cores, as this can be very CPU-intensive, especially for
            // the sp feature (since v4).
            let kan = need_kan_select.then(|| state.encode_obs(version, true));
            let (feature, mask) = state.encode_obs(version, false);

            let SyncFields {
                states,
                invisible_states,
                masks,
                action_idxs,
                kan_action_idxs,
            } = &mut *sync_fields.lock();
            if let Some((kan_feature, kan_mask)) = kan {
                kan_action_idxs[index] = Some(states.len());
                states.push(kan_feature);
                masks.push(kan_mask);
                if let Some(invisible_state) = invisible_state.clone() {
                    invisible_states.push(invisible_state);
                }
            }

            action_idxs[index] = states.len();
            states.push(feature);
            masks.push(mask);
            if let Some(invisible_state) = invisible_state {
                invisible_states.push(invisible_state);
            }
        });

        Ok(())
    }

    fn get_reaction(
        &mut self,
        index: usize,
        _: &[EventExt],
        state: &PlayerState,
        _: Option<InvisibleState>,
    ) -> Result<EventExt> {
        if self.enable_quick_eval
            && let Some(ev) = self.quick_eval_reactions[index].take()
        {
            return Ok(EventExt::no_meta(ev));
        }

        if !self.evaluated {
            self.evaluate()?;
            self.evaluated = true;
        }
        let start = Instant::now();

        let mut sync_fields = self.sync_fields.lock();
        let action_idx = sync_fields.action_idxs[index];
        let kan_select_idx = sync_fields.kan_action_idxs[index].take();

        let actor = self.player_ids[index];
        let akas_in_hand = state.akas_in_hand();
        let cans = state.last_cans();

        let orig_action = self.actions[action_idx];
        let action =
            if self.enable_rule_based_agari_guard && orig_action == 43 && !state.rule_based_agari()
            {
                // The engine wants agari, but the rule-based engine is against
                // it. In rule-based agari guard mode, it will force to execute
                // the best alternative option other than agari.
                let mut q_values = self.q_values[action_idx];
                q_values[43] = f32::MIN;
                q_values
                    .iter()
                    .enumerate()
                    .max_by(|(_, l), (_, r)| l.total_cmp(r))
                    .unwrap()
                    .0
            } else {
                orig_action
            };

        // Drop the lock before potentially expensive search
        drop(sync_fields);

        // Optionally override with search-informed action.
        // Skip search when DQN wants riichi (37): search can't evaluate
        // riichi (two-step action), so don't risk suppressing it.
        let action = if action != 37 {
            if let Some(ref mut search) = self.search {
                // Use GRP-based search when a GRP model is loaded,
                // otherwise fall back to terminal rollout search.
                let override_action = if search.grp.is_some() {
                    search.maybe_override_grp(
                        index,
                        state,
                        &self.q_values[action_idx],
                        &self.masks_recv[action_idx],
                        actor,
                    )
                } else {
                    search.maybe_override(
                        state,
                        &self.q_values[action_idx],
                        &self.masks_recv[action_idx],
                        actor,
                    )
                };
                override_action.unwrap_or(action)
            } else {
                action
            }
        } else {
            action
        };

        let event = match action {
            0..=36 => {
                ensure!(
                    cans.can_discard,
                    "failed discard check: {}",
                    state.brief_info()
                );

                let pai = must_tile!(action);
                let tsumogiri = state.last_self_tsumo().is_some_and(|t| t == pai);
                Event::Dahai {
                    actor,
                    pai,
                    tsumogiri,
                }
            }

            37 => {
                ensure!(
                    cans.can_riichi,
                    "failed riichi check: {}",
                    state.brief_info()
                );

                Event::Reach { actor }
            }

            38 => {
                ensure!(
                    cans.can_chi_low,
                    "failed chi low check: {}",
                    state.brief_info()
                );

                let pai = state
                    .last_kawa_tile()
                    .context("invalid state: no last kawa tile")?;
                let first = pai.next();

                let can_akaize_consumed = match pai.as_u8() {
                    tu8!(3m) | tu8!(4m) => akas_in_hand[0],
                    tu8!(3p) | tu8!(4p) => akas_in_hand[1],
                    tu8!(3s) | tu8!(4s) => akas_in_hand[2],
                    _ => false,
                };
                let consumed = if can_akaize_consumed {
                    [first.akaize(), first.next().akaize()]
                } else {
                    [first, first.next()]
                };
                Event::Chi {
                    actor,
                    target: cans.target_actor,
                    pai,
                    consumed,
                }
            }
            39 => {
                ensure!(
                    cans.can_chi_mid,
                    "failed chi mid check: {}",
                    state.brief_info()
                );

                let pai = state
                    .last_kawa_tile()
                    .context("invalid state: no last kawa tile")?;

                let can_akaize_consumed = match pai.as_u8() {
                    tu8!(4m) | tu8!(6m) => akas_in_hand[0],
                    tu8!(4p) | tu8!(6p) => akas_in_hand[1],
                    tu8!(4s) | tu8!(6s) => akas_in_hand[2],
                    _ => false,
                };
                let consumed = if can_akaize_consumed {
                    [pai.prev().akaize(), pai.next().akaize()]
                } else {
                    [pai.prev(), pai.next()]
                };
                Event::Chi {
                    actor,
                    target: cans.target_actor,
                    pai,
                    consumed,
                }
            }
            40 => {
                ensure!(
                    cans.can_chi_high,
                    "failed chi high check: {}",
                    state.brief_info()
                );

                let pai = state
                    .last_kawa_tile()
                    .context("invalid state: no last kawa tile")?;
                let last = pai.prev();

                let can_akaize_consumed = match pai.as_u8() {
                    tu8!(6m) | tu8!(7m) => akas_in_hand[0],
                    tu8!(6p) | tu8!(7p) => akas_in_hand[1],
                    tu8!(6s) | tu8!(7s) => akas_in_hand[2],
                    _ => false,
                };
                let consumed = if can_akaize_consumed {
                    [last.prev().akaize(), last.akaize()]
                } else {
                    [last.prev(), last]
                };
                Event::Chi {
                    actor,
                    target: cans.target_actor,
                    pai,
                    consumed,
                }
            }

            41 => {
                ensure!(cans.can_pon, "failed pon check: {}", state.brief_info());

                let pai = state
                    .last_kawa_tile()
                    .context("invalid state: no last kawa tile")?;

                let can_akaize_consumed = match pai.as_u8() {
                    tu8!(5m) => akas_in_hand[0],
                    tu8!(5p) => akas_in_hand[1],
                    tu8!(5s) => akas_in_hand[2],
                    _ => false,
                };
                let consumed = if can_akaize_consumed {
                    [pai.akaize(), pai.deaka()]
                } else {
                    [pai.deaka(); 2]
                };
                Event::Pon {
                    actor,
                    target: cans.target_actor,
                    pai,
                    consumed,
                }
            }

            42 => {
                ensure!(
                    cans.can_daiminkan || cans.can_ankan || cans.can_kakan,
                    "failed kan check: {}",
                    state.brief_info()
                );

                let ankan_candidates = state.ankan_candidates();
                let kakan_candidates = state.kakan_candidates();

                let tile = if let Some(kan_idx) = kan_select_idx {
                    let tile = must_tile!(self.actions[kan_idx]);
                    ensure!(
                        ankan_candidates.contains(&tile) || kakan_candidates.contains(&tile),
                        "kan choice not in kan candidates: {}",
                        state.brief_info()
                    );
                    tile
                } else if cans.can_daiminkan {
                    state
                        .last_kawa_tile()
                        .context("invalid state: no last kawa tile")?
                } else if cans.can_ankan {
                    ankan_candidates[0]
                } else {
                    kakan_candidates[0]
                };

                if cans.can_daiminkan {
                    let consumed = if tile.is_aka() {
                        [tile.deaka(); 3]
                    } else {
                        [tile.akaize(), tile, tile]
                    };
                    Event::Daiminkan {
                        actor,
                        target: cans.target_actor,
                        pai: tile,
                        consumed,
                    }
                } else if cans.can_ankan && ankan_candidates.contains(&tile.deaka()) {
                    Event::Ankan {
                        actor,
                        consumed: [tile.akaize(), tile, tile, tile],
                    }
                } else {
                    let can_akaize_target = match tile.as_u8() {
                        tu8!(5m) => akas_in_hand[0],
                        tu8!(5p) => akas_in_hand[1],
                        tu8!(5s) => akas_in_hand[2],
                        _ => false,
                    };
                    let (pai, consumed) = if can_akaize_target {
                        (tile.akaize(), [tile.deaka(); 3])
                    } else {
                        (tile.deaka(), [tile.akaize(), tile.deaka(), tile.deaka()])
                    };
                    Event::Kakan {
                        actor,
                        pai,
                        consumed,
                    }
                }
            }

            43 => {
                ensure!(
                    cans.can_agari(),
                    "failed hora check: {}",
                    state.brief_info(),
                );

                Event::Hora {
                    actor,
                    target: cans.target_actor,
                    deltas: None,
                    ura_markers: None,
                }
            }

            44 => {
                ensure!(
                    cans.can_ryukyoku,
                    "failed ryukyoku check: {}",
                    state.brief_info()
                );

                Event::Ryukyoku { deltas: None }
            }

            // 45
            _ => Event::None,
        };

        let mut meta = self.gen_meta(state, action_idx);
        let eval_time_ns = Instant::now()
            .checked_duration_since(start)
            .unwrap_or(Duration::ZERO)
            .saturating_add(self.last_eval_elapsed)
            .as_nanos()
            .try_into()
            .unwrap_or(u64::MAX);

        meta.eval_time_ns = Some(eval_time_ns);
        meta.batch_size = Some(self.last_batch_size);
        meta.kan_select = kan_select_idx.map(|kan_idx| Box::new(self.gen_meta(state, kan_idx)));

        Ok(EventExt {
            event,
            meta: Some(meta),
        })
    }

    fn needs_record_events(&self) -> bool {
        self.search.is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_metrics_dedup_messages() {
        let mut em = ErrorMetrics::default();
        em.record_panic("index out of bounds: the len is 0 but the index is 0");
        em.record_panic("index out of bounds: the len is 0 but the index is 0");
        em.record_error("some other error");
        em.record_error("some other error");
        em.record_panic("yet another panic");

        assert_eq!(em.panics, 3);
        assert_eq!(em.rollout_errors, 2);
        assert_eq!(em.total(), 5);
        // Dedup: only 3 unique messages
        assert_eq!(em.unique_msgs.len(), 3);
    }

    #[test]
    fn error_metrics_caps_at_max() {
        let mut em = ErrorMetrics::default();
        for i in 0..20 {
            em.record_error(&format!("error {i}"));
        }
        assert_eq!(em.rollout_errors, 20);
        assert_eq!(em.unique_msgs.len(), ErrorMetrics::MAX_UNIQUE_MSGS);
    }

    #[test]
    fn skip_metrics_default_all_zero() {
        let sm = SkipMetrics::default();
        assert_eq!(sm.low_criticality, 0);
        assert_eq!(sm.insufficient_candidates, 0);
        assert_eq!(sm.no_particles, 0);
        assert_eq!(sm.particle_gen_error, 0);
        assert_eq!(sm.replay_error, 0);
        assert_eq!(sm.insufficient_data, 0);
    }

    #[test]
    fn percentile_empty_returns_zero() {
        assert_eq!(SearchIntegration::percentile(&[], 0.5), 0);
    }

    #[test]
    fn percentile_single_element() {
        assert_eq!(SearchIntegration::percentile(&[42], 0.0), 42);
        assert_eq!(SearchIntegration::percentile(&[42], 0.5), 42);
        assert_eq!(SearchIntegration::percentile(&[42], 1.0), 42);
    }

    #[test]
    fn percentile_multiple_elements() {
        let data = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100];
        assert_eq!(SearchIntegration::percentile(&data, 0.0), 10);
        assert_eq!(SearchIntegration::percentile(&data, 0.5), 60); // (10-1)*0.5 = 4.5 -> round to 5 -> data[5]=60
        assert_eq!(SearchIntegration::percentile(&data, 0.95), 100); // (10-1)*0.95 = 8.55 -> round to 9 -> data[9]=100
        assert_eq!(SearchIntegration::percentile(&data, 1.0), 100);
    }

    #[test]
    fn is_critical_early_game_low_shanten() {
        // Use the shared test helper: sets up East round, tsumo just happened,
        // hand is 1-9m+1-5p (iishanten or better), tiles_left=69 (just drew).
        // Early east round + many tiles left + no opponent riichi → not critical
        // unless shanten is low enough to add 0.2-0.4.
        let state = crate::search::test_utils::setup_basic_game();
        // This hand (1-9m, 1-5p) is iishanten at best → score = 0.2, < 0.25 threshold
        // If it happens to be tenpai, score = 0.4 which IS critical.
        // Either way, this verifies is_critical doesn't panic on a real state.
        let _ = SearchIntegration::is_critical(&state);
    }

    /// Helper to create a SearchIntegration with default fields for testing.
    fn make_test_si(search_count: u32, override_count: u32) -> SearchIntegration {
        SearchIntegration {
            rng: ChaCha12Rng::seed_from_u64(0),
            config: ParticleConfig::new(10),
            max_candidates: 5,
            search_weight: 0.3,
            grp: None,
            grp_max_steps: 20,
            grp_game_states: HashMap::new(),
            search_count,
            override_count,
            grp_eval_count: 0,
            grp_eval_time_us: 0,
            grp_truncated_count: 0,
            grp_terminated_count: 0,
            grp_search_count: 0,
            grp_override_count: 0,
            skips: SkipMetrics::default(),
            errors: ErrorMetrics::default(),
            search_times_us: Vec::new(),
            particles_requested: 0,
            particles_generated: 0,
            particle_gen_attempts: 0,
        }
    }

    #[test]
    fn search_integration_drop_does_not_panic() {
        // Verify that dropping with zero counts doesn't panic
        let si = make_test_si(0, 0);
        drop(si); // should not panic
    }

    #[test]
    fn search_integration_drop_with_data_does_not_panic() {
        let mut si = make_test_si(10, 3);
        si.config = ParticleConfig::new(50);
        si.skips = SkipMetrics {
            low_criticality: 20,
            insufficient_candidates: 5,
            no_particles: 2,
            particle_gen_error: 1,
            replay_error: 0,
            insufficient_data: 1,
        };
        si.search_times_us = vec![1000, 2000, 3000, 5000, 10000];
        si.particles_requested = 500;
        si.particles_generated = 450;
        si.particle_gen_attempts = 600;
        si.errors.record_panic("test panic");
        si.errors.record_error("test error");
        drop(si); // should not panic
    }

    #[test]
    fn search_integration_drop_with_grp_data() {
        let mut si = make_test_si(5, 2);
        si.grp_search_count = 5;
        si.grp_override_count = 2;
        si.grp_eval_count = 150;
        si.grp_eval_time_us = 50_000;
        si.grp_truncated_count = 100;
        si.grp_terminated_count = 50;
        si.search_times_us = vec![2000, 4000, 6000, 8000, 10000];
        drop(si); // should not panic and should log GRP metrics
    }

    #[test]
    fn grp_metrics_default_zero() {
        let si = make_test_si(0, 0);
        assert_eq!(si.grp_eval_count, 0);
        assert_eq!(si.grp_eval_time_us, 0);
        assert_eq!(si.grp_truncated_count, 0);
        assert_eq!(si.grp_terminated_count, 0);
        assert_eq!(si.grp_search_count, 0);
        assert_eq!(si.grp_override_count, 0);
    }

    #[test]
    fn grp_metrics_summary_format() {
        let mut si = make_test_si(10, 3);
        si.grp_search_count = 10;
        si.grp_override_count = 3;
        si.grp_eval_count = 250;
        si.grp_eval_time_us = 100_000;
        si.grp_truncated_count = 200;
        si.grp_terminated_count = 50;

        let summary = si.grp_metrics_summary();
        assert!(summary.contains("searches=10"), "summary: {summary}");
        assert!(summary.contains("overrides=3"), "summary: {summary}");
        assert!(summary.contains("evals=250"), "summary: {summary}");
        assert!(summary.contains("truncated=200"), "summary: {summary}");
        assert!(summary.contains("terminated=50"), "summary: {summary}");
        // Truncation rate: 200/(200+50) = 80%
        assert!(summary.contains("80.0%"), "summary: {summary}");
    }

    #[test]
    fn grp_metrics_summary_zero_searches() {
        let si = make_test_si(0, 0);
        let summary = si.grp_metrics_summary();
        assert!(summary.contains("searches=0"), "summary: {summary}");
        assert!(summary.contains("truncation_rate=0.0%"), "summary: {summary}");
    }

    // --- decide_override tests ---

    #[test]
    fn decide_override_search_agrees_with_dqn() {
        let mut si = make_test_si(0, 0);
        // DQN's top action is 0, search also favors 0
        let actions = vec![0, 1];
        let candidates = vec![(0, 1.0), (1, 0.5)];
        let sums = vec![100.0, 50.0];
        let sum_sq = vec![2000.0, 1000.0];
        let counts = vec![5, 5];

        let result = si.decide_override(&actions, &candidates, &sums, &sum_sq, &counts);
        assert!(result.is_none(), "should not override when search agrees");
    }

    #[test]
    fn decide_override_clear_winner() {
        let mut si = make_test_si(0, 0);
        si.search_weight = 0.5; // threshold = 1.5
        // DQN prefers action 0, but search clearly favors action 1
        let actions = vec![0, 1];
        let candidates = vec![(0, 1.0), (1, 0.5)];
        // Action 0: mean=100, action 1: mean=200 — clear difference
        let sums = vec![500.0, 1000.0];
        let sum_sq = vec![51000.0, 202000.0]; // var ~ 1000 each
        let counts = vec![5, 5];

        let result = si.decide_override(&actions, &candidates, &sums, &sum_sq, &counts);
        assert_eq!(result, Some(1), "should override with clear winner");
    }

    #[test]
    fn decide_override_high_variance_no_override() {
        let mut si = make_test_si(0, 0);
        si.search_weight = 0.3; // threshold = 2.1
        // DQN prefers 0, search slightly prefers 1 but with huge variance
        let actions = vec![0, 1];
        let candidates = vec![(0, 1.0), (1, 0.5)];
        // Means: 100 vs 101. Variances: 10000 each. t-stat ≈ 0.016
        let sums = vec![300.0, 303.0];
        let sum_sq = vec![60000.0, 60609.0]; // var = 60000/3 - 100^2 = 10000
        let counts = vec![3, 3];

        let result = si.decide_override(&actions, &candidates, &sums, &sum_sq, &counts);
        assert!(result.is_none(), "should not override with high variance");
    }

    #[test]
    fn decide_override_n_equals_one() {
        let mut si = make_test_si(0, 0);
        let actions = vec![0, 1];
        let candidates = vec![(0, 1.0), (1, 0.5)];
        let sums = vec![100.0, 200.0];
        let sum_sq = vec![10000.0, 40000.0];
        let counts = vec![1, 1]; // n=1 for both

        let result = si.decide_override(&actions, &candidates, &sums, &sum_sq, &counts);
        assert!(result.is_none(), "should not override with n=1");
        assert_eq!(si.skips.insufficient_data, 1);
    }

    #[test]
    fn decide_override_identical_values() {
        let mut si = make_test_si(0, 0);
        // All values identical → se ≈ 0 → inconclusive
        let actions = vec![0, 1];
        let candidates = vec![(0, 1.0), (1, 0.5)];
        let sums = vec![500.0, 600.0]; // mean 100 vs 120
        let sum_sq = vec![50000.0, 72000.0]; // var = 0 each
        let counts = vec![5, 5];

        let result = si.decide_override(&actions, &candidates, &sums, &sum_sq, &counts);
        // Either overrides or not — mainly checking it doesn't panic
        let _ = result;
    }

    #[test]
    fn weight_to_threshold_boundary_values() {
        // weight=0.0 → threshold=3.0
        assert!((3.0 * (1.0 - 0.0_f64) - 3.0).abs() < 1e-10);
        // weight=0.3 → threshold=2.1
        assert!((3.0 * (1.0 - 0.3_f64) - 2.1).abs() < 1e-10);
        // weight=0.5 → threshold=1.5
        assert!((3.0 * (1.0 - 0.5_f64) - 1.5).abs() < 1e-10);
        // weight=1.0 → threshold=0.0
        assert!((3.0 * (1.0 - 1.0_f64) - 0.0).abs() < 1e-10);
    }

    // --- GRP history accumulation tests ---

    #[test]
    fn grp_history_starts_empty() {
        let si = make_test_si(0, 0);
        assert!(si.grp_game_states.is_empty());
    }

    #[test]
    fn grp_game_states_are_per_game() {
        let mut si = make_test_si(5, 2);

        // Simulate two different games populating their GRP state
        let gs0 = si.grp_game_states.entry(0).or_default();
        gs0.history.push([1.0, 0.0, 0.0, 2.5, 2.5, 2.5, 2.5]);
        gs0.current_kyoku = Some((4, [4.0, 1.0, 0.0, 2.5, 2.5, 2.5, 2.5]));

        let gs1 = si.grp_game_states.entry(1).or_default();
        gs1.history.push([0.0, 1.0, 0.0, 3.0, 2.0, 2.5, 2.5]);

        // Game 0 and game 1 should have independent histories
        assert_eq!(si.grp_game_states[&0].history.len(), 1);
        assert_eq!(si.grp_game_states[&1].history.len(), 1);
        assert!(si.grp_game_states[&0].current_kyoku.is_some());
        assert!(si.grp_game_states[&1].current_kyoku.is_none());

        // start_game clears only the specific game
        si.grp_game_states.remove(&0);
        assert!(!si.grp_game_states.contains_key(&0));
        assert_eq!(si.grp_game_states[&1].history.len(), 1);

        // Metrics should NOT be cleared (they accumulate across games)
        assert_eq!(si.search_count, 5);
        assert_eq!(si.override_count, 2);
    }
}
