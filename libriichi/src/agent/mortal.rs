use super::{BatchAgent, InvisibleState};
use crate::consts::ACTION_SPACE;
use crate::mjai::{Event, EventExt, Metadata};
use crate::search::config::ParticleConfig;
use crate::search::{particle, simulator};
use crate::state::PlayerState;
use crate::{must_tile, tu8};
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
use rand::SeedableRng;
use rand_chacha::ChaCha12Rng;

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
    inconclusive: u32,
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

    fn total(&self) -> u32 {
        self.panics + self.rollout_errors
    }
}

/// Optional search integration that overrides DQN actions with
/// particle-based rollout evaluation at critical decision points.
struct SearchIntegration {
    rng: ChaCha12Rng,
    config: ParticleConfig,
    max_candidates: usize,
    search_weight: f32,

    // Core counts
    search_count: u32,
    override_count: u32,

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

impl SearchIntegration {
    /// Decide whether to override the DQN action with a search-informed one.
    ///
    /// Returns `Some(action)` if search found a better action, `None` to keep DQN choice.
    fn maybe_override(
        &mut self,
        state: &PlayerState,
        q_values: &[f32; ACTION_SPACE],
        masks: &[bool; ACTION_SPACE],
        actor: u8,
    ) -> Option<usize> {
        let start = Instant::now();

        // Quick criticality check: skip search for trivial decisions
        if !Self::is_critical(state) {
            self.skips.low_criticality += 1;
            return None;
        }

        // Get top-k candidate actions by q-value (excluding riichi=37)
        let mut candidates: Vec<(usize, f32)> = q_values
            .iter()
            .enumerate()
            .filter(|&(i, _)| masks[i] && i != 37)
            .map(|(i, &q)| (i, q))
            .collect();
        candidates.sort_by(|a, b| b.1.total_cmp(&a.1));
        candidates.truncate(self.max_candidates);

        // Need at least 2 candidates to justify search
        if candidates.len() < 2 {
            self.skips.insufficient_candidates += 1;
            return None;
        }

        let actions: Vec<usize> = candidates.iter().map(|&(a, _)| a).collect();
        let n_target = self.config.n_particles;

        // Generate particles
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

        // Replay event history once, then reuse for all particle/action pairs.
        let replayed = match simulator::replay_player_states(state) {
            Ok(r) => r,
            Err(e) => {
                self.skips.replay_error += 1;
                log::debug!("replay error: {e}");
                self.record_elapsed(start);
                return None;
            }
        };

        // Derive event-dependent midgame context once (not per particle).
        let base = simulator::derive_midgame_context_base(state.event_history());

        // Round-robin rollouts: particle-first, action-second
        let mut action_sums: Vec<f64> = vec![0.0; actions.len()];
        let mut action_counts: Vec<usize> = vec![0; actions.len()];

        for p in &particles {
            for (ai, &action) in actions.iter().enumerate() {
                let rollout = panic::catch_unwind(panic::AssertUnwindSafe(|| {
                    simulator::simulate_action_rollout_with_base_smart(
                        state, &replayed, p, action, &base, &mut self.rng,
                    )
                }));
                match rollout {
                    Ok(Ok(result)) => {
                        action_sums[ai] += f64::from(result.deltas[actor as usize]);
                        action_counts[ai] += 1;
                    }
                    Ok(Err(e)) => {
                        let msg = e.to_string();
                        self.errors.record_error(&msg);
                        log::debug!("search rollout error: {msg}");
                    }
                    Err(panic_info) => {
                        let msg = panic_info
                            .downcast_ref::<String>()
                            .map(String::as_str)
                            .or_else(|| panic_info.downcast_ref::<&str>().copied())
                            .unwrap_or("unknown panic");
                        self.errors.record_panic(msg);
                        log::warn!("search rollout panic: {msg}");
                    }
                }
            }
        }

        // Compute mean search values
        let search_values: Vec<f64> = action_sums
            .iter()
            .zip(action_counts.iter())
            .map(|(&sum, &count)| if count > 0 { sum / count as f64 } else { f64::NEG_INFINITY })
            .collect();

        // Normalize search values to [0, 1] range
        let search_min = search_values
            .iter()
            .copied()
            .filter(|v| v.is_finite())
            .fold(f64::INFINITY, f64::min);
        let search_max = search_values
            .iter()
            .copied()
            .filter(|v| v.is_finite())
            .fold(f64::NEG_INFINITY, f64::max);
        let search_range = search_max - search_min;

        let normalized_search: Vec<f64> = if search_range > 1e-10 {
            search_values
                .iter()
                .map(|&v| {
                    if v.is_finite() {
                        (v - search_min) / search_range
                    } else {
                        0.0
                    }
                })
                .collect()
        } else {
            // All search values are the same; search is inconclusive
            self.skips.inconclusive += 1;
            self.record_elapsed(start);
            return None;
        };

        // Compute softmax policy from q-values for blending
        let candidate_q: Vec<f32> = candidates.iter().map(|&(_, q)| q).collect();
        let q_max = candidate_q
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);
        let exp_q: Vec<f64> = candidate_q
            .iter()
            .map(|&q| f64::from(q - q_max).exp())
            .collect();
        let exp_sum: f64 = exp_q.iter().sum();
        let policy: Vec<f64> = exp_q.iter().map(|&e| e / exp_sum).collect();

        // Blend: w * normalized_search + (1-w) * policy
        let w = f64::from(self.search_weight);
        let blended: Vec<f64> = normalized_search
            .iter()
            .zip(policy.iter())
            .map(|(&s, &p)| w.mul_add(s, (1.0 - w) * p))
            .collect();

        // Find best blended action
        let best_idx = blended
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.total_cmp(b))?
            .0;

        let best_action = actions[best_idx];
        let dqn_action = candidates[0].0;

        self.record_elapsed(start);

        // Only override if search disagrees with DQN
        if best_action != dqn_action {
            self.override_count += 1;
            Some(best_action)
        } else {
            None
        }
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
    ///
    /// Simplified from criticality.py (4 of 6 factors; entropy and score
    /// proximity are omitted to avoid complexity in Rust).
    /// 1. Riichi: any *opponent* in riichi increases criticality
    /// 2. Shanten: tenpai/iishanten is more critical
    /// 3. Bakaze/kyoku: later rounds are more critical
    /// 4. Danger: many tiles discarded = more information/danger
    fn is_critical(state: &PlayerState) -> bool {
        let mut score: f32 = 0.0;

        // Factor 1: Opponent riichi (0.3 if any opponent in riichi)
        // Index 0 is self (relative), 1-3 are opponents.
        // Skip self: when we're in riichi, discards are forced (tsumogiri).
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
        // tiles_left starts at 70 (decremented per draw), critical when < 30
        if state.tiles_left() < 30 {
            score += 0.15;
        }

        // Threshold: 0.25 means at least one significant factor must be present
        score >= 0.25
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
            + self.skips.inconclusive;
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
             particle_err={}, replay_err={}, inconclusive={})\n\
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
            self.skips.inconclusive,
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
                    .unwrap_or(0.5);
                Some((n_particles, seed, weight))
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

        let search = search_cfg.map(|(n_particles, seed, weight)| {
            let rng = match seed {
                Some(s) => ChaCha12Rng::seed_from_u64(s),
                None => ChaCha12Rng::from_os_rng(),
            };
            SearchIntegration {
                rng,
                config: ParticleConfig::new(n_particles),
                max_candidates: 5,
                search_weight: weight,
                search_count: 0_u32,
                override_count: 0_u32,
                skips: SkipMetrics::default(),
                errors: ErrorMetrics::default(),
                search_times_us: Vec::new(),
                particles_requested: 0_u32,
                particles_generated: 0_u32,
                particle_gen_attempts: 0_u32,
            }
        });

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
                search
                    .maybe_override(
                        state,
                        &self.q_values[action_idx],
                        &self.masks_recv[action_idx],
                        actor,
                    )
                    .unwrap_or(action)
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
        assert_eq!(sm.inconclusive, 0);
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

    #[test]
    fn search_integration_drop_does_not_panic() {
        // Verify that dropping with zero counts doesn't panic
        let si = SearchIntegration {
            rng: ChaCha12Rng::seed_from_u64(0),
            config: ParticleConfig::new(10),
            max_candidates: 5,
            search_weight: 0.5,
            search_count: 0,
            override_count: 0,
            skips: SkipMetrics::default(),
            errors: ErrorMetrics::default(),
            search_times_us: Vec::new(),
            particles_requested: 0,
            particles_generated: 0,
            particle_gen_attempts: 0,
        };
        drop(si); // should not panic
    }

    #[test]
    fn search_integration_drop_with_data_does_not_panic() {
        let mut si = SearchIntegration {
            rng: ChaCha12Rng::seed_from_u64(0),
            config: ParticleConfig::new(50),
            max_candidates: 5,
            search_weight: 0.5,
            search_count: 10,
            override_count: 3,
            skips: SkipMetrics {
                low_criticality: 20,
                insufficient_candidates: 5,
                no_particles: 2,
                particle_gen_error: 1,
                replay_error: 0,
                inconclusive: 1,
            },
            errors: ErrorMetrics::default(),
            search_times_us: vec![1000, 2000, 3000, 5000, 10000],
            particles_requested: 500,
            particles_generated: 450,
            particle_gen_attempts: 600,
        };
        si.errors.record_panic("test panic");
        si.errors.record_error("test error");
        drop(si); // should not panic
    }
}
