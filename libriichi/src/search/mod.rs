pub mod config;
pub mod grp;
pub mod heuristic;
pub mod particle;
pub mod policy;
pub mod simulator;

use crate::py_helper::add_submodule;
use crate::state::PlayerState;

use config::ParticleConfig;
use grp::GrpEvaluator;
use particle::Particle;
use simulator::{RolloutResult, TruncatedResult};

use pyo3::prelude::*;
use rand::prelude::*;
use rand::SeedableRng;
use rand_chacha::ChaCha12Rng;
use rayon::prelude::*;
use std::time::Instant;

/// PyO3-exposed search module entry point.
#[pyclass]
pub struct SearchModule {
    config: ParticleConfig,
    rng: ChaCha12Rng,
    use_smart_policy: bool,
    last_gen_attempts: usize,
    last_gen_accepted: usize,
    last_simulate_time_ms: f64,
    last_evaluate_time_ms: f64,
}

#[pymethods]
impl SearchModule {
    #[new]
    #[must_use]
    pub fn new(n_particles: usize) -> Self {
        Self {
            config: ParticleConfig::new(n_particles),
            rng: ChaCha12Rng::from_os_rng(),
            use_smart_policy: true,
            last_gen_attempts: 0,
            last_gen_accepted: 0,
            last_simulate_time_ms: 0.0,
            last_evaluate_time_ms: 0.0,
        }
    }

    /// Create with a fixed seed for reproducibility.
    #[staticmethod]
    #[must_use]
    pub fn with_seed(n_particles: usize, seed: u64) -> Self {
        Self {
            config: ParticleConfig::new(n_particles),
            rng: ChaCha12Rng::seed_from_u64(seed),
            use_smart_policy: true,
            last_gen_attempts: 0,
            last_gen_accepted: 0,
            last_simulate_time_ms: 0.0,
            last_evaluate_time_ms: 0.0,
        }
    }

    /// Generate particles from the current game state.
    pub fn generate_particles(&mut self, state: &PlayerState) -> PyResult<Vec<Particle>> {
        let (particles, attempts) =
            particle::generate_particles(state, &self.config, &mut self.rng)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        self.last_gen_attempts = attempts;
        self.last_gen_accepted = particles.len();
        Ok(particles)
    }

    /// Get the current configuration.
    #[getter]
    #[must_use]
    pub fn config(&self) -> ParticleConfig {
        self.config.clone()
    }

    /// Set the particle configuration.
    #[setter]
    pub const fn set_config(&mut self, config: ParticleConfig) {
        self.config = config;
    }

    /// Whether to use the smart heuristic policy during rollouts.
    /// When true, rollouts use shanten/ukeire/safety-based discards.
    /// When false, rollouts use tsumogiri (discard drawn tile).
    #[getter]
    #[must_use]
    pub const fn use_smart_policy(&self) -> bool {
        self.use_smart_policy
    }

    /// Set whether to use the smart heuristic policy during rollouts.
    #[setter]
    pub const fn set_use_smart_policy(&mut self, value: bool) {
        self.use_smart_policy = value;
    }

    /// Number of attempts made during the last `generate_particles` call.
    #[getter]
    #[must_use]
    pub const fn last_gen_attempts(&self) -> usize {
        self.last_gen_attempts
    }

    /// Number of accepted particles during the last `generate_particles` call.
    #[getter]
    #[must_use]
    pub const fn last_gen_accepted(&self) -> usize {
        self.last_gen_accepted
    }

    /// Wall-clock time (ms) of the last `generate_and_simulate` call.
    #[getter]
    #[must_use]
    pub const fn last_simulate_time_ms(&self) -> f64 {
        self.last_simulate_time_ms
    }

    /// Wall-clock time (ms) of the last `evaluate_actions` call.
    #[getter]
    #[must_use]
    pub const fn last_evaluate_time_ms(&self) -> f64 {
        self.last_evaluate_time_ms
    }

    /// Simulate a single particle rollout using tsumogiri strategy.
    pub fn simulate_particle(
        &self,
        state: &PlayerState,
        particle: &Particle,
    ) -> PyResult<RolloutResult> {
        simulator::simulate_particle(state, particle)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    /// Generate particles and simulate all of them in parallel, returning results.
    ///
    /// Replays event history once and derives the midgame context once,
    /// then runs rollouts in parallel across particles using rayon.
    ///
    /// Note: does not release the Python GIL via `py.allow_threads()` because
    /// test code calls this from pure Rust without a Python context. The parallel
    /// section takes ~4ms (vs seconds for arena functions that use allow_threads),
    /// so GIL contention is negligible. Revisit if Python multithreading is added.
    pub fn generate_and_simulate(
        &mut self,
        state: &PlayerState,
    ) -> PyResult<Vec<RolloutResult>> {
        let start = Instant::now();

        let (particles, attempts) =
            particle::generate_particles(state, &self.config, &mut self.rng)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        self.last_gen_attempts = attempts;
        self.last_gen_accepted = particles.len();

        let replayed = simulator::replay_player_states(state)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        // Derive event-dependent context once, reuse across all particles
        let base = simulator::derive_midgame_context_base(state.event_history());

        // Pre-generate per-particle RNG seeds
        let use_smart = self.use_smart_policy;
        let seeds: Vec<u64> = (0..particles.len())
            .map(|_| self.rng.next_u64())
            .collect();

        // Parallel rollouts across particles
        let results: Result<Vec<RolloutResult>, _> = particles
            .par_iter()
            .enumerate()
            .map(|(i, p)| {
                let board_state =
                    simulator::build_midgame_board_state_with_base(state, &replayed, p, &base)?;
                let initial_scores = board_state.board.scores;
                if use_smart {
                    let mut thread_rng = ChaCha12Rng::seed_from_u64(seeds[i]);
                    simulator::run_rollout_smart(board_state, initial_scores, &mut thread_rng)
                } else {
                    simulator::run_rollout_tsumogiri(board_state, initial_scores)
                }
            })
            .collect();

        self.last_simulate_time_ms = start.elapsed().as_secs_f64() * 1000.0;

        results.map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    /// Simulate a single particle rollout with a specific action injected.
    ///
    /// The action is applied at our player's first matching decision point,
    /// then the rest of the game runs with tsumogiri.
    pub fn simulate_action(
        &self,
        state: &PlayerState,
        particle: &Particle,
        action: usize,
    ) -> PyResult<RolloutResult> {
        simulator::simulate_action_rollout(state, particle, action)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    /// Evaluate multiple actions with truncated rollouts in parallel.
    ///
    /// Same as `evaluate_actions` but uses truncated rollouts that stop after
    /// `max_steps` heuristic steps. Returns `TruncatedResult` with leaf state
    /// information needed for GRP evaluation.
    pub fn evaluate_actions_truncated(
        &mut self,
        state: &PlayerState,
        particles: Vec<Particle>,
        actions: Vec<usize>,
        max_steps: u32,
    ) -> PyResult<std::collections::HashMap<usize, Vec<TruncatedResult>>> {
        let start = Instant::now();

        let replayed = simulator::replay_player_states(state)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        let base = simulator::derive_midgame_context_base(state.event_history());

        let use_smart = self.use_smart_policy;
        let player_id = state.player_id();
        let n_particles = particles.len();
        let seeds: Vec<u64> = (0..n_particles)
            .map(|_| self.rng.next_u64())
            .collect();

        let per_particle: Result<Vec<Vec<(usize, TruncatedResult)>>, anyhow::Error> = particles
            .into_par_iter()
            .enumerate()
            .map(|(i, particle)| {
                let board_state = simulator::build_midgame_board_state_with_base(
                    state, &replayed, &particle, &base,
                )?;
                let initial_scores = board_state.board.scores;

                let mut particle_rng = ChaCha12Rng::seed_from_u64(seeds[i]);

                let mut action_results = Vec::with_capacity(actions.len());
                for &action in &actions {
                    let action_seed = particle_rng.next_u64();
                    let mut action_rng = if use_smart {
                        Some(ChaCha12Rng::seed_from_u64(action_seed))
                    } else {
                        None
                    };

                    let result = simulator::simulate_action_rollout_prebuilt_truncated(
                        &board_state,
                        initial_scores,
                        player_id,
                        action,
                        action_rng.as_mut(),
                        max_steps,
                    )?;
                    action_results.push((action, result));
                }

                Ok(action_results)
            })
            .collect();

        self.last_evaluate_time_ms = start.elapsed().as_secs_f64() * 1000.0;

        let per_particle = per_particle
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        let mut results = std::collections::HashMap::new();
        for &action in &actions {
            results.insert(action, Vec::with_capacity(n_particles));
        }
        for particle_results in per_particle {
            for (action, result) in particle_results {
                results.get_mut(&action).unwrap().push(result);
            }
        }

        Ok(results)
    }

    /// Evaluate multiple actions across multiple particles in parallel.
    ///
    /// Replays event history once and derives midgame context once. For each
    /// particle (in parallel), builds the BoardState once and clones it K
    /// times for K actions (build-once-clone-K optimization).
    ///
    /// Uses rayon's `par_iter().map().collect()` which preserves ordering,
    /// ensuring deterministic results for the same seed.
    /// Returns a dict mapping action index to list of RolloutResults.
    pub fn evaluate_actions(
        &mut self,
        state: &PlayerState,
        particles: Vec<Particle>,
        actions: Vec<usize>,
    ) -> PyResult<std::collections::HashMap<usize, Vec<RolloutResult>>> {
        let start = Instant::now();

        // Replay once
        let replayed = simulator::replay_player_states(state)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        // Derive event-dependent context once
        let base = simulator::derive_midgame_context_base(state.event_history());

        // Pre-generate per-particle RNG seeds
        let use_smart = self.use_smart_policy;
        let player_id = state.player_id();
        let n_particles = particles.len();
        let seeds: Vec<u64> = (0..n_particles)
            .map(|_| self.rng.next_u64())
            .collect();

        // Parallel iteration over particles, sequential over actions.
        // collect() preserves particle ordering for deterministic results.
        // into_par_iter() avoids reference indirection since particles is owned.
        let per_particle: Result<Vec<Vec<(usize, RolloutResult)>>, anyhow::Error> = particles
            .into_par_iter()
            .enumerate()
            .map(|(i, particle)| {
                // Build BoardState once per particle
                let board_state = simulator::build_midgame_board_state_with_base(
                    state, &replayed, &particle, &base,
                )?;
                let initial_scores = board_state.board.scores;

                // Derive per-action seeds from the particle RNG so that each
                // action's rollout is independent — evaluating {0,1} gives the
                // same result for action 0 as evaluating {0,1,2}.
                let mut particle_rng = ChaCha12Rng::seed_from_u64(seeds[i]);

                // Sequential iteration over actions (K is small)
                let mut action_results = Vec::with_capacity(actions.len());
                for &action in &actions {
                    let action_seed = particle_rng.next_u64();
                    let mut action_rng = if use_smart {
                        Some(ChaCha12Rng::seed_from_u64(action_seed))
                    } else {
                        None
                    };

                    let rollout_result = simulator::simulate_action_rollout_prebuilt(
                        &board_state,
                        initial_scores,
                        player_id,
                        action,
                        action_rng.as_mut(),
                    )?;
                    action_results.push((action, rollout_result));
                }

                Ok(action_results)
            })
            .collect();

        self.last_evaluate_time_ms = start.elapsed().as_secs_f64() * 1000.0;

        let per_particle = per_particle
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        // Merge per-particle results in deterministic order
        let mut results = std::collections::HashMap::new();
        for &action in &actions {
            results.insert(action, Vec::with_capacity(n_particles));
        }
        for particle_results in per_particle {
            for (action, result) in particle_results {
                results.get_mut(&action).unwrap().push(result);
            }
        }

        Ok(results)
    }

    /// Collect leaf observation tensors from truncated rollouts.
    ///
    /// Runs the same particle generation + truncated rollout logic as
    /// `evaluate_actions_truncated`, but additionally collects the
    /// observation tensor and action mask at each leaf state.
    ///
    /// Returns a list of `(action_idx, obs_flat, mask_f32, terminated, deltas)`
    /// tuples that Python can batch-evaluate on GPU.
    ///
    /// - `obs_flat`: flattened `(channels * 34)` f32 observation tensor
    /// - `mask_f32`: 46-element f32 mask (1.0 = valid, 0.0 = invalid)
    /// - `terminated`: whether the rollout ended naturally
    /// - `deltas`: score deltas [i32; 4] (absolute seats)
    pub fn collect_leaf_obs(
        &mut self,
        state: &PlayerState,
        actions: Vec<usize>,
        max_steps: u32,
        obs_version: u32,
    ) -> PyResult<Vec<(usize, Vec<f32>, Vec<f32>, bool, [i32; 4])>> {
        let start = Instant::now();

        let (particles, attempts) =
            particle::generate_particles(state, &self.config, &mut self.rng)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        self.last_gen_attempts = attempts;
        self.last_gen_accepted = particles.len();

        if particles.is_empty() {
            return Ok(Vec::new());
        }

        let replayed = simulator::replay_player_states(state)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        let base = simulator::derive_midgame_context_base(state.event_history());

        let use_smart = self.use_smart_policy;
        let player_id = state.player_id();
        let seeds: Vec<u64> = (0..particles.len())
            .map(|_| self.rng.next_u64())
            .collect();

        let per_particle: Result<Vec<Vec<(usize, TruncatedResult)>>, anyhow::Error> = particles
            .into_par_iter()
            .enumerate()
            .map(|(i, particle)| {
                let board_state = simulator::build_midgame_board_state_with_base(
                    state, &replayed, &particle, &base,
                )?;
                let initial_scores = board_state.board.scores;

                let mut particle_rng = ChaCha12Rng::seed_from_u64(seeds[i]);

                let mut action_results = Vec::with_capacity(actions.len());
                for &action in &actions {
                    let action_seed = particle_rng.next_u64();
                    let mut action_rng = if use_smart {
                        Some(ChaCha12Rng::seed_from_u64(action_seed))
                    } else {
                        None
                    };

                    let result = simulator::simulate_action_rollout_prebuilt_truncated_with_obs(
                        &board_state,
                        initial_scores,
                        player_id,
                        action,
                        action_rng.as_mut(),
                        max_steps,
                        obs_version,
                    )?;
                    action_results.push((action, result));
                }

                Ok(action_results)
            })
            .collect();

        self.last_evaluate_time_ms = start.elapsed().as_secs_f64() * 1000.0;

        let per_particle = per_particle
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        // Flatten into Python-friendly tuples
        let mut entries = Vec::new();
        for particle_results in per_particle {
            for (action, result) in particle_results {
                if let (Some(obs), Some(mask)) = (result.leaf_obs, result.leaf_mask) {
                    entries.push((action, obs, mask, result.terminated, result.deltas));
                }
            }
        }

        Ok(entries)
    }
}

#[cfg(test)]
pub(crate) mod test_utils {
    use crate::mjai::Event;
    use crate::state::PlayerState;
    use crate::tile::Tile;

    /// Set up a basic game state at player 0's discard decision.
    ///
    /// Hand: 1-9m, 1-4p (13 tiles), tsumo 5p (14 tiles).
    /// Player 0 is oya. East round, kyoku 1, honba 0, all scores 25000.
    pub fn setup_basic_game() -> PlayerState {
        let mut state = PlayerState::new(0);
        state.set_record_events(true);

        let bakaze: Tile = "E".parse().unwrap();
        let dora_marker: Tile = "1m".parse().unwrap();

        let tehais = [
            [
                "1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "1p", "2p", "3p", "4p",
            ]
            .map(|s| s.parse::<Tile>().unwrap()),
            ["?"; 13].map(|s| s.parse::<Tile>().unwrap()),
            ["?"; 13].map(|s| s.parse::<Tile>().unwrap()),
            ["?"; 13].map(|s| s.parse::<Tile>().unwrap()),
        ];

        let start_event = Event::StartKyoku {
            bakaze,
            dora_marker,
            kyoku: 1,
            honba: 0,
            kyotaku: 0,
            oya: 0,
            scores: [25000; 4],
            tehais,
        };
        state.update(&start_event).unwrap();

        // First tsumo
        let tsumo_event = Event::Tsumo {
            actor: 0,
            pai: "5p".parse().unwrap(),
        };
        state.update(&tsumo_event).unwrap();

        state
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::search::test_utils::setup_basic_game;

    #[test]
    fn test_boardstate_clone_independence() {
        // Clone a BoardState, step the original forward, verify clone is unchanged.
        use crate::search::config::ParticleConfig;
        use crate::search::particle::generate_particles;

        let state = setup_basic_game();
        let config = ParticleConfig::new(1);
        let mut rng = ChaCha12Rng::seed_from_u64(42);

        let (particles, _) = generate_particles(&state, &config, &mut rng).unwrap();
        let bs = simulator::build_midgame_board_state(&state, &particles[0]).unwrap();

        // Clone the BoardState
        let clone = bs.clone();

        // Verify fields match before mutation
        assert_eq!(bs.board.scores, clone.board.scores);
        assert_eq!(bs.board.kyoku, clone.board.kyoku);
        assert_eq!(bs.tiles_left, clone.tiles_left);
        assert_eq!(bs.board.yama.len(), clone.board.yama.len());

        // Mutate the clone by stepping it forward
        let mut mutated = clone;
        let reactions: [crate::mjai::EventExt; 4] = Default::default();
        // Step the mutated copy forward — it may change tiles_left, yama, etc.
        drop(mutated.poll(reactions));

        // Original should be untouched
        assert_eq!(bs.board.scores, [25000; 4]);
        assert_eq!(bs.board.kyoku, 0);
    }

    #[test]
    fn test_simulate_action_rollout_prebuilt_correctness() {
        // Verify prebuilt function produces same results as existing
        // simulate_action_rollout_with_base_smart for the same inputs.
        use crate::search::config::ParticleConfig;
        use crate::search::particle::generate_particles;

        let state = setup_basic_game();
        let config = ParticleConfig::new(5);
        let mut rng = ChaCha12Rng::seed_from_u64(42);

        let (particles, _) = generate_particles(&state, &config, &mut rng).unwrap();
        let replayed = simulator::replay_player_states(&state).unwrap();
        let base = simulator::derive_midgame_context_base(state.event_history());

        let action = 0_usize; // discard 1m
        let player_id = state.player_id();

        for particle in &particles {
            // Method 1: existing approach (build + run_rollout internally)
            let mut rng1 = ChaCha12Rng::seed_from_u64(999);
            let result1 = simulator::simulate_action_rollout_with_base_smart(
                &state, &replayed, particle, action, &base, &mut rng1,
            )
            .unwrap();

            // Method 2: prebuilt approach (build once, clone K times)
            let mut rng2 = ChaCha12Rng::seed_from_u64(999);
            let board_state = simulator::build_midgame_board_state_with_base(
                &state, &replayed, particle, &base,
            )
            .unwrap();
            let initial_scores = board_state.board.scores;
            let result2 = simulator::simulate_action_rollout_prebuilt(
                &board_state,
                initial_scores,
                player_id,
                action,
                Some(&mut rng2),
            )
            .unwrap();

            assert_eq!(
                result1.deltas, result2.deltas,
                "prebuilt should match existing approach"
            );
            assert_eq!(result1.scores, result2.scores);
            assert_eq!(result1.has_hora, result2.has_hora);
            assert_eq!(result1.steps, result2.steps);
        }
    }

    #[test]
    fn test_generate_and_simulate_determinism() {
        // Same seed produces same results across multiple calls.
        let state = setup_basic_game();

        let mut sm1 = SearchModule::with_seed(10, 42);
        let results1 = sm1.generate_and_simulate(&state).unwrap();

        let mut sm2 = SearchModule::with_seed(10, 42);
        let results2 = sm2.generate_and_simulate(&state).unwrap();

        assert_eq!(results1.len(), results2.len());
        for (r1, r2) in results1.iter().zip(results2.iter()) {
            assert_eq!(r1.deltas, r2.deltas, "determinism: deltas should match");
            assert_eq!(r1.scores, r2.scores, "determinism: scores should match");
        }
    }

    #[test]
    fn test_evaluate_actions_determinism() {
        // Same seed produces same results across multiple calls.
        let state = setup_basic_game();

        let mut sm1 = SearchModule::with_seed(10, 42);
        let particles1 = sm1.generate_particles(&state).unwrap();
        let actions = vec![0_usize, 8, 13]; // discard 1m, 9m, 5p
        let results1 = sm1.evaluate_actions(&state, particles1.clone(), actions.clone()).unwrap();

        let mut sm2 = SearchModule::with_seed(10, 42);
        let particles2 = sm2.generate_particles(&state).unwrap();
        let results2 = sm2.evaluate_actions(&state, particles2, actions.clone()).unwrap();

        for &action in &actions {
            let r1 = &results1[&action];
            let r2 = &results2[&action];
            assert_eq!(r1.len(), r2.len(), "action {action}: result counts should match");
            for (a, b) in r1.iter().zip(r2.iter()) {
                assert_eq!(a.deltas, b.deltas, "action {action}: deltas should match");
                assert_eq!(a.scores, b.scores, "action {action}: scores should match");
            }
        }
    }

    #[test]
    fn test_evaluate_actions_correctness() {
        // Results have expected structure: each action gets N results, scores are valid.
        let state = setup_basic_game();
        let mut sm = SearchModule::with_seed(10, 42);
        let particles = sm.generate_particles(&state).unwrap();
        let n = particles.len();
        let actions = vec![0_usize, 8, 13];
        let results = sm.evaluate_actions(&state, particles, actions.clone()).unwrap();

        assert_eq!(results.len(), actions.len(), "should have one entry per action");
        for &action in &actions {
            let action_results = &results[&action];
            assert_eq!(
                action_results.len(),
                n,
                "action {action} should have {n} results"
            );
            for r in action_results {
                // Deltas should sum to at most 1000 (riichi sticks)
                let delta_sum: i32 = r.deltas.iter().sum();
                assert!(
                    delta_sum.abs() <= 1000,
                    "action {action}: delta sum {delta_sum} too large"
                );
            }
        }
    }

    #[test]
    fn test_evaluate_actions_build_once_clone_k() {
        // Verify that the build-once-clone-K pattern produces identical results
        // to the old per-pair rebuild pattern.
        //
        // We compare prebuilt (clone-based) against explicit rebuild for each action.
        use crate::search::config::ParticleConfig;
        use crate::search::particle::generate_particles;

        let state = setup_basic_game();
        let config = ParticleConfig::new(5);
        let mut rng = ChaCha12Rng::seed_from_u64(42);

        let (particles, _) = generate_particles(&state, &config, &mut rng).unwrap();
        let replayed = simulator::replay_player_states(&state).unwrap();
        let base = simulator::derive_midgame_context_base(state.event_history());
        let player_id = state.player_id();
        let actions = [0_usize, 8, 13];

        for particle in &particles {
            // Build once
            let board_state = simulator::build_midgame_board_state_with_base(
                &state, &replayed, particle, &base,
            )
            .unwrap();
            let initial_scores = board_state.board.scores;

            for &action in &actions {
                // Clone-K approach
                let mut rng_clone = ChaCha12Rng::seed_from_u64(777);
                let result_clone = simulator::simulate_action_rollout_prebuilt(
                    &board_state,
                    initial_scores,
                    player_id,
                    action,
                    Some(&mut rng_clone),
                )
                .unwrap();

                // Rebuild approach (existing)
                let mut rng_rebuild = ChaCha12Rng::seed_from_u64(777);
                let result_rebuild = simulator::simulate_action_rollout_with_base_smart(
                    &state, &replayed, particle, action, &base, &mut rng_rebuild,
                )
                .unwrap();

                assert_eq!(
                    result_clone.deltas, result_rebuild.deltas,
                    "clone-K vs rebuild: action {action} deltas should match"
                );
                assert_eq!(
                    result_clone.scores, result_rebuild.scores,
                    "clone-K vs rebuild: action {action} scores should match"
                );
            }
        }
    }

    #[test]
    fn test_timing_metrics_populated() {
        // Verify that timing fields are set after generate_and_simulate / evaluate_actions.
        let state = setup_basic_game();

        let mut sm = SearchModule::with_seed(5, 42);
        assert_eq!(sm.last_simulate_time_ms(), 0.0);
        assert_eq!(sm.last_evaluate_time_ms(), 0.0);

        drop(sm.generate_and_simulate(&state).unwrap());
        assert!(
            sm.last_simulate_time_ms() > 0.0,
            "simulate time should be > 0 after call"
        );

        let particles = sm.generate_particles(&state).unwrap();
        drop(sm.evaluate_actions(&state, particles, vec![0, 13]).unwrap());
        assert!(
            sm.last_evaluate_time_ms() > 0.0,
            "evaluate time should be > 0 after call"
        );
    }

    #[test]
    fn test_generate_and_simulate_produces_results() {
        // Basic smoke test: generate_and_simulate returns non-empty results.
        let state = setup_basic_game();
        let mut sm = SearchModule::with_seed(10, 42);
        let results = sm.generate_and_simulate(&state).unwrap();
        assert!(
            !results.is_empty(),
            "generate_and_simulate should return results"
        );
    }

    #[test]
    fn test_evaluate_actions_empty_actions() {
        let state = setup_basic_game();
        let mut sm = SearchModule::with_seed(5, 42);
        let particles = sm.generate_particles(&state).unwrap();
        let results = sm.evaluate_actions(&state, particles, vec![]).unwrap();
        assert_eq!(results.len(), 0);
    }

    #[test]
    fn test_evaluate_actions_empty_particles() {
        let state = setup_basic_game();
        let mut sm = SearchModule::with_seed(5, 42);
        let results = sm.evaluate_actions(&state, vec![], vec![0, 8]).unwrap();
        for (_, v) in &results {
            assert!(v.is_empty());
        }
    }

    #[test]
    fn test_evaluate_actions_action_independence() {
        // Verify that evaluating {0, 8} gives the same results for action 0
        // as evaluating {0, 8, 13}. This confirms per-action RNG seeding
        // makes actions independent of which other actions are evaluated.
        let state = setup_basic_game();

        let mut sm1 = SearchModule::with_seed(5, 42);
        let particles1 = sm1.generate_particles(&state).unwrap();
        let results_2 = sm1
            .evaluate_actions(&state, particles1, vec![0, 8])
            .unwrap();

        let mut sm2 = SearchModule::with_seed(5, 42);
        let particles2 = sm2.generate_particles(&state).unwrap();
        let results_3 = sm2
            .evaluate_actions(&state, particles2, vec![0, 8, 13])
            .unwrap();

        // Action 0 results should be identical regardless of action set size
        let r0_2 = &results_2[&0];
        let r0_3 = &results_3[&0];
        assert_eq!(r0_2.len(), r0_3.len());
        for (a, b) in r0_2.iter().zip(r0_3.iter()) {
            assert_eq!(
                a.deltas, b.deltas,
                "action 0: deltas should match regardless of action set"
            );
        }

        // Action 8 results should also be identical
        let r8_2 = &results_2[&8];
        let r8_3 = &results_3[&8];
        for (a, b) in r8_2.iter().zip(r8_3.iter()) {
            assert_eq!(
                a.deltas, b.deltas,
                "action 8: deltas should match regardless of action set"
            );
        }
    }

    #[test]
    fn test_evaluate_actions_truncated_basic() {
        // Truncated evaluation should return TruncatedResults with correct structure.
        let state = setup_basic_game();
        let mut sm = SearchModule::with_seed(5, 42);
        let particles = sm.generate_particles(&state).unwrap();
        let n = particles.len();
        let actions = vec![0_usize, 8, 13];
        let results = sm
            .evaluate_actions_truncated(&state, particles, actions.clone(), 10)
            .unwrap();

        assert_eq!(results.len(), actions.len());
        for &action in &actions {
            let action_results = &results[&action];
            assert_eq!(action_results.len(), n);
            for r in action_results {
                assert!(r.steps <= 10, "truncated should stop at max_steps");
                // kyoku should be reasonable
                assert!(r.kyoku <= 15, "kyoku={} too large", r.kyoku);
            }
        }
    }

    #[test]
    fn test_evaluate_actions_truncated_determinism() {
        let state = setup_basic_game();

        let mut sm1 = SearchModule::with_seed(5, 42);
        let particles1 = sm1.generate_particles(&state).unwrap();
        let results1 = sm1
            .evaluate_actions_truncated(&state, particles1, vec![0, 8], 10)
            .unwrap();

        let mut sm2 = SearchModule::with_seed(5, 42);
        let particles2 = sm2.generate_particles(&state).unwrap();
        let results2 = sm2
            .evaluate_actions_truncated(&state, particles2, vec![0, 8], 10)
            .unwrap();

        for &action in &[0, 8] {
            let r1 = &results1[&action];
            let r2 = &results2[&action];
            assert_eq!(r1.len(), r2.len());
            for (a, b) in r1.iter().zip(r2.iter()) {
                assert_eq!(a.deltas, b.deltas, "truncated determinism: deltas");
                assert_eq!(a.scores, b.scores, "truncated determinism: scores");
                assert_eq!(a.terminated, b.terminated, "truncated determinism: terminated");
                assert_eq!(a.steps, b.steps, "truncated determinism: steps");
            }
        }
    }

    #[test]
    fn test_evaluate_actions_truncated_particle_count() {
        // Each action should have exactly N results (one per particle)
        let state = setup_basic_game();
        let mut sm = SearchModule::with_seed(5, 42);
        let particles = sm.generate_particles(&state).unwrap();
        let n = particles.len();
        let actions = vec![0_usize, 8];
        let results = sm
            .evaluate_actions_truncated(&state, particles, actions.clone(), 10)
            .unwrap();

        assert_eq!(results.len(), actions.len());
        for &action in &actions {
            let action_results = &results[&action];
            assert_eq!(
                action_results.len(),
                n,
                "action {action} should have exactly {n} results (one per particle)"
            );
        }
    }

    #[test]
    fn test_evaluate_actions_truncated_action_independence() {
        // Evaluating {0, 8} should give same action-0 results as {0, 8, 13}
        let state = setup_basic_game();

        let mut sm1 = SearchModule::with_seed(5, 42);
        let particles1 = sm1.generate_particles(&state).unwrap();
        let results_2 = sm1
            .evaluate_actions_truncated(&state, particles1, vec![0, 8], 10)
            .unwrap();

        let mut sm2 = SearchModule::with_seed(5, 42);
        let particles2 = sm2.generate_particles(&state).unwrap();
        let results_3 = sm2
            .evaluate_actions_truncated(&state, particles2, vec![0, 8, 13], 10)
            .unwrap();

        // Action 0 results should be identical
        let r0_2 = &results_2[&0];
        let r0_3 = &results_3[&0];
        assert_eq!(r0_2.len(), r0_3.len());
        for (a, b) in r0_2.iter().zip(r0_3.iter()) {
            assert_eq!(a.deltas, b.deltas, "action 0 deltas should match");
            assert_eq!(a.scores, b.scores, "action 0 scores should match");
            assert_eq!(a.steps, b.steps, "action 0 steps should match");
            assert_eq!(a.terminated, b.terminated, "action 0 terminated should match");
        }

        // Action 8 results should also be identical
        let r8_2 = &results_2[&8];
        let r8_3 = &results_3[&8];
        for (a, b) in r8_2.iter().zip(r8_3.iter()) {
            assert_eq!(a.deltas, b.deltas, "action 8 deltas should match");
            assert_eq!(a.scores, b.scores, "action 8 scores should match");
        }
    }

    #[test]
    fn test_evaluate_actions_truncated_empty_particles() {
        let state = setup_basic_game();
        let mut sm = SearchModule::with_seed(5, 42);
        let results = sm
            .evaluate_actions_truncated(&state, vec![], vec![0, 8], 10)
            .unwrap();
        for (_, v) in &results {
            assert!(v.is_empty());
        }
    }

    #[test]
    fn test_evaluate_actions_truncated_max_steps_zero() {
        // max_steps=0: pure value truncation, no game steps taken
        let state = setup_basic_game();
        let mut sm = SearchModule::with_seed(5, 42);
        let particles = sm.generate_particles(&state).unwrap();
        let results = sm
            .evaluate_actions_truncated(&state, particles, vec![0, 8], 0)
            .unwrap();

        for &action in &[0_usize, 8] {
            for r in &results[&action] {
                assert_eq!(r.steps, 0, "max_steps=0 should have 0 steps");
                assert!(!r.terminated, "max_steps=0 should not be terminated");
                assert_eq!(r.deltas, [0, 0, 0, 0], "max_steps=0 should have zero deltas");
            }
        }
    }

    #[test]
    fn test_collect_leaf_obs_basic() {
        // Verify collect_leaf_obs returns entries with properly shaped obs/mask.
        let state = setup_basic_game();
        let mut sm = SearchModule::with_seed(5, 42);

        let entries = sm
            .collect_leaf_obs(&state, vec![0, 8], 100, 4)
            .unwrap();

        // Should have some entries (5 particles × 2 actions = up to 10)
        assert!(!entries.is_empty(), "should collect at least some leaf obs");

        // Verify obs shape: v4 has 1012 channels × 34 = 34408 elements
        let expected_obs_len = 1012 * 34;
        let expected_mask_len = 46;

        for (action, obs, mask, _terminated, deltas) in &entries {
            assert!(
                *action == 0 || *action == 8,
                "action should be 0 or 8, got {action}"
            );
            assert_eq!(
                obs.len(),
                expected_obs_len,
                "obs should have {expected_obs_len} elements, got {}",
                obs.len()
            );
            assert_eq!(
                mask.len(),
                expected_mask_len,
                "mask should have {expected_mask_len} elements, got {}",
                mask.len()
            );
            // Mask values should be 0.0 or 1.0
            for &m in mask {
                assert!(
                    m == 0.0 || m == 1.0,
                    "mask values should be 0.0 or 1.0, got {m}"
                );
            }
            // Deltas should be reasonable
            for &d in deltas {
                assert!(
                    d.abs() <= 200_000,
                    "delta {d} out of reasonable range"
                );
            }
        }
    }

    #[test]
    fn test_collect_leaf_obs_v3_shape() {
        // Verify obs shape for v3 (934 channels)
        let state = setup_basic_game();
        let mut sm = SearchModule::with_seed(3, 123);

        let entries = sm
            .collect_leaf_obs(&state, vec![0], 100, 3)
            .unwrap();

        let expected_obs_len = 934 * 34;
        for (_, obs, _, _, _) in &entries {
            assert_eq!(
                obs.len(),
                expected_obs_len,
                "v3 obs should have {expected_obs_len} elements, got {}",
                obs.len()
            );
        }
    }

    #[test]
    fn test_collect_leaf_obs_determinism() {
        // Same seed → same results
        let state = setup_basic_game();

        let mut sm1 = SearchModule::with_seed(5, 42);
        let entries1 = sm1
            .collect_leaf_obs(&state, vec![0, 8], 50, 4)
            .unwrap();

        let mut sm2 = SearchModule::with_seed(5, 42);
        let entries2 = sm2
            .collect_leaf_obs(&state, vec![0, 8], 50, 4)
            .unwrap();

        assert_eq!(entries1.len(), entries2.len(), "same seed should give same count");
        for ((a1, obs1, mask1, t1, d1), (a2, obs2, mask2, t2, d2)) in
            entries1.iter().zip(entries2.iter())
        {
            assert_eq!(a1, a2, "actions should match");
            assert_eq!(obs1, obs2, "obs should match");
            assert_eq!(mask1, mask2, "masks should match");
            assert_eq!(t1, t2, "terminated should match");
            assert_eq!(d1, d2, "deltas should match");
        }
    }

    #[test]
    fn test_collect_leaf_obs_empty_particles() {
        // No particles → empty result (no error)
        let state = setup_basic_game();
        let mut sm = SearchModule::with_seed(0, 42); // 0 particles
        let entries = sm
            .collect_leaf_obs(&state, vec![0], 100, 4)
            .unwrap();
        assert!(entries.is_empty(), "0 particles should give empty results");
    }

    #[test]
    fn test_collect_leaf_obs_max_steps_zero() {
        // max_steps=0: truncation at step 0 should still collect obs
        let state = setup_basic_game();
        let mut sm = SearchModule::with_seed(3, 42);
        let entries = sm
            .collect_leaf_obs(&state, vec![0], 0, 4)
            .unwrap();

        // All should be non-terminated (truncated at step 0)
        for (_, _, _, terminated, deltas) in &entries {
            assert!(!terminated, "max_steps=0 should not terminate");
            assert_eq!(*deltas, [0, 0, 0, 0], "max_steps=0 should have zero deltas");
        }
    }
}

pub(crate) fn register_module(
    py: Python<'_>,
    prefix: &str,
    super_mod: &Bound<'_, PyModule>,
) -> PyResult<()> {
    let m = PyModule::new(py, "search")?;
    m.add_class::<SearchModule>()?;
    m.add_class::<ParticleConfig>()?;
    m.add_class::<Particle>()?;
    m.add_class::<RolloutResult>()?;
    m.add_class::<TruncatedResult>()?;
    m.add_class::<GrpEvaluator>()?;
    m.add_class::<policy::PolicyEvaluator>()?;
    add_submodule(py, prefix, super_mod, &m)
}
