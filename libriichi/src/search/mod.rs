pub mod config;
pub mod heuristic;
pub mod particle;
pub mod simulator;

use crate::py_helper::add_submodule;
use crate::state::PlayerState;

use config::ParticleConfig;
use particle::Particle;
use simulator::RolloutResult;

use pyo3::prelude::*;
use rand::SeedableRng;
use rand_chacha::ChaCha12Rng;

/// PyO3-exposed search module entry point.
#[pyclass]
pub struct SearchModule {
    config: ParticleConfig,
    rng: ChaCha12Rng,
    use_smart_policy: bool,
    last_gen_attempts: usize,
    last_gen_accepted: usize,
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

    /// Simulate a single particle rollout using tsumogiri strategy.
    pub fn simulate_particle(
        &self,
        state: &PlayerState,
        particle: &Particle,
    ) -> PyResult<RolloutResult> {
        simulator::simulate_particle(state, particle)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    /// Generate particles and simulate all of them, returning results.
    ///
    /// Replays event history once and derives the midgame context once,
    /// then runs one tsumogiri rollout per particle.
    pub fn generate_and_simulate(
        &mut self,
        state: &PlayerState,
    ) -> PyResult<Vec<RolloutResult>> {
        let (particles, attempts) =
            particle::generate_particles(state, &self.config, &mut self.rng)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        self.last_gen_attempts = attempts;
        self.last_gen_accepted = particles.len();

        let replayed = simulator::replay_player_states(state)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        // Derive event-dependent context once, reuse across all particles
        let base = simulator::derive_midgame_context_base(state.event_history());

        let mut results = Vec::with_capacity(particles.len());
        for p in &particles {
            let board_state =
                simulator::build_midgame_board_state_with_base(state, &replayed, p, &base)
                    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
            let initial_scores = board_state.board.scores;
            let result = if self.use_smart_policy {
                simulator::run_rollout_smart(board_state, initial_scores, &mut self.rng)
            } else {
                simulator::run_rollout_tsumogiri(board_state, initial_scores)
            }
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
            results.push(result);
        }
        Ok(results)
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

    /// Evaluate multiple actions across multiple particles.
    ///
    /// Replays event history once and derives midgame context once, then
    /// for each particle/action pair, clones the replayed states and runs
    /// a rollout.
    /// Returns a dict mapping action index to list of RolloutResults.
    pub fn evaluate_actions(
        &mut self,
        state: &PlayerState,
        particles: Vec<Particle>,
        actions: Vec<usize>,
    ) -> PyResult<std::collections::HashMap<usize, Vec<RolloutResult>>> {
        // Replay once
        let replayed = simulator::replay_player_states(state)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        // Derive event-dependent context once
        let base = simulator::derive_midgame_context_base(state.event_history());

        let mut results = std::collections::HashMap::new();

        // Round-robin: iterate particles first, then actions, to avoid
        // sampling bias toward earlier candidates.
        for &action in &actions {
            results.insert(action, Vec::with_capacity(particles.len()));
        }

        for particle in &particles {
            for &action in &actions {
                let result = if self.use_smart_policy {
                    simulator::simulate_action_rollout_with_base_smart(
                        state, &replayed, particle, action, &base, &mut self.rng,
                    )
                } else {
                    simulator::simulate_action_rollout_with_base(
                        state, &replayed, particle, action, &base,
                    )
                };
                match result {
                    Ok(result) => results.get_mut(&action).unwrap().push(result),
                    Err(e) => {
                        return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                            "action {action}: {e}"
                        )));
                    }
                }
            }
        }

        Ok(results)
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
    add_submodule(py, prefix, super_mod, &m)
}
