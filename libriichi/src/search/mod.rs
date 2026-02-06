pub mod config;
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
}

#[pymethods]
impl SearchModule {
    #[new]
    #[must_use]
    pub fn new(n_particles: usize) -> Self {
        Self {
            config: ParticleConfig::new(n_particles),
            rng: ChaCha12Rng::from_os_rng(),
        }
    }

    /// Create with a fixed seed for reproducibility.
    #[staticmethod]
    #[must_use]
    pub fn with_seed(n_particles: usize, seed: u64) -> Self {
        Self {
            config: ParticleConfig::new(n_particles),
            rng: ChaCha12Rng::seed_from_u64(seed),
        }
    }

    /// Generate particles from the current game state.
    pub fn generate_particles(&mut self, state: &PlayerState) -> PyResult<Vec<Particle>> {
        particle::generate_particles(state, &self.config, &mut self.rng)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
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
    pub fn generate_and_simulate(
        &mut self,
        state: &PlayerState,
    ) -> PyResult<Vec<RolloutResult>> {
        let particles = particle::generate_particles(state, &self.config, &mut self.rng)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        let mut results = Vec::with_capacity(particles.len());
        for p in &particles {
            let result = simulator::simulate_particle(state, p)
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
        simulator::simulate_particle_action(state, particle, action)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    /// Evaluate multiple actions across multiple particles.
    ///
    /// For each action, runs a rollout on each particle and collects results.
    /// Returns a dict mapping action index to list of RolloutResults.
    pub fn evaluate_actions(
        &self,
        state: &PlayerState,
        particles: Vec<Particle>,
        actions: Vec<usize>,
    ) -> PyResult<std::collections::HashMap<usize, Vec<RolloutResult>>> {
        let mut results = std::collections::HashMap::new();

        for &action in &actions {
            let mut action_results = Vec::with_capacity(particles.len());
            for particle in &particles {
                match simulator::simulate_particle_action(state, particle, action) {
                    Ok(result) => action_results.push(result),
                    Err(e) => {
                        return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                            "action {action}: {e}"
                        )));
                    }
                }
            }
            results.insert(action, action_results);
        }

        Ok(results)
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
