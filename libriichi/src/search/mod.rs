pub mod config;
pub mod particle;

use crate::py_helper::add_submodule;
use crate::state::PlayerState;

use config::ParticleConfig;
use particle::Particle;

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
    pub fn set_config(&mut self, config: ParticleConfig) {
        self.config = config;
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
    add_submodule(py, prefix, super_mod, &m)
}
