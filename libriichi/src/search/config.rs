use pyo3::prelude::*;
use serde::Deserialize;

/// Configuration for particle generation.
#[pyclass]
#[derive(Clone, Debug, Deserialize)]
pub struct ParticleConfig {
    /// Number of particles to generate.
    #[pyo3(get, set)]
    pub n_particles: usize,

    /// Maximum number of sampling attempts before giving up.
    /// Set to n_particles * max_attempts_factor.
    #[pyo3(get, set)]
    pub max_attempts_factor: usize,

    /// Minimum number of particles required for a valid result.
    #[pyo3(get, set)]
    pub min_particles: usize,
}

#[pymethods]
impl ParticleConfig {
    #[new]
    #[must_use]
    pub const fn new(n_particles: usize) -> Self {
        Self {
            n_particles,
            max_attempts_factor: 20,
            min_particles: n_particles / 2,
        }
    }

    #[must_use]
    pub const fn max_attempts(&self) -> usize {
        self.n_particles * self.max_attempts_factor
    }
}

impl Default for ParticleConfig {
    fn default() -> Self {
        Self {
            n_particles: 100,
            max_attempts_factor: 20,
            min_particles: 50,
        }
    }
}
