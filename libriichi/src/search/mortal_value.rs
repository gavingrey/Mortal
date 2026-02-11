use anyhow::{Context, Result};
use ndarray::Array2;

// =============================================================================
// ort-based GPU evaluator (feature = "ort-value")
// =============================================================================
//
// Uses ONNX Runtime via the `ort` crate with CUDA execution provider.
// All leaf observations are batched into a single GPU call, giving ~100-1000x
// speedup over the tract CPU path.
//
// Requires: ORT_DYLIB_PATH set to a CUDA-enabled libonnxruntime.so at runtime.
// Falls back to CPU if CUDA is unavailable.

#[cfg(feature = "ort-value")]
pub struct MortalValueEvaluator {
    session: parking_lot::Mutex<ort::session::Session>,
    obs_channels: usize,
}

#[cfg(feature = "ort-value")]
impl MortalValueEvaluator {
    /// Load a Mortal value ONNX model using ort (ONNX Runtime).
    ///
    /// Requests CUDA execution provider; silently falls back to CPU if unavailable.
    pub fn load(onnx_path: &str, obs_channels: usize) -> Result<Self> {
        let session = ort::session::Session::builder()
            .context("failed to create ort session builder")?
            .with_execution_providers([ort::ep::CUDA::default().build()])
            .context("failed to configure CUDA execution provider")?
            .commit_from_file(onnx_path)
            .with_context(|| format!("failed to load value ONNX model from {onnx_path}"))?;

        log::info!(
            "Loaded value model via ort (CUDA requested, fallback to CPU if unavailable)"
        );

        Ok(Self {
            session: parking_lot::Mutex::new(session),
            obs_channels,
        })
    }

    /// Evaluate a single observation and return the scalar state value.
    pub fn evaluate(&self, obs: &Array2<f32>) -> Result<f64> {
        let values = self.evaluate_batch(std::slice::from_ref(obs))?;
        Ok(values[0])
    }

    /// Evaluate a batch of observations in a single inference call.
    ///
    /// On GPU this is extremely fast (~1-5ms for 150 observations).
    /// Each observation should be shape `(obs_channels, 34)`.
    pub fn evaluate_batch(&self, observations: &[Array2<f32>]) -> Result<Vec<f64>> {
        if observations.is_empty() {
            return Ok(vec![]);
        }

        let batch_size = observations.len();
        let mut data = Vec::with_capacity(batch_size * self.obs_channels * 34);
        for obs in observations {
            let (rows, cols) = obs.dim();
            anyhow::ensure!(
                rows == self.obs_channels && cols == 34,
                "expected obs shape ({}, 34), got ({rows}, {cols})",
                self.obs_channels,
            );
            data.extend(obs.iter().copied());
        }

        let input = ort::value::Tensor::from_array((
            [batch_size, self.obs_channels, 34],
            data,
        ))
        .context("failed to create ort input tensor")?;

        let mut session = self.session.lock();
        let outputs = session
            .run(ort::inputs!["obs" => input])
            .context("value head ort inference failed")?;

        // Output shape: (batch_size, 1) — extract f32 tensor data
        // try_extract_tensor returns (&Shape, &[f32])
        let (_, raw) = outputs["value"]
            .try_extract_tensor::<f32>()
            .context("failed to extract value output from ort")?;

        let values: Vec<f64> = raw.iter().map(|&v| f64::from(v)).collect();
        Ok(values)
    }
}

// =============================================================================
// tract-based CPU evaluator (fallback when ort-value feature is disabled)
// =============================================================================

#[cfg(not(feature = "ort-value"))]
use rayon::prelude::*;

#[cfg(not(feature = "ort-value"))]
use std::sync::Arc;

#[cfg(not(feature = "ort-value"))]
use tract_onnx::prelude::*;

#[cfg(not(feature = "ort-value"))]
type TypedRunnableModel<F> = SimplePlan<F, Box<dyn TypedOp>, Graph<F, Box<dyn TypedOp>>>;

#[cfg(not(feature = "ort-value"))]
pub struct MortalValueEvaluator {
    model: Arc<TypedRunnableModel<TypedFact>>,
    obs_channels: usize,
}

#[cfg(not(feature = "ort-value"))]
impl MortalValueEvaluator {
    /// Load a Mortal value ONNX model from disk using tract.
    pub fn load(onnx_path: &str, obs_channels: usize) -> Result<Self> {
        let model = tract_onnx::onnx()
            .model_for_path(onnx_path)
            .with_context(|| format!("failed to load value ONNX model from {onnx_path}"))?
            .with_input_fact(
                0,
                InferenceFact::dt_shape(
                    f32::datum_type(),
                    tvec![
                        TDim::from(1),
                        TDim::from(obs_channels as i64),
                        TDim::from(34),
                    ],
                ),
            )?
            .into_optimized()
            .context("failed to optimize value ONNX model")?
            .into_runnable()
            .context("failed to make value ONNX model runnable")?;

        Ok(Self {
            model: Arc::new(model),
            obs_channels,
        })
    }

    /// Evaluate a single observation and return the scalar state value.
    pub fn evaluate(&self, obs: &Array2<f32>) -> Result<f64> {
        let (rows, cols) = obs.dim();
        anyhow::ensure!(
            rows == self.obs_channels && cols == 34,
            "expected obs shape ({}, 34), got ({rows}, {cols})",
            self.obs_channels,
        );

        let obs_vec: Vec<f32> = obs.iter().copied().collect();
        let input_tensor =
            tract_ndarray::Array3::from_shape_vec((1, self.obs_channels, 34), obs_vec)
                .context("failed to create input tensor")?;

        let result = self
            .model
            .run(tvec![input_tensor.into_tvalue()])
            .context("value head inference failed")?;

        let output = result[0]
            .to_array_view::<f32>()
            .context("failed to extract value output")?;
        let value = f64::from(*output.iter().next().context("empty output tensor")?);

        Ok(value)
    }

    /// Evaluate multiple observations in parallel using rayon.
    pub fn evaluate_batch(&self, observations: &[Array2<f32>]) -> Result<Vec<f64>> {
        observations
            .par_iter()
            .map(|obs| self.evaluate(obs))
            .collect()
    }
}

// =============================================================================
// Shared utilities
// =============================================================================

/// Compute placement expected value for a player given final scores.
///
/// Ranks all 4 players by score (strict greater-than), then looks up the
/// actor's rank in `placement_pts`. When scores are tied, the player with
/// the tied score gets the better (lower) rank index since `filter(s > actor_score)`
/// is strict — ties don't count against the actor.
///
/// `scores`: absolute scores for players 0-3
/// `actor`: player seat (0-3) to evaluate
/// `placement_pts`: points per rank, e.g., `[6.0, 4.0, 2.0, 0.0]`
pub fn scores_to_placement_ev(scores: &[i32; 4], actor: u8, placement_pts: &[f64; 4]) -> f64 {
    let actor_score = scores[actor as usize];
    let rank = scores.iter().filter(|&&s| s > actor_score).count();
    placement_pts[rank]
}

#[cfg(test)]
mod test {
    use super::*;

    const PTS: [f64; 4] = [6.0, 4.0, 2.0, 0.0];

    #[test]
    fn test_scores_to_placement_ev_clear_leader() {
        let scores = [40000, 30000, 20000, 10000];
        assert_eq!(scores_to_placement_ev(&scores, 0, &PTS), 6.0);
    }

    #[test]
    fn test_scores_to_placement_ev_clear_last() {
        let scores = [40000, 30000, 20000, 10000];
        assert_eq!(scores_to_placement_ev(&scores, 3, &PTS), 0.0);
    }

    #[test]
    fn test_scores_to_placement_ev_tied_scores() {
        let scores = [30000, 30000, 20000, 10000];
        assert_eq!(scores_to_placement_ev(&scores, 0, &PTS), 6.0);
        assert_eq!(scores_to_placement_ev(&scores, 1, &PTS), 6.0);
    }

    #[test]
    fn test_scores_to_placement_ev_all_equal() {
        let scores = [25000; 4];
        assert_eq!(scores_to_placement_ev(&scores, 0, &PTS), 6.0);
        assert_eq!(scores_to_placement_ev(&scores, 1, &PTS), 6.0);
        assert_eq!(scores_to_placement_ev(&scores, 2, &PTS), 6.0);
        assert_eq!(scores_to_placement_ev(&scores, 3, &PTS), 6.0);
    }

    #[test]
    fn test_scores_to_placement_ev_all_players() {
        let scores = [40000, 30000, 20000, 10000];
        let total: f64 = (0..4_u8)
            .map(|a| scores_to_placement_ev(&scores, a, &PTS))
            .sum();
        assert!(
            (total - 12.0).abs() < 1e-10,
            "sum of all placement EVs should be 12.0, got {total}"
        );
    }
}
