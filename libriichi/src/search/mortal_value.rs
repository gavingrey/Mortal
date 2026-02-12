use anyhow::{Context, Result};
use ndarray::Array2;
use rayon::prelude::*;
use std::sync::Arc;

use tract_onnx::prelude::*;

/// Type alias for the tract model plan.
type TypedRunnableModel<F> = SimplePlan<F, Box<dyn TypedOp>, Graph<F, Box<dyn TypedOp>>>;

/// Mortal value head evaluator using tract ONNX runtime.
///
/// Loads a Brain + DQN v_head ONNX model exported by `export_mortal_value_onnx.py`
/// and evaluates game observations to produce scalar state values.
///
/// Input: `(1, obs_channels, 34)` float32 observation
/// Output: scalar f64 state value
///
/// Batching is implemented by looping over `evaluate` since tract works best
/// with concrete batch=1.
pub struct MortalValueEvaluator {
    model: Arc<TypedRunnableModel<TypedFact>>,
    obs_channels: usize,
}

impl MortalValueEvaluator {
    /// Load a Mortal value ONNX model from disk.
    ///
    /// `onnx_path`: path to the .onnx file exported by `export_mortal_value_onnx.py`
    /// `obs_channels`: number of observation channels (e.g., 1012 for v4)
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
    ///
    /// `obs`: a 2D array of shape `(obs_channels, 34)` containing the encoded game state.
    /// Returns the scalar value as f64.
    pub fn evaluate(&self, obs: &Array2<f32>) -> Result<f64> {
        let (rows, cols) = obs.dim();
        anyhow::ensure!(
            rows == self.obs_channels && cols == 34,
            "expected obs shape ({}, 34), got ({rows}, {cols})",
            self.obs_channels,
        );

        // Reshape to (1, obs_channels, 34) for tract
        let obs_vec: Vec<f32> = obs.iter().copied().collect();
        let input_tensor = tract_ndarray::Array3::from_shape_vec(
            (1, self.obs_channels, 34),
            obs_vec,
        )
        .context("failed to create input tensor")?;

        let result = self
            .model
            .run(tvec![input_tensor.into_tvalue()])
            .context("value head inference failed")?;

        // Extract scalar: output shape is (1, 1)
        let output = result[0]
            .to_array_view::<f32>()
            .context("failed to extract value output")?;
        let value = f64::from(*output.iter().next().context("empty output tensor")?);

        Ok(value)
    }

    /// Evaluate multiple observations in parallel using rayon.
    ///
    /// Each observation should be shape `(obs_channels, 34)`.
    /// Returns one scalar value per observation.
    ///
    /// The underlying tract `SimplePlan::run` takes `&self` and the model is
    /// wrapped in `Arc`, so concurrent evaluation across threads is safe.
    pub fn evaluate_batch(&self, observations: &[Array2<f32>]) -> Result<Vec<f64>> {
        observations
            .par_iter()
            .map(|obs| self.evaluate(obs))
            .collect()
    }
}

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
        // Player 0 has the highest score → rank 0 → 6.0
        let scores = [40000, 30000, 20000, 10000];
        assert_eq!(scores_to_placement_ev(&scores, 0, &PTS), 6.0);
    }

    #[test]
    fn test_scores_to_placement_ev_clear_last() {
        // Player 3 has the lowest score → rank 3 → 0.0
        let scores = [40000, 30000, 20000, 10000];
        assert_eq!(scores_to_placement_ev(&scores, 3, &PTS), 0.0);
    }

    #[test]
    fn test_scores_to_placement_ev_tied_scores() {
        // Players 0 and 1 tied at 30000.
        // For player 0: filter(s > 30000) → none → rank 0 → 6.0
        // For player 1: filter(s > 30000) → none → rank 0 → 6.0
        // Both get rank 0 since strict greater-than means ties don't penalize.
        let scores = [30000, 30000, 20000, 10000];
        assert_eq!(scores_to_placement_ev(&scores, 0, &PTS), 6.0);
        assert_eq!(scores_to_placement_ev(&scores, 1, &PTS), 6.0);
    }

    #[test]
    fn test_scores_to_placement_ev_all_equal() {
        // All 25000 → no score strictly greater → rank 0 → 6.0
        let scores = [25000; 4];
        assert_eq!(scores_to_placement_ev(&scores, 0, &PTS), 6.0);
        assert_eq!(scores_to_placement_ev(&scores, 1, &PTS), 6.0);
        assert_eq!(scores_to_placement_ev(&scores, 2, &PTS), 6.0);
        assert_eq!(scores_to_placement_ev(&scores, 3, &PTS), 6.0);
    }

    #[test]
    fn test_scores_to_placement_ev_all_players() {
        // With distinct scores, sum of all placement EVs equals sum of PTS.
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
