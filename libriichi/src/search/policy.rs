use crate::consts::ACTION_SPACE;

use anyhow::{Context, Result, ensure};
use pyo3::prelude::*;
use std::sync::Arc;
use tract_onnx::prelude::*;

/// Type alias for the tract model (same as grp.rs).
type TypedRunnableModel<F> = SimplePlan<F, Box<dyn TypedOp>, Graph<F, Box<dyn TypedOp>>>;

/// Policy evaluator using tract ONNX runtime.
///
/// Loads a combined Brain+DQN ONNX model (exported by `export_policy_onnx.py`)
/// and evaluates positions to produce Q-values for each action.
///
/// The model expects:
///   - obs: (1, channels, 34) float32  — observation features
///   - mask: (1, 46) float32           — action mask (0.0 or 1.0)
/// and outputs:
///   - q_values: (1, 46) float32       — Q-values per action
///
/// `obs_channels` is provided at load time (934 for v3, 1012 for v4) since
/// different model versions have different observation shapes. Use
/// `libriichi::consts::obs_shape(version)` to get the correct value.
#[pyclass]
pub struct PolicyEvaluator {
    model: Arc<TypedRunnableModel<TypedFact>>,
    /// Number of observation channels.
    obs_channels: usize,
}

#[pymethods]
impl PolicyEvaluator {
    /// Load a policy ONNX model from disk.
    ///
    /// `onnx_path`: path to the .onnx file exported by `export_policy_onnx.py`
    /// `obs_channels`: number of observation channels (e.g. 1012 for v4, 934 for v3)
    #[new]
    pub fn load(onnx_path: &str, obs_channels: usize) -> PyResult<Self> {
        Self::load_impl(onnx_path, obs_channels)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    /// Evaluate a position and return Q-values.
    ///
    /// `obs`: flat slice of length channels*34, row-major (channels, 34)
    /// `mask`: action mask, true = legal action
    ///
    /// Returns Q-values for all 46 actions (illegal actions get -inf).
    pub fn evaluate(&self, obs: Vec<f32>, mask: [bool; ACTION_SPACE]) -> PyResult<Vec<f32>> {
        self.evaluate_impl(&obs, &mask)
            .map(|arr| arr.to_vec())
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }
}

impl PolicyEvaluator {
    /// Load the ONNX model with concrete batch=1 and specified channel count.
    pub(crate) fn load_impl(onnx_path: &str, obs_channels: usize) -> Result<Self> {
        ensure!(obs_channels > 0, "obs_channels must be positive");

        let model = tract_onnx::onnx()
            .model_for_path(onnx_path)
            .with_context(|| format!("failed to load ONNX model from {onnx_path}"))?
            .with_input_fact(
                0,
                InferenceFact::dt_shape(
                    f32::datum_type(),
                    tvec![
                        TDim::from(1_i64),
                        TDim::from(obs_channels as i64),
                        TDim::from(34_i64),
                    ],
                ),
            )?
            .with_input_fact(
                1,
                InferenceFact::dt_shape(
                    f32::datum_type(),
                    tvec![
                        TDim::from(1_i64),
                        TDim::from(ACTION_SPACE as i64),
                    ],
                ),
            )?
            .into_optimized()
            .context("failed to optimize ONNX model")?
            .into_runnable()
            .context("failed to make ONNX model runnable")?;

        Ok(Self {
            model: Arc::new(model),
            obs_channels,
        })
    }

    /// Run inference and return Q-values as [f32; ACTION_SPACE].
    ///
    /// `obs`: flat slice of length obs_channels*34
    /// `mask`: action mask (true = legal)
    pub(crate) fn evaluate_impl(
        &self,
        obs: &[f32],
        mask: &[bool; ACTION_SPACE],
    ) -> Result<[f32; ACTION_SPACE]> {
        let expected_len = self.obs_channels * 34;
        ensure!(
            obs.len() == expected_len,
            "obs length mismatch: expected {} ({}*34), got {}",
            expected_len,
            self.obs_channels,
            obs.len()
        );

        // Build obs tensor: shape (1, channels, 34)
        let obs_tensor = tract_ndarray::Array3::from_shape_vec(
            (1, self.obs_channels, 34),
            obs.to_vec(),
        )
        .context("failed to create obs tensor")?;

        // Build mask tensor: shape (1, 46), bool -> f32 (true -> 1.0, false -> 0.0)
        let mask_data: Vec<f32> = mask
            .iter()
            .map(|&b| if b { 1.0_f32 } else { 0.0_f32 })
            .collect();
        let mask_tensor =
            tract_ndarray::Array2::from_shape_vec((1, ACTION_SPACE), mask_data)
                .context("failed to create mask tensor")?;

        // Run inference
        let result = self
            .model
            .run(tvec![obs_tensor.into_tvalue(), mask_tensor.into_tvalue()])
            .context("policy inference failed")?;

        // Extract q_values: must be shape (1, 46)
        let q_tensor = result[0]
            .to_array_view::<f32>()
            .context("failed to extract q_values")?;
        ensure!(
            q_tensor.shape() == [1, ACTION_SPACE],
            "expected q_values shape [1, {}], got {:?}",
            ACTION_SPACE,
            q_tensor.shape()
        );

        let mut q_values = [0.0_f32; ACTION_SPACE];
        q_values.copy_from_slice(
            q_tensor
                .as_slice()
                .context("q_values tensor is not contiguous")?,
        );

        Ok(q_values)
    }

    /// Get the number of observation channels this model expects.
    pub const fn obs_channels(&self) -> usize {
        self.obs_channels
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_mask_conversion() {
        // Verify bool -> f32 conversion logic
        let mask: [bool; ACTION_SPACE] = {
            let mut m = [false; ACTION_SPACE];
            m[0] = true;
            m[1] = true;
            m[45] = true;
            m
        };
        let mask_f32: Vec<f32> = mask
            .iter()
            .map(|&b| if b { 1.0_f32 } else { 0.0_f32 })
            .collect();
        assert_eq!(mask_f32[0], 1.0);
        assert_eq!(mask_f32[1], 1.0);
        assert_eq!(mask_f32[2], 0.0);
        assert_eq!(mask_f32[45], 1.0);
        assert_eq!(mask_f32.len(), ACTION_SPACE);
    }

    #[test]
    fn test_action_space_is_46() {
        assert_eq!(ACTION_SPACE, 46);
    }
}
