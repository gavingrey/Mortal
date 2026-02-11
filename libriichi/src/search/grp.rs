use anyhow::{Context, Result, ensure};
use pyo3::prelude::*;
use std::sync::Arc;

use tract_onnx::prelude::*;

/// Pre-computed permutations of [0,1,2,3] in lexicographic order.
/// Each permutation maps player -> rank.
/// perm[i][p] = rank of player p in permutation i.
const PERMS: [[u8; 4]; 24] = [
    [0, 1, 2, 3], [0, 1, 3, 2], [0, 2, 1, 3], [0, 2, 3, 1],
    [0, 3, 1, 2], [0, 3, 2, 1], [1, 0, 2, 3], [1, 0, 3, 2],
    [1, 2, 0, 3], [1, 2, 3, 0], [1, 3, 0, 2], [1, 3, 2, 0],
    [2, 0, 1, 3], [2, 0, 3, 1], [2, 1, 0, 3], [2, 1, 3, 0],
    [2, 3, 0, 1], [2, 3, 1, 0], [3, 0, 1, 2], [3, 0, 2, 1],
    [3, 1, 0, 2], [3, 1, 2, 0], [3, 2, 0, 1], [3, 2, 1, 0],
];

/// GRP input entry: [grand_kyoku, honba, kyotaku, s0/10k, s1/10k, s2/10k, s3/10k]
pub type GrpEntry = [f64; 7];

/// GRP (Game Result Predictor) evaluator using tract ONNX runtime.
///
/// Loads a GRP ONNX model and evaluates leaf states to produce expected
/// points based on placement probabilities.
#[pyclass]
pub struct GrpEvaluator {
    model: Arc<TypedRunnableModel<TypedFact>>,
    placement_pts: [f64; 4],
}

/// Helper type alias for the tract model.
type TypedRunnableModel<F> = SimplePlan<F, Box<dyn TypedOp>, Graph<F, Box<dyn TypedOp>>>;

#[pymethods]
impl GrpEvaluator {
    /// Load a GRP ONNX model from disk.
    ///
    /// `onnx_path`: path to the .onnx file exported by `export_grp_onnx.py`
    /// `placement_pts`: points awarded per placement, e.g. [6.0, 4.0, 2.0, 0.0]
    ///   where index 0 = 1st place, index 3 = 4th place.
    #[new]
    pub fn load(onnx_path: &str, placement_pts: [f64; 4]) -> PyResult<Self> {
        Self::load_impl(onnx_path, placement_pts)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    /// Evaluate a leaf state and return expected points for the given player.
    ///
    /// `history`: sequence of GRP entries from previous kyoku boundaries
    /// `leaf`: the current leaf state entry
    /// `player_id`: absolute seat (0-3) to evaluate for
    pub fn evaluate_leaf(
        &self,
        history: Vec<[f64; 7]>,
        leaf: [f64; 7],
        player_id: u8,
    ) -> PyResult<f64> {
        self.evaluate_leaf_impl(&history, &leaf, player_id)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }
}

impl GrpEvaluator {
    fn load_impl(onnx_path: &str, placement_pts: [f64; 4]) -> Result<Self> {
        // Create a symbolic dimension for the sequence length
        let sym_scope = tract_onnx::prelude::SymbolScope::default();
        let seq_sym = sym_scope.sym("seq");

        let model = tract_onnx::onnx()
            .model_for_path(onnx_path)
            .with_context(|| format!("failed to load ONNX model from {onnx_path}"))?
            .with_input_fact(
                0,
                InferenceFact::dt_shape(f64::datum_type(), tvec![1.into(), TDim::from(&seq_sym), 7.into()]),
            )?
            .into_optimized()
            .context("failed to optimize ONNX model")?
            .into_runnable()
            .context("failed to make ONNX model runnable")?;

        Ok(Self {
            model: Arc::new(model),
            placement_pts,
        })
    }

    /// Core evaluation: run GRP inference and compute expected points.
    pub fn evaluate_leaf_impl(
        &self,
        history: &[GrpEntry],
        leaf: &GrpEntry,
        player_id: u8,
    ) -> Result<f64> {
        ensure!(player_id < 4, "player_id must be 0-3, got {player_id}");

        // Build input sequence: history entries + leaf entry
        let seq_len = history.len() + 1;
        let mut input_data = Vec::with_capacity(seq_len * 7);
        for entry in history {
            input_data.extend_from_slice(entry);
        }
        input_data.extend_from_slice(leaf);

        // Create tensor: shape (1, seq_len, 7)
        let input_tensor = tract_ndarray::Array3::from_shape_vec(
            (1, seq_len, 7),
            input_data,
        )
        .context("failed to create input tensor")?;

        // Run inference
        let result = self.model.run(tvec![input_tensor.into_tvalue()])
            .context("GRP inference failed")?;

        // Extract logits: shape (1, 24)
        let logits_tensor = result[0]
            .to_array_view::<f64>()
            .context("failed to extract logits")?;
        let logits: Vec<f64> = logits_tensor.iter().copied().collect();
        ensure!(logits.len() == 24, "expected 24 logits, got {}", logits.len());

        // Compute placement probabilities for this player
        let probs = calc_player_probs(&logits, player_id);

        // Expected points = sum(prob[rank] * placement_pts[rank])
        let expected_pts: f64 = probs
            .iter()
            .zip(self.placement_pts.iter())
            .map(|(p, pts)| p * pts)
            .sum();

        Ok(expected_pts)
    }

    /// Evaluate a leaf state without the PyO3 boundary (for Rust-internal use).
    pub fn evaluate_leaf_rust(
        &self,
        history: &[GrpEntry],
        leaf: &GrpEntry,
        player_id: u8,
    ) -> Result<f64> {
        self.evaluate_leaf_impl(history, leaf, player_id)
    }

    /// Get a reference to the placement points.
    pub fn placement_pts(&self) -> &[f64; 4] {
        &self.placement_pts
    }
}

/// Compute placement probabilities for a single player from 24 logits.
///
/// The 24 logits correspond to 24 permutations of [0,1,2,3] in lexicographic
/// order. Each permutation maps player -> rank. We softmax the logits to get
/// permutation probabilities, then for each rank R, sum the probabilities of
/// all permutations where `perm[i][player_id] == R`.
///
/// Returns `[prob_rank0, prob_rank1, prob_rank2, prob_rank3]` where rank 0 = 1st place.
pub fn calc_player_probs(logits: &[f64], player_id: u8) -> [f64; 4] {
    debug_assert!(logits.len() == 24);
    debug_assert!(player_id < 4);

    // Softmax
    let max_logit = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let exp: Vec<f64> = logits.iter().map(|&l| (l - max_logit).exp()).collect();
    let sum: f64 = exp.iter().sum();
    let probs: Vec<f64> = exp.iter().map(|&e| e / sum).collect();

    // Sum probabilities by rank for the given player
    let pid = player_id as usize;
    let mut rank_probs = [0.0_f64; 4];
    for (i, &perm) in PERMS.iter().enumerate() {
        let rank = perm[pid] as usize;
        rank_probs[rank] += probs[i];
    }

    rank_probs
}

/// Build a GRP entry from game state.
///
/// `grand_kyoku`: absolute kyoku (E1=0, S4=7, W4=11)
/// `honba`: current honba count
/// `kyotaku`: current riichi sticks on table
/// `scores`: absolute scores for players 0-3
pub fn make_grp_entry(
    grand_kyoku: u8,
    honba: u8,
    kyotaku: u8,
    scores: &[i32; 4],
) -> GrpEntry {
    [
        f64::from(grand_kyoku),
        f64::from(honba),
        f64::from(kyotaku),
        f64::from(scores[0]) / 10000.0,
        f64::from(scores[1]) / 10000.0,
        f64::from(scores[2]) / 10000.0,
        f64::from(scores[3]) / 10000.0,
    ]
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_perms_are_valid() {
        // Each permutation should be a valid permutation of [0,1,2,3]
        for perm in &PERMS {
            let mut sorted = *perm;
            sorted.sort_unstable();
            assert_eq!(sorted, [0, 1, 2, 3]);
        }
        // Should have exactly 24 permutations (4!)
        assert_eq!(PERMS.len(), 24);
        // All permutations should be unique
        let mut unique = PERMS.to_vec();
        unique.sort_unstable();
        unique.dedup();
        assert_eq!(unique.len(), 24);
    }

    #[test]
    fn test_calc_player_probs_uniform() {
        // With uniform logits, each rank should have ~0.25 probability
        let logits = [0.0_f64; 24];
        for player_id in 0..4_u8 {
            let probs = calc_player_probs(&logits, player_id);
            let sum: f64 = probs.iter().sum();
            assert!(
                (sum - 1.0).abs() < 1e-10,
                "probabilities should sum to 1, got {sum}"
            );
            for (rank, &p) in probs.iter().enumerate() {
                assert!(
                    (p - 0.25).abs() < 1e-10,
                    "player {player_id} rank {rank}: expected ~0.25, got {p}"
                );
            }
        }
    }

    #[test]
    fn test_calc_player_probs_sum_to_one() {
        // With arbitrary logits, probabilities should sum to 1
        let logits: Vec<f64> = (0..24).map(|i| (i as f64) * 0.1 - 1.2).collect();
        for player_id in 0..4_u8 {
            let probs = calc_player_probs(&logits, player_id);
            let sum: f64 = probs.iter().sum();
            assert!(
                (sum - 1.0).abs() < 1e-10,
                "player {player_id}: probabilities should sum to 1, got {sum}"
            );
            for &p in &probs {
                assert!(p >= 0.0, "probabilities should be non-negative");
            }
        }
    }

    #[test]
    fn test_calc_player_probs_extreme() {
        // When one permutation dominates, that rank assignment should have ~1.0
        let mut logits = [-100.0_f64; 24];
        logits[0] = 100.0; // perm[0] = [0,1,2,3]: player 0 -> rank 0
        let probs = calc_player_probs(&logits, 0);
        assert!(probs[0] > 0.999, "player 0 should be rank 0 with high prob");
    }

    #[test]
    fn test_make_grp_entry() {
        let entry = make_grp_entry(3, 1, 2, &[25000, 30000, 20000, 25000]);
        assert_eq!(entry[0], 3.0);  // grand_kyoku
        assert_eq!(entry[1], 1.0);  // honba
        assert_eq!(entry[2], 2.0);  // kyotaku
        assert_eq!(entry[3], 2.5);  // 25000/10000
        assert_eq!(entry[4], 3.0);  // 30000/10000
        assert_eq!(entry[5], 2.0);  // 20000/10000
        assert_eq!(entry[6], 2.5);  // 25000/10000
    }
}
