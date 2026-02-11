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
    batch_sym: Symbol,
    seq_sym: Symbol,
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

    /// Evaluate multiple leaf states in a single batched ONNX call.
    ///
    /// `history`: shared prefix of GRP entries from previous kyoku boundaries
    /// `leaves`: leaf state entries to evaluate (one per batch element)
    /// `player_id`: absolute seat (0-3) to evaluate for
    ///
    /// Returns a Vec of expected points, one per leaf.
    pub fn evaluate_batch(
        &self,
        history: Vec<[f64; 7]>,
        leaves: Vec<[f64; 7]>,
        player_id: u8,
    ) -> PyResult<Vec<f64>> {
        self.evaluate_batch_impl(&history, &leaves, player_id)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }
}

impl GrpEvaluator {
    pub(crate) fn load_impl(onnx_path: &str, placement_pts: [f64; 4]) -> Result<Self> {
        // Create symbolic dimensions for batch size and sequence length
        let sym_scope = tract_onnx::prelude::SymbolScope::default();
        let batch_sym = sym_scope.sym("batch");
        let seq_sym = sym_scope.sym("seq");

        let model = tract_onnx::onnx()
            .model_for_path(onnx_path)
            .with_context(|| format!("failed to load ONNX model from {onnx_path}"))?
            .with_input_fact(
                0,
                InferenceFact::dt_shape(
                    f32::datum_type(),
                    tvec![TDim::from(&batch_sym), TDim::from(&seq_sym), 7.into()],
                ),
            )?
            .into_optimized()
            .context("failed to optimize ONNX model")?
            .into_runnable()
            .context("failed to make ONNX model runnable")?;

        Ok(Self {
            model: Arc::new(model),
            batch_sym,
            seq_sym,
            placement_pts,
        })
    }

    /// Core evaluation: run GRP inference and compute expected points.
    pub(crate) fn evaluate_leaf_impl(
        &self,
        history: &[GrpEntry],
        leaf: &GrpEntry,
        player_id: u8,
    ) -> Result<f64> {
        ensure!(player_id < 4, "player_id must be 0-3, got {player_id}");

        // Build input sequence: history entries + leaf entry (cast f64 -> f32 for ONNX)
        let seq_len = history.len() + 1;
        let mut input_data: Vec<f32> = Vec::with_capacity(seq_len * 7);
        for entry in history {
            input_data.extend(entry.iter().map(|&v| v as f32));
        }
        input_data.extend(leaf.iter().map(|&v| v as f32));

        // Create tensor: shape (1, seq_len, 7)
        let input_tensor = tract_ndarray::Array3::from_shape_vec(
            (1, seq_len, 7),
            input_data,
        )
        .context("failed to create input tensor")?;

        // Run inference with resolved symbolic dimensions
        let mut state = SimpleState::new(Arc::clone(&self.model))
            .context("failed to create inference state")?;
        state.session_state.resolved_symbols
            .set(&self.batch_sym, 1);
        state.session_state.resolved_symbols
            .set(&self.seq_sym, seq_len as i64);
        let result = state.run(tvec![input_tensor.into_tvalue()])
            .context("GRP inference failed")?;

        // Extract logits: shape (1, 24), cast f32 -> f64
        let logits_tensor = result[0]
            .to_array_view::<f32>()
            .context("failed to extract logits")?;
        let logits: Vec<f64> = logits_tensor.iter().map(|&v| f64::from(v)).collect();
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

    /// Batched evaluation: run GRP inference for multiple leaves in one ONNX call.
    ///
    /// All leaves share the same `history` prefix. Each sample in the batch is
    /// `history ++ [leaf_i]`. Returns expected points per leaf.
    pub(crate) fn evaluate_batch_impl(
        &self,
        history: &[GrpEntry],
        leaves: &[GrpEntry],
        player_id: u8,
    ) -> Result<Vec<f64>> {
        ensure!(player_id < 4, "player_id must be 0-3, got {player_id}");

        if leaves.is_empty() {
            return Ok(vec![]);
        }

        let batch_size = leaves.len();
        let seq_len = history.len() + 1;

        // Pre-convert shared history prefix to f32 once (avoids redundant conversion per leaf)
        let history_f32: Vec<f32> = history.iter().flat_map(|e| e.iter().map(|&v| v as f32)).collect();

        // Build flat (batch_size, seq_len, 7) tensor data
        let mut input_data: Vec<f32> = Vec::with_capacity(batch_size * seq_len * 7);
        for leaf in leaves {
            input_data.extend_from_slice(&history_f32);
            input_data.extend(leaf.iter().map(|&v| v as f32));
        }

        let input_tensor = tract_ndarray::Array3::from_shape_vec(
            (batch_size, seq_len, 7),
            input_data,
        )
        .context("failed to create batched input tensor")?;

        // Run inference with resolved symbolic dimensions
        let mut state = SimpleState::new(Arc::clone(&self.model))
            .context("failed to create inference state")?;
        state.session_state.resolved_symbols
            .set(&self.batch_sym, batch_size as i64);
        state.session_state.resolved_symbols
            .set(&self.seq_sym, seq_len as i64);
        let result = state.run(tvec![input_tensor.into_tvalue()])
            .context("batched GRP inference failed")?;

        // Extract logits: shape (batch_size, 24), cast f32 -> f64
        let logits_tensor = result[0]
            .to_array_view::<f32>()
            .context("failed to extract batched logits")?;
        ensure!(
            logits_tensor.len() == batch_size * 24,
            "expected {} logits, got {}",
            batch_size * 24,
            logits_tensor.len()
        );

        // Compute expected points for each leaf (convert f32->f64 per-sample on the stack)
        let logits_flat = logits_tensor.as_slice().context("logits tensor not contiguous")?;
        let mut results = Vec::with_capacity(batch_size);
        for i in 0..batch_size {
            let sample_f32 = &logits_flat[i * 24..(i + 1) * 24];
            let sample_logits: [f64; 24] = std::array::from_fn(|j| f64::from(sample_f32[j]));
            let probs = calc_player_probs(&sample_logits, player_id);
            let expected_pts: f64 = probs
                .iter()
                .zip(self.placement_pts.iter())
                .map(|(p, pts)| p * pts)
                .sum();
            results.push(expected_pts);
        }

        Ok(results)
    }

    /// Get a reference to the placement points.
    pub const fn placement_pts(&self) -> &[f64; 4] {
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
    assert!(logits.len() == 24, "expected 24 logits, got {}", logits.len());
    assert!(player_id < 4, "player_id must be 0-3, got {player_id}");

    // Softmax (stack-allocated)
    let max_logit = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mut exp = [0.0_f64; 24];
    let mut sum = 0.0_f64;
    for (i, &l) in logits.iter().enumerate() {
        exp[i] = (l - max_logit).exp();
        sum += exp[i];
    }
    let inv_sum = 1.0 / sum;

    // Sum probabilities by rank for the given player
    let pid = player_id as usize;
    let mut rank_probs = [0.0_f64; 4];
    for (i, &perm) in PERMS.iter().enumerate() {
        let rank = perm[pid] as usize;
        rank_probs[rank] += exp[i] * inv_sum;
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

    #[test]
    fn test_permutation_table_lexicographic_order() {
        // PERMS should be in lexicographic order
        for i in 1..PERMS.len() {
            assert!(
                PERMS[i - 1] < PERMS[i],
                "PERMS[{}]={:?} should be < PERMS[{}]={:?}",
                i - 1,
                PERMS[i - 1],
                i,
                PERMS[i]
            );
        }
    }

    #[test]
    fn test_calc_player_probs_known_values() {
        // Hand-computed test case:
        // Set logits so only permutations 0 and 6 have significant weight.
        // perm[0] = [0,1,2,3]: p0→rank0, p1→rank1, p2→rank2, p3→rank3
        // perm[6] = [1,0,2,3]: p0→rank1, p1→rank0, p2→rank2, p3→rank3
        //
        // With logits[0]=1.0, logits[6]=1.0, rest=-1000.0:
        // softmax ≈ 0.5 each for perms 0 and 6.
        //
        // Player 0: rank0 from perm[0] + rank1 from perm[6] → [0.5, 0.5, 0, 0]
        // Player 1: rank1 from perm[0] + rank0 from perm[6] → [0.5, 0.5, 0, 0]
        // Player 2: rank2 from both → [0, 0, 1.0, 0]
        // Player 3: rank3 from both → [0, 0, 0, 1.0]
        let mut logits = [-1000.0_f64; 24];
        logits[0] = 1.0;
        logits[6] = 1.0;

        let p0 = calc_player_probs(&logits, 0);
        assert!((p0[0] - 0.5).abs() < 1e-6, "p0 rank0: {}", p0[0]);
        assert!((p0[1] - 0.5).abs() < 1e-6, "p0 rank1: {}", p0[1]);
        assert!(p0[2] < 1e-6, "p0 rank2 should be ~0");
        assert!(p0[3] < 1e-6, "p0 rank3 should be ~0");

        let p1 = calc_player_probs(&logits, 1);
        assert!((p1[0] - 0.5).abs() < 1e-6, "p1 rank0: {}", p1[0]);
        assert!((p1[1] - 0.5).abs() < 1e-6, "p1 rank1: {}", p1[1]);

        let p2 = calc_player_probs(&logits, 2);
        assert!((p2[2] - 1.0).abs() < 1e-6, "p2 rank2: {}", p2[2]);

        let p3 = calc_player_probs(&logits, 3);
        assert!((p3[3] - 1.0).abs() < 1e-6, "p3 rank3: {}", p3[3]);
    }

    #[test]
    fn test_grp_expected_points_bounds() {
        // With placement_pts = [6, 4, 2, 0], expected points for any
        // probability distribution should be in [0.0, 6.0].
        let placement_pts = [6.0_f64, 4.0, 2.0, 0.0];
        let test_cases: Vec<Vec<f64>> = vec![
            vec![0.0; 24],
            (0..24).map(|i| i as f64).collect(),
            (0..24).map(|i| -(i as f64)).collect(),
        ];

        for logits in &test_cases {
            for player_id in 0..4_u8 {
                let probs = calc_player_probs(logits, player_id);
                let expected_pts: f64 = probs
                    .iter()
                    .zip(placement_pts.iter())
                    .map(|(p, pts)| p * pts)
                    .sum();
                assert!(
                    expected_pts >= -1e-10 && expected_pts <= 6.0 + 1e-10,
                    "player {player_id}: expected_pts={expected_pts} out of [0, 6] bounds"
                );
            }
        }
    }

    #[test]
    fn test_make_grp_entry_negative_scores() {
        let entry = make_grp_entry(7, 3, 0, &[-5000, 35000, 40000, 30000]);
        assert_eq!(entry[0], 7.0);
        assert_eq!(entry[1], 3.0);
        assert_eq!(entry[2], 0.0);
        assert!((entry[3] - (-0.5)).abs() < 1e-10);
        assert!((entry[4] - 3.5).abs() < 1e-10);
        assert!((entry[5] - 4.0).abs() < 1e-10);
        assert!((entry[6] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_calc_player_probs_all_players_sum() {
        // For any logits, each player's rank probabilities should sum to 1,
        // and across all players, each rank should also sum to 1.
        let logits: Vec<f64> = (0..24).map(|i| (i as f64) * 0.3 - 2.0).collect();

        let all_probs: Vec<[f64; 4]> = (0..4_u8)
            .map(|pid| calc_player_probs(&logits, pid))
            .collect();

        // Each player's probs sum to 1
        for (pid, probs) in all_probs.iter().enumerate() {
            let sum: f64 = probs.iter().sum();
            assert!(
                (sum - 1.0).abs() < 1e-10,
                "player {pid}: probs sum to {sum}"
            );
        }

        // Each rank's probs across players sum to 1
        for rank in 0..4_usize {
            let sum: f64 = all_probs.iter().map(|p| p[rank]).sum();
            assert!(
                (sum - 1.0).abs() < 1e-10,
                "rank {rank}: probs across players sum to {sum}"
            );
        }
    }
}
