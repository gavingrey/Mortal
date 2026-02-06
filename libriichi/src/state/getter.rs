use super::{ActionCandidate, PlayerState};
use crate::tile::Tile;

use pyo3::prelude::*;

#[pymethods]
impl PlayerState {
    #[getter]
    #[inline]
    #[must_use]
    pub const fn player_id(&self) -> u8 {
        self.player_id
    }
    #[getter]
    #[inline]
    #[must_use]
    pub const fn kyoku(&self) -> u8 {
        self.kyoku
    }
    #[getter]
    #[inline]
    #[must_use]
    pub const fn honba(&self) -> u8 {
        self.honba
    }
    #[getter]
    #[inline]
    #[must_use]
    pub const fn kyotaku(&self) -> u8 {
        self.kyotaku
    }
    #[getter]
    #[inline]
    #[must_use]
    pub const fn is_oya(&self) -> bool {
        self.oya == 0
    }

    #[getter]
    #[inline]
    #[must_use]
    pub const fn tehai(&self) -> [u8; 34] {
        self.tehai
    }
    #[getter]
    #[inline]
    #[must_use]
    pub const fn akas_in_hand(&self) -> [bool; 3] {
        self.akas_in_hand
    }

    #[getter]
    #[inline]
    #[must_use]
    pub fn chis(&self) -> &[u8] {
        &self.chis
    }
    #[getter]
    #[inline]
    #[must_use]
    pub fn pons(&self) -> &[u8] {
        &self.pons
    }
    #[getter]
    #[inline]
    #[must_use]
    pub fn minkans(&self) -> &[u8] {
        &self.minkans
    }
    #[getter]
    #[inline]
    #[must_use]
    pub fn ankans(&self) -> &[u8] {
        &self.ankans
    }

    #[getter]
    #[inline]
    #[must_use]
    pub const fn at_turn(&self) -> u8 {
        self.at_turn
    }
    #[getter]
    #[inline]
    #[must_use]
    pub const fn shanten(&self) -> i8 {
        self.shanten
    }
    #[getter]
    #[inline]
    #[must_use]
    pub const fn waits(&self) -> [bool; 34] {
        self.waits
    }

    #[inline]
    #[pyo3(name = "last_self_tsumo")]
    fn last_self_tsumo_py(&self) -> Option<String> {
        self.last_self_tsumo.map(|t| t.to_string())
    }
    #[inline]
    #[pyo3(name = "last_kawa_tile")]
    fn last_kawa_tile_py(&self) -> Option<String> {
        self.last_kawa_tile.map(|t| t.to_string())
    }

    #[getter]
    #[inline]
    #[must_use]
    pub const fn last_cans(&self) -> ActionCandidate {
        self.last_cans
    }

    #[inline]
    #[pyo3(name = "ankan_candidates")]
    fn ankan_candidates_py(&self) -> Vec<String> {
        self.ankan_candidates
            .iter()
            .map(|t| t.to_string())
            .collect()
    }
    #[inline]
    #[pyo3(name = "kakan_candidates")]
    fn kakan_candidates_py(&self) -> Vec<String> {
        self.kakan_candidates
            .iter()
            .map(|t| t.to_string())
            .collect()
    }

    #[getter]
    #[inline]
    #[must_use]
    pub const fn can_w_riichi(&self) -> bool {
        self.can_w_riichi
    }
    #[getter]
    #[inline]
    #[must_use]
    pub const fn self_riichi_declared(&self) -> bool {
        self.riichi_declared[0]
    }
    #[getter]
    #[inline]
    #[must_use]
    pub const fn self_riichi_accepted(&self) -> bool {
        self.riichi_accepted[0]
    }

    #[getter]
    #[inline]
    #[must_use]
    pub const fn at_furiten(&self) -> bool {
        self.at_furiten
    }
}

impl PlayerState {
    #[inline]
    #[must_use]
    pub const fn last_self_tsumo(&self) -> Option<Tile> {
        self.last_self_tsumo
    }
    #[inline]
    #[must_use]
    pub const fn last_kawa_tile(&self) -> Option<Tile> {
        self.last_kawa_tile
    }

    #[inline]
    #[must_use]
    pub fn ankan_candidates(&self) -> &[Tile] {
        &self.ankan_candidates
    }
    #[inline]
    #[must_use]
    pub fn kakan_candidates(&self) -> &[Tile] {
        &self.kakan_candidates
    }

    // --- Getters for search module ---

    #[inline]
    #[must_use]
    pub const fn tiles_seen(&self) -> [u8; 34] {
        self.tiles_seen
    }
    #[inline]
    #[must_use]
    pub const fn akas_seen(&self) -> [bool; 3] {
        self.akas_seen
    }
    #[inline]
    #[must_use]
    pub const fn tiles_left(&self) -> u8 {
        self.tiles_left
    }
    #[inline]
    #[must_use]
    pub const fn scores(&self) -> [i32; 4] {
        self.scores
    }
    #[inline]
    #[must_use]
    pub const fn riichi_declared(&self) -> [bool; 4] {
        self.riichi_declared
    }
    #[inline]
    #[must_use]
    pub const fn riichi_accepted(&self) -> [bool; 4] {
        self.riichi_accepted
    }

    /// Number of revealed dora indicators.
    #[inline]
    #[must_use]
    pub fn num_dora_indicators(&self) -> u8 {
        self.dora_indicators.len() as u8
    }

    /// Compute the number of tiles in each opponent's closed hand.
    ///
    /// Each opponent is indexed by relative seat (1, 2, 3) mapped to (0, 1, 2).
    /// Hand size = 13 - (tiles consumed by open melds) + (draws not yet discarded).
    ///
    /// We compute this from the observable meld structure:
    /// - Each chi/pon consumes 2 tiles from closed hand
    /// - Each daiminkan consumes 3 tiles from closed hand
    /// - Each ankan consumes 4 tiles from closed hand
    /// - Each kakan consumes 1 additional tile from closed hand
    ///
    /// But tracking draws precisely is hard from observer perspective. Instead,
    /// we use the fact that tiles_seen + hidden = 136, and hidden = opponent
    /// hands + wall. We know wall size (tiles_left), so:
    ///
    /// total_opponent_hand_tiles = 136 - tiles_seen_total - tiles_left
    ///
    /// For the per-opponent breakdown, we use the meld counts to determine
    /// tehai_len_div3 for each opponent, then the hand size is 3*n+1 or 3*n+2
    /// depending on whose turn it is.
    #[must_use]
    pub fn opponent_hand_sizes(&self) -> [u8; 3] {
        let mut sizes = [0u8; 3];

        for opp_rel_minus1 in 0..3 {
            let opp_rel = opp_rel_minus1 + 1;

            // Count number of melds (each reduces tehai_len_div3 by 1)
            let n_fuuro = self.fuuro_overview[opp_rel].len() as u8;
            let n_ankan = self.ankan_overview[opp_rel].len() as u8;
            let total_melds = n_fuuro + n_ankan;

            // tehai_len_div3 = 4 - total_melds
            let tehai_len_div3 = 4u8.saturating_sub(total_melds);

            // Base hand size is 3*n + 1 (waiting for draw or just discarded)
            // It's 3*n + 2 if the opponent just drew and hasn't discarded yet.
            // From our perspective, we can't always know this precisely,
            // so we use 3*n + 1 as the default (post-discard state).
            sizes[opp_rel_minus1] = tehai_len_div3 * 3 + 1;
        }

        // Adjust: the total of all opponent hands + wall must equal hidden tiles.
        // hidden = 136 - sum(tiles_seen)
        let total_seen: u16 = self.tiles_seen.iter().map(|&c| c as u16).sum();
        let total_hidden = 136u16.saturating_sub(total_seen);
        let total_opp: u16 = sizes.iter().map(|&s| s as u16).sum();
        let wall = self.tiles_left as u16;

        // If there's a mismatch (e.g., an opponent just drew), adjust.
        // The discrepancy should be exactly 0 or 1 (one opponent has 3n+2).
        let expected_opp = total_hidden.saturating_sub(wall);
        if expected_opp > total_opp {
            // One opponent has an extra tile (just drew, hasn't discarded).
            // We don't know which one for certain, but it's the current
            // turn player. We can approximate by distributing the extra tile.
            let diff = (expected_opp - total_opp) as u8;
            if diff <= 3 {
                // Distribute extra tiles starting from first opponent
                // In practice diff should be 0 or 1.
                for i in 0..diff.min(3) as usize {
                    sizes[i] += 1;
                }
            }
        }

        sizes
    }
}
