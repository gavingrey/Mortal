use super::MAX_TSUMOS_LEFT;
use super::tile::RequiredTile;
use crate::tile::Tile;
use std::cmp::Ordering;

use tinyvec::ArrayVec;

#[derive(Debug)]
pub struct Candidate {
    /// 打牌
    pub tile: Tile,
    /// 巡目ごとの聴牌確率
    pub tenpai_probs: ArrayVec<[f32; MAX_TSUMOS_LEFT]>,
    /// 巡目ごとの和了確率
    pub win_probs: ArrayVec<[f32; MAX_TSUMOS_LEFT]>,
    /// 巡目ごとの期待値
    pub exp_values: ArrayVec<[f32; MAX_TSUMOS_LEFT]>,
    /// 有効牌及び枚数の一覧
    pub required_tiles: ArrayVec<[RequiredTile; 34]>,
    pub num_required_tiles: u8,
    /// 向聴戻しになるかどうか
    pub shanten_down: bool,
}

#[derive(Default)]
pub(super) struct RawCandidate<'a> {
    pub(super) tile: Tile,
    pub(super) tenpai_probs: &'a [f32],
    pub(super) win_probs: &'a [f32],
    pub(super) exp_values: &'a [f32],
    pub(super) required_tiles: ArrayVec<[RequiredTile; 34]>,
    pub(super) shanten_down: bool,
}

#[derive(Clone, Copy)]
pub enum CandidateColumn {
    EV,
    WinProb,
    TenpaiProb,
    NotShantenDown,
    NumRequiredTiles,
    DiscardPriority,
}

impl From<RawCandidate<'_>> for Candidate {
    fn from(
        RawCandidate {
            tile,
            tenpai_probs,
            win_probs,
            exp_values,
            required_tiles,
            shanten_down,
        }: RawCandidate<'_>,
    ) -> Self {
        let num_required_tiles = required_tiles.iter().map(|r| r.count).sum();
        let tenpai_probs = tenpai_probs.iter().map(|p| p.clamp(0., 1.)).collect();
        let win_probs = win_probs.iter().map(|p| p.clamp(0., 1.)).collect();
        let exp_values = exp_values.iter().map(|v| v.max(0.)).collect();

        Self {
            tile,
            tenpai_probs,
            win_probs,
            exp_values,
            required_tiles,
            num_required_tiles,
            shanten_down,
        }
    }
}

impl Candidate {
    pub fn cmp(&self, other: &Self, by: CandidateColumn) -> Ordering {
        if self.tile == other.tile {
            return Ordering::Equal;
        }
        match by {
            CandidateColumn::EV => {
                let a = self.exp_values.first().copied().unwrap_or(0.0);
                let b = other.exp_values.first().copied().unwrap_or(0.0);
                match a.total_cmp(&b) {
                    Ordering::Equal => self.cmp(other, CandidateColumn::WinProb),
                    o => o,
                }
            }
            CandidateColumn::WinProb => {
                let a = self.win_probs.first().copied().unwrap_or(0.0);
                let b = other.win_probs.first().copied().unwrap_or(0.0);
                match a.total_cmp(&b) {
                    Ordering::Equal => self.cmp(other, CandidateColumn::TenpaiProb),
                    o => o,
                }
            }
            CandidateColumn::TenpaiProb => {
                let a = self.tenpai_probs.first().copied().unwrap_or(0.0);
                let b = other.tenpai_probs.first().copied().unwrap_or(0.0);
                match a.total_cmp(&b) {
                    Ordering::Equal => self.cmp(other, CandidateColumn::NotShantenDown),
                    o => o,
                }
            }
            CandidateColumn::NotShantenDown => match (self.shanten_down, other.shanten_down) {
                (false, true) => Ordering::Greater,
                (true, false) => Ordering::Less,
                _ => self.cmp(other, CandidateColumn::NumRequiredTiles),
            },
            CandidateColumn::NumRequiredTiles => {
                match self.num_required_tiles.cmp(&other.num_required_tiles) {
                    Ordering::Equal => self.cmp(other, CandidateColumn::DiscardPriority),
                    o => o,
                }
            }
            CandidateColumn::DiscardPriority => self.tile.cmp_discard_priority(other.tile),
        }
    }

    pub const fn csv_header(can_discard: bool) -> &'static [&'static str] {
        if can_discard {
            &[
                "Tile",
                "EV",
                "Win prob",
                "Tenpai prob",
                "Shanten down?",
                "Kinds",
                "Sum",
                "Required tiles",
            ]
        } else {
            &[
                "EV",
                "Win prob",
                "Tenpai prob",
                "Kinds",
                "Sum",
                "Required tiles",
            ]
        }
    }

    pub fn csv_row(&self, can_discard: bool) -> Vec<String> {
        let required_tiles = self
            .required_tiles
            .iter()
            .map(|r| format!("{}@{}", r.tile, r.count))
            .collect::<Vec<_>>()
            .join(",");

        // For high-shanten hands (4+), probability arrays are empty — guard access.
        let ev = self
            .exp_values
            .first()
            .map_or_else(|| "N/A".to_owned(), |v| format!("{v:.03}"));
        let win = self
            .win_probs
            .first()
            .map_or_else(|| "N/A".to_owned(), |v| format!("{:.03}", v * 100.));
        let tenpai = self
            .tenpai_probs
            .first()
            .map_or_else(|| "N/A".to_owned(), |v| format!("{:.03}", v * 100.));

        if can_discard {
            vec![
                self.tile.to_string(),
                ev,
                win,
                tenpai,
                if self.shanten_down { "Yes" } else { "No" }.to_owned(),
                self.required_tiles.len().to_string(),
                self.num_required_tiles.to_string(),
                required_tiles,
            ]
        } else {
            vec![
                ev,
                win,
                tenpai,
                self.required_tiles.len().to_string(),
                self.num_required_tiles.to_string(),
                required_tiles,
            ]
        }
    }

    #[cfg(feature = "sp_reproduce_cpp_ver")]
    #[allow(clippy::indexing_slicing)]
    pub(super) fn calibrate(mut self, real_max_tsumo: usize) -> Self {
        if self.shanten_down {
            // 向聴戻しをしない場合のパターンの確率が過小に算出されているような気がするため、
            // 帳尻をあわせるために1巡ずらしている → 本来必要ない処理なので、あとで消す
            self.tenpai_probs[0] = 0.;
            self.tenpai_probs.rotate_left(1);
            self.win_probs[0] = 0.;
            self.win_probs.rotate_left(1);
            self.exp_values[0] = 0.;
            self.exp_values.rotate_left(1);
        }
        self.tenpai_probs.rotate_right(real_max_tsumo);
        self.tenpai_probs.truncate(real_max_tsumo);
        self.win_probs.rotate_right(real_max_tsumo);
        self.win_probs.truncate(real_max_tsumo);
        self.exp_values.rotate_right(real_max_tsumo);
        self.exp_values.truncate(real_max_tsumo);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::t;

    /// Helper: create a Candidate with empty probability arrays (simulates 4+-shanten).
    fn empty_candidate(tile: Tile) -> Candidate {
        Candidate {
            tile,
            tenpai_probs: ArrayVec::new(),
            win_probs: ArrayVec::new(),
            exp_values: ArrayVec::new(),
            required_tiles: ArrayVec::new(),
            num_required_tiles: 0,
            shanten_down: false,
        }
    }

    /// Helper: create a Candidate with populated probability arrays.
    fn populated_candidate(tile: Tile, ev: f32, win: f32, tenpai: f32) -> Candidate {
        let mut exp_values = ArrayVec::new();
        exp_values.push(ev);
        let mut win_probs = ArrayVec::new();
        win_probs.push(win);
        let mut tenpai_probs = ArrayVec::new();
        tenpai_probs.push(tenpai);
        Candidate {
            tile,
            tenpai_probs,
            win_probs,
            exp_values,
            required_tiles: ArrayVec::new(),
            num_required_tiles: 0,
            shanten_down: false,
        }
    }

    #[test]
    fn csv_row_empty_arrays_discard_does_not_panic() {
        let c = empty_candidate(t!(1m));
        let row = c.csv_row(true);
        assert_eq!(row.len(), 8);
        assert_eq!(row[1], "N/A"); // EV
        assert_eq!(row[2], "N/A"); // Win prob
        assert_eq!(row[3], "N/A"); // Tenpai prob
    }

    #[test]
    fn csv_row_empty_arrays_no_discard_does_not_panic() {
        let c = empty_candidate(t!(5p));
        let row = c.csv_row(false);
        assert_eq!(row.len(), 6);
        assert_eq!(row[0], "N/A"); // EV
        assert_eq!(row[1], "N/A"); // Win prob
        assert_eq!(row[2], "N/A"); // Tenpai prob
    }

    #[test]
    fn csv_row_populated_arrays_discard() {
        let c = populated_candidate(t!(3s), 1500.0, 0.25, 0.75);
        let row = c.csv_row(true);
        assert_eq!(row.len(), 8);
        assert_eq!(row[0], "3s"); // Tile
        assert_eq!(row[1], "1500.000"); // EV
        assert_eq!(row[2], "25.000"); // Win prob %
        assert_eq!(row[3], "75.000"); // Tenpai prob %
    }

    #[test]
    fn csv_row_populated_arrays_no_discard() {
        let c = populated_candidate(t!(W), 800.0, 0.1, 0.5);
        let row = c.csv_row(false);
        assert_eq!(row.len(), 6);
        assert_eq!(row[0], "800.000"); // EV
        assert_eq!(row[1], "10.000"); // Win prob %
        assert_eq!(row[2], "50.000"); // Tenpai prob %
    }

    #[test]
    fn cmp_empty_arrays_does_not_panic() {
        let a = empty_candidate(t!(1m));
        let b = empty_candidate(t!(9m));
        // Both have EV=0.0 (empty fallback), so should fall through to lower columns
        let result = a.cmp(&b, CandidateColumn::EV);
        // Should not panic; exact ordering depends on tiebreak chain
        assert!(matches!(
            result,
            Ordering::Less | Ordering::Equal | Ordering::Greater
        ));
    }

    #[test]
    fn cmp_empty_vs_populated() {
        let empty = empty_candidate(t!(1m));
        let pop = populated_candidate(t!(9m), 1000.0, 0.5, 0.8);
        // Empty EV=0.0 < populated EV=1000.0
        assert_eq!(empty.cmp(&pop, CandidateColumn::EV), Ordering::Less);
        assert_eq!(pop.cmp(&empty, CandidateColumn::EV), Ordering::Greater);
    }

    #[test]
    fn cmp_same_tile_returns_equal() {
        let a = empty_candidate(t!(5m));
        let b = populated_candidate(t!(5m), 999.0, 0.9, 0.9);
        assert_eq!(a.cmp(&b, CandidateColumn::EV), Ordering::Equal);
    }
}
