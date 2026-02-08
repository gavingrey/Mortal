use super::action::ActionCandidate;
use super::item::{ChiPon, KawaItem, Sutehai};
use crate::algo::sp::Candidate;
use crate::hand::tiles_to_string;
use crate::mjai::Event;
use crate::must_tile;
use crate::tile::Tile;
use std::iter;

use anyhow::Result;
use derivative::Derivative;
use pyo3::prelude::*;
use serde_json as json;
use tinyvec::{ArrayVec, TinyVec};

/// `PlayerState` is the core of the lib, which holds all the observable game
/// state information from a specific seat's perspective with the ability to
/// identify the legal actions the specified player can make upon an incoming
/// mjai event, along with some helper functions to build an actual agent.
/// Notably, `PlayerState` encodes observation features into numpy arrays which
/// serve as inputs for deep learning model.
#[pyclass]
#[derive(Clone, Derivative)]
#[derivative(Default)]
pub struct PlayerState {
    pub(super) player_id: u8,

    /// Does not include aka.
    #[derivative(Default(value = "[0; 34]"))]
    pub(super) tehai: [u8; 34],

    /// Does not consider yakunashi, but does consider other kinds of
    /// furiten.
    #[derivative(Default(value = "[false; 34]"))]
    pub(super) waits: [bool; 34],

    #[derivative(Default(value = "[0; 34]"))]
    pub(super) dora_factor: [u8; 34],

    /// For calculating `waits` and `doras_seen`, also for SPCalculator.
    #[derivative(Default(value = "[0; 34]"))]
    pub(super) tiles_seen: [u8; 34],

    /// For SPCalculator.
    pub(super) akas_seen: [bool; 3],

    #[derivative(Default(value = "[false; 34]"))]
    pub(super) keep_shanten_discards: [bool; 34],

    #[derivative(Default(value = "[false; 34]"))]
    pub(super) next_shanten_discards: [bool; 34],

    #[derivative(Default(value = "[false; 34]"))]
    pub(super) forbidden_tiles: [bool; 34],

    /// Used for furiten check.
    #[derivative(Default(value = "[false; 34]"))]
    pub(super) discarded_tiles: [bool; 34],

    pub(super) bakaze: Tile,
    pub(super) jikaze: Tile,
    /// Counts from 0 unlike mjai.
    pub(super) kyoku: u8,
    pub(super) honba: u8,
    pub(super) kyotaku: u8,
    /// Rotated to be relative, so `scores[0]` is the score of the player.
    pub(super) scores: [i32; 4],
    pub(super) rank: u8,
    /// Relative to `player_id`.
    pub(super) oya: u8,
    /// Including 西入 sudden death.
    pub(super) is_all_last: bool,
    pub(super) dora_indicators: ArrayVec<[Tile; 5]>,

    /// 24 is the theoretical max size of kawa, however, since None is included
    /// in the kawa, in some very rare cases (about one in a million hanchans),
    /// the size can exceed 24.
    ///
    /// Reference:
    /// <https://detail.chiebukuro.yahoo.co.jp/qa/question_detail/q1020002370>
    pub(super) kawa: [TinyVec<[Option<KawaItem>; 24]>; 4],
    pub(super) last_tedashis: [Option<Sutehai>; 4],
    pub(super) riichi_sutehais: [Option<Sutehai>; 4],

    /// Using 34-D arrays here may be more efficient, but I don't want to mess up
    /// with aka doras.
    pub(super) kawa_overview: [ArrayVec<[Tile; 24]>; 4],
    pub(super) fuuro_overview: [ArrayVec<[ArrayVec<[Tile; 4]>; 4]>; 4],
    /// In this field all `Tile` are deaka'd.
    pub(super) ankan_overview: [ArrayVec<[Tile; 4]>; 4],

    pub(super) riichi_declared: [bool; 4],
    pub(super) riichi_accepted: [bool; 4],

    pub(super) at_turn: u8,
    pub(super) tiles_left: u8,
    pub(super) intermediate_kan: ArrayVec<[Tile; 4]>,
    pub(super) intermediate_chi_pon: Option<ChiPon>,

    pub(super) shanten: i8,

    pub(super) last_self_tsumo: Option<Tile>,
    pub(super) last_kawa_tile: Option<Tile>,
    pub(super) last_cans: ActionCandidate,

    /// Both deaka'd
    pub(super) ankan_candidates: ArrayVec<[Tile; 3]>,
    pub(super) kakan_candidates: ArrayVec<[Tile; 3]>,
    pub(super) chankan_chance: Option<()>,

    pub(super) can_w_riichi: bool,
    pub(super) is_w_riichi: bool,
    pub(super) at_rinshan: bool,
    pub(super) at_ippatsu: bool,
    pub(super) at_furiten: bool,
    pub(super) to_mark_same_cycle_furiten: Option<()>,

    /// Used for 4-kan check.
    pub(super) kans_on_board: u8,

    pub(super) is_menzen: bool,
    /// For agari calc, all deaka'd.
    pub(super) chis: ArrayVec<[u8; 4]>,
    pub(super) pons: ArrayVec<[u8; 4]>,
    pub(super) minkans: ArrayVec<[u8; 4]>,
    pub(super) ankans: ArrayVec<[u8; 4]>,

    /// Including aka, originally for agari calc usage but also encoded as a
    /// feature to the obs.
    pub(super) doras_owned: [u8; 4],
    pub(super) doras_seen: u8,

    pub(super) akas_in_hand: [bool; 3],

    /// For shanten calc.
    pub(super) tehai_len_div3: u8,

    /// Used in can_riichi, also in single-player features to get the shanten
    /// for 3n+2.
    pub(super) has_next_shanten_discard: bool,

    /// Record of all events processed by this PlayerState.
    /// Used by the search module to replay events through fresh PlayerStates
    /// for mid-game board reconstruction.
    /// Only populated when `record_events` is true.
    pub(super) event_history: Vec<Event>,

    /// When true, events are recorded in `event_history`.
    /// Defaults to false to avoid overhead in training/arena paths.
    pub(super) record_events: bool,

    /// When true, witness_tile and move_tile errors are suppressed.
    /// Used during search replay where opponent hands contain dummy tiles
    /// that may cause overflow. The hand will be patched afterward.
    pub(super) replay_mode: bool,
}

#[pymethods]
impl PlayerState {
    /// Panics if `player_id` is outside of range [0, 3].
    #[new]
    #[must_use]
    pub fn new(player_id: u8) -> Self {
        assert!(player_id < 4, "{player_id} is not in range [0, 3]");
        Self {
            player_id,
            ..Default::default()
        }
    }

    /// Returns an `ActionCandidate`.
    #[pyo3(name = "update")]
    pub(super) fn update_json(&mut self, mjai_json: &str) -> Result<ActionCandidate> {
        let event = json::from_str(mjai_json)?;
        self.update(&event)
    }

    /// Raises an exception if the action is not valid.
    #[pyo3(name = "validate_reaction")]
    pub(super) fn validate_reaction_json(&self, mjai_json: &str) -> Result<()> {
        let action = json::from_str(mjai_json)?;
        self.validate_reaction(&action)
    }

    /// For debug only.
    ///
    /// Return a human readable description of the current state.
    #[must_use]
    pub fn brief_info(&self) -> String {
        let waits = self
            .waits
            .iter()
            .enumerate()
            .filter(|&(_, &b)| b)
            .map(|(i, _)| must_tile!(i))
            .collect::<Vec<_>>();

        let zipped_kawa = self.kawa[0]
            .iter()
            .chain(iter::repeat(&None))
            .zip(self.kawa[1].iter().chain(iter::repeat(&None)))
            .zip(self.kawa[2].iter().chain(iter::repeat(&None)))
            .zip(self.kawa[3].iter().chain(iter::repeat(&None)))
            .take_while(|row| !matches!(row, &(((None, None), None), None)))
            .enumerate()
            .map(|(i, (((a, b), c), d))| {
                format!(
                    "{i:2}. {}\t{}\t{}\t{}",
                    a.as_ref()
                        .map_or_else(|| "-".to_owned(), |item| item.to_string()),
                    b.as_ref()
                        .map_or_else(|| "-".to_owned(), |item| item.to_string()),
                    c.as_ref()
                        .map_or_else(|| "-".to_owned(), |item| item.to_string()),
                    d.as_ref()
                        .map_or_else(|| "-".to_owned(), |item| item.to_string()),
                )
            })
            .collect::<Vec<_>>()
            .join("\n");

        let can_discard = self.last_cans.can_discard;
        let mut sp_tables = Candidate::csv_header(can_discard).join("\t");
        if let Ok(tables) = self.single_player_tables() {
            for candidate in tables.max_ev_table {
                sp_tables.push('\n');
                sp_tables.push_str(&candidate.csv_row(can_discard).join("\t"));
            }
        }

        format!(
            r#"player (abs): {}
oya (rel): {}
kyoku: {}{}-{}
turn: {}
jikaze: {}
score (rel): {:?}
tehai: {}
fuuro: {:?}
ankan: {:?}
tehai len: {}
shanten: {} (actual: {})
furiten: {}
waits: {waits:?}
dora indicators: {:?}
doras owned: {:?}
doras seen: {}
action candidates: {:#?}
last self tsumo: {:?}
last kawa tile: {:?}
tiles left: {}
kawa:
{zipped_kawa}
single player table (max EV):
{sp_tables}"#,
            self.player_id,
            self.oya,
            self.bakaze,
            self.kyoku + 1,
            self.honba,
            self.at_turn,
            self.jikaze,
            self.scores,
            tiles_to_string(&self.tehai, self.akas_in_hand),
            self.fuuro_overview[0],
            self.ankan_overview[0],
            self.tehai_len_div3,
            self.shanten,
            self.real_time_shanten(),
            self.at_furiten,
            self.dora_indicators,
            self.doras_owned,
            self.doras_seen,
            self.last_cans,
            self.last_self_tsumo,
            self.last_kawa_tile,
            self.tiles_left,
        )
    }
}

// Methods for search module mid-game reconstruction
impl PlayerState {
    /// Enable or disable event recording.
    /// When enabled, all events are stored in `event_history` for
    /// later replay by the search module.
    pub const fn set_record_events(&mut self, enabled: bool) {
        self.record_events = enabled;
    }

    /// Enable replay mode, which suppresses witness/move tile errors.
    /// Used during search replay where dummy tiles may cause overflows.
    pub(crate) const fn set_replay_mode(&mut self, enabled: bool) {
        self.replay_mode = enabled;
    }

    /// Replace the opponent's closed hand with the particle's actual tiles.
    ///
    /// After replaying events through a fresh PlayerState, an opponent's
    /// tehai will contain "unknown" entries from "?" tsumo events. This
    /// method patches the tehai with the actual tiles from the particle,
    /// then recalculates derived state (shanten, waits, furiten, tiles_seen,
    /// doras_owned, doras_seen).
    pub(crate) fn patch_hand(&mut self, closed_tiles: &[Tile]) {
        // Reset tehai, aka tracking, and waits
        self.tehai = [0; 34];
        self.akas_in_hand = [false; 3];
        self.waits = [false; 34];
        self.at_furiten = false;

        // Note: `discarded_tiles` is intentionally NOT reset here.
        // It tracks this player's own discard history for furiten checking,
        // and was correctly populated during the replay phase (the Dahai
        // handler sets `discarded_tiles[pai] = true` for actor_rel==0
        // regardless of replay_mode). Since patch_hand only changes the
        // closed hand contents (not the discard history), discarded_tiles
        // remains valid.

        // Set up tehai and akas_in_hand from the patched hand
        for &tile in closed_tiles {
            let tile_id = tile.deaka().as_usize();
            if tile.is_aka() {
                let aka_idx = tile.as_usize() - crate::tuz!(5mr);
                self.akas_in_hand[aka_idx] = true;
            }
            self.tehai[tile_id] += 1;
        }

        // Recompute tiles_seen and akas_seen from scratch.
        //
        // During replay_mode, tiles that entered this player's hand through
        // unknown ("?") haipai/tsumo events were never witnessed, leaving
        // tiles_seen incomplete. Rather than trying to figure out which
        // tiles were already counted vs missing, we reset tiles_seen to
        // zero and recompute from all observable game state.
        //
        // All visible tiles to this player can be decomposed as:
        //   1. Own closed hand (from `closed_tiles` parameter)
        //   2. All players' discards (kawa_overview, 4 relative seats)
        //   3. Meld-consumed tiles from hand for all players' open melds
        //      (fuuro_overview, excluding called tiles which are already
        //      counted in the calling player's kawa)
        //   4. All players' closed kans (ankan_overview, 4 copies each)
        //   5. Dora indicators
        //
        // This decomposition counts each physical tile exactly once:
        // - kawa_overview includes called tiles (they remain in discard history)
        // - fuuro consumed tiles are those from the calling player's hand (NOT in kawa)
        // - Together, kawa + fuuro_consumed covers all discards + all meld tiles
        //   without double-counting
        self.tiles_seen = [0; 34];
        self.akas_seen = [false; 3];

        const fn mark_seen(tiles_seen: &mut [u8; 34], akas_seen: &mut [bool; 3], tile: Tile) {
            let tid = tile.deaka().as_usize();
            tiles_seen[tid] += 1;
            if tile.is_aka() {
                let aka_idx = tile.as_usize() - crate::tuz!(5mr);
                akas_seen[aka_idx] = true;
            }
        }

        // 1. Own closed hand
        for &tile in closed_tiles {
            mark_seen(&mut self.tiles_seen, &mut self.akas_seen, tile);
        }

        // 2. All players' discards (kawa_overview for all 4 relative seats)
        for kawa in &self.kawa_overview {
            for &tile in kawa {
                mark_seen(&mut self.tiles_seen, &mut self.akas_seen, tile);
            }
        }

        // 3. Meld-consumed tiles from hand for ALL players' open melds.
        //    Each meld in fuuro_overview is laid out as:
        //      chi:       [consumed[0], consumed[1], called_pai]       — 2 from hand
        //      pon:       [consumed[0], consumed[1], called_pai]       — 2 from hand
        //      daiminkan: [c[0], c[1], c[2], called_pai]              — 3 from hand
        //      kakan:     [pon_c[0], pon_c[1], pon_called, kakan_pai] — 3 from hand
        //    The called tile (1 per meld) is NOT added here because it's
        //    already counted via the discarder's kawa_overview above.
        for player_fuuro in &self.fuuro_overview {
            for fuuro in player_fuuro {
                let len = fuuro.len();
                if len == 3 {
                    // Chi or pon: first 2 tiles are from hand
                    mark_seen(&mut self.tiles_seen, &mut self.akas_seen, fuuro[0]);
                    mark_seen(&mut self.tiles_seen, &mut self.akas_seen, fuuro[1]);
                } else if len == 4 {
                    // Daiminkan or kakan: all same deaka type, 3 from hand.
                    // (daiminkan: indices 0,1,2; kakan: indices 0,1,3)
                    // Since all are same deaka type, just add 3 to the count.
                    let tid = fuuro[0].deaka().as_usize();
                    self.tiles_seen[tid] += 3;
                    // Track akas from any tile in the meld (idempotent for akas_seen)
                    for &t in fuuro {
                        if t.is_aka() {
                            let aka_idx = t.as_usize() - crate::tuz!(5mr);
                            self.akas_seen[aka_idx] = true;
                        }
                    }
                }
            }
        }

        // 4. All players' closed kans (4 copies each, deaka'd)
        for player_ankan in &self.ankan_overview {
            for &tile in player_ankan {
                let tid = tile.deaka().as_usize();
                self.tiles_seen[tid] += 4;
                // Ankan of 5m/5p/5s necessarily contains the aka
                // (only one aka of each type exists in the 136-tile set)
                if tid == crate::tuz!(5m) {
                    self.akas_seen[0] = true;
                } else if tid == crate::tuz!(5p) {
                    self.akas_seen[1] = true;
                } else if tid == crate::tuz!(5s) {
                    self.akas_seen[2] = true;
                }
            }
        }

        // 5. Dora indicators
        for &tile in &self.dora_indicators {
            mark_seen(&mut self.tiles_seen, &mut self.akas_seen, tile);
        }

        // Recalculate doras_owned[0] from scratch based on new hand
        // (doras_owned for other players are unaffected)
        self.doras_owned[0] = 0;
        for (tid, &count) in self.tehai.iter().enumerate() {
            self.doras_owned[0] += count * self.dora_factor[tid];
        }
        // Add aka dora bonus for akas in hand
        for &has_aka in &self.akas_in_hand {
            if has_aka {
                self.doras_owned[0] += 1;
            }
        }
        // Add doras from our open melds (fuuro_overview[0] and ankan_overview[0])
        for fuuro in &self.fuuro_overview[0] {
            for &t in fuuro {
                self.doras_owned[0] += self.dora_factor[t.deaka().as_usize()];
                if t.is_aka() {
                    self.doras_owned[0] += 1;
                }
            }
        }
        for &t in &self.ankan_overview[0] {
            let tid = t.deaka().as_usize();
            self.doras_owned[0] += self.dora_factor[tid] * 4;
            // Ankan of 5m/5p/5s always contains the aka (worth +1 dora)
            if tid == crate::tuz!(5m) || tid == crate::tuz!(5p) || tid == crate::tuz!(5s) {
                self.doras_owned[0] += 1;
            }
        }

        // Recalculate doras_seen from scratch based on tiles_seen + akas_seen
        self.doras_seen = 0;
        for (tid, &seen) in self.tiles_seen.iter().enumerate() {
            self.doras_seen += seen * self.dora_factor[tid];
        }
        for &aka_seen in &self.akas_seen {
            if aka_seen {
                self.doras_seen += 1;
            }
        }

        // Recalculate tehai_len_div3 from actual tile count
        let total: u8 = self.tehai.iter().sum();
        self.tehai_len_div3 = total / 3;

        // Recalculate shanten and waits/furiten
        // Only update shanten if we have a valid hand size (3n+1 or 3n+2)
        let remainder = total % 3;
        if remainder == 1 {
            // 3n+1: normal waiting state
            self.update_shanten();
            self.update_waits_and_furiten();
        } else if remainder == 2 {
            // 3n+2: after draw, before discard
            self.update_shanten();
            // Don't call update_waits_and_furiten (requires 3n+1)
        } else {
            // remainder == 0: unusual state (e.g. after 4 melds with 0 closed tiles)
            self.shanten = 6; // safe default
        }
    }
}
