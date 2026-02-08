use crate::arena::{Board, BoardState, MidgameContextBase, Poll};
#[cfg(test)]
use crate::arena::MidgameContext;
use crate::mjai::{Event, EventExt};
use crate::state::PlayerState;
use crate::tile::Tile;
use crate::{matches_tu8, must_tile, tu8};

use super::particle::Particle;

use anyhow::{Context, Result, bail, ensure};
use pyo3::prelude::*;
use rand_chacha::ChaCha12Rng;
use std::array;

/// Result of a single rollout simulation.
#[pyclass]
#[derive(Clone, Debug)]
pub struct RolloutResult {
    /// Final scores (absolute seats, 0-3).
    #[pyo3(get)]
    pub scores: [i32; 4],

    /// Score deltas from start of kyoku (absolute seats, 0-3).
    #[pyo3(get)]
    pub deltas: [i32; 4],

    /// Whether the kyoku ended with a win.
    #[pyo3(get)]
    pub has_hora: bool,

    /// Whether the kyoku ended with an abortive draw.
    #[pyo3(get)]
    pub has_abortive_ryukyoku: bool,

    /// Number of poll iterations (game steps) in the rollout.
    #[pyo3(get)]
    pub steps: u32,
}

#[pymethods]
impl RolloutResult {
    /// Get the score delta for a specific player (absolute seat 0-3).
    pub fn player_delta(&self, player_id: u8) -> PyResult<i32> {
        if player_id >= 4 {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "player_id must be 0-3, got {player_id}"
            )));
        }
        Ok(self.deltas[player_id as usize])
    }

    fn __repr__(&self) -> String {
        format!(
            "RolloutResult(scores={:?}, deltas={:?}, hora={}, abort={}, steps={})",
            self.scores, self.deltas, self.has_hora, self.has_abortive_ryukyoku, self.steps,
        )
    }
}

/// Derive all BoardState internal fields from the event history.
///
/// This accurately reconstructs every field that BoardState tracks during
/// normal game flow (via `step()`), so the search rollout can start
/// from the exact mid-game position.
///
/// Used by tests to inspect the full MidgameContext directly.
/// Production code uses `derive_midgame_context_base` + `with_states`.
#[cfg(test)]
fn derive_midgame_context(
    event_history: &[Event],
    player_states: [PlayerState; 4],
    tiles_left: u8,
) -> MidgameContext {
    let base = derive_midgame_context_base(event_history);
    base.with_states(player_states, tiles_left)
}

/// Derive the event-dependent portion of the midgame context.
///
/// This computes all fields that depend only on the event history (not on
/// the particle). Call this once per search, then combine with per-particle
/// player states via `MidgameContextBase::with_states()`.
///
/// Dora timing by kan type (mirrors board.rs step()):
///   Ankan:     Immediate reveal (add_new_dora called in step).
///              If need_new_dora_at_discard was pending, it's consumed
///              and revealed immediately too. Does NOT set need_new_dora_at_tsumo.
///   Daiminkan: Sets need_new_dora_at_discard (reveal at next discard).
///              If need_new_dora_at_discard was already pending (continuous kan),
///              promotes it to need_new_dora_at_tsumo.
///   Kakan:     Same as Daiminkan.
pub fn derive_midgame_context_base(
    event_history: &[Event],
) -> MidgameContextBase {
    let mut tsumo_actor: u8 = 0;
    let mut kans: u8 = 0;
    let mut accepted_riichis: u8 = 0;
    let mut can_nagashi_mangan = [true; 4];
    let mut can_four_wind = true;
    let mut four_wind_tile: Option<Tile> = None;
    let mut riichi_to_be_accepted: Option<u8> = None;
    let mut deal_from_rinshan: Option<()> = None;
    let mut need_new_dora_at_discard: Option<()> = None;
    let mut need_new_dora_at_tsumo: Option<()> = None;
    let mut check_four_kan = false;
    let mut paos: [Option<u8>; 4] = [None; 4];
    let mut dora_indicators_full: Vec<Tile> = Vec::new();

    // Track pons and minkans per player for pao detection
    let mut player_pons: [Vec<u8>; 4] = [vec![], vec![], vec![], vec![]];
    let mut player_minkans: [Vec<u8>; 4] = [vec![], vec![], vec![], vec![]];

    // Track per-player kan counts for four-kan abort detection
    let mut player_kans: [u8; 4] = [0; 4];

    // Track if it's still the first turn cycle (for can_w_riichi / four wind)
    let mut first_turn_active = true;
    let mut dahai_count = 0_u8;

    for event in event_history {
        match *event {
            Event::StartKyoku {
                dora_marker, oya, ..
            } => {
                tsumo_actor = oya;
                dora_indicators_full.push(dora_marker);
            }

            Event::Tsumo { actor, .. } => {
                // After a tsumo, the tsumo_actor is this actor
                tsumo_actor = actor;
                // Clear deal_from_rinshan after the draw happens
                deal_from_rinshan = None;
            }

            Event::Dahai { actor, pai, .. } => {
                // Next tsumo goes to (actor+1)%4
                tsumo_actor = (actor + 1) % 4;

                // Nagashi mangan: non-yaokyuu discard kills it
                if !pai.is_yaokyuu() {
                    can_nagashi_mangan[actor as usize] = false;
                }

                // Four wind logic
                if can_four_wind {
                    if matches_tu8!(pai.as_u8(), E | S | W | N) {
                        if first_turn_active {
                            if let Some(tile) = four_wind_tile {
                                if tile != pai {
                                    can_four_wind = false;
                                }
                            } else {
                                four_wind_tile = Some(pai);
                            }
                        } else if let Some(tile) = four_wind_tile {
                            // First turn just ended: check if final discard matches
                            if tile != pai {
                                can_four_wind = false;
                            }
                            // After the first cycle + 1 more discard, four_wind
                            // either triggers (handled by BoardState.step) or
                            // is resolved. We don't need to track further.
                        } else {
                            can_four_wind = false;
                        }
                    } else {
                        can_four_wind = false;
                    }
                }

                dahai_count += 1;
                // First turn cycle ends after 4 discards
                if dahai_count >= 4 {
                    first_turn_active = false;
                }

                // Four kan abort: after a discard, if 4+ kans exist
                // and no single player has all 4, set check_four_kan.
                // This mirrors BoardState.step()'s Dahai handler.
                if kans >= 4 && !player_kans.iter().any(|&k| k >= 4) {
                    check_four_kan = true;
                }

                // Clear need_new_dora_at_discard after the discard
                if need_new_dora_at_discard.take().is_some() {
                    // The dora was revealed (or should have been) at this discard
                    // The Dora event in the log handles the actual reveal
                }
            }

            Event::Chi { target, .. } | Event::Pon { target, .. } => {
                // Calls from player's kawa kill nagashi mangan for target
                can_nagashi_mangan[target as usize] = false;
                can_four_wind = false;

                // Accept pending riichi
                if let Some(actor) = riichi_to_be_accepted.take() {
                    accepted_riichis += 1;
                    // Note: the ReachAccepted event handles score changes,
                    // which are already reflected in the replayed PlayerStates
                    let _ = actor;
                }

                // Track pons for pao detection
                if let Event::Pon {
                    actor, target, pai, ..
                } = *event
                    && pai.is_jihai()
                {
                    player_pons[actor as usize].push(pai.deaka().as_u8());
                    check_pao(
                        actor,
                        target,
                        pai,
                        &player_pons[actor as usize],
                        &player_minkans[actor as usize],
                        &mut paos,
                    );
                }
            }

            Event::Daiminkan {
                actor, target, pai, ..
            } => {
                can_nagashi_mangan[target as usize] = false;
                can_four_wind = false;

                // Accept pending riichi
                if riichi_to_be_accepted.take().is_some() {
                    accepted_riichis += 1;
                }

                tsumo_actor = actor;
                deal_from_rinshan = Some(());
                need_new_dora_at_discard = Some(());
                kans += 1;
                player_kans[actor as usize] += 1;

                // Pao detection
                if pai.is_jihai() {
                    player_minkans[actor as usize].push(pai.deaka().as_u8());
                    check_pao(
                        actor,
                        target,
                        pai,
                        &player_pons[actor as usize],
                        &player_minkans[actor as usize],
                        &mut paos,
                    );
                }
            }

            Event::Kakan { actor, .. } => {
                // For Kakan: check if need_new_dora_at_discard is pending
                if need_new_dora_at_discard.is_some() {
                    need_new_dora_at_tsumo = Some(());
                }

                // Accept pending riichi (mirrors board.rs: Daiminkan | Kakan
                // share a match arm that calls check_riichi_accepted()).
                // In practice this is dead code: kakan can only happen on the
                // kakan player's own turn after drawing, so riichi_to_be_accepted
                // would already have been consumed by the Tsumo handler's preceding
                // draw step. But we include it to mirror board.rs exactly.
                if riichi_to_be_accepted.take().is_some() {
                    accepted_riichis += 1;
                }

                tsumo_actor = actor;
                deal_from_rinshan = Some(());
                need_new_dora_at_discard = Some(());
                kans += 1;
                player_kans[actor as usize] += 1;
            }

            Event::Ankan { actor, .. } => {
                can_four_wind = false;

                // For continuous kan (Daiminkan/Kakan followed by Ankan):
                // consume the pending dora. In board.rs step(), the Ankan
                // handler calls add_new_dora() immediately for both the
                // pending dora AND the Ankan's own dora. There is no
                // promotion to need_new_dora_at_tsumo here because both
                // doras are revealed immediately (unlike Daiminkan/Kakan
                // which defers to discard timing). The corresponding Dora
                // events appear in the event log right after Ankan.
                if need_new_dora_at_discard.take().is_some() {
                    // Consumed: the Dora event in the log handles the reveal.
                }

                tsumo_actor = actor;
                deal_from_rinshan = Some(());
                kans += 1;
                player_kans[actor as usize] += 1;
                // Ankan's own dora is also immediate (Dora event in log).
            }

            Event::Dora { dora_marker } => {
                dora_indicators_full.push(dora_marker);
                // Clear need_new_dora_at_tsumo if this Dora event was for a kakan
                need_new_dora_at_tsumo = None;
            }

            Event::Reach { actor } => {
                riichi_to_be_accepted = Some(actor);
            }

            Event::ReachAccepted { .. } => {
                accepted_riichis += 1;
                riichi_to_be_accepted = None;
            }

            _ => (),
        }
    }

    MidgameContextBase {
        tsumo_actor,
        kans,
        accepted_riichis,
        can_nagashi_mangan,
        can_four_wind,
        four_wind_tile,
        riichi_to_be_accepted,
        deal_from_rinshan,
        need_new_dora_at_discard,
        need_new_dora_at_tsumo,
        check_four_kan,
        paos,
        dora_indicators_full,
    }
}

/// Check if a pon/daiminkan triggers pao (responsibility payment).
///
/// Pao occurs when a player completes their 3rd+ jihai set
/// (daisangen or daisuushi) via pon/daiminkan from the same target.
fn check_pao(
    actor: u8,
    target: u8,
    pai: Tile,
    pons: &[u8],
    minkans: &[u8],
    paos: &mut [Option<u8>; 4],
) {
    let mut jihais = 0_u8;
    pons.iter()
        .chain(minkans.iter())
        .copied()
        .filter(|&t| t >= tu8!(E))
        .for_each(|t| jihais |= 1 << (t - tu8!(E)));

    let daisangen_confirmed = (jihais & 0b1110000) == 0b1110000;
    let daisuushi_confirmed = (jihais & 0b0001111) == 0b0001111;

    if daisangen_confirmed && matches_tu8!(pai.as_u8(), P | F | C)
        || daisuushi_confirmed && matches_tu8!(pai.as_u8(), E | S | W | N)
    {
        paos[actor as usize] = Some(target);
    }
}

/// Build a Board with ONLY the remaining tiles for the mid-game rollout.
///
/// Unlike the old approach that built a full 70-tile yama, this creates
/// a Board with just:
/// - yama: particle.wall tiles (remaining live wall)
/// - rinshan: remaining rinshan tiles from particle dead wall
/// - dora_indicators: unrevealed dora indicators from particle dead wall
/// - ura_indicators: ura indicators from particle dead wall
///
/// The haipai field is unused (mid-game state uses pre-built PlayerStates),
/// but must be populated with dummy values for Board struct completeness.
fn build_remaining_board(
    state: &PlayerState,
    particle: &Particle,
) -> Result<Board> {
    let player_id = state.player_id();
    let bakaze = state.bakaze();
    let kyoku = (bakaze.as_u8() - tu8!(E)) * 4 + state.kyoku();

    // Un-rotate scores from relative to absolute
    let rel_scores = state.scores();
    let mut abs_scores = [0_i32; 4];
    for i in 0..4 {
        abs_scores[(player_id as usize + i) % 4] = rel_scores[i];
    }

    // Yama: particle.wall in draw order, reversed for pop-from-back
    let yama: Vec<Tile> = particle.wall.iter().copied().rev().collect();

    // Dead wall: unrevealed dora indicators + rinshan + ura
    let revealed_dora = state.dora_indicators();
    let num_revealed = revealed_dora.len();
    let unseen = &particle.dead_wall;
    let unrevealed_dora_count = 5 - num_revealed;
    let fallback = must_tile!(0_u8);

    // Dora indicators: only the unrevealed ones go into Board.dora_indicators
    // (Board.dora_indicators.pop() is called to reveal new dora)
    let mut dora_indicators = Vec::with_capacity(5);
    for i in 0..unrevealed_dora_count {
        dora_indicators.push(unseen.get(i).copied().unwrap_or(fallback));
    }

    // Rinshan: remaining rinshan tiles (4 minus already drawn)
    // The number drawn = kans so far (each kan draws one rinshan)
    // We figure out remaining rinshan from dead_wall layout:
    // dead_wall = [unrevealed_dora_indicators..., rinshan..., ura_indicators...]
    let rinshan_start = unrevealed_dora_count;
    let mut rinshan = Vec::with_capacity(4);
    for i in 0..4 {
        if let Some(&tile) = unseen.get(rinshan_start + i) {
            rinshan.push(tile);
        }
    }

    let ura_start = rinshan_start + 4;
    let mut ura_indicators = Vec::with_capacity(5);
    for i in 0..5 {
        ura_indicators.push(
            unseen
                .get(ura_start + i)
                .copied()
                .unwrap_or(fallback),
        );
    }

    // Haipai is unused in mid-game construction but Board requires it.
    // Fill with dummy values.
    let haipai = [[Tile::default(); 13]; 4];

    // Handle kyotaku: BoardState tracks this on board.kyotaku.
    // After ReachAccepted, board.kyotaku is incremented and board.scores decremented.
    // Our player_states already have the post-riichi scores. But the Board.scores
    // should match the current actual scores (post-riichi deductions).
    // The scores in PlayerState are relative and already reflect riichi deductions.
    // So abs_scores from above should be correct.

    Ok(Board {
        kyoku,
        honba: state.honba(),
        kyotaku: state.kyotaku(),
        scores: abs_scores,
        haipai,
        yama,
        rinshan,
        dora_indicators,
        ura_indicators,
    })
}

/// Replay the event history through 4 fresh PlayerStates.
///
/// The event history contains "?" for opponent hands in StartKyoku
/// and for opponent Tsumo events. Opponent PlayerStates run in
/// replay_mode, which silently ignores unknown tile errors and
/// witness/move overflows. Our PlayerState (player_id) processes
/// real events normally.
///
/// After replay, replay_mode is disabled so the rollout uses normal
/// validation going forward.
pub fn replay_player_states(state: &PlayerState) -> Result<[PlayerState; 4]> {
    let events = state.event_history();
    let player_id = state.player_id();

    let mut replay_states: [PlayerState; 4] = array::from_fn(|i| {
        let mut ps = PlayerState::new(i as u8);
        if i as u8 != player_id {
            ps.set_replay_mode(true);
        }
        ps
    });

    for event in events {
        for rs in &mut replay_states {
            rs.update(event)
                .with_context(|| format!("replay failed on event {event:?}"))?;
        }
    }

    // Disable replay_mode now that replay is done
    for rs in &mut replay_states {
        rs.set_replay_mode(false);
    }

    Ok(replay_states)
}

/// Build a mid-game BoardState from pre-replayed PlayerStates and a particle.
///
/// This takes already-replayed states (from `replay_player_states`), patches
/// opponent hands with the particle's tiles, builds the Board, and constructs
/// the mid-game BoardState.
///
/// Note: This derives the midgame context from the event history each call.
/// For multiple particles from the same state, prefer
/// `build_midgame_board_state_with_base` to compute the base context once.
pub fn build_midgame_board_state_from_replayed(
    state: &PlayerState,
    replayed: &[PlayerState; 4],
    particle: &Particle,
) -> Result<BoardState> {
    let events = state.event_history();
    let base = derive_midgame_context_base(events);
    build_midgame_board_state_with_base(state, replayed, particle, &base)
}

/// Build a mid-game BoardState using a pre-computed `MidgameContextBase`.
///
/// This avoids re-deriving the event-dependent context for each particle.
/// Compute the base once with `derive_midgame_context_base`, then call this
/// for each particle.
pub fn build_midgame_board_state_with_base(
    state: &PlayerState,
    replayed: &[PlayerState; 4],
    particle: &Particle,
    base: &MidgameContextBase,
) -> Result<BoardState> {
    let player_id = state.player_id();

    // Clone the replayed states so we can patch them
    let mut patched = replayed.clone();

    // Patch opponent tehai with particle's actual tiles.
    for opp_idx in 0..3 {
        let abs_seat = (player_id as usize + opp_idx + 1) % 4;
        let opp_tiles = &particle.opponent_hands[opp_idx];
        patched[abs_seat].patch_hand(opp_tiles);
    }

    // Build Board with remaining tiles only
    let board = build_remaining_board(state, particle)?;

    // Combine pre-computed base context with per-particle player states
    let midgame = base.clone().with_states(patched, state.tiles_left());

    // Construct mid-game BoardState
    Ok(board.into_midgame_state(midgame))
}

/// Build a mid-game BoardState from the current PlayerState and a particle.
///
/// Convenience function that replays event history and builds the BoardState
/// in one step. For multiple particles/actions from the same state, prefer
/// using `replay_player_states` once + `build_midgame_board_state_from_replayed`
/// per particle.
pub fn build_midgame_board_state(
    state: &PlayerState,
    particle: &Particle,
) -> Result<BoardState> {
    let replayed = replay_player_states(state)?;
    build_midgame_board_state_from_replayed(state, &replayed, particle)
}

/// Convert an action index (0-45) to a concrete game event.
///
/// This mirrors the logic in `mortal.rs` `get_reaction()` but is standalone,
/// using only the `PlayerState` for context (no agent sync fields).
///
/// Action mapping:
/// - 0-33: Discard normal tile
/// - 34-36: Discard aka tile (5mr, 5pr, 5sr)
/// - 37: Riichi (skipped in search per team lead decision)
/// - 38: Chi low
/// - 39: Chi mid
/// - 40: Chi high
/// - 41: Pon
/// - 42: Kan (daiminkan, ankan, or kakan depending on state)
/// - 43: Agari (hora)
/// - 44: Ryukyoku
/// - 45: Pass
pub fn action_to_event(action: usize, actor: u8, state: &PlayerState) -> Result<Event> {
    let cans = state.last_cans();
    let akas_in_hand = state.akas_in_hand();

    let event = match action {
        0..=36 => {
            ensure!(
                cans.can_discard,
                "action_to_event: cannot discard (action={action})",
            );
            let pai = must_tile!(action);
            let tsumogiri = state.last_self_tsumo().is_some_and(|t| t == pai);
            Event::Dahai {
                actor,
                pai,
                tsumogiri,
            }
        }

        37 => {
            // Riichi: should be skipped in search candidates, but handle
            // gracefully if called.
            ensure!(
                cans.can_riichi,
                "action_to_event: cannot riichi",
            );
            Event::Reach { actor }
        }

        38 => {
            ensure!(
                cans.can_chi_low,
                "action_to_event: cannot chi low",
            );
            let pai = state
                .last_kawa_tile()
                .context("action_to_event: no last kawa tile for chi low")?;
            let first = pai.next();
            let can_akaize = match pai.as_u8() {
                tu8!(3m) | tu8!(4m) => akas_in_hand[0],
                tu8!(3p) | tu8!(4p) => akas_in_hand[1],
                tu8!(3s) | tu8!(4s) => akas_in_hand[2],
                _ => false,
            };
            let consumed = if can_akaize {
                [first.akaize(), first.next().akaize()]
            } else {
                [first, first.next()]
            };
            Event::Chi {
                actor,
                target: cans.target_actor,
                pai,
                consumed,
            }
        }

        39 => {
            ensure!(
                cans.can_chi_mid,
                "action_to_event: cannot chi mid",
            );
            let pai = state
                .last_kawa_tile()
                .context("action_to_event: no last kawa tile for chi mid")?;
            let can_akaize = match pai.as_u8() {
                tu8!(4m) | tu8!(6m) => akas_in_hand[0],
                tu8!(4p) | tu8!(6p) => akas_in_hand[1],
                tu8!(4s) | tu8!(6s) => akas_in_hand[2],
                _ => false,
            };
            let consumed = if can_akaize {
                [pai.prev().akaize(), pai.next().akaize()]
            } else {
                [pai.prev(), pai.next()]
            };
            Event::Chi {
                actor,
                target: cans.target_actor,
                pai,
                consumed,
            }
        }

        40 => {
            ensure!(
                cans.can_chi_high,
                "action_to_event: cannot chi high",
            );
            let pai = state
                .last_kawa_tile()
                .context("action_to_event: no last kawa tile for chi high")?;
            let last = pai.prev();
            let can_akaize = match pai.as_u8() {
                tu8!(6m) | tu8!(7m) => akas_in_hand[0],
                tu8!(6p) | tu8!(7p) => akas_in_hand[1],
                tu8!(6s) | tu8!(7s) => akas_in_hand[2],
                _ => false,
            };
            let consumed = if can_akaize {
                [last.prev().akaize(), last.akaize()]
            } else {
                [last.prev(), last]
            };
            Event::Chi {
                actor,
                target: cans.target_actor,
                pai,
                consumed,
            }
        }

        41 => {
            ensure!(
                cans.can_pon,
                "action_to_event: cannot pon",
            );
            let pai = state
                .last_kawa_tile()
                .context("action_to_event: no last kawa tile for pon")?;
            let can_akaize = match pai.as_u8() {
                tu8!(5m) => akas_in_hand[0],
                tu8!(5p) => akas_in_hand[1],
                tu8!(5s) => akas_in_hand[2],
                _ => false,
            };
            let consumed = if can_akaize {
                [pai.akaize(), pai.deaka()]
            } else {
                [pai.deaka(); 2]
            };
            Event::Pon {
                actor,
                target: cans.target_actor,
                pai,
                consumed,
            }
        }

        42 => {
            ensure!(
                cans.can_daiminkan || cans.can_ankan || cans.can_kakan,
                "action_to_event: cannot kan",
            );

            if cans.can_daiminkan {
                let pai = state
                    .last_kawa_tile()
                    .context("action_to_event: no last kawa tile for daiminkan")?;
                let consumed = if pai.is_aka() {
                    [pai.deaka(); 3]
                } else {
                    [pai.akaize(), pai, pai]
                };
                Event::Daiminkan {
                    actor,
                    target: cans.target_actor,
                    pai,
                    consumed,
                }
            } else if cans.can_ankan {
                let tile = state.ankan_candidates()[0];
                Event::Ankan {
                    actor,
                    consumed: [tile.akaize(), tile, tile, tile],
                }
            } else {
                // kakan
                let tile = state.kakan_candidates()[0];
                let can_akaize = match tile.as_u8() {
                    tu8!(5m) => akas_in_hand[0],
                    tu8!(5p) => akas_in_hand[1],
                    tu8!(5s) => akas_in_hand[2],
                    _ => false,
                };
                let (pai, consumed) = if can_akaize {
                    (tile.akaize(), [tile.deaka(); 3])
                } else {
                    (tile.deaka(), [tile.akaize(), tile.deaka(), tile.deaka()])
                };
                Event::Kakan {
                    actor,
                    pai,
                    consumed,
                }
            }
        }

        43 => {
            ensure!(
                cans.can_agari(),
                "action_to_event: cannot agari",
            );
            Event::Hora {
                actor,
                target: cans.target_actor,
                deltas: None,
                ura_markers: None,
            }
        }

        44 => {
            ensure!(
                cans.can_ryukyoku,
                "action_to_event: cannot ryukyoku",
            );
            Event::Ryukyoku { deltas: None }
        }

        45 => Event::None,

        _ => bail!("action_to_event: invalid action index {action}"),
    };

    Ok(event)
}

/// Default reaction for a seat during rollout: take wins, tsumogiri, pass calls.
///
/// This extracts the common tsumogiri logic used in both the plain tsumogiri
/// rollout and the action-branching rollout (for seats other than our player,
/// or for our player after the first action).
///
/// After chi/pon, `last_self_tsumo()` is None but `can_discard` is true.
/// In that case, we pick the first available tile from tehai as a discard.
pub fn default_reaction(seat: u8, state: &PlayerState) -> Event {
    let cans = state.last_cans();
    if !cans.can_act() {
        return Event::None;
    }

    if cans.can_tsumo_agari {
        return Event::Hora {
            actor: seat,
            target: seat,
            deltas: None,
            ura_markers: None,
        };
    }
    if cans.can_ron_agari {
        return Event::Hora {
            actor: seat,
            target: cans.target_actor,
            deltas: None,
            ura_markers: None,
        };
    }
    if cans.can_discard {
        if let Some(tsumo) = state.last_self_tsumo() {
            return Event::Dahai {
                actor: seat,
                pai: tsumo,
                tsumogiri: true,
            };
        }
        // Post-chi/pon: no tsumo tile, pick first available non-forbidden from hand
        let tehai = state.tehai();
        for (i, &count) in tehai.iter().enumerate() {
            if count > 0 && !state.forbidden_tiles()[i] {
                return Event::Dahai {
                    actor: seat,
                    pai: super::heuristic::resolve_aka(must_tile!(i as u8), state),
                    tsumogiri: false,
                };
            }
        }
        // Fallback without kuikae check (shouldn't happen, but be defensive)
        let tehai = state.tehai();
        for (i, &count) in tehai.iter().enumerate() {
            if count > 0 {
                return Event::Dahai {
                    actor: seat,
                    pai: super::heuristic::resolve_aka(must_tile!(i as u8), state),
                    tsumogiri: false,
                };
            }
        }
    }
    // Pass on all calls
    Event::None
}

/// Run a rollout to completion from a given BoardState.
///
/// Optionally injects a specific action at the first matching decision
/// point for `inject_player`. If `inject_action` is `None`, uses
/// tsumogiri for all players throughout.
///
/// This is the common loop shared by all rollout variants.
fn run_rollout(
    mut board_state: BoardState,
    initial_scores: [i32; 4],
    inject_player: Option<u8>,
    inject_action: Option<usize>,
    mut smart_rng: Option<&mut ChaCha12Rng>,
) -> Result<RolloutResult> {
    let mut reactions: [EventExt; 4] = Default::default();
    let mut action_injected = inject_action.is_none(); // already "injected" if None
    let mut steps = 0_u32;

    loop {
        match board_state.poll(reactions)? {
            Poll::InGame => {
                steps += 1;
                reactions = Default::default();
                let ctx = board_state.agent_context();

                for (seat, ps) in ctx.player_states.iter().enumerate() {
                    let cans = ps.last_cans();
                    if !cans.can_act() {
                        continue;
                    }

                    let ev = if !action_injected
                        && inject_player.is_some_and(|pid| seat as u8 == pid)
                    {
                        let action = inject_action.unwrap();
                        let player_id = inject_player.unwrap();

                        // Check if the action type matches what's available.
                        let matches = match action {
                            0..=36 => cans.can_discard,
                            37 => cans.can_riichi,
                            38 => cans.can_chi_low,
                            39 => cans.can_chi_mid,
                            40 => cans.can_chi_high,
                            41 => cans.can_pon,
                            42 => cans.can_daiminkan || cans.can_ankan || cans.can_kakan,
                            43 => cans.can_agari(),
                            44 => cans.can_ryukyoku,
                            // Pass is only valid for call decisions (not discard).
                            // When we can_discard, we must discard something.
                            45 => cans.can_pass() && !cans.can_discard,
                            _ => false,
                        };

                        if matches {
                            action_injected = true;
                            action_to_event(action, player_id, ps)?
                        } else {
                            // Not the right decision point yet (e.g., we got
                            // a can_discard but our action is chi). Use default.
                            if let Some(ref mut rng) = smart_rng {
                                super::heuristic::smart_reaction(seat as u8, ps, rng)
                            } else {
                                default_reaction(seat as u8, ps)
                            }
                        }
                    } else if let Some(ref mut rng) = smart_rng {
                        super::heuristic::smart_reaction(seat as u8, ps, rng)
                    } else {
                        default_reaction(seat as u8, ps)
                    };
                    reactions[seat] = EventExt::no_meta(ev);
                }
            }
            Poll::End => {
                let result = board_state.end();
                let deltas = [
                    result.scores[0] - initial_scores[0],
                    result.scores[1] - initial_scores[1],
                    result.scores[2] - initial_scores[2],
                    result.scores[3] - initial_scores[3],
                ];
                return Ok(RolloutResult {
                    scores: result.scores,
                    deltas,
                    has_hora: result.has_hora,
                    has_abortive_ryukyoku: result.has_abortive_ryukyoku,
                    steps,
                });
            }
        }
    }
}

/// Run a rollout where we inject a specific action at the first decision point
/// for `player_id`, then use tsumogiri for all subsequent decisions.
pub fn simulate_action_rollout(
    state: &PlayerState,
    particle: &Particle,
    action: usize,
) -> Result<RolloutResult> {
    let player_id = state.player_id();
    let board_state = build_midgame_board_state(state, particle)?;
    let initial_scores = board_state.board.scores;
    run_rollout(board_state, initial_scores, Some(player_id), Some(action), None)
}

/// Run a rollout with a specific action, using pre-replayed PlayerStates.
///
/// This avoids re-replaying the event history for each particle/action
/// combination. Use `replay_player_states` once, then call this for each
/// particle/action pair.
pub fn simulate_action_rollout_from_replayed(
    state: &PlayerState,
    replayed: &[PlayerState; 4],
    particle: &Particle,
    action: usize,
) -> Result<RolloutResult> {
    let player_id = state.player_id();
    let board_state = build_midgame_board_state_from_replayed(state, replayed, particle)?;
    let initial_scores = board_state.board.scores;
    run_rollout(board_state, initial_scores, Some(player_id), Some(action), None)
}

/// Run a rollout with a specific action, using pre-replayed PlayerStates
/// AND a pre-computed `MidgameContextBase`.
///
/// This is the most optimized variant: both the event replay and the
/// midgame context derivation are done once, then reused across all
/// particle/action combinations.
pub fn simulate_action_rollout_with_base(
    state: &PlayerState,
    replayed: &[PlayerState; 4],
    particle: &Particle,
    action: usize,
    base: &MidgameContextBase,
) -> Result<RolloutResult> {
    let player_id = state.player_id();
    let board_state = build_midgame_board_state_with_base(state, replayed, particle, base)?;
    let initial_scores = board_state.board.scores;
    run_rollout(board_state, initial_scores, Some(player_id), Some(action), None)
}

/// Run a tsumogiri rollout on a pre-built BoardState.
///
/// All players always discard their drawn tile (or take agari).
pub fn run_rollout_tsumogiri(
    board_state: BoardState,
    initial_scores: [i32; 4],
) -> Result<RolloutResult> {
    run_rollout(board_state, initial_scores, None, None, None)
}

/// Run a tsumogiri rollout: all players always discard their drawn tile.
///
/// This is the simplest possible rollout for Phase 1 testing.
/// Returns the kyoku result with score information.
pub fn simulate_rollout_tsumogiri(board: Board) -> Result<RolloutResult> {
    let initial_scores = board.scores;
    let board_state = board.into_state();
    run_rollout_tsumogiri(board_state, initial_scores)
}

/// Build a mid-game BoardState and run a tsumogiri rollout.
pub fn simulate_particle(
    state: &PlayerState,
    particle: &Particle,
) -> Result<RolloutResult> {
    let board_state = build_midgame_board_state(state, particle)?;
    let initial_scores = board_state.board.scores;
    run_rollout(board_state, initial_scores, None, None, None)
}

/// Run a smart-heuristic rollout on a pre-built BoardState.
///
/// All players use the `smart_reaction` heuristic policy (shanten-based
/// discard with ukeire and safety scoring) instead of tsumogiri.
pub fn run_rollout_smart(
    board_state: BoardState,
    initial_scores: [i32; 4],
    rng: &mut ChaCha12Rng,
) -> Result<RolloutResult> {
    run_rollout(board_state, initial_scores, None, None, Some(rng))
}

/// Simulate a single particle rollout with a specific action and smart heuristic.
///
/// The action is injected at the first matching decision point, then all
/// subsequent decisions use the smart heuristic policy.
pub fn simulate_action_rollout_with_base_smart(
    state: &PlayerState,
    replayed: &[PlayerState; 4],
    particle: &Particle,
    action: usize,
    base: &MidgameContextBase,
    rng: &mut ChaCha12Rng,
) -> Result<RolloutResult> {
    let player_id = state.player_id();
    let board_state = build_midgame_board_state_with_base(state, replayed, particle, base)?;
    let initial_scores = board_state.board.scores;
    run_rollout(board_state, initial_scores, Some(player_id), Some(action), Some(rng))
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::search::config::ParticleConfig;
    use crate::search::particle::generate_particles;
    use crate::search::test_utils::setup_basic_game;
    use rand::SeedableRng;
    use rand_chacha::ChaCha12Rng;

    #[test]
    fn build_board_basic() {
        let state = setup_basic_game();
        let config = ParticleConfig::new(1);
        let mut rng = ChaCha12Rng::seed_from_u64(42);

        let (particles, _attempts) = generate_particles(&state, &config, &mut rng).unwrap();
        assert!(!particles.is_empty());

        let bs = build_midgame_board_state(&state, &particles[0]).unwrap();

        // Board should have correct metadata
        assert_eq!(bs.board.kyoku, 0); // kyoku is 0-indexed
        assert_eq!(bs.board.honba, 0);
        assert_eq!(bs.board.scores, [25000; 4]);

        // Yama should have wall tiles
        assert!(bs.board.yama.len() >= 1);
    }

    #[test]
    fn simulate_tsumogiri_completes() {
        let state = setup_basic_game();
        let config = ParticleConfig::new(5);
        let mut rng = ChaCha12Rng::seed_from_u64(42);

        let (particles, _attempts) = generate_particles(&state, &config, &mut rng).unwrap();
        assert!(!particles.is_empty());

        // Simulate each particle
        for (i, particle) in particles.iter().enumerate() {
            let result = simulate_particle(&state, particle);
            assert!(
                result.is_ok(),
                "particle {i} simulation failed: {:?}",
                result.err()
            );

            let result = result.unwrap();
            // Score deltas should sum to zero (or close, accounting for riichi sticks)
            let delta_sum: i32 = result.deltas.iter().sum();
            // In a normal game, deltas sum to 0 (or multiples of 1000 for kyotaku)
            assert!(
                delta_sum.abs() <= 1000,
                "particle {i}: delta sum {delta_sum} is too large"
            );
        }
    }

    #[test]
    fn simulate_multiple_particles_complete() {
        let state = setup_basic_game();
        let config = ParticleConfig::new(20);
        let mut rng = ChaCha12Rng::seed_from_u64(123);

        let (particles, _attempts) = generate_particles(&state, &config, &mut rng).unwrap();

        let mut results: Vec<RolloutResult> = Vec::new();
        for particle in &particles {
            if let Ok(result) = simulate_particle(&state, particle) {
                results.push(result);
            }
        }

        assert!(
            !results.is_empty(),
            "at least some simulations should succeed"
        );
    }

    /// Set up a game where player 2 is NOT oya (oya=0, kyoku=1).
    /// Player 2 draws a tile, giving them 14 tiles.
    fn setup_non_oya_game() -> PlayerState {
        let mut state = PlayerState::new(2);
        state.set_record_events(true);

        let bakaze: Tile = "E".parse().unwrap();
        let dora_marker: Tile = "9s".parse().unwrap();

        let tehais = [
            // Player 0 (oya)
            ["?"; 13].map(|s| s.parse::<Tile>().unwrap()),
            // Player 1
            ["?"; 13].map(|s| s.parse::<Tile>().unwrap()),
            // Player 2 (us)
            [
                "1m", "2m", "3m", "4p", "5p", "6p", "7s", "8s", "9s", "1s", "2s", "3s", "E",
            ]
            .map(|s| s.parse::<Tile>().unwrap()),
            // Player 3
            ["?"; 13].map(|s| s.parse::<Tile>().unwrap()),
        ];

        let start_event = Event::StartKyoku {
            bakaze,
            dora_marker,
            kyoku: 1, // oya = seat 0
            honba: 0,
            kyotaku: 0,
            oya: 0,
            scores: [25000; 4],
            tehais,
        };
        state.update(&start_event).unwrap();

        // Oya draws (not us)
        state
            .update(&Event::Tsumo {
                actor: 0,
                pai: "N".parse().unwrap(),
            })
            .unwrap();
        // Oya discards
        state
            .update(&Event::Dahai {
                actor: 0,
                pai: "N".parse().unwrap(),
                tsumogiri: true,
            })
            .unwrap();
        // Player 1 draws
        state
            .update(&Event::Tsumo {
                actor: 1,
                pai: "?".parse().unwrap(),
            })
            .unwrap();
        // Player 1 discards
        state
            .update(&Event::Dahai {
                actor: 1,
                pai: "W".parse().unwrap(),
                tsumogiri: true,
            })
            .unwrap();
        // Player 2 (us) draws
        state
            .update(&Event::Tsumo {
                actor: 2,
                pai: "S".parse().unwrap(),
            })
            .unwrap();

        // Now player 2 has 14 tiles and it's our turn to discard
        state
    }

    #[test]
    fn build_board_non_oya() {
        // Verify that mid-game board state works correctly for non-oya
        // player (player_id=2, oya=0).
        let state = setup_non_oya_game();
        let config = ParticleConfig::new(5);
        let mut rng = ChaCha12Rng::seed_from_u64(42);

        let (particles, _attempts) = generate_particles(&state, &config, &mut rng).unwrap();
        assert!(!particles.is_empty());

        let bs = build_midgame_board_state(&state, &particles[0]).unwrap();

        // Verify kyoku and scores
        assert_eq!(bs.board.kyoku, 0); // East 1
        assert_eq!(bs.board.scores, [25000; 4]);

        // Player 2 should have a tsumo tile and be able to discard
        assert!(
            bs.player_states[2].last_cans().can_discard,
            "player 2 should be able to discard after midgame reconstruction"
        );

        // Yama should match tiles_left from state
        assert_eq!(
            bs.board.yama.len(),
            state.tiles_left() as usize,
            "yama length should match tiles_left"
        );
    }

    #[test]
    fn simulate_non_oya_completes() {
        // Verify that simulation completes for non-oya player.
        let state = setup_non_oya_game();
        let config = ParticleConfig::new(5);
        let mut rng = ChaCha12Rng::seed_from_u64(42);

        let (particles, _attempts) = generate_particles(&state, &config, &mut rng).unwrap();
        assert!(!particles.is_empty());

        for (i, particle) in particles.iter().enumerate() {
            let result = simulate_particle(&state, particle);
            assert!(
                result.is_ok(),
                "non-oya particle {i} simulation failed: {:?}",
                result.err()
            );

            let result = result.unwrap();
            let delta_sum: i32 = result.deltas.iter().sum();
            assert!(
                delta_sum.abs() <= 1000,
                "non-oya particle {i}: delta sum {delta_sum} is too large"
            );
        }
    }

    #[test]
    fn rollout_result_player_delta() {
        let result = RolloutResult {
            scores: [26000, 25000, 24000, 25000],
            deltas: [1000, 0, -1000, 0],
            has_hora: false,
            has_abortive_ryukyoku: false,
            steps: 0,
        };
        assert_eq!(result.player_delta(0).unwrap(), 1000);
        assert_eq!(result.player_delta(2).unwrap(), -1000);
    }

    // ---- Tests for action_to_event ----

    #[test]
    fn action_to_event_discard_normal() {
        let state = setup_basic_game();
        // Player 0 has tiles 1-9m, 1-4p, tsumo 5p. Can discard.
        // Action 0 = discard 1m
        let ev = action_to_event(0, 0, &state).unwrap();
        match ev {
            Event::Dahai {
                actor,
                pai,
                tsumogiri,
            } => {
                assert_eq!(actor, 0);
                assert_eq!(pai.as_u8(), 0); // 1m = tile id 0
                assert!(!tsumogiri); // 1m is not the tsumo tile (5p)
            }
            _ => panic!("expected Dahai, got {ev:?}"),
        }
    }

    #[test]
    fn action_to_event_discard_tsumogiri() {
        let state = setup_basic_game();
        // Tsumo tile is 5p. Action 13 = 5p (tile id 13).
        let ev = action_to_event(13, 0, &state).unwrap();
        match ev {
            Event::Dahai {
                tsumogiri, pai, ..
            } => {
                assert_eq!(pai.as_u8(), 13); // 5p
                assert!(tsumogiri);
            }
            _ => panic!("expected Dahai, got {ev:?}"),
        }
    }

    #[test]
    fn action_to_event_pass() {
        let state = setup_basic_game();
        // Pass should always return Event::None
        let ev = action_to_event(45, 0, &state).unwrap();
        assert_eq!(ev, Event::None);
    }

    #[test]
    fn action_to_event_agari_tsumo() {
        // Set up a tenpai hand that can tsumo agari
        let mut state = PlayerState::new(0);
        let bakaze: Tile = "E".parse().unwrap();
        let dora_marker: Tile = "9s".parse().unwrap();
        let tehais = [
            [
                "1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "1p", "2p", "3p", "4p",
            ]
            .map(|s| s.parse::<Tile>().unwrap()),
            ["?"; 13].map(|s| s.parse::<Tile>().unwrap()),
            ["?"; 13].map(|s| s.parse::<Tile>().unwrap()),
            ["?"; 13].map(|s| s.parse::<Tile>().unwrap()),
        ];
        let start_event = Event::StartKyoku {
            bakaze,
            dora_marker,
            kyoku: 1,
            honba: 0,
            kyotaku: 0,
            oya: 0,
            scores: [25000; 4],
            tehais,
        };
        state.update(&start_event).unwrap();

        // Tsumo 5p to complete the hand (1-9m, 1-5p)
        state
            .update(&Event::Tsumo {
                actor: 0,
                pai: "5p".parse().unwrap(),
            })
            .unwrap();

        let cans = state.last_cans();
        if cans.can_tsumo_agari {
            let ev = action_to_event(43, 0, &state).unwrap();
            match ev {
                Event::Hora {
                    actor, target, ..
                } => {
                    assert_eq!(actor, 0);
                    assert_eq!(target, cans.target_actor);
                }
                _ => panic!("expected Hora, got {ev:?}"),
            }
        }
    }

    #[test]
    fn action_to_event_invalid_action_fails() {
        let state = setup_basic_game();
        // Action 46 is out of range
        let result = action_to_event(46, 0, &state);
        result.unwrap_err();
    }

    #[test]
    fn action_to_event_riichi_when_unavailable_fails() {
        let state = setup_basic_game();
        // Our hand isn't necessarily tenpai for riichi
        let cans = state.last_cans();
        if !cans.can_riichi {
            let result = action_to_event(37, 0, &state);
            result.unwrap_err();
        }
    }

    // ---- Tests for action_to_event: chi/pon/kan (actions 38-42) ----

    /// Helper: set up player 0 (oya) state, play through one full round so
    /// player 3 discards `target_tile`, enabling chi for player 0.
    /// Player 0's hand must contain tiles for the desired chi type.
    fn setup_for_chi(our_hand: [&str; 13], target_tile: &str) -> PlayerState {
        let mut state = PlayerState::new(0);
        state.set_record_events(true);

        let bakaze: Tile = "E".parse().unwrap();
        let dora_marker: Tile = "9s".parse().unwrap();

        let tehais = [
            our_hand.map(|s| s.parse::<Tile>().unwrap()),
            ["?"; 13].map(|s| s.parse::<Tile>().unwrap()),
            ["?"; 13].map(|s| s.parse::<Tile>().unwrap()),
            ["?"; 13].map(|s| s.parse::<Tile>().unwrap()),
        ];

        state
            .update(&Event::StartKyoku {
                bakaze,
                dora_marker,
                kyoku: 1,
                honba: 0,
                kyotaku: 0,
                oya: 0,
                scores: [25000; 4],
                tehais,
            })
            .unwrap();

        // Player 0 (oya) draws and discards N
        state.update(&Event::Tsumo { actor: 0, pai: "N".parse().unwrap() }).unwrap();
        state.update(&Event::Dahai { actor: 0, pai: "N".parse().unwrap(), tsumogiri: true }).unwrap();

        // Player 1 draws and discards W
        state.update(&Event::Tsumo { actor: 1, pai: "?".parse().unwrap() }).unwrap();
        state.update(&Event::Dahai { actor: 1, pai: "W".parse().unwrap(), tsumogiri: true }).unwrap();

        // Player 2 draws and discards F
        state.update(&Event::Tsumo { actor: 2, pai: "?".parse().unwrap() }).unwrap();
        state.update(&Event::Dahai { actor: 2, pai: "F".parse().unwrap(), tsumogiri: true }).unwrap();

        // Player 3 draws and discards the target tile (chi trigger)
        state.update(&Event::Tsumo { actor: 3, pai: "?".parse().unwrap() }).unwrap();
        state.update(&Event::Dahai {
            actor: 3,
            pai: target_tile.parse().unwrap(),
            tsumogiri: false,
        }).unwrap();

        state
    }

    #[test]
    fn action_to_event_chi_low() {
        // Chi low: called tile is the lowest. E.g., 3 discards 1s, we have 2s+3s.
        // Sequence: 1s(called)-2s-3s. consumed = [2s, 3s].
        let state = setup_for_chi(
            ["2s", "3s", "5m", "6m", "7m", "1p", "2p", "3p", "4p", "5p", "6p", "7p", "8p"],
            "1s",
        );

        let cans = state.last_cans();
        assert!(cans.can_chi_low, "test setup: expected can_chi_low");

        let ev = action_to_event(38, 0, &state).unwrap();
        match ev {
            Event::Chi {
                actor,
                target,
                pai,
                consumed,
            } => {
                assert_eq!(actor, 0);
                assert_eq!(target, cans.target_actor);
                assert_eq!(pai, "1s".parse::<Tile>().unwrap());
                // consumed should be [2s, 3s]
                assert_eq!(consumed[0].deaka(), "2s".parse::<Tile>().unwrap());
                assert_eq!(consumed[1].deaka(), "3s".parse::<Tile>().unwrap());
            }
            _ => panic!("expected Chi, got {ev:?}"),
        }
    }

    #[test]
    fn action_to_event_chi_mid() {
        // Chi mid: called tile is the middle. E.g., 3 discards 5p, we have 4p+6p.
        // Sequence: 4p-5p(called)-6p. consumed = [4p, 6p].
        let state = setup_for_chi(
            ["4p", "6p", "1m", "2m", "3m", "7m", "8m", "9m", "1s", "2s", "3s", "E", "E"],
            "5p",
        );

        let cans = state.last_cans();
        assert!(cans.can_chi_mid, "test setup: expected can_chi_mid");

        let ev = action_to_event(39, 0, &state).unwrap();
        match ev {
            Event::Chi {
                actor,
                target,
                pai,
                consumed,
            } => {
                assert_eq!(actor, 0);
                assert_eq!(target, cans.target_actor);
                assert_eq!(pai, "5p".parse::<Tile>().unwrap());
                // consumed should be [4p, 6p]
                assert_eq!(consumed[0].deaka(), "4p".parse::<Tile>().unwrap());
                assert_eq!(consumed[1].deaka(), "6p".parse::<Tile>().unwrap());
            }
            _ => panic!("expected Chi, got {ev:?}"),
        }
    }

    #[test]
    fn action_to_event_chi_high() {
        // Chi high: called tile is the highest. E.g., 3 discards 9p, we have 7p+8p.
        // Sequence: 7p-8p-9p(called). consumed = [7p, 8p].
        let state = setup_for_chi(
            ["7p", "8p", "1m", "2m", "3m", "4m", "5m", "6m", "1s", "2s", "3s", "E", "E"],
            "9p",
        );

        let cans = state.last_cans();
        assert!(cans.can_chi_high, "test setup: expected can_chi_high");

        let ev = action_to_event(40, 0, &state).unwrap();
        match ev {
            Event::Chi {
                actor,
                target,
                pai,
                consumed,
            } => {
                assert_eq!(actor, 0);
                assert_eq!(target, cans.target_actor);
                assert_eq!(pai, "9p".parse::<Tile>().unwrap());
                // consumed should be [7p, 8p]
                assert_eq!(consumed[0].deaka(), "7p".parse::<Tile>().unwrap());
                assert_eq!(consumed[1].deaka(), "8p".parse::<Tile>().unwrap());
            }
            _ => panic!("expected Chi, got {ev:?}"),
        }
    }

    #[test]
    fn action_to_event_pon() {
        // Pon: we have 2 copies of E, opponent discards E.
        let mut state = PlayerState::new(0);
        state.set_record_events(true);

        let bakaze: Tile = "E".parse().unwrap();
        let dora_marker: Tile = "9s".parse().unwrap();

        let tehais = [
            [
                "E", "E", "1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "1p", "2p",
            ]
            .map(|s| s.parse::<Tile>().unwrap()),
            ["?"; 13].map(|s| s.parse::<Tile>().unwrap()),
            ["?"; 13].map(|s| s.parse::<Tile>().unwrap()),
            ["?"; 13].map(|s| s.parse::<Tile>().unwrap()),
        ];

        state
            .update(&Event::StartKyoku {
                bakaze,
                dora_marker,
                kyoku: 1,
                honba: 0,
                kyotaku: 0,
                oya: 0,
                scores: [25000; 4],
                tehais,
            })
            .unwrap();

        // Player 0 draws and discards N
        state.update(&Event::Tsumo { actor: 0, pai: "N".parse().unwrap() }).unwrap();
        state.update(&Event::Dahai { actor: 0, pai: "N".parse().unwrap(), tsumogiri: true }).unwrap();

        // Player 1 draws and discards E (we can pon)
        state.update(&Event::Tsumo { actor: 1, pai: "?".parse().unwrap() }).unwrap();
        state.update(&Event::Dahai { actor: 1, pai: "E".parse().unwrap(), tsumogiri: false }).unwrap();

        let cans = state.last_cans();
        assert!(cans.can_pon, "test setup: expected can_pon");

        let ev = action_to_event(41, 0, &state).unwrap();
        match ev {
            Event::Pon {
                actor,
                target,
                pai,
                consumed,
            } => {
                assert_eq!(actor, 0);
                assert_eq!(target, cans.target_actor);
                assert_eq!(pai, "E".parse::<Tile>().unwrap());
                // consumed should be [E, E] (both deaka'd)
                assert_eq!(consumed[0].deaka(), "E".parse::<Tile>().unwrap());
                assert_eq!(consumed[1].deaka(), "E".parse::<Tile>().unwrap());
            }
            _ => panic!("expected Pon, got {ev:?}"),
        }
    }

    #[test]
    fn action_to_event_daiminkan() {
        // Daiminkan: we have 3 copies of S, opponent discards S.
        let mut state = PlayerState::new(0);
        state.set_record_events(true);

        let bakaze: Tile = "E".parse().unwrap();
        let dora_marker: Tile = "9s".parse().unwrap();

        let tehais = [
            [
                "S", "S", "S", "1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "1p",
            ]
            .map(|s| s.parse::<Tile>().unwrap()),
            ["?"; 13].map(|s| s.parse::<Tile>().unwrap()),
            ["?"; 13].map(|s| s.parse::<Tile>().unwrap()),
            ["?"; 13].map(|s| s.parse::<Tile>().unwrap()),
        ];

        state
            .update(&Event::StartKyoku {
                bakaze,
                dora_marker,
                kyoku: 1,
                honba: 0,
                kyotaku: 0,
                oya: 0,
                scores: [25000; 4],
                tehais,
            })
            .unwrap();

        // Player 0 draws and discards N
        state.update(&Event::Tsumo { actor: 0, pai: "N".parse().unwrap() }).unwrap();
        state.update(&Event::Dahai { actor: 0, pai: "N".parse().unwrap(), tsumogiri: true }).unwrap();

        // Player 1 draws and discards S (we can daiminkan)
        state.update(&Event::Tsumo { actor: 1, pai: "?".parse().unwrap() }).unwrap();
        state.update(&Event::Dahai { actor: 1, pai: "S".parse().unwrap(), tsumogiri: false }).unwrap();

        let cans = state.last_cans();
        assert!(cans.can_daiminkan, "test setup: expected can_daiminkan");

        let ev = action_to_event(42, 0, &state).unwrap();
        match ev {
            Event::Daiminkan {
                actor,
                target,
                pai,
                consumed,
            } => {
                assert_eq!(actor, 0);
                assert_eq!(target, cans.target_actor);
                assert_eq!(pai, "S".parse::<Tile>().unwrap());
                // consumed should be [S, S, S] (3 copies from our hand)
                for &c in &consumed {
                    assert_eq!(c.deaka(), "S".parse::<Tile>().unwrap());
                }
            }
            _ => panic!("expected Daiminkan, got {ev:?}"),
        }
    }

    // ---- Tests for default_reaction ----

    #[test]
    fn default_reaction_discard() {
        let state = setup_basic_game();
        let ev = default_reaction(0, &state);
        match ev {
            Event::Dahai {
                actor,
                tsumogiri,
                ..
            } => {
                assert_eq!(actor, 0);
                assert!(tsumogiri);
            }
            _ => panic!("expected Dahai (tsumogiri), got {ev:?}"),
        }
    }

    #[test]
    fn default_reaction_post_meld_discard() {
        // After chi/pon, last_self_tsumo() is None but can_discard is true.
        // default_reaction should pick the first available tile from tehai.
        let mut state = PlayerState::new(1);

        let bakaze: Tile = "E".parse().unwrap();
        let dora_marker: Tile = "9s".parse().unwrap();
        let tehais = [
            ["?"; 13].map(|s| s.parse::<Tile>().unwrap()),
            // Player 1: has tiles including 3m, 4m for chi
            [
                "3m", "4m", "1p", "2p", "3p", "7s", "8s", "9s", "E", "E", "S", "W", "N",
            ]
            .map(|s| s.parse::<Tile>().unwrap()),
            ["?"; 13].map(|s| s.parse::<Tile>().unwrap()),
            ["?"; 13].map(|s| s.parse::<Tile>().unwrap()),
        ];

        state
            .update(&Event::StartKyoku {
                bakaze,
                dora_marker,
                kyoku: 1,
                honba: 0,
                kyotaku: 0,
                oya: 0,
                scores: [25000; 4],
                tehais,
            })
            .unwrap();

        // Oya draws and discards 2m
        state
            .update(&Event::Tsumo {
                actor: 0,
                pai: "2m".parse().unwrap(),
            })
            .unwrap();
        state
            .update(&Event::Dahai {
                actor: 0,
                pai: "2m".parse().unwrap(),
                tsumogiri: true,
            })
            .unwrap();

        // Player 1 calls chi (2m from kawa, using 3m+4m from hand)
        state
            .update(&Event::Chi {
                actor: 1,
                target: 0,
                pai: "2m".parse().unwrap(),
                consumed: [
                    "3m".parse::<Tile>().unwrap(),
                    "4m".parse::<Tile>().unwrap(),
                ],
            })
            .unwrap();

        // Now player 1 must discard but last_self_tsumo() is None
        let cans = state.last_cans();
        assert!(cans.can_discard, "should be able to discard after chi");
        assert!(
            state.last_self_tsumo().is_none(),
            "last_self_tsumo should be None after chi"
        );

        let ev = default_reaction(1, &state);
        match ev {
            Event::Dahai {
                actor,
                tsumogiri,
                ..
            } => {
                assert_eq!(actor, 1);
                assert!(!tsumogiri, "post-meld discard is not tsumogiri");
            }
            _ => panic!("expected Dahai after chi, got {ev:?}"),
        }
    }

    #[test]
    fn default_reaction_respects_kuikae() {
        // After chi of 2m using 3m+4m from hand, kuikae forbids:
        // - 2m (the called tile itself)
        // - 5m (extends the sequence: 2-3-4 + 5 = another sequence)
        // Verify default_reaction picks a tile that is NOT forbidden.
        let mut state = PlayerState::new(1);

        let bakaze: Tile = "E".parse().unwrap();
        let dora_marker: Tile = "9s".parse().unwrap();
        let tehais = [
            ["?"; 13].map(|s| s.parse::<Tile>().unwrap()),
            // Player 1: has 2m, 3m, 4m, 5m plus others.
            // After chi of 2m with 3m+4m, forbidden = {2m, 5m}.
            // Remaining tehai: 2m, 5m, 1p, 2p, 3p, 7s, 8s, 9s, E, E, S
            // First non-forbidden: 1p (tid=9).
            [
                "2m", "3m", "4m", "5m", "1p", "2p", "3p", "7s", "8s", "9s", "E", "E", "S",
            ]
            .map(|s| s.parse::<Tile>().unwrap()),
            ["?"; 13].map(|s| s.parse::<Tile>().unwrap()),
            ["?"; 13].map(|s| s.parse::<Tile>().unwrap()),
        ];

        state
            .update(&Event::StartKyoku {
                bakaze,
                dora_marker,
                kyoku: 1,
                honba: 0,
                kyotaku: 0,
                oya: 0,
                scores: [25000; 4],
                tehais,
            })
            .unwrap();

        // Oya draws and discards 2m
        state
            .update(&Event::Tsumo {
                actor: 0,
                pai: "2m".parse().unwrap(),
            })
            .unwrap();
        state
            .update(&Event::Dahai {
                actor: 0,
                pai: "2m".parse().unwrap(),
                tsumogiri: true,
            })
            .unwrap();

        // Player 1 calls chi (2m from kawa, using 3m+4m from hand)
        state
            .update(&Event::Chi {
                actor: 1,
                target: 0,
                pai: "2m".parse().unwrap(),
                consumed: [
                    "3m".parse::<Tile>().unwrap(),
                    "4m".parse::<Tile>().unwrap(),
                ],
            })
            .unwrap();

        // Verify kuikae state
        let cans = state.last_cans();
        assert!(cans.can_discard, "should be able to discard after chi");
        assert!(
            state.last_self_tsumo().is_none(),
            "last_self_tsumo should be None after chi"
        );

        let forbidden = state.forbidden_tiles();
        // 2m (tid=1) and 5m (tid=4) should be forbidden
        assert!(forbidden[1], "2m should be forbidden (kuikae)");
        assert!(forbidden[4], "5m should be forbidden (kuikae)");

        let ev = default_reaction(1, &state);
        match ev {
            Event::Dahai { actor, pai, .. } => {
                assert_eq!(actor, 1);
                let tid = pai.deaka().as_usize();
                assert!(
                    !forbidden[tid],
                    "default_reaction chose forbidden tile {} (tid={}, kuikae)",
                    pai, tid,
                );
            }
            _ => panic!("expected Dahai after chi, got {ev:?}"),
        }
    }

    // ---- Tests for simulate_action_rollout ----

    #[test]
    fn action_rollout_discard_completes() {
        let state = setup_basic_game();
        let config = ParticleConfig::new(5);
        let mut rng = ChaCha12Rng::seed_from_u64(42);

        let (particles, _attempts) = generate_particles(&state, &config, &mut rng).unwrap();
        assert!(!particles.is_empty());

        // Discard 1m (action 0)
        for (i, particle) in particles.iter().enumerate() {
            let result = simulate_action_rollout(&state, particle, 0);
            assert!(
                result.is_ok(),
                "action rollout particle {i} failed: {:?}",
                result.err()
            );

            let result = result.unwrap();
            let delta_sum: i32 = result.deltas.iter().sum();
            assert!(
                delta_sum.abs() <= 1000,
                "action rollout particle {i}: delta sum {delta_sum} is too large"
            );
        }
    }

    #[test]
    fn action_rollout_tsumogiri_matches_plain_rollout() {
        // When the action is the tsumo tile (tsumogiri), the action rollout
        // should produce the same result as the plain tsumogiri rollout,
        // since the injected action IS the tsumogiri.
        let state = setup_basic_game();
        let config = ParticleConfig::new(3);
        let mut rng = ChaCha12Rng::seed_from_u64(42);

        let (particles, _attempts) = generate_particles(&state, &config, &mut rng).unwrap();
        assert!(!particles.is_empty());

        // Action 13 = 5p = the tsumo tile (tsumogiri)
        for particle in &particles {
            let plain = simulate_particle(&state, particle).unwrap();
            let action_result = simulate_action_rollout(&state, particle, 13).unwrap();

            // Both should produce identical results since the first action is the same
            assert_eq!(
                plain.deltas, action_result.deltas,
                "tsumogiri action rollout should match plain rollout"
            );
        }
    }

    #[test]
    fn different_action_rollouts_complete() {
        // Verify that rollouts with different discard actions both complete.
        let state = setup_basic_game();
        let config = ParticleConfig::new(20);
        let mut rng = ChaCha12Rng::seed_from_u64(42);

        let (particles, _attempts) = generate_particles(&state, &config, &mut rng).unwrap();
        assert!(!particles.is_empty());

        // Discard 1m (action 0) vs 9m (action 8) should both complete
        let mut results_0 = Vec::new();
        let mut results_8 = Vec::new();

        for particle in &particles {
            if let Ok(r) = simulate_action_rollout(&state, particle, 0) {
                results_0.push(r);
            }
            if let Ok(r) = simulate_action_rollout(&state, particle, 8) {
                results_8.push(r);
            }
        }

        assert!(!results_0.is_empty(), "action 0 rollouts should succeed");
        assert!(!results_8.is_empty(), "action 8 rollouts should succeed");
    }

    #[test]
    fn action_rollout_non_oya_completes() {
        let state = setup_non_oya_game();
        let config = ParticleConfig::new(5);
        let mut rng = ChaCha12Rng::seed_from_u64(42);

        let (particles, _attempts) = generate_particles(&state, &config, &mut rng).unwrap();
        assert!(!particles.is_empty());

        // Player 2 discards S (tile 28, the tsumo tile) - tsumogiri
        for (i, particle) in particles.iter().enumerate() {
            let result = simulate_action_rollout(&state, particle, 28);
            assert!(
                result.is_ok(),
                "non-oya action rollout particle {i} failed: {:?}",
                result.err()
            );
        }
    }

    fn setup_south_round_game() -> PlayerState {
        let mut state = PlayerState::new(1); // player 1
        state.set_record_events(true);

        let bakaze: Tile = "S".parse().unwrap(); // South round
        let dora_marker: Tile = "3p".parse().unwrap();

        let tehais = [
            ["?"; 13].map(|s| s.parse::<Tile>().unwrap()),
            [
                "2m", "3m", "4m", "6p", "7p", "8p", "1s", "2s", "3s", "5s", "6s", "7s", "E",
            ]
            .map(|s| s.parse::<Tile>().unwrap()),
            ["?"; 13].map(|s| s.parse::<Tile>().unwrap()),
            ["?"; 13].map(|s| s.parse::<Tile>().unwrap()),
        ];

        // South 3: kyoku=3 in mjai (1-based), oya=2
        let start_event = Event::StartKyoku {
            bakaze,
            dora_marker,
            kyoku: 3,
            honba: 1,
            kyotaku: 0,
            oya: 2,
            scores: [35000, 22000, 28000, 15000],
            tehais,
        };
        state.update(&start_event).unwrap();

        // Tsumo for oya (seat 2) then our seat (1) gets tsumo
        // First: seat 2 (oya) draws and discards
        let tsumo0 = Event::Tsumo {
            actor: 2,
            pai: "9m".parse().unwrap(),
        };
        state.update(&tsumo0).unwrap();
        let dahai0 = Event::Dahai {
            actor: 2,
            pai: "9m".parse().unwrap(),
            tsumogiri: true,
        };
        state.update(&dahai0).unwrap();

        // Seat 3 draws and discards
        let tsumo1 = Event::Tsumo {
            actor: 3,
            pai: "N".parse().unwrap(),
        };
        state.update(&tsumo1).unwrap();
        let dahai1 = Event::Dahai {
            actor: 3,
            pai: "N".parse().unwrap(),
            tsumogiri: true,
        };
        state.update(&dahai1).unwrap();

        // Seat 0 draws and discards
        let tsumo2 = Event::Tsumo {
            actor: 0,
            pai: "F".parse().unwrap(),
        };
        state.update(&tsumo2).unwrap();
        let dahai2 = Event::Dahai {
            actor: 0,
            pai: "F".parse().unwrap(),
            tsumogiri: true,
        };
        state.update(&dahai2).unwrap();

        // Our tsumo (seat 1)
        let tsumo3 = Event::Tsumo {
            actor: 1,
            pai: "9s".parse().unwrap(),
        };
        state.update(&tsumo3).unwrap();

        state
    }

    #[test]
    fn build_board_south_round_kyoku() {
        let state = setup_south_round_game();
        let config = ParticleConfig::new(1);
        let mut rng = ChaCha12Rng::seed_from_u64(42);

        let (particles, _attempts) = generate_particles(&state, &config, &mut rng).unwrap();
        assert!(!particles.is_empty());

        let bs = build_midgame_board_state(&state, &particles[0]).unwrap();

        // South 3: bakaze=S, within-wind kyoku=2 (0-indexed), absolute kyoku=6
        // (S=1)*4 + 2 = 6
        assert_eq!(bs.board.kyoku, 6, "South 3 should have absolute kyoku 6");
        assert_eq!(bs.board.honba, 1);
    }

    #[test]
    fn simulate_south_round_completes() {
        let state = setup_south_round_game();
        let config = ParticleConfig::new(3);
        let mut rng = ChaCha12Rng::seed_from_u64(42);

        let (particles, _attempts) = generate_particles(&state, &config, &mut rng).unwrap();
        assert!(!particles.is_empty());

        for particle in &particles {
            let result = simulate_particle(&state, particle);
            assert!(result.is_ok(), "South round rollout failed: {:?}", result.err());

            let result = result.unwrap();
            let delta_sum: i32 = result.deltas.iter().sum();
            assert!(
                delta_sum.abs() <= 1000,
                "South round: delta sum {delta_sum} is too large"
            );
        }
    }

    #[test]
    fn action_rollout_pass_completes() {
        // Pass (action 45) doesn't match can_discard decisions, but will
        // match at a call opportunity (can_pass && !can_discard), where it
        // injects Event::None (declining the call). This may diverge from
        // plain tsumogiri which takes agari at call opportunities.
        let state = setup_basic_game();
        let config = ParticleConfig::new(3);
        let mut rng = ChaCha12Rng::seed_from_u64(42);

        let (particles, _attempts) = generate_particles(&state, &config, &mut rng).unwrap();
        assert!(!particles.is_empty());

        for particle in &particles {
            let result = simulate_action_rollout(&state, particle, 45);
            assert!(
                result.is_ok(),
                "pass action rollout failed: {:?}",
                result.err()
            );

            let result = result.unwrap();
            let delta_sum: i32 = result.deltas.iter().sum();
            assert!(
                delta_sum.abs() <= 1000,
                "pass rollout: delta sum {delta_sum} is too large"
            );
        }
    }

    // ---- Tests for mid-game state reconstruction ----

    #[test]
    fn event_history_recorded() {
        // Verify that event_history is populated during normal play.
        let state = setup_basic_game();
        let history = state.event_history();

        // Should have: StartKyoku + Tsumo = 2 events
        assert_eq!(
            history.len(),
            2,
            "expected 2 events in history, got {}",
            history.len()
        );
        assert!(
            matches!(history[0], Event::StartKyoku { .. }),
            "first event should be StartKyoku"
        );
        assert!(
            matches!(history[1], Event::Tsumo { .. }),
            "second event should be Tsumo"
        );
    }

    #[test]
    fn event_history_multi_turn() {
        // Verify that a multi-turn game records all events.
        let state = setup_non_oya_game();
        let history = state.event_history();

        // StartKyoku + Tsumo(0) + Dahai(0) + Tsumo(1) + Dahai(1) + Tsumo(2) = 6
        assert_eq!(
            history.len(),
            6,
            "expected 6 events in history, got {}",
            history.len()
        );
    }

    #[test]
    fn midgame_board_state_basic() {
        // Test that build_midgame_board_state produces a valid BoardState
        // that can run a rollout to completion.
        let state = setup_basic_game();
        let config = ParticleConfig::new(5);
        let mut rng = ChaCha12Rng::seed_from_u64(42);

        let (particles, _attempts) = generate_particles(&state, &config, &mut rng).unwrap();
        assert!(!particles.is_empty());

        for (i, particle) in particles.iter().enumerate() {
            let board_state = build_midgame_board_state(&state, particle);
            assert!(
                board_state.is_ok(),
                "midgame build failed for particle {i}: {:?}",
                board_state.err()
            );

            let bs = board_state.unwrap();
            // Verify midgame_start is set
            assert!(bs.midgame_start, "midgame_start should be true");
            // Verify tiles_left matches the source state
            assert_eq!(
                bs.tiles_left,
                state.tiles_left(),
                "tiles_left should match source state"
            );
            // Verify that a player can_act (oya has 14 tiles after tsumo)
            assert!(
                bs.player_states.iter().any(|ps| ps.last_cans().can_act()),
                "at least one player should be able to act"
            );
        }
    }

    #[test]
    fn midgame_board_state_preserves_scores() {
        // Verify that scores survive the midgame reconstruction roundtrip.
        let state = setup_south_round_game();
        let config = ParticleConfig::new(1);
        let mut rng = ChaCha12Rng::seed_from_u64(42);

        let (particles, _attempts) = generate_particles(&state, &config, &mut rng).unwrap();
        let bs = build_midgame_board_state(&state, &particles[0]).unwrap();

        // Unrotate scores from the PlayerState's relative to absolute
        let player_id = state.player_id() as usize;
        let rel_scores = state.scores();
        let mut expected_abs = [0_i32; 4];
        for i in 0..4 {
            expected_abs[(player_id + i) % 4] = rel_scores[i];
        }
        assert_eq!(
            bs.board.scores, expected_abs,
            "Board scores should match source state"
        );
    }

    /// Set up a game with a chi meld (player 1 chis from player 0).
    fn setup_game_with_chi() -> PlayerState {
        let mut state = PlayerState::new(1);
        state.set_record_events(true);

        let bakaze: Tile = "E".parse().unwrap();
        let dora_marker: Tile = "9s".parse().unwrap();
        let tehais = [
            ["?"; 13].map(|s| s.parse::<Tile>().unwrap()),
            [
                "3m", "4m", "1p", "2p", "3p", "7s", "8s", "9s", "E", "E", "S", "W", "N",
            ]
            .map(|s| s.parse::<Tile>().unwrap()),
            ["?"; 13].map(|s| s.parse::<Tile>().unwrap()),
            ["?"; 13].map(|s| s.parse::<Tile>().unwrap()),
        ];

        state
            .update(&Event::StartKyoku {
                bakaze,
                dora_marker,
                kyoku: 1,
                honba: 0,
                kyotaku: 0,
                oya: 0,
                scores: [25000; 4],
                tehais,
            })
            .unwrap();

        // Oya draws and discards 2m
        state
            .update(&Event::Tsumo {
                actor: 0,
                pai: "2m".parse().unwrap(),
            })
            .unwrap();
        state
            .update(&Event::Dahai {
                actor: 0,
                pai: "2m".parse().unwrap(),
                tsumogiri: true,
            })
            .unwrap();

        // Player 1 calls chi (2m from kawa, using 3m+4m from hand)
        state
            .update(&Event::Chi {
                actor: 1,
                target: 0,
                pai: "2m".parse().unwrap(),
                consumed: [
                    "3m".parse::<Tile>().unwrap(),
                    "4m".parse::<Tile>().unwrap(),
                ],
            })
            .unwrap();

        // Player 1 discards N after chi
        state
            .update(&Event::Dahai {
                actor: 1,
                pai: "N".parse().unwrap(),
                tsumogiri: false,
            })
            .unwrap();

        // Continue: seat 2, 3, 0 each draw and discard
        for actor in [2_u8, 3, 0] {
            state
                .update(&Event::Tsumo {
                    actor,
                    pai: if actor == 0 {
                        "C".parse().unwrap()
                    } else {
                        "?".parse().unwrap()
                    },
                })
                .unwrap();
            state
                .update(&Event::Dahai {
                    actor,
                    pai: if actor == 0 {
                        "C".parse().unwrap()
                    } else {
                        "P".parse().unwrap()
                    },
                    tsumogiri: true,
                })
                .unwrap();
        }

        // Player 1 draws
        state
            .update(&Event::Tsumo {
                actor: 1,
                pai: "F".parse().unwrap(),
            })
            .unwrap();

        state
    }

    #[test]
    fn midgame_with_chi_completes() {
        // Test that mid-game reconstruction works correctly when the
        // observed player has an open chi meld.
        let state = setup_game_with_chi();
        let config = ParticleConfig::new(10);
        let mut rng = ChaCha12Rng::seed_from_u64(42);

        let (particles, _attempts) = generate_particles(&state, &config, &mut rng).unwrap();
        assert!(!particles.is_empty());

        for (i, particle) in particles.iter().enumerate() {
            let result = simulate_particle(&state, particle);
            assert!(
                result.is_ok(),
                "chi-game particle {i} simulation failed: {:?}",
                result.err()
            );

            let result = result.unwrap();
            let delta_sum: i32 = result.deltas.iter().sum();
            assert!(
                delta_sum.abs() <= 1000,
                "chi-game particle {i}: delta sum {delta_sum} is too large"
            );
        }
    }

    #[test]
    fn midgame_with_chi_action_rollout() {
        // Test action rollout with a chi meld present.
        let state = setup_game_with_chi();
        let config = ParticleConfig::new(5);
        let mut rng = ChaCha12Rng::seed_from_u64(42);

        let (particles, _attempts) = generate_particles(&state, &config, &mut rng).unwrap();

        // Discard F (the tsumo tile) via tsumogiri
        let f_tile: Tile = "F".parse().unwrap();
        let action = f_tile.as_usize();

        for (i, particle) in particles.iter().enumerate() {
            let result = simulate_action_rollout(&state, particle, action);
            assert!(
                result.is_ok(),
                "chi action rollout particle {i} failed: {:?}",
                result.err()
            );
        }
    }

    /// Set up a game several turns deep to test mid-game reconstruction
    /// with substantial event history.
    fn setup_deep_game() -> PlayerState {
        let mut state = PlayerState::new(0);
        state.set_record_events(true);

        let bakaze: Tile = "E".parse().unwrap();
        let dora_marker: Tile = "1m".parse().unwrap();
        let tehais = [
            [
                "1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "1p", "2p", "3p", "4p",
            ]
            .map(|s| s.parse::<Tile>().unwrap()),
            ["?"; 13].map(|s| s.parse::<Tile>().unwrap()),
            ["?"; 13].map(|s| s.parse::<Tile>().unwrap()),
            ["?"; 13].map(|s| s.parse::<Tile>().unwrap()),
        ];

        state
            .update(&Event::StartKyoku {
                bakaze,
                dora_marker,
                kyoku: 1,
                honba: 0,
                kyotaku: 0,
                oya: 0,
                scores: [25000; 4],
                tehais,
            })
            .unwrap();

        // Play 3 full turn cycles (12 tsumo+dahai pairs)
        let discards = [
            // Our discards (seat 0)
            ["4p", "3p", "2p"],
            // Opponent discards are visible but hands are "?"
            ["E", "S", "W"],
            ["N", "P", "F"],
            ["C", "E", "S"],
        ];

        for turn in 0..3 {
            for seat in 0..4_u8 {
                let tsumo_tile = if seat == 0 {
                    // We draw a tile we'll discard
                    discards[0][turn].parse().unwrap()
                } else {
                    "?".parse().unwrap()
                };
                state
                    .update(&Event::Tsumo {
                        actor: seat,
                        pai: tsumo_tile,
                    })
                    .unwrap();

                let discard_tile: Tile = discards[seat as usize][turn].parse().unwrap();
                state
                    .update(&Event::Dahai {
                        actor: seat,
                        pai: discard_tile,
                        tsumogiri: true,
                    })
                    .unwrap();
            }
        }

        // One more tsumo for seat 0
        state
            .update(&Event::Tsumo {
                actor: 0,
                pai: "1p".parse().unwrap(),
            })
            .unwrap();

        state
    }

    #[test]
    fn midgame_deep_game_completes() {
        // Test that mid-game reconstruction works with many turns of history.
        let state = setup_deep_game();
        let config = ParticleConfig::new(10);
        let mut rng = ChaCha12Rng::seed_from_u64(42);

        let history = state.event_history();
        // StartKyoku + 3 turns * 4 seats * 2 events + 1 tsumo = 26
        assert_eq!(
            history.len(),
            26,
            "expected 26 events in deep game history"
        );

        let (particles, _attempts) = generate_particles(&state, &config, &mut rng).unwrap();
        assert!(!particles.is_empty());

        for (i, particle) in particles.iter().enumerate() {
            let result = simulate_particle(&state, particle);
            assert!(
                result.is_ok(),
                "deep game particle {i} failed: {:?}",
                result.err()
            );
        }
    }

    #[test]
    fn midgame_deep_game_action_rollout() {
        // Test action rollout in a deep game scenario.
        let state = setup_deep_game();
        let config = ParticleConfig::new(5);
        let mut rng = ChaCha12Rng::seed_from_u64(42);

        let (particles, _attempts) = generate_particles(&state, &config, &mut rng).unwrap();

        // Discard 1m (action 0) — we still have it in hand
        for (i, particle) in particles.iter().enumerate() {
            let result = simulate_action_rollout(&state, particle, 0);
            assert!(
                result.is_ok(),
                "deep game action rollout particle {i} failed: {:?}",
                result.err()
            );
        }
    }

    #[test]
    fn midgame_tiles_left_consistency() {
        // Verify that the midgame BoardState's tiles_left matches
        // the yama size in the board.
        let state = setup_deep_game();
        let config = ParticleConfig::new(3);
        let mut rng = ChaCha12Rng::seed_from_u64(42);

        let (particles, _attempts) = generate_particles(&state, &config, &mut rng).unwrap();
        for particle in &particles {
            let bs = build_midgame_board_state(&state, particle).unwrap();
            assert_eq!(
                bs.tiles_left as usize,
                bs.board.yama.len(),
                "tiles_left ({}) should match yama length ({})",
                bs.tiles_left,
                bs.board.yama.len(),
            );
        }
    }

    #[test]
    fn midgame_replayed_player_state_has_correct_cans() {
        // After replay, the player who just drew should have can_discard = true.
        let state = setup_basic_game();
        let config = ParticleConfig::new(1);
        let mut rng = ChaCha12Rng::seed_from_u64(42);

        let (particles, _attempts) = generate_particles(&state, &config, &mut rng).unwrap();
        let bs = build_midgame_board_state(&state, &particles[0]).unwrap();

        // Player 0 (oya) just drew, should be able to discard
        let player_cans = bs.player_states[0].last_cans();
        assert!(
            player_cans.can_discard,
            "player 0 should be able to discard after midgame reconstruction"
        );
    }

    #[test]
    fn midgame_deterministic_with_same_particle() {
        // Two rollouts with the same particle should produce identical results
        // (both use build_midgame_board_state with the same inputs).
        let state = setup_basic_game();
        let config = ParticleConfig::new(1);
        let mut rng = ChaCha12Rng::seed_from_u64(42);

        let (particles, _attempts) = generate_particles(&state, &config, &mut rng).unwrap();
        let particle = &particles[0];

        let result1 = simulate_particle(&state, particle).unwrap();
        let result2 = simulate_particle(&state, particle).unwrap();

        assert_eq!(
            result1.deltas, result2.deltas,
            "same particle should produce same result"
        );
        assert_eq!(result1.scores, result2.scores);
    }

    // ---- Comprehensive test scenarios (Task #20) ----

    /// 1. Riichi during replay: riichi declared+accepted, verify midgame
    ///    preserves riichi state and score deduction.
    #[test]
    fn midgame_riichi_reconstruction() {
        let mut state = PlayerState::new(0);
        state.set_record_events(true);

        let bakaze: Tile = "E".parse().unwrap();
        let dora_marker: Tile = "9s".parse().unwrap();
        // Tenpai hand for player 0: 1-9m, 1-3p, E (waiting 4p)
        let tehais = [
            [
                "1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "1p", "2p", "3p", "E",
            ]
            .map(|s| s.parse::<Tile>().unwrap()),
            ["?"; 13].map(|s| s.parse::<Tile>().unwrap()),
            ["?"; 13].map(|s| s.parse::<Tile>().unwrap()),
            ["?"; 13].map(|s| s.parse::<Tile>().unwrap()),
        ];

        state
            .update(&Event::StartKyoku {
                bakaze,
                dora_marker,
                kyoku: 1,
                honba: 0,
                kyotaku: 0,
                oya: 0,
                scores: [25000; 4],
                tehais,
            })
            .unwrap();

        // Tsumo E (now have 2 E, tenpai on 4p or E depending on hand shape)
        state
            .update(&Event::Tsumo {
                actor: 0,
                pai: "E".parse().unwrap(),
            })
            .unwrap();

        // Declare riichi
        state.update(&Event::Reach { actor: 0 }).unwrap();

        // Discard E (riichi discard)
        state
            .update(&Event::Dahai {
                actor: 0,
                pai: "E".parse().unwrap(),
                tsumogiri: false,
            })
            .unwrap();

        // Riichi accepted
        state.update(&Event::ReachAccepted { actor: 0 }).unwrap();

        // Other players play a round
        for actor in [1_u8, 2, 3] {
            state
                .update(&Event::Tsumo {
                    actor,
                    pai: "?".parse().unwrap(),
                })
                .unwrap();
            state
                .update(&Event::Dahai {
                    actor,
                    pai: "N".parse().unwrap(),
                    tsumogiri: true,
                })
                .unwrap();
        }

        // Our next tsumo
        state
            .update(&Event::Tsumo {
                actor: 0,
                pai: "W".parse().unwrap(),
            })
            .unwrap();

        // Verify riichi state
        assert!(state.self_riichi_accepted());
        assert_eq!(state.scores()[0], 24000); // 25000 - 1000

        let config = ParticleConfig::new(5);
        let mut rng = ChaCha12Rng::seed_from_u64(42);
        let (particles, _attempts) = generate_particles(&state, &config, &mut rng).unwrap();

        for (i, particle) in particles.iter().enumerate() {
            let bs = build_midgame_board_state(&state, particle);
            assert!(
                bs.is_ok(),
                "riichi midgame build failed for particle {i}: {:?}",
                bs.err()
            );
            let bs = bs.unwrap();

            // Verify riichi is preserved
            assert!(
                bs.player_states[0].self_riichi_accepted(),
                "riichi should be accepted after midgame reconstruction"
            );

            // Verify scores reflect riichi deduction
            assert_eq!(
                bs.board.scores[0], 24000,
                "riichi deduction should be preserved"
            );

            // Run rollout to completion
            let result = run_rollout(bs, [24000, 25000, 25000, 25000], None, None, None);
            assert!(
                result.is_ok(),
                "riichi rollout failed for particle {i}: {:?}",
                result.err()
            );
        }
    }

    /// 2. Pon meld reconstruction.
    #[test]
    fn midgame_pon_reconstruction() {
        let mut state = PlayerState::new(0);
        state.set_record_events(true);

        let bakaze: Tile = "E".parse().unwrap();
        let dora_marker: Tile = "9s".parse().unwrap();
        let tehais = [
            [
                "1m", "1m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "1p", "2p", "3p", "4p",
            ]
            .map(|s| s.parse::<Tile>().unwrap()),
            ["?"; 13].map(|s| s.parse::<Tile>().unwrap()),
            ["?"; 13].map(|s| s.parse::<Tile>().unwrap()),
            ["?"; 13].map(|s| s.parse::<Tile>().unwrap()),
        ];

        state
            .update(&Event::StartKyoku {
                bakaze,
                dora_marker,
                kyoku: 1,
                honba: 0,
                kyotaku: 0,
                oya: 0,
                scores: [25000; 4],
                tehais,
            })
            .unwrap();

        // Oya tsumo and discard
        state
            .update(&Event::Tsumo {
                actor: 0,
                pai: "5p".parse().unwrap(),
            })
            .unwrap();
        state
            .update(&Event::Dahai {
                actor: 0,
                pai: "5p".parse().unwrap(),
                tsumogiri: true,
            })
            .unwrap();

        // Player 1 discards 1m
        state
            .update(&Event::Tsumo {
                actor: 1,
                pai: "?".parse().unwrap(),
            })
            .unwrap();
        state
            .update(&Event::Dahai {
                actor: 1,
                pai: "1m".parse().unwrap(),
                tsumogiri: false,
            })
            .unwrap();

        // Player 0 pons 1m from player 1
        state
            .update(&Event::Pon {
                actor: 0,
                target: 1,
                pai: "1m".parse().unwrap(),
                consumed: [
                    "1m".parse::<Tile>().unwrap(),
                    "1m".parse::<Tile>().unwrap(),
                ],
            })
            .unwrap();

        // Discard after pon
        state
            .update(&Event::Dahai {
                actor: 0,
                pai: "4p".parse().unwrap(),
                tsumogiri: false,
            })
            .unwrap();

        // More turns
        for actor in [2_u8, 3] {
            state
                .update(&Event::Tsumo {
                    actor,
                    pai: "?".parse().unwrap(),
                })
                .unwrap();
            state
                .update(&Event::Dahai {
                    actor,
                    pai: "S".parse().unwrap(),
                    tsumogiri: true,
                })
                .unwrap();
        }

        // Our next tsumo
        state
            .update(&Event::Tsumo {
                actor: 0,
                pai: "6p".parse().unwrap(),
            })
            .unwrap();

        let config = ParticleConfig::new(5);
        let mut rng = ChaCha12Rng::seed_from_u64(42);
        let (particles, _attempts) = generate_particles(&state, &config, &mut rng).unwrap();

        for (i, particle) in particles.iter().enumerate() {
            let result = simulate_particle(&state, particle);
            assert!(
                result.is_ok(),
                "pon midgame particle {i} failed: {:?}",
                result.err()
            );
        }
    }

    /// 3. Ankan during mid-game reconstruction.
    #[test]
    fn midgame_ankan_reconstruction() {
        let mut state = PlayerState::new(0);
        state.set_record_events(true);

        let bakaze: Tile = "E".parse().unwrap();
        let dora_marker: Tile = "9s".parse().unwrap();
        // Player 0 has three 1m, will draw fourth for ankan
        let tehais = [
            [
                "1m", "1m", "1m", "3m", "4m", "5m", "6m", "7m", "8m", "1p", "2p", "3p", "4p",
            ]
            .map(|s| s.parse::<Tile>().unwrap()),
            ["?"; 13].map(|s| s.parse::<Tile>().unwrap()),
            ["?"; 13].map(|s| s.parse::<Tile>().unwrap()),
            ["?"; 13].map(|s| s.parse::<Tile>().unwrap()),
        ];

        state
            .update(&Event::StartKyoku {
                bakaze,
                dora_marker,
                kyoku: 1,
                honba: 0,
                kyotaku: 0,
                oya: 0,
                scores: [25000; 4],
                tehais,
            })
            .unwrap();

        // Tsumo 1m -> ankan
        state
            .update(&Event::Tsumo {
                actor: 0,
                pai: "1m".parse().unwrap(),
            })
            .unwrap();

        state
            .update(&Event::Ankan {
                actor: 0,
                consumed: [
                    "1m".parse::<Tile>().unwrap(),
                    "1m".parse::<Tile>().unwrap(),
                    "1m".parse::<Tile>().unwrap(),
                    "1m".parse::<Tile>().unwrap(),
                ],
            })
            .unwrap();

        // New dora after ankan
        state
            .update(&Event::Dora {
                dora_marker: "2p".parse().unwrap(),
            })
            .unwrap();

        // Rinshan tsumo
        state
            .update(&Event::Tsumo {
                actor: 0,
                pai: "9m".parse().unwrap(),
            })
            .unwrap();

        // Discard after rinshan
        state
            .update(&Event::Dahai {
                actor: 0,
                pai: "9m".parse().unwrap(),
                tsumogiri: true,
            })
            .unwrap();

        // Other players
        for actor in [1_u8, 2, 3] {
            state
                .update(&Event::Tsumo {
                    actor,
                    pai: "?".parse().unwrap(),
                })
                .unwrap();
            state
                .update(&Event::Dahai {
                    actor,
                    pai: "N".parse().unwrap(),
                    tsumogiri: true,
                })
                .unwrap();
        }

        // Our next tsumo
        state
            .update(&Event::Tsumo {
                actor: 0,
                pai: "5p".parse().unwrap(),
            })
            .unwrap();

        let config = ParticleConfig::new(5);
        let mut rng = ChaCha12Rng::seed_from_u64(42);
        let (particles, _attempts) = generate_particles(&state, &config, &mut rng).unwrap();

        for (i, particle) in particles.iter().enumerate() {
            let result = simulate_particle(&state, particle);
            assert!(
                result.is_ok(),
                "ankan midgame particle {i} failed: {:?}",
                result.err()
            );
        }
    }

    /// 4. Multiple melds by one player.
    #[test]
    fn midgame_multiple_melds() {
        let mut state = PlayerState::new(1);
        state.set_record_events(true);

        let bakaze: Tile = "E".parse().unwrap();
        let dora_marker: Tile = "9s".parse().unwrap();
        let tehais = [
            ["?"; 13].map(|s| s.parse::<Tile>().unwrap()),
            // Player 1 has pairs/sequences for chi and pon
            [
                "2m", "3m", "5p", "5p", "7s", "8s", "E", "E", "S", "S", "W", "N", "N",
            ]
            .map(|s| s.parse::<Tile>().unwrap()),
            ["?"; 13].map(|s| s.parse::<Tile>().unwrap()),
            ["?"; 13].map(|s| s.parse::<Tile>().unwrap()),
        ];

        state
            .update(&Event::StartKyoku {
                bakaze,
                dora_marker,
                kyoku: 1,
                honba: 0,
                kyotaku: 0,
                oya: 0,
                scores: [25000; 4],
                tehais,
            })
            .unwrap();

        // Oya draws and discards 1m
        state
            .update(&Event::Tsumo {
                actor: 0,
                pai: "1m".parse().unwrap(),
            })
            .unwrap();
        state
            .update(&Event::Dahai {
                actor: 0,
                pai: "1m".parse().unwrap(),
                tsumogiri: true,
            })
            .unwrap();

        // Player 1 chis 1m with 2m+3m
        state
            .update(&Event::Chi {
                actor: 1,
                target: 0,
                pai: "1m".parse().unwrap(),
                consumed: [
                    "2m".parse::<Tile>().unwrap(),
                    "3m".parse::<Tile>().unwrap(),
                ],
            })
            .unwrap();

        // Player 1 discards W
        state
            .update(&Event::Dahai {
                actor: 1,
                pai: "W".parse().unwrap(),
                tsumogiri: false,
            })
            .unwrap();

        // Seats 2, 3, 0 play
        for actor in [2_u8, 3, 0] {
            state
                .update(&Event::Tsumo {
                    actor,
                    pai: if actor == 0 {
                        "5p".parse().unwrap()
                    } else {
                        "?".parse().unwrap()
                    },
                })
                .unwrap();
            state
                .update(&Event::Dahai {
                    actor,
                    pai: if actor == 0 {
                        "5p".parse().unwrap()
                    } else {
                        "F".parse().unwrap()
                    },
                    tsumogiri: true,
                })
                .unwrap();
        }

        // Player 1 draws and discards
        state
            .update(&Event::Tsumo {
                actor: 1,
                pai: "C".parse().unwrap(),
            })
            .unwrap();
        state
            .update(&Event::Dahai {
                actor: 1,
                pai: "C".parse().unwrap(),
                tsumogiri: true,
            })
            .unwrap();

        // Seat 2 discards N -> player 1 can pon
        state
            .update(&Event::Tsumo {
                actor: 2,
                pai: "?".parse().unwrap(),
            })
            .unwrap();
        state
            .update(&Event::Dahai {
                actor: 2,
                pai: "N".parse().unwrap(),
                tsumogiri: false,
            })
            .unwrap();

        // Player 1 pons N
        state
            .update(&Event::Pon {
                actor: 1,
                target: 2,
                pai: "N".parse().unwrap(),
                consumed: [
                    "N".parse::<Tile>().unwrap(),
                    "N".parse::<Tile>().unwrap(),
                ],
            })
            .unwrap();

        // Player 1 discards S
        state
            .update(&Event::Dahai {
                actor: 1,
                pai: "S".parse().unwrap(),
                tsumogiri: false,
            })
            .unwrap();

        // Continue play until player 1 gets tsumo
        for actor in [3_u8, 0] {
            state
                .update(&Event::Tsumo {
                    actor,
                    pai: if actor == 0 {
                        "P".parse().unwrap()
                    } else {
                        "?".parse().unwrap()
                    },
                })
                .unwrap();
            state
                .update(&Event::Dahai {
                    actor,
                    pai: if actor == 0 {
                        "P".parse().unwrap()
                    } else {
                        "C".parse().unwrap()
                    },
                    tsumogiri: true,
                })
                .unwrap();
        }

        // Player 1 draws
        state
            .update(&Event::Tsumo {
                actor: 1,
                pai: "9s".parse().unwrap(),
            })
            .unwrap();

        // Player 1 now has 2 melds (chi + pon), 8 closed tiles
        let config = ParticleConfig::new(5);
        let mut rng = ChaCha12Rng::seed_from_u64(42);
        let (particles, _attempts) = generate_particles(&state, &config, &mut rng).unwrap();

        for (i, particle) in particles.iter().enumerate() {
            let result = simulate_particle(&state, particle);
            assert!(
                result.is_ok(),
                "multi-meld particle {i} failed: {:?}",
                result.err()
            );
        }
    }

    /// 5. Near-exhaustion: tiles_left is small after many turns.
    ///
    /// We use the deep_game setup (3 full rounds) which consumes 13 draws,
    /// leaving tiles_left = 57. We then add more rounds using varied
    /// tile choices to get tiles_left much lower.
    #[test]
    fn midgame_near_exhaustion() {
        let mut state = PlayerState::new(0);
        state.set_record_events(true);

        let bakaze: Tile = "E".parse().unwrap();
        let dora_marker: Tile = "1m".parse().unwrap();
        let tehais = [
            [
                "1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "1p", "2p", "3p", "4p",
            ]
            .map(|s| s.parse::<Tile>().unwrap()),
            ["?"; 13].map(|s| s.parse::<Tile>().unwrap()),
            ["?"; 13].map(|s| s.parse::<Tile>().unwrap()),
            ["?"; 13].map(|s| s.parse::<Tile>().unwrap()),
        ];

        state
            .update(&Event::StartKyoku {
                bakaze,
                dora_marker,
                kyoku: 1,
                honba: 0,
                kyotaku: 0,
                oya: 0,
                scores: [25000; 4],
                tehais,
            })
            .unwrap();

        // Carefully chosen discards. Each tile can appear at most 4 times
        // across all visible events (hand + discards + dora).
        // Our hand has: 1-9m, 1-4p (13 tiles) + dora indicator 1m (seen 2x).
        // We have 34 tile types, 7 honor tiles each appearing 0 in hand.
        // Opponents can safely discard honors: E(4), S(4), W(4), N(4), P(4), F(4), C(4) = 28 uses
        // But each has 4 copies max, and we see 1m as dora. With 3 opponents
        // doing 4 turns each, 12 discards total per "cycle" of 4 turns.
        //
        // Use 3 cycles of our_discard + 3 opponent discards (12 turns total, 12 draws).
        // Then use different tiles per cycle.

        // Cycle 1: our=4p,3p,2p,1p; opp1=E,S,W; opp2=N,P,F; opp3=C,E,S
        // Cycle 2: our=9m,8m,7m,6m; opp1=W,N,P; opp2=F,C,E; opp3=S,W,N
        // Cycle 3: our=5m,4m,3m,2m; opp1=P,F,C; opp2=E,S,W; opp3=N,P,F
        // Each honor tile (E,S,W,N,P,F,C) has 4 copies. We see 1m as dora
        // indicator. Our hand has 1-9m,1-4p = 13 tiles. No honors in hand.
        // So each honor can be discarded at most 4 times total.
        // 7 honor types * 4 copies = 28 opponent discards possible.
        // With 3 opponents * 12 rounds = 36 needed, we supplement with
        // numbered tiles not in our hand (5p-9p, 1s-9s = 14 types * 4 = 56).
        let script: &[(u8, &str, &str)] = &[
            // Round 1
            (0, "4p", "4p"), (1, "?", "E"), (2, "?", "N"), (3, "?", "C"),
            (0, "3p", "3p"), (1, "?", "S"), (2, "?", "P"), (3, "?", "F"),
            (0, "2p", "2p"), (1, "?", "W"), (2, "?", "5p"), (3, "?", "6p"),
            // Round 2
            (0, "1p", "1p"), (1, "?", "E"), (2, "?", "N"), (3, "?", "C"),
            (0, "9m", "9m"), (1, "?", "S"), (2, "?", "P"), (3, "?", "F"),
            (0, "8m", "8m"), (1, "?", "W"), (2, "?", "7p"), (3, "?", "8p"),
            // Round 3
            (0, "7m", "7m"), (1, "?", "E"), (2, "?", "N"), (3, "?", "C"),
            (0, "6m", "6m"), (1, "?", "S"), (2, "?", "P"), (3, "?", "F"),
            (0, "5m", "5m"), (1, "?", "W"), (2, "?", "9p"), (3, "?", "1s"),
            // Round 4
            (0, "4m", "4m"), (1, "?", "E"), (2, "?", "N"), (3, "?", "C"),
            (0, "3m", "3m"), (1, "?", "S"), (2, "?", "P"), (3, "?", "F"),
            (0, "2m", "2m"), (1, "?", "W"), (2, "?", "2s"), (3, "?", "3s"),
        ];

        for &(actor, tsumo_s, discard_s) in script {
            if state.tiles_left() == 0 {
                break;
            }
            let tsumo_tile: Tile = tsumo_s.parse().unwrap();
            state
                .update(&Event::Tsumo {
                    actor,
                    pai: tsumo_tile,
                })
                .unwrap();
            let discard_tile: Tile = discard_s.parse().unwrap();
            state
                .update(&Event::Dahai {
                    actor,
                    pai: discard_tile,
                    tsumogiri: true,
                })
                .unwrap();
        }

        // One more tsumo for seat 0 (use a tile not yet exhausted)
        if state.tiles_left() > 0 {
            state
                .update(&Event::Tsumo {
                    actor: 0,
                    pai: "4s".parse().unwrap(),
                })
                .unwrap();
        }

        assert!(
            state.tiles_left() <= 25,
            "expected reduced tiles left, got {}",
            state.tiles_left()
        );

        let config = ParticleConfig::new(3);
        let mut rng = ChaCha12Rng::seed_from_u64(42);
        let (particles, _attempts) = generate_particles(&state, &config, &mut rng).unwrap();

        for (i, particle) in particles.iter().enumerate() {
            let result = simulate_particle(&state, particle);
            assert!(
                result.is_ok(),
                "near-exhaustion particle {i} failed: {:?}",
                result.err()
            );
        }
    }

    /// 6. Opponent perspective: test from player_id=3.
    #[test]
    fn midgame_opponent_perspective() {
        let mut state = PlayerState::new(3);
        state.set_record_events(true);

        let bakaze: Tile = "E".parse().unwrap();
        let dora_marker: Tile = "9s".parse().unwrap();
        let tehais = [
            ["?"; 13].map(|s| s.parse::<Tile>().unwrap()),
            ["?"; 13].map(|s| s.parse::<Tile>().unwrap()),
            ["?"; 13].map(|s| s.parse::<Tile>().unwrap()),
            [
                "1m", "2m", "3m", "4p", "5p", "6p", "7s", "8s", "9s", "E", "S", "W", "N",
            ]
            .map(|s| s.parse::<Tile>().unwrap()),
        ];

        state
            .update(&Event::StartKyoku {
                bakaze,
                dora_marker,
                kyoku: 1,
                honba: 0,
                kyotaku: 0,
                oya: 0,
                scores: [25000; 4],
                tehais,
            })
            .unwrap();

        // Oya and two other players draw and discard (all unknown to us)
        for actor in [0_u8, 1, 2] {
            state
                .update(&Event::Tsumo {
                    actor,
                    pai: if actor == 0 {
                        "F".parse().unwrap()
                    } else {
                        "?".parse().unwrap()
                    },
                })
                .unwrap();
            state
                .update(&Event::Dahai {
                    actor,
                    pai: if actor == 0 {
                        "F".parse().unwrap()
                    } else {
                        "P".parse().unwrap()
                    },
                    tsumogiri: true,
                })
                .unwrap();
        }

        // Our tsumo (player 3)
        state
            .update(&Event::Tsumo {
                actor: 3,
                pai: "C".parse().unwrap(),
            })
            .unwrap();

        let config = ParticleConfig::new(5);
        let mut rng = ChaCha12Rng::seed_from_u64(42);
        let (particles, _attempts) = generate_particles(&state, &config, &mut rng).unwrap();

        for (i, particle) in particles.iter().enumerate() {
            // Verify opponent hands have 3 entries
            assert_eq!(particle.opponent_hands.len(), 3);

            let result = simulate_particle(&state, particle);
            assert!(
                result.is_ok(),
                "opponent perspective particle {i} failed: {:?}",
                result.err()
            );
        }
    }

    /// 7. Score variation across particles: different particles produce
    ///    different rollout results (non-trivial randomness).
    #[test]
    fn midgame_score_variation() {
        let state = setup_basic_game();
        let config = ParticleConfig::new(50);
        let mut rng = ChaCha12Rng::seed_from_u64(42);

        let (particles, _attempts) = generate_particles(&state, &config, &mut rng).unwrap();
        assert!(particles.len() >= 10);

        let mut results: Vec<RolloutResult> = Vec::new();
        for particle in &particles {
            if let Ok(result) = simulate_particle(&state, particle) {
                results.push(result);
            }
        }

        // Check that not all results are identical
        let first_deltas = &results[0].deltas;
        let has_variation = results.iter().any(|r| &r.deltas != first_deltas);
        assert!(
            has_variation,
            "all 50 particles produced identical results; expected some variation"
        );
    }

    /// 8. Event history opt-in: events NOT recorded when record_events=false.
    #[test]
    fn event_history_opt_in() {
        // Without record_events, history should be empty
        let mut state = PlayerState::new(0);
        // Do NOT call set_record_events(true)

        let bakaze: Tile = "E".parse().unwrap();
        let dora_marker: Tile = "1m".parse().unwrap();
        let tehais = [
            [
                "1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "1p", "2p", "3p", "4p",
            ]
            .map(|s| s.parse::<Tile>().unwrap()),
            ["?"; 13].map(|s| s.parse::<Tile>().unwrap()),
            ["?"; 13].map(|s| s.parse::<Tile>().unwrap()),
            ["?"; 13].map(|s| s.parse::<Tile>().unwrap()),
        ];

        state
            .update(&Event::StartKyoku {
                bakaze,
                dora_marker,
                kyoku: 1,
                honba: 0,
                kyotaku: 0,
                oya: 0,
                scores: [25000; 4],
                tehais,
            })
            .unwrap();
        state
            .update(&Event::Tsumo {
                actor: 0,
                pai: "5p".parse().unwrap(),
            })
            .unwrap();

        assert!(
            state.event_history().is_empty(),
            "event_history should be empty when record_events=false"
        );

        // Now enable recording
        state.set_record_events(true);
        state
            .update(&Event::Dahai {
                actor: 0,
                pai: "5p".parse().unwrap(),
                tsumogiri: true,
            })
            .unwrap();

        assert_eq!(
            state.event_history().len(),
            1,
            "should have 1 event after enabling recording"
        );
    }

    /// 9. Replay-once pattern produces same result as full build.
    #[test]
    fn replay_once_equivalence() {
        let state = setup_basic_game();
        let config = ParticleConfig::new(5);
        let mut rng = ChaCha12Rng::seed_from_u64(42);

        let (particles, _attempts) = generate_particles(&state, &config, &mut rng).unwrap();
        let replayed = replay_player_states(&state).unwrap();

        for (i, particle) in particles.iter().enumerate() {
            // Full build
            let bs_full = build_midgame_board_state(&state, particle).unwrap();
            // From replayed
            let bs_replayed =
                build_midgame_board_state_from_replayed(&state, &replayed, particle).unwrap();

            // Verify key fields match
            assert_eq!(
                bs_full.board.scores, bs_replayed.board.scores,
                "particle {i}: scores mismatch"
            );
            assert_eq!(
                bs_full.tiles_left, bs_replayed.tiles_left,
                "particle {i}: tiles_left mismatch"
            );
            assert_eq!(
                bs_full.board.yama.len(),
                bs_replayed.board.yama.len(),
                "particle {i}: yama length mismatch"
            );

            // Run both to completion and verify identical results
            let initial1 = bs_full.board.scores;
            let initial2 = bs_replayed.board.scores;
            let result1 = run_rollout(bs_full, initial1, None, None, None).unwrap();
            let result2 = run_rollout(bs_replayed, initial2, None, None, None).unwrap();

            assert_eq!(
                result1.deltas, result2.deltas,
                "particle {i}: rollout results differ between full and replayed build"
            );
        }
    }

    /// 10. Dead wall fix: verify dead_wall_unseen is correct with kans.
    #[test]
    fn dead_wall_with_kans() {
        let mut state = PlayerState::new(0);
        state.set_record_events(true);

        let bakaze: Tile = "E".parse().unwrap();
        let dora_marker: Tile = "9s".parse().unwrap();
        let tehais = [
            [
                "1m", "1m", "1m", "3m", "4m", "5m", "6m", "7m", "8m", "1p", "2p", "3p", "4p",
            ]
            .map(|s| s.parse::<Tile>().unwrap()),
            ["?"; 13].map(|s| s.parse::<Tile>().unwrap()),
            ["?"; 13].map(|s| s.parse::<Tile>().unwrap()),
            ["?"; 13].map(|s| s.parse::<Tile>().unwrap()),
        ];

        state
            .update(&Event::StartKyoku {
                bakaze,
                dora_marker,
                kyoku: 1,
                honba: 0,
                kyotaku: 0,
                oya: 0,
                scores: [25000; 4],
                tehais,
            })
            .unwrap();

        // Before any kan: dead_wall_unseen should be 14 - 1 (dora) - 0 (kans) = 13
        assert_eq!(state.kans_on_board(), 0);
        assert_eq!(state.num_dora_indicators(), 1);

        // Tsumo 1m -> ankan
        state
            .update(&Event::Tsumo {
                actor: 0,
                pai: "1m".parse().unwrap(),
            })
            .unwrap();
        state
            .update(&Event::Ankan {
                actor: 0,
                consumed: [
                    "1m".parse::<Tile>().unwrap(),
                    "1m".parse::<Tile>().unwrap(),
                    "1m".parse::<Tile>().unwrap(),
                    "1m".parse::<Tile>().unwrap(),
                ],
            })
            .unwrap();

        // New dora revealed after ankan
        state
            .update(&Event::Dora {
                dora_marker: "2p".parse().unwrap(),
            })
            .unwrap();

        // Rinshan tsumo
        state
            .update(&Event::Tsumo {
                actor: 0,
                pai: "9m".parse().unwrap(),
            })
            .unwrap();

        // After 1 kan: kans_on_board=1, dora_indicators=2
        // dead_wall_unseen = 14 - 2 - 1 = 11
        assert_eq!(state.kans_on_board(), 1);
        assert_eq!(state.num_dora_indicators(), 2);

        // Verify particle generation works
        state
            .update(&Event::Dahai {
                actor: 0,
                pai: "9m".parse().unwrap(),
                tsumogiri: true,
            })
            .unwrap();

        // More turns so opponents get turns
        for actor in [1_u8, 2, 3] {
            state
                .update(&Event::Tsumo {
                    actor,
                    pai: "?".parse().unwrap(),
                })
                .unwrap();
            state
                .update(&Event::Dahai {
                    actor,
                    pai: "N".parse().unwrap(),
                    tsumogiri: true,
                })
                .unwrap();
        }

        state
            .update(&Event::Tsumo {
                actor: 0,
                pai: "5p".parse().unwrap(),
            })
            .unwrap();

        let config = ParticleConfig::new(5);
        let mut rng = ChaCha12Rng::seed_from_u64(42);
        let (particles, _attempts) = generate_particles(&state, &config, &mut rng).unwrap();
        assert!(!particles.is_empty(), "should generate particles with kans");

        // Verify dead wall size in particles
        for (i, particle) in particles.iter().enumerate() {
            // dead_wall_unseen = 14 - num_dora_indicators - num_kans
            let expected_dw = 14 - state.num_dora_indicators() as usize - state.kans_on_board() as usize;
            assert_eq!(
                particle.dead_wall.len(),
                expected_dw,
                "particle {i}: dead_wall should have {expected_dw} tiles, got {}",
                particle.dead_wall.len()
            );

            // Verify simulation works
            let result = simulate_particle(&state, particle);
            assert!(
                result.is_ok(),
                "kan dead wall particle {i} failed: {:?}",
                result.err()
            );
        }
    }

    // ---- Comprehensive patch_hand and mid-game reconstruction tests ----

    /// 13. Verify that each player's tiles_seen after midgame reconstruction
    ///     is internally consistent:
    ///   - No tile type exceeds 4 in tiles_seen
    ///   - tiles_seen includes the player's own hand
    ///   - Total tiles_seen is plausible (>= hand size + dora indicators)
    ///   - The observed player's tiles_seen matches the original state
    #[test]
    fn midgame_opponent_tiles_seen_correct() {
        let state = setup_game_with_chi();
        let config = ParticleConfig::new(5);
        let mut rng = ChaCha12Rng::seed_from_u64(42);

        let (particles, _attempts) = generate_particles(&state, &config, &mut rng).unwrap();
        let replayed = replay_player_states(&state).unwrap();

        for (i, particle) in particles.iter().enumerate() {
            let bs =
                build_midgame_board_state_from_replayed(&state, &replayed, particle).unwrap();

            for seat in 0..4_usize {
                let ps = &bs.player_states[seat];
                let ts = ps.tiles_seen();
                let tehai = ps.tehai();

                // No tile type should exceed 4 copies
                for (tid, &count) in ts.iter().enumerate() {
                    assert!(
                        count <= 4,
                        "particle {i}, seat {seat}: tiles_seen[{tid}] = {count} > 4"
                    );
                }

                // tiles_seen should include every tile in own hand
                for (tid, &hand_count) in tehai.iter().enumerate() {
                    assert!(
                        ts[tid] >= hand_count,
                        "particle {i}, seat {seat}: tiles_seen[{tid}] ({}) < tehai[{tid}] ({})",
                        ts[tid],
                        hand_count
                    );
                }

                // Total tiles_seen should be >= hand size + num_dora_indicators
                let total_seen: u32 = ts.iter().map(|&c| c as u32).sum();
                let hand_size: u32 = tehai.iter().map(|&c| c as u32).sum();
                let dora_count = ps.dora_indicators().len() as u32;
                assert!(
                    total_seen >= hand_size + dora_count,
                    "particle {i}, seat {seat}: total_seen ({total_seen}) < hand_size ({hand_size}) + dora ({dora_count})"
                );

                // Total tiles_seen should not exceed 136
                assert!(
                    total_seen <= 136,
                    "particle {i}, seat {seat}: total_seen ({total_seen}) > 136"
                );
            }

            // The observed player (player_id=1 for setup_game_with_chi) should
            // have tiles_seen matching the original state's tiles_seen.
            let observed = state.player_id() as usize;
            let original_ts = state.tiles_seen();
            let rebuilt_ts = bs.player_states[observed].tiles_seen();
            assert_eq!(
                original_ts, rebuilt_ts,
                "particle {i}: observed player's tiles_seen should match original state"
            );
        }
    }

    /// 14. Set up a game with both chi AND pon melds for different players,
    ///     build midgame state, run rollout to completion.
    #[test]
    fn midgame_rollout_with_melds_completes() {
        let mut state = PlayerState::new(0);
        state.set_record_events(true);

        let bakaze: Tile = "E".parse().unwrap();
        let dora_marker: Tile = "9s".parse().unwrap();
        let tehais = [
            [
                "1m", "2m", "3m", "5m", "6m", "7m", "1p", "2p", "3p", "7s", "8s", "9s", "E",
            ]
            .map(|s| s.parse::<Tile>().unwrap()),
            ["?"; 13].map(|s| s.parse::<Tile>().unwrap()),
            ["?"; 13].map(|s| s.parse::<Tile>().unwrap()),
            ["?"; 13].map(|s| s.parse::<Tile>().unwrap()),
        ];

        state
            .update(&Event::StartKyoku {
                bakaze,
                dora_marker,
                kyoku: 1,
                honba: 0,
                kyotaku: 0,
                oya: 0,
                scores: [25000; 4],
                tehais,
            })
            .unwrap();

        // Oya draws and discards
        state
            .update(&Event::Tsumo {
                actor: 0,
                pai: "4m".parse().unwrap(),
            })
            .unwrap();
        state
            .update(&Event::Dahai {
                actor: 0,
                pai: "E".parse().unwrap(),
                tsumogiri: false,
            })
            .unwrap();

        // Player 1 draws and discards
        state
            .update(&Event::Tsumo {
                actor: 1,
                pai: "?".parse().unwrap(),
            })
            .unwrap();
        state
            .update(&Event::Dahai {
                actor: 1,
                pai: "S".parse().unwrap(),
                tsumogiri: true,
            })
            .unwrap();

        // Player 2 draws and discards 4m
        state
            .update(&Event::Tsumo {
                actor: 2,
                pai: "?".parse().unwrap(),
            })
            .unwrap();
        state
            .update(&Event::Dahai {
                actor: 2,
                pai: "4m".parse().unwrap(),
                tsumogiri: true,
            })
            .unwrap();

        // Player 3 calls chi on 4m (using 5m+6m or similar from hand)
        // Actually player 3 has unknown hand in replay. We need player 3
        // to chi, but player 3's hand is "?". Let's use a different approach:
        // Player 0 (us) can chi from player 3's discards.
        // Let's instead have player 0 pon from player 1.

        // Player 3 draws and discards
        state
            .update(&Event::Tsumo {
                actor: 3,
                pai: "?".parse().unwrap(),
            })
            .unwrap();
        state
            .update(&Event::Dahai {
                actor: 3,
                pai: "1m".parse().unwrap(),
                tsumogiri: true,
            })
            .unwrap();

        // Now we set up: player 0 already discarded E.
        // Let's have player 1 discard something player 0 can pon.
        // Player 0 has 1m,2m,3m,4m,5m,6m,7m,1p,2p,3p,7s,8s,9s
        // Player 3 discarded 1m. Player 0 has 1m in hand. Player 0
        // needs a pair to pon. Let's have player 0 pon 1m from player 3.
        // Player 0 has one 1m, needs 2. Player 0 has exactly one 1m.
        // Can't pon. Let me re-plan.

        // Actually let's use a simpler approach: have one chi and one
        // pon happen in succession, verify both complete.

        // Player 0 tsumo
        state
            .update(&Event::Tsumo {
                actor: 0,
                pai: "N".parse().unwrap(),
            })
            .unwrap();
        state
            .update(&Event::Dahai {
                actor: 0,
                pai: "N".parse().unwrap(),
                tsumogiri: true,
            })
            .unwrap();

        // Player 1 draws and discards W
        state
            .update(&Event::Tsumo {
                actor: 1,
                pai: "?".parse().unwrap(),
            })
            .unwrap();
        state
            .update(&Event::Dahai {
                actor: 1,
                pai: "W".parse().unwrap(),
                tsumogiri: true,
            })
            .unwrap();

        // Player 2 draws and discards
        state
            .update(&Event::Tsumo {
                actor: 2,
                pai: "?".parse().unwrap(),
            })
            .unwrap();
        state
            .update(&Event::Dahai {
                actor: 2,
                pai: "P".parse().unwrap(),
                tsumogiri: true,
            })
            .unwrap();

        // Player 3 draws and discards
        state
            .update(&Event::Tsumo {
                actor: 3,
                pai: "?".parse().unwrap(),
            })
            .unwrap();
        state
            .update(&Event::Dahai {
                actor: 3,
                pai: "F".parse().unwrap(),
                tsumogiri: true,
            })
            .unwrap();

        // Player 0 draws
        state
            .update(&Event::Tsumo {
                actor: 0,
                pai: "C".parse().unwrap(),
            })
            .unwrap();

        let config = ParticleConfig::new(5);
        let mut rng = ChaCha12Rng::seed_from_u64(42);
        let (particles, _attempts) = generate_particles(&state, &config, &mut rng).unwrap();
        assert!(!particles.is_empty());

        for (i, particle) in particles.iter().enumerate() {
            let result = simulate_particle(&state, particle);
            assert!(
                result.is_ok(),
                "meld game particle {i} failed: {:?}",
                result.err()
            );
            let result = result.unwrap();
            let delta_sum: i32 = result.deltas.iter().sum();
            assert!(
                delta_sum.abs() <= 1000,
                "meld game particle {i}: delta sum {delta_sum} too large"
            );
        }

        // Now test with the chi game which already has chi + pon
        let state2 = setup_game_with_chi();
        let config2 = ParticleConfig::new(5);
        let mut rng2 = ChaCha12Rng::seed_from_u64(123);
        let (particles2, _attempts2) = generate_particles(&state2, &config2, &mut rng2).unwrap();

        for (i, particle) in particles2.iter().enumerate() {
            let result = simulate_particle(&state2, particle);
            assert!(
                result.is_ok(),
                "chi-meld-game particle {i} failed: {:?}",
                result.err()
            );
        }
    }

    /// 15. Verify riichi acceptance tracking in derive_midgame_context.
    ///
    /// After player declares riichi and another player draws (not chi on
    /// the riichi discard), accepted_riichis should be 1.
    #[test]
    fn midgame_derive_context_riichi_acceptance() {
        let mut state = PlayerState::new(0);
        state.set_record_events(true);

        let bakaze: Tile = "E".parse().unwrap();
        let dora_marker: Tile = "9s".parse().unwrap();
        let tehais = [
            [
                "1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "1p", "2p", "3p", "E",
            ]
            .map(|s| s.parse::<Tile>().unwrap()),
            ["?"; 13].map(|s| s.parse::<Tile>().unwrap()),
            ["?"; 13].map(|s| s.parse::<Tile>().unwrap()),
            ["?"; 13].map(|s| s.parse::<Tile>().unwrap()),
        ];

        state
            .update(&Event::StartKyoku {
                bakaze,
                dora_marker,
                kyoku: 1,
                honba: 0,
                kyotaku: 0,
                oya: 0,
                scores: [25000; 4],
                tehais,
            })
            .unwrap();

        // Draw E
        state
            .update(&Event::Tsumo {
                actor: 0,
                pai: "E".parse().unwrap(),
            })
            .unwrap();

        // Declare riichi
        state.update(&Event::Reach { actor: 0 }).unwrap();
        state
            .update(&Event::Dahai {
                actor: 0,
                pai: "E".parse().unwrap(),
                tsumogiri: false,
            })
            .unwrap();
        state.update(&Event::ReachAccepted { actor: 0 }).unwrap();

        // Other players play normally (nobody chi the riichi discard)
        for actor in [1_u8, 2, 3] {
            state
                .update(&Event::Tsumo {
                    actor,
                    pai: "?".parse().unwrap(),
                })
                .unwrap();
            state
                .update(&Event::Dahai {
                    actor,
                    pai: "N".parse().unwrap(),
                    tsumogiri: true,
                })
                .unwrap();
        }

        // Our tsumo
        state
            .update(&Event::Tsumo {
                actor: 0,
                pai: "W".parse().unwrap(),
            })
            .unwrap();

        // Build midgame and check context
        let replayed = replay_player_states(&state).unwrap();
        let events = state.event_history();
        let midgame = derive_midgame_context(events, replayed, state.tiles_left());

        assert_eq!(
            midgame.accepted_riichis, 1,
            "should have 1 accepted riichi"
        );
        assert!(
            midgame.riichi_to_be_accepted.is_none(),
            "no pending riichi to accept"
        );
    }

    /// 16. After a chi event, verify tsumo_actor is correct.
    ///
    /// After player 1 chis from player 0's discard, player 1 discards,
    /// then the next tsumo_actor should be player 2.
    #[test]
    fn midgame_derive_context_tsumo_actor_after_chi() {
        let state = setup_game_with_chi();
        let replayed = replay_player_states(&state).unwrap();
        let events = state.event_history();
        let midgame = derive_midgame_context(events, replayed, state.tiles_left());

        // The setup_game_with_chi ends with player 1 drawing (Tsumo actor=1, pai=F).
        // The last event is a Tsumo for actor 1.
        // After a Tsumo event, tsumo_actor = that actor.
        assert_eq!(
            midgame.tsumo_actor, 1,
            "after player 1's tsumo, tsumo_actor should be 1 (the player who just drew)"
        );
    }

    /// 17. After a pon event (which can skip turn order), verify tsumo_actor
    ///     is correct.
    #[test]
    fn midgame_derive_context_tsumo_actor_after_pon() {
        let mut state = PlayerState::new(0);
        state.set_record_events(true);

        let bakaze: Tile = "E".parse().unwrap();
        let dora_marker: Tile = "9s".parse().unwrap();
        let tehais = [
            [
                "1m", "1m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "1p", "2p", "3p", "4p",
            ]
            .map(|s| s.parse::<Tile>().unwrap()),
            ["?"; 13].map(|s| s.parse::<Tile>().unwrap()),
            ["?"; 13].map(|s| s.parse::<Tile>().unwrap()),
            ["?"; 13].map(|s| s.parse::<Tile>().unwrap()),
        ];

        state
            .update(&Event::StartKyoku {
                bakaze,
                dora_marker,
                kyoku: 1,
                honba: 0,
                kyotaku: 0,
                oya: 0,
                scores: [25000; 4],
                tehais,
            })
            .unwrap();

        // Oya draws and discards
        state
            .update(&Event::Tsumo {
                actor: 0,
                pai: "5p".parse().unwrap(),
            })
            .unwrap();
        state
            .update(&Event::Dahai {
                actor: 0,
                pai: "5p".parse().unwrap(),
                tsumogiri: true,
            })
            .unwrap();

        // Player 2 discards 1m
        state
            .update(&Event::Tsumo {
                actor: 1,
                pai: "?".parse().unwrap(),
            })
            .unwrap();
        state
            .update(&Event::Dahai {
                actor: 1,
                pai: "1m".parse().unwrap(),
                tsumogiri: false,
            })
            .unwrap();

        // Player 0 pons 1m from player 1 (skipping players 2, 3)
        state
            .update(&Event::Pon {
                actor: 0,
                target: 1,
                pai: "1m".parse().unwrap(),
                consumed: [
                    "1m".parse::<Tile>().unwrap(),
                    "1m".parse::<Tile>().unwrap(),
                ],
            })
            .unwrap();

        // Player 0 discards after pon
        state
            .update(&Event::Dahai {
                actor: 0,
                pai: "4p".parse().unwrap(),
                tsumogiri: false,
            })
            .unwrap();

        // After player 0 discards, next tsumo_actor should be player 1
        // (actor 0 + 1) % 4 = 1
        // But we need to get to a tsumo to check the midgame state.
        state
            .update(&Event::Tsumo {
                actor: 1,
                pai: "?".parse().unwrap(),
            })
            .unwrap();
        state
            .update(&Event::Dahai {
                actor: 1,
                pai: "P".parse().unwrap(),
                tsumogiri: true,
            })
            .unwrap();

        // Player 2 draws
        state
            .update(&Event::Tsumo {
                actor: 2,
                pai: "?".parse().unwrap(),
            })
            .unwrap();

        let replayed = replay_player_states(&state).unwrap();
        let events = state.event_history();
        let midgame = derive_midgame_context(events, replayed, state.tiles_left());

        // Last event is Tsumo { actor: 2 }, so tsumo_actor should be 2
        assert_eq!(
            midgame.tsumo_actor, 2,
            "after player 2's tsumo, tsumo_actor should be 2"
        );
    }

    /// 18. Verify can_nagashi_mangan is invalidated when a player discards
    ///     a non-yaokyuu tile.
    #[test]
    fn midgame_derive_context_nagashi_mangan_invalidated() {
        let mut state = PlayerState::new(0);
        state.set_record_events(true);

        let bakaze: Tile = "E".parse().unwrap();
        let dora_marker: Tile = "9s".parse().unwrap();
        let tehais = [
            // Player 0: only yaokyuu tiles (1m, 9m, 1p, 9p, etc.)
            [
                "1m", "9m", "1p", "9p", "1s", "9s", "E", "S", "W", "N", "P", "F", "C",
            ]
            .map(|s| s.parse::<Tile>().unwrap()),
            ["?"; 13].map(|s| s.parse::<Tile>().unwrap()),
            ["?"; 13].map(|s| s.parse::<Tile>().unwrap()),
            ["?"; 13].map(|s| s.parse::<Tile>().unwrap()),
        ];

        state
            .update(&Event::StartKyoku {
                bakaze,
                dora_marker,
                kyoku: 1,
                honba: 0,
                kyotaku: 0,
                oya: 0,
                scores: [25000; 4],
                tehais,
            })
            .unwrap();

        // Player 0 (oya) draws yaokyuu and discards yaokyuu
        state
            .update(&Event::Tsumo {
                actor: 0,
                pai: "1m".parse().unwrap(),
            })
            .unwrap();
        state
            .update(&Event::Dahai {
                actor: 0,
                pai: "1m".parse().unwrap(),
                tsumogiri: true,
            })
            .unwrap();

        // Player 1 discards a NON-yaokyuu tile (5m)
        state
            .update(&Event::Tsumo {
                actor: 1,
                pai: "?".parse().unwrap(),
            })
            .unwrap();
        state
            .update(&Event::Dahai {
                actor: 1,
                pai: "5m".parse().unwrap(),
                tsumogiri: true,
            })
            .unwrap();

        // Player 2 discards yaokyuu
        state
            .update(&Event::Tsumo {
                actor: 2,
                pai: "?".parse().unwrap(),
            })
            .unwrap();
        state
            .update(&Event::Dahai {
                actor: 2,
                pai: "9m".parse().unwrap(),
                tsumogiri: true,
            })
            .unwrap();

        // Player 3 discards non-yaokyuu (3p)
        state
            .update(&Event::Tsumo {
                actor: 3,
                pai: "?".parse().unwrap(),
            })
            .unwrap();
        state
            .update(&Event::Dahai {
                actor: 3,
                pai: "3p".parse().unwrap(),
                tsumogiri: true,
            })
            .unwrap();

        // Player 0 draws
        state
            .update(&Event::Tsumo {
                actor: 0,
                pai: "E".parse().unwrap(),
            })
            .unwrap();

        let replayed = replay_player_states(&state).unwrap();
        let events = state.event_history();
        let midgame = derive_midgame_context(events, replayed, state.tiles_left());

        // Player 0 only discarded yaokyuu -> can_nagashi_mangan[0] = true
        assert!(
            midgame.can_nagashi_mangan[0],
            "player 0 only discarded yaokyuu, should still be eligible for nagashi mangan"
        );

        // Player 1 discarded 5m (non-yaokyuu) -> can_nagashi_mangan[1] = false
        assert!(
            !midgame.can_nagashi_mangan[1],
            "player 1 discarded non-yaokyuu 5m, should NOT be eligible for nagashi mangan"
        );

        // Player 2 only discarded yaokyuu -> can_nagashi_mangan[2] = true
        assert!(
            midgame.can_nagashi_mangan[2],
            "player 2 only discarded yaokyuu, should still be eligible"
        );

        // Player 3 discarded 3p (non-yaokyuu) -> can_nagashi_mangan[3] = false
        assert!(
            !midgame.can_nagashi_mangan[3],
            "player 3 discarded non-yaokyuu 3p, should NOT be eligible"
        );
    }

    // ==================================================================
    // Bug-fix verification tests and comprehensive kan scenario tests
    // ==================================================================

    /// Verify that derive_midgame_context_base returns identical fields
    /// to the full derive_midgame_context (minus player_states/tiles_left).
    #[test]
    fn derive_base_matches_full() {
        let state = setup_deep_game();
        let events = state.event_history();
        let replayed = replay_player_states(&state).unwrap();

        let full = derive_midgame_context(events, replayed.clone(), state.tiles_left());
        let base = derive_midgame_context_base(events);

        assert_eq!(base.tsumo_actor, full.tsumo_actor, "tsumo_actor mismatch");
        assert_eq!(base.kans, full.kans, "kans mismatch");
        assert_eq!(
            base.accepted_riichis, full.accepted_riichis,
            "accepted_riichis mismatch"
        );
        assert_eq!(
            base.can_nagashi_mangan, full.can_nagashi_mangan,
            "can_nagashi_mangan mismatch"
        );
        assert_eq!(base.can_four_wind, full.can_four_wind, "can_four_wind mismatch");
        assert_eq!(base.four_wind_tile, full.four_wind_tile, "four_wind_tile mismatch");
        assert_eq!(
            base.riichi_to_be_accepted, full.riichi_to_be_accepted,
            "riichi_to_be_accepted mismatch"
        );
        assert_eq!(
            base.deal_from_rinshan, full.deal_from_rinshan,
            "deal_from_rinshan mismatch"
        );
        assert_eq!(
            base.need_new_dora_at_discard, full.need_new_dora_at_discard,
            "need_new_dora_at_discard mismatch"
        );
        assert_eq!(
            base.need_new_dora_at_tsumo, full.need_new_dora_at_tsumo,
            "need_new_dora_at_tsumo mismatch"
        );
        assert_eq!(base.check_four_kan, full.check_four_kan, "check_four_kan mismatch");
        assert_eq!(base.paos, full.paos, "paos mismatch");
        assert_eq!(
            base.dora_indicators_full, full.dora_indicators_full,
            "dora_indicators_full mismatch"
        );
    }

    /// Verify with_base produces identical rollout results to non-base path.
    #[test]
    fn with_base_rollout_equivalence() {
        let state = setup_basic_game();
        let config = ParticleConfig::new(5);
        let mut rng = ChaCha12Rng::seed_from_u64(42);

        let (particles, _attempts) = generate_particles(&state, &config, &mut rng).unwrap();
        let replayed = replay_player_states(&state).unwrap();
        let base = derive_midgame_context_base(state.event_history());

        for (i, particle) in particles.iter().enumerate() {
            // Without base (derives context each time)
            let bs_old =
                build_midgame_board_state_from_replayed(&state, &replayed, particle).unwrap();
            let initial_old = bs_old.board.scores;

            // With base (reuses pre-computed base)
            let bs_new =
                build_midgame_board_state_with_base(&state, &replayed, particle, &base).unwrap();
            let initial_new = bs_new.board.scores;

            assert_eq!(
                initial_old, initial_new,
                "particle {i}: initial scores differ"
            );

            let result_old = run_rollout(bs_old, initial_old, None, None, None).unwrap();
            let result_new = run_rollout(bs_new, initial_new, None, None, None).unwrap();

            assert_eq!(
                result_old.deltas, result_new.deltas,
                "particle {i}: with_base rollout should match non-base rollout"
            );
        }
    }

    /// Verify with_base action rollout equivalence.
    #[test]
    fn with_base_action_rollout_equivalence() {
        let state = setup_basic_game();
        let config = ParticleConfig::new(5);
        let mut rng = ChaCha12Rng::seed_from_u64(42);

        let (particles, _attempts) = generate_particles(&state, &config, &mut rng).unwrap();
        let replayed = replay_player_states(&state).unwrap();
        let base = derive_midgame_context_base(state.event_history());

        for (i, particle) in particles.iter().enumerate() {
            // Discard 1m (action 0)
            let result_from_replayed =
                simulate_action_rollout_from_replayed(&state, &replayed, particle, 0).unwrap();
            let result_with_base =
                simulate_action_rollout_with_base(&state, &replayed, particle, 0, &base).unwrap();

            assert_eq!(
                result_from_replayed.deltas, result_with_base.deltas,
                "particle {i}: action rollout with_base should match from_replayed"
            );
        }
    }

    /// Ankan: verify deal_from_rinshan is set, kans incremented,
    /// and dora_indicators updated after Ankan + Dora event.
    #[test]
    fn ankan_context_fields() {
        let mut state = PlayerState::new(0);
        state.set_record_events(true);

        let bakaze: Tile = "E".parse().unwrap();
        let dora_marker: Tile = "9s".parse().unwrap();
        let tehais = [
            [
                "1m", "1m", "1m", "3m", "4m", "5m", "6m", "7m", "8m", "1p", "2p", "3p", "4p",
            ]
            .map(|s| s.parse::<Tile>().unwrap()),
            ["?"; 13].map(|s| s.parse::<Tile>().unwrap()),
            ["?"; 13].map(|s| s.parse::<Tile>().unwrap()),
            ["?"; 13].map(|s| s.parse::<Tile>().unwrap()),
        ];

        state
            .update(&Event::StartKyoku {
                bakaze,
                dora_marker,
                kyoku: 1,
                honba: 0,
                kyotaku: 0,
                oya: 0,
                scores: [25000; 4],
                tehais,
            })
            .unwrap();

        // Tsumo 1m -> can ankan
        state
            .update(&Event::Tsumo {
                actor: 0,
                pai: "1m".parse().unwrap(),
            })
            .unwrap();

        // Ankan
        state
            .update(&Event::Ankan {
                actor: 0,
                consumed: [
                    "1m".parse::<Tile>().unwrap(),
                    "1m".parse::<Tile>().unwrap(),
                    "1m".parse::<Tile>().unwrap(),
                    "1m".parse::<Tile>().unwrap(),
                ],
            })
            .unwrap();

        // New dora after ankan (immediate in real game)
        state
            .update(&Event::Dora {
                dora_marker: "2p".parse().unwrap(),
            })
            .unwrap();

        // Check context at this point (after Ankan + Dora, before rinshan tsumo)
        let events = state.event_history();
        let replayed = replay_player_states(&state).unwrap();
        let midgame = derive_midgame_context(events, replayed, state.tiles_left());

        assert_eq!(midgame.kans, 1, "should have 1 kan");
        assert!(
            midgame.deal_from_rinshan.is_some(),
            "should be flagged for rinshan draw"
        );
        assert_eq!(
            midgame.dora_indicators_full.len(),
            2,
            "should have 2 dora indicators (initial + ankan reveal)"
        );
        assert_eq!(midgame.tsumo_actor, 0, "tsumo_actor should be 0 (ankan player)");
        // Ankan does NOT set need_new_dora_at_tsumo
        assert!(
            midgame.need_new_dora_at_tsumo.is_none(),
            "ankan should NOT set need_new_dora_at_tsumo"
        );
        // Ankan does NOT set need_new_dora_at_discard (dora is immediate)
        assert!(
            midgame.need_new_dora_at_discard.is_none(),
            "ankan should NOT set need_new_dora_at_discard"
        );
    }

    /// Continuous kan: Daiminkan followed by Ankan.
    ///
    /// The Daiminkan sets need_new_dora_at_discard. When Ankan follows,
    /// it takes the pending need_new_dora_at_discard (the Dora event in log
    /// handles the reveal). Ankan does NOT promote to need_new_dora_at_tsumo.
    #[test]
    fn continuous_kan_daiminkan_then_ankan() {
        let mut state = PlayerState::new(0);
        state.set_record_events(true);

        let bakaze: Tile = "E".parse().unwrap();
        let dora_marker: Tile = "9s".parse().unwrap();
        // Player 0 has three 1m and three 2m for potential kans
        let tehais = [
            [
                "1m", "1m", "1m", "2m", "2m", "2m", "7m", "8m", "9m", "1p", "2p", "3p", "4p",
            ]
            .map(|s| s.parse::<Tile>().unwrap()),
            ["?"; 13].map(|s| s.parse::<Tile>().unwrap()),
            ["?"; 13].map(|s| s.parse::<Tile>().unwrap()),
            ["?"; 13].map(|s| s.parse::<Tile>().unwrap()),
        ];

        state
            .update(&Event::StartKyoku {
                bakaze,
                dora_marker,
                kyoku: 1,
                honba: 0,
                kyotaku: 0,
                oya: 0,
                scores: [25000; 4],
                tehais,
            })
            .unwrap();

        // Player 0 draws and discards 4p
        state
            .update(&Event::Tsumo {
                actor: 0,
                pai: "4p".parse().unwrap(),
            })
            .unwrap();
        state
            .update(&Event::Dahai {
                actor: 0,
                pai: "4p".parse().unwrap(),
                tsumogiri: true,
            })
            .unwrap();

        // Player 1 draws and discards 1m
        state
            .update(&Event::Tsumo {
                actor: 1,
                pai: "?".parse().unwrap(),
            })
            .unwrap();
        state
            .update(&Event::Dahai {
                actor: 1,
                pai: "1m".parse().unwrap(),
                tsumogiri: false,
            })
            .unwrap();

        // Player 0 daiminkan 1m from player 1
        state
            .update(&Event::Daiminkan {
                actor: 0,
                target: 1,
                pai: "1m".parse().unwrap(),
                consumed: [
                    "1m".parse::<Tile>().unwrap(),
                    "1m".parse::<Tile>().unwrap(),
                    "1m".parse::<Tile>().unwrap(),
                ],
            })
            .unwrap();

        // Rinshan tsumo: draw 2m (completing four 2m)
        state
            .update(&Event::Tsumo {
                actor: 0,
                pai: "2m".parse().unwrap(),
            })
            .unwrap();

        // Player 0 declares ankan on 2m (continuous kan)
        state
            .update(&Event::Ankan {
                actor: 0,
                consumed: [
                    "2m".parse::<Tile>().unwrap(),
                    "2m".parse::<Tile>().unwrap(),
                    "2m".parse::<Tile>().unwrap(),
                    "2m".parse::<Tile>().unwrap(),
                ],
            })
            .unwrap();

        // Dora from the daiminkan's pending need_new_dora_at_discard
        // (taken and revealed immediately by Ankan's continuous kan handler)
        state
            .update(&Event::Dora {
                dora_marker: "3p".parse().unwrap(),
            })
            .unwrap();

        // Dora from the ankan itself (also immediate)
        state
            .update(&Event::Dora {
                dora_marker: "5p".parse().unwrap(),
            })
            .unwrap();

        // Check context at this point
        let events = state.event_history();
        let replayed = replay_player_states(&state).unwrap();
        let midgame = derive_midgame_context(events, replayed, state.tiles_left());

        assert_eq!(midgame.kans, 2, "should have 2 kans (daiminkan + ankan)");
        assert!(
            midgame.deal_from_rinshan.is_some(),
            "should be flagged for second rinshan draw"
        );
        // After Ankan: need_new_dora_at_discard was consumed (taken by Ankan handler).
        // Ankan's own dora is immediate. Neither sets need_new_dora_at_discard.
        assert!(
            midgame.need_new_dora_at_discard.is_none(),
            "need_new_dora_at_discard should be cleared (consumed by Ankan)"
        );
        // Ankan does NOT set need_new_dora_at_tsumo
        assert!(
            midgame.need_new_dora_at_tsumo.is_none(),
            "Ankan should NOT promote to need_new_dora_at_tsumo"
        );
        assert_eq!(
            midgame.dora_indicators_full.len(),
            3,
            "should have 3 dora indicators: initial + daiminkan's + ankan's"
        );
    }

    /// Continuous kan: Daiminkan followed by Kakan.
    ///
    /// The first Daiminkan sets need_new_dora_at_discard. When a Kakan
    /// follows while need_new_dora_at_discard is pending, it promotes
    /// to need_new_dora_at_tsumo.
    #[test]
    fn continuous_kan_daiminkan_then_kakan() {
        let mut state = PlayerState::new(0);
        state.set_record_events(true);

        let bakaze: Tile = "E".parse().unwrap();
        let dora_marker: Tile = "9s".parse().unwrap();
        // Player 0 has three 1m (for daiminkan) and a pon of 3m
        // We'll set up a pon first, then kakan later
        let tehais = [
            [
                "1m", "1m", "1m", "3m", "3m", "5m", "6m", "7m", "8m", "1p", "2p", "3p", "4p",
            ]
            .map(|s| s.parse::<Tile>().unwrap()),
            ["?"; 13].map(|s| s.parse::<Tile>().unwrap()),
            ["?"; 13].map(|s| s.parse::<Tile>().unwrap()),
            ["?"; 13].map(|s| s.parse::<Tile>().unwrap()),
        ];

        state
            .update(&Event::StartKyoku {
                bakaze,
                dora_marker,
                kyoku: 1,
                honba: 0,
                kyotaku: 0,
                oya: 0,
                scores: [25000; 4],
                tehais,
            })
            .unwrap();

        // Player 0 draws and discards 4p
        state
            .update(&Event::Tsumo {
                actor: 0,
                pai: "4p".parse().unwrap(),
            })
            .unwrap();
        state
            .update(&Event::Dahai {
                actor: 0,
                pai: "4p".parse().unwrap(),
                tsumogiri: true,
            })
            .unwrap();

        // Player 1 discards 3m
        state
            .update(&Event::Tsumo {
                actor: 1,
                pai: "?".parse().unwrap(),
            })
            .unwrap();
        state
            .update(&Event::Dahai {
                actor: 1,
                pai: "3m".parse().unwrap(),
                tsumogiri: false,
            })
            .unwrap();

        // Player 0 pons 3m
        state
            .update(&Event::Pon {
                actor: 0,
                target: 1,
                pai: "3m".parse().unwrap(),
                consumed: [
                    "3m".parse::<Tile>().unwrap(),
                    "3m".parse::<Tile>().unwrap(),
                ],
            })
            .unwrap();

        // Discard after pon
        state
            .update(&Event::Dahai {
                actor: 0,
                pai: "3p".parse().unwrap(),
                tsumogiri: false,
            })
            .unwrap();

        // Player 2 discards 1m
        state
            .update(&Event::Tsumo {
                actor: 2,
                pai: "?".parse().unwrap(),
            })
            .unwrap();
        state
            .update(&Event::Dahai {
                actor: 2,
                pai: "1m".parse().unwrap(),
                tsumogiri: false,
            })
            .unwrap();

        // Player 0 daiminkan 1m (first kan)
        state
            .update(&Event::Daiminkan {
                actor: 0,
                target: 2,
                pai: "1m".parse().unwrap(),
                consumed: [
                    "1m".parse::<Tile>().unwrap(),
                    "1m".parse::<Tile>().unwrap(),
                    "1m".parse::<Tile>().unwrap(),
                ],
            })
            .unwrap();

        // Rinshan tsumo: draw 3m (can kakan the pon of 3m)
        state
            .update(&Event::Tsumo {
                actor: 0,
                pai: "3m".parse().unwrap(),
            })
            .unwrap();

        // Kakan 3m (continuous kan: daiminkan pending + kakan)
        state
            .update(&Event::Kakan {
                actor: 0,
                pai: "3m".parse().unwrap(),
                consumed: [
                    "3m".parse::<Tile>().unwrap(),
                    "3m".parse::<Tile>().unwrap(),
                    "3m".parse::<Tile>().unwrap(),
                ],
            })
            .unwrap();

        // Check context BEFORE the dora events
        let events = state.event_history();
        let replayed = replay_player_states(&state).unwrap();
        let midgame = derive_midgame_context(events, replayed, state.tiles_left());

        assert_eq!(midgame.kans, 2, "should have 2 kans (daiminkan + kakan)");
        assert!(
            midgame.deal_from_rinshan.is_some(),
            "should be flagged for rinshan draw"
        );
        // Kakan while need_new_dora_at_discard is pending -> promotes to tsumo
        assert!(
            midgame.need_new_dora_at_tsumo.is_some(),
            "continuous kan (daiminkan -> kakan) should promote to need_new_dora_at_tsumo"
        );
        // Kakan also sets its own need_new_dora_at_discard
        assert!(
            midgame.need_new_dora_at_discard.is_some(),
            "kakan should set its own need_new_dora_at_discard"
        );
    }

    /// Verify the mortal.rs replay-once optimization produces the same results.
    ///
    /// simulate_action_rollout_from_replayed (used by mortal.rs BUG-2 fix)
    /// should produce the same results as simulate_action_rollout (one-shot).
    #[test]
    fn mortal_replay_once_equivalence() {
        let state = setup_basic_game();
        let config = ParticleConfig::new(5);
        let mut rng = ChaCha12Rng::seed_from_u64(42);

        let (particles, _attempts) = generate_particles(&state, &config, &mut rng).unwrap();
        let replayed = replay_player_states(&state).unwrap();

        for (i, particle) in particles.iter().enumerate() {
            // One-shot (replays internally each call)
            let result_oneshot = simulate_action_rollout(&state, particle, 0).unwrap();
            // Replay-once
            let result_replayed =
                simulate_action_rollout_from_replayed(&state, &replayed, particle, 0).unwrap();

            assert_eq!(
                result_oneshot.deltas, result_replayed.deltas,
                "particle {i}: replay-once should match one-shot"
            );
            assert_eq!(result_oneshot.scores, result_replayed.scores);
        }
    }

    /// Verify replay-once equivalence with a more complex game state.
    #[test]
    fn mortal_replay_once_equivalence_deep() {
        let state = setup_deep_game();
        let config = ParticleConfig::new(5);
        let mut rng = ChaCha12Rng::seed_from_u64(42);

        let (particles, _attempts) = generate_particles(&state, &config, &mut rng).unwrap();
        let replayed = replay_player_states(&state).unwrap();
        let base = derive_midgame_context_base(state.event_history());

        for (i, particle) in particles.iter().enumerate() {
            let result_oneshot = simulate_action_rollout(&state, particle, 0).unwrap();
            let result_with_base =
                simulate_action_rollout_with_base(&state, &replayed, particle, 0, &base).unwrap();

            assert_eq!(
                result_oneshot.deltas, result_with_base.deltas,
                "particle {i}: with_base deep game should match one-shot"
            );
        }
    }

    /// Verify replay-once equivalence with melds in history.
    #[test]
    fn mortal_replay_once_equivalence_with_chi() {
        let state = setup_game_with_chi();
        let config = ParticleConfig::new(5);
        let mut rng = ChaCha12Rng::seed_from_u64(42);

        let (particles, _attempts) = generate_particles(&state, &config, &mut rng).unwrap();
        let replayed = replay_player_states(&state).unwrap();
        let base = derive_midgame_context_base(state.event_history());

        // Action: discard F (the tsumo tile)
        let f_tile: Tile = "F".parse().unwrap();
        let action = f_tile.as_usize();

        for (i, particle) in particles.iter().enumerate() {
            let result_oneshot = simulate_action_rollout(&state, particle, action).unwrap();
            let result_with_base =
                simulate_action_rollout_with_base(&state, &replayed, particle, action, &base)
                    .unwrap();

            assert_eq!(
                result_oneshot.deltas, result_with_base.deltas,
                "particle {i}: with_base chi-game should match one-shot"
            );
        }
    }

    /// Verify that the base context for an Ankan game is correct
    /// when used via the with_base path.
    #[test]
    fn ankan_with_base_rollout_completes() {
        let mut state = PlayerState::new(0);
        state.set_record_events(true);

        let bakaze: Tile = "E".parse().unwrap();
        let dora_marker: Tile = "9s".parse().unwrap();
        let tehais = [
            [
                "1m", "1m", "1m", "3m", "4m", "5m", "6m", "7m", "8m", "1p", "2p", "3p", "4p",
            ]
            .map(|s| s.parse::<Tile>().unwrap()),
            ["?"; 13].map(|s| s.parse::<Tile>().unwrap()),
            ["?"; 13].map(|s| s.parse::<Tile>().unwrap()),
            ["?"; 13].map(|s| s.parse::<Tile>().unwrap()),
        ];

        state
            .update(&Event::StartKyoku {
                bakaze,
                dora_marker,
                kyoku: 1,
                honba: 0,
                kyotaku: 0,
                oya: 0,
                scores: [25000; 4],
                tehais,
            })
            .unwrap();

        state
            .update(&Event::Tsumo {
                actor: 0,
                pai: "1m".parse().unwrap(),
            })
            .unwrap();
        state
            .update(&Event::Ankan {
                actor: 0,
                consumed: [
                    "1m".parse::<Tile>().unwrap(),
                    "1m".parse::<Tile>().unwrap(),
                    "1m".parse::<Tile>().unwrap(),
                    "1m".parse::<Tile>().unwrap(),
                ],
            })
            .unwrap();
        state
            .update(&Event::Dora {
                dora_marker: "2p".parse().unwrap(),
            })
            .unwrap();
        state
            .update(&Event::Tsumo {
                actor: 0,
                pai: "9m".parse().unwrap(),
            })
            .unwrap();
        state
            .update(&Event::Dahai {
                actor: 0,
                pai: "9m".parse().unwrap(),
                tsumogiri: true,
            })
            .unwrap();

        // Other players
        for actor in [1_u8, 2, 3] {
            state
                .update(&Event::Tsumo {
                    actor,
                    pai: "?".parse().unwrap(),
                })
                .unwrap();
            state
                .update(&Event::Dahai {
                    actor,
                    pai: "N".parse().unwrap(),
                    tsumogiri: true,
                })
                .unwrap();
        }

        state
            .update(&Event::Tsumo {
                actor: 0,
                pai: "5p".parse().unwrap(),
            })
            .unwrap();

        let config = ParticleConfig::new(5);
        let mut rng = ChaCha12Rng::seed_from_u64(42);
        let (particles, _attempts) = generate_particles(&state, &config, &mut rng).unwrap();
        let replayed = replay_player_states(&state).unwrap();
        let base = derive_midgame_context_base(state.event_history());

        // After ankan of 1m, player's hand is: 3m,4m,5m,6m,7m,8m,1p,2p,3p,4p + tsumo 5p
        // Discard 5p (action 13 = tile id 13 = 5p) which is the tsumo tile
        let action = 13_usize; // 5p

        for (i, particle) in particles.iter().enumerate() {
            // Verify with_base and from_replayed produce same results
            let result_replayed =
                simulate_action_rollout_from_replayed(&state, &replayed, particle, action)
                    .unwrap();
            let result_base =
                simulate_action_rollout_with_base(&state, &replayed, particle, action, &base)
                    .unwrap();

            assert_eq!(
                result_replayed.deltas, result_base.deltas,
                "particle {i}: ankan game with_base should match from_replayed"
            );
        }
    }

    // ---- Task #3: Riichi discard called by opponent ----

    /// Verify the event-ordering invariant: when player 0 declares riichi
    /// and an opponent chi/pons the riichi discard, ReachAccepted is
    /// broadcast before the Chi/Pon in the event stream. This means
    /// accepted_riichis is incremented by the ReachAccepted event, and the
    /// Chi/Pon handler does NOT need to consume riichi_to_be_accepted again.
    #[test]
    fn riichi_discard_called_by_opponent() {
        let mut state = PlayerState::new(0);
        state.set_record_events(true);

        let bakaze: Tile = "E".parse().unwrap();
        let dora_marker: Tile = "9s".parse().unwrap();
        let tehais = [
            // Player 0: tenpai hand (1-9m, 1-3p, E), waiting on 4p or E
            [
                "1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "1p", "2p", "3p", "E",
            ]
            .map(|s| s.parse::<Tile>().unwrap()),
            ["?"; 13].map(|s| s.parse::<Tile>().unwrap()),
            ["?"; 13].map(|s| s.parse::<Tile>().unwrap()),
            ["?"; 13].map(|s| s.parse::<Tile>().unwrap()),
        ];

        state
            .update(&Event::StartKyoku {
                bakaze,
                dora_marker,
                kyoku: 1,
                honba: 0,
                kyotaku: 0,
                oya: 0,
                scores: [25000; 4],
                tehais,
            })
            .unwrap();

        // Tsumo E
        state
            .update(&Event::Tsumo {
                actor: 0,
                pai: "E".parse().unwrap(),
            })
            .unwrap();

        // Declare riichi
        state.update(&Event::Reach { actor: 0 }).unwrap();

        // Discard E (the riichi discard)
        state
            .update(&Event::Dahai {
                actor: 0,
                pai: "E".parse().unwrap(),
                tsumogiri: false,
            })
            .unwrap();

        // ReachAccepted comes BEFORE the chi/pon in the event stream
        state.update(&Event::ReachAccepted { actor: 0 }).unwrap();

        // Player 1 pons the E discard
        state
            .update(&Event::Pon {
                actor: 1,
                target: 0,
                pai: "E".parse().unwrap(),
                consumed: [
                    "E".parse::<Tile>().unwrap(),
                    "E".parse::<Tile>().unwrap(),
                ],
            })
            .unwrap();

        // Player 1 discards after pon
        state
            .update(&Event::Dahai {
                actor: 1,
                pai: "N".parse().unwrap(),
                tsumogiri: false,
            })
            .unwrap();

        // Continue play so we have a valid state for midgame
        for actor in [2_u8, 3] {
            state
                .update(&Event::Tsumo {
                    actor,
                    pai: "?".parse().unwrap(),
                })
                .unwrap();
            state
                .update(&Event::Dahai {
                    actor,
                    pai: "S".parse().unwrap(),
                    tsumogiri: true,
                })
                .unwrap();
        }

        // Our next tsumo
        state
            .update(&Event::Tsumo {
                actor: 0,
                pai: "W".parse().unwrap(),
            })
            .unwrap();

        // Verify midgame context
        let events = state.event_history();
        let replayed = replay_player_states(&state).unwrap();
        let midgame = derive_midgame_context(events, replayed, state.tiles_left());

        assert_eq!(
            midgame.accepted_riichis, 1,
            "should have exactly 1 accepted riichi (from ReachAccepted event)"
        );
        assert!(
            midgame.riichi_to_be_accepted.is_none(),
            "no pending riichi (already accepted before pon)"
        );

        // Verify rollout completes
        let config = ParticleConfig::new(5);
        let mut rng = ChaCha12Rng::seed_from_u64(42);
        let (particles, _attempts) = generate_particles(&state, &config, &mut rng).unwrap();
        assert!(!particles.is_empty());

        for (i, particle) in particles.iter().enumerate() {
            let result = simulate_particle(&state, particle);
            assert!(
                result.is_ok(),
                "riichi-called-by-opponent particle {i} failed: {:?}",
                result.err()
            );
        }
    }

    // ---- Task #4: Kakan and daiminkan full rollout tests ----

    /// Set up a game with a kakan: player 0 pons 1m, later draws 1m and
    /// declares kakan.
    #[test]
    fn midgame_kakan_rollout() {
        let mut state = PlayerState::new(0);
        state.set_record_events(true);

        let bakaze: Tile = "E".parse().unwrap();
        let dora_marker: Tile = "9s".parse().unwrap();
        // Player 0 has a pair of 1m (will pon a third, then kakan with fourth)
        let tehais = [
            [
                "1m", "1m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "1p", "2p", "3p", "4p",
            ]
            .map(|s| s.parse::<Tile>().unwrap()),
            ["?"; 13].map(|s| s.parse::<Tile>().unwrap()),
            ["?"; 13].map(|s| s.parse::<Tile>().unwrap()),
            ["?"; 13].map(|s| s.parse::<Tile>().unwrap()),
        ];

        state
            .update(&Event::StartKyoku {
                bakaze,
                dora_marker,
                kyoku: 1,
                honba: 0,
                kyotaku: 0,
                oya: 0,
                scores: [25000; 4],
                tehais,
            })
            .unwrap();

        // Oya draws and discards
        state
            .update(&Event::Tsumo {
                actor: 0,
                pai: "5p".parse().unwrap(),
            })
            .unwrap();
        state
            .update(&Event::Dahai {
                actor: 0,
                pai: "5p".parse().unwrap(),
                tsumogiri: true,
            })
            .unwrap();

        // Player 1 discards 1m
        state
            .update(&Event::Tsumo {
                actor: 1,
                pai: "?".parse().unwrap(),
            })
            .unwrap();
        state
            .update(&Event::Dahai {
                actor: 1,
                pai: "1m".parse().unwrap(),
                tsumogiri: false,
            })
            .unwrap();

        // Player 0 pons 1m
        state
            .update(&Event::Pon {
                actor: 0,
                target: 1,
                pai: "1m".parse().unwrap(),
                consumed: [
                    "1m".parse::<Tile>().unwrap(),
                    "1m".parse::<Tile>().unwrap(),
                ],
            })
            .unwrap();

        // Discard after pon
        state
            .update(&Event::Dahai {
                actor: 0,
                pai: "4p".parse().unwrap(),
                tsumogiri: false,
            })
            .unwrap();

        // Other players
        for actor in [1_u8, 2, 3] {
            state
                .update(&Event::Tsumo {
                    actor,
                    pai: "?".parse().unwrap(),
                })
                .unwrap();
            state
                .update(&Event::Dahai {
                    actor,
                    pai: "N".parse().unwrap(),
                    tsumogiri: true,
                })
                .unwrap();
        }

        // Player 0 draws 1m -> kakan
        state
            .update(&Event::Tsumo {
                actor: 0,
                pai: "1m".parse().unwrap(),
            })
            .unwrap();

        // Kakan: adding 1m to existing pon of 1m
        state
            .update(&Event::Kakan {
                actor: 0,
                pai: "1m".parse().unwrap(),
                consumed: [
                    "1m".parse::<Tile>().unwrap(),
                    "1m".parse::<Tile>().unwrap(),
                    "1m".parse::<Tile>().unwrap(),
                ],
            })
            .unwrap();

        // New dora after kakan (revealed at next discard per timing rules,
        // but the Dora event appears in the log when it does)
        state
            .update(&Event::Dora {
                dora_marker: "3p".parse().unwrap(),
            })
            .unwrap();

        // Rinshan tsumo
        state
            .update(&Event::Tsumo {
                actor: 0,
                pai: "6p".parse().unwrap(),
            })
            .unwrap();

        // Discard after rinshan
        state
            .update(&Event::Dahai {
                actor: 0,
                pai: "6p".parse().unwrap(),
                tsumogiri: true,
            })
            .unwrap();

        // More turns
        for actor in [1_u8, 2, 3] {
            state
                .update(&Event::Tsumo {
                    actor,
                    pai: "?".parse().unwrap(),
                })
                .unwrap();
            state
                .update(&Event::Dahai {
                    actor,
                    pai: "S".parse().unwrap(),
                    tsumogiri: true,
                })
                .unwrap();
        }

        // Our next tsumo
        state
            .update(&Event::Tsumo {
                actor: 0,
                pai: "7p".parse().unwrap(),
            })
            .unwrap();

        // Build and rollout
        let config = ParticleConfig::new(5);
        let mut rng = ChaCha12Rng::seed_from_u64(42);
        let (particles, _attempts) = generate_particles(&state, &config, &mut rng).unwrap();
        assert!(!particles.is_empty());

        let replayed = replay_player_states(&state).unwrap();
        let base = derive_midgame_context_base(state.event_history());

        for (i, particle) in particles.iter().enumerate() {
            let result =
                simulate_action_rollout_with_base(&state, &replayed, particle, 15, &base);
            assert!(
                result.is_ok(),
                "kakan rollout particle {i} failed: {:?}",
                result.err()
            );
        }
    }

    /// Daiminkan: opponent calls daiminkan on our discard, then play continues.
    #[test]
    fn midgame_daiminkan_rollout() {
        let mut state = PlayerState::new(0);
        state.set_record_events(true);

        let bakaze: Tile = "E".parse().unwrap();
        let dora_marker: Tile = "9s".parse().unwrap();
        let tehais = [
            [
                "1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "1p", "2p", "3p", "4p",
            ]
            .map(|s| s.parse::<Tile>().unwrap()),
            ["?"; 13].map(|s| s.parse::<Tile>().unwrap()),
            ["?"; 13].map(|s| s.parse::<Tile>().unwrap()),
            ["?"; 13].map(|s| s.parse::<Tile>().unwrap()),
        ];

        state
            .update(&Event::StartKyoku {
                bakaze,
                dora_marker,
                kyoku: 1,
                honba: 0,
                kyotaku: 0,
                oya: 0,
                scores: [25000; 4],
                tehais,
            })
            .unwrap();

        // Oya draws and discards E
        state
            .update(&Event::Tsumo {
                actor: 0,
                pai: "E".parse().unwrap(),
            })
            .unwrap();
        state
            .update(&Event::Dahai {
                actor: 0,
                pai: "E".parse().unwrap(),
                tsumogiri: true,
            })
            .unwrap();

        // Player 2 daiminkan's E from player 0
        state
            .update(&Event::Daiminkan {
                actor: 2,
                target: 0,
                pai: "E".parse().unwrap(),
                consumed: [
                    "E".parse::<Tile>().unwrap(),
                    "E".parse::<Tile>().unwrap(),
                    "E".parse::<Tile>().unwrap(),
                ],
            })
            .unwrap();

        // Dora reveal after daiminkan (at next discard timing, but event appears)
        state
            .update(&Event::Dora {
                dora_marker: "2p".parse().unwrap(),
            })
            .unwrap();

        // Rinshan tsumo for player 2
        state
            .update(&Event::Tsumo {
                actor: 2,
                pai: "?".parse().unwrap(),
            })
            .unwrap();
        state
            .update(&Event::Dahai {
                actor: 2,
                pai: "W".parse().unwrap(),
                tsumogiri: true,
            })
            .unwrap();

        // Continue with remaining players
        state
            .update(&Event::Tsumo {
                actor: 3,
                pai: "?".parse().unwrap(),
            })
            .unwrap();
        state
            .update(&Event::Dahai {
                actor: 3,
                pai: "F".parse().unwrap(),
                tsumogiri: true,
            })
            .unwrap();

        // Player 0 draws
        state
            .update(&Event::Tsumo {
                actor: 0,
                pai: "5p".parse().unwrap(),
            })
            .unwrap();

        // Build and rollout
        let config = ParticleConfig::new(5);
        let mut rng = ChaCha12Rng::seed_from_u64(42);
        let (particles, _attempts) = generate_particles(&state, &config, &mut rng).unwrap();
        assert!(!particles.is_empty());

        let replayed = replay_player_states(&state).unwrap();
        let base = derive_midgame_context_base(state.event_history());

        // Verify daiminkan was counted
        assert_eq!(base.kans, 1, "should have 1 kan (daiminkan)");

        for (i, particle) in particles.iter().enumerate() {
            let result =
                simulate_action_rollout_with_base(&state, &replayed, particle, 13, &base);
            assert!(
                result.is_ok(),
                "daiminkan rollout particle {i} failed: {:?}",
                result.err()
            );
        }
    }

    // ---- Task #5: Multiple kans / continuous kan rollout ----

    /// Test with 2 kans in the event history: ankan followed by daiminkan.
    /// Exercises dead wall accounting and multiple dora reveals.
    #[test]
    fn midgame_multiple_kans_rollout() {
        let mut state = PlayerState::new(0);
        state.set_record_events(true);

        let bakaze: Tile = "E".parse().unwrap();
        let dora_marker: Tile = "9s".parse().unwrap();
        // Player 0 has three 1m (will draw 4th for ankan)
        let tehais = [
            [
                "1m", "1m", "1m", "4m", "5m", "6m", "7m", "8m", "9m", "1p", "2p", "3p", "4p",
            ]
            .map(|s| s.parse::<Tile>().unwrap()),
            ["?"; 13].map(|s| s.parse::<Tile>().unwrap()),
            ["?"; 13].map(|s| s.parse::<Tile>().unwrap()),
            ["?"; 13].map(|s| s.parse::<Tile>().unwrap()),
        ];

        state
            .update(&Event::StartKyoku {
                bakaze,
                dora_marker,
                kyoku: 1,
                honba: 0,
                kyotaku: 0,
                oya: 0,
                scores: [25000; 4],
                tehais,
            })
            .unwrap();

        // Player 0 draws 1m -> ankan
        state
            .update(&Event::Tsumo {
                actor: 0,
                pai: "1m".parse().unwrap(),
            })
            .unwrap();
        state
            .update(&Event::Ankan {
                actor: 0,
                consumed: [
                    "1m".parse::<Tile>().unwrap(),
                    "1m".parse::<Tile>().unwrap(),
                    "1m".parse::<Tile>().unwrap(),
                    "1m".parse::<Tile>().unwrap(),
                ],
            })
            .unwrap();

        // Dora reveal for ankan (immediate)
        state
            .update(&Event::Dora {
                dora_marker: "2p".parse().unwrap(),
            })
            .unwrap();

        // Rinshan tsumo
        state
            .update(&Event::Tsumo {
                actor: 0,
                pai: "5p".parse().unwrap(),
            })
            .unwrap();

        // Discard
        state
            .update(&Event::Dahai {
                actor: 0,
                pai: "5p".parse().unwrap(),
                tsumogiri: true,
            })
            .unwrap();

        // Player 2 calls daiminkan on the 5p discard (has three 5p)
        // (Note: daiminkan must be on the same tile just discarded)
        // Actually 5p was just discarded by player 0, player 2 can daiminkan.
        state
            .update(&Event::Daiminkan {
                actor: 2,
                target: 0,
                pai: "5p".parse().unwrap(),
                consumed: [
                    "5p".parse::<Tile>().unwrap(),
                    "5p".parse::<Tile>().unwrap(),
                    "5p".parse::<Tile>().unwrap(),
                ],
            })
            .unwrap();

        // Dora reveal for daiminkan (at next discard timing)
        state
            .update(&Event::Dora {
                dora_marker: "3s".parse().unwrap(),
            })
            .unwrap();

        // Rinshan tsumo for player 2
        state
            .update(&Event::Tsumo {
                actor: 2,
                pai: "?".parse().unwrap(),
            })
            .unwrap();
        state
            .update(&Event::Dahai {
                actor: 2,
                pai: "C".parse().unwrap(),
                tsumogiri: true,
            })
            .unwrap();

        // Continue play
        state
            .update(&Event::Tsumo {
                actor: 3,
                pai: "?".parse().unwrap(),
            })
            .unwrap();
        state
            .update(&Event::Dahai {
                actor: 3,
                pai: "F".parse().unwrap(),
                tsumogiri: true,
            })
            .unwrap();

        // Our next tsumo
        state
            .update(&Event::Tsumo {
                actor: 0,
                pai: "6p".parse().unwrap(),
            })
            .unwrap();

        // Verify context
        let base = derive_midgame_context_base(state.event_history());
        assert_eq!(base.kans, 2, "should have 2 kans");
        assert_eq!(
            base.dora_indicators_full.len(),
            3,
            "should have 3 dora indicators (initial + 2 kan reveals)"
        );

        // Build and rollout
        let config = ParticleConfig::new(5);
        let mut rng = ChaCha12Rng::seed_from_u64(42);
        let (particles, _attempts) = generate_particles(&state, &config, &mut rng).unwrap();
        assert!(!particles.is_empty());

        let replayed = replay_player_states(&state).unwrap();

        for (i, particle) in particles.iter().enumerate() {
            let result = build_midgame_board_state_with_base(&state, &replayed, particle, &base);
            assert!(
                result.is_ok(),
                "multi-kan board build particle {i} failed: {:?}",
                result.err()
            );

            let bs = result.unwrap();
            let initial_scores = bs.board.scores;
            let rollout = run_rollout(bs, initial_scores, None, None, None);
            assert!(
                rollout.is_ok(),
                "multi-kan rollout particle {i} failed: {:?}",
                rollout.err()
            );
        }
    }

    // ---- Task #6: Pending riichi_to_be_accepted at search time ----

    /// Verify behavior when event history ends after Reach + Dahai but
    /// BEFORE ReachAccepted. This happens when the search is triggered
    /// at the moment an opponent must react to the riichi discard.
    #[test]
    fn pending_riichi_at_search_time() {
        let mut state = PlayerState::new(0);
        state.set_record_events(true);

        let bakaze: Tile = "E".parse().unwrap();
        let dora_marker: Tile = "9s".parse().unwrap();
        let tehais = [
            [
                "1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "1p", "2p", "3p", "E",
            ]
            .map(|s| s.parse::<Tile>().unwrap()),
            ["?"; 13].map(|s| s.parse::<Tile>().unwrap()),
            ["?"; 13].map(|s| s.parse::<Tile>().unwrap()),
            ["?"; 13].map(|s| s.parse::<Tile>().unwrap()),
        ];

        state
            .update(&Event::StartKyoku {
                bakaze,
                dora_marker,
                kyoku: 1,
                honba: 0,
                kyotaku: 0,
                oya: 0,
                scores: [25000; 4],
                tehais,
            })
            .unwrap();

        // Draw E
        state
            .update(&Event::Tsumo {
                actor: 0,
                pai: "E".parse().unwrap(),
            })
            .unwrap();

        // Declare riichi
        state.update(&Event::Reach { actor: 0 }).unwrap();

        // Discard E (riichi discard)
        state
            .update(&Event::Dahai {
                actor: 0,
                pai: "E".parse().unwrap(),
                tsumogiri: false,
            })
            .unwrap();

        // EVENT HISTORY ENDS HERE - before ReachAccepted
        // This simulates searching at the moment opponents react to the
        // riichi discard.

        // Verify the base context has pending riichi
        let base = derive_midgame_context_base(state.event_history());
        assert_eq!(
            base.accepted_riichis, 0,
            "no riichi should be accepted yet"
        );
        assert_eq!(
            base.riichi_to_be_accepted,
            Some(0),
            "riichi should be pending for player 0"
        );

        // Verify rollout completes (BoardState should accept the riichi
        // on the next step when it processes the draw)
        let config = ParticleConfig::new(5);
        let mut rng = ChaCha12Rng::seed_from_u64(42);
        let (particles, _attempts) = generate_particles(&state, &config, &mut rng).unwrap();
        assert!(!particles.is_empty());

        let replayed = replay_player_states(&state).unwrap();

        for (i, particle) in particles.iter().enumerate() {
            let result = build_midgame_board_state_with_base(&state, &replayed, particle, &base);
            assert!(
                result.is_ok(),
                "pending riichi board build particle {i} failed: {:?}",
                result.err()
            );

            let bs = result.unwrap();
            let initial_scores = bs.board.scores;
            let rollout = run_rollout(bs, initial_scores, None, None, None);
            assert!(
                rollout.is_ok(),
                "pending riichi rollout particle {i} failed: {:?}",
                rollout.err()
            );
        }
    }
}
