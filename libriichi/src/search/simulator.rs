use crate::arena::{Board, Poll};
use crate::mjai::{Event, EventExt};
use crate::state::PlayerState;
use crate::tile::Tile;
use crate::{must_tile, tu8};

use super::particle::Particle;

use anyhow::{Context, Result, bail, ensure};
use pyo3::prelude::*;

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
            "RolloutResult(scores={:?}, deltas={:?}, hora={}, abort={})",
            self.scores, self.deltas, self.has_hora, self.has_abortive_ryukyoku,
        )
    }
}

/// Construct a Board from the current game state and a particle sample.
///
/// The Board represents a hypothetical kyoku where:
/// - Each player starts with their current hand tiles
/// - The wall contains the particle's sampled remaining tiles
/// - Dora indicators and dead wall are assigned from unused tiles
///
/// This is a "snapshot" Board, NOT a replay from the beginning.
/// It starts a fresh kyoku with the given tile distribution.
pub fn build_board_from_particle(
    state: &PlayerState,
    particle: &Particle,
) -> Result<Board> {
    let player_id = state.player_id();

    // PlayerState.kyoku() is 0-indexed within-wind (0-3).
    // Board.kyoku is absolute (0-7): 0-3 = East 1-4, 4-7 = South 1-4.
    // Reconstruct absolute kyoku from bakaze + within-wind kyoku.
    let bakaze = state.bakaze();
    let kyoku = (bakaze.as_u8() - crate::tu8!(E)) * 4 + state.kyoku();

    // Oya is absolute seat = kyoku % 4
    // Un-rotate scores from relative (scores[0]=self) to absolute
    let rel_scores = state.scores();
    let mut abs_scores = [0_i32; 4];
    for i in 0..4 {
        abs_scores[(player_id as usize + i) % 4] = rel_scores[i];
    }

    // Build haipai: player's hand + opponent hands from particle
    let mut haipai = [[Tile::default(); 13]; 4];

    // Our hand: convert tehai counts back to tiles (up to 13 tiles)
    let tehai = state.tehai();
    let akas = state.akas_in_hand();
    let mut our_tiles = Vec::with_capacity(14);
    for (tid, &count) in tehai.iter().enumerate() {
        if count == 0 {
            continue;
        }
        // Check if this tile type has an aka variant we hold
        let aka_idx = match tid {
            x if x == crate::tuz!(5m) => Some(0),
            x if x == crate::tuz!(5p) => Some(1),
            x if x == crate::tuz!(5s) => Some(2),
            _ => None,
        };
        if let Some(ai) = aka_idx {
            if akas[ai] {
                our_tiles.push(must_tile!(tu8!(5mr) + ai as u8));
                for _ in 0..count - 1 {
                    our_tiles.push(must_tile!(tid));
                }
            } else {
                for _ in 0..count {
                    our_tiles.push(must_tile!(tid));
                }
            }
        } else {
            for _ in 0..count {
                our_tiles.push(must_tile!(tid));
            }
        }
    }

    // For haipai we need exactly 13 tiles. If we have 14 (just drew),
    // the extra tile goes to the front of yama (will be drawn as first tsumo).
    let our_tsumo_tile = if our_tiles.len() > 13 {
        Some(our_tiles.pop().unwrap())
    } else {
        None
    };

    ensure!(
        our_tiles.len() == 13,
        "expected 13 tiles for our haipai, got {}",
        our_tiles.len(),
    );
    haipai[player_id as usize].copy_from_slice(&our_tiles);

    // Opponent hands from particle -> haipai
    for opp in 0..3 {
        let abs_seat = (player_id as usize + opp + 1) % 4;
        let hand = &particle.opponent_hands[opp];

        if hand.len() >= 13 {
            haipai[abs_seat].copy_from_slice(&hand[..13]);
        } else {
            for (i, &tile) in hand.iter().enumerate() {
                haipai[abs_seat][i] = tile;
            }
            // Phase 1 limitation: opponent has fewer than 13 tiles (has melds).
            // We pad with dummy tiles to satisfy Board's 13-tile haipai requirement.
            // This means the simulation starts a fresh kyoku that doesn't preserve
            // the actual meld state. Phase 2 will need proper mid-game continuation.
            for slot in &mut haipai[abs_seat][hand.len()..13] {
                *slot = must_tile!(0_u8); // dummy padding
            }
        }
    }

    // Build yama (wall). BoardState expects exactly 70 tiles.
    //
    // The particle has tiles_left wall tiles + possibly our tsumo tile.
    // At game start that's 69 + 1 = 70, but mid-game we have fewer because
    // tiles were drawn (and discarded) before our current state.
    //
    // To reach 70, we need padding tiles. These represent the draws that
    // already happened in the real game. We reconstruct them from the
    // "consumed" tiles: tiles that are in tiles_seen but NOT in our hand
    // and NOT in the particle (opponent hands + wall).
    //
    // Consumed = tiles_seen - our_hand (these are discards, open melds,
    // and dora indicators that we've witnessed).
    let oya = (kyoku % 4) as usize;
    let our_offset = (player_id as usize + 4 - oya) % 4;

    let tiles_in_particle = particle.wall.len();
    let tiles_with_tsumo = tiles_in_particle + our_tsumo_tile.is_some() as usize;
    let padding_needed = 70_usize.saturating_sub(tiles_with_tsumo);

    // Compute consumed tiles: tiles we've observed that are NOT in our hand.
    // These are discards, open meld tiles, and dora indicators - all visible
    // and NOT in the particle (which only contains hidden tiles).
    // We can safely reuse these as padding for the yama.
    let tiles_seen = state.tiles_seen();
    let tehai = state.tehai();
    let mut consumed_counts = [0_u8; 34];
    for (tid, count) in consumed_counts.iter_mut().enumerate() {
        *count = tiles_seen[tid].saturating_sub(tehai[tid]);
    }

    // Expand consumed counts into actual tiles for padding
    let mut padding_tiles = Vec::with_capacity(padding_needed);
    if padding_needed > 0 {
        'outer: for (tid, &count) in consumed_counts.iter().enumerate() {
            for _ in 0..count {
                padding_tiles.push(must_tile!(tid));
                if padding_tiles.len() >= padding_needed {
                    break 'outer;
                }
            }
        }
    }

    // Build yama: index 0 = drawn last, end = drawn first (pop from back).
    let mut yama = Vec::with_capacity(70);

    // Future wall tiles (drawn after current state, lowest indices).
    // particle.wall is in draw order, reverse for pop-from-back.
    yama.extend(particle.wall.iter().copied().rev());

    // Padding tiles (drawn before current state, higher indices = popped first).
    yama.extend(padding_tiles.iter().copied());

    // Insert our tsumo tile at the correct yama position for our turn.
    // Draw order: oya (0th pop), oya+1 (1st pop), ...
    // Our turn is the our_offset-th pop.
    // In a yama of length N, the k-th pop is at index N-1-k.
    if let Some(tile) = our_tsumo_tile {
        let insert_idx = yama.len() - our_offset;
        yama.insert(insert_idx, tile);
    }

    ensure!(
        yama.len() == 70,
        "yama should have 70 tiles, got {} (wall={}, tsumo={}, padding={}/{})",
        yama.len(),
        tiles_in_particle,
        our_tsumo_tile.is_some() as usize,
        padding_tiles.len(),
        padding_needed,
    );

    // Dead wall: rinshan (4 tiles) + dora indicators (5) + ura indicators (5) = 14
    //
    // The particle's dead_wall contains the unseen dead wall tiles
    // (14 - num_revealed_dora_indicators). Combined with the actual revealed
    // dora indicators from the game state, we have all 14 tiles.
    //
    // Layout: dora_indicators = [revealed..., unseen_dora...]
    //         rinshan = [unseen...]
    //         ura_indicators = [unseen...]
    let revealed_dora = state.dora_indicators();
    let num_revealed = revealed_dora.len();
    let unseen = &particle.dead_wall;

    // Split unseen dead wall: unrevealed dora, then rinshan, then ura
    let unrevealed_dora_count = 5 - num_revealed;
    let fallback = must_tile!(0_u8); // only used if unseen is unexpectedly short

    let mut dora_indicators = Vec::with_capacity(5);
    dora_indicators.extend_from_slice(revealed_dora);
    for i in 0..unrevealed_dora_count {
        dora_indicators.push(unseen.get(i).copied().unwrap_or(fallback));
    }

    let mut rinshan = Vec::with_capacity(4);
    for i in 0..4 {
        rinshan.push(unseen.get(unrevealed_dora_count + i).copied().unwrap_or(fallback));
    }

    let mut ura_indicators = Vec::with_capacity(5);
    for i in 0..5 {
        ura_indicators.push(
            unseen
                .get(unrevealed_dora_count + 4 + i)
                .copied()
                .unwrap_or(fallback),
        );
    }

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
        // Post-chi/pon: no tsumo tile, pick first available from hand
        let tehai = state.tehai();
        for (i, &count) in tehai.iter().enumerate() {
            if count > 0 {
                return Event::Dahai {
                    actor: seat,
                    pai: must_tile!(i),
                    tsumogiri: false,
                };
            }
        }
    }
    // Pass on all calls
    Event::None
}

/// Run a rollout where we inject a specific action at the first decision point
/// for `player_id`, then use tsumogiri for all subsequent decisions.
///
/// The board starts a fresh kyoku via `build_board_from_particle`. Between
/// StartKyoku and our first decision, other players may act (tsumo/discard).
/// We only inject `action` at the correct decision point matching the action
/// type (can_discard for discard actions, can_chi for chi, etc.).
///
/// After our action is injected, the rest of the game runs with tsumogiri.
pub fn simulate_action_rollout(
    state: &PlayerState,
    particle: &Particle,
    action: usize,
) -> Result<RolloutResult> {
    let player_id = state.player_id();
    let board = build_board_from_particle(state, particle)?;
    let initial_scores = board.scores;
    let mut board_state = board.into_state();

    let mut reactions: [EventExt; 4] = Default::default();
    let mut action_injected = false;

    loop {
        match board_state.poll(reactions)? {
            Poll::InGame => {
                reactions = Default::default();
                let ctx = board_state.agent_context();

                for (seat, ps) in ctx.player_states.iter().enumerate() {
                    let cans = ps.last_cans();
                    if !cans.can_act() {
                        continue;
                    }

                    let ev = if seat as u8 == player_id && !action_injected {
                        // This is our player's decision point. Check if
                        // the action type matches what's available.
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
                            default_reaction(seat as u8, ps)
                        }
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
                });
            }
        }
    }
}

/// Convenience function: build a board from state + particle, then simulate
/// with a specific action injected for our player.
pub fn simulate_particle_action(
    state: &PlayerState,
    particle: &Particle,
    action: usize,
) -> Result<RolloutResult> {
    simulate_action_rollout(state, particle, action)
}

/// Run a tsumogiri rollout: all players always discard their drawn tile.
///
/// This is the simplest possible rollout for Phase 1 testing.
/// Returns the kyoku result with score information.
pub fn simulate_rollout_tsumogiri(board: Board) -> Result<RolloutResult> {
    let initial_scores = board.scores;
    let mut board_state = board.into_state();

    let mut reactions: [EventExt; 4] = Default::default();

    loop {
        match board_state.poll(reactions)? {
            Poll::InGame => {
                reactions = Default::default();
                let ctx = board_state.agent_context();
                for (seat, ps) in ctx.player_states.iter().enumerate() {
                    let ev = default_reaction(seat as u8, ps);
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
                });
            }
        }
    }
}

/// Convenience function: build a board from state + particle, then simulate.
pub fn simulate_particle(
    state: &PlayerState,
    particle: &Particle,
) -> Result<RolloutResult> {
    let board = build_board_from_particle(state, particle)?;
    simulate_rollout_tsumogiri(board)
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::search::config::ParticleConfig;
    use crate::search::particle::generate_particles;
    use rand::SeedableRng;
    use rand_chacha::ChaCha12Rng;

    fn setup_basic_game() -> PlayerState {
        let mut state = PlayerState::new(0);

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

        // First tsumo
        let tsumo_event = Event::Tsumo {
            actor: 0,
            pai: "5p".parse().unwrap(),
        };
        state.update(&tsumo_event).unwrap();

        state
    }

    #[test]
    fn build_board_basic() {
        let state = setup_basic_game();
        let config = ParticleConfig::new(1);
        let mut rng = ChaCha12Rng::seed_from_u64(42);

        let particles = generate_particles(&state, &config, &mut rng).unwrap();
        assert!(!particles.is_empty());

        let board = build_board_from_particle(&state, &particles[0]).unwrap();

        // Board should have correct metadata
        assert_eq!(board.kyoku, 0); // kyoku is 0-indexed
        assert_eq!(board.honba, 0);
        assert_eq!(board.scores, [25000; 4]);

        // Our haipai should have 13 tiles (the 14th is in yama)
        // Yama should have wall tiles + our tsumo tile
        assert!(board.yama.len() >= 1); // At least our tsumo tile
    }

    #[test]
    fn simulate_tsumogiri_completes() {
        let state = setup_basic_game();
        let config = ParticleConfig::new(5);
        let mut rng = ChaCha12Rng::seed_from_u64(42);

        let particles = generate_particles(&state, &config, &mut rng).unwrap();
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

        let particles = generate_particles(&state, &config, &mut rng).unwrap();

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
        // Verify that the tsumo tile is correctly placed in yama
        // for a non-oya player (player_id=2, oya=0).
        let state = setup_non_oya_game();
        let config = ParticleConfig::new(5);
        let mut rng = ChaCha12Rng::seed_from_u64(42);

        let particles = generate_particles(&state, &config, &mut rng).unwrap();
        assert!(!particles.is_empty());

        let board = build_board_from_particle(&state, &particles[0]).unwrap();

        // oya = kyoku % 4 = 0. Our player is seat 2, offset k=2.
        // The tsumo tile should be the 3rd pop (0-indexed: 2nd) from yama.
        // Verify that the first pop (for oya) is NOT our tsumo tile "S".
        let tsumo_tile: Tile = "S".parse().unwrap();
        let yama_len = board.yama.len();

        // The tile at the end of yama (first pop, for oya) should not be S
        // unless it happens to be a wall tile.
        // More importantly, the tile at position yama_len-1-2 should be S.
        assert_eq!(
            board.yama[yama_len - 1 - 2],
            tsumo_tile,
            "tsumo tile should be at offset 2 from end (3rd pop, for seat 2)"
        );
    }

    #[test]
    fn simulate_non_oya_completes() {
        // Verify that simulation completes for non-oya player.
        let state = setup_non_oya_game();
        let config = ParticleConfig::new(5);
        let mut rng = ChaCha12Rng::seed_from_u64(42);

        let particles = generate_particles(&state, &config, &mut rng).unwrap();
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
        assert!(result.is_err());
    }

    #[test]
    fn action_to_event_riichi_when_unavailable_fails() {
        let state = setup_basic_game();
        // Our hand isn't necessarily tenpai for riichi
        let cans = state.last_cans();
        if !cans.can_riichi {
            let result = action_to_event(37, 0, &state);
            assert!(result.is_err());
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

    // ---- Tests for simulate_action_rollout ----

    #[test]
    fn action_rollout_discard_completes() {
        let state = setup_basic_game();
        let config = ParticleConfig::new(5);
        let mut rng = ChaCha12Rng::seed_from_u64(42);

        let particles = generate_particles(&state, &config, &mut rng).unwrap();
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

        let particles = generate_particles(&state, &config, &mut rng).unwrap();
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

        let particles = generate_particles(&state, &config, &mut rng).unwrap();
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

        let particles = generate_particles(&state, &config, &mut rng).unwrap();
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

        let particles = generate_particles(&state, &config, &mut rng).unwrap();
        assert!(!particles.is_empty());

        let board = build_board_from_particle(&state, &particles[0]).unwrap();

        // South 3: bakaze=S, within-wind kyoku=2 (0-indexed), absolute kyoku=6
        // (S=1)*4 + 2 = 6
        assert_eq!(board.kyoku, 6, "South 3 should have absolute kyoku 6");
        assert_eq!(board.honba, 1);
    }

    #[test]
    fn simulate_south_round_completes() {
        let state = setup_south_round_game();
        let config = ParticleConfig::new(3);
        let mut rng = ChaCha12Rng::seed_from_u64(42);

        let particles = generate_particles(&state, &config, &mut rng).unwrap();
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

        let particles = generate_particles(&state, &config, &mut rng).unwrap();
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
}
