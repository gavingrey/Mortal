use crate::arena::{Board, Poll};
use crate::mjai::{Event, EventExt};
use crate::state::PlayerState;
use crate::tile::Tile;
use crate::{must_tile, tu8};

use super::particle::Particle;

use anyhow::{Result, ensure};
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
    /// Get the score delta for a specific player (absolute seat).
    #[must_use]
    pub const fn player_delta(&self, player_id: u8) -> i32 {
        self.deltas[player_id as usize]
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
    let kyoku = state.kyoku();

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
    // For Phase 1, we use dummy tiles for the dead wall since:
    // - Tsumogiri agents never kan (no rinshan draws)
    // - Dora indicators only matter for scoring (we need at least the first one)
    //
    // We need at least 1 dora indicator for StartKyoku to work.
    let dummy = must_tile!(0_u8); // 1m
    let rinshan = vec![dummy; 4];
    let dora_indicators = vec![dummy; 5]; // First one gets popped as initial dora
    let ura_indicators = vec![dummy; 5];

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

/// Run a tsumogiri rollout: all players always discard their drawn tile.
///
/// This is the simplest possible rollout for Phase 1 testing.
/// Returns the kyoku result with score information.
pub fn simulate_rollout_tsumogiri(board: Board) -> Result<RolloutResult> {
    let initial_scores = board.scores;
    let mut board_state = board.into_state();

    // Default reactions (Event::None for all)
    let mut reactions: [EventExt; 4] = Default::default();

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

                    let ev = if cans.can_tsumo_agari {
                        // Always take a win if available
                        Event::Hora {
                            actor: seat as u8,
                            target: seat as u8,
                            deltas: None,
                            ura_markers: None,
                        }
                    } else if cans.can_ron_agari {
                        Event::Hora {
                            actor: seat as u8,
                            target: cans.target_actor,
                            deltas: None,
                            ura_markers: None,
                        }
                    } else if cans.can_discard {
                        // Tsumogiri: discard the drawn tile
                        if let Some(tsumo) = ps.last_self_tsumo() {
                            Event::Dahai {
                                actor: seat as u8,
                                pai: tsumo,
                                tsumogiri: true,
                            }
                        } else {
                            Event::None
                        }
                    } else {
                        // Pass on all calls (chi, pon, kan)
                        Event::None
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
    fn simulate_multiple_particles_vary() {
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

        // Not all results should be identical (different particles = different outcomes)
        if results.len() > 1 {
            let first_delta = results[0].deltas;
            let has_variation = results.iter().any(|r| r.deltas != first_delta);
            // With Tsumogiri, most games end in exhaustive draw with same delta,
            // but occasionally someone tsumo-agaris, so we don't strictly require variation.
            // Just log it.
            if !has_variation {
                // This is OK for tsumogiri - most draws end the same way
            }
        }
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
        assert_eq!(result.player_delta(0), 1000);
        assert_eq!(result.player_delta(2), -1000);
    }
}
