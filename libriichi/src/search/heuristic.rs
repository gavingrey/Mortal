//! Heuristic rollout policy for search simulations.
//!
//! Provides a `smart_reaction` function that makes better-than-tsumogiri
//! decisions during rollouts, using shanten-based discard selection with
//! ukeire (acceptance count) and safety scoring.

use crate::algo::agari::has_valid_agari;
use crate::algo::shanten;
use crate::mjai::Event;
use crate::state::PlayerState;
use crate::tile::Tile;
use crate::{must_tile, tuz};

use rand::Rng;
use rand_chacha::ChaCha12Rng;
use tinyvec::ArrayVec;

/// Build a safety lookup for all 34 tile types based on observable state.
///
/// Higher values = safer to discard.
/// - 10: genbutsu (in a riichi player's discard pond)
/// -  5: suji partner of a genbutsu tile
/// -  4: 3+ copies already visible (unlikely to be a winning tile)
/// -  0: unknown safety
fn build_safety_array(state: &PlayerState) -> [u8; 34] {
    let mut safety = [0_u8; 34];
    let riichi_accepted = state.riichi_accepted();
    let kawa = state.kawa_overview();
    let tiles_seen = state.tiles_seen();

    // Mark tiles with 3+ copies seen as somewhat safe
    for tid in 0..34_usize {
        if tiles_seen[tid] >= 3 {
            safety[tid] = safety[tid].max(4);
        }
    }

    // For each opponent in riichi, mark genbutsu and suji
    for opp in 1..4_usize {
        if !riichi_accepted[opp] {
            continue;
        }

        for tile in &kawa[opp] {
            let tid = tile.deaka().as_u8() as usize;
            if tid < 34 {
                // Genbutsu: tile discarded by riichi player
                safety[tid] = safety[tid].max(10);

                // Suji: same suit, ±3 from position
                let suit = tid / 9;
                let pos = tid % 9;
                if suit < 3 {
                    // Number tile (man/pin/sou)
                    if pos + 3 < 9 {
                        let partner = suit * 9 + pos + 3;
                        safety[partner] = safety[partner].max(5);
                    }
                    if pos >= 3 {
                        let partner = suit * 9 + pos - 3;
                        safety[partner] = safety[partner].max(5);
                    }
                }
            }
        }
    }

    safety
}

/// Choose a discard tile using shanten + ukeire + safety heuristic.
///
/// Pass 1: Find best (lowest) shanten among all candidate discards.
/// Pass 2: For candidates at best shanten, compute ukeire + safety score.
/// Pass 3: Top-k sampling from the best candidates.
fn choose_discard(state: &PlayerState, rng: &mut ChaCha12Rng) -> Tile {
    let mut tehai = state.tehai();
    let tiles_seen = state.tiles_seen();
    let forbidden = state.forbidden_tiles();
    let safety = build_safety_array(state);

    // Compute len_div3 for shanten calculation
    let n_melds = state.chis().len()
        + state.pons().len()
        + state.minkans().len()
        + state.ankans().len();
    let len_div3 = (4 - n_melds) as u8;

    // Pass 1: Find best shanten among all candidates
    let mut best_shanten = i8::MAX;
    let mut shanten_cache = [i8::MAX; 34];
    for tid in 0..34_usize {
        if tehai[tid] == 0 || forbidden[tid] {
            continue;
        }
        tehai[tid] -= 1;
        let sh = shanten::calc_all(&tehai, len_div3);
        shanten_cache[tid] = sh;
        if sh < best_shanten {
            best_shanten = sh;
        }
        tehai[tid] += 1;
    }

    // Pass 2: For candidates at best shanten, compute ukeire + safety
    #[derive(Default)]
    struct Candidate {
        tid: usize,
        score: i32,
    }
    let mut candidates: ArrayVec<[Candidate; 34]> = ArrayVec::new();

    for tid in 0..34_usize {
        if tehai[tid] == 0 || forbidden[tid] {
            continue;
        }

        let sh = shanten_cache[tid];

        if sh == best_shanten {
            // Need tehai modification for ukeire calculation
            tehai[tid] -= 1;

            // Compute ukeire: how many unseen tiles would reduce shanten?
            let mut ukeire: i32 = 0;
            for t in 0..34_usize {
                let unseen = 4_u8.saturating_sub(tiles_seen[t]);
                if unseen == 0 {
                    continue;
                }
                // Can we add this tile? Need tehai[t] < 4
                if tehai[t] >= 4 {
                    continue;
                }
                tehai[t] += 1;
                let new_sh = shanten::calc_all(&tehai, len_div3);
                tehai[t] -= 1;

                if new_sh < sh {
                    ukeire += unseen as i32;
                }
            }

            // Isolated yaokyuu bonus: if tile is yaokyuu and count was 1
            // (now 0 after removing), give a small bonus for discarding it
            let isolated_bonus = if must_tile!(tid as u8).is_yaokyuu()
                && tehai[tid] == 0
            {
                2_i32
            } else {
                0_i32
            };

            let score = ukeire * 10 + safety[tid] as i32 * 5 + isolated_bonus;
            candidates.push(Candidate { tid, score });

            tehai[tid] += 1;
        }
    }

    // Pass 3: Top-k sampling (k=3)
    candidates.sort_by(|a, b| b.score.cmp(&a.score));
    let k = candidates.len().min(3);
    if k == 0 {
        // Fallback: pick first non-forbidden available tile
        for (tid, (&count, &is_forbidden)) in state.tehai().iter().zip(forbidden.iter()).enumerate() {
            if count > 0 && !is_forbidden {
                return resolve_aka(must_tile!(tid as u8), state);
            }
        }
        // Last resort: pick any available tile (shouldn't happen, but be defensive)
        for (tid, &count) in state.tehai().iter().enumerate() {
            if count > 0 {
                return resolve_aka(must_tile!(tid as u8), state);
            }
        }
        // This really shouldn't happen
        return must_tile!(0_u8);
    }

    let chosen = &candidates[rng.random_range(0..k)];
    let tile = must_tile!(chosen.tid as u8);

    // Convert to aka tile when the player holds the aka version of a 5
    resolve_aka(tile, state)
}

/// If `tile` is a regular 5m/5p/5s and the player holds the aka version,
/// return the aka tile instead. This ensures the Dahai event carries the
/// correct tile identity for dora tracking.
pub(crate) fn resolve_aka(tile: Tile, state: &PlayerState) -> Tile {
    let tid = tile.as_u8() as usize;
    let akas = state.akas_in_hand();
    if tid == tuz!(5m) && akas[0] {
        must_tile!(34_u8) // 5mr
    } else if tid == tuz!(5p) && akas[1] {
        must_tile!(35_u8) // 5pr
    } else if tid == tuz!(5s) && akas[2] {
        must_tile!(36_u8) // 5sr
    } else {
        tile
    }
}

/// Heuristic reaction function for search rollouts.
///
/// Priority:
/// 1. Accept tsumo agari (self-draw win)
/// 2. Accept ron agari (win off discard)
/// 3. Discard using shanten/ukeire/safety heuristic
/// 4. Pass all calls (chi/pon/kan)
pub fn smart_reaction(seat: u8, state: &PlayerState, rng: &mut ChaCha12Rng) -> Event {
    let cans = state.last_cans();

    if !cans.can_act() {
        return Event::None;
    }

    // Accept wins only if the hand pattern is in the agari table.
    // shanten::calc_all() can report shanten=-1 for patterns that
    // AGARI_TABLE doesn't contain, which causes "not a hora hand" errors.
    if cans.can_tsumo_agari && has_valid_agari(&state.tehai()) {
        return Event::Hora {
            actor: seat,
            target: seat,
            deltas: None,
            ura_markers: None,
        };
    }
    if cans.can_ron_agari && has_valid_agari(&state.tehai()) {
        return Event::Hora {
            actor: seat,
            target: cans.target_actor,
            deltas: None,
            ura_markers: None,
        };
    }

    // Riichi players must tsumogiri
    if state.self_riichi_accepted()
        && let Some(tsumo) = state.last_self_tsumo()
    {
        return Event::Dahai {
            actor: seat,
            pai: tsumo,
            tsumogiri: true,
        };
    }

    // Smart discard
    if cans.can_discard {
        let tile = choose_discard(state, rng);
        // Check if the chosen tile matches the tsumo tile
        let tsumogiri = state
            .last_self_tsumo()
            .is_some_and(|tsumo| tsumo == tile);
        return Event::Dahai {
            actor: seat,
            pai: tile,
            tsumogiri,
        };
    }

    // Pass all calls (chi, pon, kan, ryukyoku)
    Event::None
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::mjai::Event;
    use crate::search::config::ParticleConfig;
    use crate::search::particle::generate_particles;
    use crate::search::simulator;
    use crate::search::test_utils::setup_basic_game;
    use crate::state::PlayerState;
    use crate::tile::Tile;

    use rand::SeedableRng;

    /// Helper: set up a game where player has a clear best discard.
    /// Hand: 1m 2m 3m 4p 5p 6p 7s 8s 9s E E E N, tsumo W
    /// Discarding N or W keeps shanten at 0; discarding something from
    /// a complete set would raise shanten.
    fn setup_clear_discard_game() -> PlayerState {
        let mut state = PlayerState::new(0);
        state.set_record_events(true);

        let bakaze: Tile = "E".parse().unwrap();
        let dora_marker: Tile = "1m".parse().unwrap();

        let tehais = [
            [
                "1m", "2m", "3m", "4p", "5p", "6p", "7s", "8s", "9s", "E", "E", "E", "N",
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

        let tsumo_event = Event::Tsumo {
            actor: 0,
            pai: "W".parse().unwrap(),
        };
        state.update(&tsumo_event).unwrap();

        state
    }

    #[test]
    fn test_shanten_optimal_discard() {
        // Hand: 1m 2m 3m | 4p 5p 6p | 7s 8s 9s | E E E | N + tsumo W
        // Tenpai (shanten=0). The optimal discard is N or W (isolated honors).
        // Discarding any tile from a mentsu would break it and raise shanten.
        let state = setup_clear_discard_game();
        let mut rng = ChaCha12Rng::seed_from_u64(42);

        for _ in 0..20 {
            let tile = choose_discard(&state, &mut rng);
            let tid = tile.as_u8() as usize;
            // N=30 (North), W=29 (West) — both are isolated yaokyuu tiles
            assert!(
                tid == 30 || tid == 29,
                "expected N(30) or W(29), got tile id {tid}"
            );
        }
    }

    #[test]
    fn test_accepts_hora() {
        // Set up a tsumo agari situation
        let mut state = PlayerState::new(0);
        state.set_record_events(true);

        let bakaze: Tile = "E".parse().unwrap();
        let dora_marker: Tile = "9s".parse().unwrap();

        // Tenpai hand waiting on 1p; tsumo 1p
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

        // Tsumo the winning tile (E, completing the pair+mentsu)
        state
            .update(&Event::Tsumo {
                actor: 0,
                pai: "E".parse().unwrap(),
            })
            .unwrap();

        let cans = state.last_cans();
        assert!(
            cans.can_tsumo_agari,
            "test setup error: expected can_tsumo_agari"
        );
        let mut rng = ChaCha12Rng::seed_from_u64(42);
        let ev = smart_reaction(0, &state, &mut rng);
        match ev {
            Event::Hora { actor, target, .. } => {
                assert_eq!(actor, 0);
                assert_eq!(target, 0); // tsumo agari: target = self
            }
            _ => panic!("expected Hora event for tsumo agari, got {ev:?}"),
        }
    }

    #[test]
    fn test_passes_calls() {
        // Set up a state where player has chi/pon options but NOT a discard
        // This happens when it's another player's discard and we can call.
        let mut state = PlayerState::new(1);
        state.set_record_events(true);

        let bakaze: Tile = "E".parse().unwrap();
        let dora_marker: Tile = "9s".parse().unwrap();

        let tehais = [
            ["?"; 13].map(|s| s.parse::<Tile>().unwrap()),
            // Player 1 has tiles for chi: 3m 4m can chi on 2m or 5m
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

        // Oya draws and discards 2m (player 1 can chi with 3m+4m)
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

        let cans = state.last_cans();
        // Player 1 should have chi option but not discard
        assert!(
            cans.can_chi_low || cans.can_chi_mid || cans.can_chi_high,
            "test setup error: expected can_chi"
        );
        assert!(
            !cans.can_discard,
            "test setup error: expected no can_discard"
        );
        let mut rng = ChaCha12Rng::seed_from_u64(42);
        let ev = smart_reaction(1, &state, &mut rng);
        assert!(
            matches!(ev, Event::None),
            "should pass on call opportunities, got {ev:?}"
        );
    }

    #[test]
    fn test_top_k_diversity() {
        // With a hand that has multiple viable discards at the same shanten,
        // verify that top-k sampling produces diverse results.
        // Hand: 1m 2m 3m 4m 5m 6m 7m 8m 9m 1p 2p 3p 4p + tsumo 5p
        // Many tiles are equivalent in shanten terms.
        let state = setup_basic_game();
        let mut rng = ChaCha12Rng::seed_from_u64(42);

        let mut seen_tiles = std::collections::HashSet::new();
        for _ in 0..100 {
            let tile = choose_discard(&state, &mut rng);
            seen_tiles.insert(tile.as_u8());
        }

        assert!(
            seen_tiles.len() >= 2,
            "expected at least 2 different discards from top-k sampling, got {}",
            seen_tiles.len()
        );
    }

    #[test]
    fn test_full_rollout_completes() {
        // Verify that a full game rollout using smart_reaction completes
        // without errors.
        let state = setup_basic_game();
        let config = ParticleConfig::new(5);
        let mut rng = ChaCha12Rng::seed_from_u64(42);

        let (particles, _attempts) = generate_particles(&state, &config, &mut rng).unwrap();
        assert!(!particles.is_empty());

        let replayed = simulator::replay_player_states(&state).unwrap();
        let base = simulator::derive_midgame_context_base(state.event_history());

        for (i, particle) in particles.iter().enumerate() {
            let board_state =
                simulator::build_midgame_board_state_with_base(&state, &replayed, particle, &base)
                    .unwrap();
            let initial_scores = board_state.board.scores;
            let result = simulator::run_rollout_smart(board_state, initial_scores, &mut rng);
            assert!(
                result.is_ok(),
                "smart rollout particle {i} failed: {:?}",
                result.err()
            );

            let result = result.unwrap();
            let delta_sum: i32 = result.deltas.iter().sum();
            assert!(
                delta_sum.abs() <= 1000,
                "smart rollout particle {i}: delta sum {delta_sum} is too large"
            );
        }
    }

    #[test]
    fn test_safety_array_genbutsu() {
        // Verify that build_safety_array correctly marks genbutsu tiles
        // when an opponent is in riichi.
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

        // Player 0 draws + discards to advance turns
        state.update(&Event::Tsumo { actor: 0, pai: "N".parse().unwrap() }).unwrap();
        state.update(&Event::Dahai { actor: 0, pai: "N".parse().unwrap(), tsumogiri: true }).unwrap();

        // Player 1 draws, declares riichi, discards 5m
        state.update(&Event::Tsumo { actor: 1, pai: "?".parse().unwrap() }).unwrap();
        state.update(&Event::Reach { actor: 1 }).unwrap();
        state.update(&Event::Dahai { actor: 1, pai: "5m".parse().unwrap(), tsumogiri: false }).unwrap();
        state.update(&Event::ReachAccepted { actor: 1 }).unwrap();

        // Player 2 draws + discards
        state.update(&Event::Tsumo { actor: 2, pai: "?".parse().unwrap() }).unwrap();
        state.update(&Event::Dahai { actor: 2, pai: "W".parse().unwrap(), tsumogiri: true }).unwrap();

        // Player 3 draws + discards
        state.update(&Event::Tsumo { actor: 3, pai: "?".parse().unwrap() }).unwrap();
        state.update(&Event::Dahai { actor: 3, pai: "S".parse().unwrap(), tsumogiri: true }).unwrap();

        // Player 0 draws again (now we can check safety)
        state.update(&Event::Tsumo { actor: 0, pai: "C".parse().unwrap() }).unwrap();

        let safety = build_safety_array(&state);

        // 5m (tile id 4) should be genbutsu (safety 10) — discarded by riichi player 1
        assert_eq!(safety[4], 10, "5m should be genbutsu (10)");
        // 2m (tile id 1) is suji partner of 5m (pos 4 - 3 = 1), safety 5
        assert_eq!(safety[1], 5, "2m should be suji (5)");
        // 8m (tile id 7) is suji partner of 5m (pos 4 + 3 = 7), safety 5
        assert_eq!(safety[7], 5, "8m should be suji (5)");
    }

    // T1: After chi/pon, smart_reaction returns a valid discard
    #[test]
    fn test_post_chipon_discard() {
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

        // Player 1 calls chi (2m from kawa, using 3m+4m)
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

        let mut rng = ChaCha12Rng::seed_from_u64(42);
        let ev = smart_reaction(1, &state, &mut rng);
        match ev {
            Event::Dahai {
                actor,
                tsumogiri,
                pai,
            } => {
                assert_eq!(actor, 1);
                assert!(!tsumogiri, "post-meld discard is not tsumogiri");
                // Verify the discarded tile was actually in hand
                let tid = pai.deaka().as_u8() as usize;
                assert!(
                    state.tehai()[tid] > 0,
                    "discarded tile {pai:?} (tid={tid}) should be in tehai"
                );
            }
            _ => panic!("expected Dahai after chi, got {ev:?}"),
        }
    }

    // T2: Verify higher-ukeire discard is preferred
    #[test]
    fn test_ukeire_tiebreak() {
        // Hand: 1m 2m 3m | 4p 5p 6p | 7s 8s 9s | E E | N N + tsumo W
        // Shanten 0 (tenpai). Discarding N or W both maintain tenpai.
        // But N has a pair (count=2), W is isolated (count=1).
        // Discarding W should be preferred (isolated yaokyuu bonus + similar ukeire).
        let mut state = PlayerState::new(0);
        state.set_record_events(true);

        let bakaze: Tile = "E".parse().unwrap();
        let dora_marker: Tile = "1m".parse().unwrap();

        let tehais = [
            [
                "1m", "2m", "3m", "4p", "5p", "6p", "7s", "8s", "9s", "E", "E", "N", "N",
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
                pai: "W".parse().unwrap(),
            })
            .unwrap();

        // Run many iterations. W (isolated yaokyuu) should be chosen more
        // often than N (which forms a pair and provides more waits if kept).
        let mut w_count = 0;
        let mut n_count = 0;
        for seed in 0..50_u64 {
            let mut rng = ChaCha12Rng::seed_from_u64(seed);
            let tile = choose_discard(&state, &mut rng);
            let tid = tile.as_u8() as usize;
            if tid == 29 {
                // W
                w_count += 1;
            } else if tid == 30 {
                // N
                n_count += 1;
            }
        }

        assert!(
            w_count > n_count,
            "W should be preferred over N (W={w_count}, N={n_count})"
        );
    }

    // T3: Verify genbutsu tile is preferred under opponent riichi
    #[test]
    fn test_safety_under_riichi() {
        // Hand with two honor tiles at equal shanten. One is genbutsu
        // (discarded by riichi player), the other is not.
        let mut state = PlayerState::new(0);
        state.set_record_events(true);

        let bakaze: Tile = "E".parse().unwrap();
        let dora_marker: Tile = "9s".parse().unwrap();

        // Hand: 1m2m3m 4p5p6p 7s8s9s EE + W C (isolated, both yaokyuu)
        let tehais = [
            [
                "1m", "2m", "3m", "4p", "5p", "6p", "7s", "8s", "9s", "E", "E", "W", "C",
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

        // Player 0 draws + discards to advance turns
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

        // Player 1 declares riichi and discards W (making W genbutsu)
        state
            .update(&Event::Tsumo {
                actor: 1,
                pai: "?".parse().unwrap(),
            })
            .unwrap();
        state.update(&Event::Reach { actor: 1 }).unwrap();
        state
            .update(&Event::Dahai {
                actor: 1,
                pai: "W".parse().unwrap(),
                tsumogiri: false,
            })
            .unwrap();
        state
            .update(&Event::ReachAccepted { actor: 1 })
            .unwrap();

        // Players 2 and 3 draw + discard
        state
            .update(&Event::Tsumo {
                actor: 2,
                pai: "?".parse().unwrap(),
            })
            .unwrap();
        state
            .update(&Event::Dahai {
                actor: 2,
                pai: "S".parse().unwrap(),
                tsumogiri: true,
            })
            .unwrap();
        state
            .update(&Event::Tsumo {
                actor: 3,
                pai: "?".parse().unwrap(),
            })
            .unwrap();
        state
            .update(&Event::Dahai {
                actor: 3,
                pai: "S".parse().unwrap(),
                tsumogiri: true,
            })
            .unwrap();

        // Player 0 draws again
        state
            .update(&Event::Tsumo {
                actor: 0,
                pai: "F".parse().unwrap(),
            })
            .unwrap();

        // Now we have W (genbutsu, safety=10), C (unknown, safety=0),
        // F (unknown, safety=0) as candidates.
        // Verify that the safety array correctly scores W higher.
        let safety = build_safety_array(&state);
        assert_eq!(
            safety[29], 10,
            "W (tid 29) should be genbutsu (safety 10)"
        );
        assert_eq!(safety[33], 0, "C (tid 33) should have safety 0");
        // F (tid 32) should also be 0 unless discarded by riichi player
        assert_eq!(safety[32], 0, "F (tid 32) should have safety 0");

        // W should be the top-scored candidate and chosen at least sometimes.
        // With top-k=3 and 3 candidates (W, C, F), each gets ~1/3 chance,
        // but W is always in top-k due to highest score.
        let mut w_count = 0;
        let total = 50_u64;
        for seed in 0..total {
            let mut rng = ChaCha12Rng::seed_from_u64(seed);
            let tile = choose_discard(&state, &mut rng);
            let tid = tile.as_u8() as usize;
            if tid == 29 {
                // W
                w_count += 1;
            }
        }

        assert!(
            w_count > 0,
            "W (genbutsu) should be chosen at least once in {total} attempts"
        );
    }

    // T4: Smart action rollout completes without errors
    #[test]
    fn test_smart_action_rollout_completes() {
        let state = setup_basic_game();
        let config = ParticleConfig::new(5);
        let mut rng = ChaCha12Rng::seed_from_u64(42);

        let (particles, _attempts) = generate_particles(&state, &config, &mut rng).unwrap();
        assert!(!particles.is_empty());

        let replayed = simulator::replay_player_states(&state).unwrap();
        let base = simulator::derive_midgame_context_base(state.event_history());

        // Discard 1m (action 0) with smart heuristic
        for (i, particle) in particles.iter().enumerate() {
            let result = simulator::simulate_action_rollout_with_base_smart(
                &state, &replayed, particle, 0, &base, &mut rng,
            );
            assert!(
                result.is_ok(),
                "smart action rollout particle {i} failed: {:?}",
                result.err()
            );

            let result = result.unwrap();
            let delta_sum: i32 = result.deltas.iter().sum();
            assert!(
                delta_sum.abs() <= 1000,
                "smart action rollout particle {i}: delta sum {delta_sum} is too large"
            );
        }
    }

    // T6: Riichi forces tsumogiri
    #[test]
    fn test_riichi_forces_tsumogiri() {
        // Set up a state where our player has declared and accepted riichi.
        // After riichi, any draw must be tsumogiri'd.
        let mut state = PlayerState::new(0);
        state.set_record_events(true);

        let bakaze: Tile = "E".parse().unwrap();
        let dora_marker: Tile = "9s".parse().unwrap();

        // Tenpai hand: 1m2m3m 4p5p6p 7s8s9s EE + N
        let tehais = [
            [
                "1m", "2m", "3m", "4p", "5p", "6p", "7s", "8s", "9s", "E", "E", "N", "N",
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

        // Player 0 draws, declares riichi, discards N
        state
            .update(&Event::Tsumo {
                actor: 0,
                pai: "W".parse().unwrap(),
            })
            .unwrap();
        state.update(&Event::Reach { actor: 0 }).unwrap();
        state
            .update(&Event::Dahai {
                actor: 0,
                pai: "W".parse().unwrap(),
                tsumogiri: true,
            })
            .unwrap();
        state
            .update(&Event::ReachAccepted { actor: 0 })
            .unwrap();

        // Opponents draw and discard
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
        state
            .update(&Event::Tsumo {
                actor: 2,
                pai: "?".parse().unwrap(),
            })
            .unwrap();
        state
            .update(&Event::Dahai {
                actor: 2,
                pai: "S".parse().unwrap(),
                tsumogiri: true,
            })
            .unwrap();
        state
            .update(&Event::Tsumo {
                actor: 3,
                pai: "?".parse().unwrap(),
            })
            .unwrap();
        state
            .update(&Event::Dahai {
                actor: 3,
                pai: "S".parse().unwrap(),
                tsumogiri: true,
            })
            .unwrap();

        // Player 0 draws C (not a winning tile)
        state
            .update(&Event::Tsumo {
                actor: 0,
                pai: "C".parse().unwrap(),
            })
            .unwrap();

        assert!(
            state.self_riichi_accepted(),
            "test setup error: expected riichi accepted"
        );

        let mut rng = ChaCha12Rng::seed_from_u64(42);
        let ev = smart_reaction(0, &state, &mut rng);
        match ev {
            Event::Dahai {
                actor,
                pai,
                tsumogiri,
            } => {
                assert_eq!(actor, 0);
                assert!(tsumogiri, "riichi player must tsumogiri");
                // The discarded tile should be the tsumo tile (C = tile id 33)
                assert_eq!(
                    pai.deaka().as_u8(),
                    33,
                    "riichi tsumogiri should discard the drawn tile C (id 33)"
                );
            }
            _ => panic!("expected Dahai for riichi player, got {ev:?}"),
        }
    }
}
