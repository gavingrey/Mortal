//! Tier 1 search-as-features for the LuckyJ-style probe.
//!
//! Per-opponent attack/defense features that go beyond Tier 0's per-discard
//! aggregates. All channels are shape `(1, 34)` aligned with the existing
//! observation format. Channel layout (9 channels total = 306 floats, row-major):
//!
//!   0..=2   per-opp genbutsu     (3 × 34 binary, only when that opp is in riichi)
//!   3..=5   per-opp suji         (3 × 34 binary, only when that opp is in riichi)
//!   6..=8   per-opp tenpai prob  (3 × 34 broadcast scalar from particle shanten check)
//!
//! Opp index 0/1/2 corresponds to relative seats 1/2/3 — same convention as
//! the Tier 0 `_build_search_features` aggregator in `probe_search_features.py`.
//!
//! Genbutsu: tile is in opp's discard pond AND opp has accepted riichi → 1, else 0.
//! Suji:     tile is suji partner of any tile in opp's pond AND opp has accepted
//!           riichi → 1. Honors and terminals have no suji partners; only number
//!           tiles 4..=6 ± 3 within the same suit count.
//! Tenpai prob: fraction of particles where shanten of opp's closed hand == 0.
//!           len_div3 derived from closed hand size: (hand_len - 1) / 3.

use crate::algo::shanten;
use crate::state::PlayerState;
use crate::tile::Tile;

use super::particle::Particle;

/// Number of feature channels produced.
pub const N_CHANNELS: usize = 9;

/// Total length of the flat output vector (channels × 34).
pub const FEATURE_LEN: usize = N_CHANNELS * 34;

/// Convert a slice of tiles to a 34-indexed count array (aka tiles collapsed
/// to their non-aka counterpart).
fn tiles_to_counts(tiles: &[Tile]) -> [u8; 34] {
    let mut counts = [0_u8; 34];
    for &t in tiles {
        let tid = t.deaka().as_u8() as usize;
        if tid < 34 {
            counts[tid] += 1;
        }
    }
    counts
}

/// Compute Tier 1 search features for the given state and particles.
///
/// Returns a flat `Vec<f32>` of length `FEATURE_LEN` in row-major order:
/// `out[c * 34 + tid]` is channel `c`'s value for tile id `tid`.
pub fn compute_tier1_features(state: &PlayerState, particles: &[Particle]) -> Vec<f32> {
    let mut feats = vec![0.0_f32; FEATURE_LEN];

    let riichi_accepted = state.riichi_accepted();
    let kawa = state.kawa_overview();

    // Channels 0..=5: per-opp genbutsu and suji (rule-based, deterministic from state).
    for opp_idx in 0..3_usize {
        // opp_idx 0 = relative seat 1 (next opponent in turn order)
        let rel_seat = opp_idx + 1;
        if !riichi_accepted[rel_seat] {
            // Channels stay at 0 — opponent not in riichi, no defensive
            // commitment from us yet.
            continue;
        }

        let genbutsu_ch = opp_idx;       // 0, 1, 2
        let suji_ch = 3 + opp_idx;       // 3, 4, 5
        let genbutsu_off = genbutsu_ch * 34;
        let suji_off = suji_ch * 34;

        for &tile in &kawa[rel_seat] {
            let tid = tile.deaka().as_u8() as usize;
            if tid >= 34 {
                continue;
            }
            // Genbutsu: this exact tile is in opp's pond.
            feats[genbutsu_off + tid] = 1.0;

            // Suji: same suit, ±3 from position. Only number tiles (suit < 3).
            let suit = tid / 9;
            let pos = tid % 9;
            if suit < 3 {
                if pos + 3 < 9 {
                    feats[suji_off + suit * 9 + pos + 3] = 1.0;
                }
                if pos >= 3 {
                    feats[suji_off + suit * 9 + pos - 3] = 1.0;
                }
            }
        }
    }

    // Channels 6..=8: per-opp tenpai probability.
    // Empty particles → leave at 0 (consistent with no-information default).
    if !particles.is_empty() {
        let n = particles.len();
        let mut tenpai_count = [0_u32; 3];

        for particle in particles {
            for opp_idx in 0..3_usize {
                let hand = &particle.opponent_hands[opp_idx];
                let counts = tiles_to_counts(hand);
                // len_div3 = (closed_hand_size - 1) / 3 for between-turn states.
                // Hand sizes < 1 are degenerate and can't be tenpai.
                let len = hand.len();
                if len == 0 {
                    continue;
                }
                let len_div3 = ((len - 1) / 3) as u8;
                let sh = shanten::calc_all(&counts, len_div3);
                if sh == 0 {
                    tenpai_count[opp_idx] += 1;
                }
            }
        }

        for opp_idx in 0..3_usize {
            let tenpai_ch = 6 + opp_idx; // 6, 7, 8
            let off = tenpai_ch * 34;
            let prob = tenpai_count[opp_idx] as f32 / n as f32;
            for tid in 0..34 {
                feats[off + tid] = prob;
            }
        }
    }

    feats
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::mjai::Event;
    use crate::state::PlayerState;
    use crate::tile::Tile;
    use super::super::particle::Particle;

    /// Build a state where opp 1 (relative seat 1, absolute seat 1 since we are seat 0)
    /// has declared and accepted riichi after discarding 5m. Returns the state at our
    /// next turn so we can inspect safety from our perspective.
    fn setup_opp1_riichi_state() -> PlayerState {
        let mut state = PlayerState::new(0);
        state.set_record_events(true);

        let bakaze: Tile = "E".parse().unwrap();
        let dora_marker: Tile = "9s".parse().unwrap();

        let tehais = [
            ["1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "1p", "2p", "3p", "4p"]
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

        // Player 0 draws + discards (our turn complete)
        state
            .update(&Event::Tsumo { actor: 0, pai: "N".parse().unwrap() })
            .unwrap();
        state
            .update(&Event::Dahai { actor: 0, pai: "N".parse().unwrap(), tsumogiri: true })
            .unwrap();

        // Player 1 declares riichi, discards 5m, riichi accepted
        state
            .update(&Event::Tsumo { actor: 1, pai: "?".parse().unwrap() })
            .unwrap();
        state.update(&Event::Reach { actor: 1 }).unwrap();
        state
            .update(&Event::Dahai { actor: 1, pai: "5m".parse().unwrap(), tsumogiri: false })
            .unwrap();
        state.update(&Event::ReachAccepted { actor: 1 }).unwrap();

        // Players 2 + 3 draw and discard
        state
            .update(&Event::Tsumo { actor: 2, pai: "?".parse().unwrap() })
            .unwrap();
        state
            .update(&Event::Dahai { actor: 2, pai: "S".parse().unwrap(), tsumogiri: true })
            .unwrap();
        state
            .update(&Event::Tsumo { actor: 3, pai: "?".parse().unwrap() })
            .unwrap();
        state
            .update(&Event::Dahai { actor: 3, pai: "S".parse().unwrap(), tsumogiri: true })
            .unwrap();

        // Player 0 draws — now we can inspect safety
        state
            .update(&Event::Tsumo { actor: 0, pai: "C".parse().unwrap() })
            .unwrap();

        state
    }

    #[test]
    fn test_genbutsu_flag_for_riichi_opp() {
        let state = setup_opp1_riichi_state();
        let particles: Vec<Particle> = vec![];
        let feats = compute_tier1_features(&state, &particles);

        // Channel 0 = opp 1 genbutsu. 5m is tile id 4.
        let tid_5m = 4_usize;
        let ch0 = &feats[0..34];
        assert_eq!(ch0[tid_5m], 1.0, "5m should be genbutsu for opp 1");

        // Channels 1, 2 (opp 2, 3 genbutsu) should be all zero — they are not in riichi.
        let ch1 = &feats[34..68];
        let ch2 = &feats[68..102];
        assert!(ch1.iter().all(|&v| v == 0.0), "opp 2 not in riichi → no genbutsu");
        assert!(ch2.iter().all(|&v| v == 0.0), "opp 3 not in riichi → no genbutsu");

        // Other tiles in opp 1's pond? Only 5m was discarded after riichi.
        // But riichi-pre-discards (e.g., the tsumogiri before reach) are also in kawa_overview.
        // We don't strictly assert the rest of ch0 is all zero — only that 5m is set.
    }

    #[test]
    fn test_suji_for_riichi_opp() {
        let state = setup_opp1_riichi_state();
        let particles: Vec<Particle> = vec![];
        let feats = compute_tier1_features(&state, &particles);

        // Channel 3 = opp 1 suji. 5m's suji partners are 2m (tid 1) and 8m (tid 7).
        let ch3 = &feats[3 * 34..4 * 34];
        assert_eq!(ch3[1], 1.0, "2m should be suji of 5m");
        assert_eq!(ch3[7], 1.0, "8m should be suji of 5m");

        // Channels 4, 5 (opp 2, 3 suji) should be all zero.
        let ch4 = &feats[4 * 34..5 * 34];
        let ch5 = &feats[5 * 34..6 * 34];
        assert!(ch4.iter().all(|&v| v == 0.0), "opp 2 not in riichi → no suji");
        assert!(ch5.iter().all(|&v| v == 0.0), "opp 3 not in riichi → no suji");
    }

    #[test]
    fn test_tenpai_prob_no_particles_is_zero() {
        let state = setup_opp1_riichi_state();
        let feats = compute_tier1_features(&state, &[]);

        // Channels 6, 7, 8 should be all zero with no particles.
        for ch in 6..=8 {
            let off = ch * 34;
            for tid in 0..34 {
                assert_eq!(feats[off + tid], 0.0, "channel {ch} tid {tid} should be 0");
            }
        }
    }

    #[test]
    fn test_tenpai_prob_with_known_particles() {
        // Construct two synthetic particles by hand:
        //   particle 0: opp 1 has a tenpai hand (1m2m3m 4p5p6p 7s8s9s EE NN, waiting on N)
        //               opp 2 / opp 3 have garbage (1z 1m × 13 etc.)
        //   particle 1: opp 1 has a non-tenpai hand
        //
        // Expected: opp 1 tenpai prob = 0.5, opps 2/3 tenpai prob = 0.0 (depends on hand).
        let state = setup_opp1_riichi_state();

        // Tenpai hand: 3 sequences + 2 pairs → shanten 0, shanpon wait on E or N.
        let tenpai_hand: Vec<Tile> = ["1m", "2m", "3m", "4p", "5p", "6p", "7s", "8s", "9s",
                                       "E", "E", "N", "N"]
            .iter().map(|s| s.parse().unwrap()).collect();
        // Non-tenpai: 13 distinct tiles, number tiles spaced 3 apart (no sequences possible),
        // 4 isolated honors. Far from tenpai by any rule (standard / chiitoi / kokushi).
        let garbage_hand: Vec<Tile> = ["1m", "4m", "7m", "2p", "5p", "8p", "3s", "6s", "9s",
                                        "E", "S", "W", "N"]
            .iter().map(|s| s.parse().unwrap()).collect();

        let particle_tenpai = Particle {
            opponent_hands: [tenpai_hand.clone(), garbage_hand.clone(), garbage_hand.clone()],
            wall: vec![],
            dead_wall: vec![],
            weight: 1.0,
        };
        let particle_garbage = Particle {
            opponent_hands: [garbage_hand.clone(), garbage_hand.clone(), garbage_hand.clone()],
            wall: vec![],
            dead_wall: vec![],
            weight: 1.0,
        };

        let particles = vec![particle_tenpai, particle_garbage];
        let feats = compute_tier1_features(&state, &particles);

        // Channel 6 = opp 1 tenpai prob. Should be 0.5 (1 of 2 particles tenpai).
        let off_6 = 6 * 34;
        assert!(
            (feats[off_6] - 0.5).abs() < 1e-6,
            "opp 1 tenpai prob should be 0.5, got {}",
            feats[off_6]
        );
        // Broadcast: all 34 entries equal.
        for tid in 0..34 {
            assert!(
                (feats[off_6 + tid] - feats[off_6]).abs() < 1e-9,
                "channel 6 should be broadcast"
            );
        }

        // Channels 7, 8 = opp 2, 3 tenpai prob. With garbage hands, should be 0.0.
        let off_7 = 7 * 34;
        let off_8 = 8 * 34;
        assert_eq!(feats[off_7], 0.0, "opp 2 garbage hand → tenpai prob 0");
        assert_eq!(feats[off_8], 0.0, "opp 3 garbage hand → tenpai prob 0");
    }

    #[test]
    fn test_no_riichi_yields_zero_safety_channels() {
        // Fresh state: no riichi declared yet → all genbutsu/suji channels should be 0.
        let mut state = PlayerState::new(0);
        state.set_record_events(true);

        let bakaze: Tile = "E".parse().unwrap();
        let dora_marker: Tile = "9s".parse().unwrap();
        let tehais = [
            ["1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "1p", "2p", "3p", "4p"]
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
            .update(&Event::Tsumo { actor: 0, pai: "5p".parse().unwrap() })
            .unwrap();

        let feats = compute_tier1_features(&state, &[]);
        // All 6 safety channels should be entirely zero.
        for ch in 0..6 {
            let off = ch * 34;
            for tid in 0..34 {
                assert_eq!(feats[off + tid], 0.0, "channel {ch} tid {tid} should be 0");
            }
        }
    }
}
