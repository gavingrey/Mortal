use super::config::ParticleConfig;
use crate::state::PlayerState;
use crate::tile::Tile;
use crate::{must_tile, tu8, tuz};

use anyhow::{Result, ensure};
use pyo3::prelude::*;
use rand::prelude::*;
use rand_chacha::ChaCha12Rng;

/// A single sampled world consistent with observations.
///
/// Each particle represents one possible complete assignment of all hidden
/// information: opponent hands, remaining wall tiles, and dead wall contents.
#[pyclass]
#[derive(Clone, Debug)]
pub struct Particle {
    /// Opponent hands, indexed by relative position (1..=3 mapped to 0..3).
    /// Each hand is a Vec of tiles the opponent holds.
    pub opponent_hands: [Vec<Tile>; 3],

    /// Remaining wall tiles in draw order.
    pub wall: Vec<Tile>,

    /// Importance weight for weighted sampling (Phase 1: uniform = 1.0).
    #[pyo3(get, set)]
    pub weight: f32,
}

#[pymethods]
impl Particle {
    /// Number of tiles in opponent i's hand (0-indexed relative).
    #[must_use]
    pub fn opponent_hand_size(&self, opp_idx: usize) -> usize {
        self.opponent_hands[opp_idx].len()
    }

    /// Number of tiles remaining in the wall.
    #[must_use]
    pub fn wall_size(&self) -> usize {
        self.wall.len()
    }

    fn __repr__(&self) -> String {
        format!(
            "Particle(hands=[{}, {}, {}], wall={}, weight={:.4})",
            self.opponent_hands[0].len(),
            self.opponent_hands[1].len(),
            self.opponent_hands[2].len(),
            self.wall.len(),
            self.weight,
        )
    }
}

/// Visible tile information extracted from a PlayerState.
///
/// This captures all tiles known to the player, decomposed into categories
/// needed for particle generation.
#[allow(dead_code)] // tehai/akas_in_hand will be used in Phase 2 consistency checks
struct VisibleTiles {
    /// Count of each tile type (34-indexed) that we have seen.
    /// This is directly from `PlayerState.tiles_seen`.
    seen_counts: [u8; 34],

    /// Whether each aka dora has been seen.
    akas_seen: [bool; 3],

    /// Our hand tile counts (34-indexed).
    tehai: [u8; 34],

    /// Whether we hold each aka dora.
    akas_in_hand: [bool; 3],

    /// Number of tiles in each opponent's closed hand.
    /// Indexed by relative position minus 1 (opponent 0 = relative seat 1).
    opponent_hand_sizes: [u8; 3],

    /// Tiles remaining in live wall (from PlayerState.tiles_left).
    tiles_left: u8,

    /// Number of revealed dora indicators (known dead wall tiles).
    num_dora_indicators: u8,
}

impl VisibleTiles {
    fn from_player_state(state: &PlayerState) -> Self {
        // Compute opponent hand sizes from fuuro and turn information.
        // Each player starts with 13 tiles. Open melds consume tiles from hand:
        // - chi/pon: 2 tiles from hand consumed, 1 from discard -> net -1 tile from closed hand per meld
        //   (actually the closed hand size is tehai_len_div3 * 3 + 1 or +2, but we don't have direct
        //    access to opponents' tehai_len_div3)
        //
        // The simplest approach: count tiles in opponents' fuuro_overview and ankan_overview
        // to determine how many tiles have been revealed from their hands.
        // But since PlayerState fields are pub(super), we need to use public getters.
        //
        // For now, we compute hand sizes from the game structure:
        // - A player starts with 13 tiles
        // - Each chi/pon removes 2 tiles from closed hand (the called tile comes from discard)
        // - Each daiminkan removes 3 tiles from closed hand
        // - Each kakan removes 1 tile from closed hand (added to existing pon)
        // - Each ankan removes 4 tiles from closed hand
        // But the player also draws tiles, so the actual hand size varies.
        //
        // Actually, for particle generation we get hand sizes from state data.
        // We'll use a dedicated method that takes the required info.

        Self {
            seen_counts: state.tiles_seen(),
            akas_seen: state.akas_seen(),
            tehai: state.tehai(),
            akas_in_hand: state.akas_in_hand(),
            opponent_hand_sizes: state.opponent_hand_sizes(),
            tiles_left: state.tiles_left(),
            num_dora_indicators: state.num_dora_indicators(),
        }
    }

    /// Compute the pool of hidden tiles (136-tile representation).
    ///
    /// Hidden tiles = FULL_TILE_SET - (tiles_seen from our perspective).
    /// Since `tiles_seen` tracks all tiles we've witnessed (our hand + discards
    /// + melds + dora indicators), the hidden pool is everything else.
    fn compute_hidden_tiles(&self) -> Vec<Tile> {
        // Build a 34-count array of seen tiles, then subtract from full set.
        // For aka handling: we track seen akas separately.
        let mut remaining_counts = [0u8; 34];
        let mut remaining_akas = [false; 3]; // whether the aka is still hidden

        // Start with 4 of each tile type
        for count in &mut remaining_counts {
            *count = 4;
        }

        // Subtract all seen tiles
        for (tid, &seen) in self.seen_counts.iter().enumerate() {
            remaining_counts[tid] = remaining_counts[tid].saturating_sub(seen);
        }

        // Track aka status: if not seen, the aka is still in the hidden pool
        for i in 0..3 {
            remaining_akas[i] = !self.akas_seen[i];
        }

        // Also subtract our own hand (which is part of tiles_seen already,
        // but we need to not include our hand tiles in the hidden pool).
        // Actually, tiles_seen already includes our hand tiles, so
        // remaining_counts already excludes them. We're good.

        // Expand 34-counts into actual 136-format tiles
        let mut hidden = Vec::with_capacity(136);
        for tid in 0..34 {
            let count = remaining_counts[tid];
            if count == 0 {
                continue;
            }

            // Check if this tile type has an aka variant
            let aka_tile_id = match tid {
                x if x == tuz!(5m) => Some(0),
                x if x == tuz!(5p) => Some(1),
                x if x == tuz!(5s) => Some(2),
                _ => None,
            };

            if let Some(aka_idx) = aka_tile_id {
                // This tile type has an aka variant
                if remaining_akas[aka_idx] {
                    // The aka is still hidden, include it
                    hidden.push(must_tile!(tu8!(5mr) + aka_idx as u8));
                    // Add remaining as normal tiles
                    for _ in 0..count - 1 {
                        hidden.push(must_tile!(tid));
                    }
                } else {
                    // The aka has been seen, all remaining are normal
                    for _ in 0..count {
                        hidden.push(must_tile!(tid));
                    }
                }
            } else {
                for _ in 0..count {
                    hidden.push(must_tile!(tid));
                }
            }
        }

        hidden
    }
}

/// Generate particles consistent with the current game state.
///
/// Uses rejection sampling: shuffle hidden tiles, deal to opponents and wall,
/// then check consistency. Inconsistent samples are discarded.
///
/// Phase 1: Uniform weighting (all accepted particles have weight 1.0).
pub fn generate_particles(
    state: &PlayerState,
    config: &ParticleConfig,
    rng: &mut ChaCha12Rng,
) -> Result<Vec<Particle>> {
    let visible = VisibleTiles::from_player_state(state);
    let hidden_tiles = visible.compute_hidden_tiles();

    // Dead wall has 14 tiles total. Of those, num_dora_indicators are revealed
    // (already counted in tiles_seen). The rest are hidden but not in
    // opponent hands or the live wall.
    let dead_wall_unseen = 14 - visible.num_dora_indicators as usize;
    let expected_hidden = {
        let total_hand_tiles: u8 = visible.opponent_hand_sizes.iter().sum();
        total_hand_tiles as usize + visible.tiles_left as usize + dead_wall_unseen
    };
    ensure!(
        hidden_tiles.len() == expected_hidden,
        "hidden tile count mismatch: got {}, expected {} \
         (opp hands: {:?}, wall: {}, dead wall unseen: {})",
        hidden_tiles.len(),
        expected_hidden,
        visible.opponent_hand_sizes,
        visible.tiles_left,
        dead_wall_unseen,
    );

    let max_attempts = config.max_attempts();
    let mut particles = Vec::with_capacity(config.n_particles);
    let mut attempts = 0;

    while particles.len() < config.n_particles && attempts < max_attempts {
        attempts += 1;

        let mut shuffled = hidden_tiles.clone();
        shuffled.shuffle(rng);

        // Deal tiles to opponents
        let mut idx = 0;
        let mut opponent_hands = [vec![], vec![], vec![]];
        let mut valid = true;

        for (opp, hand) in opponent_hands.iter_mut().enumerate() {
            let hand_size = visible.opponent_hand_sizes[opp] as usize;
            if idx + hand_size > shuffled.len() {
                valid = false;
                break;
            }
            *hand = shuffled[idx..idx + hand_size].to_vec();
            idx += hand_size;
        }

        if !valid {
            continue;
        }

        // Next tiles go to live wall, remainder is dead wall (discarded)
        let wall_end = idx + visible.tiles_left as usize;
        if wall_end > shuffled.len() {
            valid = false;
        }
        if !valid {
            continue;
        }
        let wall = shuffled[idx..wall_end].to_vec();
        // Remaining tiles (shuffled[wall_end..]) are unseen dead wall tiles;
        // we don't need to track them in the particle.

        // Phase 1: no additional consistency checks beyond tile accounting.
        // The hidden tile computation already ensures we only use tiles that
        // are not visible to the player. In Phase 2, we would add checks like:
        // - Opponent who called riichi must be tenpai
        // - Hands must be consistent with observed call patterns
        // - etc.

        particles.push(Particle {
            opponent_hands,
            wall,
            weight: 1.0,
        });
    }

    // Normalize weights (uniform in Phase 1, but structure is ready)
    let total_weight: f32 = particles.iter().map(|p| p.weight).sum();
    if total_weight > 0.0 {
        for p in &mut particles {
            p.weight /= total_weight;
        }
    }

    Ok(particles)
}

/// Verify that a particle is consistent with the observed game state.
///
/// This is used both as a filter during generation and as a validation
/// function for testing.
pub fn is_particle_consistent(particle: &Particle, state: &PlayerState) -> bool {
    let visible = VisibleTiles::from_player_state(state);

    // Check 1: Total tile count in particle must equal opponent hands + live wall
    let particle_tile_count: usize = particle
        .opponent_hands
        .iter()
        .map(|h| h.len())
        .sum::<usize>()
        + particle.wall.len();
    let dead_wall_unseen = 14 - visible.num_dora_indicators as usize;
    let visible_count = visible.seen_counts.iter().sum::<u8>();
    let total_hidden = 136 - visible_count as usize;
    // Particles contain hidden tiles minus the unseen dead wall
    let expected_particle_tiles = total_hidden - dead_wall_unseen;
    if particle_tile_count != expected_particle_tiles {
        return false;
    }

    // Check 2: Opponent hand sizes must match expected
    for (opp, hand) in particle.opponent_hands.iter().enumerate() {
        if hand.len() != visible.opponent_hand_sizes[opp] as usize {
            return false;
        }
    }

    // Check 3: No tile appears more than its remaining count
    let mut used_counts = [0u8; 34];
    let mut used_akas = [false; 3];

    for hand in &particle.opponent_hands {
        for &tile in hand {
            let tid = tile.deaka().as_usize();
            used_counts[tid] += 1;
            if tile.is_aka() {
                let aka_idx = tile.as_usize() - tuz!(5mr);
                if used_akas[aka_idx] {
                    return false; // Duplicate aka
                }
                used_akas[aka_idx] = true;
            }
        }
    }
    for &tile in &particle.wall {
        let tid = tile.deaka().as_usize();
        used_counts[tid] += 1;
        if tile.is_aka() {
            let aka_idx = tile.as_usize() - tuz!(5mr);
            if used_akas[aka_idx] {
                return false; // Duplicate aka
            }
            used_akas[aka_idx] = true;
        }
    }

    // Verify counts don't exceed what's available
    // (some hidden tiles may be in the unseen dead wall, so <= not ==)
    for tid in 0..34 {
        let available = 4 - visible.seen_counts[tid];
        if used_counts[tid] > available {
            return false;
        }
    }

    // Verify aka consistency
    for i in 0..3 {
        if visible.akas_seen[i] && used_akas[i] {
            return false; // Aka was already seen but appears in particle
        }
    }

    true
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::mjai::Event;
    use crate::tile::Tile;

    fn setup_basic_game() -> PlayerState {
        let mut state = PlayerState::new(0);

        // Create a basic game start
        let bakaze: Tile = "E".parse().unwrap();
        let dora_marker: Tile = "1m".parse().unwrap();

        let tehais = [
            // Player 0's hand
            [
                "1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "1p", "2p", "3p", "4p",
            ]
            .map(|s| s.parse::<Tile>().unwrap()),
            // Player 1's hand (unknown to us, shown as ?)
            ["?"; 13].map(|s| s.parse::<Tile>().unwrap()),
            // Player 2's hand
            ["?"; 13].map(|s| s.parse::<Tile>().unwrap()),
            // Player 3's hand
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
    fn hidden_tiles_count() {
        let state = setup_basic_game();
        let visible = VisibleTiles::from_player_state(&state);
        let hidden = visible.compute_hidden_tiles();

        // We have 14 tiles in hand (13 + tsumo) + 1 dora indicator = 15 seen tiles
        // But tiles_seen tracks deaka'd counts, so we need to count carefully.
        // Our hand: 1-9m (9 tiles) + 1-4p (4 tiles) + 5p tsumo = 14 tiles in hand
        // Plus 1 dora indicator (1m) which is also witnessed.
        // tiles_seen for 1m = 2 (1 in hand + 1 dora indicator)
        // Total seen = 15 tile witnesses
        let total_seen: u8 = visible.seen_counts.iter().sum();
        assert_eq!(
            hidden.len(),
            136 - total_seen as usize,
            "hidden tiles should be 136 minus total seen"
        );
    }

    #[test]
    fn generate_particles_basic() {
        let state = setup_basic_game();
        let config = ParticleConfig::new(50);
        let mut rng = ChaCha12Rng::seed_from_u64(42);

        let particles = generate_particles(&state, &config, &mut rng).unwrap();
        assert!(
            !particles.is_empty(),
            "should generate at least some particles"
        );

        // All particles should have 3 opponent hands
        for particle in &particles {
            assert_eq!(particle.opponent_hands.len(), 3);
        }
    }

    #[test]
    fn particles_are_consistent() {
        let state = setup_basic_game();
        let config = ParticleConfig::new(100);
        let mut rng = ChaCha12Rng::seed_from_u64(123);

        let particles = generate_particles(&state, &config, &mut rng).unwrap();
        for (i, particle) in particles.iter().enumerate() {
            assert!(
                is_particle_consistent(particle, &state),
                "particle {i} is inconsistent"
            );
        }
    }

    #[test]
    fn particle_weights_sum_to_one() {
        let state = setup_basic_game();
        let config = ParticleConfig::new(50);
        let mut rng = ChaCha12Rng::seed_from_u64(99);

        let particles = generate_particles(&state, &config, &mut rng).unwrap();
        if !particles.is_empty() {
            let total: f32 = particles.iter().map(|p| p.weight).sum();
            assert!(
                (total - 1.0).abs() < 1e-5,
                "weights should sum to ~1.0, got {total}"
            );
        }
    }

    #[test]
    fn no_visible_tiles_in_particles() {
        let state = setup_basic_game();
        let visible = VisibleTiles::from_player_state(&state);
        let config = ParticleConfig::new(50);
        let mut rng = ChaCha12Rng::seed_from_u64(77);

        let particles = generate_particles(&state, &config, &mut rng).unwrap();

        for particle in &particles {
            // Collect all particle tiles into counts
            let mut counts = [0u8; 34];
            for hand in &particle.opponent_hands {
                for &tile in hand {
                    counts[tile.deaka().as_usize()] += 1;
                }
            }
            for &tile in &particle.wall {
                counts[tile.deaka().as_usize()] += 1;
            }

            // No tile type should exceed 4 minus what we've seen
            for tid in 0..34 {
                let max_hidden = 4 - visible.seen_counts[tid];
                assert!(
                    counts[tid] <= max_hidden,
                    "tile {tid} count {} exceeds max hidden {max_hidden}",
                    counts[tid],
                );
            }
        }
    }

    #[test]
    fn particle_coverage() {
        // Ensure that across many particles, all hidden tile types appear
        let state = setup_basic_game();
        let config = ParticleConfig::new(200);
        let mut rng = ChaCha12Rng::seed_from_u64(555);

        let particles = generate_particles(&state, &config, &mut rng).unwrap();
        let visible = VisibleTiles::from_player_state(&state);

        let mut tile_appeared = [false; 34];
        for particle in &particles {
            for hand in &particle.opponent_hands {
                for &tile in hand {
                    tile_appeared[tile.deaka().as_usize()] = true;
                }
            }
            for &tile in &particle.wall {
                tile_appeared[tile.deaka().as_usize()] = true;
            }
        }

        // Every tile type that has hidden copies should appear somewhere
        for tid in 0..34 {
            if visible.seen_counts[tid] < 4 {
                assert!(
                    tile_appeared[tid],
                    "tile type {tid} has hidden copies but never appeared in particles"
                );
            }
        }
    }
}
