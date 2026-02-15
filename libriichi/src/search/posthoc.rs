//! Post-hoc search pipeline for ExIt training (Step 4.3b).
//!
//! Reads self-play game logs (MJSON .json.gz) plus Q-value sidecars (.msgpack),
//! replays game events to reconstruct player states, identifies critical decisions,
//! runs particle-based search, and outputs improved policy targets.

use crate::consts::ACTION_SPACE;
use crate::mjai::{Event, EventExt, Metadata};
use crate::state::PlayerState;

use super::config::ParticleConfig;
use super::particle;
use super::simulator;

use anyhow::{Context, Result, ensure};
use rand::prelude::*;
use rand_chacha::ChaCha12Rng;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the post-hoc search pipeline.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PostHocConfig {
    /// Weight for search values in the blend (0.0 = pure logged, 1.0 = pure search).
    pub blend_weight: f32,
    /// Temperature for softmax normalization.
    pub temperature: f32,
    /// Criticality threshold: decisions with top-2 Q-value gap below this are "critical".
    pub criticality_threshold: f32,
    /// Number of particles for search.
    pub n_particles: usize,
    /// Maximum rollout steps per particle.
    pub max_rollout_steps: u32,
    /// Whether to use the smart heuristic policy during rollouts.
    pub use_smart_policy: bool,
    /// Actions to evaluate per decision (top-K by Q-value).
    pub top_k_actions: usize,
    /// RNG seed for reproducibility (None = random).
    pub seed: Option<u64>,
    /// Whether to z-score normalize search values before softmax blending.
    /// Search values are raw score deltas (~[-50K, +50K]) while Q-values are
    /// log-scale (~[-15, 3]). Without normalization, softmax produces degenerate
    /// one-hot distributions from the huge search value spreads.
    pub normalize_search: bool,
}

impl Default for PostHocConfig {
    fn default() -> Self {
        Self {
            blend_weight: 0.3,
            temperature: 1.0,
            criticality_threshold: 0.5,
            n_particles: 50,
            max_rollout_steps: 100,
            use_smart_policy: true,
            top_k_actions: 5,
            seed: None,
            normalize_search: true,
        }
    }
}

// ---------------------------------------------------------------------------
// Q-value sidecar structures
// ---------------------------------------------------------------------------

/// A single player's decision record from the Q-value sidecar.
#[derive(Debug, Clone, Deserialize)]
pub struct SidecarDecision {
    /// Compact Q-values (only for valid actions, ordered by mask bits).
    #[serde(rename = "q")]
    pub q_values: Vec<f32>,
    /// Mask bits (u64, bit i set means action i is valid).
    #[serde(rename = "m")]
    pub mask_bits: u64,
    /// Action index taken.
    #[serde(rename = "a")]
    pub action: Option<u8>,
    /// Whether the action was greedy.
    #[serde(rename = "g")]
    pub is_greedy: Option<bool>,
}

/// Top-level sidecar format.
#[derive(Debug, Deserialize)]
pub struct Sidecar {
    pub version: u32,
    pub seed: Option<(u64, u64)>,
    /// Per-player decision lists, keyed by player ID string ("0"-"3").
    pub players: HashMap<String, Vec<SidecarDecision>>,
}

// ---------------------------------------------------------------------------
// Output structures
// ---------------------------------------------------------------------------

/// A single decision record output by the post-hoc pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionRecord {
    /// Game identifier (filename stem).
    pub game_id: String,
    /// Decision index within the game (across all players).
    pub decision_idx: u32,
    /// Acting player (0-3).
    pub actor: u8,
    /// Action taken in self-play (0-45).
    pub logged_action: u8,
    /// Full 46-slot Q-values (invalid actions = NEG_INFINITY).
    pub logged_q_values: Vec<f32>,
    /// Action validity mask (46 slots).
    pub mask: Vec<bool>,
    /// Per-action search values (46 slots, invalid = 0.0).
    pub search_values: Vec<f32>,
    /// Best action according to blended policy.
    pub improved_action: u8,
    /// Blended soft target distribution (46 slots).
    pub improved_distribution: Vec<f32>,
    /// How much search changed the decision (KL divergence proxy).
    pub search_effect: f32,
    /// Number of particles actually used.
    pub num_particles: u32,
}

// ---------------------------------------------------------------------------
// Core logic
// ---------------------------------------------------------------------------

/// Expand compact Q-values (only valid actions) into a full 46-slot array
/// using mask_bits to determine which slots are valid.
pub fn expand_q_values(compact_q: &[f32], mask_bits: u64) -> ([f32; ACTION_SPACE], [bool; ACTION_SPACE]) {
    let mut full_q = [f32::NEG_INFINITY; ACTION_SPACE];
    let mut mask = [false; ACTION_SPACE];
    let mut q_idx = 0_usize;

    for i in 0..ACTION_SPACE {
        if mask_bits & (1_u64 << i) != 0 {
            mask[i] = true;
            if q_idx < compact_q.len() {
                full_q[i] = compact_q[q_idx];
                q_idx += 1;
            }
        }
    }

    (full_q, mask)
}

/// Determine whether a decision is "critical" (worth searching).
///
/// A decision is critical when the gap between the best and second-best
/// Q-value is below the threshold — meaning the model is uncertain.
pub fn is_critical(q_values: &[f32; ACTION_SPACE], mask: &[bool; ACTION_SPACE], threshold: f32) -> bool {
    let mut valid: Vec<f32> = q_values
        .iter()
        .zip(mask.iter())
        .filter(|&(_, m)| *m)
        .map(|(&q, _)| q)
        .filter(|q| q.is_finite())
        .collect();

    if valid.len() <= 1 {
        return false;
    }

    valid.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    (valid[0] - valid[1]) < threshold
}

/// Numerically stable softmax over values, respecting the action mask.
///
/// Invalid actions (where mask is false) get probability 0.0.
/// Returns a 46-element probability distribution.
pub fn masked_softmax(values: &[f32; ACTION_SPACE], mask: &[bool; ACTION_SPACE], temperature: f32) -> [f32; ACTION_SPACE] {
    let mut result = [0.0_f32; ACTION_SPACE];
    let tau = if temperature.abs() < 1e-8 { 1e-8 } else { temperature };

    // Find max among valid values for numerical stability
    let max_val = values
        .iter()
        .zip(mask.iter())
        .filter(|&(_, m)| *m)
        .map(|(&v, _)| v)
        .filter(|v| v.is_finite())
        .fold(f32::NEG_INFINITY, f32::max);

    if !max_val.is_finite() {
        return result;
    }

    // Compute exp((v - max) / tau) for valid actions
    let mut sum = 0.0_f32;
    for i in 0..ACTION_SPACE {
        if mask[i] && values[i].is_finite() {
            let exp_val = ((values[i] - max_val) / tau).exp();
            result[i] = exp_val;
            sum += exp_val;
        }
    }

    // Normalize
    if sum > 0.0 {
        for v in &mut result {
            *v /= sum;
        }
    }

    result
}

/// Z-score normalize search values so they're comparable to Q-value scale.
///
/// Search values are raw score deltas in Mahjong points (~[-50K, +50K]).
/// Q-values are log-scale (~[-15, 3]). Without normalization, softmax on
/// raw search values produces a degenerate one-hot distribution.
///
/// After z-scoring, values are in standard-deviation units (~[-2, +2]),
/// making the temperature parameter work consistently for both distributions.
pub fn normalize_search_values(
    search_values: &[f32; ACTION_SPACE],
    mask: &[bool; ACTION_SPACE],
) -> [f32; ACTION_SPACE] {
    let mut result = [0.0_f32; ACTION_SPACE];

    // Collect valid search values
    let valid: Vec<f32> = search_values
        .iter()
        .zip(mask.iter())
        .filter(|&(_, &m)| m)
        .map(|(&v, _)| v)
        .collect();

    if valid.len() <= 1 {
        // Single or no valid actions — nothing to normalize
        return result;
    }

    let n = valid.len() as f32;
    let mean = valid.iter().sum::<f32>() / n;
    let variance = valid.iter().map(|&v| (v - mean).powi(2)).sum::<f32>() / n;
    let stddev = variance.sqrt().max(1e-8);

    for i in 0..ACTION_SPACE {
        if mask[i] {
            result[i] = (search_values[i] - mean) / stddev;
        }
    }

    result
}

/// Blend search values with logged Q-values to produce an improved policy.
///
/// Returns (improved_action, improved_distribution, search_effect).
pub fn blend(
    search_values: &[f32; ACTION_SPACE],
    logged_q: &[f32; ACTION_SPACE],
    mask: &[bool; ACTION_SPACE],
    w: f32,
    tau: f32,
) -> (u8, [f32; ACTION_SPACE], f32) {
    let p_search = masked_softmax(search_values, mask, tau);
    let p_logged = masked_softmax(logged_q, mask, tau);

    let mut pi_improved = [0.0_f32; ACTION_SPACE];
    for i in 0..ACTION_SPACE {
        pi_improved[i] = w.mul_add(p_search[i], (1.0 - w) * p_logged[i]);
    }

    // Argmax
    let improved_action = pi_improved
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map_or(0, |(i, _)| i as u8);

    // Search effect: L1 distance between p_logged and pi_improved
    let search_effect: f32 = p_logged
        .iter()
        .zip(pi_improved.iter())
        .map(|(a, b)| (a - b).abs())
        .sum::<f32>()
        / 2.0;

    (improved_action, pi_improved, search_effect)
}

/// Get the top-K valid action indices by Q-value (descending).
fn top_k_actions(q_values: &[f32; ACTION_SPACE], mask: &[bool; ACTION_SPACE], k: usize) -> Vec<usize> {
    let mut valid: Vec<(usize, f32)> = q_values
        .iter()
        .enumerate()
        .zip(mask.iter())
        .filter(|&((_, _), m)| *m)
        .map(|((i, &q), _)| (i, q))
        .filter(|&(_, q)| q.is_finite())
        .collect();

    valid.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    valid.truncate(k);
    valid.into_iter().map(|(i, _)| i).collect()
}

// ---------------------------------------------------------------------------
// Game replay & event alignment
// ---------------------------------------------------------------------------

/// Parsed decision point from MJSON replay.
struct MjsonDecision {
    /// Index of this NN-evaluated decision in the game (across all players).
    decision_idx: u32,
    /// The acting player.
    actor: u8,
    /// The event itself (for action extraction).
    event: Event,
    /// The metadata (contains q_values, mask_bits, etc.).
    meta: Metadata,
    /// Snapshot of the PlayerState at this decision point (before the event).
    player_state: PlayerState,
}

/// Event types that represent player decisions with NN evaluation.
const fn is_action_event(event: &Event) -> bool {
    matches!(
        event,
        Event::Dahai { .. }
            | Event::Reach { .. }
            | Event::Chi { .. }
            | Event::Pon { .. }
            | Event::Ankan { .. }
            | Event::Kakan { .. }
            | Event::Daiminkan { .. }
            | Event::None
    )
}

/// Extract the action index (0-45) from an MJSON event.
fn event_to_action(event: &Event) -> Option<u8> {
    match event {
        Event::Dahai { pai, .. } => {
            Some(pai.as_u8())
        }
        Event::Reach { .. } => Some(37),
        Event::Chi { pai, consumed, .. } => {
            // Determine chi type from called tile position
            let pai_norm = pai.deaka().as_u8();
            let mut c_norms: Vec<u8> = consumed.iter().map(|t| t.deaka().as_u8()).collect();
            c_norms.sort();
            if c_norms.len() == 2 {
                if pai_norm < c_norms[0] {
                    Some(38) // chi_low
                } else if pai_norm > c_norms[1] {
                    Some(40) // chi_high
                } else {
                    Some(39) // chi_mid
                }
            } else {
                Some(38) // fallback
            }
        }
        Event::Pon { .. } => Some(41),
        Event::Daiminkan { .. } | Event::Kakan { .. } | Event::Ankan { .. } => Some(42),
        Event::None => Some(45), // pass
        _ => None,
    }
}

/// Parse a .json.gz MJSON game file and extract events with metadata.
///
/// Returns a vector of `EventExt` (event + optional metadata) for the entire game.
fn parse_mjson_game(path: &Path) -> Result<Vec<EventExt>> {
    let file = fs::File::open(path)
        .with_context(|| format!("failed to open {}", path.display()))?;
    let decoder = flate2::read::GzDecoder::new(file);
    let reader = BufReader::new(decoder);

    let mut events = Vec::new();
    for (line_no, line) in reader.lines().enumerate() {
        let line = line.with_context(|| format!("line {}", line_no + 1))?;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let event_ext: EventExt = serde_json::from_str(trimmed)
            .with_context(|| format!("failed to parse line {}: {trimmed}", line_no + 1))?;
        events.push(event_ext);
    }
    Ok(events)
}

/// Load a Q-value sidecar (.msgpack) file.
fn load_sidecar(path: &Path) -> Result<Sidecar> {
    let data = fs::read(path)
        .with_context(|| format!("failed to read {}", path.display()))?;
    rmp_serde::from_slice(&data)
        .with_context(|| format!("failed to deserialize {}", path.display()))
}

/// Replay a game and extract decision points with aligned sidecar data.
///
/// For each kyoku in the game:
///  1. Create fresh PlayerStates with record_events enabled.
///  2. Feed events through PlayerState::update().
///  3. At each NN-evaluated decision point, capture the PlayerState snapshot.
///  4. Align with the sidecar's per-player decision records.
fn replay_and_extract(
    events: &[EventExt],
) -> Result<Vec<MjsonDecision>> {
    let mut decisions = Vec::new();
    let mut player_states: [PlayerState; 4] = std::array::from_fn(|i| {
        let mut ps = PlayerState::new(i as u8);
        ps.set_record_events(true);
        ps
    });

    // Track per-player decision indices for sidecar alignment
    let mut player_decision_idx: [usize; 4] = [0; 4];
    let mut global_decision_idx: u32 = 0;

    for event_ext in events {
        let event = &event_ext.event;

        // Check if this is a decision event with metadata containing Q-values
        if let Some(meta) = event_ext.meta.as_ref().filter(|m| m.q_values.is_some())
            && is_action_event(event)
        {
            let Some(actor) = event.actor() else { continue };

            // Snapshot the player state BEFORE processing this event
            let state_snapshot = player_states[actor as usize].clone();

            decisions.push(MjsonDecision {
                decision_idx: global_decision_idx,
                actor,
                event: event.clone(),
                meta: meta.clone(),
                player_state: state_snapshot,
            });

            player_decision_idx[actor as usize] += 1;
            global_decision_idx += 1;
        }

        // Process the event through all player states
        if matches!(event, Event::StartKyoku { .. }) {
            // Reset player states for new kyoku
            player_states = std::array::from_fn(|i| {
                let mut ps = PlayerState::new(i as u8);
                ps.set_record_events(true);
                ps
            });
            // Reset per-player counters for sidecar alignment within kyoku
            // Note: sidecar indices are global per player across the whole game,
            // so we do NOT reset player_decision_idx here.
        }

        for ps in &mut player_states {
            ps.update(event)
                .with_context(|| format!("replay failed on event {event:?}"))?;
        }
    }

    Ok(decisions)
}

// ---------------------------------------------------------------------------
// Search integration
// ---------------------------------------------------------------------------

/// Run particle-based search for a single decision point.
///
/// Returns per-action mean score deltas as search values (46 slots).
fn run_search(
    state: &PlayerState,
    actions: &[usize],
    config: &PostHocConfig,
    rng: &mut ChaCha12Rng,
) -> Result<[f32; ACTION_SPACE]> {
    let mut search_values = [0.0_f32; ACTION_SPACE];

    if actions.is_empty() {
        return Ok(search_values);
    }

    let particle_config = ParticleConfig::new(config.n_particles);
    let (particles, _attempts) = particle::generate_particles(state, &particle_config, rng)
        .context("failed to generate particles")?;

    if particles.is_empty() {
        return Ok(search_values);
    }

    let replayed = simulator::replay_player_states(state)
        .context("failed to replay player states")?;
    let base = simulator::derive_midgame_context_base(state.event_history());

    let player_id = state.player_id();
    let use_smart = config.use_smart_policy;
    let n_particles = particles.len();
    let seeds: Vec<u64> = (0..n_particles).map(|_| rng.next_u64()).collect();

    type ActionDeltas = (usize, [i32; 4]);

    // Parallel rollouts: for each particle, evaluate all actions
    let per_particle: Result<Vec<Vec<ActionDeltas>>> = particles
        .into_par_iter()
        .enumerate()
        .map(|(i, p)| {
            let board_state = simulator::build_midgame_board_state_with_base(
                state, &replayed, &p, &base,
            )?;
            let initial_scores = board_state.board.scores;

            let mut particle_rng = ChaCha12Rng::seed_from_u64(seeds[i]);
            let mut action_results = Vec::with_capacity(actions.len());

            for &action in actions {
                let action_seed = particle_rng.next_u64();
                let mut action_rng = if use_smart {
                    Some(ChaCha12Rng::seed_from_u64(action_seed))
                } else {
                    None
                };

                let result = simulator::simulate_action_rollout_prebuilt(
                    &board_state,
                    initial_scores,
                    player_id,
                    action,
                    action_rng.as_mut(),
                )?;
                action_results.push((action, result.deltas));
            }

            Ok(action_results)
        })
        .collect();

    let per_particle = per_particle?;

    // Aggregate: mean delta for the acting player across particles
    let mut action_sums: HashMap<usize, f64> = HashMap::new();
    let mut action_counts: HashMap<usize, usize> = HashMap::new();

    for particle_results in &per_particle {
        for &(action, deltas) in particle_results {
            let delta = deltas[player_id as usize] as f64;
            *action_sums.entry(action).or_insert(0.0) += delta;
            *action_counts.entry(action).or_insert(0) += 1;
        }
    }

    for (&action, &sum) in &action_sums {
        let count = action_counts[&action];
        if count > 0 && action < ACTION_SPACE {
            search_values[action] = (sum / count as f64) as f32;
        }
    }

    Ok(search_values)
}

// ---------------------------------------------------------------------------
// Pipeline
// ---------------------------------------------------------------------------

/// Process a single game: replay, identify critical decisions, search, blend.
pub fn process_game(
    game_path: &Path,
    sidecar_path: &Path,
    config: &PostHocConfig,
    rng: &mut ChaCha12Rng,
) -> Result<Vec<DecisionRecord>> {
    let game_id_raw = game_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("unknown");
    // Strip .json from stem if present (file is .json.gz)
    let game_id = game_id_raw.strip_suffix(".json").unwrap_or(game_id_raw).to_owned();

    let events = parse_mjson_game(game_path)
        .with_context(|| format!("parsing game {game_id}"))?;
    let sidecar = load_sidecar(sidecar_path)
        .with_context(|| format!("loading sidecar for {game_id}"))?;

    ensure!(sidecar.version == 1, "unsupported sidecar version {}", sidecar.version);

    let mjson_decisions = replay_and_extract(&events)
        .with_context(|| format!("replaying game {game_id}"))?;

    // For each decision, align with sidecar, check criticality, search, blend
    // We need to track per-player sidecar indices
    let mut player_sidecar_idx: [usize; 4] = [0; 4];
    let mut records = Vec::new();

    for decision in &mjson_decisions {
        let actor = decision.actor;
        let sidecar_idx = player_sidecar_idx[actor as usize];
        player_sidecar_idx[actor as usize] += 1;

        // Get sidecar entry for this decision
        let player_key = actor.to_string();
        let player_decisions = sidecar.players.get(&player_key);
        let sidecar_entry = player_decisions.and_then(|d| d.get(sidecar_idx));

        // Use sidecar Q-values if available, otherwise fall back to MJSON metadata
        let (full_q, mask) = if let Some(entry) = sidecar_entry {
            expand_q_values(&entry.q_values, entry.mask_bits)
        } else if let Some(ref q_vals) = decision.meta.q_values {
            // Fallback: use MJSON metadata (requires mask_bits)
            let mask_bits = decision.meta.mask_bits.unwrap_or(0);
            expand_q_values(q_vals, mask_bits)
        } else {
            continue; // No Q-values available
        };

        // Get the logged action
        let logged_action = if let Some(entry) = sidecar_entry {
            entry.action.unwrap_or_else(|| event_to_action(&decision.event).unwrap_or(0))
        } else {
            event_to_action(&decision.event).unwrap_or(0)
        };

        // Check criticality
        if !is_critical(&full_q, &mask, config.criticality_threshold) {
            continue;
        }

        // Get top-K actions to search
        let actions = top_k_actions(&full_q, &mask, config.top_k_actions);
        if actions.is_empty() {
            continue;
        }

        // Run search
        let search_values = match run_search(&decision.player_state, &actions, config, rng) {
            Ok(sv) => sv,
            Err(e) => {
                eprintln!("search failed for {game_id} decision {}: {e}", decision.decision_idx);
                continue;
            }
        };

        // Normalize search values before blending (if enabled)
        let blendable_sv = if config.normalize_search {
            normalize_search_values(&search_values, &mask)
        } else {
            search_values
        };

        // Blend
        let (improved_action, improved_dist, search_effect) = blend(
            &blendable_sv,
            &full_q,
            &mask,
            config.blend_weight,
            config.temperature,
        );

        records.push(DecisionRecord {
            game_id: game_id.clone(),
            decision_idx: decision.decision_idx,
            actor,
            logged_action,
            logged_q_values: full_q.to_vec(),
            mask: mask.to_vec(),
            search_values: search_values.to_vec(),
            improved_action,
            improved_distribution: improved_dist.to_vec(),
            search_effect,
            num_particles: config.n_particles as u32,
        });
    }

    Ok(records)
}

/// Discover game/sidecar pairs in the given directories.
pub fn discover_game_pairs(
    game_dir: &Path,
    sidecar_dir: &Path,
) -> Result<Vec<(PathBuf, PathBuf)>> {
    let mut pairs = Vec::new();

    let entries: Vec<_> = fs::read_dir(game_dir)
        .with_context(|| format!("reading game dir {}", game_dir.display()))?
        .filter_map(|e| e.ok())
        .collect();

    for entry in &entries {
        let path = entry.path();
        let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");

        if !name.ends_with(".json.gz") {
            continue;
        }

        // Derive sidecar name: strip .json.gz, add .msgpack
        let stem = name.strip_suffix(".json.gz").unwrap_or(name);
        let sidecar_name = format!("{stem}.msgpack");
        let sidecar_path = sidecar_dir.join(&sidecar_name);

        if sidecar_path.exists() {
            pairs.push((path.clone(), sidecar_path));
        }
    }

    pairs.sort_by(|a, b| a.0.cmp(&b.0));
    Ok(pairs)
}

/// Process a batch of games sequentially, collecting all decision records.
///
/// Games are processed sequentially to keep RNG deterministic; within each
/// game, particle rollouts run in parallel via rayon.
pub fn process_batch(
    pairs: &[(PathBuf, PathBuf)],
    config: &PostHocConfig,
) -> Result<Vec<DecisionRecord>> {
    let mut rng = match config.seed {
        Some(seed) => ChaCha12Rng::seed_from_u64(seed),
        None => ChaCha12Rng::from_os_rng(),
    };

    let mut all_records = Vec::new();
    let total = pairs.len();

    for (i, (game_path, sidecar_path)) in pairs.iter().enumerate() {
        let game_name = game_path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("?");

        match process_game(game_path, sidecar_path, config, &mut rng) {
            Ok(records) => {
                let n = records.len();
                if n > 0 {
                    eprintln!(
                        "[{}/{}] {}: {} critical decisions searched",
                        i + 1,
                        total,
                        game_name,
                        n,
                    );
                }
                all_records.extend(records);
            }
            Err(e) => {
                eprintln!("[{}/{}] {}: ERROR: {e:#}", i + 1, total, game_name);
            }
        }
    }

    Ok(all_records)
}

/// Write decision records to a MessagePack file.
pub fn write_records(records: &[DecisionRecord], output_path: &Path) -> Result<()> {
    let data = rmp_serde::to_vec(records)
        .context("failed to serialize records")?;
    fs::write(output_path, &data)
        .with_context(|| format!("failed to write {}", output_path.display()))?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_expand_q_values_basic() {
        // mask_bits = 0b111 (actions 0, 1, 2 valid)
        let compact = vec![1.0, 2.0, 3.0];
        let mask_bits = 0b111_u64;
        let (full_q, mask) = expand_q_values(&compact, mask_bits);

        assert!(mask[0]);
        assert!(mask[1]);
        assert!(mask[2]);
        assert!(!mask[3]);
        assert!((full_q[0] - 1.0).abs() < 1e-6);
        assert!((full_q[1] - 2.0).abs() < 1e-6);
        assert!((full_q[2] - 3.0).abs() < 1e-6);
        assert!(full_q[3] == f32::NEG_INFINITY);
    }

    #[test]
    fn test_expand_q_values_sparse() {
        // mask_bits: actions 0 and 45 valid
        let compact = vec![10.0, 20.0];
        let mask_bits = 1_u64 | (1_u64 << 45);
        let (full_q, mask) = expand_q_values(&compact, mask_bits);

        assert!(mask[0]);
        assert!(!mask[1]);
        assert!(mask[45]);
        assert!((full_q[0] - 10.0).abs() < 1e-6);
        assert!((full_q[45] - 20.0).abs() < 1e-6);
    }

    #[test]
    fn test_is_critical_large_gap() {
        let mut q = [f32::NEG_INFINITY; ACTION_SPACE];
        let mut mask = [false; ACTION_SPACE];
        q[0] = 5.0;
        q[1] = 1.0;
        mask[0] = true;
        mask[1] = true;

        // Gap = 4.0, threshold = 0.5 → not critical
        assert!(!is_critical(&q, &mask, 0.5));
    }

    #[test]
    fn test_is_critical_small_gap() {
        let mut q = [f32::NEG_INFINITY; ACTION_SPACE];
        let mut mask = [false; ACTION_SPACE];
        q[0] = 1.0;
        q[1] = 0.8;
        mask[0] = true;
        mask[1] = true;

        // Gap = 0.2, threshold = 0.5 → critical
        assert!(is_critical(&q, &mask, 0.5));
    }

    #[test]
    fn test_is_critical_single_action() {
        let mut q = [f32::NEG_INFINITY; ACTION_SPACE];
        let mut mask = [false; ACTION_SPACE];
        q[0] = 1.0;
        mask[0] = true;

        // Single valid action → not critical
        assert!(!is_critical(&q, &mask, 0.5));
    }

    #[test]
    fn test_masked_softmax_basic() {
        let mut values = [f32::NEG_INFINITY; ACTION_SPACE];
        let mut mask = [false; ACTION_SPACE];
        values[0] = 1.0;
        values[1] = 1.0;
        mask[0] = true;
        mask[1] = true;

        let probs = masked_softmax(&values, &mask, 1.0);

        // Equal values → equal probabilities
        assert!((probs[0] - 0.5).abs() < 1e-5);
        assert!((probs[1] - 0.5).abs() < 1e-5);
        assert!((probs[2]).abs() < 1e-8);
    }

    #[test]
    fn test_masked_softmax_temperature() {
        let mut values = [f32::NEG_INFINITY; ACTION_SPACE];
        let mut mask = [false; ACTION_SPACE];
        values[0] = 2.0;
        values[1] = 0.0;
        mask[0] = true;
        mask[1] = true;

        // Low temperature → sharper distribution
        let probs_low = masked_softmax(&values, &mask, 0.1);
        let probs_high = masked_softmax(&values, &mask, 10.0);

        assert!(probs_low[0] > probs_high[0]); // Lower temp → more peaked
    }

    #[test]
    fn test_masked_softmax_no_valid() {
        let values = [f32::NEG_INFINITY; ACTION_SPACE];
        let mask = [false; ACTION_SPACE];
        let probs = masked_softmax(&values, &mask, 1.0);

        // All zeros
        for p in &probs {
            assert!((*p).abs() < 1e-8);
        }
    }

    #[test]
    fn test_blend_pure_search() {
        let mut search = [0.0_f32; ACTION_SPACE];
        let mut logged = [f32::NEG_INFINITY; ACTION_SPACE];
        let mut mask = [false; ACTION_SPACE];

        search[0] = 100.0;
        search[1] = 50.0;
        logged[0] = -1.0;
        logged[1] = 1.0;
        mask[0] = true;
        mask[1] = true;

        // w=1.0 → pure search → action 0 should win
        let (action, _, _) = blend(&search, &logged, &mask, 1.0, 1.0);
        assert_eq!(action, 0);
    }

    #[test]
    fn test_blend_pure_logged() {
        let mut search = [0.0_f32; ACTION_SPACE];
        let mut logged = [f32::NEG_INFINITY; ACTION_SPACE];
        let mut mask = [false; ACTION_SPACE];

        search[0] = 100.0;
        search[1] = 50.0;
        logged[0] = -1.0;
        logged[1] = 1.0;
        mask[0] = true;
        mask[1] = true;

        // w=0.0 → pure logged → action 1 should win
        let (action, _, _) = blend(&search, &logged, &mask, 0.0, 1.0);
        assert_eq!(action, 1);
    }

    #[test]
    fn test_blend_distribution_sums_to_one() {
        let mut search = [0.0_f32; ACTION_SPACE];
        let mut logged = [f32::NEG_INFINITY; ACTION_SPACE];
        let mut mask = [false; ACTION_SPACE];

        for i in 0..5 {
            search[i] = (i as f32) * 10.0;
            logged[i] = -(i as f32);
            mask[i] = true;
        }

        let (_, dist, _) = blend(&search, &logged, &mask, 0.3, 1.0);
        let sum: f32 = dist.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "distribution sum = {sum}");
    }

    #[test]
    fn test_top_k_actions() {
        let mut q = [f32::NEG_INFINITY; ACTION_SPACE];
        let mut mask = [false; ACTION_SPACE];

        q[0] = 1.0;
        q[5] = 3.0;
        q[10] = 2.0;
        q[15] = 0.5;
        mask[0] = true;
        mask[5] = true;
        mask[10] = true;
        mask[15] = true;

        let top3 = top_k_actions(&q, &mask, 3);
        assert_eq!(top3, vec![5, 10, 0]);
    }

    #[test]
    fn test_top_k_fewer_valid_than_k() {
        let mut q = [f32::NEG_INFINITY; ACTION_SPACE];
        let mut mask = [false; ACTION_SPACE];

        q[0] = 1.0;
        q[1] = 2.0;
        mask[0] = true;
        mask[1] = true;

        let top5 = top_k_actions(&q, &mask, 5);
        assert_eq!(top5.len(), 2);
    }

    #[test]
    fn test_normalize_search_values_equal() {
        // All equal values → all zero after normalization
        let mut sv = [0.0_f32; ACTION_SPACE];
        let mut mask = [false; ACTION_SPACE];
        sv[0] = 1000.0;
        sv[1] = 1000.0;
        sv[2] = 1000.0;
        mask[0] = true;
        mask[1] = true;
        mask[2] = true;

        let result = normalize_search_values(&sv, &mask);

        for i in 0..3 {
            assert!(result[i].abs() < 1e-5, "equal values should normalize to ~0, got {}", result[i]);
        }
        // Invalid slots should remain 0
        assert!(result[3].abs() < 1e-8);
    }

    #[test]
    fn test_normalize_search_values_known() {
        // Known spread: values [100, 200, 300] → mean=200, stddev≈81.65
        // z-scores: [-1.2247, 0.0, +1.2247]
        let mut sv = [0.0_f32; ACTION_SPACE];
        let mut mask = [false; ACTION_SPACE];
        sv[0] = 100.0;
        sv[1] = 200.0;
        sv[2] = 300.0;
        mask[0] = true;
        mask[1] = true;
        mask[2] = true;

        let result = normalize_search_values(&sv, &mask);

        assert!((result[1]).abs() < 1e-5, "mean value should normalize to ~0");
        assert!(result[0] < -1.0, "below-mean should be negative: {}", result[0]);
        assert!(result[2] > 1.0, "above-mean should be positive: {}", result[2]);
        // Symmetric: result[0] ≈ -result[2]
        assert!((result[0] + result[2]).abs() < 1e-5, "should be symmetric");
    }

    #[test]
    fn test_normalize_search_values_single() {
        // Single valid action → unchanged (returns all zeros)
        let mut sv = [0.0_f32; ACTION_SPACE];
        let mut mask = [false; ACTION_SPACE];
        sv[0] = 5000.0;
        mask[0] = true;

        let result = normalize_search_values(&sv, &mask);

        // Single action: nothing to normalize
        assert!(result[0].abs() < 1e-8);
    }

    #[test]
    fn test_normalize_search_values_large_spread() {
        // Realistic scale: [-30000, +10000] → should compress to ~[-1, +1]
        let mut sv = [0.0_f32; ACTION_SPACE];
        let mut mask = [false; ACTION_SPACE];
        sv[0] = -30000.0;
        sv[1] = 10000.0;
        mask[0] = true;
        mask[1] = true;

        let result = normalize_search_values(&sv, &mask);

        // After z-scoring, values should be in [-2, +2] range
        assert!(result[0].abs() < 3.0, "should be in std-dev units: {}", result[0]);
        assert!(result[1].abs() < 3.0, "should be in std-dev units: {}", result[1]);
        // They should be symmetric opposites
        assert!((result[0] + result[1]).abs() < 1e-5);
    }

    #[test]
    fn test_blend_with_normalized_search_not_one_hot() {
        // Main motivation test: normalized search values should produce
        // a non-degenerate blend (not one-hot from raw score deltas).
        let mut search = [0.0_f32; ACTION_SPACE];
        let mut logged = [f32::NEG_INFINITY; ACTION_SPACE];
        let mut mask = [false; ACTION_SPACE];

        // Realistic raw search values (Mahjong points)
        search[0] = 5000.0;
        search[1] = 3000.0;
        search[2] = -1000.0;
        logged[0] = -2.0;
        logged[1] = -1.5;
        logged[2] = -3.0;
        mask[0] = true;
        mask[1] = true;
        mask[2] = true;

        // Without normalization: softmax(5000, 3000, -1000) ≈ [1, 0, 0] (degenerate)
        let (_, raw_dist, _) = blend(&search, &logged, &mask, 0.3, 1.0);
        // The search component will be almost entirely on action 0
        assert!(raw_dist[0] > 0.99 * 0.3, "raw search should dominate action 0");

        // With normalization: spread becomes ~[-1, 0, +1], softmax is smoother
        let normed = normalize_search_values(&search, &mask);
        let (_, norm_dist, _) = blend(&normed, &logged, &mask, 0.3, 1.0);
        // Distribution should be more spread — action 1 should get meaningful weight
        assert!(norm_dist[1] > 0.1, "normalized blend should give action 1 meaningful weight, got {}", norm_dist[1]);
    }
}
