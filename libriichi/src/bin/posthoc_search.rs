//! Post-hoc search pipeline CLI.
//!
//! Usage:
//!   posthoc_search --game-dir /path/to/games --sidecar-dir /path/to/qvalues \
//!                  --output /path/to/output.msgpack [options]

use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result};
use clap::Parser;
use riichi::search::posthoc::{self, PostHocConfig};

#[derive(Parser, Debug)]
#[command(
    name = "posthoc_search",
    about = "Post-hoc search pipeline for ExIt training targets"
)]
struct Args {
    /// Directory containing .json.gz game logs.
    #[arg(long)]
    game_dir: PathBuf,

    /// Directory containing .msgpack Q-value sidecars.
    #[arg(long)]
    sidecar_dir: PathBuf,

    /// Output path for MessagePack decision records.
    #[arg(long)]
    output: PathBuf,

    /// Blend weight for search values (0.0 = pure logged, 1.0 = pure search).
    #[arg(long, default_value = "0.3")]
    blend_weight: f32,

    /// Temperature for softmax normalization.
    #[arg(long, default_value = "1.0")]
    temperature: f32,

    /// Criticality threshold (gap between top-2 Q-values).
    #[arg(long, default_value = "0.5")]
    criticality_threshold: f32,

    /// Number of particles for search.
    #[arg(long, default_value = "50")]
    n_particles: usize,

    /// Maximum rollout steps per particle.
    #[arg(long, default_value = "100")]
    max_rollout_steps: u32,

    /// Whether to use smart heuristic policy during rollouts.
    #[arg(long, default_value = "true")]
    use_smart_policy: bool,

    /// Number of top actions to evaluate per decision.
    #[arg(long, default_value = "5")]
    top_k_actions: usize,

    /// RNG seed for reproducibility.
    #[arg(long)]
    seed: Option<u64>,

    /// Disable z-score normalization of search values before blending.
    /// By default, search values (raw Mahjong points) are z-scored to be
    /// comparable with Q-values (log-scale). Use this flag to blend raw values.
    #[arg(long)]
    no_normalize_search: bool,

    /// Number of rayon threads (0 = auto).
    #[arg(long, default_value = "0")]
    threads: usize,

    /// Maximum number of games to process (0 = all).
    #[arg(long, default_value = "0")]
    max_games: usize,
}

fn main() -> Result<()> {
    let args = Args::parse();

    // Configure rayon thread pool
    if args.threads > 0 {
        rayon::ThreadPoolBuilder::new()
            .num_threads(args.threads)
            .build_global()
            .context("failed to configure rayon thread pool")?;
    }

    let config = PostHocConfig {
        blend_weight: args.blend_weight,
        temperature: args.temperature,
        criticality_threshold: args.criticality_threshold,
        n_particles: args.n_particles,
        max_rollout_steps: args.max_rollout_steps,
        use_smart_policy: args.use_smart_policy,
        top_k_actions: args.top_k_actions,
        seed: args.seed,
        normalize_search: !args.no_normalize_search,
    };

    eprintln!("=== Post-Hoc Search Pipeline ===");
    eprintln!("Game dir:     {}", args.game_dir.display());
    eprintln!("Sidecar dir:  {}", args.sidecar_dir.display());
    eprintln!("Output:       {}", args.output.display());
    eprintln!("Config:       blend_weight={}, temperature={}, criticality_threshold={}",
        config.blend_weight, config.temperature, config.criticality_threshold);
    eprintln!("              n_particles={}, max_rollout_steps={}, top_k={}",
        config.n_particles, config.max_rollout_steps, config.top_k_actions);
    eprintln!("              use_smart_policy={}, normalize_search={}, seed={:?}",
        config.use_smart_policy, config.normalize_search, config.seed);
    eprintln!();

    // Discover game/sidecar pairs
    let start = Instant::now();
    let mut pairs = posthoc::discover_game_pairs(&args.game_dir, &args.sidecar_dir)
        .context("failed to discover game pairs")?;

    if pairs.is_empty() {
        eprintln!("No game/sidecar pairs found. Check your directories.");
        return Ok(());
    }

    eprintln!("Found {} game/sidecar pairs", pairs.len());

    if args.max_games > 0 && pairs.len() > args.max_games {
        pairs.truncate(args.max_games);
        eprintln!("Limiting to {} games", args.max_games);
    }

    // Process games
    let records = posthoc::process_batch(&pairs, &config)
        .context("batch processing failed")?;

    let elapsed = start.elapsed();

    // Write output
    posthoc::write_records(&records, &args.output)
        .context("failed to write output")?;

    // Summary
    let output_size = std::fs::metadata(&args.output)
        .map(|m| m.len())
        .unwrap_or(0);

    eprintln!();
    eprintln!("=== Summary ===");
    eprintln!("Games processed:     {}", pairs.len());
    eprintln!("Critical decisions:  {}", records.len());
    eprintln!("Avg per game:        {:.1}", records.len() as f64 / pairs.len().max(1) as f64);
    eprintln!("Output size:         {:.1} MB", output_size as f64 / 1_048_576.0);
    eprintln!("Total time:          {:.1}s", elapsed.as_secs_f64());
    eprintln!("Throughput:          {:.2} games/sec", pairs.len() as f64 / elapsed.as_secs_f64());

    if !records.is_empty() {
        let avg_effect: f32 = records.iter().map(|r| r.search_effect).sum::<f32>()
            / records.len() as f32;
        let changed = records.iter().filter(|r| r.improved_action != r.logged_action).count();
        eprintln!("Avg search effect:   {avg_effect:.4}");
        eprintln!("Decisions changed:   {} ({:.1}%)",
            changed,
            changed as f64 / records.len() as f64 * 100.0,
        );
    }

    Ok(())
}
