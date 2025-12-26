//! Engram CLI - Generate art from neural metrics.

use anyhow::Result;
use clap::{Parser, ValueEnum};
use engram::generators::glyph::{GlyphGenerator, GlyphStyle};
use engram::generators::stipple::{StippleGenerator, StippleStyle};
use engram::generators::svg_lines::{LineStyle, SvgLineGenerator};
use engram::generators::Generator;
use engram::metrics::NeuralMetrics;
use neuronic::{capture_snapshots, AggregatedMetrics, CaptureConfig, SubscriberConfig};
use std::fs;
use std::path::PathBuf;
use std::time::Duration;

#[derive(Parser)]
#[command(name = "engram")]
#[command(about = "Generate art from neural network activity metrics")]
#[command(version)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(clap::Subcommand)]
enum Commands {
    /// Generate a single piece from random/sample data
    Generate {
        /// Generator type to use
        #[arg(short, long, value_enum, default_value = "dendrite")]
        generator: GeneratorType,

        /// Output file path
        #[arg(short, long)]
        output: Option<PathBuf>,

        /// Seed for random generation (uses random if not specified)
        #[arg(short, long)]
        seed: Option<u64>,

        /// Width of the output (for SVG generators)
        #[arg(long, default_value = "512")]
        width: u32,

        /// Height of the output (for SVG generators)
        #[arg(long, default_value = "512")]
        height: u32,
    },

    /// Capture live metrics from neuronic/RabbitMQ and generate art
    Capture {
        /// RabbitMQ URL
        #[arg(long, default_value = "amqp://guest:guest@localhost:5672")]
        rabbitmq_url: String,

        /// RabbitMQ exchange
        #[arg(long, default_value = "amq.topic")]
        exchange: String,

        /// Topic pattern to subscribe to
        #[arg(short, long, default_value = "buswatch.snapshot")]
        topic: String,

        /// Number of snapshots to capture
        #[arg(short, long, default_value = "10")]
        count: usize,

        /// Interval between snapshots in seconds
        #[arg(short, long, default_value = "1")]
        interval: u64,

        /// Generator type to use
        #[arg(short, long, value_enum, default_value = "dendrite")]
        generator: GeneratorType,

        /// Output file path
        #[arg(short, long)]
        output: Option<PathBuf>,

        /// Width of the output
        #[arg(long, default_value = "512")]
        width: u32,

        /// Height of the output
        #[arg(long, default_value = "512")]
        height: u32,

        /// Also save metrics as JSON
        #[arg(long)]
        save_metrics: bool,
    },

    /// Generate a batch of pieces from sample data
    Batch {
        /// Number of pieces to generate
        #[arg(short, long, default_value = "100")]
        count: usize,

        /// Generator type to use
        #[arg(short, long, value_enum, default_value = "dendrite")]
        generator: GeneratorType,

        /// Output directory
        #[arg(short, long, default_value = "output")]
        output_dir: PathBuf,

        /// Starting seed
        #[arg(short, long, default_value = "0")]
        start_seed: u64,

        /// Width of the output
        #[arg(long, default_value = "512")]
        width: u32,

        /// Height of the output
        #[arg(long, default_value = "512")]
        height: u32,
    },

    /// Generate samples of all generator types
    Showcase {
        /// Output directory
        #[arg(short, long, default_value = "output/showcase")]
        output_dir: PathBuf,

        /// Seed for consistent results
        #[arg(short, long, default_value = "42")]
        seed: u64,
    },
}

#[derive(Clone, ValueEnum)]
enum GeneratorType {
    // Glyph styles
    GlyphMinimal,
    GlyphCircuit,
    GlyphStipple,
    GlyphNeural,
    GlyphAdaptive,

    // SVG line styles
    Flow,
    Dendrite,
    Radial,
    Grid,

    // Stipple styles
    StippleGradient,
    StippleClustered,
    StippleFlow,
    Halftone,
}

impl GeneratorType {
    fn name(&self) -> &'static str {
        match self {
            GeneratorType::GlyphMinimal => "glyph_minimal",
            GeneratorType::GlyphCircuit => "glyph_circuit",
            GeneratorType::GlyphStipple => "glyph_stipple",
            GeneratorType::GlyphNeural => "glyph_neural",
            GeneratorType::GlyphAdaptive => "glyph_adaptive",
            GeneratorType::Flow => "flow",
            GeneratorType::Dendrite => "dendrite",
            GeneratorType::Radial => "radial",
            GeneratorType::Grid => "grid",
            GeneratorType::StippleGradient => "stipple_gradient",
            GeneratorType::StippleClustered => "stipple_clustered",
            GeneratorType::StippleFlow => "stipple_flow",
            GeneratorType::Halftone => "halftone",
        }
    }

    fn create(&self, width: u32, height: u32) -> Box<dyn Generator> {
        match self {
            GeneratorType::GlyphMinimal => Box::new(GlyphGenerator::new(
                width as usize,
                height as usize,
                GlyphStyle::Minimal,
            )),
            GeneratorType::GlyphCircuit => Box::new(GlyphGenerator::new(
                width as usize,
                height as usize,
                GlyphStyle::Circuit,
            )),
            GeneratorType::GlyphStipple => Box::new(GlyphGenerator::new(
                width as usize,
                height as usize,
                GlyphStyle::Stipple,
            )),
            GeneratorType::GlyphNeural => Box::new(GlyphGenerator::new(
                width as usize,
                height as usize,
                GlyphStyle::Neural,
            )),
            GeneratorType::GlyphAdaptive => Box::new(GlyphGenerator::new(
                width as usize,
                height as usize,
                GlyphStyle::Adaptive,
            )),
            GeneratorType::Flow => Box::new(SvgLineGenerator::new(width, height, LineStyle::Flow)),
            GeneratorType::Dendrite => {
                Box::new(SvgLineGenerator::new(width, height, LineStyle::Dendrite))
            }
            GeneratorType::Radial => {
                Box::new(SvgLineGenerator::new(width, height, LineStyle::Radial))
            }
            GeneratorType::Grid => Box::new(SvgLineGenerator::new(width, height, LineStyle::Grid)),
            GeneratorType::StippleGradient => {
                Box::new(StippleGenerator::new(width, height, StippleStyle::Gradient))
            }
            GeneratorType::StippleClustered => Box::new(StippleGenerator::new(
                width,
                height,
                StippleStyle::Clustered,
            )),
            GeneratorType::StippleFlow => {
                Box::new(StippleGenerator::new(width, height, StippleStyle::Flow))
            }
            GeneratorType::Halftone => {
                Box::new(StippleGenerator::new(width, height, StippleStyle::Halftone))
            }
        }
    }

    fn all() -> Vec<GeneratorType> {
        vec![
            GeneratorType::GlyphMinimal,
            GeneratorType::GlyphCircuit,
            GeneratorType::GlyphStipple,
            GeneratorType::GlyphNeural,
            GeneratorType::GlyphAdaptive,
            GeneratorType::Flow,
            GeneratorType::Dendrite,
            GeneratorType::Radial,
            GeneratorType::Grid,
            GeneratorType::StippleGradient,
            GeneratorType::StippleClustered,
            GeneratorType::StippleFlow,
            GeneratorType::Halftone,
        ]
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive("engram=info".parse().unwrap()),
        )
        .init();

    let cli = Cli::parse();

    match cli.command {
        Commands::Generate {
            generator,
            output,
            seed,
            width,
            height,
        } => {
            let gen = generator.create(width, height);
            let seed = seed.unwrap_or_else(rand::random);
            let metrics = NeuralMetrics::sample(seed);

            println!("Generating {} with seed {}...", gen.name(), seed);
            let result = gen.generate(&metrics);

            let output_path = output
                .unwrap_or_else(|| PathBuf::from(format!("engram_{}.{}", seed, gen.extension())));

            fs::write(&output_path, &result)?;
            println!("Saved to {}", output_path.display());
        }

        Commands::Capture {
            rabbitmq_url,
            exchange,
            topic,
            count,
            interval,
            generator,
            output,
            width,
            height,
            save_metrics,
        } => {
            println!("Connecting to RabbitMQ at {}...", rabbitmq_url);
            println!("Subscribing to topic: {}", topic);
            println!(
                "Capturing {} snapshots at {}s intervals...",
                count, interval
            );

            let subscriber_config = SubscriberConfig {
                url: rabbitmq_url,
                exchange,
            };

            let capture_config = CaptureConfig {
                count,
                interval: Duration::from_secs(interval),
                timeout: Duration::from_secs(30),
                ..Default::default()
            };

            let snapshots = capture_snapshots(&subscriber_config, &topic, &capture_config).await?;

            println!("Captured {} snapshots", snapshots.len());

            // Aggregate metrics
            let aggregated = AggregatedMetrics::from_snapshots(&snapshots);
            let metrics = NeuralMetrics::from_aggregated(&aggregated, &snapshots);

            println!("\nMetrics summary:");
            println!("  Nodes: {}", metrics.node_count);
            println!("  Connections: {}", metrics.connection_count);
            println!("  Synapse rate: {:.1} msg/s", metrics.synapse_rate);
            println!("  Peak burst: {:.1} msg/s", metrics.peak_burst);
            println!("  Total throughput: {} messages", metrics.total_throughput);
            println!(
                "  Health: {:.0}% healthy, {:.0}% warning, {:.0}% critical",
                metrics.health_ratio * 100.0,
                metrics.warning_ratio * 100.0,
                metrics.critical_ratio * 100.0
            );

            // Generate art
            let gen = generator.create(width, height);
            println!("\nGenerating {} art...", gen.name());
            let result = gen.generate(&metrics);

            let timestamp = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs();

            let output_path = output.unwrap_or_else(|| {
                PathBuf::from(format!("engram_live_{}.{}", timestamp, gen.extension()))
            });

            fs::write(&output_path, &result)?;
            println!("Saved art to {}", output_path.display());

            // Optionally save metrics
            if save_metrics {
                let metrics_path = output_path.with_extension("json");
                let metrics_json = serde_json::to_string_pretty(&metrics)?;
                fs::write(&metrics_path, metrics_json)?;
                println!("Saved metrics to {}", metrics_path.display());
            }
        }

        Commands::Batch {
            count,
            generator,
            output_dir,
            start_seed,
            width,
            height,
        } => {
            fs::create_dir_all(&output_dir)?;

            let gen = generator.create(width, height);
            println!("Generating {} {} pieces...", count, gen.name());

            for i in 0..count {
                let seed = start_seed + i as u64;
                let metrics = NeuralMetrics::sample(seed);
                let result = gen.generate(&metrics);

                let filename = format!("{}_{:04}.{}", gen.name(), i, gen.extension());
                let path = output_dir.join(&filename);
                fs::write(&path, &result)?;

                if (i + 1) % 10 == 0 {
                    println!("  Generated {}/{}", i + 1, count);
                }
            }

            println!("Done! {} pieces saved to {}", count, output_dir.display());
        }

        Commands::Showcase { output_dir, seed } => {
            fs::create_dir_all(&output_dir)?;

            let metrics = NeuralMetrics::sample(seed);
            println!("Generating showcase with seed {}...", seed);
            println!("Metrics: {:?}", metrics);

            for gen_type in GeneratorType::all() {
                let gen = gen_type.create(512, 512);
                let result = gen.generate(&metrics);

                let filename = format!("{}.{}", gen_type.name(), gen.extension());
                let path = output_dir.join(&filename);
                fs::write(&path, &result)?;
                println!("  Created {}", filename);
            }

            println!("Done! Showcase saved to {}", output_dir.display());
        }
    }

    Ok(())
}
