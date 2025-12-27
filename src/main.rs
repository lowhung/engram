//! Engram CLI - Generate bold graphic visualizations from neural network topology.

use anyhow::Result;
use clap::{Parser, ValueEnum};
use engram::config::EngramConfig;
use engram::generators::graph::{palette::ColorPalette, GraphGenerator, GraphStyle};
use engram::generators::Generator;
use engram::metrics::NeuralMetrics;
use neuronic::{capture_snapshots, AggregatedMetrics, CaptureConfig, SubscriberConfig};
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Duration;

#[derive(Parser)]
#[command(name = "engram")]
#[command(about = "Generate dendrite-like art from neural network topology")]
#[command(version)]
struct Cli {
    /// Config file path
    #[arg(long, default_value = "engram.toml")]
    config: PathBuf,

    #[command(subcommand)]
    command: Commands,
}

#[derive(clap::Subcommand)]
enum Commands {
    /// Generate a single piece from sample data
    Generate {
        /// Generator style
        #[arg(short, long, value_enum, default_value = "organic")]
        style: GraphStyleArg,

        /// Output file path
        #[arg(short, long)]
        output: Option<PathBuf>,

        /// Seed for generation
        #[arg(short = 'S', long)]
        seed: Option<u64>,

        /// Width of the output
        #[arg(long)]
        width: Option<u32>,

        /// Height of the output
        #[arg(long)]
        height: Option<u32>,

        /// Background color (hex, e.g. "#0a0a0a")
        #[arg(long)]
        background: Option<String>,

        /// Node colors (comma-separated hex values, e.g. "#3988A4,#67C2D4,#D0944D")
        #[arg(long)]
        node_colors: Option<String>,

        /// Edge color (hex, e.g. "#ffffff")
        #[arg(long)]
        edge_color: Option<String>,
    },

    /// Capture live metrics from neuronic and generate art
    Capture {
        /// RabbitMQ URL
        #[arg(long)]
        rabbitmq_url: Option<String>,

        /// RabbitMQ exchange
        #[arg(long)]
        exchange: Option<String>,

        /// Topic pattern to subscribe to
        #[arg(short, long)]
        topic: Option<String>,

        /// Number of snapshots to capture
        #[arg(short, long)]
        count: Option<usize>,

        /// Interval between snapshots in seconds
        #[arg(short, long)]
        interval: Option<u64>,

        /// Generator style
        #[arg(short, long, value_enum, default_value = "organic")]
        style: GraphStyleArg,

        /// Output file path
        #[arg(short, long)]
        output: Option<PathBuf>,

        /// Width of the output
        #[arg(long)]
        width: Option<u32>,

        /// Height of the output
        #[arg(long)]
        height: Option<u32>,

        /// Also save metrics as JSON
        #[arg(long)]
        save_metrics: bool,
    },

    /// Generate all styles for comparison
    Showcase {
        /// Output directory
        #[arg(short, long)]
        output_dir: Option<PathBuf>,

        /// Seed for consistent results
        #[arg(short = 'S', long, default_value = "42")]
        seed: u64,

        /// Width of the output
        #[arg(long)]
        width: Option<u32>,

        /// Height of the output
        #[arg(long)]
        height: Option<u32>,
    },
}

#[derive(Clone, ValueEnum, Debug)]
enum GraphStyleArg {
    /// Force-directed organic layout
    Organic,
    /// Modules arranged in a circle
    Circular,
    /// Top-to-bottom flow layout
    Hierarchical,
}

impl GraphStyleArg {
    fn to_style(&self) -> GraphStyle {
        match self {
            GraphStyleArg::Organic => GraphStyle::Organic,
            GraphStyleArg::Circular => GraphStyle::Circular,
            GraphStyleArg::Hierarchical => GraphStyle::Hierarchical,
        }
    }

    fn name(&self) -> &'static str {
        match self {
            GraphStyleArg::Organic => "organic",
            GraphStyleArg::Circular => "circular",
            GraphStyleArg::Hierarchical => "hierarchical",
        }
    }

    fn all() -> Vec<GraphStyleArg> {
        vec![
            GraphStyleArg::Organic,
            GraphStyleArg::Circular,
            GraphStyleArg::Hierarchical,
        ]
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive("engram=info".parse().unwrap()),
        )
        .init();

    let cli = Cli::parse();
    let config = EngramConfig::load(Path::new(&cli.config))?;

    match cli.command {
        Commands::Generate {
            style,
            output,
            seed,
            width,
            height,
            background,
            node_colors,
            edge_color,
        } => {
            let width = width.unwrap_or(config.output.width);
            let height = height.unwrap_or(config.output.height);
            let seed = seed.unwrap_or_else(rand::random);

            // Build palette - random by default, or custom if specified
            let palette = {
                let mut p = if background.is_some() || node_colors.is_some() || edge_color.is_some()
                {
                    ColorPalette::default()
                } else {
                    // Use randomized palette based on seed
                    ColorPalette::random(seed)
                };
                if let Some(bg) = background {
                    p.background = bg;
                }
                if let Some(colors) = node_colors {
                    p.node_colors = colors.split(',').map(|s| s.trim().to_string()).collect();
                }
                if let Some(edge) = edge_color {
                    p.edge_color = edge;
                }
                p
            };

            let gen = GraphGenerator::new(width, height, style.to_style()).with_palette(palette);
            let metrics = NeuralMetrics::sample(seed);

            println!("Generating {} with seed {}...", style.name(), seed);
            println!(
                "  {} nodes, {} edges",
                metrics.graph.nodes.len(),
                metrics.graph.edges.len()
            );

            let result = gen.generate(&metrics);

            let output_dir = PathBuf::from(&config.output.directory);
            fs::create_dir_all(&output_dir)?;

            let output_path =
                output.unwrap_or_else(|| output_dir.join(format!("engram_{}.svg", seed)));

            fs::write(&output_path, &result)?;
            println!("Saved to {}", output_path.display());
        }

        Commands::Capture {
            rabbitmq_url,
            exchange,
            topic,
            count,
            interval,
            style,
            output,
            width,
            height,
            save_metrics,
        } => {
            let rabbitmq_url = rabbitmq_url.unwrap_or_else(|| config.rabbitmq.url.clone());
            let exchange = exchange.unwrap_or_else(|| config.rabbitmq.exchange.clone());
            let topic = topic.unwrap_or_else(|| config.capture.topic.clone());
            let count = count.unwrap_or(config.capture.count);
            let interval = interval.unwrap_or(config.capture.interval);
            let width = width.unwrap_or(config.output.width);
            let height = height.unwrap_or(config.output.height);
            let save_metrics = save_metrics || config.output.save_metrics;

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
                timeout: Duration::from_secs(config.capture.timeout),
                ..Default::default()
            };

            let snapshots = capture_snapshots(&subscriber_config, &topic, &capture_config).await?;

            println!("Captured {} snapshots", snapshots.len());

            let aggregated = AggregatedMetrics::from_snapshots(&snapshots);
            let metrics = NeuralMetrics::from_aggregated(&aggregated, &snapshots);

            println!("\nTopology:");
            println!("  Nodes: {}", metrics.graph.nodes.len());
            println!("  Edges: {}", metrics.graph.edges.len());
            println!("  Topics: {}", metrics.graph.topics().len());
            println!("\nActivity:");
            println!("  Rate: {:.1} msg/s", metrics.synapse_rate);
            println!("  Throughput: {} messages", metrics.total_throughput);

            let gen = GraphGenerator::new(width, height, style.to_style());
            println!("\nGenerating {} visualization...", style.name());
            let result = gen.generate(&metrics);

            let output_dir = PathBuf::from(&config.output.directory);
            fs::create_dir_all(&output_dir)?;

            let timestamp = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs();

            let output_path =
                output.unwrap_or_else(|| output_dir.join(format!("engram_live_{}.svg", timestamp)));

            fs::write(&output_path, &result)?;
            println!("Saved to {}", output_path.display());

            if save_metrics {
                let metrics_path = output_path.with_extension("json");
                let metrics_json = serde_json::to_string_pretty(&metrics)?;
                fs::write(&metrics_path, metrics_json)?;
                println!("Saved metrics to {}", metrics_path.display());
            }
        }

        Commands::Showcase {
            output_dir, seed, ..
        } => {
            let output_dir = output_dir
                .unwrap_or_else(|| PathBuf::from(&config.output.directory).join("showcase"));

            fs::create_dir_all(&output_dir)?;

            let metrics = NeuralMetrics::sample(seed);
            println!("Generating showcase with seed {}...", seed);
            println!(
                "  {} nodes, {} edges",
                metrics.graph.nodes.len(),
                metrics.graph.edges.len()
            );

            // Generate multiple sizes
            let sizes: [(u32, &str); 4] = [(774, "774"), (2048, "2k"), (4096, "4k"), (8192, "8k")];

            for style in GraphStyleArg::all() {
                for (size, label) in &sizes {
                    let gen = GraphGenerator::new(*size, *size, style.to_style());
                    let result = gen.generate(&metrics);

                    let filename = format!("{}_{}.svg", style.name(), label);
                    let path = output_dir.join(&filename);
                    fs::write(&path, &result)?;
                    println!("  Created {}", filename);
                }
            }

            println!("Done! Showcase saved to {}", output_dir.display());
        }
    }

    Ok(())
}
