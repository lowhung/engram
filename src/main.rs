//! Engram CLI - Generate art from neural metrics.

use clap::{Parser, ValueEnum};
use engram::generators::glyph::{GlyphGenerator, GlyphStyle};
use engram::generators::stipple::{StippleGenerator, StippleStyle};
use engram::generators::svg_lines::{LineStyle, SvgLineGenerator};
use engram::generators::Generator;
use engram::metrics::NeuralMetrics;
use std::fs;
use std::path::PathBuf;

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
    /// Generate a single piece
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

    /// Generate a batch of pieces
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

fn main() {
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

            fs::write(&output_path, &result).expect("Failed to write output");
            println!("Saved to {}", output_path.display());
        }

        Commands::Batch {
            count,
            generator,
            output_dir,
            start_seed,
            width,
            height,
        } => {
            fs::create_dir_all(&output_dir).expect("Failed to create output directory");

            let gen = generator.create(width, height);
            println!("Generating {} {} pieces...", count, gen.name());

            for i in 0..count {
                let seed = start_seed + i as u64;
                let metrics = NeuralMetrics::sample(seed);
                let result = gen.generate(&metrics);

                let filename = format!("{}_{:04}.{}", gen.name(), i, gen.extension());
                let path = output_dir.join(&filename);
                fs::write(&path, &result).expect("Failed to write output");

                if (i + 1) % 10 == 0 {
                    println!("  Generated {}/{}", i + 1, count);
                }
            }

            println!("Done! {} pieces saved to {}", count, output_dir.display());
        }

        Commands::Showcase { output_dir, seed } => {
            fs::create_dir_all(&output_dir).expect("Failed to create output directory");

            let metrics = NeuralMetrics::sample(seed);
            println!("Generating showcase with seed {}...", seed);
            println!("Metrics: {:?}", metrics);

            for gen_type in GeneratorType::all() {
                let gen = gen_type.create(512, 512);
                let result = gen.generate(&metrics);

                let filename = format!("{}.{}", gen_type.name(), gen.extension());
                let path = output_dir.join(&filename);
                fs::write(&path, &result).expect("Failed to write output");
                println!("  Created {}", filename);
            }

            println!("Done! Showcase saved to {}", output_dir.display());
        }
    }
}
