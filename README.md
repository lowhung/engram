# Engram

Generative art from neural network activity metrics.

Inspired by [Autoglyphs](https://github.com/protofire/autoglyphs) and built to work with [neuronic](https://github.com/lowhung/neuronic), Engram creates unique black & white visual pieces by encoding live message bus activity into minimalist art.

## Features

- **Live capture**: Connect to RabbitMQ and generate art from real system activity
- **Multiple generators**: ASCII glyphs, SVG lines, stipple patterns
- **Deterministic**: Same metrics always produce the same art
- **Configurable**: TOML config with environment variable overrides

## Installation

```bash
cargo install --path .
```

## Usage

### Generate from sample data

```bash
# Single piece with random seed
engram generate -g dendrite

# Batch of 100 pieces
engram batch -c 100 -g halftone -o output/

# Showcase all generator types
engram showcase
```

### Capture live metrics

```bash
# Connect to RabbitMQ and generate art from live data
engram capture \
  --rabbitmq-url amqp://guest:guest@localhost:5672 \
  --topic buswatch.snapshot \
  --count 10 \
  --generator dendrite \
  --save-metrics
```

## Generators

### Glyph (ASCII/Unicode text art)
- `glyph-minimal` - Sparse ASCII characters
- `glyph-circuit` - Box-drawing characters
- `glyph-stipple` - Dot characters
- `glyph-neural` - Neural-inspired symbols
- `glyph-adaptive` - Adapts charset to activity level

### SVG Lines
- `flow` - Flowing curves
- `dendrite` - Organic branching structures
- `radial` - Concentric rings with rays
- `grid` - Node/connection patterns

### Stipple
- `stipple-gradient` - Density gradients
- `stipple-clustered` - Clustered around activity centers
- `stipple-flow` - Dots following flow lines
- `halftone` - Variable dot size grids

## Configuration

Copy `config.default.toml` to `engram.toml` and customize:

```toml
[rabbitmq]
url = "amqp://127.0.0.1:5672/%2f"
exchange = "caryatid"

[capture]
topic = "buswatch.snapshot"
count = 10
interval = 1

[output]
width = 512
height = 512

[generator]
default = "dendrite"
```

Override with environment variables:
```bash
ENGRAM_RABBITMQ_URL=amqp://localhost engram capture
```

## Metrics

Each piece encodes:
- Node count and topology
- Message throughput and rates
- Activity distribution (skew)
- Health status ratios
- Peak burst detection

The metrics are hashed to create a deterministic seed, so the same system state always produces the same art.

## License

MIT
