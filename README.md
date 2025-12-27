# Engram

Generative art from neural network topology. Visualizes module connections from [neuronic](https://github.com/lowhung/neuronic) as dendrite-like graphs.

## Usage

```bash
# Generate from sample data
engram generate --style organic --seed 42

# Capture live topology from neuronic
engram capture --style circular

# Generate all styles
engram showcase
```

## Styles

- **organic** - Force-directed layout with curved edges
- **circular** - Modules arranged on a ring, ordered by depth
- **hierarchical** - Top-to-bottom flow (sources → sinks)

## Visual Features

The generated art reflects actual graph topology:

- **Node roles** - Sources (data producers) have cyan-shifted hues and outer rings; sinks (consumers) have purple-shifted hues; processors show balanced colors
- **Topology-aware positioning** - Nodes positioned by graph depth (sources at top, sinks at bottom) with connectivity influencing centrality
- **Dynamic coloring** - HSL-based blue palette (190-230 hue range) with per-node variation
- **Edge curves** - Curvature based on topic identity and depth difference; stroke width reflects message rate

## Configuration

Copy `config.default.toml` to `engram.toml` and adjust settings.

## License

MIT
