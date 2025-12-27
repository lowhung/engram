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
- **circular** - Modules arranged on a ring
- **hierarchical** - Top-to-bottom flow (sources → sinks)
- **constellation** - Star-like nodes with dotted connections

## Configuration

Copy `config.default.toml` to `engram.toml` and adjust settings.

## License

MIT
