//! ASCII/Unicode glyph-based generator inspired by Autoglyphs.
//!
//! Produces text-based art using a limited character set,
//! where patterns emerge from neural metrics.

use crate::generators::Generator;
use crate::metrics::NeuralMetrics;
use rand::{Rng, SeedableRng};

/// Character sets for different visual densities.
const SPARSE: &[char] = &[' ', '.', '·'];
const LIGHT: &[char] = &[' ', '.', '·', ':', '\'', '`'];
const MEDIUM: &[char] = &['.', ':', '+', '-', '|', '/', '\\'];
const DENSE: &[char] = &['#', '@', '%', '&', '*', '+', '='];
const NEURAL: &[char] = &['○', '●', '◐', '◑', '◒', '◓', '│', '─', '┼', '╱', '╲'];
const LINES: &[char] = &['│', '─', '┌', '┐', '└', '┘', '├', '┤', '┬', '┴', '┼'];
const DOTS: &[char] = &[' ', '·', '•', '●', '○', '◦'];

pub struct GlyphGenerator {
    pub width: usize,
    pub height: usize,
    pub style: GlyphStyle,
}

#[derive(Debug, Clone, Copy)]
pub enum GlyphStyle {
    /// Classic ASCII, sparse
    Minimal,
    /// Box-drawing characters
    Circuit,
    /// Dot patterns
    Stipple,
    /// Neural-inspired symbols
    Neural,
    /// Adaptive based on metrics
    Adaptive,
}

impl Default for GlyphGenerator {
    fn default() -> Self {
        Self {
            width: 64,
            height: 64,
            style: GlyphStyle::Adaptive,
        }
    }
}

impl GlyphGenerator {
    pub fn new(width: usize, height: usize, style: GlyphStyle) -> Self {
        Self {
            width,
            height,
            style,
        }
    }

    fn select_charset(&self, metrics: &crate::metrics::NormalizedMetrics) -> &'static [char] {
        match self.style {
            GlyphStyle::Minimal => SPARSE,
            GlyphStyle::Circuit => LINES,
            GlyphStyle::Stipple => DOTS,
            GlyphStyle::Neural => NEURAL,
            GlyphStyle::Adaptive => {
                if metrics.activity < 0.3 {
                    SPARSE
                } else if metrics.activity < 0.5 {
                    LIGHT
                } else if metrics.activity < 0.7 {
                    MEDIUM
                } else {
                    DENSE
                }
            }
        }
    }

    /// Generate pattern using cellular automata influenced by metrics.
    fn generate_cellular(&self, metrics: &NeuralMetrics, rng: &mut impl Rng) -> Vec<Vec<char>> {
        let norm = metrics.normalized();
        let charset = self.select_charset(&norm);

        let mut grid: Vec<Vec<char>> = vec![vec![' '; self.width]; self.height];

        // Seed initial points based on node_count
        let seed_count = (norm.density * 20.0) as usize + 5;
        for _ in 0..seed_count {
            let x = rng.gen_range(0..self.width);
            let y = rng.gen_range(0..self.height);
            grid[y][x] = charset[rng.gen_range(0..charset.len())];
        }

        // Evolve based on branching and connectivity
        let iterations = (norm.depth * 10.0) as usize + 3;
        for _ in 0..iterations {
            let mut next = grid.clone();
            for y in 1..self.height - 1 {
                for x in 1..self.width - 1 {
                    let neighbors = self.count_neighbors(&grid, x, y);
                    let threshold = (norm.connectivity * 4.0) as usize + 1;

                    if neighbors >= threshold && grid[y][x] == ' ' {
                        if rng.gen::<f64>() < norm.branching * 0.3 {
                            next[y][x] = charset[rng.gen_range(0..charset.len())];
                        }
                    }
                }
            }
            grid = next;
        }

        grid
    }

    /// Generate wave/flow pattern based on synapse rate.
    fn generate_flow(&self, metrics: &NeuralMetrics, _rng: &mut impl Rng) -> Vec<Vec<char>> {
        let norm = metrics.normalized();
        let charset = self.select_charset(&norm);

        let mut grid: Vec<Vec<char>> = vec![vec![' '; self.width]; self.height];

        let frequency = norm.activity * 0.3 + 0.1;
        let amplitude = norm.intensity * (self.height as f64 / 4.0);
        let phase_offset = norm.skew * std::f64::consts::PI * 2.0;

        for x in 0..self.width {
            let base_y = (self.height as f64 / 2.0
                + (x as f64 * frequency + phase_offset).sin() * amplitude)
                as usize;

            let spread = (norm.branching * 3.0) as usize + 1;
            for dy in 0..spread {
                let y = base_y.saturating_add(dy).min(self.height - 1);
                let y2 = base_y.saturating_sub(dy);

                let char_idx = (dy as f64 / spread as f64 * charset.len() as f64) as usize;
                let c = charset[char_idx.min(charset.len() - 1)];

                grid[y][x] = c;
                if y2 < self.height {
                    grid[y2][x] = c;
                }
            }
        }

        grid
    }

    /// Generate concentric patterns from activity centers.
    fn generate_radial(&self, metrics: &NeuralMetrics, rng: &mut impl Rng) -> Vec<Vec<char>> {
        let norm = metrics.normalized();
        let charset = self.select_charset(&norm);

        let mut grid: Vec<Vec<char>> = vec![vec![' '; self.width]; self.height];

        // Create activity centers based on node topology
        let center_count = (norm.density * 5.0) as usize + 1;
        let centers: Vec<(f64, f64)> = (0..center_count)
            .map(|_| {
                (
                    rng.gen_range(0.0..self.width as f64),
                    rng.gen_range(0.0..self.height as f64),
                )
            })
            .collect();

        for y in 0..self.height {
            for x in 0..self.width {
                let mut min_dist = f64::MAX;
                for &(cx, cy) in &centers {
                    let dist = ((x as f64 - cx).powi(2) + (y as f64 - cy).powi(2)).sqrt();
                    min_dist = min_dist.min(dist);
                }

                let ring_width = (1.0 - norm.connectivity) * 5.0 + 2.0;
                let ring = (min_dist / ring_width) as usize;

                if ring % 2 == 0 || rng.gen::<f64>() < norm.intensity * 0.3 {
                    let idx = ring % charset.len();
                    grid[y][x] = charset[idx];
                }
            }
        }

        grid
    }

    fn count_neighbors(&self, grid: &[Vec<char>], x: usize, y: usize) -> usize {
        let mut count = 0;
        for dy in -1i32..=1 {
            for dx in -1i32..=1 {
                if dx == 0 && dy == 0 {
                    continue;
                }
                let nx = (x as i32 + dx) as usize;
                let ny = (y as i32 + dy) as usize;
                if ny < grid.len() && nx < grid[0].len() && grid[ny][nx] != ' ' {
                    count += 1;
                }
            }
        }
        count
    }
}

impl Generator for GlyphGenerator {
    fn name(&self) -> &'static str {
        "glyph"
    }

    fn generate(&self, metrics: &NeuralMetrics) -> String {
        let seed = metrics.to_seed();
        let seed_u64 = u64::from_le_bytes(seed[0..8].try_into().unwrap());
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed_u64);

        let norm = metrics.normalized();

        // Choose pattern type based on metrics
        let grid = if norm.branching > 0.6 {
            self.generate_cellular(metrics, &mut rng)
        } else if norm.activity > 0.5 {
            self.generate_flow(metrics, &mut rng)
        } else {
            self.generate_radial(metrics, &mut rng)
        };

        // Convert grid to string
        grid.iter()
            .map(|row| row.iter().collect::<String>())
            .collect::<Vec<_>>()
            .join("\n")
    }

    fn extension(&self) -> &'static str {
        "txt"
    }
}
