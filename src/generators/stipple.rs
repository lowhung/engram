//! Stipple/dot pattern generator.
//!
//! Creates pointillist-style art where dot density and placement
//! encode neural activity patterns.

use crate::generators::Generator;
use crate::metrics::NeuralMetrics;
use rand::{Rng, SeedableRng};
use std::f64::consts::PI;

pub struct StippleGenerator {
    pub width: u32,
    pub height: u32,
    pub style: StippleStyle,
}

#[derive(Debug, Clone, Copy)]
pub enum StippleStyle {
    /// Uniform random distribution with density gradients
    Gradient,
    /// Clustered around activity centers
    Clustered,
    /// Following flow lines
    Flow,
    /// Halftone-inspired concentric patterns
    Halftone,
}

impl Default for StippleGenerator {
    fn default() -> Self {
        Self {
            width: 512,
            height: 512,
            style: StippleStyle::Gradient,
        }
    }
}

impl StippleGenerator {
    pub fn new(width: u32, height: u32, style: StippleStyle) -> Self {
        Self {
            width,
            height,
            style,
        }
    }

    fn generate_gradient(&self, metrics: &NeuralMetrics, rng: &mut impl Rng) -> String {
        let norm = metrics.normalized();
        let mut dots = Vec::new();

        let base_count = (norm.density * 2000.0) as usize + 200;

        for _ in 0..base_count {
            let x = rng.gen_range(0.0..self.width as f64);
            let y = rng.gen_range(0.0..self.height as f64);

            // Density varies across the canvas based on activity pattern
            let density_factor = match (norm.skew * 4.0) as usize % 4 {
                0 => x / self.width as f64,  // left-to-right gradient
                1 => y / self.height as f64, // top-to-bottom
                2 => {
                    let cx = self.width as f64 / 2.0;
                    let cy = self.height as f64 / 2.0;
                    1.0 - ((x - cx).powi(2) + (y - cy).powi(2)).sqrt() / (cx.max(cy))
                } // radial from center
                _ => ((x / self.width as f64 * PI * 2.0).sin() + 1.0) / 2.0, // wave pattern
            };

            if rng.gen::<f64>() < density_factor * norm.activity + 0.1 {
                let r = 0.5 + rng.gen::<f64>() * norm.intensity * 3.0;
                dots.push(format!(
                    r#"<circle cx="{:.1}" cy="{:.1}" r="{:.1}" fill="black"/>"#,
                    x, y, r
                ));
            }
        }

        self.wrap_svg(&dots.join("\n"))
    }

    fn generate_clustered(&self, metrics: &NeuralMetrics, rng: &mut impl Rng) -> String {
        let norm = metrics.normalized();
        let mut dots = Vec::new();

        // Create cluster centers based on node count
        let cluster_count = (norm.density * 8.0) as usize + 2;
        let clusters: Vec<(f64, f64, f64)> = (0..cluster_count)
            .map(|_| {
                (
                    rng.gen_range(50.0..self.width as f64 - 50.0),
                    rng.gen_range(50.0..self.height as f64 - 50.0),
                    rng.gen_range(30.0..100.0) * norm.branching, // radius
                )
            })
            .collect();

        let dots_per_cluster = (norm.activity * 300.0) as usize + 50;

        for (cx, cy, radius) in &clusters {
            for _ in 0..dots_per_cluster {
                // Gaussian-ish distribution around center
                let angle = rng.gen_range(0.0..PI * 2.0);
                let dist = rng.gen::<f64>().sqrt() * radius; // sqrt for uniform disk

                let x = cx + angle.cos() * dist;
                let y = cy + angle.sin() * dist;

                if x >= 0.0 && x < self.width as f64 && y >= 0.0 && y < self.height as f64 {
                    // Dot size decreases toward edge
                    let edge_factor = 1.0 - (dist / radius).min(1.0);
                    let r = 0.3 + edge_factor * norm.intensity * 2.0;

                    dots.push(format!(
                        r#"<circle cx="{:.1}" cy="{:.1}" r="{:.1}" fill="black"/>"#,
                        x, y, r
                    ));
                }
            }
        }

        // Add some noise dots outside clusters
        let noise_count = (norm.connectivity * 200.0) as usize;
        for _ in 0..noise_count {
            let x = rng.gen_range(0.0..self.width as f64);
            let y = rng.gen_range(0.0..self.height as f64);
            dots.push(format!(
                r#"<circle cx="{:.1}" cy="{:.1}" r="0.5" fill="black"/>"#,
                x, y
            ));
        }

        self.wrap_svg(&dots.join("\n"))
    }

    fn generate_flow(&self, metrics: &NeuralMetrics, rng: &mut impl Rng) -> String {
        let norm = metrics.normalized();
        let mut dots = Vec::new();

        let line_count = (norm.depth * 20.0) as usize + 5;
        let dots_per_line = (norm.activity * 50.0) as usize + 10;

        for _ in 0..line_count {
            let mut x = rng.gen_range(0.0..self.width as f64);
            let mut y = rng.gen_range(0.0..self.height as f64);
            let mut angle = rng.gen_range(0.0..PI * 2.0);

            for i in 0..dots_per_line {
                // Perlin-like flow field simulation
                let noise = ((x * 0.01).sin() + (y * 0.01).cos()) * PI;
                angle += noise * 0.1 + rng.gen_range(-0.1..0.1);

                let step = 8.0 + norm.branching * 10.0;
                x += angle.cos() * step;
                y += angle.sin() * step;

                if x < 0.0 || x >= self.width as f64 || y < 0.0 || y >= self.height as f64 {
                    break;
                }

                // Dot size varies along the path
                let t = i as f64 / dots_per_line as f64;
                let r = 0.5 + (1.0 - (t - 0.5).abs() * 2.0) * norm.intensity * 2.5;

                dots.push(format!(
                    r#"<circle cx="{:.1}" cy="{:.1}" r="{:.1}" fill="black"/>"#,
                    x, y, r
                ));
            }
        }

        self.wrap_svg(&dots.join("\n"))
    }

    fn generate_halftone(&self, metrics: &NeuralMetrics, rng: &mut impl Rng) -> String {
        let norm = metrics.normalized();
        let mut dots = Vec::new();

        let cell_size = 8.0 + (1.0 - norm.density) * 12.0;
        let cols = (self.width as f64 / cell_size) as usize;
        let rows = (self.height as f64 / cell_size) as usize;

        let cx = self.width as f64 / 2.0;
        let cy = self.height as f64 / 2.0;
        let max_dist = (cx.powi(2) + cy.powi(2)).sqrt();

        for row in 0..rows {
            for col in 0..cols {
                let x = col as f64 * cell_size + cell_size / 2.0;
                let y = row as f64 * cell_size + cell_size / 2.0;

                // Calculate "darkness" at this point
                let dist = ((x - cx).powi(2) + (y - cy).powi(2)).sqrt();
                let base_darkness = match (norm.skew * 3.0) as usize % 3 {
                    0 => 1.0 - dist / max_dist,                     // radial
                    1 => x / self.width as f64,                     // horizontal
                    _ => ((x * 0.02 + y * 0.02).sin() + 1.0) / 2.0, // moire
                };

                let darkness = base_darkness * norm.activity + rng.gen::<f64>() * 0.2;

                if darkness > 0.1 {
                    let max_r = cell_size / 2.0 * 0.9;
                    let r = darkness.min(1.0) * max_r * norm.intensity + 0.5;

                    dots.push(format!(
                        r#"<circle cx="{:.1}" cy="{:.1}" r="{:.1}" fill="black"/>"#,
                        x, y, r
                    ));
                }
            }
        }

        self.wrap_svg(&dots.join("\n"))
    }

    fn wrap_svg(&self, content: &str) -> String {
        format!(
            r#"<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {} {}" width="{}" height="{}">
  <rect width="100%" height="100%" fill="white"/>
  {}
</svg>"#,
            self.width, self.height, self.width, self.height, content
        )
    }
}

impl Generator for StippleGenerator {
    fn name(&self) -> &'static str {
        "stipple"
    }

    fn generate(&self, metrics: &NeuralMetrics) -> String {
        let seed = metrics.to_seed();
        let seed_u64 = u64::from_le_bytes(seed[0..8].try_into().unwrap());
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed_u64);

        match self.style {
            StippleStyle::Gradient => self.generate_gradient(metrics, &mut rng),
            StippleStyle::Clustered => self.generate_clustered(metrics, &mut rng),
            StippleStyle::Flow => self.generate_flow(metrics, &mut rng),
            StippleStyle::Halftone => self.generate_halftone(metrics, &mut rng),
        }
    }

    fn extension(&self) -> &'static str {
        "svg"
    }
}
