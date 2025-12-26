//! Interference pattern generator.
//!
//! Creates moiré-like patterns where each node in the graph becomes
//! a wave source. The interference between waves creates emergent
//! visual complexity from simple rules.

use crate::generators::Generator;
use crate::metrics::NeuralMetrics;
use sha2::{Digest, Sha256};
use std::f64::consts::PI;

pub struct InterferenceGenerator {
    pub width: u32,
    pub height: u32,
    pub style: InterferenceStyle,
}

#[derive(Debug, Clone, Copy)]
pub enum InterferenceStyle {
    /// Concentric circles from each node, interference where they overlap
    Ripples,
    /// Radial lines from each node
    Rays,
    /// Combination of ripples and rays
    Combined,
    /// Grid-based wave interference
    WaveGrid,
}

impl Default for InterferenceGenerator {
    fn default() -> Self {
        Self {
            width: 512,
            height: 512,
            style: InterferenceStyle::Ripples,
        }
    }
}

impl InterferenceGenerator {
    pub fn new(width: u32, height: u32, style: InterferenceStyle) -> Self {
        Self {
            width,
            height,
            style,
        }
    }

    /// Hash a string to get a deterministic position in the canvas
    fn hash_to_position(&self, name: &str, salt: u8) -> (f64, f64) {
        let mut hasher = Sha256::new();
        hasher.update(name.as_bytes());
        hasher.update([salt]);
        let hash = hasher.finalize();

        // Use first 8 bytes for x, next 8 for y
        let x_bytes: [u8; 8] = hash[0..8].try_into().unwrap();
        let y_bytes: [u8; 8] = hash[8..16].try_into().unwrap();

        let x_raw = u64::from_le_bytes(x_bytes) as f64 / u64::MAX as f64;
        let y_raw = u64::from_le_bytes(y_bytes) as f64 / u64::MAX as f64;

        // Add margin to keep nodes away from edges
        let margin = 0.1;
        let x = margin + x_raw * (1.0 - 2.0 * margin);
        let y = margin + y_raw * (1.0 - 2.0 * margin);

        (x * self.width as f64, y * self.height as f64)
    }

    /// Hash a string to get a deterministic value 0-1
    fn hash_to_value(&self, name: &str, salt: u8) -> f64 {
        let mut hasher = Sha256::new();
        hasher.update(name.as_bytes());
        hasher.update([salt]);
        let hash = hasher.finalize();
        let bytes: [u8; 8] = hash[0..8].try_into().unwrap();
        u64::from_le_bytes(bytes) as f64 / u64::MAX as f64
    }

    /// Generate ripple interference pattern
    fn generate_ripples(&self, metrics: &NeuralMetrics) -> String {
        let mut elements = Vec::new();

        // Each node becomes a wave source
        let sources: Vec<(f64, f64, f64, f64)> = metrics
            .node_names
            .iter()
            .map(|name| {
                let (x, y) = self.hash_to_position(name, 0);
                let frequency = 0.02 + self.hash_to_value(name, 1) * 0.03;
                let phase = self.hash_to_value(name, 2) * PI * 2.0;
                (x, y, frequency, phase)
            })
            .collect();

        if sources.is_empty() {
            return self.wrap_svg("");
        }

        // Calculate max rings based on canvas size
        let max_radius = ((self.width.pow(2) + self.height.pow(2)) as f64).sqrt();

        // Base wavelength on activity
        let base_wavelength = 15.0 + (1.0 - metrics.normalized().activity) * 20.0;

        // Draw concentric circles from each source
        for (cx, cy, freq_mod, _phase) in &sources {
            let wavelength = base_wavelength * (0.8 + freq_mod * 0.4);
            let num_rings = (max_radius / wavelength) as usize;

            for i in 0..num_rings {
                let radius = (i as f64 + 0.5) * wavelength;

                // Fade out with distance
                let opacity = (1.0 - radius / max_radius).max(0.0).powf(0.5);
                if opacity < 0.05 {
                    continue;
                }

                // Stroke width decreases with radius
                let stroke_width = (0.3 + 0.7 * (1.0 - radius / max_radius)).max(0.2);

                elements.push(format!(
                    r#"<circle cx="{:.1}" cy="{:.1}" r="{:.1}" fill="none" stroke="black" stroke-width="{:.2}" stroke-opacity="{:.2}"/>"#,
                    cx, cy, radius, stroke_width, opacity
                ));
            }
        }

        self.wrap_svg(&elements.join("\n"))
    }

    /// Generate ray interference pattern
    fn generate_rays(&self, metrics: &NeuralMetrics) -> String {
        let mut elements = Vec::new();

        let sources: Vec<(f64, f64, usize)> = metrics
            .node_names
            .iter()
            .map(|name| {
                let (x, y) = self.hash_to_position(name, 0);
                // Number of rays based on hash
                let num_rays = 12 + (self.hash_to_value(name, 3) * 24.0) as usize;
                (x, y, num_rays)
            })
            .collect();

        if sources.is_empty() {
            return self.wrap_svg("");
        }

        let max_radius = ((self.width.pow(2) + self.height.pow(2)) as f64).sqrt();

        for (cx, cy, num_rays) in &sources {
            let angle_step = PI * 2.0 / *num_rays as f64;

            for i in 0..*num_rays {
                let angle = i as f64 * angle_step;
                let x2 = cx + angle.cos() * max_radius;
                let y2 = cy + angle.sin() * max_radius;

                elements.push(format!(
                    r#"<line x1="{:.1}" y1="{:.1}" x2="{:.1}" y2="{:.1}" stroke="black" stroke-width="0.3" stroke-opacity="0.4"/>"#,
                    cx, cy, x2, y2
                ));
            }
        }

        self.wrap_svg(&elements.join("\n"))
    }

    /// Generate combined ripples and rays
    fn generate_combined(&self, metrics: &NeuralMetrics) -> String {
        let mut elements = Vec::new();

        let sources: Vec<(f64, f64, f64, usize)> = metrics
            .node_names
            .iter()
            .map(|name| {
                let (x, y) = self.hash_to_position(name, 0);
                let freq_mod = self.hash_to_value(name, 1);
                let num_rays = 8 + (self.hash_to_value(name, 3) * 16.0) as usize;
                (x, y, freq_mod, num_rays)
            })
            .collect();

        if sources.is_empty() {
            return self.wrap_svg("");
        }

        let max_radius = ((self.width.pow(2) + self.height.pow(2)) as f64).sqrt();
        let base_wavelength = 20.0 + (1.0 - metrics.normalized().activity) * 15.0;

        // Draw rays first (behind)
        for (cx, cy, _, num_rays) in &sources {
            let angle_step = PI * 2.0 / *num_rays as f64;

            for i in 0..*num_rays {
                let angle = i as f64 * angle_step;
                let x2 = cx + angle.cos() * max_radius;
                let y2 = cy + angle.sin() * max_radius;

                elements.push(format!(
                    r#"<line x1="{:.1}" y1="{:.1}" x2="{:.1}" y2="{:.1}" stroke="black" stroke-width="0.2" stroke-opacity="0.2"/>"#,
                    cx, cy, x2, y2
                ));
            }
        }

        // Draw ripples on top
        for (cx, cy, freq_mod, _) in &sources {
            let wavelength = base_wavelength * (0.8 + freq_mod * 0.4);
            let num_rings = (max_radius / wavelength) as usize;

            for i in 0..num_rings {
                let radius = (i as f64 + 0.5) * wavelength;
                let opacity = (1.0 - radius / max_radius).max(0.0).powf(0.6) * 0.7;
                if opacity < 0.03 {
                    continue;
                }

                let stroke_width = (0.4 + 0.6 * (1.0 - radius / max_radius)).max(0.15);

                elements.push(format!(
                    r#"<circle cx="{:.1}" cy="{:.1}" r="{:.1}" fill="none" stroke="black" stroke-width="{:.2}" stroke-opacity="{:.2}"/>"#,
                    cx, cy, radius, stroke_width, opacity
                ));
            }
        }

        self.wrap_svg(&elements.join("\n"))
    }

    /// Generate wave grid - samples interference at grid points
    fn generate_wave_grid(&self, metrics: &NeuralMetrics) -> String {
        let mut elements = Vec::new();

        let sources: Vec<(f64, f64, f64, f64)> = metrics
            .node_names
            .iter()
            .map(|name| {
                let (x, y) = self.hash_to_position(name, 0);
                let frequency = 0.03 + self.hash_to_value(name, 1) * 0.04;
                let phase = self.hash_to_value(name, 2) * PI * 2.0;
                (x, y, frequency, phase)
            })
            .collect();

        if sources.is_empty() {
            return self.wrap_svg("");
        }

        // Grid resolution
        let cell_size = 6.0;
        let cols = (self.width as f64 / cell_size) as usize;
        let rows = (self.height as f64 / cell_size) as usize;

        for row in 0..rows {
            for col in 0..cols {
                let px = (col as f64 + 0.5) * cell_size;
                let py = (row as f64 + 0.5) * cell_size;

                // Calculate interference at this point
                let mut wave_sum = 0.0;
                for (sx, sy, freq, phase) in &sources {
                    let dist = ((px - sx).powi(2) + (py - sy).powi(2)).sqrt();
                    let wave = (dist * freq + phase).sin();
                    wave_sum += wave;
                }

                // Normalize to 0-1
                let normalized = if !sources.is_empty() {
                    (wave_sum / sources.len() as f64 + 1.0) / 2.0
                } else {
                    0.5
                };

                // Only draw if above threshold (creates the pattern)
                if normalized > 0.5 {
                    let intensity = (normalized - 0.5) * 2.0;
                    let radius = cell_size * 0.4 * intensity;

                    if radius > 0.5 {
                        elements.push(format!(
                            r#"<circle cx="{:.1}" cy="{:.1}" r="{:.1}" fill="black"/>"#,
                            px, py, radius
                        ));
                    }
                }
            }
        }

        self.wrap_svg(&elements.join("\n"))
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

impl Generator for InterferenceGenerator {
    fn name(&self) -> &'static str {
        "interference"
    }

    fn generate(&self, metrics: &NeuralMetrics) -> String {
        match self.style {
            InterferenceStyle::Ripples => self.generate_ripples(metrics),
            InterferenceStyle::Rays => self.generate_rays(metrics),
            InterferenceStyle::Combined => self.generate_combined(metrics),
            InterferenceStyle::WaveGrid => self.generate_wave_grid(metrics),
        }
    }

    fn extension(&self) -> &'static str {
        "svg"
    }
}
