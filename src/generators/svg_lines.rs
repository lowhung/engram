//! SVG line and curve generator.
//!
//! Creates vector graphics with flowing lines that represent
//! neural pathways and activity patterns.

use crate::generators::Generator;
use crate::metrics::NeuralMetrics;
use rand::{Rng, SeedableRng};
use std::f64::consts::PI;

pub struct SvgLineGenerator {
    pub width: u32,
    pub height: u32,
    pub style: LineStyle,
}

#[derive(Debug, Clone, Copy)]
pub enum LineStyle {
    /// Flowing curves like neural pathways
    Flow,
    /// Sharp angular connections
    Angular,
    /// Concentric circles/rings
    Radial,
    /// Grid-based with intersections
    Grid,
    /// Organic branching structures
    Dendrite,
}

impl Default for SvgLineGenerator {
    fn default() -> Self {
        Self {
            width: 512,
            height: 512,
            style: LineStyle::Flow,
        }
    }
}

impl SvgLineGenerator {
    pub fn new(width: u32, height: u32, style: LineStyle) -> Self {
        Self {
            width,
            height,
            style,
        }
    }

    fn generate_flow(&self, metrics: &NeuralMetrics, rng: &mut impl Rng) -> String {
        let norm = metrics.normalized();
        let mut paths = Vec::new();

        let line_count = (norm.density * 30.0) as usize + 5;

        for _ in 0..line_count {
            let start_x = rng.gen_range(0.0..self.width as f64);
            let start_y = rng.gen_range(0.0..self.height as f64);

            let mut points = vec![(start_x, start_y)];
            let segments = (norm.depth * 8.0) as usize + 3;

            let mut angle = rng.gen_range(0.0..PI * 2.0);
            let step = 20.0 + norm.activity * 30.0;

            for _ in 0..segments {
                let (last_x, last_y) = points.last().unwrap();

                // Angle influenced by activity and some randomness
                angle += rng.gen_range(-0.5..0.5) * (1.0 - norm.connectivity);

                let new_x = last_x + angle.cos() * step;
                let new_y = last_y + angle.sin() * step;

                points.push((new_x, new_y));
            }

            // Build bezier path
            let stroke_width = 0.5 + norm.intensity * 2.0;
            let path = self.points_to_bezier(&points, stroke_width);
            paths.push(path);
        }

        self.wrap_svg(&paths.join("\n"))
    }

    fn generate_dendrite(&self, metrics: &NeuralMetrics, rng: &mut impl Rng) -> String {
        let norm = metrics.normalized();
        let mut paths = Vec::new();

        // Start from center or random points
        let root_count = (norm.density * 3.0) as usize + 1;

        for _ in 0..root_count {
            let start_x = self.width as f64 / 2.0 + rng.gen_range(-100.0..100.0);
            let start_y = self.height as f64 / 2.0 + rng.gen_range(-100.0..100.0);

            self.branch(
                start_x,
                start_y,
                rng.gen_range(0.0..PI * 2.0),
                50.0,
                (norm.depth * 5.0) as usize + 2,
                norm.branching,
                2.0,
                &mut paths,
                rng,
            );
        }

        self.wrap_svg(&paths.join("\n"))
    }

    fn branch(
        &self,
        x: f64,
        y: f64,
        angle: f64,
        length: f64,
        depth: usize,
        branch_prob: f64,
        stroke_width: f64,
        paths: &mut Vec<String>,
        rng: &mut impl Rng,
    ) {
        if depth == 0 || length < 5.0 {
            return;
        }

        let end_x = x + angle.cos() * length;
        let end_y = y + angle.sin() * length;

        paths.push(format!(
            r#"<line x1="{:.1}" y1="{:.1}" x2="{:.1}" y2="{:.1}" stroke="black" stroke-width="{:.1}" stroke-linecap="round"/>"#,
            x, y, end_x, end_y, stroke_width
        ));

        let new_length = length * 0.75;
        let new_width = (stroke_width * 0.8).max(0.5);

        // Continue main branch
        self.branch(
            end_x,
            end_y,
            angle + rng.gen_range(-0.3..0.3),
            new_length,
            depth - 1,
            branch_prob,
            new_width,
            paths,
            rng,
        );

        // Maybe spawn side branches
        if rng.gen::<f64>() < branch_prob {
            let branch_angle = if rng.gen() { PI / 4.0 } else { -PI / 4.0 };
            self.branch(
                end_x,
                end_y,
                angle + branch_angle + rng.gen_range(-0.2..0.2),
                new_length * 0.8,
                depth - 1,
                branch_prob * 0.7,
                new_width,
                paths,
                rng,
            );
        }
    }

    fn generate_radial(&self, metrics: &NeuralMetrics, rng: &mut impl Rng) -> String {
        let norm = metrics.normalized();
        let mut elements = Vec::new();

        let cx = self.width as f64 / 2.0;
        let cy = self.height as f64 / 2.0;

        let ring_count = (norm.depth * 10.0) as usize + 3;
        let max_radius = self.width.min(self.height) as f64 / 2.0 - 10.0;

        for i in 0..ring_count {
            let radius = (i as f64 / ring_count as f64) * max_radius;
            let stroke_width = 0.5 + (1.0 - i as f64 / ring_count as f64) * norm.intensity * 2.0;

            // Full circle or arc based on connectivity
            if norm.connectivity > 0.5 || rng.gen::<f64>() < 0.3 {
                elements.push(format!(
                    r#"<circle cx="{:.1}" cy="{:.1}" r="{:.1}" fill="none" stroke="black" stroke-width="{:.1}"/>"#,
                    cx, cy, radius, stroke_width
                ));
            } else {
                // Partial arc
                let start_angle = rng.gen_range(0.0..PI * 2.0);
                let arc_length = rng.gen_range(PI / 2.0..PI * 1.5);
                let end_angle = start_angle + arc_length;

                let x1 = cx + radius * start_angle.cos();
                let y1 = cy + radius * start_angle.sin();
                let x2 = cx + radius * end_angle.cos();
                let y2 = cy + radius * end_angle.sin();

                let large_arc = if arc_length > PI { 1 } else { 0 };

                elements.push(format!(
                    r#"<path d="M {:.1} {:.1} A {:.1} {:.1} 0 {} 1 {:.1} {:.1}" fill="none" stroke="black" stroke-width="{:.1}"/>"#,
                    x1, y1, radius, radius, large_arc, x2, y2, stroke_width
                ));
            }
        }

        // Add radial lines based on branching
        let ray_count = (norm.branching * 20.0) as usize;
        for i in 0..ray_count {
            let angle = (i as f64 / ray_count as f64) * PI * 2.0;
            let inner_r = max_radius * norm.skew * 0.3;
            let outer_r = max_radius * (0.7 + rng.gen::<f64>() * 0.3);

            let x1 = cx + inner_r * angle.cos();
            let y1 = cy + inner_r * angle.sin();
            let x2 = cx + outer_r * angle.cos();
            let y2 = cy + outer_r * angle.sin();

            elements.push(format!(
                r#"<line x1="{:.1}" y1="{:.1}" x2="{:.1}" y2="{:.1}" stroke="black" stroke-width="0.5"/>"#,
                x1, y1, x2, y2
            ));
        }

        self.wrap_svg(&elements.join("\n"))
    }

    fn generate_grid(&self, metrics: &NeuralMetrics, rng: &mut impl Rng) -> String {
        let norm = metrics.normalized();
        let mut elements = Vec::new();

        let cell_size = 20.0 + (1.0 - norm.density) * 30.0;
        let cols = (self.width as f64 / cell_size) as usize;
        let rows = (self.height as f64 / cell_size) as usize;

        // Create connection matrix based on metrics
        for row in 0..rows {
            for col in 0..cols {
                let x = col as f64 * cell_size + cell_size / 2.0;
                let y = row as f64 * cell_size + cell_size / 2.0;

                // Node at intersection
                if rng.gen::<f64>() < norm.density * 0.5 {
                    let r = 1.0 + norm.intensity * 3.0;
                    elements.push(format!(
                        r#"<circle cx="{:.1}" cy="{:.1}" r="{:.1}" fill="black"/>"#,
                        x, y, r
                    ));
                }

                // Horizontal connection
                if col < cols - 1 && rng.gen::<f64>() < norm.connectivity {
                    let next_x = (col + 1) as f64 * cell_size + cell_size / 2.0;
                    elements.push(format!(
                        r#"<line x1="{:.1}" y1="{:.1}" x2="{:.1}" y2="{:.1}" stroke="black" stroke-width="0.5"/>"#,
                        x, y, next_x, y
                    ));
                }

                // Vertical connection
                if row < rows - 1 && rng.gen::<f64>() < norm.connectivity {
                    let next_y = (row + 1) as f64 * cell_size + cell_size / 2.0;
                    elements.push(format!(
                        r#"<line x1="{:.1}" y1="{:.1}" x2="{:.1}" y2="{:.1}" stroke="black" stroke-width="0.5"/>"#,
                        x, y, x, next_y
                    ));
                }
            }
        }

        self.wrap_svg(&elements.join("\n"))
    }

    fn points_to_bezier(&self, points: &[(f64, f64)], stroke_width: f64) -> String {
        if points.len() < 2 {
            return String::new();
        }

        let mut d = format!("M {:.1} {:.1}", points[0].0, points[0].1);

        for i in 1..points.len() {
            let (x, y) = points[i];
            let (px, py) = points[i - 1];

            // Simple quadratic bezier with midpoint control
            let cx = (px + x) / 2.0;
            let cy = (py + y) / 2.0;

            d.push_str(&format!(" Q {:.1} {:.1} {:.1} {:.1}", px, py, cx, cy));
        }

        format!(
            r#"<path d="{}" fill="none" stroke="black" stroke-width="{:.1}" stroke-linecap="round"/>"#,
            d, stroke_width
        )
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

impl Generator for SvgLineGenerator {
    fn name(&self) -> &'static str {
        "svg_lines"
    }

    fn generate(&self, metrics: &NeuralMetrics) -> String {
        let seed = metrics.to_seed();
        let seed_u64 = u64::from_le_bytes(seed[0..8].try_into().unwrap());
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed_u64);

        match self.style {
            LineStyle::Flow => self.generate_flow(metrics, &mut rng),
            LineStyle::Dendrite => self.generate_dendrite(metrics, &mut rng),
            LineStyle::Radial => self.generate_radial(metrics, &mut rng),
            LineStyle::Grid => self.generate_grid(metrics, &mut rng),
            LineStyle::Angular => self.generate_flow(metrics, &mut rng), // TODO: implement
        }
    }

    fn extension(&self) -> &'static str {
        "svg"
    }
}
