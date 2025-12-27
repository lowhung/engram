//! Graph topology generator.
//!
//! Visualizes the actual module adjacency graph as a dendrite-like
//! network diagram. Modules are nodes, topics are connections.

use crate::generators::Generator;
use crate::metrics::{GraphEdge, NeuralMetrics};
use rand::{Rng, SeedableRng};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::f64::consts::PI;

pub struct GraphGenerator {
    pub width: u32,
    pub height: u32,
    pub style: GraphStyle,
}

#[derive(Debug, Clone, Copy)]
pub enum GraphStyle {
    /// Organic force-directed layout with curved edges
    Organic,
    /// Circular arrangement with modules on a ring
    Circular,
    /// Hierarchical top-to-bottom flow
    Hierarchical,
    /// Constellation-like with clusters
    Constellation,
}

impl Default for GraphGenerator {
    fn default() -> Self {
        Self {
            width: 512,
            height: 512,
            style: GraphStyle::Organic,
        }
    }
}

/// A positioned node for rendering.
struct PositionedNode {
    name: String,
    x: f64,
    y: f64,
    radius: f64,
    throughput: u64,
}

impl GraphGenerator {
    pub fn new(width: u32, height: u32, style: GraphStyle) -> Self {
        Self {
            width,
            height,
            style,
        }
    }

    /// Hash a string to a deterministic float in [0, 1).
    fn hash_to_float(s: &str) -> f64 {
        let mut hasher = Sha256::new();
        hasher.update(s.as_bytes());
        let hash = hasher.finalize();
        let val = u32::from_le_bytes([hash[0], hash[1], hash[2], hash[3]]);
        val as f64 / u32::MAX as f64
    }

    /// Position nodes using a force-directed-like layout.
    fn position_nodes_organic(
        &self,
        metrics: &NeuralMetrics,
        rng: &mut impl Rng,
    ) -> Vec<PositionedNode> {
        let graph = &metrics.graph;
        let padding = 60.0;
        let w = self.width as f64 - padding * 2.0;
        let h = self.height as f64 - padding * 2.0;

        // Initial positions based on name hash + some randomness
        let mut positions: Vec<PositionedNode> = graph
            .nodes
            .iter()
            .map(|node| {
                let hash = Self::hash_to_float(&node.name);
                let angle = hash * PI * 2.0;
                let radius_factor = 0.3 + Self::hash_to_float(&format!("{}_r", node.name)) * 0.5;

                let cx = self.width as f64 / 2.0;
                let cy = self.height as f64 / 2.0;

                // Start in a rough circular pattern
                let x = cx + angle.cos() * w * 0.3 * radius_factor + rng.gen_range(-20.0..20.0);
                let y = cy + angle.sin() * h * 0.3 * radius_factor + rng.gen_range(-20.0..20.0);

                // Node size based on throughput
                let base_radius = 8.0;
                let throughput_factor = (node.throughput as f64).log10().max(1.0) / 6.0;
                let radius = base_radius + throughput_factor * 12.0;

                PositionedNode {
                    name: node.name.clone(),
                    x: x.clamp(padding, self.width as f64 - padding),
                    y: y.clamp(padding, self.height as f64 - padding),
                    radius,
                    throughput: node.throughput,
                }
            })
            .collect();

        // Simple force-directed iterations
        let iterations = 50;
        let repulsion = 5000.0;
        let attraction = 0.01;

        // Build edge lookup
        let edges: Vec<(usize, usize)> = graph
            .edges
            .iter()
            .filter_map(|e| {
                let src_idx = positions.iter().position(|n| n.name == e.source)?;
                let tgt_idx = positions.iter().position(|n| n.name == e.target)?;
                Some((src_idx, tgt_idx))
            })
            .collect();

        for _ in 0..iterations {
            let mut forces: Vec<(f64, f64)> = vec![(0.0, 0.0); positions.len()];

            // Repulsion between all nodes
            for i in 0..positions.len() {
                for j in (i + 1)..positions.len() {
                    let dx = positions[j].x - positions[i].x;
                    let dy = positions[j].y - positions[i].y;
                    let dist = (dx * dx + dy * dy).sqrt().max(1.0);

                    let force = repulsion / (dist * dist);
                    let fx = (dx / dist) * force;
                    let fy = (dy / dist) * force;

                    forces[i].0 -= fx;
                    forces[i].1 -= fy;
                    forces[j].0 += fx;
                    forces[j].1 += fy;
                }
            }

            // Attraction along edges
            for &(src, tgt) in &edges {
                let dx = positions[tgt].x - positions[src].x;
                let dy = positions[tgt].y - positions[src].y;
                let dist = (dx * dx + dy * dy).sqrt().max(1.0);

                let force = dist * attraction;
                let fx = (dx / dist) * force;
                let fy = (dy / dist) * force;

                forces[src].0 += fx;
                forces[src].1 += fy;
                forces[tgt].0 -= fx;
                forces[tgt].1 -= fy;
            }

            // Apply forces with damping
            let damping = 0.85;
            for (i, pos) in positions.iter_mut().enumerate() {
                pos.x = (pos.x + forces[i].0 * damping).clamp(padding, self.width as f64 - padding);
                pos.y =
                    (pos.y + forces[i].1 * damping).clamp(padding, self.height as f64 - padding);
            }
        }

        positions
    }

    /// Position nodes in a circle.
    fn position_nodes_circular(&self, metrics: &NeuralMetrics) -> Vec<PositionedNode> {
        let graph = &metrics.graph;
        let cx = self.width as f64 / 2.0;
        let cy = self.height as f64 / 2.0;
        let radius = (self.width.min(self.height) as f64 / 2.0) - 80.0;

        graph
            .nodes
            .iter()
            .enumerate()
            .map(|(i, node)| {
                let angle = (i as f64 / graph.nodes.len() as f64) * PI * 2.0 - PI / 2.0;
                let x = cx + angle.cos() * radius;
                let y = cy + angle.sin() * radius;

                let base_radius = 6.0;
                let throughput_factor = (node.throughput as f64).log10().max(1.0) / 6.0;
                let node_radius = base_radius + throughput_factor * 10.0;

                PositionedNode {
                    name: node.name.clone(),
                    x,
                    y,
                    radius: node_radius,
                    throughput: node.throughput,
                }
            })
            .collect()
    }

    /// Position nodes in hierarchical layers.
    fn position_nodes_hierarchical(
        &self,
        metrics: &NeuralMetrics,
        _rng: &mut impl Rng,
    ) -> Vec<PositionedNode> {
        let graph = &metrics.graph;

        // Simple layering: nodes with no incoming edges are "sources" (top),
        // nodes with no outgoing edges are "sinks" (bottom), rest in middle
        let mut layers: Vec<Vec<&str>> = vec![vec![], vec![], vec![]];

        for node in &graph.nodes {
            let has_incoming = graph.edges.iter().any(|e| e.target == node.name);
            let has_outgoing = graph.edges.iter().any(|e| e.source == node.name);

            match (has_incoming, has_outgoing) {
                (false, true) => layers[0].push(&node.name), // source
                (true, false) => layers[2].push(&node.name), // sink
                _ => layers[1].push(&node.name),             // middle
            }
        }

        // If a layer is empty, redistribute
        if layers[0].is_empty() && layers[1].is_empty() {
            layers[1] = layers[2].clone();
            layers[2].clear();
        }

        let padding = 60.0;
        let layer_height = (self.height as f64 - padding * 2.0) / 3.0;

        let mut positions = Vec::new();

        for (layer_idx, layer) in layers.iter().enumerate() {
            if layer.is_empty() {
                continue;
            }

            let y = padding + layer_height * (layer_idx as f64 + 0.5);
            let spacing = (self.width as f64 - padding * 2.0) / (layer.len() + 1) as f64;

            for (i, name) in layer.iter().enumerate() {
                let x = padding + spacing * (i + 1) as f64;

                let node = graph.nodes.iter().find(|n| &n.name == name).unwrap();
                let base_radius = 6.0;
                let throughput_factor = (node.throughput as f64).log10().max(1.0) / 6.0;
                let node_radius = base_radius + throughput_factor * 10.0;

                positions.push(PositionedNode {
                    name: name.to_string(),
                    x,
                    y,
                    radius: node_radius,
                    throughput: node.throughput,
                });
            }
        }

        positions
    }

    /// Draw curved edges between nodes.
    fn draw_edges(
        &self,
        positions: &[PositionedNode],
        edges: &[GraphEdge],
        rng: &mut impl Rng,
    ) -> Vec<String> {
        let pos_map: HashMap<&str, &PositionedNode> =
            positions.iter().map(|p| (p.name.as_str(), p)).collect();

        edges
            .iter()
            .filter_map(|edge| {
                let src = pos_map.get(edge.source.as_str())?;
                let tgt = pos_map.get(edge.target.as_str())?;

                // Calculate edge with some curve
                let dx = tgt.x - src.x;
                let dy = tgt.y - src.y;
                let dist = (dx * dx + dy * dy).sqrt();

                // Skip if too close
                if dist < src.radius + tgt.radius + 5.0 {
                    return None;
                }

                // Start/end points at node edges
                let angle = dy.atan2(dx);
                let start_x = src.x + angle.cos() * src.radius;
                let start_y = src.y + angle.sin() * src.radius;
                let end_x = tgt.x - angle.cos() * tgt.radius;
                let end_y = tgt.y - angle.sin() * tgt.radius;

                // Control point for bezier curve (perpendicular offset)
                let mid_x = (start_x + end_x) / 2.0;
                let mid_y = (start_y + end_y) / 2.0;
                let perp_angle = angle + PI / 2.0;
                let curve_amount = rng.gen_range(-30.0..30.0);
                let ctrl_x = mid_x + perp_angle.cos() * curve_amount;
                let ctrl_y = mid_y + perp_angle.sin() * curve_amount;

                // Stroke width based on rate
                let stroke_width = if let Some(rate) = edge.rate {
                    0.5 + (rate.log10().max(0.0) / 3.0).min(2.0)
                } else {
                    0.75
                };

                Some(format!(
                    r#"<path d="M {:.1} {:.1} Q {:.1} {:.1} {:.1} {:.1}" fill="none" stroke="black" stroke-width="{:.2}" opacity="0.6"/>"#,
                    start_x, start_y, ctrl_x, ctrl_y, end_x, end_y, stroke_width
                ))
            })
            .collect()
    }

    /// Draw nodes as circles.
    fn draw_nodes(&self, positions: &[PositionedNode]) -> Vec<String> {
        positions
            .iter()
            .map(|node| {
                // Outer circle
                let outer = format!(
                    r#"<circle cx="{:.1}" cy="{:.1}" r="{:.1}" fill="white" stroke="black" stroke-width="1.5"/>"#,
                    node.x, node.y, node.radius
                );

                // Inner fill based on activity (darker = more active)
                let inner_radius = node.radius * 0.6;
                let fill_opacity = (node.throughput as f64).log10().max(1.0) / 7.0;
                let inner = format!(
                    r#"<circle cx="{:.1}" cy="{:.1}" r="{:.1}" fill="black" opacity="{:.2}"/>"#,
                    node.x, node.y, inner_radius, fill_opacity.min(0.8)
                );

                format!("{}\n  {}", outer, inner)
            })
            .collect()
    }

    /// Generate organic style graph.
    fn generate_organic(&self, metrics: &NeuralMetrics, rng: &mut impl Rng) -> String {
        let positions = self.position_nodes_organic(metrics, rng);
        let edges = self.draw_edges(&positions, &metrics.graph.edges, rng);
        let nodes = self.draw_nodes(&positions);

        self.wrap_svg(&format!("{}\n{}", edges.join("\n"), nodes.join("\n")))
    }

    /// Generate circular style graph.
    fn generate_circular(&self, metrics: &NeuralMetrics, rng: &mut impl Rng) -> String {
        let positions = self.position_nodes_circular(metrics);
        let edges = self.draw_edges(&positions, &metrics.graph.edges, rng);
        let nodes = self.draw_nodes(&positions);

        self.wrap_svg(&format!("{}\n{}", edges.join("\n"), nodes.join("\n")))
    }

    /// Generate hierarchical style graph.
    fn generate_hierarchical(&self, metrics: &NeuralMetrics, rng: &mut impl Rng) -> String {
        let positions = self.position_nodes_hierarchical(metrics, rng);
        let edges = self.draw_edges(&positions, &metrics.graph.edges, rng);
        let nodes = self.draw_nodes(&positions);

        self.wrap_svg(&format!("{}\n{}", edges.join("\n"), nodes.join("\n")))
    }

    /// Generate constellation style with organic clusters.
    fn generate_constellation(&self, metrics: &NeuralMetrics, rng: &mut impl Rng) -> String {
        let positions = self.position_nodes_organic(metrics, rng);

        // Draw edges as thin lines with dots
        let edges: Vec<String> = metrics
            .graph
            .edges
            .iter()
            .filter_map(|edge| {
                let src = positions.iter().find(|p| p.name == edge.source)?;
                let tgt = positions.iter().find(|p| p.name == edge.target)?;

                let dx = tgt.x - src.x;
                let dy = tgt.y - src.y;
                let dist = (dx * dx + dy * dy).sqrt();

                if dist < src.radius + tgt.radius + 5.0 {
                    return None;
                }

                let angle = dy.atan2(dx);
                let start_x = src.x + angle.cos() * src.radius;
                let start_y = src.y + angle.sin() * src.radius;
                let end_x = tgt.x - angle.cos() * tgt.radius;
                let end_y = tgt.y - angle.sin() * tgt.radius;

                // Dotted line effect
                Some(format!(
                    r#"<line x1="{:.1}" y1="{:.1}" x2="{:.1}" y2="{:.1}" stroke="black" stroke-width="0.5" stroke-dasharray="2,3" opacity="0.5"/>"#,
                    start_x, start_y, end_x, end_y
                ))
            })
            .collect();

        // Draw nodes as stars/points
        let nodes: Vec<String> = positions
            .iter()
            .map(|node| {
                let r = node.radius;

                // Multi-pointed star effect
                let points: Vec<String> = (0..8)
                    .map(|i| {
                        let angle = (i as f64 / 8.0) * PI * 2.0;
                        let point_r = if i % 2 == 0 { r } else { r * 0.4 };
                        let x = node.x + angle.cos() * point_r;
                        let y = node.y + angle.sin() * point_r;
                        format!("{:.1},{:.1}", x, y)
                    })
                    .collect();

                format!(r#"<polygon points="{}" fill="black"/>"#, points.join(" "))
            })
            .collect();

        self.wrap_svg(&format!("{}\n{}", edges.join("\n"), nodes.join("\n")))
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

impl Generator for GraphGenerator {
    fn name(&self) -> &'static str {
        match self.style {
            GraphStyle::Organic => "graph_organic",
            GraphStyle::Circular => "graph_circular",
            GraphStyle::Hierarchical => "graph_hierarchical",
            GraphStyle::Constellation => "graph_constellation",
        }
    }

    fn generate(&self, metrics: &NeuralMetrics) -> String {
        let seed = metrics.to_seed();
        let seed_u64 = u64::from_le_bytes(seed[0..8].try_into().unwrap());
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed_u64);

        match self.style {
            GraphStyle::Organic => self.generate_organic(metrics, &mut rng),
            GraphStyle::Circular => self.generate_circular(metrics, &mut rng),
            GraphStyle::Hierarchical => self.generate_hierarchical(metrics, &mut rng),
            GraphStyle::Constellation => self.generate_constellation(metrics, &mut rng),
        }
    }

    fn extension(&self) -> &'static str {
        "svg"
    }
}
