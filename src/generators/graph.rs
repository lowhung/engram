//! Graph topology generator.
//!
//! Visualizes the actual module adjacency graph as a dendrite-like
//! network diagram. Modules are nodes, topics are connections.
//!
//! The visual structure reflects the actual graph topology:
//! - Node roles (source/sink/processor) influence position and appearance
//! - Edge weights and rates affect curve intensity
//! - Graph depth and branching factor shape the overall layout
//! - Randomization adds organic variation while preserving structure

use crate::generators::Generator;
use crate::metrics::{GraphEdge, GraphNode, NeuralMetrics};
use rand::{Rng, SeedableRng};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::f64::consts::PI;

/// Extended color palette with role-based hue ranges.
///
/// The palette extends beyond pure blues to include:
/// - Cyan/teal for sources (data producers)
/// - Deep blue for processors (transformers)
/// - Purple/violet for sinks (data consumers)
/// - Warm accents (amber/orange) for high-activity elements
mod palette {
    /// Background color
    pub const BG: &str = "#000000";

    /// Base hue ranges by role (HSL hue values)
    pub const HUE_SOURCE: f64 = 175.0; // Cyan/teal - data flows OUT
    pub const HUE_PROCESSOR: f64 = 210.0; // Blue - data flows THROUGH
    pub const HUE_SINK: f64 = 255.0; // Purple/violet - data flows IN

    /// Hue variation allowed within each role
    pub const HUE_VARIANCE: f64 = 15.0;

    /// Warm accent hue for high activity (amber/orange)
    pub const HUE_ACCENT: f64 = 35.0;

    /// Convert HSL to hex color string
    pub fn hsl_to_hex(h: f64, s: f64, l: f64) -> String {
        // Normalize hue to 0-360
        let h = ((h % 360.0) + 360.0) % 360.0;

        let c = (1.0 - (2.0 * l - 1.0).abs()) * s;
        let x = c * (1.0 - ((h / 60.0) % 2.0 - 1.0).abs());
        let m = l - c / 2.0;

        let (r, g, b) = if h < 60.0 {
            (c, x, 0.0)
        } else if h < 120.0 {
            (x, c, 0.0)
        } else if h < 180.0 {
            (0.0, c, x)
        } else if h < 240.0 {
            (0.0, x, c)
        } else if h < 300.0 {
            (x, 0.0, c)
        } else {
            (c, 0.0, x)
        };

        let r = ((r + m) * 255.0) as u8;
        let g = ((g + m) * 255.0) as u8;
        let b = ((b + m) * 255.0) as u8;

        format!("#{:02x}{:02x}{:02x}", r, g, b)
    }

    /// Get base hue for a node role
    pub fn role_hue(role: NodeRole) -> f64 {
        match role {
            NodeRole::Source => HUE_SOURCE,
            NodeRole::Processor => HUE_PROCESSOR,
            NodeRole::Sink => HUE_SINK,
        }
    }

    /// Get a glow color (lighter, desaturated version)
    pub fn glow_color(base_hue: f64) -> String {
        hsl_to_hex(base_hue, 0.6, 0.85)
    }

    /// Get an accent color for high-activity elements
    /// Blends toward warm amber based on intensity (0.0-1.0)
    pub fn accent_blend(base_hue: f64, intensity: f64) -> f64 {
        // Blend from base hue toward accent hue based on intensity
        let blend = intensity.clamp(0.0, 1.0);
        base_hue * (1.0 - blend * 0.3) + HUE_ACCENT * (blend * 0.3)
    }

    /// Role of a node in the graph topology
    #[derive(Debug, Clone, Copy, PartialEq)]
    pub enum NodeRole {
        Source,    // Only writes (data producers)
        Sink,      // Only reads (data consumers)
        Processor, // Both reads and writes (transformers)
    }
}

use palette::NodeRole;

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
}

impl Default for GraphGenerator {
    fn default() -> Self {
        Self {
            width: 2048,
            height: 2048,
            style: GraphStyle::Organic,
        }
    }
}

/// A positioned node for rendering with topology-aware properties.
struct PositionedNode {
    name: String,
    x: f64,
    y: f64,
    radius: f64,
    throughput: u64,
    rate: Option<f64>,
    /// Role in the graph topology
    role: NodeRole,
    /// Depth level in the graph (0 = source, higher = further downstream)
    depth: u32,
    /// Number of incoming connections
    in_degree: usize,
    /// Number of outgoing connections
    out_degree: usize,
    /// Assigned hue for consistent coloring
    hue: f64,
    /// Betweenness centrality (0.0-1.0) - how much of a bridge/hub
    centrality: f64,
    /// Clustering coefficient (0.0-1.0) - how clique-y the neighborhood
    clustering: f64,
}

impl GraphGenerator {
    pub fn new(width: u32, height: u32, style: GraphStyle) -> Self {
        Self {
            width,
            height,
            style,
        }
    }

    /// Scale factor for high-res output.
    fn scale(&self) -> f64 {
        self.width as f64 / 512.0
    }

    /// Hash a string to a deterministic float in [0, 1).
    fn hash_to_float(s: &str) -> f64 {
        let mut hasher = Sha256::new();
        hasher.update(s.as_bytes());
        let hash = hasher.finalize();
        let val = u32::from_le_bytes([hash[0], hash[1], hash[2], hash[3]]);
        val as f64 / u32::MAX as f64
    }

    /// Determine the role of a node based on its read/write patterns.
    fn node_role(node: &GraphNode) -> NodeRole {
        let has_reads = !node.reads.is_empty();
        let has_writes = !node.writes.is_empty();

        match (has_reads, has_writes) {
            (false, true) => NodeRole::Source,
            (true, false) => NodeRole::Sink,
            _ => NodeRole::Processor,
        }
    }

    /// Calculate the depth of each node in the graph (BFS from sources).
    fn calculate_depths(nodes: &[GraphNode], edges: &[GraphEdge]) -> HashMap<String, u32> {
        let mut depths: HashMap<String, u32> = HashMap::new();

        // Find sources (nodes with no incoming edges)
        let has_incoming: std::collections::HashSet<&str> =
            edges.iter().map(|e| e.target.as_str()).collect();

        // Initialize sources at depth 0
        let mut queue: Vec<(String, u32)> = Vec::new();
        for node in nodes {
            if !has_incoming.contains(node.name.as_str()) {
                depths.insert(node.name.clone(), 0);
                queue.push((node.name.clone(), 0));
            }
        }

        // If no sources found, start all nodes at depth 0
        if queue.is_empty() {
            for node in nodes {
                depths.insert(node.name.clone(), 0);
            }
            return depths;
        }

        // BFS to assign depths
        while let Some((current, current_depth)) = queue.pop() {
            for edge in edges {
                if edge.source == current {
                    let target_depth = current_depth + 1;
                    let entry = depths.entry(edge.target.clone()).or_insert(target_depth);
                    if *entry > target_depth {
                        *entry = target_depth;
                        queue.push((edge.target.clone(), target_depth));
                    }
                }
            }
        }

        // Assign remaining unvisited nodes a default depth
        for node in nodes {
            depths.entry(node.name.clone()).or_insert(1);
        }

        depths
    }

    /// Count incoming and outgoing edges for each node.
    fn calculate_degrees(
        nodes: &[GraphNode],
        edges: &[GraphEdge],
    ) -> (HashMap<String, usize>, HashMap<String, usize>) {
        let mut in_degrees: HashMap<String, usize> = HashMap::new();
        let mut out_degrees: HashMap<String, usize> = HashMap::new();

        for node in nodes {
            in_degrees.insert(node.name.clone(), 0);
            out_degrees.insert(node.name.clone(), 0);
        }

        for edge in edges {
            *out_degrees.entry(edge.source.clone()).or_insert(0) += 1;
            *in_degrees.entry(edge.target.clone()).or_insert(0) += 1;
        }

        (in_degrees, out_degrees)
    }

    /// Get activity level normalized to 0.0-1.0.
    fn activity_level(throughput: u64, rate: Option<f64>) -> f64 {
        let activity = rate.unwrap_or_else(|| (throughput as f64).log10().max(0.0) * 10.0);
        (activity / 200.0).min(1.0)
    }

    /// Position nodes using a force-directed-like layout.
    ///
    /// Uses graph topology to inform initial placement:
    /// - Nodes are positioned based on depth (sources at top, sinks at bottom)
    /// - High-degree nodes get more central positions
    /// - Randomization adds organic variation while preserving structure
    fn position_nodes_organic(
        &self,
        metrics: &NeuralMetrics,
        rng: &mut impl Rng,
    ) -> Vec<PositionedNode> {
        let graph = &metrics.graph;
        let scale = self.scale();
        let padding = 120.0 * scale;
        let w = self.width as f64 - padding * 2.0;
        let h = self.height as f64 - padding * 2.0;

        // Calculate topology information
        let depths = Self::calculate_depths(&graph.nodes, &graph.edges);
        let (in_degrees, out_degrees) = Self::calculate_degrees(&graph.nodes, &graph.edges);
        let max_depth = depths.values().copied().max().unwrap_or(1).max(1);

        // Calculate centrality metrics
        let centralities = graph.betweenness_centrality();
        let clusterings = graph.clustering_coefficients();

        // Initial positions based on topology + randomness
        let mut positions: Vec<PositionedNode> = graph
            .nodes
            .iter()
            .map(|node| {
                let role = Self::node_role(node);
                let depth = *depths.get(&node.name).unwrap_or(&0);
                let in_deg = *in_degrees.get(&node.name).unwrap_or(&0);
                let out_deg = *out_degrees.get(&node.name).unwrap_or(&0);
                let centrality = *centralities.get(&node.name).unwrap_or(&0.0);
                let clustering = *clusterings.get(&node.name).unwrap_or(&0.0);

                // Assign hue based on role with hash-based variation
                let role_hue = palette::role_hue(role);
                let hash_offset =
                    (Self::hash_to_float(&node.name) - 0.5) * palette::HUE_VARIANCE * 2.0;
                let hue = role_hue + hash_offset + rng.gen_range(-3.0..3.0);

                let cx = self.width as f64 / 2.0;
                let cy = self.height as f64 / 2.0;

                // Y position influenced by depth (sources higher, sinks lower)
                let depth_ratio = depth as f64 / max_depth as f64;
                let base_y = padding + depth_ratio * h * 0.7;

                // X position based on hash but clustered by role
                // High centrality nodes pulled toward center
                let hash = Self::hash_to_float(&node.name);
                let role_offset = match role {
                    NodeRole::Source => -0.2,
                    NodeRole::Sink => 0.2,
                    NodeRole::Processor => 0.0,
                };
                let center_pull = centrality * 0.3; // High centrality = more central
                let base_x = cx + (hash - 0.5 + role_offset) * w * 0.6 * (1.0 - center_pull);

                // Add jitter for organic feel - more jitter for less connected nodes
                // High centrality nodes get less jitter (more stable)
                let connectivity = (in_deg + out_deg) as f64;
                let jitter_factor = (1.0 / (1.0 + connectivity * 0.2)) * (1.0 - centrality * 0.5);
                let jitter = 60.0 * scale * jitter_factor;
                let x = base_x + rng.gen_range(-jitter..jitter);
                let y = if centrality > 0.1 {
                    // High centrality nodes also pulled toward vertical center
                    base_y * (1.0 - centrality * 0.2) + cy * centrality * 0.2
                } else {
                    base_y
                } + rng.gen_range(-jitter..jitter);

                // Node size based on throughput, connectivity, and centrality
                let base_radius = 14.0 * scale;
                let throughput_factor = (node.throughput as f64).log10().max(1.0) / 6.0;
                let degree_factor = (connectivity / 10.0).min(1.0);
                let centrality_factor = centrality * 2.0; // Centrality boosts size
                let radius = base_radius
                    + (throughput_factor * 0.5 + degree_factor * 0.25 + centrality_factor * 0.25)
                        * 28.0
                        * scale;

                PositionedNode {
                    name: node.name.clone(),
                    x: x.clamp(padding, self.width as f64 - padding),
                    y: y.clamp(padding, self.height as f64 - padding),
                    radius,
                    throughput: node.throughput,
                    rate: node.rate,
                    role,
                    depth,
                    in_degree: in_deg,
                    out_degree: out_deg,
                    hue,
                    centrality,
                    clustering,
                }
            })
            .collect();

        // Simple force-directed iterations
        let iterations = 80;
        let repulsion = 80000.0 * scale * scale;
        let attraction = 0.008;

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

    /// Position nodes in a circle, ordered by depth/role for visual flow.
    fn position_nodes_circular(
        &self,
        metrics: &NeuralMetrics,
        rng: &mut impl Rng,
    ) -> Vec<PositionedNode> {
        let graph = &metrics.graph;
        let scale = self.scale();
        let cx = self.width as f64 / 2.0;
        let cy = self.height as f64 / 2.0;
        let radius = (self.width.min(self.height) as f64 / 2.0) - 160.0 * scale;

        // Calculate topology
        let depths = Self::calculate_depths(&graph.nodes, &graph.edges);
        let (in_degrees, out_degrees) = Self::calculate_degrees(&graph.nodes, &graph.edges);
        let centralities = graph.betweenness_centrality();
        let clusterings = graph.clustering_coefficients();

        // Sort nodes by depth for better visual flow around the circle
        let mut sorted_nodes: Vec<_> = graph.nodes.iter().collect();
        sorted_nodes.sort_by_key(|n| depths.get(&n.name).copied().unwrap_or(0));

        sorted_nodes
            .iter()
            .enumerate()
            .map(|(i, node)| {
                let role = Self::node_role(node);
                let depth = *depths.get(&node.name).unwrap_or(&0);
                let in_deg = *in_degrees.get(&node.name).unwrap_or(&0);
                let out_deg = *out_degrees.get(&node.name).unwrap_or(&0);
                let centrality = *centralities.get(&node.name).unwrap_or(&0.0);
                let clustering = *clusterings.get(&node.name).unwrap_or(&0.0);

                // Assign hue based on role with hash-based variation
                let role_hue = palette::role_hue(role);
                let hash_offset =
                    (Self::hash_to_float(&node.name) - 0.5) * palette::HUE_VARIANCE * 2.0;
                let hue = role_hue + hash_offset + rng.gen_range(-3.0..3.0);

                let angle = (i as f64 / sorted_nodes.len() as f64) * PI * 2.0 - PI / 2.0;

                // Vary radius slightly based on role - sources slightly outside, sinks inside
                // High centrality nodes pulled toward center
                let radius_offset = match role {
                    NodeRole::Source => 20.0 * scale,
                    NodeRole::Sink => -20.0 * scale,
                    NodeRole::Processor => 0.0,
                };
                let centrality_pull = centrality * 40.0 * scale;
                let node_orbit =
                    radius + radius_offset - centrality_pull + rng.gen_range(-10.0..10.0) * scale;

                let x = cx + angle.cos() * node_orbit;
                let y = cy + angle.sin() * node_orbit;

                let base_radius = 12.0 * scale;
                let throughput_factor = (node.throughput as f64).log10().max(1.0) / 6.0;
                let connectivity = (in_deg + out_deg) as f64;
                let degree_factor = (connectivity / 10.0).min(1.0);
                let centrality_factor = centrality * 2.0;
                let node_radius = base_radius
                    + (throughput_factor * 0.5 + degree_factor * 0.25 + centrality_factor * 0.25)
                        * 24.0
                        * scale;

                PositionedNode {
                    name: node.name.clone(),
                    x,
                    y,
                    radius: node_radius,
                    throughput: node.throughput,
                    rate: node.rate,
                    role,
                    depth,
                    in_degree: in_deg,
                    out_degree: out_deg,
                    hue,
                    centrality,
                    clustering,
                }
            })
            .collect()
    }

    /// Position nodes in hierarchical layers based on actual graph depth.
    ///
    /// Uses graph topology for layered positioning:
    /// - Nodes grouped by depth (distance from sources)
    /// - High centrality nodes positioned more centrally within their layer
    /// - Node sizes influenced by centrality and clustering
    fn position_nodes_hierarchical(
        &self,
        metrics: &NeuralMetrics,
        rng: &mut impl Rng,
    ) -> Vec<PositionedNode> {
        let graph = &metrics.graph;
        let scale = self.scale();

        // Calculate actual depths from graph topology
        let depths = Self::calculate_depths(&graph.nodes, &graph.edges);
        let (in_degrees, out_degrees) = Self::calculate_degrees(&graph.nodes, &graph.edges);
        let max_depth = depths.values().copied().max().unwrap_or(0);

        // Calculate centrality metrics
        let centralities = graph.betweenness_centrality();
        let clusterings = graph.clustering_coefficients();

        // Group nodes by depth
        let mut layers: Vec<Vec<&GraphNode>> = vec![vec![]; (max_depth + 1) as usize];
        for node in &graph.nodes {
            let depth = *depths.get(&node.name).unwrap_or(&0) as usize;
            if depth < layers.len() {
                layers[depth].push(node);
            }
        }

        // Remove empty layers
        layers.retain(|l| !l.is_empty());

        let padding = 120.0 * scale;
        let cx = self.width as f64 / 2.0;
        let num_layers = layers.len().max(1);
        let layer_height = (self.height as f64 - padding * 2.0) / num_layers as f64;

        let mut positions = Vec::new();

        for (layer_idx, layer) in layers.iter().enumerate() {
            if layer.is_empty() {
                continue;
            }

            let base_y = padding + layer_height * (layer_idx as f64 + 0.5);

            // Sort nodes within layer by centrality (high centrality = more central position)
            let mut layer_nodes: Vec<_> = layer.iter().collect();
            layer_nodes.sort_by(|a, b| {
                let ca = centralities.get(&a.name).unwrap_or(&0.0);
                let cb = centralities.get(&b.name).unwrap_or(&0.0);
                cb.partial_cmp(ca).unwrap_or(std::cmp::Ordering::Equal)
            });

            let spacing = (self.width as f64 - padding * 2.0) / (layer.len() + 1) as f64;

            for (i, node) in layer_nodes.iter().enumerate() {
                let role = Self::node_role(node);
                let depth = *depths.get(&node.name).unwrap_or(&0);
                let in_deg = *in_degrees.get(&node.name).unwrap_or(&0);
                let out_deg = *out_degrees.get(&node.name).unwrap_or(&0);
                let centrality = *centralities.get(&node.name).unwrap_or(&0.0);
                let clustering = *clusterings.get(&node.name).unwrap_or(&0.0);

                // Assign hue based on role with hash-based variation
                let role_hue = palette::role_hue(role);
                let hash_offset =
                    (Self::hash_to_float(&node.name) - 0.5) * palette::HUE_VARIANCE * 2.0;
                let hue = role_hue + hash_offset + rng.gen_range(-3.0..3.0);

                // Position within layer - high centrality nodes get positioned toward center
                let natural_x = padding + spacing * (i + 1) as f64;
                let center_pull = centrality * 0.3;
                let base_x = natural_x * (1.0 - center_pull) + cx * center_pull;

                // Add slight jitter for organic feel - less jitter for high centrality
                let jitter = 15.0 * scale * (1.0 - centrality * 0.5);
                let x = base_x + rng.gen_range(-jitter..jitter);
                let y = base_y + rng.gen_range(-jitter..jitter);

                // Node size based on throughput, connectivity, and centrality
                let base_radius = 12.0 * scale;
                let throughput_factor = (node.throughput as f64).log10().max(1.0) / 6.0;
                let connectivity = (in_deg + out_deg) as f64;
                let degree_factor = (connectivity / 10.0).min(1.0);
                let centrality_factor = centrality * 2.0;
                let node_radius = base_radius
                    + (throughput_factor * 0.5 + degree_factor * 0.25 + centrality_factor * 0.25)
                        * 22.0
                        * scale;

                positions.push(PositionedNode {
                    name: node.name.clone(),
                    x,
                    y,
                    radius: node_radius,
                    throughput: node.throughput,
                    rate: node.rate,
                    role,
                    depth,
                    in_degree: in_deg,
                    out_degree: out_deg,
                    hue,
                    centrality,
                    clustering,
                });
            }
        }

        positions
    }

    /// Draw curved edges between nodes with topology-aware styling.
    ///
    /// Edge curves are influenced by:
    /// - Flow direction (top-to-bottom curves more naturally)
    /// - Rate/activity affects stroke width and opacity
    /// - Topic hash determines curve direction for consistency
    fn draw_edges(
        &self,
        positions: &[PositionedNode],
        edges: &[GraphEdge],
        rng: &mut impl Rng,
    ) -> Vec<String> {
        let scale = self.scale();
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
                if dist < src.radius + tgt.radius + 10.0 * scale {
                    return None;
                }

                // Start/end points at node edges
                let angle = dy.atan2(dx);
                let start_x = src.x + angle.cos() * src.radius;
                let start_y = src.y + angle.sin() * src.radius;
                let end_x = tgt.x - angle.cos() * tgt.radius;
                let end_y = tgt.y - angle.sin() * tgt.radius;

                // Curve amount based on topic hash (for consistency) + depth difference
                let topic_hash = Self::hash_to_float(&edge.topic);
                let depth_diff = (tgt.depth as i32 - src.depth as i32).abs() as f64;

                // More curve for longer distances and cross-depth connections
                let base_curve = 30.0 + depth_diff * 15.0;
                // Direction based on topic hash (deterministic per topic)
                let curve_direction = if topic_hash > 0.5 { 1.0 } else { -1.0 };
                // Add randomness but keep it bounded
                let curve_amount = (base_curve + rng.gen_range(-20.0..20.0)) * curve_direction * scale;

                // Control point for bezier curve (perpendicular offset)
                let mid_x = (start_x + end_x) / 2.0;
                let mid_y = (start_y + end_y) / 2.0;
                let perp_angle = angle + PI / 2.0;
                let ctrl_x = mid_x + perp_angle.cos() * curve_amount;
                let ctrl_y = mid_y + perp_angle.sin() * curve_amount;

                // Stroke width based on rate and connectivity
                let base_width = 1.2 * scale;
                let rate_factor = edge.rate
                    .map(|r| (r.log10().max(0.0) / 2.0).min(2.5))
                    .unwrap_or(0.0);
                // Thicker lines for high-degree connections
                let degree_factor = ((src.out_degree + tgt.in_degree) as f64 / 10.0).min(1.0);
                let stroke_width = base_width + (rate_factor * 0.7 + degree_factor * 0.3) * 2.5 * scale;

                // Color with hue variation - blend source and target hues
                // High activity edges shift toward warm accent color
                let base_edge_hue = (src.hue + tgt.hue) / 2.0 + rng.gen_range(-3.0..3.0);
                let activity = edge.rate.map(|r| (r / 100.0).min(1.0)).unwrap_or(0.2);
                let edge_hue = palette::accent_blend(base_edge_hue, activity);

                // More dramatic saturation variation based on activity
                let saturation = 0.4 + activity * 0.5; // 40-90%
                let lightness = 0.25 + activity * 0.25; // 25-50%
                let _base_color = palette::hsl_to_hex(edge_hue, saturation, lightness);

                // Opacity based on rate
                let opacity = if edge.rate.map(|r| r > 20.0).unwrap_or(false) {
                    0.6 + activity * 0.3
                } else {
                    0.3 + activity * 0.2
                };

                // Generate multiple fiber strokes for active edges
                let fiber_count = if activity > 0.5 { 3 } else if activity > 0.2 { 2 } else { 1 };

                let mut paths = Vec::new();

                for i in 0..fiber_count {
                    // Offset each fiber slightly perpendicular to the path
                    let fiber_offset = if fiber_count > 1 {
                        let spread = stroke_width * 0.8;
                        let offset_ratio = (i as f64 - (fiber_count - 1) as f64 / 2.0) / (fiber_count as f64);
                        offset_ratio * spread
                    } else {
                        0.0
                    };

                    // Apply offset to control point
                    let fiber_ctrl_x = ctrl_x + perp_angle.cos() * fiber_offset;
                    let fiber_ctrl_y = ctrl_y + perp_angle.sin() * fiber_offset;

                    // Individual fiber width (thinner than total)
                    let fiber_width = if fiber_count > 1 {
                        stroke_width / fiber_count as f64 * 1.2
                    } else {
                        stroke_width
                    };

                    // Slight opacity variation per fiber
                    let fiber_opacity = opacity * (0.8 + 0.2 * (i as f64 / fiber_count as f64));

                    // Slight hue shift per fiber for visual interest
                    let fiber_hue = edge_hue + (i as f64 - fiber_count as f64 / 2.0) * 3.0;
                    let fiber_color = palette::hsl_to_hex(fiber_hue, saturation, lightness);

                    paths.push(format!(
                        r#"<path d="M {:.1} {:.1} Q {:.1} {:.1} {:.1} {:.1}" fill="none" stroke="{}" stroke-width="{:.2}" opacity="{:.2}" stroke-linecap="round"/>"#,
                        start_x, start_y, fiber_ctrl_x, fiber_ctrl_y, end_x, end_y, fiber_color, fiber_width, fiber_opacity
                    ));
                }

                Some(paths.join("\n"))
            })
            .collect()
    }

    /// Draw nodes as glowing circles with role-based visual hierarchy.
    ///
    /// Uses SVG filters for smooth gaussian blur glows and radial gradients
    /// for natural light falloff. Visual distinctions by role:
    /// - Sources: Cyan-shifted, outer ring, intense glow
    /// - Sinks: Purple-shifted, softer glow
    /// - Processors: Balanced appearance
    ///
    /// Centrality and clustering influence appearance:
    /// - High centrality: Larger glow, more prominent stroke
    /// - High clustering: Subtle secondary ring (interconnected neighborhood)
    fn draw_nodes(&self, positions: &[PositionedNode], _rng: &mut impl Rng) -> Vec<String> {
        let scale = self.scale();

        positions
            .iter()
            .flat_map(|node| {
                let activity = Self::activity_level(node.throughput, node.rate);
                let node_id = node.name.replace(|c: char| !c.is_alphanumeric(), "_");

                let mut elements = Vec::new();

                // Glow radius varies by connectivity and centrality
                // High centrality = more prominent glow (hub nodes stand out)
                let connectivity_factor =
                    1.0 + ((node.in_degree + node.out_degree) as f64 / 15.0).min(0.5);
                let centrality_glow_boost = 1.0 + node.centrality * 0.5;

                // Outer glow using radial gradient (smooth falloff)
                let glow_multiplier = match node.role {
                    NodeRole::Source => 2.5,
                    NodeRole::Sink => 1.8,
                    NodeRole::Processor => 2.0,
                };
                let glow_radius =
                    node.radius * glow_multiplier * connectivity_factor * centrality_glow_boost;

                // Use filter for high-activity or high-centrality nodes
                let filter_attr = if activity > 0.5 || node.centrality > 0.3 {
                    r#" filter="url(#intenseGlow)""#
                } else {
                    ""
                };

                elements.push(format!(
                    r#"<circle cx="{:.1}" cy="{:.1}" r="{:.1}" fill="url(#glowGrad_{})"{}/>"#,
                    node.x, node.y, glow_radius, node_id, filter_attr
                ));

                // Core node with radial gradient fill
                // High centrality nodes get brighter, more saturated strokes
                let centrality_saturation_boost = node.centrality * 0.15;
                let centrality_lightness_boost = node.centrality * 0.1;
                let stroke_color = match node.role {
                    NodeRole::Source => palette::hsl_to_hex(
                        node.hue - 5.0,
                        0.8 + centrality_saturation_boost,
                        0.7 + centrality_lightness_boost,
                    ),
                    NodeRole::Sink => palette::hsl_to_hex(
                        node.hue + 5.0,
                        0.7 + centrality_saturation_boost,
                        0.6 + centrality_lightness_boost,
                    ),
                    NodeRole::Processor => palette::hsl_to_hex(
                        node.hue,
                        0.75 + centrality_saturation_boost,
                        0.65 + centrality_lightness_boost,
                    ),
                };
                // High centrality = thicker stroke (more prominent)
                let centrality_stroke_boost = node.centrality * 0.5;
                let stroke_width = match node.role {
                    NodeRole::Source => (2.0 + centrality_stroke_boost) * scale,
                    NodeRole::Sink => (1.0 + centrality_stroke_boost) * scale,
                    NodeRole::Processor => (1.5 + centrality_stroke_boost) * scale,
                };
                elements.push(format!(
                    r#"<circle cx="{:.1}" cy="{:.1}" r="{:.1}" fill="url(#nodeGrad_{})" stroke="{}" stroke-width="{:.1}" filter="url(#softGlow)"/>"#,
                    node.x, node.y, node.radius, node_id, stroke_color, stroke_width
                ));

                // Bright center highlight - gives that "lit from within" look
                // High centrality nodes have slightly larger, brighter centers
                let center_radius =
                    node.radius * (0.25 + activity * 0.15 + node.centrality * 0.1);
                let glow_color = palette::glow_color(node.hue);
                let center_opacity = 0.5 + activity * 0.4 + node.centrality * 0.1;
                elements.push(format!(
                    r#"<circle cx="{:.1}" cy="{:.1}" r="{:.1}" fill="{}" opacity="{:.2}"/>"#,
                    node.x - node.radius * 0.2,
                    node.y - node.radius * 0.2,
                    center_radius,
                    glow_color,
                    center_opacity.min(1.0)
                ));

                // Sources get an additional outer ring to show they're emitters
                if node.role == NodeRole::Source {
                    let ring_radius = node.radius * 1.2;
                    elements.push(format!(
                        r#"<circle cx="{:.1}" cy="{:.1}" r="{:.1}" fill="none" stroke="{}" stroke-width="{:.1}" opacity="0.5" filter="url(#softGlow)"/>"#,
                        node.x, node.y, ring_radius, stroke_color, 1.5 * scale
                    ));
                }

                // High clustering nodes get a subtle secondary halo ring
                // This indicates they're part of a tightly interconnected neighborhood
                if node.clustering > 0.4 {
                    let cluster_ring_radius = node.radius * 1.4;
                    let cluster_opacity = (node.clustering - 0.4) * 0.5; // 0.0 to 0.3
                    let cluster_color =
                        palette::hsl_to_hex(node.hue + 15.0, 0.5, 0.6); // Slightly shifted hue
                    elements.push(format!(
                        r#"<circle cx="{:.1}" cy="{:.1}" r="{:.1}" fill="none" stroke="{}" stroke-width="{:.1}" opacity="{:.2}" stroke-dasharray="{:.1} {:.1}"/>"#,
                        node.x,
                        node.y,
                        cluster_ring_radius,
                        cluster_color,
                        1.0 * scale,
                        cluster_opacity,
                        3.0 * scale,
                        3.0 * scale
                    ));
                }

                elements
            })
            .collect()
    }

    /// Generate organic style graph.
    fn generate_organic(&self, metrics: &NeuralMetrics, rng: &mut impl Rng) -> String {
        let positions = self.position_nodes_organic(metrics, rng);
        let defs = self.generate_defs(&positions);
        let edges = self.draw_edges(&positions, &metrics.graph.edges, rng);
        let nodes = self.draw_nodes(&positions, rng);

        self.wrap_svg(
            &defs,
            &format!("{}\n{}", edges.join("\n"), nodes.join("\n")),
        )
    }

    /// Generate circular style graph.
    fn generate_circular(&self, metrics: &NeuralMetrics, rng: &mut impl Rng) -> String {
        let positions = self.position_nodes_circular(metrics, rng);
        let defs = self.generate_defs(&positions);
        let edges = self.draw_edges(&positions, &metrics.graph.edges, rng);
        let nodes = self.draw_nodes(&positions, rng);

        self.wrap_svg(
            &defs,
            &format!("{}\n{}", edges.join("\n"), nodes.join("\n")),
        )
    }

    /// Generate hierarchical style graph.
    fn generate_hierarchical(&self, metrics: &NeuralMetrics, rng: &mut impl Rng) -> String {
        let positions = self.position_nodes_hierarchical(metrics, rng);
        let defs = self.generate_defs(&positions);
        let edges = self.draw_edges(&positions, &metrics.graph.edges, rng);
        let nodes = self.draw_nodes(&positions, rng);

        self.wrap_svg(
            &defs,
            &format!("{}\n{}", edges.join("\n"), nodes.join("\n")),
        )
    }

    /// Generate SVG filter definitions for glow effects and background textures.
    fn generate_defs(&self, positions: &[PositionedNode]) -> String {
        let scale = self.scale();
        let mut defs = String::from("<defs>\n");

        // Noise texture filter for background grain
        defs.push_str(&format!(
            r#"  <filter id="noiseFilter" x="0%" y="0%" width="100%" height="100%">
    <feTurbulence type="fractalNoise" baseFrequency="{:.4}" numOctaves="3" result="noise"/>
    <feColorMatrix type="matrix" values="0 0 0 0 0.03  0 0 0 0 0.05  0 0 0 0 0.08  0 0 0 0.15 0"/>
  </filter>
"#,
            0.5 / scale // Adjust frequency based on scale
        ));

        // Vignette gradient (radial fade from transparent center to dark edges)
        defs.push_str(
            "  <radialGradient id=\"vignette\" cx=\"50%\" cy=\"50%\" r=\"70%\" fx=\"50%\" fy=\"50%\">\n\
             \x20\x20\x20\x20<stop offset=\"0%\" stop-color=\"#000000\" stop-opacity=\"0\"/>\n\
             \x20\x20\x20\x20<stop offset=\"60%\" stop-color=\"#000000\" stop-opacity=\"0\"/>\n\
             \x20\x20\x20\x20<stop offset=\"100%\" stop-color=\"#000000\" stop-opacity=\"0.7\"/>\n\
             \x20\x20</radialGradient>\n",
        );

        // Soft glow filter - used for outer node halos
        let blur_std = 8.0 * scale;
        defs.push_str(&format!(
            r#"  <filter id="softGlow" x="-100%" y="-100%" width="300%" height="300%">
    <feGaussianBlur in="SourceGraphic" stdDeviation="{:.1}" result="blur"/>
    <feMerge>
      <feMergeNode in="blur"/>
      <feMergeNode in="blur"/>
      <feMergeNode in="SourceGraphic"/>
    </feMerge>
  </filter>
"#,
            blur_std
        ));

        // Intense glow for high-activity nodes
        let intense_blur = 12.0 * scale;
        defs.push_str(&format!(
            r#"  <filter id="intenseGlow" x="-150%" y="-150%" width="400%" height="400%">
    <feGaussianBlur in="SourceGraphic" stdDeviation="{:.1}" result="blur"/>
    <feMerge>
      <feMergeNode in="blur"/>
      <feMergeNode in="blur"/>
      <feMergeNode in="blur"/>
      <feMergeNode in="SourceGraphic"/>
    </feMerge>
  </filter>
"#,
            intense_blur
        ));

        // Edge glow filter - subtler
        let edge_blur = 4.0 * scale;
        defs.push_str(&format!(
            r#"  <filter id="edgeGlow" x="-50%" y="-50%" width="200%" height="200%">
    <feGaussianBlur in="SourceGraphic" stdDeviation="{:.1}" result="blur"/>
    <feMerge>
      <feMergeNode in="blur"/>
      <feMergeNode in="SourceGraphic"/>
    </feMerge>
  </filter>
"#,
            edge_blur
        ));

        // Generate radial gradients for each node
        for node in positions {
            let activity = Self::activity_level(node.throughput, node.rate);

            // High activity nodes get warm-shifted hue
            let node_hue = palette::accent_blend(node.hue, activity * 0.5);

            // Core gradient (bright center fading to node color)
            // More dramatic saturation range based on activity
            let center_saturation = 0.5 + activity * 0.3;
            let edge_saturation = 0.6 + activity * 0.35;
            let center_lightness = 0.65 + activity * 0.2;
            let edge_lightness = 0.28 + activity * 0.17;
            let center_color = palette::hsl_to_hex(node_hue, center_saturation, center_lightness);
            let edge_color = palette::hsl_to_hex(node_hue, edge_saturation, edge_lightness);

            defs.push_str(&format!(
                r#"  <radialGradient id="nodeGrad_{}" cx="30%" cy="30%" r="70%" fx="30%" fy="30%">
    <stop offset="0%" stop-color="{}"/>
    <stop offset="100%" stop-color="{}"/>
  </radialGradient>
"#,
                node.name.replace(|c: char| !c.is_alphanumeric(), "_"),
                center_color,
                edge_color
            ));

            // Glow gradient (for the outer halo)
            let glow_color = palette::glow_color(node.hue);
            defs.push_str(&format!(
                r#"  <radialGradient id="glowGrad_{}">
    <stop offset="0%" stop-color="{}" stop-opacity="0.4"/>
    <stop offset="60%" stop-color="{}" stop-opacity="0.15"/>
    <stop offset="100%" stop-color="{}" stop-opacity="0"/>
  </radialGradient>
"#,
                node.name.replace(|c: char| !c.is_alphanumeric(), "_"),
                glow_color,
                glow_color,
                glow_color
            ));
        }

        defs.push_str("</defs>");
        defs
    }

    fn wrap_svg(&self, defs: &str, content: &str) -> String {
        format!(
            r#"<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {} {}" width="{}" height="{}">
  {}
  <!-- Background -->
  <rect width="100%" height="100%" fill="{}"/>
  <!-- Noise texture overlay -->
  <rect width="100%" height="100%" filter="url(#noiseFilter)" opacity="1"/>
  <!-- Graph content -->
  {}
  <!-- Vignette overlay -->
  <rect width="100%" height="100%" fill="url(#vignette)"/>
</svg>"#,
            self.width,
            self.height,
            self.width,
            self.height,
            defs,
            palette::BG,
            content
        )
    }
}

impl Generator for GraphGenerator {
    fn name(&self) -> &'static str {
        match self.style {
            GraphStyle::Organic => "graph_organic",
            GraphStyle::Circular => "graph_circular",
            GraphStyle::Hierarchical => "graph_hierarchical",
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
        }
    }

    fn extension(&self) -> &'static str {
        "svg"
    }
}
