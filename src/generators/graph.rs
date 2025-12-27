//! Graph topology generator.
//!
//! Visualizes the actual module adjacency graph as a bold, graphic
//! network diagram. Clean, solid colors - no gradients, no glows.
//!
//! The visual structure reflects the actual graph topology:
//! - Node roles (source/sink/processor) influence position and appearance
//! - Multi-ring or solid nodes based on randomization
//! - Always black edges, solid strokes, no arrowheads

use crate::generators::Generator;
use crate::metrics::{GraphEdge, GraphNode, NeuralMetrics, NodeTemporalMetrics};
use rand::{Rng, SeedableRng};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::f64::consts::PI;

/// Color palette - solid colors, configurable.
pub mod palette {
    /// The color palette configuration
    #[derive(Debug, Clone)]
    pub struct ColorPalette {
        pub background: String,
        pub node_colors: Vec<String>,
        pub edge_color: String,
    }

    impl Default for ColorPalette {
        fn default() -> Self {
            // Default: use a curated subset of bold Catppuccin colors
            Self {
                background: BASE.to_string(),
                node_colors: vec![
                    MAUVE.to_string(),
                    PINK.to_string(),
                    PEACH.to_string(),
                    SAPPHIRE.to_string(),
                    LAVENDER.to_string(),
                ],
                edge_color: TEXT.to_string(),
            }
        }
    }

    impl ColorPalette {
        /// Get a node color by index (cycles through available colors)
        pub fn node_color(&self, index: usize) -> &str {
            &self.node_colors[index % self.node_colors.len()]
        }

        /// Create a new palette with custom colors
        pub fn new(background: &str, node_colors: Vec<&str>, edge_color: &str) -> Self {
            Self {
                background: background.to_string(),
                node_colors: node_colors.into_iter().map(|s| s.to_string()).collect(),
                edge_color: edge_color.to_string(),
            }
        }

        /// Create from a list of hex color strings for nodes
        pub fn with_node_colors(node_colors: Vec<&str>) -> Self {
            Self {
                background: BASE.to_string(),
                node_colors: node_colors.into_iter().map(|s| s.to_string()).collect(),
                edge_color: TEXT.to_string(),
            }
        }

        /// Create a randomized palette using the seed
        /// Picks a random background and a subset of node colors from all Catppuccin colors
        pub fn random(seed: u64) -> Self {
            use rand::seq::SliceRandom;
            use rand::{Rng, SeedableRng};
            let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

            // Dark/neutral backgrounds
            let dark_bgs = [BASE, MANTLE, CRUST, SURFACE0, SURFACE1, SURFACE2];
            // Vibrant backgrounds
            let vibrant_bgs = [
                ROSEWATER, FLAMINGO, PINK, MAUVE, RED, MAROON, PEACH, YELLOW, GREEN, TEAL, SKY,
                SAPPHIRE, BLUE, LAVENDER,
            ];

            // 50/50 chance of dark vs vibrant background
            let background = if rng.gen_bool(0.5) {
                dark_bgs[rng.gen_range(0..dark_bgs.len())]
            } else {
                vibrant_bgs[rng.gen_range(0..vibrant_bgs.len())]
            };

            // All colors for nodes
            let all_colors = [
                BASE, MANTLE, CRUST, SURFACE0, SURFACE1, SURFACE2, ROSEWATER, FLAMINGO, PINK,
                MAUVE, RED, MAROON, PEACH, YELLOW, GREEN, TEAL, SKY, SAPPHIRE, BLUE, LAVENDER,
            ];

            // For node colors, pick from colors that contrast with background
            // Remove the background color from available node colors
            let available_for_nodes: Vec<&str> = all_colors
                .iter()
                .filter(|&&c| c != background)
                .copied()
                .collect();

            // Pick 3-6 colors randomly
            let color_count = rng.gen_range(3..=6);
            let mut colors = available_for_nodes;
            colors.shuffle(&mut rng);
            let node_colors: Vec<String> = colors
                .iter()
                .take(color_count)
                .map(|s| s.to_string())
                .collect();

            // Edge color: use a contrasting neutral
            // If background is dark, use light text; if light/vibrant, use dark
            let dark_backgrounds = [BASE, MANTLE, CRUST];
            let edge_color = if dark_backgrounds.contains(&background) {
                TEXT // Light text on dark
            } else {
                CRUST // Dark on light/vibrant
            };

            Self {
                background: background.to_string(),
                node_colors,
                edge_color: edge_color.to_string(),
            }
        }
    }

    // Catppuccin Mocha palette (dark variant)
    pub const ROSEWATER: &str = "#f5e0dc";
    pub const FLAMINGO: &str = "#f2cdcd";
    pub const PINK: &str = "#f5c2e7";
    pub const MAUVE: &str = "#cba6f7";
    pub const RED: &str = "#f38ba8";
    pub const MAROON: &str = "#eba0ac";
    pub const PEACH: &str = "#fab387";
    pub const YELLOW: &str = "#f9e2af";
    pub const GREEN: &str = "#a6e3a1";
    pub const TEAL: &str = "#94e2d5";
    pub const SKY: &str = "#89dceb";
    pub const SAPPHIRE: &str = "#74c7ec";
    pub const BLUE: &str = "#89b4fa";
    pub const LAVENDER: &str = "#b4befe";

    // Catppuccin Mocha neutrals
    pub const TEXT: &str = "#cdd6f4";
    pub const SUBTEXT1: &str = "#bac2de";
    pub const SUBTEXT0: &str = "#a6adc8";
    pub const OVERLAY2: &str = "#9399b2";
    pub const OVERLAY1: &str = "#7f849c";
    pub const OVERLAY0: &str = "#6c7086";
    pub const SURFACE2: &str = "#585b70";
    pub const SURFACE1: &str = "#45475a";
    pub const SURFACE0: &str = "#313244";
    pub const BASE: &str = "#1e1e2e";
    pub const MANTLE: &str = "#181825";
    pub const CRUST: &str = "#11111b";

    /// Role of a node in the graph topology
    #[derive(Debug, Clone, Copy, PartialEq)]
    pub enum NodeRole {
        Source,    // Only writes (data producers)
        Sink,      // Only reads (data consumers)
        Processor, // Both reads and writes (transformers)
    }
}

use palette::{ColorPalette, NodeRole};

pub struct GraphGenerator {
    pub width: u32,
    pub height: u32,
    pub style: GraphStyle,
    pub palette: ColorPalette,
    /// Maximum nodes to display (downsamples larger graphs)
    pub max_nodes: Option<usize>,
}

#[derive(Debug, Clone, Copy)]
pub enum GraphStyle {
    /// Organic force-directed layout with curved edges
    Organic,
    /// Circular arrangement with modules on a ring
    Circular,
    /// Hierarchical top-to-bottom flow
    Hierarchical,
    /// Grid-based layout
    Grid,
    /// Spiral outward from center
    Spiral,
    /// Concentric rings based on node depth
    Radial,
    /// Tight clusters connected by long edges
    Clustered,
    /// Random scattered positions
    Scatter,
    /// Fixed NxN matrix with nodes at grid intersections, edges route around obstacles
    Matrix,
}

impl Default for GraphGenerator {
    fn default() -> Self {
        Self {
            width: 2048,
            height: 2048,
            style: GraphStyle::Organic,
            palette: ColorPalette::default(),
            max_nodes: Some(12),
        }
    }
}

/// A positioned node for rendering with topology-aware properties.
#[allow(dead_code)]
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
    /// Index into the color palette for this node
    color_index: usize,
    /// Betweenness centrality (0.0-1.0) - how much of a bridge/hub
    centrality: f64,
    /// Clustering coefficient (0.0-1.0) - how clique-y the neighborhood
    clustering: f64,
    /// PageRank importance score (0.0-1.0)
    pagerank: f64,
    /// Whether this node is part of a cycle/feedback loop
    in_cycle: bool,
    /// Temporal metrics from multi-snapshot capture
    temporal: NodeTemporalMetrics,
    /// Number of concentric rings (1 = solid, 2-4 = concentric)
    ring_count: u8,
}

impl GraphGenerator {
    pub fn new(width: u32, height: u32, style: GraphStyle) -> Self {
        Self {
            width,
            height,
            style,
            palette: ColorPalette::default(),
            max_nodes: Some(12), // Default to sparse samples
        }
    }

    pub fn with_palette(mut self, palette: ColorPalette) -> Self {
        self.palette = palette;
        self
    }

    pub fn with_max_nodes(mut self, max_nodes: Option<usize>) -> Self {
        self.max_nodes = max_nodes;
        self
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
        let pageranks = graph.pagerank(0.85, 20);
        let cycle_nodes: std::collections::HashSet<String> =
            graph.find_cycle_nodes().into_iter().collect();

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
                let pagerank = *pageranks.get(&node.name).unwrap_or(&0.0);
                let in_cycle = cycle_nodes.contains(&node.name);
                let temporal = metrics
                    .temporal
                    .nodes
                    .get(&node.name)
                    .cloned()
                    .unwrap_or_default();

                // Assign color index based on node name hash (deterministic)
                let hash = Self::hash_to_float(&node.name);
                let color_index = (hash * 1000.0) as usize;

                // Randomly decide ring count: 1 (solid) or 2-4 (concentric)
                // ~50% solid, ~50% concentric with varying ring counts
                let ring_count = if rng.gen_bool(0.5) {
                    1 // Solid circle
                } else {
                    rng.gen_range(2..=4) // Concentric rings
                };

                let cx = self.width as f64 / 2.0;
                let cy = self.height as f64 / 2.0;

                // Y position influenced by depth (sources higher, sinks lower)
                let depth_ratio = depth as f64 / max_depth as f64;
                let base_y = padding + depth_ratio * h * 0.7;

                // X position based on hash but clustered by role
                // High centrality/pagerank nodes pulled toward center
                let hash = Self::hash_to_float(&node.name);
                let role_offset = match role {
                    NodeRole::Source => -0.2,
                    NodeRole::Sink => 0.2,
                    NodeRole::Processor => 0.0,
                };
                let importance = (centrality + pagerank) / 2.0; // Combined importance
                let center_pull = importance * 0.35;
                let base_x = cx + (hash - 0.5 + role_offset) * w * 0.6 * (1.0 - center_pull);

                // Add jitter for organic feel - more jitter for less connected nodes
                // High centrality nodes get less jitter (more stable)
                // High rate_variance nodes get MORE jitter (visually unstable)
                let connectivity = (in_deg + out_deg) as f64;
                let stability_factor = 1.0 - (temporal.rate_variance / 2.0).min(1.0);
                let jitter_factor = (1.0 / (1.0 + connectivity * 0.2))
                    * (1.0 - importance * 0.5)
                    * (0.5 + (1.0 - stability_factor) * 0.5);
                let jitter = 60.0 * scale * jitter_factor;
                let x = base_x + rng.gen_range(-jitter..jitter);
                let y = if importance > 0.1 {
                    // High importance nodes also pulled toward vertical center
                    base_y * (1.0 - importance * 0.2) + cy * importance * 0.2
                } else {
                    base_y
                } + rng.gen_range(-jitter..jitter);

                // Node size based on throughput, connectivity, centrality, and pagerank
                let base_radius = 14.0 * scale;
                let throughput_factor = (node.throughput as f64).log10().max(1.0) / 6.0;
                let degree_factor = (connectivity / 10.0).min(1.0);
                let importance_factor = importance * 2.0; // Combined importance boosts size
                let radius = base_radius
                    + (throughput_factor * 0.4 + degree_factor * 0.25 + importance_factor * 0.35)
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
                    color_index,
                    centrality,
                    clustering,
                    pagerank,
                    in_cycle,
                    temporal,
                    ring_count,
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
        let pageranks = graph.pagerank(0.85, 20);
        let cycle_nodes: std::collections::HashSet<String> =
            graph.find_cycle_nodes().into_iter().collect();

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
                let pagerank = *pageranks.get(&node.name).unwrap_or(&0.0);
                let in_cycle = cycle_nodes.contains(&node.name);
                let temporal = metrics
                    .temporal
                    .nodes
                    .get(&node.name)
                    .cloned()
                    .unwrap_or_default();

                // Assign color index based on node name hash (deterministic)
                let hash = Self::hash_to_float(&node.name);
                let color_index = (hash * 1000.0) as usize;

                // Randomly decide ring count: 1 (solid) or 2-4 (concentric)
                let ring_count = if rng.gen_bool(0.5) {
                    1
                } else {
                    rng.gen_range(2..=4)
                };

                let angle = (i as f64 / sorted_nodes.len() as f64) * PI * 2.0 - PI / 2.0;

                // Vary radius slightly based on role - sources slightly outside, sinks inside
                // High importance nodes pulled toward center
                let importance = (centrality + pagerank) / 2.0;
                let radius_offset = match role {
                    NodeRole::Source => 20.0 * scale,
                    NodeRole::Sink => -20.0 * scale,
                    NodeRole::Processor => 0.0,
                };
                let importance_pull = importance * 50.0 * scale;
                let node_orbit =
                    radius + radius_offset - importance_pull + rng.gen_range(-10.0..10.0) * scale;

                let x = cx + angle.cos() * node_orbit;
                let y = cy + angle.sin() * node_orbit;

                let base_radius = 12.0 * scale;
                let throughput_factor = (node.throughput as f64).log10().max(1.0) / 6.0;
                let connectivity = (in_deg + out_deg) as f64;
                let degree_factor = (connectivity / 10.0).min(1.0);
                let importance_factor = importance * 2.0;
                let node_radius = base_radius
                    + (throughput_factor * 0.4 + degree_factor * 0.25 + importance_factor * 0.35)
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
                    color_index,
                    centrality,
                    clustering,
                    pagerank,
                    in_cycle,
                    temporal,
                    ring_count,
                }
            })
            .collect()
    }

    /// Position nodes in hierarchical layers based on actual graph depth.
    ///
    /// Uses graph topology for layered positioning:
    /// - Nodes grouped by depth (distance from sources)
    /// - High centrality/pagerank nodes positioned more centrally within their layer
    /// - Node sizes influenced by centrality, pagerank, and clustering
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
        let pageranks = graph.pagerank(0.85, 20);
        let cycle_nodes: std::collections::HashSet<String> =
            graph.find_cycle_nodes().into_iter().collect();

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

            // Sort nodes within layer by importance (high importance = more central position)
            let mut layer_nodes: Vec<_> = layer.iter().collect();
            layer_nodes.sort_by(|a, b| {
                let ca = centralities.get(&a.name).unwrap_or(&0.0);
                let pa = pageranks.get(&a.name).unwrap_or(&0.0);
                let cb = centralities.get(&b.name).unwrap_or(&0.0);
                let pb = pageranks.get(&b.name).unwrap_or(&0.0);
                let importance_a = (ca + pa) / 2.0;
                let importance_b = (cb + pb) / 2.0;
                importance_b
                    .partial_cmp(&importance_a)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

            let spacing = (self.width as f64 - padding * 2.0) / (layer.len() + 1) as f64;

            for (i, node) in layer_nodes.iter().enumerate() {
                let role = Self::node_role(node);
                let depth = *depths.get(&node.name).unwrap_or(&0);
                let in_deg = *in_degrees.get(&node.name).unwrap_or(&0);
                let out_deg = *out_degrees.get(&node.name).unwrap_or(&0);
                let centrality = *centralities.get(&node.name).unwrap_or(&0.0);
                let clustering = *clusterings.get(&node.name).unwrap_or(&0.0);
                let pagerank = *pageranks.get(&node.name).unwrap_or(&0.0);
                let in_cycle = cycle_nodes.contains(&node.name);
                let temporal = metrics
                    .temporal
                    .nodes
                    .get(&node.name)
                    .cloned()
                    .unwrap_or_default();

                // Assign color index based on node name hash (deterministic)
                let hash = Self::hash_to_float(&node.name);
                let color_index = (hash * 1000.0) as usize;

                // Randomly decide ring count: 1 (solid) or 2-4 (concentric)
                let ring_count = if rng.gen_bool(0.5) {
                    1
                } else {
                    rng.gen_range(2..=4)
                };

                // Position within layer - high importance nodes get positioned toward center
                let importance = (centrality + pagerank) / 2.0;
                let natural_x = padding + spacing * (i + 1) as f64;
                let center_pull = importance * 0.35;
                let base_x = natural_x * (1.0 - center_pull) + cx * center_pull;

                // Add slight jitter for organic feel - less jitter for high importance
                let jitter = 15.0 * scale * (1.0 - importance * 0.5);
                let x = base_x + rng.gen_range(-jitter..jitter);
                let y = base_y + rng.gen_range(-jitter..jitter);

                // Node size based on throughput, connectivity, and importance
                let base_radius = 12.0 * scale;
                let throughput_factor = (node.throughput as f64).log10().max(1.0) / 6.0;
                let connectivity = (in_deg + out_deg) as f64;
                let degree_factor = (connectivity / 10.0).min(1.0);
                let importance_factor = importance * 2.0;
                let node_radius = base_radius
                    + (throughput_factor * 0.4 + degree_factor * 0.25 + importance_factor * 0.35)
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
                    color_index,
                    centrality,
                    clustering,
                    pagerank,
                    in_cycle,
                    temporal,
                    ring_count,
                });
            }
        }

        positions
    }

    /// Draw edges between nodes.
    ///
    /// Edges take on the color of their source node.
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

                // Curve amount based on topic hash (for consistency)
                let topic_hash = Self::hash_to_float(&edge.topic);
                let depth_diff = (tgt.depth as i32 - src.depth as i32).abs() as f64;
                let base_curve = 30.0 + depth_diff * 15.0;
                let curve_direction = if topic_hash > 0.5 { 1.0 } else { -1.0 };
                let curve_amount = (base_curve + rng.gen_range(-20.0..20.0)) * curve_direction * scale;

                // Control point for bezier curve
                let mid_x = (start_x + end_x) / 2.0;
                let mid_y = (start_y + end_y) / 2.0;
                let perp_angle = angle + PI / 2.0;
                let ctrl_x = mid_x + perp_angle.cos() * curve_amount;
                let ctrl_y = mid_y + perp_angle.sin() * curve_amount;

                // Stroke width based on rate and connectivity
                let rate = edge.rate.unwrap_or(10.0);
                let base_width = 1.5 * scale;
                let rate_factor = (rate.log10().max(0.0) / 2.0).min(2.0);
                let degree_factor = ((src.out_degree + tgt.in_degree) as f64 / 10.0).min(1.0);
                let stroke_width = base_width + (rate_factor * 0.6 + degree_factor * 0.4) * 2.0 * scale;

                // Edge color from source node
                let edge_color = self.palette.node_color(src.color_index);

                Some(format!(
                    r#"<path d="M {:.1} {:.1} Q {:.1} {:.1} {:.1} {:.1}" fill="none" stroke="{}" stroke-width="{:.2}" stroke-linecap="round"/>"#,
                    start_x, start_y, ctrl_x, ctrl_y, end_x, end_y, edge_color, stroke_width
                ))
            })
            .collect()
    }

    /// Draw nodes as solid circles or concentric rings.
    ///
    /// No gradients, no glow - just bold solid shapes.
    fn draw_nodes(&self, positions: &[PositionedNode]) -> Vec<String> {
        positions
            .iter()
            .flat_map(|node| {
                let fill_color = self.palette.node_color(node.color_index);
                let mut elements = Vec::new();

                if node.ring_count == 1 {
                    // Solid filled circle
                    elements.push(format!(
                        r#"<circle cx="{:.1}" cy="{:.1}" r="{:.1}" fill="{}"/>"#,
                        node.x, node.y, node.radius, fill_color
                    ));
                } else {
                    // Concentric rings - draw from outside in
                    let ring_spacing = node.radius / (node.ring_count as f64 + 0.5);
                    let stroke_width = ring_spacing * 0.8;

                    for i in 0..node.ring_count {
                        let ring_radius = node.radius - (i as f64 * ring_spacing) - ring_spacing * 0.5;
                        if ring_radius > 0.0 {
                            elements.push(format!(
                                r#"<circle cx="{:.1}" cy="{:.1}" r="{:.1}" fill="none" stroke="{}" stroke-width="{:.1}"/>"#,
                                node.x, node.y, ring_radius, fill_color, stroke_width
                            ));
                        }
                    }
                }

                elements
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
        let positions = self.position_nodes_circular(metrics, rng);
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

    /// Position nodes in a grid pattern.
    fn position_nodes_grid(
        &self,
        metrics: &NeuralMetrics,
        rng: &mut impl Rng,
    ) -> Vec<PositionedNode> {
        let graph = &metrics.graph;
        let scale = self.scale();
        let padding = 120.0 * scale;

        let n = graph.nodes.len();
        let cols = (n as f64).sqrt().ceil() as usize;
        let rows = (n + cols - 1) / cols;

        let cell_w = (self.width as f64 - padding * 2.0) / cols as f64;
        let cell_h = (self.height as f64 - padding * 2.0) / rows as f64;

        let depths = Self::calculate_depths(&graph.nodes, &graph.edges);
        let (in_degrees, out_degrees) = Self::calculate_degrees(&graph.nodes, &graph.edges);
        let centralities = graph.betweenness_centrality();
        let clusterings = graph.clustering_coefficients();
        let pageranks = graph.pagerank(0.85, 20);
        let cycle_nodes: std::collections::HashSet<String> =
            graph.find_cycle_nodes().into_iter().collect();

        graph
            .nodes
            .iter()
            .enumerate()
            .map(|(i, node)| {
                let row = i / cols;
                let col = i % cols;

                let base_x = padding + cell_w * (col as f64 + 0.5);
                let base_y = padding + cell_h * (row as f64 + 0.5);

                // Add jitter
                let jitter = 20.0 * scale;
                let x = base_x + rng.gen_range(-jitter..jitter);
                let y = base_y + rng.gen_range(-jitter..jitter);

                let role = Self::node_role(node);
                let depth = *depths.get(&node.name).unwrap_or(&0);
                let in_deg = *in_degrees.get(&node.name).unwrap_or(&0);
                let out_deg = *out_degrees.get(&node.name).unwrap_or(&0);
                let centrality = *centralities.get(&node.name).unwrap_or(&0.0);
                let clustering = *clusterings.get(&node.name).unwrap_or(&0.0);
                let pagerank = *pageranks.get(&node.name).unwrap_or(&0.0);
                let in_cycle = cycle_nodes.contains(&node.name);
                let temporal = metrics
                    .temporal
                    .nodes
                    .get(&node.name)
                    .cloned()
                    .unwrap_or_default();

                let hash = Self::hash_to_float(&node.name);
                let color_index = (hash * 1000.0) as usize;
                let ring_count = if rng.gen_bool(0.5) {
                    1
                } else {
                    rng.gen_range(2..=4)
                };

                let importance = (centrality + pagerank) / 2.0;
                let base_radius = 15.0 * scale;
                let node_radius = base_radius + importance * 20.0 * scale;

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
                    color_index,
                    centrality,
                    clustering,
                    pagerank,
                    in_cycle,
                    temporal,
                    ring_count,
                }
            })
            .collect()
    }

    /// Generate grid style graph.
    fn generate_grid(&self, metrics: &NeuralMetrics, rng: &mut impl Rng) -> String {
        let positions = self.position_nodes_grid(metrics, rng);
        let edges = self.draw_edges(&positions, &metrics.graph.edges, rng);
        let nodes = self.draw_nodes(&positions);
        self.wrap_svg(&format!("{}\n{}", edges.join("\n"), nodes.join("\n")))
    }

    /// Position nodes in a spiral pattern from center outward.
    fn position_nodes_spiral(
        &self,
        metrics: &NeuralMetrics,
        rng: &mut impl Rng,
    ) -> Vec<PositionedNode> {
        let graph = &metrics.graph;
        let scale = self.scale();
        let cx = self.width as f64 / 2.0;
        let cy = self.height as f64 / 2.0;
        let max_radius = (self.width.min(self.height) as f64 / 2.0) - 100.0 * scale;

        let depths = Self::calculate_depths(&graph.nodes, &graph.edges);
        let (in_degrees, out_degrees) = Self::calculate_degrees(&graph.nodes, &graph.edges);
        let centralities = graph.betweenness_centrality();
        let clusterings = graph.clustering_coefficients();
        let pageranks = graph.pagerank(0.85, 20);
        let cycle_nodes: std::collections::HashSet<String> =
            graph.find_cycle_nodes().into_iter().collect();

        let n = graph.nodes.len();
        let turns = 2.5; // Number of spiral rotations

        graph
            .nodes
            .iter()
            .enumerate()
            .map(|(i, node)| {
                let t = i as f64 / n.max(1) as f64;
                let angle = t * turns * 2.0 * PI;
                let r = t * max_radius + 50.0 * scale;

                let base_x = cx + angle.cos() * r;
                let base_y = cy + angle.sin() * r;

                let jitter = 15.0 * scale;
                let x = base_x + rng.gen_range(-jitter..jitter);
                let y = base_y + rng.gen_range(-jitter..jitter);

                let role = Self::node_role(node);
                let depth = *depths.get(&node.name).unwrap_or(&0);
                let in_deg = *in_degrees.get(&node.name).unwrap_or(&0);
                let out_deg = *out_degrees.get(&node.name).unwrap_or(&0);
                let centrality = *centralities.get(&node.name).unwrap_or(&0.0);
                let clustering = *clusterings.get(&node.name).unwrap_or(&0.0);
                let pagerank = *pageranks.get(&node.name).unwrap_or(&0.0);
                let in_cycle = cycle_nodes.contains(&node.name);
                let temporal = metrics
                    .temporal
                    .nodes
                    .get(&node.name)
                    .cloned()
                    .unwrap_or_default();

                let hash = Self::hash_to_float(&node.name);
                let color_index = (hash * 1000.0) as usize;
                let ring_count = if rng.gen_bool(0.5) {
                    1
                } else {
                    rng.gen_range(2..=4)
                };

                let importance = (centrality + pagerank) / 2.0;
                let base_radius = 12.0 * scale;
                let node_radius = base_radius + importance * 18.0 * scale;

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
                    color_index,
                    centrality,
                    clustering,
                    pagerank,
                    in_cycle,
                    temporal,
                    ring_count,
                }
            })
            .collect()
    }

    /// Generate spiral style graph.
    fn generate_spiral(&self, metrics: &NeuralMetrics, rng: &mut impl Rng) -> String {
        let positions = self.position_nodes_spiral(metrics, rng);
        let edges = self.draw_edges(&positions, &metrics.graph.edges, rng);
        let nodes = self.draw_nodes(&positions);
        self.wrap_svg(&format!("{}\n{}", edges.join("\n"), nodes.join("\n")))
    }

    /// Position nodes in concentric rings based on depth.
    fn position_nodes_radial(
        &self,
        metrics: &NeuralMetrics,
        rng: &mut impl Rng,
    ) -> Vec<PositionedNode> {
        let graph = &metrics.graph;
        let scale = self.scale();
        let cx = self.width as f64 / 2.0;
        let cy = self.height as f64 / 2.0;
        let max_radius = (self.width.min(self.height) as f64 / 2.0) - 100.0 * scale;

        let depths = Self::calculate_depths(&graph.nodes, &graph.edges);
        let (in_degrees, out_degrees) = Self::calculate_degrees(&graph.nodes, &graph.edges);
        let centralities = graph.betweenness_centrality();
        let clusterings = graph.clustering_coefficients();
        let pageranks = graph.pagerank(0.85, 20);
        let cycle_nodes: std::collections::HashSet<String> =
            graph.find_cycle_nodes().into_iter().collect();

        let max_depth = depths.values().copied().max().unwrap_or(0).max(1);

        // Group nodes by depth
        let mut layers: Vec<Vec<&GraphNode>> = vec![vec![]; (max_depth + 1) as usize];
        for node in &graph.nodes {
            let depth = *depths.get(&node.name).unwrap_or(&0) as usize;
            if depth < layers.len() {
                layers[depth].push(node);
            }
        }

        let mut positions = Vec::new();
        let ring_spacing = max_radius / (max_depth + 1) as f64;

        for (depth_idx, layer) in layers.iter().enumerate() {
            let ring_radius = ring_spacing * (depth_idx as f64 + 1.0);
            let n = layer.len();

            for (i, node) in layer.iter().enumerate() {
                let angle = (i as f64 / n.max(1) as f64) * 2.0 * PI + rng.gen_range(-0.1..0.1);
                let r = ring_radius + rng.gen_range(-15.0..15.0) * scale;

                let x = cx + angle.cos() * r;
                let y = cy + angle.sin() * r;

                let role = Self::node_role(node);
                let depth = *depths.get(&node.name).unwrap_or(&0);
                let in_deg = *in_degrees.get(&node.name).unwrap_or(&0);
                let out_deg = *out_degrees.get(&node.name).unwrap_or(&0);
                let centrality = *centralities.get(&node.name).unwrap_or(&0.0);
                let clustering = *clusterings.get(&node.name).unwrap_or(&0.0);
                let pagerank = *pageranks.get(&node.name).unwrap_or(&0.0);
                let in_cycle = cycle_nodes.contains(&node.name);
                let temporal = metrics
                    .temporal
                    .nodes
                    .get(&node.name)
                    .cloned()
                    .unwrap_or_default();

                let hash = Self::hash_to_float(&node.name);
                let color_index = (hash * 1000.0) as usize;
                let ring_count = if rng.gen_bool(0.5) {
                    1
                } else {
                    rng.gen_range(2..=4)
                };

                let importance = (centrality + pagerank) / 2.0;
                let base_radius = 12.0 * scale;
                let node_radius = base_radius + importance * 20.0 * scale;

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
                    color_index,
                    centrality,
                    clustering,
                    pagerank,
                    in_cycle,
                    temporal,
                    ring_count,
                });
            }
        }

        positions
    }

    /// Generate radial style graph.
    fn generate_radial(&self, metrics: &NeuralMetrics, rng: &mut impl Rng) -> String {
        let positions = self.position_nodes_radial(metrics, rng);
        let edges = self.draw_edges(&positions, &metrics.graph.edges, rng);
        let nodes = self.draw_nodes(&positions);
        self.wrap_svg(&format!("{}\n{}", edges.join("\n"), nodes.join("\n")))
    }

    /// Position nodes in tight clusters with long inter-cluster edges.
    fn position_nodes_clustered(
        &self,
        metrics: &NeuralMetrics,
        rng: &mut impl Rng,
    ) -> Vec<PositionedNode> {
        let graph = &metrics.graph;
        let scale = self.scale();
        let padding = 150.0 * scale;

        let depths = Self::calculate_depths(&graph.nodes, &graph.edges);
        let (in_degrees, out_degrees) = Self::calculate_degrees(&graph.nodes, &graph.edges);
        let centralities = graph.betweenness_centrality();
        let clusterings = graph.clustering_coefficients();
        let pageranks = graph.pagerank(0.85, 20);
        let cycle_nodes: std::collections::HashSet<String> =
            graph.find_cycle_nodes().into_iter().collect();

        // Create 3-5 cluster centers
        let num_clusters = rng.gen_range(3..=5).min(graph.nodes.len());
        let cluster_centers: Vec<(f64, f64)> = (0..num_clusters)
            .map(|_| {
                (
                    rng.gen_range(padding..(self.width as f64 - padding)),
                    rng.gen_range(padding..(self.height as f64 - padding)),
                )
            })
            .collect();

        graph
            .nodes
            .iter()
            .enumerate()
            .map(|(i, node)| {
                // Assign to a cluster based on index
                let cluster_idx = i % num_clusters;
                let (cx, cy) = cluster_centers[cluster_idx];

                // Tight clustering around center
                let cluster_radius = 80.0 * scale;
                let angle = rng.gen_range(0.0..2.0 * PI);
                let r = rng.gen_range(0.0..cluster_radius);

                let x = cx + angle.cos() * r;
                let y = cy + angle.sin() * r;

                let role = Self::node_role(node);
                let depth = *depths.get(&node.name).unwrap_or(&0);
                let in_deg = *in_degrees.get(&node.name).unwrap_or(&0);
                let out_deg = *out_degrees.get(&node.name).unwrap_or(&0);
                let centrality = *centralities.get(&node.name).unwrap_or(&0.0);
                let clustering = *clusterings.get(&node.name).unwrap_or(&0.0);
                let pagerank = *pageranks.get(&node.name).unwrap_or(&0.0);
                let in_cycle = cycle_nodes.contains(&node.name);
                let temporal = metrics
                    .temporal
                    .nodes
                    .get(&node.name)
                    .cloned()
                    .unwrap_or_default();

                let hash = Self::hash_to_float(&node.name);
                let color_index = (hash * 1000.0) as usize;
                let ring_count = if rng.gen_bool(0.5) {
                    1
                } else {
                    rng.gen_range(2..=4)
                };

                let importance = (centrality + pagerank) / 2.0;
                let base_radius = 10.0 * scale;
                let node_radius = base_radius + importance * 15.0 * scale;

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
                    color_index,
                    centrality,
                    clustering,
                    pagerank,
                    in_cycle,
                    temporal,
                    ring_count,
                }
            })
            .collect()
    }

    /// Generate clustered style graph.
    fn generate_clustered(&self, metrics: &NeuralMetrics, rng: &mut impl Rng) -> String {
        let positions = self.position_nodes_clustered(metrics, rng);
        let edges = self.draw_edges(&positions, &metrics.graph.edges, rng);
        let nodes = self.draw_nodes(&positions);
        self.wrap_svg(&format!("{}\n{}", edges.join("\n"), nodes.join("\n")))
    }

    /// Position nodes randomly scattered across the canvas.
    fn position_nodes_scatter(
        &self,
        metrics: &NeuralMetrics,
        rng: &mut impl Rng,
    ) -> Vec<PositionedNode> {
        let graph = &metrics.graph;
        let scale = self.scale();
        let padding = 100.0 * scale;

        let depths = Self::calculate_depths(&graph.nodes, &graph.edges);
        let (in_degrees, out_degrees) = Self::calculate_degrees(&graph.nodes, &graph.edges);
        let centralities = graph.betweenness_centrality();
        let clusterings = graph.clustering_coefficients();
        let pageranks = graph.pagerank(0.85, 20);
        let cycle_nodes: std::collections::HashSet<String> =
            graph.find_cycle_nodes().into_iter().collect();

        graph
            .nodes
            .iter()
            .map(|node| {
                let x = rng.gen_range(padding..(self.width as f64 - padding));
                let y = rng.gen_range(padding..(self.height as f64 - padding));

                let role = Self::node_role(node);
                let depth = *depths.get(&node.name).unwrap_or(&0);
                let in_deg = *in_degrees.get(&node.name).unwrap_or(&0);
                let out_deg = *out_degrees.get(&node.name).unwrap_or(&0);
                let centrality = *centralities.get(&node.name).unwrap_or(&0.0);
                let clustering = *clusterings.get(&node.name).unwrap_or(&0.0);
                let pagerank = *pageranks.get(&node.name).unwrap_or(&0.0);
                let in_cycle = cycle_nodes.contains(&node.name);
                let temporal = metrics
                    .temporal
                    .nodes
                    .get(&node.name)
                    .cloned()
                    .unwrap_or_default();

                let hash = Self::hash_to_float(&node.name);
                let color_index = (hash * 1000.0) as usize;
                let ring_count = if rng.gen_bool(0.5) {
                    1
                } else {
                    rng.gen_range(2..=4)
                };

                let importance = (centrality + pagerank) / 2.0;
                let base_radius = 12.0 * scale;
                let node_radius = base_radius + importance * 22.0 * scale;

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
                    color_index,
                    centrality,
                    clustering,
                    pagerank,
                    in_cycle,
                    temporal,
                    ring_count,
                }
            })
            .collect()
    }

    /// Generate scatter style graph.
    fn generate_scatter(&self, metrics: &NeuralMetrics, rng: &mut impl Rng) -> String {
        let positions = self.position_nodes_scatter(metrics, rng);
        let edges = self.draw_edges(&positions, &metrics.graph.edges, rng);
        let nodes = self.draw_nodes(&positions);
        self.wrap_svg(&format!("{}\n{}", edges.join("\n"), nodes.join("\n")))
    }

    /// Position nodes on a fixed NxN matrix grid.
    ///
    /// Creates grid positions and randomly assigns nodes to slots.
    /// Empty slots remain empty, creating a sparse matrix appearance.
    fn position_nodes_matrix(
        &self,
        metrics: &NeuralMetrics,
        rng: &mut impl Rng,
    ) -> Vec<PositionedNode> {
        let graph = &metrics.graph;
        let scale = self.scale();

        // Density based on metrics:
        // - High branching_factor (many connections) = spread out for clarity
        // - High synapse_rate (busy) = spread out for clarity
        // - High activity_skew (concentrated) = tighter layout
        let branching_spread = (metrics.branching_factor / 4.0).min(1.0) * 0.4; // 0-0.4
        let rate_spread = (metrics.synapse_rate / 300.0).min(1.0) * 0.3; // 0-0.3
        let skew_tight = metrics.activity_skew * 0.3; // 0-0.3 (subtracts)

        let density_factor = (1.2 + branching_spread + rate_spread - skew_tight).clamp(1.1, 2.0);
        let padding = (60.0 + branching_spread * 60.0) * scale; // 60-84 based on branching

        let n = graph.nodes.len();
        // Grid size varies based on metric-driven density
        let grid_size = ((n as f64 * density_factor).sqrt().ceil() as usize).max(3);

        let cell_w = (self.width as f64 - padding * 2.0) / grid_size as f64;
        let cell_h = (self.height as f64 - padding * 2.0) / grid_size as f64;

        // Generate all possible grid positions
        let mut slots: Vec<(usize, usize)> = (0..grid_size)
            .flat_map(|row| (0..grid_size).map(move |col| (row, col)))
            .collect();

        // Shuffle and take only as many as we have nodes
        use rand::seq::SliceRandom;
        slots.shuffle(rng);
        slots.truncate(n);

        let depths = Self::calculate_depths(&graph.nodes, &graph.edges);
        let (in_degrees, out_degrees) = Self::calculate_degrees(&graph.nodes, &graph.edges);
        let centralities = graph.betweenness_centrality();
        let clusterings = graph.clustering_coefficients();
        let pageranks = graph.pagerank(0.85, 20);
        let cycle_nodes: std::collections::HashSet<String> =
            graph.find_cycle_nodes().into_iter().collect();

        graph
            .nodes
            .iter()
            .enumerate()
            .map(|(i, node)| {
                let (row, col) = slots[i];

                // Fixed grid position (center of cell)
                let x = padding + cell_w * (col as f64 + 0.5);
                let y = padding + cell_h * (row as f64 + 0.5);

                let role = Self::node_role(node);
                let depth = *depths.get(&node.name).unwrap_or(&0);
                let in_deg = *in_degrees.get(&node.name).unwrap_or(&0);
                let out_deg = *out_degrees.get(&node.name).unwrap_or(&0);
                let centrality = *centralities.get(&node.name).unwrap_or(&0.0);
                let clustering = *clusterings.get(&node.name).unwrap_or(&0.0);
                let pagerank = *pageranks.get(&node.name).unwrap_or(&0.0);
                let in_cycle = cycle_nodes.contains(&node.name);
                let temporal = metrics
                    .temporal
                    .nodes
                    .get(&node.name)
                    .cloned()
                    .unwrap_or_default();

                let hash = Self::hash_to_float(&node.name);
                let color_index = (hash * 1000.0) as usize;
                let ring_count = if rng.gen_bool(0.5) {
                    1
                } else {
                    rng.gen_range(2..=4)
                };

                let importance = (centrality + pagerank) / 2.0;
                let base_radius = 18.0 * scale;
                let node_radius = base_radius + importance * 25.0 * scale;

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
                    color_index,
                    centrality,
                    clustering,
                    pagerank,
                    in_cycle,
                    temporal,
                    ring_count,
                }
            })
            .collect()
    }

    /// Draw edges that route around other nodes.
    ///
    /// Uses curved paths that deflect away from obstacles.
    fn draw_edges_avoiding_nodes(
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

                let dx = tgt.x - src.x;
                let dy = tgt.y - src.y;
                let dist = (dx * dx + dy * dy).sqrt();

                if dist < src.radius + tgt.radius + 10.0 * scale {
                    return None;
                }

                // Start/end at node edges
                let angle = dy.atan2(dx);
                let start_x = src.x + angle.cos() * src.radius;
                let start_y = src.y + angle.sin() * src.radius;
                let end_x = tgt.x - angle.cos() * tgt.radius;
                let end_y = tgt.y - angle.sin() * tgt.radius;

                // Check for nodes along the path and calculate deflection
                let mid_x = (start_x + end_x) / 2.0;
                let mid_y = (start_y + end_y) / 2.0;

                // Find nodes that might be in the way (excluding src and tgt)
                let mut max_deflection = 0.0f64;
                let mut deflection_direction = 1.0f64;

                for node in positions {
                    if node.name == src.name || node.name == tgt.name {
                        continue;
                    }

                    // Check if node is near the line between src and tgt
                    let node_dx = node.x - src.x;
                    let node_dy = node.y - src.y;

                    // Project node onto the line
                    let line_len_sq = dx * dx + dy * dy;
                    if line_len_sq < 1.0 {
                        continue;
                    }
                    let t = ((node_dx * dx + node_dy * dy) / line_len_sq).clamp(0.0, 1.0);

                    // Point on line closest to node
                    let closest_x = src.x + t * dx;
                    let closest_y = src.y + t * dy;

                    // Distance from node center to line
                    let dist_to_line = ((node.x - closest_x).powi(2) + (node.y - closest_y).powi(2)).sqrt();

                    // If the line would pass through or near the node, we need to deflect
                    let clearance = node.radius + 30.0 * scale;
                    if dist_to_line < clearance && t > 0.1 && t < 0.9 {
                        // Calculate required deflection
                        let needed_deflection = clearance - dist_to_line + 50.0 * scale;
                        if needed_deflection > max_deflection {
                            max_deflection = needed_deflection;
                            // Deflect perpendicular to the line, away from the obstacle
                            let perp_x = -(dy / dist);
                            let perp_y = dx / dist;
                            let to_node_x = node.x - mid_x;
                            let to_node_y = node.y - mid_y;
                            // Choose direction away from obstacle
                            deflection_direction = if perp_x * to_node_x + perp_y * to_node_y > 0.0 {
                                -1.0
                            } else {
                                1.0
                            };
                        }
                    }
                }

                // Base curve from topic hash
                let topic_hash = Self::hash_to_float(&edge.topic);
                let base_curve = 40.0 * scale;
                let hash_direction = if topic_hash > 0.5 { 1.0 } else { -1.0 };

                // Final curve amount: base curve + obstacle avoidance
                let curve_amount = if max_deflection > 0.0 {
                    max_deflection * deflection_direction
                } else {
                    (base_curve + rng.gen_range(-20.0..20.0) * scale) * hash_direction
                };

                // Control point for bezier curve
                let perp_angle = angle + PI / 2.0;
                let ctrl_x = mid_x + perp_angle.cos() * curve_amount;
                let ctrl_y = mid_y + perp_angle.sin() * curve_amount;

                // Stroke width
                let rate = edge.rate.unwrap_or(10.0);
                let base_width = 2.0 * scale;
                let rate_factor = (rate.log10().max(0.0) / 2.0).min(2.0);
                let stroke_width = base_width + rate_factor * 2.0 * scale;

                let edge_color = self.palette.node_color(src.color_index);

                Some(format!(
                    r#"<path d="M {:.1} {:.1} Q {:.1} {:.1} {:.1} {:.1}" fill="none" stroke="{}" stroke-width="{:.2}" stroke-linecap="round"/>"#,
                    start_x, start_y, ctrl_x, ctrl_y, end_x, end_y, edge_color, stroke_width
                ))
            })
            .collect()
    }

    /// Generate matrix style graph with obstacle-avoiding edges.
    fn generate_matrix(&self, metrics: &NeuralMetrics, rng: &mut impl Rng) -> String {
        let positions = self.position_nodes_matrix(metrics, rng);
        let edges = self.draw_edges_avoiding_nodes(&positions, &metrics.graph.edges, rng);
        let nodes = self.draw_nodes(&positions);
        self.wrap_svg(&format!("{}\n{}", edges.join("\n"), nodes.join("\n")))
    }

    /// Wrap content in a simple SVG with solid background.
    fn wrap_svg(&self, content: &str) -> String {
        format!(
            r#"<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {} {}" width="{}" height="{}">
  <rect width="100%" height="100%" fill="{}"/>
  {}
</svg>"#,
            self.width, self.height, self.width, self.height, self.palette.background, content
        )
    }
}

impl Generator for GraphGenerator {
    fn name(&self) -> &'static str {
        match self.style {
            GraphStyle::Organic => "graph_organic",
            GraphStyle::Circular => "graph_circular",
            GraphStyle::Hierarchical => "graph_hierarchical",
            GraphStyle::Grid => "graph_grid",
            GraphStyle::Spiral => "graph_spiral",
            GraphStyle::Radial => "graph_radial",
            GraphStyle::Clustered => "graph_clustered",
            GraphStyle::Scatter => "graph_scatter",
            GraphStyle::Matrix => "graph_matrix",
        }
    }

    fn generate(&self, metrics: &NeuralMetrics) -> String {
        let seed = metrics.to_seed();
        let seed_u64 = u64::from_le_bytes(seed[0..8].try_into().unwrap());
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed_u64);

        // Downsample the graph if needed - use random count between 9-17
        let metrics = if self.max_nodes.is_some() {
            let target_nodes = rng.gen_range(9..=17);
            if metrics.graph.nodes.len() > target_nodes {
                let downsampled_graph = metrics.graph.downsample(target_nodes, seed_u64);
                NeuralMetrics {
                    graph: downsampled_graph,
                    node_count: target_nodes as u32,
                    connection_count: metrics.connection_count,
                    ..metrics.clone()
                }
            } else {
                metrics.clone()
            }
        } else {
            metrics.clone()
        };

        match self.style {
            GraphStyle::Organic => self.generate_organic(&metrics, &mut rng),
            GraphStyle::Circular => self.generate_circular(&metrics, &mut rng),
            GraphStyle::Hierarchical => self.generate_hierarchical(&metrics, &mut rng),
            GraphStyle::Grid => self.generate_grid(&metrics, &mut rng),
            GraphStyle::Spiral => self.generate_spiral(&metrics, &mut rng),
            GraphStyle::Radial => self.generate_radial(&metrics, &mut rng),
            GraphStyle::Clustered => self.generate_clustered(&metrics, &mut rng),
            GraphStyle::Scatter => self.generate_scatter(&metrics, &mut rng),
            GraphStyle::Matrix => self.generate_matrix(&metrics, &mut rng),
        }
    }

    fn extension(&self) -> &'static str {
        "svg"
    }
}
