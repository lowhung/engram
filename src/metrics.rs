//! Neural activity metrics that drive art generation.
//!
//! Converts neuronic graph snapshots into normalized parameters
//! that generators use to create unique visual pieces.
//!
//! Temporal metrics capture behavioral patterns over time:
//! - Rate variance: how stable/jittery a node's activity is
//! - Burst detection: sudden spikes in activity
//! - Connection stability: how long nodes have been active

use neuronic::{AggregatedMetrics, GraphSnapshot};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{HashMap, HashSet};

/// Health status of a node or edge.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum HealthStatus {
    #[default]
    Healthy,
    Warning,
    Critical,
}

/// A node in the adjacency graph with its topic connections.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphNode {
    /// Module name.
    pub name: String,
    /// Topics this module reads from.
    pub reads: Vec<String>,
    /// Topics this module writes to.
    pub writes: Vec<String>,
    /// Activity rate if available.
    pub rate: Option<f64>,
    /// Total messages processed.
    pub throughput: u64,
    /// Health status of this node.
    pub health: HealthStatus,
}

/// An edge representing a connection between two modules via a topic.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphEdge {
    /// Source module (producer).
    pub source: String,
    /// Target module (consumer).
    pub target: String,
    /// Topic connecting them.
    pub topic: String,
    /// Message rate on this connection.
    pub rate: Option<f64>,
    /// Backlog (unread messages) if available.
    pub backlog: Option<u64>,
    /// Pending time in microseconds if available.
    pub pending_us: Option<u64>,
    /// Health status of this edge.
    pub health: HealthStatus,
}

/// The adjacency graph representing module connections.
///
/// Derived from neuronic snapshots by matching producers to consumers
/// via shared topics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdjacencyGraph {
    /// All nodes (modules) in the graph.
    pub nodes: Vec<GraphNode>,
    /// All edges (topic connections between modules).
    pub edges: Vec<GraphEdge>,
    /// Map from topic name to producers.
    pub producers: HashMap<String, Vec<String>>,
    /// Map from topic name to consumers.
    pub consumers: HashMap<String, Vec<String>>,
}

impl AdjacencyGraph {
    /// Build an adjacency graph from a neuronic snapshot.
    pub fn from_snapshot(snapshot: &GraphSnapshot) -> Self {
        let mut producers: HashMap<String, Vec<String>> = HashMap::new();
        let mut consumers: HashMap<String, Vec<String>> = HashMap::new();

        // Build nodes and collect topic relationships
        let nodes: Vec<GraphNode> = snapshot
            .nodes
            .iter()
            .map(|n| {
                // Register as producer for write topics
                for topic in &n.write_topics {
                    producers
                        .entry(topic.clone())
                        .or_default()
                        .push(n.name.clone());
                }
                // Register as consumer for read topics
                for topic in &n.read_topics {
                    consumers
                        .entry(topic.clone())
                        .or_default()
                        .push(n.name.clone());
                }

                // Convert neuronic HealthStatus to our HealthStatus
                let health = match n.health {
                    neuronic::HealthStatus::Healthy => HealthStatus::Healthy,
                    neuronic::HealthStatus::Warning => HealthStatus::Warning,
                    neuronic::HealthStatus::Critical => HealthStatus::Critical,
                };

                GraphNode {
                    name: n.name.clone(),
                    reads: n.read_topics.clone(),
                    writes: n.write_topics.clone(),
                    rate: n.rate(),
                    throughput: n.throughput(),
                    health,
                }
            })
            .collect();

        // Build edges by matching producers to consumers
        let mut edges = Vec::new();
        for (topic, topic_producers) in &producers {
            if let Some(topic_consumers) = consumers.get(topic) {
                for producer in topic_producers {
                    for consumer in topic_consumers {
                        // Skip self-loops
                        if producer != consumer {
                            // Find the topic edge in the snapshot for rate and health info
                            let edge_info = snapshot.edges.iter().find(|e| &e.topic == topic);

                            let rate = edge_info.and_then(|e| e.rate);
                            let backlog = edge_info.and_then(|e| e.backlog);
                            let pending_us = edge_info.and_then(|e| e.pending_us);
                            let health = edge_info
                                .map(|e| match e.health {
                                    neuronic::HealthStatus::Healthy => HealthStatus::Healthy,
                                    neuronic::HealthStatus::Warning => HealthStatus::Warning,
                                    neuronic::HealthStatus::Critical => HealthStatus::Critical,
                                })
                                .unwrap_or(HealthStatus::Healthy);

                            edges.push(GraphEdge {
                                source: producer.clone(),
                                target: consumer.clone(),
                                topic: topic.clone(),
                                rate,
                                backlog,
                                pending_us,
                                health,
                            });
                        }
                    }
                }
            }
        }

        Self {
            nodes,
            edges,
            producers,
            consumers,
        }
    }

    /// Get all unique topics in the graph.
    pub fn topics(&self) -> Vec<String> {
        let mut topics: HashSet<String> = HashSet::new();
        for node in &self.nodes {
            topics.extend(node.reads.iter().cloned());
            topics.extend(node.writes.iter().cloned());
        }
        let mut result: Vec<String> = topics.into_iter().collect();
        result.sort();
        result
    }

    /// Get modules that a given module sends to (via topics).
    pub fn outgoing(&self, module: &str) -> Vec<&str> {
        self.edges
            .iter()
            .filter(|e| e.source == module)
            .map(|e| e.target.as_str())
            .collect()
    }

    /// Get modules that send to a given module (via topics).
    pub fn incoming(&self, module: &str) -> Vec<&str> {
        self.edges
            .iter()
            .filter(|e| e.target == module)
            .map(|e| e.source.as_str())
            .collect()
    }

    /// Calculate betweenness centrality for all nodes.
    ///
    /// Betweenness centrality measures how often a node lies on shortest paths
    /// between other nodes. High betweenness = important bridge/hub.
    ///
    /// Returns a map from node name to centrality score (0.0 to 1.0 normalized).
    pub fn betweenness_centrality(&self) -> HashMap<String, f64> {
        let node_names: Vec<&str> = self.nodes.iter().map(|n| n.name.as_str()).collect();
        let n = node_names.len();

        if n < 2 {
            return self.nodes.iter().map(|n| (n.name.clone(), 0.0)).collect();
        }

        let mut centrality: HashMap<String, f64> = node_names
            .iter()
            .map(|&name| (name.to_string(), 0.0))
            .collect();

        // Build adjacency list for faster lookups
        let mut adj: HashMap<&str, Vec<&str>> = HashMap::new();
        for name in &node_names {
            adj.insert(name, self.outgoing(name));
        }

        // For each source node, do BFS and count shortest paths
        for &source in &node_names {
            // BFS to find shortest paths from source
            let mut dist: HashMap<&str, i32> = HashMap::new();
            let mut num_paths: HashMap<&str, f64> = HashMap::new();
            let mut predecessors: HashMap<&str, Vec<&str>> = HashMap::new();

            for &name in &node_names {
                dist.insert(name, -1);
                num_paths.insert(name, 0.0);
                predecessors.insert(name, Vec::new());
            }

            dist.insert(source, 0);
            num_paths.insert(source, 1.0);

            let mut queue = std::collections::VecDeque::new();
            let mut stack = Vec::new();
            queue.push_back(source);

            while let Some(v) = queue.pop_front() {
                stack.push(v);
                let v_dist = dist[v];

                for &w in adj.get(v).unwrap_or(&Vec::new()) {
                    // First time seeing w?
                    if dist[w] < 0 {
                        dist.insert(w, v_dist + 1);
                        queue.push_back(w);
                    }
                    // Shortest path to w via v?
                    if dist[w] == v_dist + 1 {
                        *num_paths.get_mut(w).unwrap() += num_paths[v];
                        predecessors.get_mut(w).unwrap().push(v);
                    }
                }
            }

            // Accumulate dependencies
            let mut dependency: HashMap<&str, f64> =
                node_names.iter().map(|&name| (name, 0.0)).collect();

            while let Some(w) = stack.pop() {
                for &v in &predecessors[w] {
                    let fraction = num_paths[v] / num_paths[w];
                    *dependency.get_mut(v).unwrap() += fraction * (1.0 + dependency[w]);
                }
                if w != source {
                    *centrality.get_mut(w).unwrap() += dependency[w];
                }
            }
        }

        // Normalize by (n-1)*(n-2) for directed graphs
        let norm = ((n - 1) * (n - 2)) as f64;
        if norm > 0.0 {
            for val in centrality.values_mut() {
                *val /= norm;
            }
        }

        centrality
    }

    /// Calculate PageRank-style importance scores for all nodes.
    ///
    /// PageRank measures importance based on incoming connections from
    /// other important nodes. Nodes that receive many connections from
    /// highly-ranked nodes get higher scores.
    ///
    /// Returns a map from node name to PageRank score (normalized 0.0 to 1.0).
    pub fn pagerank(&self, damping: f64, iterations: usize) -> HashMap<String, f64> {
        let n = self.nodes.len();
        if n == 0 {
            return HashMap::new();
        }

        // Initialize all nodes with equal rank
        let initial_rank = 1.0 / n as f64;
        let mut ranks: HashMap<String, f64> = self
            .nodes
            .iter()
            .map(|node| (node.name.clone(), initial_rank))
            .collect();

        // Build outgoing edge counts for each node
        let mut out_degrees: HashMap<&str, usize> = HashMap::new();
        for node in &self.nodes {
            out_degrees.insert(&node.name, 0);
        }
        for edge in &self.edges {
            *out_degrees.entry(&edge.source).or_insert(0) += 1;
        }

        // Iterative PageRank calculation
        for _ in 0..iterations {
            let mut new_ranks: HashMap<String, f64> = HashMap::new();

            // Base rank from damping factor (random jump probability)
            let base_rank = (1.0 - damping) / n as f64;

            for node in &self.nodes {
                let mut incoming_rank = 0.0;

                // Sum contributions from all incoming edges
                for edge in &self.edges {
                    if edge.target == node.name {
                        let source_rank = ranks.get(&edge.source).copied().unwrap_or(0.0);
                        let source_out_degree =
                            *out_degrees.get(edge.source.as_str()).unwrap_or(&1);
                        incoming_rank += source_rank / source_out_degree as f64;
                    }
                }

                new_ranks.insert(node.name.clone(), base_rank + damping * incoming_rank);
            }

            ranks = new_ranks;
        }

        // Normalize to 0.0-1.0 range
        let max_rank = ranks.values().cloned().fold(0.0f64, f64::max);
        if max_rank > 0.0 {
            for rank in ranks.values_mut() {
                *rank /= max_rank;
            }
        }

        ranks
    }

    /// Detect cycles in the graph using DFS.
    ///
    /// Returns a list of nodes that participate in at least one cycle.
    /// Nodes in cycles may represent feedback loops in the system.
    pub fn find_cycle_nodes(&self) -> Vec<String> {
        let node_names: Vec<&str> = self.nodes.iter().map(|n| n.name.as_str()).collect();

        // Build adjacency list
        let mut adj: HashMap<&str, Vec<&str>> = HashMap::new();
        for name in &node_names {
            adj.insert(name, Vec::new());
        }
        for edge in &self.edges {
            adj.entry(edge.source.as_str())
                .or_default()
                .push(edge.target.as_str());
        }

        let mut cycle_nodes: std::collections::HashSet<String> = std::collections::HashSet::new();

        // For each node, do DFS to find if it can reach itself
        for &start in &node_names {
            let mut visited: HashMap<&str, bool> = HashMap::new();
            let mut rec_stack: HashMap<&str, bool> = HashMap::new();
            let mut path: Vec<&str> = Vec::new();

            self.dfs_find_cycles(
                start,
                &adj,
                &mut visited,
                &mut rec_stack,
                &mut path,
                &mut cycle_nodes,
            );
        }

        cycle_nodes.into_iter().collect()
    }

    /// Helper function for cycle detection DFS.
    fn dfs_find_cycles<'a>(
        &self,
        node: &'a str,
        adj: &HashMap<&'a str, Vec<&'a str>>,
        visited: &mut HashMap<&'a str, bool>,
        rec_stack: &mut HashMap<&'a str, bool>,
        path: &mut Vec<&'a str>,
        cycle_nodes: &mut std::collections::HashSet<String>,
    ) {
        visited.insert(node, true);
        rec_stack.insert(node, true);
        path.push(node);

        if let Some(neighbors) = adj.get(node) {
            for &neighbor in neighbors {
                if !visited.get(neighbor).copied().unwrap_or(false) {
                    self.dfs_find_cycles(neighbor, adj, visited, rec_stack, path, cycle_nodes);
                } else if rec_stack.get(neighbor).copied().unwrap_or(false) {
                    // Found a cycle - mark all nodes from neighbor to current as cycle nodes
                    let mut in_cycle = false;
                    for &p in path.iter() {
                        if p == neighbor {
                            in_cycle = true;
                        }
                        if in_cycle {
                            cycle_nodes.insert(p.to_string());
                        }
                    }
                }
            }
        }

        path.pop();
        rec_stack.insert(node, false);
    }

    /// Calculate clustering coefficient for all nodes.
    ///
    /// Clustering coefficient measures how connected a node's neighbors are
    /// to each other. High clustering = tight-knit local community.
    ///
    /// Returns a map from node name to coefficient (0.0 to 1.0).
    pub fn clustering_coefficients(&self) -> HashMap<String, f64> {
        let mut coefficients: HashMap<String, f64> = HashMap::new();

        // Build bidirectional neighbor sets (treat as undirected for clustering)
        let mut neighbors: HashMap<&str, HashSet<&str>> = HashMap::new();
        for node in &self.nodes {
            neighbors.insert(node.name.as_str(), HashSet::new());
        }
        for edge in &self.edges {
            neighbors
                .get_mut(edge.source.as_str())
                .map(|s| s.insert(edge.target.as_str()));
            neighbors
                .get_mut(edge.target.as_str())
                .map(|s| s.insert(edge.source.as_str()));
        }

        for node in &self.nodes {
            let node_neighbors = &neighbors[node.name.as_str()];
            let k = node_neighbors.len();

            if k < 2 {
                coefficients.insert(node.name.clone(), 0.0);
                continue;
            }

            // Count edges between neighbors
            let mut neighbor_edges = 0;
            let neighbor_vec: Vec<&&str> = node_neighbors.iter().collect();
            for i in 0..neighbor_vec.len() {
                for j in (i + 1)..neighbor_vec.len() {
                    if neighbors[*neighbor_vec[i]].contains(*neighbor_vec[j]) {
                        neighbor_edges += 1;
                    }
                }
            }

            // Clustering coefficient = 2 * edges / (k * (k-1))
            let max_edges = k * (k - 1) / 2;
            let coeff = if max_edges > 0 {
                neighbor_edges as f64 / max_edges as f64
            } else {
                0.0
            };

            coefficients.insert(node.name.clone(), coeff);
        }

        coefficients
    }

    /// Generate sample adjacency graph for testing.
    pub fn sample(seed: u64) -> Self {
        use rand::{Rng, SeedableRng};
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

        let node_count = rng.gen_range(5..15);
        let topic_count = rng.gen_range(3..10);

        // Generate topic names
        let topics: Vec<String> = (0..topic_count).map(|i| format!("topic.{}", i)).collect();

        // Generate nodes with random topic subscriptions
        let mut producers: HashMap<String, Vec<String>> = HashMap::new();
        let mut consumers: HashMap<String, Vec<String>> = HashMap::new();

        let nodes: Vec<GraphNode> = (0..node_count)
            .map(|i| {
                let name = format!("module_{}", i);

                // Each node writes to 1-3 topics
                let write_count = rng.gen_range(1..=3.min(topic_count));
                let writes: Vec<String> = (0..write_count)
                    .map(|_| topics[rng.gen_range(0..topic_count)].clone())
                    .collect::<HashSet<_>>()
                    .into_iter()
                    .collect();

                // Each node reads from 1-3 topics
                let read_count = rng.gen_range(1..=3.min(topic_count));
                let reads: Vec<String> = (0..read_count)
                    .map(|_| topics[rng.gen_range(0..topic_count)].clone())
                    .collect::<HashSet<_>>()
                    .into_iter()
                    .collect();

                // Register relationships
                for topic in &writes {
                    producers
                        .entry(topic.clone())
                        .or_default()
                        .push(name.clone());
                }
                for topic in &reads {
                    consumers
                        .entry(topic.clone())
                        .or_default()
                        .push(name.clone());
                }

                // Random health status (mostly healthy)
                let health = match rng.gen_range(0..10) {
                    0 => HealthStatus::Critical,
                    1..=2 => HealthStatus::Warning,
                    _ => HealthStatus::Healthy,
                };

                GraphNode {
                    name,
                    reads,
                    writes,
                    rate: Some(rng.gen_range(10.0..500.0)),
                    throughput: rng.gen_range(1000..100000),
                    health,
                }
            })
            .collect();

        // Build edges
        let mut edges = Vec::new();
        for (topic, topic_producers) in &producers {
            if let Some(topic_consumers) = consumers.get(topic) {
                for producer in topic_producers {
                    for consumer in topic_consumers {
                        if producer != consumer {
                            // Random health and backlog for edges
                            let health = match rng.gen_range(0..10) {
                                0 => HealthStatus::Critical,
                                1..=2 => HealthStatus::Warning,
                                _ => HealthStatus::Healthy,
                            };
                            let backlog = if rng.gen_bool(0.3) {
                                Some(rng.gen_range(0..500))
                            } else {
                                None
                            };
                            let pending_us = if rng.gen_bool(0.3) {
                                Some(rng.gen_range(0..1_000_000))
                            } else {
                                None
                            };

                            edges.push(GraphEdge {
                                source: producer.clone(),
                                target: consumer.clone(),
                                topic: topic.clone(),
                                rate: Some(rng.gen_range(10.0..200.0)),
                                backlog,
                                pending_us,
                                health,
                            });
                        }
                    }
                }
            }
        }

        Self {
            nodes,
            edges,
            producers,
            consumers,
        }
    }
}

/// Temporal metrics for a single node across multiple snapshots.
///
/// Captures behavioral patterns over time to inform visual properties.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct NodeTemporalMetrics {
    /// Coefficient of variation in rate (std_dev / mean). Higher = more jittery.
    /// Range: 0.0 (perfectly stable) to 1.0+ (highly variable)
    pub rate_variance: f64,

    /// Maximum rate spike relative to mean. Higher = more bursty.
    /// Range: 1.0 (no bursts) to 10.0+ (extreme bursts)
    pub burst_intensity: f64,

    /// Whether a burst was detected in recent snapshots.
    pub has_recent_burst: bool,

    /// Fraction of snapshots where this node was present.
    /// Range: 0.0 (appeared once) to 1.0 (present in all snapshots)
    pub presence_ratio: f64,

    /// Whether this node was present in the first snapshot (long-lived).
    pub is_original: bool,

    /// Whether this node appeared in later snapshots (newcomer).
    pub is_newcomer: bool,

    /// Average rate across all snapshots.
    pub avg_rate: f64,

    /// Trend direction: positive = increasing activity, negative = decreasing.
    /// Range: -1.0 to 1.0
    pub trend: f64,
}

/// Temporal metrics for a connection (edge) across multiple snapshots.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct EdgeTemporalMetrics {
    /// Coefficient of variation in rate. Higher = more variable.
    pub rate_variance: f64,

    /// Fraction of snapshots where this edge was active.
    pub stability: f64,

    /// Whether this connection existed from the start.
    pub is_original: bool,
}

/// Collection of temporal metrics for the entire graph.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct TemporalMetrics {
    /// Per-node temporal metrics.
    pub nodes: HashMap<String, NodeTemporalMetrics>,

    /// Per-edge temporal metrics (key: "source->target").
    pub edges: HashMap<String, EdgeTemporalMetrics>,

    /// Number of snapshots used to calculate these metrics.
    pub snapshot_count: usize,

    /// Time span covered in milliseconds.
    pub time_span_ms: u64,
}

impl TemporalMetrics {
    /// Calculate temporal metrics from a series of snapshots.
    pub fn from_snapshots(snapshots: &[GraphSnapshot]) -> Self {
        if snapshots.is_empty() {
            return Self::default();
        }

        let snapshot_count = snapshots.len();
        let time_span_ms = if snapshot_count > 1 {
            snapshots.last().unwrap().timestamp_ms - snapshots.first().unwrap().timestamp_ms
        } else {
            0
        };

        // Collect per-node rate time series
        let mut node_rates: HashMap<String, Vec<Option<f64>>> = HashMap::new();
        let mut node_first_seen: HashMap<String, usize> = HashMap::new();

        for (i, snapshot) in snapshots.iter().enumerate() {
            // Track which nodes are in this snapshot
            let nodes_in_snapshot: HashSet<String> =
                snapshot.nodes.iter().map(|n| n.name.clone()).collect();

            for node in &snapshot.nodes {
                let series = node_rates.entry(node.name.clone()).or_default();

                // Pad with None for snapshots before this node appeared
                while series.len() < i {
                    series.push(None);
                }
                series.push(node.rate());

                node_first_seen.entry(node.name.clone()).or_insert(i);
            }

            // For nodes not in this snapshot, record None
            for (name, series) in node_rates.iter_mut() {
                if !nodes_in_snapshot.contains(name) && series.len() == i {
                    series.push(None);
                }
            }
        }

        // Calculate per-node temporal metrics
        let mut nodes: HashMap<String, NodeTemporalMetrics> = HashMap::new();

        for (name, rates) in &node_rates {
            let valid_rates: Vec<f64> = rates.iter().filter_map(|r| *r).collect();
            let presence_count = valid_rates.len();

            if presence_count == 0 {
                nodes.insert(name.clone(), NodeTemporalMetrics::default());
                continue;
            }

            let avg_rate = valid_rates.iter().sum::<f64>() / presence_count as f64;
            let max_rate = valid_rates.iter().cloned().fold(0.0f64, f64::max);

            // Calculate variance
            let variance = if presence_count > 1 && avg_rate > 0.0 {
                let sum_sq: f64 = valid_rates.iter().map(|r| (r - avg_rate).powi(2)).sum();
                let std_dev = (sum_sq / presence_count as f64).sqrt();
                (std_dev / avg_rate).min(2.0) // Cap at 2.0
            } else {
                0.0
            };

            // Burst detection: max > 2x average
            let burst_intensity = if avg_rate > 0.0 {
                max_rate / avg_rate
            } else {
                1.0
            };
            let has_recent_burst = if valid_rates.len() >= 2 {
                // Check if last rate is significantly above average
                let last_rate = valid_rates.last().unwrap_or(&0.0);
                *last_rate > avg_rate * 1.5
            } else {
                false
            };

            // Trend calculation: simple linear regression slope sign
            let trend = if valid_rates.len() >= 3 {
                let n = valid_rates.len() as f64;
                let sum_x: f64 = (0..valid_rates.len()).map(|i| i as f64).sum();
                let sum_y: f64 = valid_rates.iter().sum();
                let sum_xy: f64 = valid_rates
                    .iter()
                    .enumerate()
                    .map(|(i, r)| i as f64 * r)
                    .sum();
                let sum_xx: f64 = (0..valid_rates.len()).map(|i| (i * i) as f64).sum();

                let slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x);
                // Normalize slope to -1..1 range based on average rate
                if avg_rate > 0.0 {
                    (slope / avg_rate).clamp(-1.0, 1.0)
                } else {
                    0.0
                }
            } else {
                0.0
            };

            let first_seen = *node_first_seen.get(name).unwrap_or(&0);

            nodes.insert(
                name.clone(),
                NodeTemporalMetrics {
                    rate_variance: variance,
                    burst_intensity,
                    has_recent_burst,
                    presence_ratio: presence_count as f64 / snapshot_count as f64,
                    is_original: first_seen == 0,
                    is_newcomer: first_seen > 0 && first_seen >= snapshot_count / 2,
                    avg_rate,
                    trend,
                },
            );
        }

        // Calculate per-edge temporal metrics by building adjacency graphs for each snapshot
        // and tracking which edges appear over time
        let mut edge_presence: HashMap<String, Vec<bool>> = HashMap::new();
        let mut edge_rates: HashMap<String, Vec<Option<f64>>> = HashMap::new();

        for (i, snapshot) in snapshots.iter().enumerate() {
            // Build adjacency graph to get actual source->target edges
            let adj_graph = AdjacencyGraph::from_snapshot(snapshot);

            let edges_in_snapshot: HashSet<String> = adj_graph
                .edges
                .iter()
                .map(|e| format!("{}:{}", e.source, e.target))
                .collect();

            for edge in &adj_graph.edges {
                let key = format!("{}:{}", edge.source, edge.target);

                let presence = edge_presence.entry(key.clone()).or_default();
                while presence.len() < i {
                    presence.push(false);
                }
                presence.push(true);

                let rates = edge_rates.entry(key).or_default();
                while rates.len() < i {
                    rates.push(None);
                }
                rates.push(edge.rate);
            }

            // Mark missing edges
            for (key, presence) in edge_presence.iter_mut() {
                if !edges_in_snapshot.contains(key) && presence.len() == i {
                    presence.push(false);
                }
            }
        }

        let mut edges: HashMap<String, EdgeTemporalMetrics> = HashMap::new();

        for (key, presence) in &edge_presence {
            let active_count = presence.iter().filter(|&&p| p).count();
            let stability = active_count as f64 / snapshot_count as f64;
            let is_original = presence.first().copied().unwrap_or(false);

            let rate_variance = if let Some(rates) = edge_rates.get(key) {
                let valid_rates: Vec<f64> = rates.iter().filter_map(|r| *r).collect();
                if valid_rates.len() > 1 {
                    let avg = valid_rates.iter().sum::<f64>() / valid_rates.len() as f64;
                    if avg > 0.0 {
                        let sum_sq: f64 = valid_rates.iter().map(|r| (r - avg).powi(2)).sum();
                        let std_dev = (sum_sq / valid_rates.len() as f64).sqrt();
                        (std_dev / avg).min(2.0)
                    } else {
                        0.0
                    }
                } else {
                    0.0
                }
            } else {
                0.0
            };

            edges.insert(
                key.clone(),
                EdgeTemporalMetrics {
                    rate_variance,
                    stability,
                    is_original,
                },
            );
        }

        Self {
            nodes,
            edges,
            snapshot_count,
            time_span_ms,
        }
    }

    /// Generate sample temporal metrics for testing.
    pub fn sample(seed: u64, node_names: &[String]) -> Self {
        use rand::{Rng, SeedableRng};
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

        let mut nodes = HashMap::new();
        for name in node_names {
            nodes.insert(
                name.clone(),
                NodeTemporalMetrics {
                    rate_variance: rng.gen_range(0.0..1.5),
                    burst_intensity: rng.gen_range(1.0..5.0),
                    has_recent_burst: rng.gen_bool(0.2),
                    presence_ratio: rng.gen_range(0.5..1.0),
                    is_original: rng.gen_bool(0.7),
                    is_newcomer: rng.gen_bool(0.2),
                    avg_rate: rng.gen_range(10.0..200.0),
                    trend: rng.gen_range(-0.5..0.5),
                },
            );
        }

        Self {
            nodes,
            edges: HashMap::new(),
            snapshot_count: 5,
            time_span_ms: 30000,
        }
    }
}

/// Core metrics captured from neural network activity.
///
/// This can be constructed from live neuronic snapshots or
/// generated randomly for testing/demo purposes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuralMetrics {
    /// Unique identifier for this snapshot (typically timestamp).
    pub id: u64,
    /// Number of active nodes in the network.
    pub node_count: u32,
    /// Number of connections between nodes.
    pub connection_count: u32,
    /// Messages per second (synapse rate).
    pub synapse_rate: f64,
    /// Peak activity burst observed.
    pub peak_burst: f64,
    /// Activity distribution across nodes (0.0 = uniform, 1.0 = concentrated).
    pub activity_skew: f64,
    /// Estimated network depth based on topology.
    pub topology_depth: u32,
    /// Branching factor (average connections per node).
    pub branching_factor: f64,
    /// Health distribution: fraction healthy.
    pub health_ratio: f64,
    /// Health distribution: fraction warning.
    pub warning_ratio: f64,
    /// Health distribution: fraction critical.
    pub critical_ratio: f64,
    /// Total message throughput observed.
    pub total_throughput: u64,
    /// Names of nodes (for deterministic hashing).
    pub node_names: Vec<String>,
    /// Names of topics (for deterministic hashing).
    pub topic_names: Vec<String>,
    /// The adjacency graph with full topology information.
    pub graph: AdjacencyGraph,
    /// Temporal metrics from multi-snapshot capture (behavioral patterns over time).
    pub temporal: TemporalMetrics,
}

impl NeuralMetrics {
    /// Create metrics from a single neuronic snapshot.
    pub fn from_snapshot(snapshot: &GraphSnapshot) -> Self {
        let node_count = snapshot.node_count as u32;
        let connection_count = snapshot.edge_count as u32;

        // Build the adjacency graph
        let graph = AdjacencyGraph::from_snapshot(snapshot);

        // Calculate activity skew from rate distribution
        let rates: Vec<f64> = snapshot.nodes.iter().filter_map(|n| n.rate()).collect();
        let activity_skew = if rates.len() > 1 {
            let mean = rates.iter().sum::<f64>() / rates.len() as f64;
            let variance =
                rates.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / rates.len() as f64;
            let std_dev = variance.sqrt();
            // Coefficient of variation as skew measure
            if mean > 0.0 {
                (std_dev / mean).min(1.0)
            } else {
                0.0
            }
        } else {
            0.0
        };

        // Estimate depth from topic chains (simplified)
        let topology_depth = (snapshot.edge_count as f64 / snapshot.node_count.max(1) as f64)
            .sqrt()
            .ceil() as u32;

        let branching_factor = if node_count > 0 {
            connection_count as f64 / node_count as f64
        } else {
            0.0
        };

        let total = snapshot.node_count.max(1);
        let health_ratio = snapshot.healthy_count as f64 / total as f64;
        let warning_ratio = snapshot.warning_count as f64 / total as f64;
        let critical_ratio = snapshot.critical_count as f64 / total as f64;

        Self {
            id: snapshot.timestamp_ms,
            node_count,
            connection_count,
            synapse_rate: snapshot.total_rate.unwrap_or(0.0),
            peak_burst: snapshot.total_rate.unwrap_or(0.0), // Single snapshot, same as rate
            activity_skew,
            topology_depth,
            branching_factor,
            health_ratio,
            warning_ratio,
            critical_ratio,
            total_throughput: snapshot.total_reads + snapshot.total_writes,
            node_names: snapshot.nodes.iter().map(|n| n.name.clone()).collect(),
            topic_names: snapshot.edges.iter().map(|e| e.topic.clone()).collect(),
            graph,
            // Single snapshot has no temporal data
            temporal: TemporalMetrics::default(),
        }
    }

    /// Create metrics from aggregated neuronic snapshots.
    ///
    /// This provides richer data from observing the system over time.
    pub fn from_aggregated(agg: &AggregatedMetrics, snapshots: &[GraphSnapshot]) -> Self {
        // Use last snapshot for the graph (most complete picture)
        let graph = if let Some(last) = snapshots.last() {
            AdjacencyGraph::from_snapshot(last)
        } else {
            AdjacencyGraph::sample(0)
        };

        // Calculate temporal metrics from all snapshots
        let temporal = TemporalMetrics::from_snapshots(snapshots);

        // Use first snapshot for base data, augment with aggregated stats
        let base = if !snapshots.is_empty() {
            Self::from_snapshot(&snapshots[0])
        } else {
            Self::sample(0)
        };

        // Calculate peak burst from all snapshots
        let peak_burst = snapshots
            .iter()
            .filter_map(|s| s.total_rate)
            .fold(0.0f64, |acc, r| acc.max(r));

        // Calculate activity skew across time
        let rates: Vec<f64> = snapshots.iter().filter_map(|s| s.total_rate).collect();
        let activity_skew = if rates.len() > 1 {
            let mean = rates.iter().sum::<f64>() / rates.len() as f64;
            let variance =
                rates.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / rates.len() as f64;
            let std_dev = variance.sqrt();
            if mean > 0.0 {
                (std_dev / mean).min(1.0)
            } else {
                0.0
            }
        } else {
            0.0
        };

        Self {
            id: agg.time_span_ms,
            node_count: agg.avg_node_count.ceil() as u32,
            connection_count: agg.avg_edge_count.ceil() as u32,
            synapse_rate: agg.avg_rate.unwrap_or(0.0),
            peak_burst,
            activity_skew,
            topology_depth: base.topology_depth,
            branching_factor: if agg.avg_node_count > 0.0 {
                agg.avg_edge_count / agg.avg_node_count
            } else {
                0.0
            },
            health_ratio: base.health_ratio,
            warning_ratio: base.warning_ratio,
            critical_ratio: base.critical_ratio,
            total_throughput: agg.message_delta,
            node_names: agg.node_names.clone(),
            topic_names: agg.topic_names.clone(),
            graph,
            temporal,
        }
    }

    /// Generate a deterministic seed from these metrics.
    /// Same metrics always produce the same art.
    pub fn to_seed(&self) -> [u8; 32] {
        let mut hasher = Sha256::new();
        hasher.update(self.id.to_le_bytes());
        hasher.update(self.node_count.to_le_bytes());
        hasher.update(self.connection_count.to_le_bytes());
        hasher.update(self.synapse_rate.to_le_bytes());
        hasher.update(self.peak_burst.to_le_bytes());
        hasher.update(self.activity_skew.to_le_bytes());
        hasher.update(self.topology_depth.to_le_bytes());
        hasher.update(self.branching_factor.to_le_bytes());
        hasher.update(self.total_throughput.to_le_bytes());
        // Include node names for more uniqueness
        for name in &self.node_names {
            hasher.update(name.as_bytes());
        }
        for topic in &self.topic_names {
            hasher.update(topic.as_bytes());
        }
        hasher.finalize().into()
    }

    /// Normalize metrics to 0.0-1.0 range for use in generators.
    pub fn normalized(&self) -> NormalizedMetrics {
        NormalizedMetrics {
            density: (self.node_count as f64 / 50.0).min(1.0),
            connectivity: (self.connection_count as f64 / (self.node_count.max(1) * 5) as f64)
                .min(1.0),
            activity: (self.synapse_rate / 1000.0).min(1.0),
            intensity: (self.peak_burst / 500.0).min(1.0),
            skew: self.activity_skew.clamp(0.0, 1.0),
            depth: (self.topology_depth as f64 / 10.0).min(1.0),
            branching: (self.branching_factor / 5.0).min(1.0),
            health: self.health_ratio,
            warning: self.warning_ratio,
            critical: self.critical_ratio,
            throughput: (self.total_throughput as f64 / 1_000_000.0).min(1.0),
        }
    }

    /// Create sample metrics for testing/demo purposes.
    pub fn sample(id: u64) -> Self {
        use rand::{Rng, SeedableRng};
        let mut rng = rand::rngs::StdRng::seed_from_u64(id);

        // Generate adjacency graph first
        let graph = AdjacencyGraph::sample(id);

        let node_count = graph.nodes.len() as u32;
        let connection_count = graph.edges.len() as u32;
        let node_names: Vec<String> = graph.nodes.iter().map(|n| n.name.clone()).collect();

        // Generate sample temporal metrics
        let temporal = TemporalMetrics::sample(id, &node_names);

        Self {
            id,
            node_count,
            connection_count,
            synapse_rate: rng.gen_range(10.0..500.0),
            peak_burst: rng.gen_range(5.0..100.0),
            activity_skew: rng.gen_range(0.0..1.0),
            topology_depth: rng.gen_range(2..8),
            branching_factor: rng.gen_range(1.5..4.0),
            health_ratio: rng.gen_range(0.5..1.0),
            warning_ratio: rng.gen_range(0.0..0.3),
            critical_ratio: rng.gen_range(0.0..0.1),
            total_throughput: rng.gen_range(1000..1_000_000),
            node_names,
            topic_names: graph.topics(),
            graph,
            temporal,
        }
    }
}

/// Metrics normalized to 0.0-1.0 for direct use in generation algorithms.
#[derive(Debug, Clone, Copy)]
pub struct NormalizedMetrics {
    /// Node density (0 = sparse, 1 = dense).
    pub density: f64,
    /// How interconnected (0 = isolated, 1 = fully meshed).
    pub connectivity: f64,
    /// Activity level (0 = quiet, 1 = very active).
    pub activity: f64,
    /// Peak intensity (0 = calm, 1 = intense bursts).
    pub intensity: f64,
    /// Activity distribution (0 = uniform, 1 = concentrated).
    pub skew: f64,
    /// Network depth (0 = shallow, 1 = deep).
    pub depth: f64,
    /// Branching factor (0 = linear, 1 = highly branched).
    pub branching: f64,
    /// Fraction of healthy nodes.
    pub health: f64,
    /// Fraction of warning nodes.
    pub warning: f64,
    /// Fraction of critical nodes.
    pub critical: f64,
    /// Normalized throughput.
    pub throughput: f64,
}
