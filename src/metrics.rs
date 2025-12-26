//! Neural activity metrics that drive art generation.
//!
//! Converts neuronic graph snapshots into normalized parameters
//! that generators use to create unique visual pieces.

use neuronic::{AggregatedMetrics, GraphSnapshot};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

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
}

impl NeuralMetrics {
    /// Create metrics from a single neuronic snapshot.
    pub fn from_snapshot(snapshot: &GraphSnapshot) -> Self {
        let node_count = snapshot.node_count as u32;
        let connection_count = snapshot.edge_count as u32;

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
        }
    }

    /// Create metrics from aggregated neuronic snapshots.
    ///
    /// This provides richer data from observing the system over time.
    pub fn from_aggregated(agg: &AggregatedMetrics, snapshots: &[GraphSnapshot]) -> Self {
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

        let node_count = rng.gen_range(5..50);
        let connection_count = rng.gen_range(10..200);

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
            node_names: (0..node_count).map(|i| format!("module_{}", i)).collect(),
            topic_names: (0..connection_count.min(20))
                .map(|i| format!("topic_{}", i))
                .collect(),
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
