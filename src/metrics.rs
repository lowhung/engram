//! Neural activity metrics that drive art generation.
//!
//! These structures mirror concepts from neuronic - synapse rates,
//! node topology, message patterns - and encode them into parameters
//! that generators use to create unique pieces.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// Core metrics captured from neural network activity.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuralMetrics {
    /// Unique identifier for this snapshot
    pub id: u64,
    /// Number of active nodes in the network
    pub node_count: u32,
    /// Number of connections between nodes
    pub connection_count: u32,
    /// Messages per second (synapse rate)
    pub synapse_rate: f64,
    /// Peak activity burst observed
    pub peak_burst: f64,
    /// Activity distribution across nodes (0.0 = uniform, 1.0 = concentrated)
    pub activity_skew: f64,
    /// Depth of the network topology
    pub topology_depth: u32,
    /// Branching factor (average connections per node)
    pub branching_factor: f64,
}

impl NeuralMetrics {
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
        hasher.finalize().into()
    }

    /// Normalize metrics to 0.0-1.0 range for use in generators.
    pub fn normalized(&self) -> NormalizedMetrics {
        NormalizedMetrics {
            density: (self.node_count as f64 / 100.0).min(1.0),
            connectivity: (self.connection_count as f64 / (self.node_count.max(1) * 10) as f64).min(1.0),
            activity: (self.synapse_rate / 1000.0).min(1.0),
            intensity: (self.peak_burst / 100.0).min(1.0),
            skew: self.activity_skew.clamp(0.0, 1.0),
            depth: (self.topology_depth as f64 / 10.0).min(1.0),
            branching: (self.branching_factor / 5.0).min(1.0),
        }
    }

    /// Create sample metrics for testing/demo purposes.
    pub fn sample(id: u64) -> Self {
        use rand::{Rng, SeedableRng};
        let mut rng = rand::rngs::StdRng::seed_from_u64(id);

        Self {
            id,
            node_count: rng.gen_range(5..50),
            connection_count: rng.gen_range(10..200),
            synapse_rate: rng.gen_range(10.0..500.0),
            peak_burst: rng.gen_range(5.0..50.0),
            activity_skew: rng.gen_range(0.0..1.0),
            topology_depth: rng.gen_range(2..8),
            branching_factor: rng.gen_range(1.5..4.0),
        }
    }
}

/// Metrics normalized to 0.0-1.0 for direct use in generation algorithms.
#[derive(Debug, Clone, Copy)]
pub struct NormalizedMetrics {
    /// Node density (0 = sparse, 1 = dense)
    pub density: f64,
    /// How interconnected (0 = isolated, 1 = fully meshed)
    pub connectivity: f64,
    /// Activity level (0 = quiet, 1 = very active)
    pub activity: f64,
    /// Peak intensity (0 = calm, 1 = intense bursts)
    pub intensity: f64,
    /// Activity distribution (0 = uniform, 1 = concentrated)
    pub skew: f64,
    /// Network depth (0 = shallow, 1 = deep)
    pub depth: f64,
    /// Branching factor (0 = linear, 1 = highly branched)
    pub branching: f64,
}
