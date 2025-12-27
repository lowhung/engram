//! Art generators - visualize neural network topology as dendrite-like graphs.

pub mod graph;

use crate::metrics::NeuralMetrics;

/// Trait for all art generators.
pub trait Generator {
    /// Name of this generator style.
    fn name(&self) -> &'static str;

    /// Generate art from the given metrics.
    fn generate(&self, metrics: &NeuralMetrics) -> String;

    /// File extension for this generator's output.
    fn extension(&self) -> &'static str;
}
