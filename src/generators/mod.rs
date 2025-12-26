//! Art generators - each produces a different visual style from neural metrics.

pub mod glyph;
pub mod stipple;
pub mod svg_lines;

use crate::metrics::NeuralMetrics;

/// Trait for all art generators.
pub trait Generator {
    /// Name of this generator style.
    fn name(&self) -> &'static str;

    /// Generate art from the given metrics.
    /// Returns the output as a string (ASCII, SVG, etc.)
    fn generate(&self, metrics: &NeuralMetrics) -> String;

    /// File extension for this generator's output.
    fn extension(&self) -> &'static str;
}
