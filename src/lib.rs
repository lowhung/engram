//! Engram - Generative art from neural network activity metrics.
//!
//! Inspired by neuronic and autoglyphs, Engram creates unique visual
//! pieces by encoding neural activity patterns into minimalist black
//! and white art.

pub mod config;
pub mod generators;
pub mod metrics;

pub use config::EngramConfig;
pub use generators::Generator;
pub use metrics::{NeuralMetrics, NormalizedMetrics};
