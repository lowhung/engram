//! Configuration loading for Engram.
//!
//! Configuration is loaded from TOML files with environment variable overrides.

use anyhow::Result;
use config::{Config, Environment, File};
use serde::Deserialize;
use std::path::Path;

pub const DEFAULT_CONFIG_FILE: &str = "config.default.toml";

#[derive(Debug, Clone, Deserialize, Default)]
pub struct EngramConfig {
    #[serde(default)]
    pub rabbitmq: RabbitmqConfig,

    #[serde(default)]
    pub capture: CaptureConfig,

    #[serde(default)]
    pub filter: FilterConfig,

    #[serde(default)]
    pub output: OutputConfig,

    #[serde(default)]
    pub generator: GeneratorConfig,

    #[serde(default)]
    pub health: HealthConfig,
}

#[derive(Debug, Clone, Deserialize)]
pub struct RabbitmqConfig {
    #[serde(default = "default_rabbitmq_url")]
    pub url: String,

    #[serde(default = "default_rabbitmq_exchange")]
    pub exchange: String,
}

impl Default for RabbitmqConfig {
    fn default() -> Self {
        Self {
            url: default_rabbitmq_url(),
            exchange: default_rabbitmq_exchange(),
        }
    }
}

fn default_rabbitmq_url() -> String {
    "amqp://127.0.0.1:5672/%2f".to_string()
}

fn default_rabbitmq_exchange() -> String {
    "caryatid".to_string()
}

#[derive(Debug, Clone, Deserialize)]
pub struct CaptureConfig {
    #[serde(default = "default_topic")]
    pub topic: String,

    #[serde(default = "default_count")]
    pub count: usize,

    #[serde(default = "default_interval")]
    pub interval: u64,

    #[serde(default = "default_timeout")]
    pub timeout: u64,
}

impl Default for CaptureConfig {
    fn default() -> Self {
        Self {
            topic: default_topic(),
            count: default_count(),
            interval: default_interval(),
            timeout: default_timeout(),
        }
    }
}

fn default_topic() -> String {
    "buswatch.snapshot".to_string()
}

fn default_count() -> usize {
    10
}

fn default_interval() -> u64 {
    1
}

fn default_timeout() -> u64 {
    30
}

#[derive(Debug, Clone, Deserialize)]
pub struct FilterConfig {
    #[serde(default = "default_ignored_topics")]
    pub ignored_topics: Vec<String>,
}

impl Default for FilterConfig {
    fn default() -> Self {
        Self {
            ignored_topics: default_ignored_topics(),
        }
    }
}

fn default_ignored_topics() -> Vec<String> {
    vec!["cardano.query.".to_string()]
}

#[derive(Debug, Clone, Deserialize)]
pub struct OutputConfig {
    #[serde(default = "default_directory")]
    pub directory: String,

    #[serde(default = "default_width")]
    pub width: u32,

    #[serde(default = "default_height")]
    pub height: u32,

    #[serde(default)]
    pub save_metrics: bool,
}

impl Default for OutputConfig {
    fn default() -> Self {
        Self {
            directory: default_directory(),
            width: default_width(),
            height: default_height(),
            save_metrics: false,
        }
    }
}

fn default_directory() -> String {
    "output".to_string()
}

fn default_width() -> u32 {
    512
}

fn default_height() -> u32 {
    512
}

#[derive(Debug, Clone, Deserialize)]
pub struct GeneratorConfig {
    #[serde(default = "default_generator")]
    pub default: String,
}

impl Default for GeneratorConfig {
    fn default() -> Self {
        Self {
            default: default_generator(),
        }
    }
}

fn default_generator() -> String {
    "dendrite".to_string()
}

#[derive(Debug, Clone, Deserialize)]
pub struct HealthConfig {
    #[serde(default = "default_backlog_warning")]
    pub backlog_warning: u64,

    #[serde(default = "default_backlog_critical")]
    pub backlog_critical: u64,

    #[serde(default = "default_pending_warning_ms")]
    pub pending_warning_ms: u64,

    #[serde(default = "default_pending_critical_ms")]
    pub pending_critical_ms: u64,
}

impl Default for HealthConfig {
    fn default() -> Self {
        Self {
            backlog_warning: default_backlog_warning(),
            backlog_critical: default_backlog_critical(),
            pending_warning_ms: default_pending_warning_ms(),
            pending_critical_ms: default_pending_critical_ms(),
        }
    }
}

fn default_backlog_warning() -> u64 {
    100
}

fn default_backlog_critical() -> u64 {
    1000
}

fn default_pending_warning_ms() -> u64 {
    500
}

fn default_pending_critical_ms() -> u64 {
    2000
}

impl EngramConfig {
    pub fn load(path: &Path) -> Result<Self> {
        let config = Config::builder()
            .add_source(File::with_name(DEFAULT_CONFIG_FILE).required(false))
            .add_source(File::from(path).required(false))
            .add_source(Environment::with_prefix("ENGRAM").separator("_"))
            .build()?;

        let engram_config: EngramConfig = config.try_deserialize().unwrap_or_default();
        Ok(engram_config)
    }
}
