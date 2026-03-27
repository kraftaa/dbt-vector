use pyo3::prelude::*;
use pyo3::types::PyModule;
use rand::{rngs::StdRng, Rng, SeedableRng};
use reqwest::blocking::Client;
use reqwest::header::{HeaderMap, HeaderValue, AUTHORIZATION, CONTENT_TYPE};
use serde::Deserialize;
use std::env;
use std::thread::sleep;
use std::time::Duration;
use thiserror::Error;

const OPENAI_EMBED_URL: &str = "https://api.openai.com/v1/embeddings";
const DEFAULT_MODEL: &str = "text-embedding-3-small";
const DEFAULT_MAX_BATCH: usize = 128;
const DEFAULT_RETRIES: usize = 3;
const DEFAULT_TIMEOUT_SECS: u64 = 60;
const BASE_BACKOFF_MS: u64 = 200;
const JITTER_MS: u64 = 100;

#[derive(Debug, Error)]
enum EmbedError {
    #[error("missing OPENAI_API_KEY env var")]
    MissingApiKey,
    #[error("http error: {0}")]
    Http(String),
    #[error("response parse error: {0}")]
    Parse(String),
}

#[derive(Deserialize)]
struct EmbedResponse {
    data: Vec<EmbedItem>,
}

#[derive(Deserialize)]
struct EmbedItem {
    embedding: Vec<f32>,
}

#[derive(Clone, Debug)]
struct Config {
    max_batch: usize,
    retries: usize,
    timeout_secs: u64,
    model: String,
}

fn parse_env_usize(key: &str, default: usize) -> usize {
    env::var(key)
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|v| *v > 0)
        .unwrap_or(default)
}

fn parse_env_u64(key: &str, default: u64) -> u64 {
    env::var(key)
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .filter(|v| *v > 0)
        .unwrap_or(default)
}

fn load_config(model: Option<String>) -> Config {
    Config {
        max_batch: parse_env_usize("EMBED_MAX_BATCH", DEFAULT_MAX_BATCH),
        retries: parse_env_usize("EMBED_RETRIES", DEFAULT_RETRIES),
        timeout_secs: parse_env_u64("EMBED_TIMEOUT_SECS", DEFAULT_TIMEOUT_SECS),
        model: model.unwrap_or_else(|| DEFAULT_MODEL.to_string()),
    }
}

fn http_client(api_key: &str, timeout_secs: u64) -> Result<Client, EmbedError> {
    let mut headers = HeaderMap::new();
    headers.insert(
        AUTHORIZATION,
        HeaderValue::from_str(&format!("Bearer {}", api_key)).map_err(|e| EmbedError::Http(e.to_string()))?,
    );
    headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
    Client::builder()
        .default_headers(headers)
        .timeout(Duration::from_secs(timeout_secs))
        .build()
        .map_err(|e| EmbedError::Http(e.to_string()))
}

fn embed_url() -> String {
    env::var("OPENAI_EMBED_URL").unwrap_or_else(|_| OPENAI_EMBED_URL.to_string())
}

fn send_with_retry(
    body: &serde_json::Value,
    client: &Client,
    retries: usize,
    rng: &mut StdRng,
) -> Result<reqwest::blocking::Response, EmbedError> {
    let mut attempt = 0;
    let url = embed_url();
    loop {
        let resp = client
            .post(&url)
            .header("X-Retry-Attempt", attempt.to_string())
            .json(body)
            .send()
            .map_err(|e| EmbedError::Http(e.to_string()))?;

        if resp.status().is_success() {
            return Ok(resp);
        }

        let status = resp.status();
        if attempt + 1 >= retries || !(status.is_server_error() || status.as_u16() == 429) {
            return Err(EmbedError::Http(format!(
                "status {}: {}",
                status,
                resp.text().unwrap_or_default()
            )));
        }

        let base = BASE_BACKOFF_MS.saturating_mul(2u64.saturating_pow(attempt as u32));
        let jitter = rng.gen_range(0..=JITTER_MS);
        let backoff = Duration::from_millis(base + jitter);
        sleep(backoff);
        attempt += 1;
    }
}

fn embed_batch_internal(
    texts: &[String],
    cfg: &Config,
    client: &Client,
) -> Result<Vec<Vec<f32>>, EmbedError> {
    #[derive(serde::Serialize)]
    struct EmbedRequest<'a> {
        model: &'a str,
        input: &'a [String],
    }

    let mut all_embeddings: Vec<Vec<f32>> = Vec::with_capacity(texts.len());
    let mut rng = StdRng::from_entropy();
    for chunk in texts.chunks(cfg.max_batch) {
        let body = serde_json::to_value(EmbedRequest {
            model: &cfg.model,
            input: chunk,
        })
            .map_err(|e| EmbedError::Parse(e.to_string()))?;
        let resp = send_with_retry(&body, client, cfg.retries, &mut rng)?;
        let parsed: EmbedResponse = resp.json().map_err(|e| EmbedError::Parse(e.to_string()))?;
        all_embeddings.extend(parsed.data.into_iter().map(|item| item.embedding));
    }

    Ok(all_embeddings)
}

impl From<EmbedError> for pyo3::PyErr {
    fn from(e: EmbedError) -> Self {
        pyo3::exceptions::PyRuntimeError::new_err(e.to_string())
    }
}

/// Generate embeddings via OpenAI. Requires OPENAI_API_KEY in the environment.
#[pyfunction]
pub fn embed_batch(texts: Vec<String>, model: Option<String>) -> PyResult<Vec<Vec<f32>>> {
    let api_key = env::var("OPENAI_API_KEY").map_err(|_| EmbedError::MissingApiKey)?;
    let cfg = load_config(model);
    let client = http_client(&api_key, cfg.timeout_secs)?;
    embed_batch_internal(&texts, &cfg, &client).map_err(Into::into)
}

#[pymodule]
fn embedding_engine(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(embed_batch, m)?)?;
    Ok(())
}
