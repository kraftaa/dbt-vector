use pyo3::prelude::*;
use pyo3::types::PyModule;
use reqwest::blocking::Client;
use reqwest::header::{HeaderMap, HeaderValue, AUTHORIZATION, CONTENT_TYPE};
use serde::Deserialize;
use std::env;
use std::thread::sleep;
use std::time::Duration;
use thiserror::Error;

const OPENAI_EMBED_URL: &str = "https://api.openai.com/v1/embeddings";
const DEFAULT_MODEL: &str = "text-embedding-3-small";
const MAX_BATCH: usize = 128;
const RETRIES: usize = 3;
const TIMEOUT_SECS: u64 = 60;

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

fn http_client(api_key: &str) -> Result<Client, EmbedError> {
    let mut headers = HeaderMap::new();
    headers.insert(
        AUTHORIZATION,
        HeaderValue::from_str(&format!("Bearer {}", api_key)).map_err(|e| EmbedError::Http(e.to_string()))?,
    );
    headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
    Client::builder()
        .default_headers(headers)
        .timeout(Duration::from_secs(TIMEOUT_SECS))
        .build()
        .map_err(|e| EmbedError::Http(e.to_string()))
}

fn embed_url() -> String {
    env::var("OPENAI_EMBED_URL").unwrap_or_else(|_| OPENAI_EMBED_URL.to_string())
}

fn send_with_retry(body: &serde_json::Value, client: &Client) -> Result<reqwest::blocking::Response, EmbedError> {
    let mut attempt = 0;
    let url = embed_url();
    loop {
        let resp = client
            .post(&url)
            .json(body)
            .send()
            .map_err(|e| EmbedError::Http(e.to_string()))?;

        if resp.status().is_success() {
            return Ok(resp);
        }

        let status = resp.status();
        if attempt + 1 >= RETRIES || !(status.is_server_error() || status.as_u16() == 429) {
            return Err(EmbedError::Http(format!(
                "status {}: {}",
                status,
                resp.text().unwrap_or_default()
            )));
        }

        let backoff = Duration::from_millis(500 * 2u64.saturating_pow(attempt as u32));
        sleep(backoff);
        attempt += 1;
    }
}

fn embed_batch_internal(texts: &[String], model: &str, client: &Client) -> Result<Vec<Vec<f32>>, EmbedError> {
    #[derive(serde::Serialize)]
    struct EmbedRequest<'a> {
        model: &'a str,
        input: &'a [String],
    }

    let mut all_embeddings: Vec<Vec<f32>> = Vec::with_capacity(texts.len());
    for chunk in texts.chunks(MAX_BATCH) {
        let body = serde_json::to_value(EmbedRequest { model, input: chunk })
            .map_err(|e| EmbedError::Parse(e.to_string()))?;
        let resp = send_with_retry(&body, client)?;
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
    let client = http_client(&api_key)?;
    let model_to_use = model.unwrap_or_else(|| DEFAULT_MODEL.to_string());
    embed_batch_internal(&texts, &model_to_use, &client).map_err(Into::into)
}

#[pymodule]
fn embedding_engine(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(embed_batch, m)?)?;
    Ok(())
}
