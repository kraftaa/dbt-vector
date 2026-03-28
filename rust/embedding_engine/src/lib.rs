use ndarray::Array2;
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::Value;
use rand::{rngs::StdRng, Rng, SeedableRng};
use reqwest::blocking::Client;
use reqwest::header::{HeaderMap, HeaderValue, AUTHORIZATION, CONTENT_TYPE};
use serde::Deserialize;
use std::env;
use std::ops::Deref;
use std::path::PathBuf;
use std::thread::sleep;
use std::time::Duration;
use thiserror::Error;
use tokenizers::Tokenizer;

const OPENAI_EMBED_URL: &str = "https://api.openai.com/v1/embeddings";
const DEFAULT_MODEL: &str = "text-embedding-3-small";
const DEFAULT_MAX_BATCH: usize = 128;
const DEFAULT_RETRIES: usize = 3;
const DEFAULT_TIMEOUT_SECS: u64 = 60;
const BASE_BACKOFF_MS: u64 = 200;
const JITTER_MS: u64 = 100;

#[derive(Debug, Error)]
pub enum EmbedError {
    #[error("missing OPENAI_API_KEY env var")]
    MissingApiKey,
    #[error("missing EMBED_LOCAL_MODEL_PATH env var for local provider")]
    MissingLocalModelPath,
    #[error("http error: {0}")]
    Http(String),
    #[error("response parse error: {0}")]
    Parse(String),
    #[error("local embedding error: {0}")]
    Local(String),
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
enum Provider {
    OpenAi,
    Bedrock,
    Local,
}

#[derive(Clone, Debug)]
struct Config {
    max_batch: usize,
    retries: usize,
    timeout_secs: u64,
    model: String,
    provider: Provider,
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
    let provider = match env::var("EMBED_PROVIDER")
        .unwrap_or_else(|_| "openai".to_string())
        .to_lowercase()
        .as_str()
    {
        "bedrock" => Provider::Bedrock,
        "local" => Provider::Local,
        _ => Provider::OpenAi,
    };

    Config {
        max_batch: parse_env_usize("EMBED_MAX_BATCH", DEFAULT_MAX_BATCH),
        retries: parse_env_usize("EMBED_RETRIES", DEFAULT_RETRIES),
        timeout_secs: parse_env_u64("EMBED_TIMEOUT_SECS", DEFAULT_TIMEOUT_SECS),
        model: model.unwrap_or_else(|| DEFAULT_MODEL.to_string()),
        provider,
    }
}

fn http_client(api_key: &str, timeout_secs: u64) -> Result<Client, EmbedError> {
    let mut headers = HeaderMap::new();
    headers.insert(
        AUTHORIZATION,
        HeaderValue::from_str(&format!("Bearer {}", api_key))
            .map_err(|e| EmbedError::Http(e.to_string()))?,
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

fn embed_local(texts: &[String], cfg: &Config) -> Result<Vec<Vec<f32>>, EmbedError> {
    embed_local_onnx(texts, cfg)
}

fn embed_local_onnx(texts: &[String], _cfg: &Config) -> Result<Vec<Vec<f32>>, EmbedError> {
    // Expect a directory with model.onnx and tokenizer.json (HF export)
    let model_dir = env::var("EMBED_LOCAL_MODEL_PATH")
        .map(PathBuf::from)
        .map_err(|_| EmbedError::MissingLocalModelPath)?;
    let model_path = model_dir.join("model.onnx");
    let tokenizer_path = model_dir.join("tokenizer.json");

    let mut tokenizer = Tokenizer::from_file(tokenizer_path)
        .map_err(|e| EmbedError::Local(format!("tokenizer load: {e}")))?;

    let max_len: usize = env::var("EMBED_MAX_LEN")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(256);

    tokenizer.with_padding(Some(tokenizers::PaddingParams {
        strategy: tokenizers::PaddingStrategy::Fixed(max_len),
        direction: tokenizers::PaddingDirection::Right,
        pad_id: 0,
        pad_type_id: 0,
        pad_token: "[PAD]".into(),
        ..Default::default()
    }));
    let _ = tokenizer.with_truncation(Some(tokenizers::TruncationParams {
        max_length: max_len,
        strategy: tokenizers::TruncationStrategy::LongestFirst,
        ..Default::default()
    }));

    let encodings = tokenizer
        .encode_batch(texts.iter().map(|s| s.as_str()).collect::<Vec<_>>(), true)
        .map_err(|e| EmbedError::Local(format!("tokenize batch: {e}")))?;

    let seq_len = max_len;
    let batch = encodings.len();

    let mut ids: Vec<i64> = Vec::with_capacity(batch * seq_len);
    let mut mask: Vec<i64> = Vec::with_capacity(batch * seq_len);
    for enc in &encodings {
        let mut t = enc.get_ids().to_vec();
        let mut m = enc.get_attention_mask().to_vec();
        t.resize(seq_len, 0);
        m.resize(seq_len, 0);
        ids.extend(t.into_iter().map(|v| v as i64));
        mask.extend(m.into_iter().map(|v| v as i64));
    }

    let mut session =
        Session::builder().map_err(|e| EmbedError::Local(format!("ort session builder: {e}")))?;
    session = session
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .map_err(|e| EmbedError::Local(format!("ort opt level: {e}")))?;
    session = session
        .with_intra_threads(num_cpus::get())
        .map_err(|e| EmbedError::Local(format!("ort threads: {e}")))?;
    let mut session = session
        .commit_from_file(model_path)
        .map_err(|e| EmbedError::Local(format!("ort session: {e}")))?;

    let input_ids: Array2<i64> = Array2::from_shape_vec((batch, seq_len), ids)
        .map_err(|e| EmbedError::Local(format!("ids reshape: {e}")))?;
    let attention_mask: Array2<i64> = Array2::from_shape_vec((batch, seq_len), mask)
        .map_err(|e| EmbedError::Local(format!("mask reshape: {e}")))?;
    let token_type_ids: Array2<i64> = Array2::zeros((batch, seq_len));

    let input_ids_val =
        Value::from_array(input_ids).map_err(|e| EmbedError::Local(format!("value ids: {e}")))?;
    let attention_mask_val = Value::from_array(attention_mask.clone())
        .map_err(|e| EmbedError::Local(format!("value mask: {e}")))?;
    let token_type_ids_val = Value::from_array(token_type_ids)
        .map_err(|e| EmbedError::Local(format!("value token_type_ids: {e}")))?;

    let inputs = ort::inputs! {
        "input_ids" => input_ids_val,
        "attention_mask" => attention_mask_val,
        "token_type_ids" => token_type_ids_val
    };

    let first_name = {
        let outputs = session.outputs();
        outputs
            .first()
            .ok_or_else(|| EmbedError::Local("no outputs in model".into()))?
            .name()
            .to_string()
    };

    let outputs = session
        .run(inputs)
        .map_err(|e| EmbedError::Local(format!("ort run: {e}")))?;

    // Expect first output: [batch, seq, hidden]
    let output = outputs
        .get(first_name.as_str())
        .ok_or_else(|| EmbedError::Local("no output tensor".into()))?;
    let (shape, data) = output
        .try_extract_tensor::<f32>()
        .map_err(|e| EmbedError::Local(format!("extract tensor: {e}")))?;
    let dims: &[i64] = shape.deref();
    if dims.len() != 3 {
        return Err(EmbedError::Local(format!(
            "expected 3D output, got shape {:?}",
            dims
        )));
    }
    let batch_dim = dims[0] as usize;
    let seq_len_dim = dims[1] as usize;
    let hidden = dims[2] as usize;

    // data is laid out [batch][seq][hidden] contiguous
    // mean pooling with attention mask
    let mut results = Vec::with_capacity(batch_dim);
    for b in 0..batch_dim {
        let mut sum = vec![0f32; hidden];
        let mut count = 0f32;
        for t in 0..seq_len_dim {
            if attention_mask[(b, t)] == 0 {
                continue;
            }
            count += 1.0;
            for (h, slot) in sum.iter_mut().enumerate().take(hidden) {
                let idx = b * seq_len_dim * hidden + t * hidden + h;
                *slot += data[idx];
            }
        }
        if count > 0.0 {
            for v in sum.iter_mut() {
                *v /= count;
            }
        }
        results.push(sum);
    }

    Ok(results)
}

fn embed_bedrock(texts: &[String], cfg: &Config) -> Result<Vec<Vec<f32>>, EmbedError> {
    use aws_config::BehaviorVersion;
    use aws_sdk_bedrockruntime::primitives::Blob;
    use tokio::runtime::Runtime;

    let rt = Runtime::new().map_err(|e| EmbedError::Http(e.to_string()))?;
    rt.block_on(async {
        let shared_config = aws_config::load_defaults(BehaviorVersion::latest()).await;
        let client = aws_sdk_bedrockruntime::Client::new(&shared_config);
        let mut out = Vec::with_capacity(texts.len());
        for t in texts {
            let body = serde_json::json!({ "inputText": t });
            let resp = client
                .invoke_model()
                .model_id(&cfg.model)
                .content_type("application/json")
                .accept("application/json")
                .body(Blob::new(body.to_string()))
                .send()
                .await
                .map_err(|e| EmbedError::Http(e.to_string()))?;
            let bytes = resp.body.into_inner();
            let parsed: serde_json::Value =
                serde_json::from_slice(&bytes).map_err(|e| EmbedError::Parse(e.to_string()))?;
            let embed_arr = parsed
                .get("embedding")
                .and_then(|v| v.as_array())
                .ok_or_else(|| EmbedError::Parse("missing embedding array".to_string()))?;
            let vecf: Vec<f32> = embed_arr
                .iter()
                .filter_map(|v| v.as_f64())
                .map(|v| v as f32)
                .collect();
            out.push(vecf);
        }
        Ok(out)
    })
}

/// Generate embeddings (OpenAI/Bedrock/local ONNX).
pub fn embed_batch(texts: Vec<String>, model: Option<String>) -> Result<Vec<Vec<f32>>, EmbedError> {
    let cfg = load_config(model);
    match cfg.provider {
        Provider::Local => embed_local(&texts, &cfg),
        Provider::Bedrock => embed_bedrock(&texts, &cfg),
        Provider::OpenAi => {
            let api_key = env::var("OPENAI_API_KEY").map_err(|_| EmbedError::MissingApiKey)?;
            let client = http_client(&api_key, cfg.timeout_secs)?;
            embed_batch_internal(&texts, &cfg, &client)
        }
    }
}
