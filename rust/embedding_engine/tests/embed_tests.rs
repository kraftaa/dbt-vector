use embedding_engine::embed_batch;
use httpmock::Method::POST;
use httpmock::MockServer;
use once_cell::sync::Lazy;
use std::env;
use std::sync::Mutex;

static TEST_LOCK: Lazy<Mutex<()>> = Lazy::new(|| Mutex::new(()));

#[test]
fn chunks_requests_to_max_batch() {
    let _guard = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    env::remove_var("OPENAI_API_KEY");
    env::remove_var("OPENAI_EMBED_URL");
    env::remove_var("EMBED_MAX_BATCH");
    env::set_var("EMBED_MAX_BATCH", "128");
    env::set_var("EMBED_RETRIES", "1");
    env::set_var("EMBED_PROVIDER", "openai");
    let server = MockServer::start();

    let m_all = server.mock(|when, then| {
        when.method(POST).path("/v1/embeddings");
        then.status(200).json_body(serde_json::json!({
            "data": (0..128).map(|_| serde_json::json!({"embedding": [0.1, 0.2]})).collect::<Vec<_>>()
        }));
    });

    env::set_var("OPENAI_API_KEY", "test-key");
    env::set_var(
        "OPENAI_EMBED_URL",
        format!("{}{}", server.base_url(), "/v1/embeddings"),
    );

    let texts: Vec<String> = (0..256).map(|i| format!("doc-{}", i)).collect();

    let res = embed_batch(texts, Some("dummy".to_string()));

    assert!(res.is_ok());
    let embeddings = res.unwrap();
    assert_eq!(embeddings.len(), 256);
    m_all.assert_hits(2);
}

#[test]
fn retries_on_429_then_succeeds() {
    let _guard = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    env::remove_var("OPENAI_API_KEY");
    env::remove_var("OPENAI_EMBED_URL");
    env::remove_var("EMBED_RETRIES");
    env::set_var("EMBED_RETRIES", "2");
    env::set_var("EMBED_PROVIDER", "openai");
    let server = MockServer::start();

    let _m429 = server.mock(|when, then| {
        when.method(POST)
            .path("/v1/embeddings")
            .header("X-Retry-Attempt", "0");
        then.status(429).body("rate limited");
    });
    let m200 = server.mock(|when, then| {
        when.method(POST)
            .path("/v1/embeddings")
            .header("X-Retry-Attempt", "1");
        then.status(200).json_body(serde_json::json!({
            "data": [ {"embedding": [0.5, 0.6]} ]
        }));
    });

    env::set_var("OPENAI_API_KEY", "test-key");
    env::set_var(
        "OPENAI_EMBED_URL",
        format!("{}{}", server.base_url(), "/v1/embeddings"),
    );

    let texts = vec!["one".to_string()];
    let res = embed_batch(texts, Some("dummy".to_string()));
    assert!(res.is_ok());
    assert_eq!(res.unwrap().len(), 1);
    assert_eq!(m200.hits(), 1);
    m200.assert();
}
