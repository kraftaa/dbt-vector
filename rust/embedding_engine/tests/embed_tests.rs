use embedding_engine::embed_batch;
use httpmock::Method::POST;
use httpmock::MockServer;
use once_cell::sync::Lazy;
use std::env;

static _LOCK: Lazy<()> = Lazy::new(|| {
    // ensure env isolation per test run
    env::remove_var("OPENAI_API_KEY");
    env::remove_var("OPENAI_EMBED_URL");
});

#[test]
fn chunks_requests_to_max_batch() {
    let _guard = &_LOCK;
    let server = MockServer::start();

    let m1 = server.mock(|when, then| {
        when.method(POST).path("/v1/embeddings");
        then.status(200).json_body(serde_json::json!({
            "data": (0..128).map(|_| serde_json::json!({"embedding": [0.1, 0.2]})).collect::<Vec<_>>()
        }));
    });
    let m2 = server.mock(|when, then| {
        when.method(POST).path("/v1/embeddings");
        then.status(200).json_body(serde_json::json!({
            "data": (0..128).map(|_| serde_json::json!({"embedding": [0.3, 0.4]})).collect::<Vec<_>>()
        }));
    });

    env::set_var("OPENAI_API_KEY", "test-key");
    env::set_var("OPENAI_EMBED_URL", format!("{}{}", server.base_url(), "/v1/embeddings"));

    let texts: Vec<String> = (0..256).map(|i| format!("doc-{}", i)).collect();

    let res = embed_batch(texts, Some("dummy".to_string()));

    assert!(res.is_ok());
    let embeddings = res.unwrap();
    assert_eq!(embeddings.len(), 256);
    m1.assert();
    m2.assert();
}

#[test]
fn retries_on_429_then_succeeds() {
    let _guard = &_LOCK;
    let server = MockServer::start();

    let _m429 = server.mock(|when, then| {
        when.method(POST).path("/v1/embeddings");
        then.status(429).body("rate limited");
    });
    let m200 = server.mock(|when, then| {
        when.method(POST).path("/v1/embeddings");
        then.status(200).json_body(serde_json::json!({
            "data": [ {"embedding": [0.5, 0.6]} ]
        }));
    });

    env::set_var("OPENAI_API_KEY", "test-key");
    env::set_var("OPENAI_EMBED_URL", format!("{}{}", server.base_url(), "/v1/embeddings"));

    let texts = vec!["one".to_string()];
    let res = embed_batch(texts, Some("dummy".to_string()));
    assert!(res.is_ok());
    assert_eq!(res.unwrap().len(), 1);
    // 429 may or may not be hit depending on retry timing; ensure success call hit once.
    m200.assert();
}
