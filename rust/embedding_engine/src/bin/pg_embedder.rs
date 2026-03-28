use embedding_engine::embed_batch;
use pgvector::Vector;
use postgres::{Client, NoTls};
use std::env;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let schema = env::var("SCHEMA").unwrap_or_else(|_| "public".to_string());
    let index_name = env::var("INDEX_NAME").unwrap_or_else(|_| "knowledge_base".to_string());
    let model = env::var("EMBED_MODEL").ok();
    let host = env::var("PGHOST").unwrap_or_else(|_| "localhost".to_string());
    let port: u16 = env::var("PGPORT")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(5432);
    let user = env::var("PGUSER").unwrap_or_else(|_| "postgres".to_string());
    let password = env::var("PGPASSWORD").unwrap_or_else(|_| "".to_string());
    let dbname = env::var("PGDATABASE").unwrap_or_else(|_| "postgres".to_string());

    let conn_str = format!(
        "host={} port={} user={} password={} dbname={}",
        host, port, user, password, dbname
    );
    let mut client = Client::connect(&conn_str, NoTls)?;

    let src_table = format!("\"{}\".\"{}__vector_src\"", schema, index_name);
    let tgt_table = format!("\"{}\".\"{}\"", schema, index_name);

    let rows: Vec<(String, String)> = client
        .query(
            &format!(
                "select doc_id::text, \"text\"::text from {} \
                 where \"text\" is not null and length(trim(\"text\"))>0",
                src_table
            ),
            &[],
        )?
        .into_iter()
        .map(|r| (r.get(0), r.get(1)))
        .collect();

    if rows.is_empty() {
        println!("no rows to embed");
        return Ok(());
    }

    // Default to OpenAI if EMBED_PROVIDER is unset.
    if env::var("EMBED_PROVIDER").is_err() {
        env::set_var("EMBED_PROVIDER", "openai");
    }

    let texts: Vec<String> = rows.iter().map(|(_, t)| t.clone()).collect();
    let vectors = embed_batch(texts, model).map_err(|e| e.to_string())?;
    let total = vectors.len();

    let stmt = client.prepare(&format!(
        "update {} set embedding = $1 where doc_id = $2",
        tgt_table
    ))?;

    for ((doc_id, _), vec) in rows.into_iter().zip(vectors.into_iter()) {
        let vec_param = Vector::from(vec);
        client.execute(&stmt, &[&vec_param, &doc_id])?;
    }

    // optional: drop the source table
    let _ = client.execute(&format!("drop table if exists {}", src_table), &[]);

    println!("embedded {} rows into {}.{}", total, schema, index_name);
    Ok(())
}
