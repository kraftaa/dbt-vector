use embedding_engine::embed_batch;
use pgvector::Vector;
use postgres::{Client, NoTls};
use std::env;

fn parse_env_i64(key: &str, default: i64) -> i64 {
    env::var(key)
        .ok()
        .and_then(|s| s.parse::<i64>().ok())
        .filter(|v| *v > 0)
        .unwrap_or(default)
}

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
    let db_batch_size = parse_env_i64("EMBED_DB_BATCH_SIZE", 1000);

    let conn_str = format!(
        "host={} port={} user={} password={} dbname={}",
        host, port, user, password, dbname
    );
    let mut client = Client::connect(&conn_str, NoTls)?;

    let src_table = format!("\"{}\".\"{}__vector_src\"", schema, index_name);
    let tgt_table = format!("\"{}\".\"{}\"", schema, index_name);

    // Default to OpenAI if EMBED_PROVIDER is unset.
    if env::var("EMBED_PROVIDER").is_err() {
        env::set_var("EMBED_PROVIDER", "openai");
    }

    let stmt = client.prepare(&format!(
        "update {} set embedding = $1 where doc_id = $2",
        tgt_table
    ))?;
    let mut total: usize = 0;
    let mut last_doc_id: Option<String> = None;
    loop {
        let rows = if let Some(last) = &last_doc_id {
            client.query(
                &format!(
                    "select doc_id::text, \"text\"::text from {} \
                     where \"text\" is not null and length(trim(\"text\")) > 0 \
                       and doc_id::text > $2 \
                     order by doc_id::text \
                     limit $1",
                    src_table
                ),
                &[&db_batch_size, last],
            )?
        } else {
            client.query(
                &format!(
                    "select doc_id::text, \"text\"::text from {} \
                     where \"text\" is not null and length(trim(\"text\")) > 0 \
                     order by doc_id::text \
                     limit $1",
                    src_table
                ),
                &[&db_batch_size],
            )?
        };

        if rows.is_empty() {
            break;
        }

        let batch: Vec<(String, String)> = rows.into_iter().map(|r| (r.get(0), r.get(1))).collect();
        let texts: Vec<String> = batch.iter().map(|(_, t)| t.clone()).collect();
        let vectors = embed_batch(texts, model.clone()).map_err(|e| e.to_string())?;

        for ((doc_id, _), vec) in batch.iter().zip(vectors.into_iter()) {
            let vec_param = Vector::from(vec);
            client.execute(&stmt, &[&vec_param, doc_id])?;
        }

        total += batch.len();
        last_doc_id = batch.last().map(|(d, _)| d.clone());
    }

    if total == 0 {
        println!("no rows to embed");
        return Ok(());
    }

    // optional: drop the source table
    let _ = client.execute(&format!("drop table if exists {}", src_table), &[]);

    println!("embedded {} rows into {}.{}", total, schema, index_name);
    Ok(())
}
