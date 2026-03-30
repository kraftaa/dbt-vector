use embedding_engine::embed_batch;
use pgvector::Vector;
use postgres::{Client, NoTls};
use serde_json::json;
use std::collections::HashSet;
use std::env;

#[derive(Debug)]
struct Args {
    schema: String,
    index_name: String,
    query: String,
    top_k: i64,
    format: String,
    include_distance: bool,
    columns: Vec<String>,
}

fn parse_env_i32(key: &str, default: i32) -> i32 {
    env::var(key)
        .ok()
        .and_then(|v| v.parse::<i32>().ok())
        .filter(|v| *v > 0)
        .unwrap_or(default)
}

fn parse_arg_value(args: &[String], key: &str) -> Option<String> {
    args.windows(2).find(|w| w[0] == key).map(|w| w[1].clone())
}

fn parse_flag(args: &[String], key: &str) -> bool {
    args.iter().any(|a| a == key)
}

fn parse_columns(raw: &str) -> Result<Vec<String>, String> {
    let mut out: Vec<String> = Vec::new();
    for part in raw.split(',') {
        let col = part.trim();
        if col.is_empty() {
            continue;
        }
        validate_identifier(col, "column")?;
        if !out.iter().any(|c| c == col) {
            out.push(col.to_string());
        }
    }
    Ok(out)
}

fn validate_identifier(value: &str, name: &str) -> Result<(), String> {
    let mut chars = value.chars();
    let Some(first) = chars.next() else {
        return Err(format!(
            "invalid {} '{}': only [A-Za-z_][A-Za-z0-9_]* allowed",
            name, value
        ));
    };
    let first_ok = first.is_ascii_alphabetic() || first == '_';
    let rest_ok = chars.all(|c| c.is_ascii_alphanumeric() || c == '_');
    if !first_ok || !rest_ok {
        return Err(format!(
            "invalid {} '{}': only [A-Za-z_][A-Za-z0-9_]* allowed",
            name, value
        ));
    }
    Ok(())
}

fn parse_args() -> Result<Args, String> {
    let raw: Vec<String> = env::args().collect();
    let schema = parse_arg_value(&raw, "--schema")
        .or_else(|| env::var("SCHEMA").ok())
        .unwrap_or_else(|| "public".to_string());
    let index_name = parse_arg_value(&raw, "--index")
        .or_else(|| parse_arg_value(&raw, "--index-name"))
        .or_else(|| env::var("INDEX_NAME").ok())
        .unwrap_or_else(|| "knowledge_base".to_string());
    let query =
        parse_arg_value(&raw, "--query").ok_or_else(|| "--query is required".to_string())?;
    let top_k = parse_arg_value(&raw, "--top-k")
        .and_then(|s| s.parse::<i64>().ok())
        .filter(|v| *v > 0)
        .unwrap_or(5);
    let format = parse_arg_value(&raw, "--format").unwrap_or_else(|| "table".to_string());
    let include_distance = parse_flag(&raw, "--include-distance");
    let columns_raw = parse_arg_value(&raw, "--columns")
        .or_else(|| env::var("VECTOR_SEARCH_COLUMNS").ok())
        .unwrap_or_else(|| "doc_id,text,source,created_at".to_string());
    let columns = parse_columns(&columns_raw)?;
    if !columns.iter().any(|c| c == "doc_id") {
        return Err("--columns must include doc_id".to_string());
    }
    if !columns.iter().any(|c| c == "text") {
        return Err("--columns must include text".to_string());
    }

    validate_identifier(&schema, "schema")?;
    validate_identifier(&index_name, "index")?;
    if format != "table" && format != "json" {
        return Err("--format must be one of: table, json".to_string());
    }

    Ok(Args {
        schema,
        index_name,
        query,
        top_k,
        format,
        include_distance,
        columns,
    })
}

fn conn_str_from_env() -> String {
    let host = env::var("PGHOST").unwrap_or_else(|_| "localhost".to_string());
    let port = env::var("PGPORT").unwrap_or_else(|_| "5432".to_string());
    let user = env::var("PGUSER").unwrap_or_else(|_| "postgres".to_string());
    let password = env::var("PGPASSWORD").unwrap_or_else(|_| "".to_string());
    let dbname = env::var("PGDATABASE").unwrap_or_else(|_| "postgres".to_string());
    format!(
        "host={} port={} user={} password={} dbname={}",
        host, port, user, password, dbname
    )
}

fn table_columns(
    client: &mut Client,
    schema: &str,
    table: &str,
) -> Result<HashSet<String>, Box<dyn std::error::Error>> {
    let rows = client.query(
        "select column_name::text \
         from information_schema.columns \
         where table_schema = $1 and table_name = $2",
        &[&schema, &table],
    )?;
    Ok(rows
        .into_iter()
        .map(|r| r.get::<_, String>(0))
        .collect::<HashSet<_>>())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = parse_args().map_err(|e| format!("arg error: {e}"))?;
    let model = env::var("EMBED_MODEL").ok();
    if env::var("EMBED_PROVIDER").is_err() {
        env::set_var("EMBED_PROVIDER", "openai");
    }

    let query_vec = embed_batch(vec![args.query.clone()], model)
        .map_err(|e| format!("embed query failed: {e}"))?
        .into_iter()
        .next()
        .ok_or_else(|| "no query embedding generated".to_string())?;

    let mut client = Client::connect(&conn_str_from_env(), NoTls)?;
    let available_cols = table_columns(&mut client, &args.schema, &args.index_name)?;
    for required in ["embedding", "doc_id", "text"] {
        if !available_cols.contains(required) {
            return Err(format!(
                "index table {}.{} is missing required column '{}'",
                args.schema, args.index_name, required
            )
            .into());
        }
    }
    let table = format!("\"{}\".\"{}\"", args.schema, args.index_name);

    let mut select_parts: Vec<String> = Vec::new();
    for col in &args.columns {
        let quoted = format!("\"{}\"", col);
        let select_expr = if available_cols.contains(col) {
            format!("{}::text as {}", quoted, quoted)
        } else {
            format!("null::text as {}", quoted)
        };
        select_parts.push(select_expr);
    }
    select_parts.push("embedding <=> $1 as distance".to_string());
    let sql = format!(
        "select {} from {} order by embedding <=> $1 limit $2",
        select_parts.join(", "),
        table
    );

    let probes = parse_env_i32("VECTOR_SEARCH_PROBES", 10);
    client.batch_execute(&format!("set ivfflat.probes = {};", probes))?;

    let vec_param = Vector::from(query_vec);
    let mut rows = client.query(&sql, &[&vec_param, &args.top_k])?;
    if rows.is_empty() {
        // Fallback to exact scan when IVFFLAT returns no candidates.
        client.batch_execute(
            "set enable_indexscan = off; \
             set enable_bitmapscan = off; \
             set enable_indexonlyscan = off;",
        )?;
        rows = client.query(&sql, &[&vec_param, &args.top_k])?;
    }

    if args.format == "json" {
        let payload: Vec<serde_json::Value> = rows
            .iter()
            .map(|row| {
                let mut obj = serde_json::Map::new();
                for (idx, col) in args.columns.iter().enumerate() {
                    let val: Option<String> = row.get(idx);
                    obj.insert(col.clone(), json!(val));
                }
                if args.include_distance {
                    obj.insert(
                        "distance".to_string(),
                        json!(row.get::<_, f64>(args.columns.len())),
                    );
                }
                serde_json::Value::Object(obj)
            })
            .collect();
        println!("{}", serde_json::to_string_pretty(&payload)?);
        return Ok(());
    }

    println!(
        "Top {} results from {}.{}",
        args.top_k, args.schema, args.index_name
    );
    if rows.is_empty() {
        println!("(no rows returned)");
        return Ok(());
    }
    let doc_id_idx = args.columns.iter().position(|c| c == "doc_id").unwrap_or(0);
    let text_idx = args.columns.iter().position(|c| c == "text").unwrap_or(1);
    let distance_idx = args.columns.len();
    let meta_cols: Vec<(usize, &String)> = args
        .columns
        .iter()
        .enumerate()
        .filter(|(_, c)| *c != "doc_id" && *c != "text")
        .collect();

    for (i, row) in rows.iter().enumerate() {
        let doc_id: Option<String> = row.get(doc_id_idx);
        let text: Option<String> = row.get(text_idx);
        let distance: f64 = row.get(distance_idx);
        let mut meta_parts: Vec<String> = Vec::new();
        for (idx, col) in &meta_cols {
            let val: Option<String> = row.get(*idx);
            if let Some(v) = val {
                meta_parts.push(format!("{}={}", col, v));
            }
        }
        let meta_str = if meta_parts.is_empty() {
            String::new()
        } else {
            format!(" {}", meta_parts.join(" "))
        };
        if args.include_distance {
            println!(
                "{}. doc_id={}{} distance={:.6}",
                i + 1,
                doc_id.unwrap_or_default(),
                meta_str,
                distance
            );
        } else {
            println!(
                "{}. doc_id={}{}",
                i + 1,
                doc_id.unwrap_or_default(),
                meta_str
            );
        }
        println!("   {}", text.unwrap_or_default());
    }

    Ok(())
}
