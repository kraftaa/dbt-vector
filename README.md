# dbt-vectors (prototype scaffold)

> Make vector indexes a first-class materialization in dbt. This repo is an MVP scaffold to prove the concept.

## Why
- dbt today only materializes SQL artifacts (table, view, incremental, ephemeral).
- Vector pipelines require SQL + embeddings + upsert to a vector DB; teams currently stitch that with ad-hoc external scripts.
- A custom `vector_index` materialization can run inside `dbt build`, generating embeddings, handling incremental logic, and writing to pgvector/Pinecone/Qdrant.

## What’s here
- **dbt package skeleton** with a `vector_index` materialization and dispatchable macros (pgvector working).
- **Rust embedder** (`rust/embedding_engine`) that can generate embeddings via OpenAI, Amazon Bedrock, or a local ONNX model (no Python needed).
- **`./bin/vectorize` runner**: orchestrates `dbt run` for the model and then calls the Rust embedder to write embeddings into Postgres/pgvector.
- **Examples** to show how a model is defined and run.

## Prerequisites

`dbt-vectorize` does not vendor dbt. It uses whatever dbt binary you point it to (`DBT=...`) or find on PATH.

Verify your existing dbt + adapter:
```bash
dbt --version
```
You should see a plugin like `postgres` under "Plugins".

If you do not have dbt + postgres adapter installed:
```bash
python -m pip install "dbt-core~=1.9" "dbt-postgres~=1.9"
```

You also need pgvector available in Postgres:
- install the extension package on the **Postgres server** (`vector.control` must exist on that server)
- enable it in each **database** you want to use

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

(`pgvector` is the project name; the SQL extension name is `vector`.)

## Repo layout
- `dbt_project.yml` – declares this as a dbt package and exposes macros.
- `macros/materializations/vector_index.sql` – Jinja materialization scaffold (pgvector first, adapters dispatchable).
- `macros/adapters/vector_index_pgvector.sql` – pgvector adapter macro that creates/loads the target table.
- `bin/vectorize` – orchestration command that runs dbt and then Rust embedding.
- `rust/embedding_engine` – Rust crate and `pg_embedder` binary used for embedding generation/upsert.

## Next steps (MVP path)
1. Harden Rust embedding provider support (OpenAI/Bedrock/local ONNX) with better diagnostics and retries. ⏳
2. Expand adapter macros beyond pgvector (Pinecone/Qdrant). ⏳
3. Add end-to-end integration tests for dbt + pgvector + `pg_embedder`. ⏳
4. Publish package docs and a reproducible quickstart. ⏳

## Example model (current)
```sql
{{ config(
    materialized='vector_index',
    vector_db='pgvector',
    index_name='knowledge_base',
    embedding_model='text-embedding-3-small',
    dimensions=(env_var('EMBED_DIMS', 1536) | int),
    metadata_columns=['source', 'created_at', 'doc_id']
) }}

select
    doc_id,
    chunk_text as text,
    source,
    created_at
from {{ ref('staging_documents') }}
where is_active = true
```

Running `./bin/vectorize --select vector_knowledge_base` should:
- fetch incremental rows
- generate embeddings via Rust engine
- upsert to pgvector (or Pinecone/Qdrant via adapters)
- emit metrics (processed, failed, latency) and freshness tests

## Run locally (preferred: existing local Postgres)

1) Ensure Postgres is running, reachable (`PGHOST/PGPORT/PGUSER/PGDATABASE`), and has `vector` enabled:
```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

2) Choose a provider and matching dimensions:
```
# Local ONNX (MiniLM, 384 dims)
EMBED_PROVIDER=local
EMBED_MODEL=sentence-transformers/all-MiniLM-L6-v2
EMBED_LOCAL_MODEL_PATH=$PWD/ml_model   # contains model.onnx + tokenizer.json
EMBED_DIMS=384

# OpenAI
EMBED_PROVIDER=openai
EMBED_MODEL=text-embedding-3-small
EMBED_DIMS=1536   # or a smaller dim if you request it from OpenAI

# Bedrock Titan v2 (defaults)
EMBED_PROVIDER=bedrock
EMBED_MODEL=amazon.titan-embed-text-v2:0
EMBED_DIMS=1024   # or 512/256 if you override
```

3) Run vectorization (dbt model + embedding upsert):
```
PGHOST=localhost PGPORT=5432 PGUSER=postgres PGDATABASE=postgres \
EMBED_PROVIDER=... EMBED_MODEL=... EMBED_DIMS=... \
./bin/vectorize --select vector_knowledge_base
```

Shortcut with env file:
```
cp .env.vectorize.example .env.vectorize
./bin/vectorize --select vector_knowledge_base
```
`bin/vectorize` auto-loads `.env.vectorize` if present. Use `VECTORIZE_ENV_FILE=/path/to/file` to load a different env file.

Expected CLI output (example):
```
[vectorize] running dbt model vector_knowledge_base (provider=local, model=sentence-transformers/all-MiniLM-L6-v2)
...
Done. PASS=1 WARN=0 ERROR=0 SKIP=0 TOTAL=1
[vectorize] generating embeddings via Rust into public.knowledge_base
embedded 20 rows into public.knowledge_base
[vectorize] done.
```

Quick verification in Postgres:
```sql
SELECT count(*) AS rows FROM public.knowledge_base;

SELECT
  doc_id,
  (embedding::float4[])[1:8] AS first_8_dims,
  source,
  created_at
FROM public.knowledge_base
LIMIT 5;
```

## Optional Docker Postgres

Use this only if you want a disposable local pgvector instance:
```
docker-compose up -d postgres
```
If Docker/Colima is not running, this will fail with a daemon connection error.

## Build pip package (`dbt-vectorize`)

Build from repo root (factorlens-style, bundles Rust binary in wheel):
```bash
./scripts/build_wheel_with_binary.sh
```

Artifacts will be written to `dist/`.
Install locally:
```bash
python -m pip install dist/dbt_vectorize-*.whl
```

CLI entrypoint after install:
```bash
dbt-vectorize --select vector_knowledge_base
```

CI release wheel build (macOS arm64 + Linux x86_64):
- workflow file: `.github/workflows/release.yml`
- trigger manually from Actions or push a `v*` tag
- outputs platform-specific wheels under workflow artifacts / GitHub release assets

### Supported embedding dimensions (set `EMBED_DIMS` to match)
- OpenAI `text-embedding-3-small`: 1536 (can request smaller via API parameter)
- OpenAI `text-embedding-3-large`: 3072 (can request smaller)
- Bedrock Titan embed text v2: 1024 (or 512/256)
- Bedrock Titan embed text v1: 1024 (or 512/256)
- Bedrock Cohere Embed v4: 1536 (or 1024/512/256)
- Local MiniLM (all-MiniLM-L6-v2 ONNX): 384

## Notes
- The Rust embedder is Python-free.
- Keep your Postgres `vector` column dimension aligned with `EMBED_DIMS`.
- IVFFLAT indexes warn on very small datasets; that’s expected. Rebuild after you have more rows.
