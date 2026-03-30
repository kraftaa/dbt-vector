# dbt-vectors (prototype scaffold)

> Make vector indexes a first-class materialization in dbt. This repo is an MVP scaffold to prove the concept.

## Why
- dbt today only materializes SQL artifacts (table, view, incremental, ephemeral).
- Vector pipelines require SQL + embeddings + upsert to a vector DB; teams currently stitch that with ad-hoc external scripts.
- A custom `vector_index` materialization can run inside `dbt build`, generating embeddings, handling incremental logic, and writing to pgvector/Pinecone/Qdrant.

## What’s here
- **dbt package skeleton** with a `vector_index` materialization and dispatchable macros (pgvector working).
- **Rust embedder** (`rust/embedding_engine`) that can generate embeddings via OpenAI, Amazon Bedrock, or a local ONNX model (no Python needed).
- **`dbt-vectorize` CLI** with subcommands:
  - `dbt-vectorize build --select ...`
  - `dbt-vectorize embed --index-name ... --schema ...` (embed only, skip dbt run)
  - `dbt-vectorize search --select ... --query ...`
  - `dbt-vectorize inspect --select ...`
- Backward-compatible alias: `dbt-vectorize --select my_model` behaves like `dbt-vectorize build --select my_model`.
- **`./bin/vectorize` runner** still works as a compatibility alias for build flows.
- **Examples** to show how a model is defined and run.

## Use via `packages.yml` (recommended for Jupyter/other dbt projects)

You do not need to copy `macros/` manually.

Fast path (one command inside your dbt project root):

```bash
dbt-vectorize init --project-dir . --revision v0.1.5
```

`init` updates/creates `packages.yml` and runs `dbt deps`.

In your consumer dbt project, add `packages.yml`:

```yaml
packages:
  - git: "https://github.com/kraftaa/dbt-vector.git"
    revision: "v0.1.5"
```

Reference file in this repo: `examples/consumer/packages.yml`.

Then install package macros/materializations:

```bash
dbt deps
```

If you already have `packages.yml`, `init` preserves existing package entries and only adds/updates the `dbt-vector` entry.

After `dbt deps`, `materialized='vector_index'` is available in your models.

Minimal files needed in Jupyter pod:
- `dbt_project.yml`
- `profiles.yml`
- `models/*.sql` (your vector models)
- `packages.yml` (snippet above)

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

Running `dbt-vectorize build --select vector_knowledge_base` should:
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
dbt-vectorize build --select vector_knowledge_base
```

You can override runtime settings either via env vars or flags:
```
dbt-vectorize build \
  --select vector_knowledge_base_incremental \
  --embed-provider local \
  --embed-model sentence-transformers/all-MiniLM-L6-v2 \
  --embed-dims 384 \
  --embed-db-batch-size 500
```

Incremental variant (embed only new/changed rows):
```
PGHOST=localhost PGPORT=5432 PGUSER=postgres PGDATABASE=postgres \
EMBED_PROVIDER=... EMBED_MODEL=... EMBED_DIMS=... \
dbt-vectorize build --select vector_knowledge_base_incremental
```
This model uses `embed_incremental=true` and only sends rows to the embedder when:
- `doc_id` is new, or
- `created_at` (`updated_at_column`) is newer than `source_updated_at`, or
- `text` changed.

Large delta handling:
- `pg_embedder` processes source rows in database batches (default `EMBED_DB_BATCH_SIZE=1000`).
- Set `EMBED_DB_BATCH_SIZE` lower/higher to trade memory for throughput.
- pgvector macro creates supporting indexes (key/timestamp/vector) on target for production use.

Shortcut with env file:
```
cp .env.vectorize.example .env.vectorize
./bin/vectorize --select vector_knowledge_base
```
`bin/vectorize` auto-loads `.env.vectorize` if present. Use `VECTORIZE_ENV_FILE=/path/to/file` to load a different env file.

Search (semantic nearest-neighbor):
```bash
dbt-vectorize search \
  --select vector_knowledge_base \
  --query "oauth callback issues" \
  --top-k 5 \
  --format table \
  --include-distance
```

Production search recipe:
```bash
DBT=/opt/homebrew/bin/dbt \
EMBED_PROVIDER=local \
EMBED_MODEL=sentence-transformers/all-MiniLM-L6-v2 \
EMBED_LOCAL_MODEL_PATH=$PWD/ml_model \
EMBED_DIMS=384 \
VECTOR_SEARCH_PROBES=50 \
dbt-vectorize search \
  --select vector_knowledge_base_2000_varied \
  --query "oauth callback issues" \
  --top-k 5 \
  --include-distance
```

Search tuning notes:
- `--top-k`: how many nearest rows to return.
- `VECTOR_SEARCH_PROBES`: pgvector IVFFLAT probe count (`10` default). Higher = better recall, slower query.
- If IVFFLAT returns no candidates, search falls back to an exact scan automatically.
- Use `--format json` for programmatic consumption.
- `--columns doc_id,text,source,created_at` controls returned fields (must include `doc_id,text`).

Inspect resolved model config:
```bash
dbt-vectorize inspect --select vector_knowledge_base
```

Embed-only rerun (no dbt run):
```bash
dbt-vectorize embed \
  --index-name knowledge_base_2000_varied \
  --schema public \
  --embed-db-batch-size 250 \
  --embed-max-batch 250 \
  --verbose
```
Use this when `...__vector_src` already exists (e.g., after a prior `--keep-source` build).

Verbose batch logging (example: 2000 rows in chunks of 250):
```bash
dbt-vectorize build \
  --select vector_knowledge_base_2000 \
  --embed-db-batch-size 250 \
  --embed-max-batch 250 \
  --verbose
```
Expected embedder logs:
```text
[pg_embedder] db batch 1 fetched 250 rows (limit=250, total_before=0)
...
[pg_embedder] db batch 8 fetched 250 rows (limit=250, total_before=1750)
embedded 2000 rows into public.knowledge_base_2000
```

Build safety/debug flags:
- `--limit N` embeds only the first `N` rows from `__vector_src` (debug only).
- `--allow-partial` is required with `--limit` to avoid accidental partial indexes.
- `--keep-source` keeps `...__vector_src` after embedding for inspection (default drops it).

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

Run dbt checks for incremental embeddings:
```bash
EMBED_DIMS=384 dbt test --profiles-dir . --project-dir . --select vector_knowledge_base_incremental
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
dbt-vectorize build --select vector_knowledge_base
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
