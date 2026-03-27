# dbt-vectors (prototype scaffold)

> Make vector indexes a first-class materialization in dbt. This repo is an MVP scaffold to prove the concept.

## Why
- dbt today only materializes SQL artifacts (table, view, incremental, ephemeral).
- Vector pipelines require SQL + embeddings + upsert to a vector DB; teams currently stitch that with ad‑hoc Python scripts.
- A custom `vector_index` materialization can run inside `dbt build`, generating embeddings, handling incremental logic, and writing to pgvector/Pinecone/Qdrant.

## What’s here
- **dbt package skeleton** with a `vector_index` materialization stub and dispatchable macros.
- **Python bridge** (`dbt_vectors.embedding`) intended to wrap a Rust embedding engine via PyO3.
- **Rust crate stub** (`rust/embedding_engine`) ready for OpenAI or local model clients and a batched embedding API.
- **Examples** to show how a model might be defined once code is filled in.

## Repo layout
- `dbt_project.yml` – declares this as a dbt package and exposes macros.
- `macros/materializations/vector_index.sql` – Jinja materialization scaffold (pgvector first, adapters dispatchable).
- `macros/embedding_generate.sql` – macro that calls the Python bridge (swappable per adapter/model).
- `python/dbt_vectors` – Python package placeholder for dbt-run-time helpers and PyO3 bindings.
- `rust/embedding_engine` – Rust crate stub for the high-performance embedding engine.

## Next steps (MVP path)
1. Implement Rust embedding engine with OpenAI + optional local model, expose PyO3 bindings. ✅ (stubbed to OpenAI API; needs compile and key)
2. Fill `dbt_vectors.embedding` with batching/retry/backoff and adapter-specific upsert utilities. ✅ (PyO3 + Python fallback)
3. Finish `vector_index` materialization logic (create index, incremental upsert, freshness logging). ✅ (pgvector prototype; add Pinecone/Qdrant next)
4. Add tests: unit (Python + Rust) and dbt integration (pgvector target container). ⏳
5. Publish as a dbt package + PyPI package, add docs + quickstart. ⏳

## Example model (goal state)
```sql
{{ config(
    materialized='vector_index',
    vector_db='pgvector',
    index_name='knowledge_base',
    embedding_model='text-embedding-3-small',
    dimensions=1536,
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

Running `dbt build --select vector_knowledge_base` should:
- fetch incremental rows
- generate embeddings via Rust engine
- upsert to pgvector (or Pinecone/Qdrant via adapters)
- emit metrics (processed, failed, latency) and freshness tests
```

## Local integration harness (pgvector)
```
docker-compose up -d postgres
OPENAI_API_KEY=sk-... dbt --project-dir examples --profiles-dir examples run --select vector_knowledge_base
```

The example profile points at the local Postgres/pgvector container (user/password/db: `dbt`). Build the Rust extension first if you want the PyO3 path:
```
cd rust/embedding_engine && maturin develop
```
If the Rust module is not built, the Python fallback uses the `openai` package (install via `pip install 'dbt-vectors[openai]'`).

## Rust embedding engine: runtime config & testing

Key env vars (defaults in parentheses):
- `OPENAI_API_KEY` (required): API key used for requests.
- `OPENAI_EMBED_URL` (`https://api.openai.com/v1/embeddings`): override for self-hosted/endpoints.
- `EMBED_MODEL` (`text-embedding-3-small`): model name passed to the API.
- `EMBED_MAX_BATCH` (`128`): chunk size for requests; texts are split into chunks of this size.
- `EMBED_RETRIES` (`3`): max attempts on 429/5xx with exponential backoff + jitter.
- `EMBED_TIMEOUT_SECS` (`60`): per-request timeout.

Run tests on macOS (Homebrew Python 3.12):
```bash
export PYO3_PYTHON=$(brew --prefix python@3.12)/bin/python3.12
export PYO3_LIB_DIR=$(brew --prefix python@3.12)/Frameworks/Python.framework/Versions/3.12/lib
export PYO3_INCLUDE_DIR=$(brew --prefix python@3.12)/Frameworks/Python.framework/Versions/3.12/include/python3.12
export MACOSX_DEPLOYMENT_TARGET=12.0
export RUSTFLAGS="-L${PYO3_LIB_DIR} -lpython3.12 -Wl,-rpath,${PYO3_LIB_DIR}"
cd rust/embedding_engine
cargo clean
cargo test -q
```

Run tests in Docker (quiet, no local toolchain needed):
```bash
docker run --rm --platform linux/amd64 \
  -v "$PWD":/workspace \
  -w /workspace/rust/embedding_engine \
  rustlang/rust:nightly-slim \
  bash -lc 'export PATH=/usr/local/cargo/bin:$PATH DEBIAN_FRONTEND=noninteractive \
            MALLOC_CONF="abort:false,prof:false,background_thread:false,stats_print:false" && \
            apt-get update >/dev/null && \
            apt-get install -y python3 python3-dev pkg-config libssl-dev >/dev/null && \
            cargo test -q'
```
