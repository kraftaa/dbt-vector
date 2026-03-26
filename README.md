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
1. Implement Rust embedding engine with OpenAI + optional local model, expose PyO3 bindings.
2. Fill `dbt_vectors.embedding` with batching/retry/backoff and adapter-specific upsert utilities.
3. Finish `vector_index` materialization logic (create index, incremental upsert, freshness logging).
4. Add tests: unit (Python + Rust) and dbt integration (pgvector target container).
5. Publish as a dbt package + PyPI package, add docs + quickstart.

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
