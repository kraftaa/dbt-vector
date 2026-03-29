-- Fails if embedding dimensionality differs from EMBED_DIMS (default 384 for local MiniLM).
select
  doc_id
from {{ ref('vector_knowledge_base_incremental') }}
where array_length(embedding::float4[], 1) <> {{ env_var('EMBED_DIMS', 384) | int }}
