-- Fails if embedding is all zeros.
select
  doc_id
from {{ ref('vector_knowledge_base_incremental') }} t
where not exists (
  select 1
  from unnest(t.embedding::float4[]) as v
  where abs(v::float8) > 1e-12
)

