-- Fails if any row has a null embedding.
select
  doc_id
from {{ ref('vector_knowledge_base_incremental') }}
where embedding is null

