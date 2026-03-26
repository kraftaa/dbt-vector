{{ config(
    materialized='vector_index',
    vector_db='pgvector',
    index_name='knowledge_base',
    embedding_model='text-embedding-3-small',
    dimensions=1536,
    metadata_columns=['source', 'created_at', 'doc_id'],
    unique_key='doc_id',
    updated_at_column='created_at'
) }}

select
    doc_id,
    chunk_text as text,
    source,
    created_at
from {{ ref('staging_documents') }}
where is_active = true
