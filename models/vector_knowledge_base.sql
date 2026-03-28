{{ config(
    materialized='vector_index',
    vector_db='pgvector',
    index_name='knowledge_base',
    embedding_model=env_var('EMBED_MODEL', 'text-embedding-3-small'),
    unique_key='doc_id',
    text_column='text',
    dimensions=(env_var('EMBED_DIMS', 1536) | int),
    metadata_columns=['source','doc_id','created_at'],
    updated_at_column='created_at',
    transaction=False
) }}

select
    doc_id,
    body,
    body as text,
    source,
    created_at
from {{ ref('staging_docs') }}

union all

select
    doc_id,
    body,
    body as text,
    source,
    created_at
from {{ ref('staging_docs_extra') }}
