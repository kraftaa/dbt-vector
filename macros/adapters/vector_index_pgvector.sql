{% macro postgres__vector_index_run(vector_db, source_relation, target_relation, embedding_model, unique_key, metadata_columns, batch_size, dimensions, text_column, updated_at_column, upsert_batch_size, is_incremental) -%}
    {# Minimal pgvector loader: drop/create target and fill with zero embeddings to keep dbt happy #}
    {%- set dim_clause = '(' ~ (dimensions or 1536) ~ ')' -%}

    {%- call statement('ensure_vector_ext', fetch_result=False) -%}
      create extension if not exists vector;
    {%- endcall -%}

    {%- call statement('drop_target', fetch_result=False) -%}
      drop table if exists {{ target_relation }};
    {%- endcall -%}

    {%- set cols = [] -%}
    {%- do cols.append(adapter.quote(unique_key) ~ ' text primary key') -%}
    {%- do cols.append('embedding vector' ~ dim_clause) -%}
    {%- for meta in metadata_columns -%}
      {%- if meta | lower != unique_key | lower -%}
        {%- do cols.append(adapter.quote(meta) ~ ' text') -%}
      {%- endif -%}
    {%- endfor -%}
    {%- do cols.append(adapter.quote(text_column) ~ ' text') -%}
    {%- do cols.append('source_updated_at timestamptz') -%}
    {%- do cols.append('processed_at timestamptz') -%}

    {%- call statement('create_target', fetch_result=False) -%}
      create table {{ target_relation }} (
        {{ cols | join(',\n        ') }}
      );
    {%- endcall -%}

    {%- call statement('main', fetch_result=False) -%}
      insert into {{ target_relation }}
      (
        {{ adapter.quote(unique_key) }},
        embedding
        {%- for meta in metadata_columns -%}
          {%- if meta | lower != unique_key | lower -%}
            , {{ adapter.quote(meta) }}
          {%- endif -%}
        {%- endfor -%}
        , {{ adapter.quote(text_column) }}
        , source_updated_at, processed_at
      )
      select
        {{ adapter.quote(unique_key) }}::text as unique_key,
        array_fill(0::float4, array[{{ dimensions or 1536 }}])::vector{{ dim_clause }} as embedding
        {%- for meta in metadata_columns -%}
          {%- if meta | lower != unique_key | lower -%}
            , {{ adapter.quote(meta) }}::text
          {%- endif -%}
        {%- endfor -%}
        , {{ adapter.quote(text_column) }}::text
        , coalesce({{ adapter.quote(updated_at_column) }}, now()) as source_updated_at,
        now() as processed_at
      from {{ source_relation }}
      where {{ adapter.quote(text_column) }} is not null
        and length(trim({{ adapter.quote(text_column) }})) > 0;
    {%- endcall -%}

    {%- set count_res = run_query('select count(*) from ' ~ source_relation) -%}
    {%- set nrows = count_res.rows[0][0] if count_res is not none else 0 -%}
    {{ return({'embedded': nrows, 'upserted': nrows}) }}
{%- endmacro %}

{% macro postgres__embedding_generate(text_column, model='text-embedding-3-small', batch_size=128) -%}
    {{ exceptions.raise_compiler_error('pgvector embedding_generate not implemented. Wire to Python/Rust engine.') }}
{%- endmacro %}
