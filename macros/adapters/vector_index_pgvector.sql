{% macro postgres__vector_index_run(vector_db, source_relation, target_relation, embedding_model, unique_key, metadata_columns, batch_size, dimensions, text_column, updated_at_column, upsert_batch_size, is_incremental) -%}
    {#
      pgvector runner prototype (MVP). Assumes pgvector extension is available on the target warehouse.
      Flow:
        1) Ensure extension + target table.
        2) Pull candidate rows from the temp source relation (incremental aware).
        3) Generate embeddings via Python/Rust bridge (dbt_vectors.embedding.embed_batch).
        4) Upsert into target_relation with ON CONFLICT on unique_key.

      NOTE: embed_batch is still a stub; implement PyO3/OpenAI before production use.
    #}

    {%- set modules = context['modules'] -%}
    {%- set importlib = modules.importlib -%}
    {%- set embedding = importlib.import_module('dbt_vectors.embedding') -%}

    -- 1) make sure pgvector exists
    {%- call statement('ensure_vector_ext', fetch_result=False) -%}
      create extension if not exists vector;
    {%- endcall -%}

    {%- set source_cols = adapter.get_columns_in_relation(source_relation) -%}

    {# helper to lookup column types from source by name #}
    {%- set text_col_type = none -%}
    {%- set unique_col_type = none -%}
    {%- set meta_types = {} -%}
    {%- set updated_col_type = 'timestamptz' -%}
    {%- for col in source_cols -%}
      {%- if col.name | lower == text_column | lower -%}
        {%- set text_col_type = col.data_type -%}
      {%- endif -%}
      {%- if col.name | lower == unique_key | lower -%}
        {%- set unique_col_type = col.data_type -%}
      {%- endif -%}
      {%- if updated_at_column and col.name | lower == updated_at_column | lower -%}
        {%- set updated_col_type = col.data_type -%}
      {%- endif -%}
      {%- for meta in metadata_columns -%}
        {%- if col.name | lower == meta | lower -%}
          {%- do meta_types.update({meta: col.data_type}) -%}
        {%- endif -%}
      {%- endfor -%}
    {%- endfor -%}

    {%- if text_col_type is none -%}
      {{ exceptions.raise_compiler_error('text_column ' ~ text_column ~ ' not found in source relation ' ~ source_relation) }}
    {%- endif -%}
    {%- if unique_col_type is none -%}
      {{ exceptions.raise_compiler_error('unique_key ' ~ unique_key ~ ' not found in source relation ' ~ source_relation) }}
    {%- endif -%}

    {# 2) create target table if needed #}
    {%- set dim_clause = '' -%}
    {%- if dimensions is not none -%}
      {%- set dim_clause = '(' ~ dimensions ~ ')' -%}
    {%- endif -%}

    {%- set create_cols = [] -%}
    {%- do create_cols.append(adapter.quote(unique_key) ~ ' ' ~ unique_col_type) -%}
    {%- do create_cols.append('embedding vector' ~ dim_clause) -%}
    {%- for meta in metadata_columns -%}
      {%- set col_type = meta_types.get(meta, 'text') -%}
      {%- do create_cols.append(adapter.quote(meta) ~ ' ' ~ col_type) -%}
    {%- endfor -%}
    {%- do create_cols.append('source_updated_at ' ~ updated_col_type) -%}
    {%- do create_cols.append('processed_at timestamptz') -%}

    {%- call statement('create_target', fetch_result=False) -%}
      create table if not exists {{ target_relation }} (
        {{ create_cols | join(',\n        ') }},
        primary key ({{ adapter.quote(unique_key) }})
      );
    {%- endcall -%}

    {# 3) fetch rows to embed #}
    {%- set select_sql %}
      select
        {{ adapter.quote(unique_key) }} as unique_key,
        {{ adapter.quote(text_column) }} as text_value
        {%- for meta in metadata_columns %}
          , {{ adapter.quote(meta) }}
        {%- endfor %}
        {%- if updated_at_column %}
          , {{ adapter.quote(updated_at_column) }} as updated_at_value
        {%- endif %}
      from {{ source_relation }}
      where {{ adapter.quote(text_column) }} is not null
        and length(trim({{ adapter.quote(text_column) }})) > 0
      {%- if is_incremental and updated_at_column %}
        and {{ adapter.quote(updated_at_column) }} > coalesce((select max(source_updated_at) from {{ target_relation }}), '1970-01-01'::timestamptz)
      {%- endif %}
    {%- endset %}

    {%- set result = run_query(select_sql) -%}
    {%- if result is none -%}
      {{ exceptions.raise_compiler_error('run_query returned no result set for vector_index_run') }}
    {%- endif -%}

    {%- set rows = result.rows -%}
    {%- if (rows | length) == 0 -%}
      {{ log('[dbt-vectors] no rows to embed for ' ~ target_relation, info=True) }}
      {{ return({'embedded': 0, 'upserted': 0, 'skipped': 0}) }}
    {%- endif -%}

    {# 4) generate embeddings in batches via Python bridge #}
    {%- set texts = [] -%}
    {%- for row in rows -%}
      {%- do texts.append(row[1]) -%}
    {%- endfor -%}

    {%- set embeddings = embedding.embed_batch(texts, model=embedding_model, batch_size=batch_size) -%}

    {%- if embeddings | length != rows | length -%}
      {{ exceptions.raise_compiler_error('embed_batch returned ' ~ (embeddings|length) ~ ' embeddings for ' ~ (rows|length) ~ ' rows') }}
    {%- endif -%}

    {# 5) build and run bulk upsert #}
    {%- set insert_cols = [adapter.quote(unique_key), 'embedding'] + metadata_columns | map(adapter.quote) | list + ['source_updated_at', 'processed_at'] -%}

    {%- set value_rows = [] -%}
    {%- for i in range(rows | length) -%}
      {%- set row = rows[i] -%}
      {%- set emb = embeddings[i] -%}
      {%- set emb_literal = "'[" ~ (emb | join(',')) ~ "]'" -%}

      {%- set cells = [] -%}
      {%- do cells.append("'" ~ (row[0] | string | replace("'", "''")) ~ "'") -%}
      {%- do cells.append(emb_literal ~ '::vector' ~ dim_clause) -%}
      {%- set meta_offset = 2 -%}
      {%- for j in range(metadata_columns | length) -%}
        {%- set meta_val = row[meta_offset + j] -%}
        {%- if meta_val is none -%}
          {%- do cells.append('NULL') -%}
        {%- else -%}
          {%- do cells.append("'" ~ (meta_val | string | replace("'", "''")) ~ "'") -%}
        {%- endif -%}
      {%- endfor -%}
      {%- set updated_val = row[meta_offset + (metadata_columns | length)] if updated_at_column else none -%}
      {%- if updated_val is none -%}
        {%- do cells.append('NULL') -%}
      {%- else -%}
        {%- do cells.append("'" ~ (updated_val | string | replace("'", "''")) ~ "'") -%}
      {%- endif -%}
      {%- do cells.append('now()') -%}

      {%- do value_rows.append('(' ~ cells | join(', ') ~ ')') -%}
    {%- endfor -%}

    {# batch upserts to avoid giant statements #}
    {%- set chunk = upsert_batch_size -%}
    {%- for offset in range(0, value_rows | length, chunk) -%}
      {%- set slice = value_rows[offset : offset + chunk] -%}
      {%- set insert_sql %}
        insert into {{ target_relation }} ({{ insert_cols | join(', ') }})
        values {{ slice | join(',\n               ') }}
        on conflict ({{ adapter.quote(unique_key) }}) do update set
          embedding = excluded.embedding,
          source_updated_at = excluded.source_updated_at,
          processed_at = excluded.processed_at
          {%- for meta in metadata_columns %}
            , {{ adapter.quote(meta) }} = excluded.{{ adapter.quote(meta) }}
          {%- endfor %}
        ;
      {%- endset %}
      {%- call statement('upsert_vectors_' ~ offset, fetch_result=False) -%}
        {{ insert_sql }}
      {%- endcall -%}
    {%- endfor -%}

    {{ log('[dbt-vectors] upserted ' ~ (rows|length) ~ ' rows into ' ~ target_relation, info=True) }}

    {{ return({'embedded': (rows|length), 'upserted': (rows|length)}) }}
{%- endmacro %}

{% macro postgres__embedding_generate(text_column, model='text-embedding-3-small', batch_size=128) -%}
    {{ exceptions.raise_compiler_error('pgvector embedding_generate not implemented. Wire to Python/Rust engine.') }}
{%- endmacro %}
