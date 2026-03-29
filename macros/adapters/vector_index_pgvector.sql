{% macro postgres__vector_index_run(vector_db, source_relation, target_relation, embedding_model, unique_key, metadata_columns, batch_size, dimensions, text_column, updated_at_column, upsert_batch_size, embed_incremental, is_incremental) -%}
    {# pgvector loader: full-refresh by default, optional incremental delta mode #}
    {%- set dim_clause = '(' ~ (dimensions or 1536) ~ ')' -%}

    {%- call statement('ensure_vector_ext', fetch_result=False) -%}
      create extension if not exists vector;
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

    {%- if embed_incremental -%}
      {%- call statement('idx_source_key', fetch_result=False) -%}
        create index if not exists {{ source_relation.identifier }}__doc_id_idx
        on {{ source_relation }} ({{ adapter.quote(unique_key) }});
      {%- endcall -%}
      {%- call statement('idx_source_updated', fetch_result=False) -%}
        create index if not exists {{ source_relation.identifier }}__updated_at_idx
        on {{ source_relation }} ({{ adapter.quote(updated_at_column) }});
      {%- endcall -%}

      {%- call statement('create_target_if_missing', fetch_result=False) -%}
        create table if not exists {{ target_relation }} (
          {{ cols | join(',\n          ') }}
        );
      {%- endcall -%}

      {%- set delta_relation = adapter.Relation.create(
          database=None,
          schema=source_relation.schema,
          identifier=source_relation.identifier ~ '__delta',
          type='table'
      ) -%}

      {%- call statement('drop_delta_pre', fetch_result=False) -%}
        drop table if exists {{ delta_relation }};
      {%- endcall -%}

      {%- call statement('build_delta', fetch_result=False) -%}
        create table {{ delta_relation }} as
        select src.*
        from {{ source_relation }} src
        left join {{ target_relation }} tgt
          on tgt.{{ adapter.quote(unique_key) }} = src.{{ adapter.quote(unique_key) }}::text
        where src.{{ adapter.quote(text_column) }} is not null
          and length(trim(src.{{ adapter.quote(text_column) }})) > 0
          and (
            tgt.{{ adapter.quote(unique_key) }} is null
            or coalesce(src.{{ adapter.quote(updated_at_column) }}::timestamptz, 'epoch'::timestamptz)
               > coalesce(tgt.source_updated_at, 'epoch'::timestamptz)
            or coalesce(src.{{ adapter.quote(text_column) }}::text, '')
               <> coalesce(tgt.{{ adapter.quote(text_column) }}, '')
          );
      {%- endcall -%}

      {%- call statement('replace_source_with_delta', fetch_result=False) -%}
        truncate table {{ source_relation }};
        insert into {{ source_relation }} select * from {{ delta_relation }};
      {%- endcall -%}

      {%- call statement('drop_delta_post', fetch_result=False) -%}
        drop table if exists {{ delta_relation }};
      {%- endcall -%}

      {%- set upsert_update_cols = [] -%}
      {%- do upsert_update_cols.append('embedding = excluded.embedding') -%}
      {%- do upsert_update_cols.append(adapter.quote(text_column) ~ ' = excluded.' ~ adapter.quote(text_column)) -%}
      {%- do upsert_update_cols.append('source_updated_at = excluded.source_updated_at') -%}
      {%- do upsert_update_cols.append('processed_at = excluded.processed_at') -%}
      {%- for meta in metadata_columns -%}
        {%- if meta | lower != unique_key | lower -%}
          {%- do upsert_update_cols.append(adapter.quote(meta) ~ ' = excluded.' ~ adapter.quote(meta)) -%}
        {%- endif -%}
      {%- endfor -%}

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
          , coalesce({{ adapter.quote(updated_at_column) }}::timestamptz, now()) as source_updated_at,
          now() as processed_at
        from {{ source_relation }}
        on conflict ({{ adapter.quote(unique_key) }}) do update
        set {{ upsert_update_cols | join(', ') }};
      {%- endcall -%}

      {%- call statement('idx_target_time', fetch_result=False) -%}
        create index if not exists {{ target_relation.identifier }}__source_updated_at_idx
        on {{ target_relation }} (source_updated_at);
      {%- endcall -%}
      {%- call statement('idx_target_embedding', fetch_result=False) -%}
        create index if not exists {{ target_relation.identifier }}__embedding_idx
        on {{ target_relation }} using ivfflat (embedding vector_cosine_ops) with (lists = 100);
      {%- endcall -%}
    {%- else -%}
      {%- call statement('drop_target', fetch_result=False) -%}
        drop table if exists {{ target_relation }};
      {%- endcall -%}

      {%- call statement('create_target', fetch_result=False) -%}
        create table {{ target_relation }} (
          {{ cols | join(',\n          ') }}
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
          , coalesce({{ adapter.quote(updated_at_column) }}::timestamptz, now()) as source_updated_at,
          now() as processed_at
        from {{ source_relation }}
        where {{ adapter.quote(text_column) }} is not null
          and length(trim({{ adapter.quote(text_column) }})) > 0;
      {%- endcall -%}

      {%- call statement('idx_target_time', fetch_result=False) -%}
        create index if not exists {{ target_relation.identifier }}__source_updated_at_idx
        on {{ target_relation }} (source_updated_at);
      {%- endcall -%}
      {%- call statement('idx_target_embedding', fetch_result=False) -%}
        create index if not exists {{ target_relation.identifier }}__embedding_idx
        on {{ target_relation }} using ivfflat (embedding vector_cosine_ops) with (lists = 100);
      {%- endcall -%}
    {%- endif -%}

    {%- set count_res = run_query('select count(*) from ' ~ source_relation) -%}
    {%- set nrows = count_res.rows[0][0] if count_res is not none else 0 -%}
    {{ return({'embedded': nrows, 'upserted': nrows}) }}
{%- endmacro %}

{% macro postgres__embedding_generate(text_column, model='text-embedding-3-small', batch_size=128) -%}
    {{ exceptions.raise_compiler_error('pgvector embedding_generate not implemented. Wire to Python/Rust engine.') }}
{%- endmacro %}
