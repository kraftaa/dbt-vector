{% materialization vector_index, default -%}
    {# run outside a single transaction so that created tables persist even if adapter issues a rollback at end #}
    {{ config(transaction=False) }}
    {#
      Prototype materialization for vector indexes. It compiles the model SQL to a temp table,
      then calls a dispatched macro to generate embeddings and upsert to the target vector DB.

      Configuration (examples):
        materialized: vector_index
        vector_db: pgvector | pinecone | qdrant
        index_name: knowledge_base
        embedding_model: text-embedding-3-small
        dimensions: 1536
        metadata_columns: ["source", "created_at", "doc_id"]
        text_column: text
        updated_at_column: updated_at
        unique_key: doc_id
        batch_size: 128
        upsert_batch_size: 500
    #}

    {%- set vector_db = config.get('vector_db',
        var('dbt_vectorize', {}).get(
          'default_vector_db',
          var('dbt_vectors', {}).get('default_vector_db', 'pgvector')
        )
      ) -%}
    {%- set index_name = config.get('index_name', this.identifier) -%}
    {%- set unique_key = (config.get('unique_key') or 'id') -%}
    {%- set dimensions = config.get('dimensions') -%}
    {%- set metadata_columns = config.get('metadata_columns', []) -%}
    {%- set embedding_model = config.get('embedding_model', 'text-embedding-3-small') -%}
    {%- set batch_size = config.get('batch_size', 128) -%}
    {%- set text_column = config.get('text_column', 'text') -%}
    {%- set updated_at_column = config.get('updated_at_column', 'updated_at') -%}
    {%- set upsert_batch_size = config.get('upsert_batch_size', 500) -%}
    {%- set embed_incremental = config.get('embed_incremental', false) -%}

    {%- set target_schema = this.schema -%}
    {%- set target_relation = adapter.Relation.create(
        database=None,
        schema=target_schema,
        identifier=index_name,
        type='table'
      ) -%}
    {%- set tmp_relation = adapter.Relation.create(
        database=None,
        schema=target_schema,
        identifier=index_name ~ '__vector_src',
        type='table'
      ) -%}
    {%- set src_relation = tmp_relation -%}

    {{ log("[dbt-vectorize] materializing vector_index => " ~ target_relation, info=True) }}

    -- Clean up any leftover temp relation from previous runs
    {%- call statement('drop_tmp_pre', fetch_result=False, auto_begin=True) -%}
      drop table if exists {{ tmp_relation }};
    {%- endcall -%}

    -- Step 1: build the source rows to embed (uses the model SQL)
    {%- call statement('build_source', fetch_result=False) -%}
      {{ create_table_as(False, tmp_relation, sql) }}
    {%- endcall -%}

    {%- set col_probe = run_query("select column_name from information_schema.columns where table_schema = '" ~ tmp_relation.schema ~ "' and table_name = '" ~ tmp_relation.identifier ~ "' order by ordinal_position") -%}
    {%- if col_probe is not none -%}
      {%- set _src_cols = col_probe.rows | map(attribute=0) | list -%}
      {{ log("[dbt-vectorize] source columns: " ~ (_src_cols | join(', ')), info=True) }}
      {%- if text_column not in _src_cols -%}
        {%- if 'text' in _src_cols -%}
          {{ log("[dbt-vectorize] overriding text_column to 'text' (matched existing column)", info=True) }}
          {%- set text_column = 'text' -%}
        {%- elif 'body' in _src_cols -%}
          {{ log("[dbt-vectorize] overriding text_column to 'body' (matched existing column)", info=True) }}
          {%- set text_column = 'body' -%}
        {%- else -%}
          {{ exceptions.raise_compiler_error("text_column '" ~ text_column ~ "' not found in source columns: " ~ (_src_cols | join(', '))) }}
        {%- endif -%}
      {%- endif -%}
    {%- else -%}
      {{ log("[dbt-vectorize] source column probe returned none", info=True) }}
    {%- endif -%}

    -- Step 2: call adapter-specific runner to embed + upsert
    {{ log("[dbt-vectorize] about to dispatch src=" ~ src_relation ~ " target=" ~ target_relation, info=True) }}
    {%- set run_result = adapter.dispatch('vector_index_run', 'dbt_vectorize')(
        vector_db=vector_db,
        source_relation=src_relation,
        target_relation=target_relation,
        embedding_model=embedding_model,
        unique_key=unique_key,
        metadata_columns=metadata_columns,
        batch_size=batch_size,
        dimensions=dimensions,
        text_column=text_column,
        updated_at_column=updated_at_column,
        upsert_batch_size=upsert_batch_size,
        embed_incremental=embed_incremental,
        is_incremental=is_incremental()
      ) -%}
    {{ log("[dbt-vectorize] passed source_relation=" ~ src_relation ~ " target_relation=" ~ target_relation, info=True) }}

    -- Optional: emit metrics table/logging (placeholder)
    {{ log("[dbt-vectorize] run result: " ~ run_result, info=True) }}

    -- Step 3: keep temp relation for embedder phase.
    -- pg_embedder is responsible for dropping it after embeddings are written
    -- unless EMBED_KEEP_SOURCE is enabled.
    {%- if env_var('DBT_VECTORIZE_DROP_SOURCE_IN_DBT', '0') | lower in ['1', 'true', 'yes', 'on'] -%}
      {%- call statement('drop_tmp', fetch_result=False) -%}
        drop table if exists {{ tmp_relation }};
      {%- endcall -%}
      {{ log('[dbt-vectorize] dropped temp source table in dbt phase (DBT_VECTORIZE_DROP_SOURCE_IN_DBT enabled)', info=True) }}
    {%- else -%}
      {{ log('[dbt-vectorize] keeping temp source table for embedder phase', info=True) }}
    {%- endif -%}

    {# Explicitly commit so a later adapter-issued ROLLBACK doesn't undo the work #}
    {%- call statement('commit_vector_index', fetch_result=False, auto_begin=False) -%}
      commit;
    {%- endcall -%}

    {{ return({'relations': [target_relation]}) }}
{% endmaterialization %}
