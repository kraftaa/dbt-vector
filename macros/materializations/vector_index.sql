{% materialization vector_index, default -%}
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

    {%- set vector_db = config.get('vector_db', var('dbt_vectors', {}).get('default_vector_db', 'pgvector')) -%}
    {%- set index_name = config.get('index_name', this.identifier) -%}
    {%- set unique_key = config.get('unique_key', 'id') -%}
    {%- set dimensions = config.get('dimensions') -%}
    {%- set metadata_columns = config.get('metadata_columns', []) -%}
    {%- set embedding_model = config.get('embedding_model', 'text-embedding-3-small') -%}
    {%- set batch_size = config.get('batch_size', 128) -%}
    {%- set text_column = config.get('text_column', 'text') -%}
    {%- set updated_at_column = config.get('updated_at_column', 'updated_at') -%}
    {%- set upsert_batch_size = config.get('upsert_batch_size', 500) -%}

    {%- set target_relation = this.incorporate(type='table', identifier=index_name) -%}
    {%- set tmp_relation = this.incorporate(type='table', identifier=this.identifier ~ '__vector_src') -%}

    {{ log("[dbt-vectors] materializing vector_index => " ~ target_relation, info=True) }}

    -- Step 1: build the source rows to embed (uses the model SQL)
    {%- call statement('build_source', fetch_result=False) -%}
      {{ create_table_as(False, tmp_relation, sql) }}
    {%- endcall -%}

    -- Step 2: call adapter-specific runner to embed + upsert
    {%- set run_result = adapter.dispatch('vector_index_run', 'dbt_vectors')(
        vector_db=vector_db,
        source_relation=tmp_relation,
        target_relation=target_relation,
        embedding_model=embedding_model,
        unique_key=unique_key,
        metadata_columns=metadata_columns,
        batch_size=batch_size,
        dimensions=dimensions,
        text_column=text_column,
        updated_at_column=updated_at_column,
        upsert_batch_size=upsert_batch_size,
        is_incremental=is_incremental()
      ) -%}

    -- Optional: emit metrics table/logging (placeholder)
    {{ log("[dbt-vectors] run result: " ~ run_result, info=True) }}

    -- Step 3: clean up temp relation
    {%- call statement('drop_tmp', fetch_result=False) -%}
      {{ adapter.drop_relation(tmp_relation) }}
    {%- endcall -%}

    {{ return({'relations': [target_relation]}) }}
{% endmaterialization %}
