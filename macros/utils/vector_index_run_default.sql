{% macro dbt_vectors__vector_index_run(vector_db, source_relation, target_relation, embedding_model, unique_key, metadata_columns, batch_size, dimensions, text_column, updated_at_column, upsert_batch_size, is_incremental) -%}
    {#
      Adapter-dispatch entrypoint. It should:
        1) read rows from source_relation
        2) generate embeddings (embedding_generate)
        3) upsert into target_relation according to vector_db
        4) return a result dict/struct for logging

      This default implementation simply raises; adapters must override.
    #}
    {{ exceptions.raise_compiler_error('vector_index_run is not implemented for adapter ' ~ adapter.type()) }}
{%- endmacro %}
