{% macro embedding_generate(text_column, model='text-embedding-3-small', batch_size=128) -%}
    {#
      Dispatchable macro that should return a list/array of embeddings for the given text column.
      The default implementation is intentionally a stub.
    #}
    {{ adapter.dispatch('embedding_generate', 'dbt_vectorize')(text_column=text_column, model=model, batch_size=batch_size) }}
{%- endmacro %}
