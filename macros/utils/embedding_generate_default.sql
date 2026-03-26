{% macro dbt_vectors__embedding_generate(text_column, model='text-embedding-3-small', batch_size=128) -%}
    {#
      Placeholder implementation: delegates to Python helper via run_query on a SQL wrapper.
      In a real implementation, this would call a Python UDF or external function that
      invokes the Rust embedding engine. For now, raise to make the missing piece obvious.
    #}
    {{ exceptions.raise_compiler_error('embedding_generate is not implemented yet. Provide an adapter-specific macro or Python binding.') }}
{%- endmacro %}
