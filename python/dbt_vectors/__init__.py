"""dbt-vectors Python helpers.

This package is the runtime bridge between dbt macros and the Rust embedding engine.
"""

__all__ = ["embed_batch", "UpsertResult"]

from .embedding import embed_batch, UpsertResult  # noqa: E402

__version__ = "0.0.1"
