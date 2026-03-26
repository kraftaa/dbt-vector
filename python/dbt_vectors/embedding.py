from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Mapping, Sequence


@dataclass
class UpsertResult:
    processed: int
    failed: int
    latency_ms: float | None = None
    detail: Mapping[str, object] | None = None


def embed_batch(
    texts: Sequence[str],
    model: str = "text-embedding-3-small",
    batch_size: int = 128,
) -> List[List[float]]:
    """Generate embeddings for a batch of text.

    Preferred path: call the Rust PyO3 module `embedding_engine.embed_batch`.
    Fallback: use the Python OpenAI client if installed.
    """

    try:
        import embedding_engine  # type: ignore

        return embedding_engine.embed_batch(list(texts), model)
    except ModuleNotFoundError:
        pass  # fall through to Python client

    try:
        from openai import OpenAI  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover - missing dep
        raise NotImplementedError(
            "embed_batch requires the Rust extension (embedding_engine) or the openai package."
        ) from exc

    client = OpenAI()
    out: List[List[float]] = []
    # simple batching to avoid large payloads
    for i in range(0, len(texts), batch_size):
        chunk = list(texts[i : i + batch_size])
        resp = client.embeddings.create(model=model, input=chunk)
        out.extend([item.embedding for item in resp.data])
    return out
