"""Voyage embeddings in small, rate-limit-friendly batches with retry."""

from __future__ import annotations

import time

import voyageai
from voyageai.error import RateLimitError

BATCH_SIZE = 8
BATCH_DELAY_SECONDS = 20
MAX_RETRIES = 5
BACKOFF_START_SECONDS = 30


class EmbeddingStopped(RuntimeError):
    pass


def embed_texts(client: voyageai.Client, texts: list[str], model: str, dimensions: int,
                input_type: str) -> list[list[float]]:
    wait = BACKOFF_START_SECONDS
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            res = client.embed(texts, model=model, input_type=input_type,
                               output_dimension=dimensions)
        except RateLimitError:
            if attempt == MAX_RETRIES:
                raise EmbeddingStopped("Voyage rate limit persisted after retries") from None
            print(f"  rate limited; waiting {wait}s (retry {attempt}/{MAX_RETRIES - 1})")
            time.sleep(wait)
            wait *= 2
            continue
        vectors = [[float(x) for x in v] for v in res.embeddings]
        if len(vectors) != len(texts) or any(len(v) != dimensions for v in vectors):
            raise EmbeddingStopped(f"Voyage returned vectors that are not {dimensions}-dimensional")
        return vectors
    raise EmbeddingStopped("embedding failed")


def delay_between_batches() -> None:
    time.sleep(BATCH_DELAY_SECONDS)
