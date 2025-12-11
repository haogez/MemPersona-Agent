from __future__ import annotations

import hashlib
from typing import List

import httpx
import numpy as np
from tenacity import AsyncRetrying, retry_if_exception_type, stop_after_attempt, wait_exponential

from mem_persona_agent.config import settings


async def embed(text: str) -> List[float]:
    """Generate embedding vector for text."""
    if not settings.llm_api_base or not settings.llm_api_key:
        return _fallback_embed(text)

    payload = {
        "model": settings.embed_model_name,
        "input": text,
    }
    headers = {"Authorization": f"Bearer {settings.llm_api_key}"}

    async for attempt in AsyncRetrying(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=4),
        retry=retry_if_exception_type(httpx.HTTPError),
    ):
        with attempt:
            async with httpx.AsyncClient(base_url=settings.llm_api_base, timeout=30.0) as client:
                response = await client.post("/embeddings", headers=headers, json=payload)
                response.raise_for_status()
                data = response.json()
                vector = data.get("data", [{}])[0].get("embedding")
                if vector is None:  # pragma: no cover - defensive
                    raise RuntimeError("Embedding response missing data")
                return [float(x) for x in vector]

    raise RuntimeError("Embedding failed")


def _fallback_embed(text: str) -> List[float]:
    # deterministic pseudo embedding using hash
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "big"))
    return [float(x) for x in rng.random(16)]
