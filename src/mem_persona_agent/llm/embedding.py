from __future__ import annotations

import hashlib
import logging
from typing import List

import httpx
import numpy as np
from tenacity import AsyncRetrying, retry_if_exception_type, stop_after_attempt, wait_exponential

from mem_persona_agent.config import settings
from mem_persona_agent.utils import StatsCollector, estimate_tokens

logger = logging.getLogger(__name__)


class EmbedError(RuntimeError):
    def __init__(self, message: str, status_code: int | None = None, response_text: str | None = None):
        super().__init__(message)
        self.status_code = status_code
        self.response_text = response_text


def _truncate_text(text: str | None, max_len: int = 800) -> str | None:
    if text is None:
        return None
    if len(text) <= max_len:
        return text
    return f"{text[:max_len]}...<truncated {len(text)} chars>"


async def embed(text: str, *, stats: StatsCollector | None = None) -> List[float]:
    """Generate embedding vector for text."""
    if not settings.llm_api_base or not settings.llm_api_key:
        if stats:
            stats.record_llm_call(
                model=settings.embedding_model_name,
                prompt_tokens=estimate_tokens(text),
                completion_tokens=0,
            )
        return _fallback_embed(text)

    payload = {
        "model": settings.embedding_model_name,
        "input": text,
    }
    headers = {"Authorization": f"Bearer {settings.llm_api_key}"}
    url = f"{settings.llm_api_base.rstrip('/')}/embeddings"

    try:
        async for attempt in AsyncRetrying(
            reraise=True,
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=1, max=4),
            retry=retry_if_exception_type(httpx.HTTPError),
        ):
            with attempt:
                async with httpx.AsyncClient(timeout=30.0) as client:
                    response = await client.post(url, headers=headers, json=payload)
                    response.raise_for_status()
                    data = response.json()
                    usage = data.get("usage") if isinstance(data, dict) else None
                    if stats:
                        prompt_tokens = 0
                        if isinstance(usage, dict):
                            prompt_tokens = (
                                usage.get("prompt_tokens")
                                or usage.get("total_tokens")
                                or 0
                            )
                        if not prompt_tokens:
                            prompt_tokens = estimate_tokens(text)
                        stats.record_llm_call(
                            model=settings.embedding_model_name,
                            prompt_tokens=int(prompt_tokens),
                            completion_tokens=0,
                        )
                    vector = None
                    if isinstance(data, dict) and isinstance(data.get("data"), list) and data.get("data"):
                        vector = data.get("data", [{}])[0].get("embedding")
                    if vector is None:  # pragma: no cover - defensive
                        raise RuntimeError("Embedding response missing data")
                    return [float(x) for x in vector]
    except httpx.HTTPStatusError as exc:
        status_code = exc.response.status_code if exc.response else None
        response_text = None
        if exc.response is not None:
            try:
                response_text = _truncate_text(exc.response.text)
            except Exception:
                try:
                    response_text = _truncate_text(exc.response.content.decode("utf-8", errors="replace"))
                except Exception:
                    response_text = None
        err_msg = f"Embedding HTTP {status_code} model={settings.embedding_model_name} url={url} body={response_text}"
        logger.error("embedding request failed: %s", err_msg)
        raise EmbedError(
            err_msg,
            status_code=status_code,
            response_text=response_text,
        ) from exc
    except httpx.HTTPError as exc:
        err_msg = f"Embedding HTTP error model={settings.embedding_model_name} url={url}: {exc}"
        raise EmbedError(err_msg, status_code=None) from exc
    except Exception as exc:
        err_msg = f"Embedding failed model={settings.embedding_model_name} url={url}: {exc}"
        raise EmbedError(err_msg, status_code=None) from exc

    raise EmbedError("Embedding failed: unknown", status_code=None)


def _fallback_embed(text: str) -> List[float]:
    # deterministic pseudo embedding using hash
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "big"))
    dims = settings.embed_dimensions if settings.embed_dimensions > 0 else 16
    return [float(x) for x in rng.random(dims)]
