from __future__ import annotations

import json
from typing import Any, Dict, List

import httpx
from tenacity import AsyncRetrying, retry_if_exception_type, stop_after_attempt, wait_exponential

from mem_persona_agent.config import settings


class ChatError(RuntimeError):
    pass


class ChatClient:
    """Async chat completion client with OpenAI compatible interface."""

    def __init__(self, model: str | None = None):
        self.model = model or settings.llm_model_name

    async def chat(self, messages: List[Dict[str, str]], temperature: float = 0.7) -> str:
        # Provide a mockable path when API is not configured
        if not settings.llm_api_base or not settings.llm_api_key:
            return self._fallback(messages)

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "stream": False,
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
                    response = await client.post("/chat/completions", headers=headers, json=payload)
                    response.raise_for_status()
                    data = response.json()
                    try:
                        return data["choices"][0]["message"]["content"]
                    except (KeyError, IndexError) as exc:  # pragma: no cover - defensive
                        raise ChatError("Unexpected response structure") from exc

        raise ChatError("Failed to get chat completion")

    def _fallback(self, messages: List[Dict[str, str]]) -> str:
        """Deterministic local fallback for tests."""
        last_user = next((m for m in reversed(messages) if m["role"] == "user"), {"content": ""})
        system = next((m for m in messages if m["role"] == "system"), {"content": ""})
        return json.dumps({"system": system.get("content", ""), "echo": last_user.get("content", "")}, ensure_ascii=False)
