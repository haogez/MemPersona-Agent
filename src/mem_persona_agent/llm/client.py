from __future__ import annotations

import json
from typing import Any, Dict, List, AsyncGenerator
import logging

import httpx
from tenacity import AsyncRetrying, retry_if_exception_type, stop_after_attempt, wait_exponential

from mem_persona_agent.config import settings
from mem_persona_agent.utils import StatsCollector, estimate_message_tokens, estimate_tokens

logger = logging.getLogger(__name__)


class ChatError(RuntimeError):
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


class ChatClient:
    """Async chat completion client with OpenAI compatible interface."""

    def __init__(self, model: str | None = None, stats: StatsCollector | None = None):
        self.model = model or settings.llm_model_name
        self.stats = stats

    async def chat(self, messages: List[Dict[str, str]], temperature: float = 0.7) -> str:
        # Provide a mockable path when API is not configured
        if not settings.llm_api_base or not settings.llm_api_key:
            content = self._fallback(messages)
            self._record_usage(messages, content, usage=None)
            return content

        headers = {"Authorization": f"Bearer {settings.llm_api_key}"}
        timeout = None if settings.llm_timeout_seconds <= 0 else settings.llm_timeout_seconds

        async for attempt in AsyncRetrying(
            reraise=True,
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=1, max=4),
            retry=retry_if_exception_type(httpx.HTTPError),
        ):
            with attempt:
                request_url = None
                async with httpx.AsyncClient(base_url=settings.llm_api_base, timeout=timeout) as client:
                    payload = {
                        "model": self.model,
                        "messages": messages,
                        "temperature": temperature,
                        "stream": False,
                    }
                    response = await client.post("/chat/completions", headers=headers, json=payload)
                    request_url = str(response.request.url)
                try:
                    response.raise_for_status()
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
                    if not request_url and exc.response is not None and exc.response.request is not None:
                        request_url = str(exc.response.request.url)
                    err_msg = f"Chat HTTP {status_code} model={self.model} url={request_url} body={response_text}"
                    logger.error("chat completion failed: %s", err_msg)
                    raise ChatError(
                        err_msg,
                        status_code=status_code,
                        response_text=response_text,
                    ) from exc
                data = response.json()
                try:
                    content = data["choices"][0]["message"]["content"]
                    self._record_usage(messages, content, usage=data.get("usage"))
                    return content
                except (KeyError, IndexError) as exc:  # pragma: no cover - defensive
                    raise ChatError("Unexpected response structure") from exc

        raise ChatError("Failed to get chat completion")

    async def stream_chat(self, messages: List[Dict[str, str]], temperature: float = 0.7) -> AsyncGenerator[str, None]:
        """Streaming chat completion yielding tokens."""
        if not settings.llm_api_base or not settings.llm_api_key:
            # fallback: yield once
            yield self._fallback(messages)
            return

        headers = {"Authorization": f"Bearer {settings.llm_api_key}"}
        timeout = None if settings.llm_timeout_seconds <= 0 else settings.llm_timeout_seconds

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "stream": True,
        }
        async with httpx.AsyncClient(base_url=settings.llm_api_base, timeout=timeout) as client:
            async with client.stream("POST", "/chat/completions", headers=headers, json=payload) as resp:
                try:
                    resp.raise_for_status()
                except httpx.HTTPStatusError as exc:
                    status_code = exc.response.status_code if exc.response else None
                    response_text = None
                    if exc.response is not None:
                        try:
                            response_text = _truncate_text(exc.response.text)
                        except Exception:
                            try:
                                response_text = _truncate_text(
                                    exc.response.content.decode("utf-8", errors="replace")
                                )
                            except Exception:
                                response_text = None
                    request_url = None
                    if exc.response is not None and exc.response.request is not None:
                        request_url = str(exc.response.request.url)
                    err_msg = (
                        f"Chat stream HTTP {status_code} model={self.model} url={request_url} body={response_text}"
                    )
                    logger.error("chat stream failed: %s", err_msg)
                    raise ChatError(
                        err_msg,
                        status_code=status_code,
                        response_text=response_text,
                    ) from exc
                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    if line.startswith(("event:", "id:", ":")):
                        continue
                    data = line
                    if data.startswith("data:"):
                        data = data[5:].strip()
                    if not data:
                        continue
                    if data.strip() == "[DONE]":
                        break
                    try:
                        obj = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    token = self._extract_stream_token(obj)
                    if token:
                        yield token

    def _fallback(self, messages: List[Dict[str, str]]) -> str:
        """Deterministic local fallback for tests."""
        last_user = next((m for m in reversed(messages) if m["role"] == "user"), {"content": ""})
        system = next((m for m in messages if m["role"] == "system"), {"content": ""})
        return json.dumps({"system": system.get("content", ""), "echo": last_user.get("content", "")}, ensure_ascii=False)

    def _record_usage(self, messages: List[Dict[str, str]], content: str, usage: Dict[str, Any] | None) -> None:
        if not self.stats:
            return
        prompt_tokens = None
        completion_tokens = None
        if isinstance(usage, dict):
            prompt_tokens = usage.get("prompt_tokens")
            completion_tokens = usage.get("completion_tokens")
        if prompt_tokens is None:
            prompt_tokens = estimate_message_tokens(messages)
        if completion_tokens is None:
            completion_tokens = estimate_tokens(content)
        self.stats.record_llm_call(
            model=self.model,
            prompt_tokens=int(prompt_tokens),
            completion_tokens=int(completion_tokens),
        )

    def _extract_stream_token(self, obj: Any) -> str | None:
        if not isinstance(obj, dict):
            return None
        candidates = obj.get("candidates")
        if isinstance(candidates, list) and candidates:
            content = candidates[0].get("content")
            if isinstance(content, dict):
                parts_text = content.get("partsText")
                if parts_text:
                    return str(parts_text)
                parts = content.get("parts") or []
                tokens: List[str] = []
                for part in parts:
                    if isinstance(part, str):
                        tokens.append(part)
                    elif isinstance(part, dict):
                        text = part.get("text")
                        if text:
                            tokens.append(str(text))
                if tokens:
                    return "".join(tokens)
            if isinstance(content, str) and content:
                return content
        for key in ("text", "output_text"):
            val = obj.get(key)
            if isinstance(val, str) and val:
                return val
        choices = obj.get("choices")
        if isinstance(choices, list) and choices:
            choice = choices[0] or {}
            delta = choice.get("delta") or choice.get("message") or {}
            if isinstance(delta, dict):
                token = delta.get("content") or delta.get("text") or delta.get("token")
                if token:
                    return str(token)
            if isinstance(delta, str) and delta:
                return delta
            token = choice.get("text") or choice.get("content")
            if token:
                return str(token)
        delta = obj.get("delta")
        if isinstance(delta, dict):
            token = delta.get("token")
            if token:
                return str(token)
        for key in ("output_text", "text", "content"):
            val = obj.get(key)
            if isinstance(val, str) and val:
                return val
        return None
