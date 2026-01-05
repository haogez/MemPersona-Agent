from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any, AsyncGenerator, Dict, List

import httpx

from mem_persona_agent.config import settings
from mem_persona_agent.utils import StatsCollector, estimate_message_tokens, estimate_tokens

logger = logging.getLogger(__name__)

_SYSTEM_GUARD = "你必须直接给出最终答案。不要展示思考过程、推理步骤或中间分析。不要输出 <think> 或 </think>。"


class ChatError(RuntimeError):
    def __init__(self, message: str, status_code: int | None = None, response_text: str | None = None):
        super().__init__(message)
        self.status_code = status_code
        self.response_text = response_text


class ChatClient:
    """Async chat completion client with OpenAI compatible interface (vLLM HTTP)."""

    def __init__(self, model: str | None = None, stats: StatsCollector | None = None, model_path: str | None = None):
        self.model = model or settings.llm_model_name or "qwen3-14b"
        self.stats = stats
        self.base_url = settings.llm_api_base or "http://127.0.0.1:18000/v1"
        self.api_key = settings.llm_api_key or "dummy"

    async def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float | None = None,
        top_p: float | None = None,
        max_new_tokens: int | None = None,
    ) -> str:
        guarded = self._apply_system_guard(messages)
        temp_val = settings.llm_temperature if temperature is None else temperature
        top_p_val = settings.llm_top_p if top_p is None else top_p
        max_tokens_val = self._resolve_max_tokens(guarded, max_new_tokens)
        payload = {
            "model": self.model,
            "messages": guarded,
            "temperature": temp_val,
            "top_p": top_p_val,
            "max_tokens": max_tokens_val,
            "stream": settings.llm_stream_default,
        }
        headers = {"Authorization": f"Bearer {self.api_key}"}
        timeout = None if settings.llm_timeout_seconds <= 0 else settings.llm_timeout_seconds
        try:
            async with httpx.AsyncClient(base_url=self.base_url, timeout=timeout) as client:
                resp = await client.post("/chat/completions", json=payload, headers=headers)
                resp.raise_for_status()
                data = resp.json()
                content = data["choices"][0]["message"]["content"]
        except httpx.HTTPError as exc:
            status_code = exc.response.status_code if exc.response else None
            body = None
            try:
                body = exc.response.text if exc.response is not None else None
            except Exception:
                body = None
            raise ChatError(f"vLLM chat failed: {exc}", status_code=status_code, response_text=body) from exc
        content = self._strip_think_text(content)
        self._record_usage(guarded, content, usage=None)
        return content

    async def stream_chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float | None = None,
        top_p: float | None = None,
        max_new_tokens: int | None = None,
    ) -> AsyncGenerator[str, None]:
        """Streaming chat completion yielding tokens from vLLM HTTP."""
        guarded = self._apply_system_guard(messages)
        temp_val = settings.llm_temperature if temperature is None else temperature
        top_p_val = settings.llm_top_p if top_p is None else top_p
        max_tokens_val = self._resolve_max_tokens(guarded, max_new_tokens, is_stream=True)
        payload = {
            "model": self.model,
            "messages": guarded,
            "temperature": temp_val,
            "top_p": top_p_val,
            "max_tokens": max_tokens_val,
            "stream": True,
        }
        headers = {"Authorization": f"Bearer {self.api_key}"}
        timeout = None if settings.llm_timeout_seconds <= 0 else settings.llm_timeout_seconds

        filter_state = {"buffer": "", "in_think": False}
        collected: list[str] = []

        async with httpx.AsyncClient(base_url=self.base_url, timeout=timeout) as client:
            async with client.stream("POST", "/chat/completions", json=payload, headers=headers) as resp:
                try:
                    resp.raise_for_status()
                except httpx.HTTPError as exc:
                    status_code = exc.response.status_code if exc.response else None
                    body = None
                    try:
                        body = exc.response.text if exc.response is not None else None
                    except Exception:
                        body = None
                    raise ChatError(f"vLLM stream failed: {exc}", status_code=status_code, response_text=body) from exc
                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    if line.startswith(("event:", "id:", ":")):
                        continue
                    data_line = line[5:].strip() if line.startswith("data:") else line.strip()
                    if not data_line:
                        continue
                    if data_line == "[DONE]":
                        break
                    try:
                        obj = json.loads(data_line)
                    except json.JSONDecodeError:
                        continue
                    delta = None
                    try:
                        delta = obj["choices"][0]["delta"].get("content")
                    except Exception:
                        delta = None
                    if not delta:
                        continue
                    token = delta if isinstance(delta, str) else "".join(delta) if isinstance(delta, list) else None
                    if not token:
                        continue
                    for clean in self._strip_think_stream(token, state=filter_state):
                        collected.append(clean)
                        yield clean
        if collected:
            try:
                self._record_usage(guarded, "".join(collected), usage=None)
            except Exception:  # pragma: no cover - defensive
                logger.debug("failed to record usage for stream", exc_info=True)

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

    def _apply_system_guard(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        if not messages:
            return [{"role": "system", "content": _SYSTEM_GUARD}]
        injected = False
        new_messages: List[Dict[str, str]] = []
        for msg in messages:
            if msg.get("role") == "system":
                content = (msg.get("content") or "") + ("\n" if msg.get("content") else "") + _SYSTEM_GUARD
                new_messages.append({**msg, "content": content})
                injected = True
            else:
                new_messages.append(msg)
        if not injected:
            new_messages.insert(0, {"role": "system", "content": _SYSTEM_GUARD})
        return new_messages

    def _strip_think_text(self, text: str) -> str:
        if not text:
            return text
        return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)

    def _strip_think_stream(self, token: str, *, state: Dict[str, Any]) -> List[str]:
        state["buffer"] += token
        output: List[str] = []
        while state["buffer"]:
            if state.get("in_think"):
                end = state["buffer"].find("</think>")
                if end == -1:
                    state["buffer"] = state["buffer"][-8:]
                    break
                state["buffer"] = state["buffer"][end + len("</think>") :]
                state["in_think"] = False
                continue
            start = state["buffer"].find("<think>")
            if start == -1:
                output.append(state["buffer"])
                state["buffer"] = ""
                break
            if start > 0:
                output.append(state["buffer"][:start])
            state["buffer"] = state["buffer"][start + len("<think>") :]
            state["in_think"] = True
        return [t for t in output if t]

    def _resolve_max_tokens(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: int | None,
        is_stream: bool = False,
    ) -> int:
        if max_new_tokens is not None:
            return int(max_new_tokens)
        # Heuristic: persona/memory generation uses long budget; others use dialog budget.
        text = " ".join([m.get("content") or "" for m in messages if m.get("role") == "system"]).lower()
        persona_keywords = (
            "persona",
            "memory",
            "scene",
            "worldrule",
            "inspiration",
            "related",
            "detail",
        )
        if any(k in text for k in persona_keywords):
            return settings.llm_max_tokens_persona
        return settings.llm_max_tokens_dialog
