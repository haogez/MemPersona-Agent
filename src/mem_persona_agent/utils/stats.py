from __future__ import annotations

import json
import logging
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

from mem_persona_agent.config import settings

logger = logging.getLogger(__name__)


DEFAULT_MODEL_PRICES: Dict[str, Dict[str, float]] = {
    # per 1k tokens (USD); override with MODEL_PRICES_JSON if needed
    "gpt-4.1-mini": {"input": 0.00015, "output": 0.00060},
}


def _load_model_prices() -> Dict[str, Dict[str, float]]:
    raw = getattr(settings, "model_prices_json", "") or os.getenv("MODEL_PRICES_JSON", "")
    if not raw:
        return dict(DEFAULT_MODEL_PRICES)
    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            return data
    except Exception:
        logger.warning("MODEL_PRICES_JSON invalid; using defaults")
    return dict(DEFAULT_MODEL_PRICES)


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    chinese = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    ascii_chars = sum(1 for ch in text if "a" <= ch.lower() <= "z" or "0" <= ch <= "9")
    other = max(0, len(text) - chinese - ascii_chars)
    return chinese + max(1, ascii_chars // 4) + max(0, other // 2)


def estimate_message_tokens(messages: Iterable[Dict[str, Any]]) -> int:
    return sum(estimate_tokens(m.get("content", "")) for m in messages if isinstance(m, dict))


@dataclass
class StageStats:
    name: str
    duration_ms: float = 0.0
    llm_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0
    retries: int = 0
    invalid_reasons: Dict[str, int] = field(default_factory=dict)
    notes: Dict[str, Any] = field(default_factory=dict)


class StatsCollector:
    def __init__(self, *, model_name: Optional[str] = None, prices: Optional[Dict[str, Dict[str, float]]] = None):
        self.model_name = model_name or settings.llm_model_name
        self.prices = prices or _load_model_prices()
        self.stages: Dict[str, StageStats] = {}
        self._stage_stack: List[str] = []

    @property
    def current_stage(self) -> Optional[str]:
        return self._stage_stack[-1] if self._stage_stack else None

    def _ensure_stage(self, name: str) -> StageStats:
        if name not in self.stages:
            self.stages[name] = StageStats(name=name)
        return self.stages[name]

    @contextmanager
    def stage(self, name: str):
        self._stage_stack.append(name)
        start = time.perf_counter()
        try:
            yield
        finally:
            duration_ms = (time.perf_counter() - start) * 1000.0
            self._ensure_stage(name).duration_ms += duration_ms
            self._stage_stack.pop()

    def record_retry(self, stage: Optional[str] = None) -> None:
        stage_name = stage or self.current_stage or "unknown"
        self._ensure_stage(stage_name).retries += 1

    def record_invalid(self, reason: str, *, stage: Optional[str] = None) -> None:
        stage_name = stage or self.current_stage or "unknown"
        stats = self._ensure_stage(stage_name)
        stats.invalid_reasons[reason] = stats.invalid_reasons.get(reason, 0) + 1

    def add_note(self, key: str, value: Any, *, stage: Optional[str] = None) -> None:
        stage_name = stage or self.current_stage or "unknown"
        self._ensure_stage(stage_name).notes[key] = value

    def record_llm_call(
        self,
        *,
        model: Optional[str],
        prompt_tokens: int,
        completion_tokens: int,
        stage: Optional[str] = None,
    ) -> None:
        stage_name = stage or self.current_stage or "unknown"
        stats = self._ensure_stage(stage_name)
        stats.llm_calls += 1
        stats.prompt_tokens += int(prompt_tokens)
        stats.completion_tokens += int(completion_tokens)
        total = int(prompt_tokens) + int(completion_tokens)
        stats.total_tokens += total
        stats.cost_usd += self._estimate_cost(model or self.model_name, int(prompt_tokens), int(completion_tokens))

    def _estimate_cost(self, model: str, prompt_tokens: int, completion_tokens: int) -> float:
        price = self.prices.get(model) or self.prices.get(self.model_name) or {}
        in_price = float(price.get("input", 0.0))
        out_price = float(price.get("output", 0.0))
        return (prompt_tokens / 1000.0) * in_price + (completion_tokens / 1000.0) * out_price

    def to_dict(self) -> Dict[str, Any]:
        output: Dict[str, Any] = {}
        for name, stats in self.stages.items():
            output[name] = {
                "duration_ms": round(stats.duration_ms, 2),
                "llm_calls": stats.llm_calls,
                "prompt_tokens": stats.prompt_tokens,
                "completion_tokens": stats.completion_tokens,
                "total_tokens": stats.total_tokens,
                "cost_usd": round(stats.cost_usd, 6),
                "retries": stats.retries,
                "invalid_reasons": stats.invalid_reasons,
                "notes": stats.notes,
            }
        return output

    def totals(self) -> Dict[str, Any]:
        totals = {
            "llm_calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "cost_usd": 0.0,
            "retries": 0,
        }
        for stats in self.stages.values():
            totals["llm_calls"] += stats.llm_calls
            totals["prompt_tokens"] += stats.prompt_tokens
            totals["completion_tokens"] += stats.completion_tokens
            totals["total_tokens"] += stats.total_tokens
            totals["cost_usd"] += stats.cost_usd
            totals["retries"] += stats.retries
        totals["cost_usd"] = round(totals["cost_usd"], 6)
        return totals

    def log_summary(self) -> None:
        for name, stats in self.stages.items():
            logger.info(
                "stage=%s duration_ms=%.2f calls=%s tokens=%s cost_usd=%.6f",
                name,
                stats.duration_ms,
                stats.llm_calls,
                stats.total_tokens,
                stats.cost_usd,
            )
