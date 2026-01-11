from __future__ import annotations

import os
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from .utils import OUTPUT_DIR

DEFAULT_SYSTEM_PROMPT = (
    "你是问答助手。禁止输出思考/分析/解释。只输出最终答案："
    "如果是选择题仅输出选项字母（如 A/B/C/D），否则输出最简短答案。"
    "不要输出<think>标签，不要多余文字。"
)
DEFAULT_MAX_CONTEXT_CHARS = 120_000


def configure_llm_env(model_card: str | None, base_url: str | None, api_key: str | None) -> None:
    if base_url:
        os.environ["LLM_API_BASE_URL"] = base_url
    if api_key:
        os.environ["LLM_API_KEY"] = api_key
    if model_card:
        os.environ["LLM_MODEL_NAME"] = model_card
        os.environ["CHAT_MODEL_NAME"] = model_card


def set_seed(seed: int | None) -> None:
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)


def model_tag_from_card(model_card: str | None) -> str:
    raw = (model_card or os.getenv("LLM_MODEL_NAME") or "model").strip()
    if not raw:
        raw = "model"
    tag = re.sub(r"[^A-Za-z0-9]+", "_", raw).strip("_").lower()
    return tag or "model"


def split_tag(split: str | None) -> str:
    return split or "default"


def build_output_paths(
    method_name: str,
    model_tag: str,
    split: str | None,
    out_dir: str | None,
) -> Tuple[Path, Path]:
    base = Path(out_dir) if out_dir else OUTPUT_DIR
    pred_path = base / "predictions" / f"{method_name}_{model_tag}_{split_tag(split)}.jsonl"
    metrics_path = base / "metrics" / f"{method_name}_{model_tag}_{split_tag(split)}.json"
    return pred_path, metrics_path


def scene_id_from_sample(sample: Any, idx: int) -> str:
    if isinstance(sample, dict):
        return str(sample.get("id") or sample.get("sample_id") or sample.get("idx") or f"S{idx:06d}")
    try:
        return str(sample)
    except Exception:
        return f"S{idx:06d}"


def get_field(sample: Any, key: str, default: str = "") -> str:
    if isinstance(sample, dict):
        return str(sample.get(key, default))
    return default


def trim_context(question: str, context: str, max_chars: int = DEFAULT_MAX_CONTEXT_CHARS) -> str:
    if not context:
        return ""
    max_ctx_len = max_chars - len(question) - 500
    if max_ctx_len > 0 and len(context) > max_ctx_len:
        return context[:max_ctx_len]
    return context


def build_messages(question: str, context: str) -> List[Dict[str, str]]:
    user_msg = question
    if context:
        user_msg = f"【Context】\n{context}\n\n【Question】\n{question}"
    return [
        {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]
