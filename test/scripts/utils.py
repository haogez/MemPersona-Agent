from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import httpx
from datasets import load_dataset
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "test" / "data"
OUTPUT_DIR = ROOT / "test" / "outputs"


def ensure_dirs() -> None:
    for path in [
        DATA_DIR / "time_lite",
        DATA_DIR / "processed",
        OUTPUT_DIR / "logs",
        OUTPUT_DIR / "predictions",
        OUTPUT_DIR / "metrics",
    ]:
        path.mkdir(parents=True, exist_ok=True)


def load_timelite(split: str | None = None):
    ensure_dirs()
    cache_dir = DATA_DIR / "time_lite"
    ds = load_dataset("SylvainWei/TIME-Lite", cache_dir=str(cache_dir))
    if split and split in ds:
        return ds[split]
    # prefer test > validation > train > first
    for name in ("test", "validation", "train"):
        if name in ds:
            return ds[name]
    first = list(ds.keys())[0]
    return ds[first]


def pick_fields(sample: Dict[str, Any]) -> Tuple[str, str, str | None]:
    keys = list(sample.keys())
    # build a lower-case lookup to handle TIME-Lite capitalized fields such as "Question"/"Gold Answer"/"Context"
    lowered = {k.lower().replace(" ", "_"): k for k in keys}
    def find_key(candidates: Tuple[str, ...]) -> str | None:
        for cand in candidates:
            if cand in lowered:
                return lowered[cand]
            if cand in keys:
                return cand
        return None

    q_key = find_key(("question", "query", "prompt"))
    a_key = find_key(("gold_answer", "answer", "answers", "label"))
    c_key = find_key(("context", "passage", "dialogue", "story"))
    if q_key is None or a_key is None:
        raise ValueError(f"Cannot find question/answer fields in sample keys={keys}")
    return q_key, a_key, c_key


def normalize_answer(text: str) -> str:
    if text is None:
        return ""
    t = text.strip()
    # 优先提取类似 "A." "B:" 这样的选项字母
    m = re.match(r"\s*([A-Za-z])(?:[\s\.\):\-]|$)", t)
    if m:
        return m.group(1).lower()
    t = t.lower()
    t = re.sub(r"[\s\u3000]+", " ", t)
    t = re.sub(r"[.。]+$", "", t)
    t = t.strip()
    return t


def strip_think(text: str) -> str:
    """Remove <think>...</think> blocks and stray tags."""
    if text is None:
        return ""
    # remove full blocks
    t = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    # remove standalone tags
    t = re.sub(r"</?think>", "", t, flags=re.IGNORECASE)
    return t.strip()


def judge_equiv(question: str, candidate: str, gold: str) -> bool:
    """
    Ask the model to judge if candidate == gold (semantically), expect 'true' or 'false'.
    Falls back to normalized string equality if parsing fails.
    """
    sys_prompt = (
        "你是严格的判题器。判断候选答案是否等价于标准答案。"
        "只能输出 true 或 false，不要思考、不解释、不输出其他字符。"
    )
    user_prompt = "\n".join(
        [
            "【问题】",
            question.strip(),
            "【候选答案】",
            candidate.strip(),
            "【标准答案】",
            gold.strip(),
            "只输出 true 或 false。",
        ]
    )
    try:
        resp = call_vllm(
            [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=4,
            temperature=0.0,
            top_p=0.9,
        )
        resp_clean = strip_think(resp).strip().lower()
        if resp_clean in {"true", "false"}:
            return resp_clean == "true"
        if resp_clean.startswith("t"):
            return True
        if resp_clean.startswith("f"):
            return False
    except Exception:
        pass
    return normalize_answer(candidate) == normalize_answer(gold)


def scene_summary_from_sample(sample: Dict[str, Any]) -> str:
    for key in ("scene_gist", "summary_7whr", "summary", "context", "passage", "Context"):
        if key in sample and sample.get(key):
            return str(sample.get(key))
    return ""


def save_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def load_jsonl(path: Path) -> Dict[str, Dict[str, Any]]:
    if not path.exists():
        return {}
    records: Dict[str, Dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            rec_id = str(rec.get("id") or rec.get("sample_id") or len(records))
            records[rec_id] = rec
    return records


def hf_split_text(text: str) -> List[str]:
    if not text:
        return []
    # split by sentence end or newline
    parts = re.split(r"[。.!?？\n]+", text)
    return [p.strip() for p in parts if p and p.strip()]


def call_vllm(
    messages: List[Dict[str, str]],
    *,
    max_tokens: int = 512,
    temperature: float = 0.0,
    top_p: float = 0.9,
    enable_thinking: bool = False,   # ✅加这个
) -> str:
    base_url = os.getenv("LLM_API_BASE_URL", "http://127.0.0.1:18000/v1")
    api_key = os.getenv("LLM_API_KEY", "dummy")
    model = os.getenv("LLM_MODEL_NAME", "qwen3-8b")
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "stream": False,
        "chat_template_kwargs": {      # ✅加这个
            "enable_thinking": enable_thinking
        },
    }
    headers = {"Authorization": f"Bearer {api_key}"}
    with httpx.Client(base_url=base_url, timeout=60.0) as client:
        resp = client.post("/chat/completions", json=payload, headers=headers)
        resp.raise_for_status()
        data = resp.json()
        return str(data["choices"][0]["message"]["content"])



def running_accuracy(correct: int, total: int) -> float:
    return (correct / total) if total else 0.0


def print_progress(i: int, n: int, correct: int) -> None:
    acc = running_accuracy(correct, i)
    sys.stdout.write(f"\rProcessed {i}/{n} | acc={acc:.4f}")
    sys.stdout.flush()
