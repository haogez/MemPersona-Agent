from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

from tqdm import tqdm
from .utils import (
    DATA_DIR,
    OUTPUT_DIR,
    call_vllm,
    ensure_dirs,
    judge_equiv,
    load_timelite,
    normalize_answer,
    pick_fields,
    print_progress,
    strip_think,
)


def _scene_id(sample: object, idx: int) -> str:
    if isinstance(sample, dict):
        return str(sample.get("id") or sample.get("sample_id") or idx)
    try:
        return str(sample)
    except Exception:
        return f"{idx}"


def _get_field(sample: object, key: str, default: str = "") -> str:
    if isinstance(sample, dict):
        return str(sample.get(key, default))
    return default


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Full-context eval on TIME-Lite")
    parser.add_argument("--split", type=str, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)

    ensure_dirs()
    data = load_timelite(args.split)
    if args.limit:
        data = data.select(range(args.limit))
    sample0 = data[0]
    q_key, a_key, c_key = pick_fields(sample0)

    pred_path = OUTPUT_DIR / "predictions" / "fullcontext.jsonl"
    done = {}
    if args.resume and pred_path.exists():
        for line in pred_path.open("r", encoding="utf-8"):
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            done[str(rec.get("id"))] = rec
    out_f = pred_path.open("w", encoding="utf-8")

    correct = 0
    all_recs = []
    total = len(data)
    pbar = tqdm(total=len(data), desc="fullcontext")
    max_chars = 120_000  # approx <= 40k tokens
    for idx, sample in enumerate(data):
        sid = _scene_id(sample, idx)
        if sid in done:
            rec = done[sid]
            correct += 1 if rec.get("correct") else 0
            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            all_recs.append(rec)
            pbar.update(1)
            pbar.set_postfix({"acc": f"{correct/max(1,len(all_recs)):.4f}"})
            continue
        if not isinstance(sample, dict):
            pbar.update(1)
            continue
        question = _get_field(sample, q_key, "")
        context = _get_field(sample, c_key, "")
        if context:
            max_ctx_len = max_chars - len(question) - 500
            if max_ctx_len > 0 and len(context) > max_ctx_len:
                context = context[:max_ctx_len]
        user_msg = question
        if context:
            user_msg = f"【Context】\n{context}\n\n【Question】\n{question}"
        messages = [
            {
                "role": "system",
                "content": "你是问答助手。禁止输出思考/分析/解释。只输出最终答案：如果是选择题仅输出选项字母（如 A/B/C/D），否则输出最简短答案。不要输出<think>标签，不要多余文字。",
            },
            {"role": "user", "content": user_msg},
        ]
        try:
            pred_raw = call_vllm(
                messages,
                max_tokens=650,
                temperature=0.0,
                enable_thinking=False,   # ✅ 显式关闭 Qwen3 thinking
            )
        except Exception as exc:
            pred_raw = f"ERROR: {exc}"
        pred_clean = strip_think(pred_raw)
        pred_norm = normalize_answer(pred_clean)
        gold_text = _get_field(sample, a_key, "")
        is_correct = judge_equiv(question, pred_clean, gold_text)
        correct += 1 if is_correct else 0
        rec = {
            "id": sid,
            "question": question,
            "pred": pred_clean,
            "pred_raw": pred_raw,
            "pred_norm": pred_norm,
            "gold": gold_text,
            "correct": is_correct,
        }
        out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        all_recs.append(rec)
        pbar.update(1)
        pbar.set_postfix({"acc": f"{correct/max(1,len(all_recs)):.4f}"})
    out_f.close()
    acc = correct / total if total else 0.0
    metrics_path = OUTPUT_DIR / "metrics" / "fullcontext.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(
        json.dumps({"accuracy": acc, "total": total, "samples": all_recs}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\nFull-context accuracy: {acc:.4f}")


if __name__ == "__main__":
    main()
