from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from tqdm import tqdm

from .eval_utils import (
    build_messages,
    build_output_paths,
    configure_llm_env,
    get_field,
    model_tag_from_card,
    scene_id_from_sample,
    set_seed,
    trim_context,
)
from .regular_rag_impl import RegularRagIndex
from .task_accuracy import compute_accuracy
from .utils import (
    DATA_DIR,
    call_vllm,
    ensure_dirs,
    judge_equiv,
    load_jsonl,
    load_timelite,
    normalize_answer,
    pick_fields,
    strip_think,
)


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Regular RAG eval on TIME-Lite")
    parser.add_argument("--dataset", type=str, default="time_lite")
    parser.add_argument("--split", type=str, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model_card", type=str, default=None)
    parser.add_argument("--base_url", type=str, default=None)
    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--chunk_tokens", type=int, default=220)
    parser.add_argument("--dim", type=int, default=1024)
    parser.add_argument("--rebuild_index", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)

    configure_llm_env(args.model_card, args.base_url, args.api_key)
    set_seed(args.seed)
    ensure_dirs()

    data = load_timelite(args.split)
    limit = args.max_samples or args.limit
    if limit:
        data = data.select(range(limit))
    q_key, a_key, c_key = pick_fields(data[0])

    cache_path = DATA_DIR / "processed" / "regular_rag_index.pkl"
    index = RegularRagIndex.load_or_build(
        data,
        c_key,
        cache_path,
        dim=args.dim,
        chunk_tokens=args.chunk_tokens,
        rebuild=args.rebuild_index,
    )

    model_tag = model_tag_from_card(args.model_card)
    pred_path, metrics_path = build_output_paths("regular_rag", model_tag, args.split, args.out_dir)
    done: Dict[str, Dict[str, Any]] = load_jsonl(pred_path) if args.resume else {}
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    out_f = pred_path.open("w", encoding="utf-8")

    correct = 0
    all_recs: List[Dict[str, Any]] = []
    total = len(data)
    pbar = tqdm(total=total, desc="regular_rag")

    for idx, sample in enumerate(data):
        sid = scene_id_from_sample(sample, idx)
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

        question = get_field(sample, q_key, "")
        gold_text = get_field(sample, a_key, "")

        hits = index.search(question, sid, args.k)
        context_parts = [hit[2] for hit in hits]
        context = trim_context(question, "\n\n".join(context_parts))

        messages = build_messages(question, context)
        try:
            pred_raw = call_vllm(messages, max_tokens=650, temperature=0.0, enable_thinking=False)
        except Exception as exc:
            pred_raw = f"ERROR: {exc}"
        pred_clean = strip_think(pred_raw)
        pred_norm = normalize_answer(pred_clean)
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
            "extra": {
                "retrieval": [{"chunk_id": idx, "score": score, "text": text[:500]} for idx, score, text in hits],
                "k": args.k,
            },
        }
        out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        all_recs.append(rec)
        pbar.update(1)
        pbar.set_postfix({"acc": f"{correct/max(1,len(all_recs)):.4f}"})

    out_f.close()
    acc = compute_accuracy(all_recs)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(
        json.dumps({"accuracy": acc, "total": total, "samples": all_recs}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\nRegular RAG accuracy: {acc:.4f}")


if __name__ == "__main__":
    main()
