from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
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
from .graph_rag_utils import SceneGraph, load_graph_snapshot
from .rag_embeddings import hash_vectorize
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


METHOD_NOTE = (
    "GraphRAG-Global: 基于 timelite_graph.jsonl 构建全局摘要库，"
    "按问题检索 top-k 摘要拼接为上下文。"
)


def build_global_summaries(graphs: Dict[str, SceneGraph], summary_path: Path, *, rebuild: bool) -> List[Dict[str, Any]]:
    if summary_path.exists() and not rebuild:
        summaries = []
        with summary_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                summaries.append(json.loads(line))
        return summaries
    summaries: List[Dict[str, Any]] = []
    for sid, graph in graphs.items():
        events = sorted(graph.events, key=lambda ev: ev.get("order_index", 1e9))
        top_events = events[: min(6, len(events))]
        summary_text = " / ".join(ev.get("text", "") for ev in top_events if ev.get("text"))
        summaries.append({"scene_id": sid, "summary": summary_text, "event_ids": [ev["event_id"] for ev in top_events]})
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as f:
        for rec in summaries:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return summaries


def rank_summaries(question: str, summaries: List[Dict[str, Any]], top_k: int) -> List[Tuple[str, float, str]]:
    q_vec = hash_vectorize(question)
    scored: List[Tuple[str, float, str]] = []
    for rec in summaries:
        summ = rec.get("summary") or ""
        vec = hash_vectorize(summ)
        score = float(np.dot(q_vec, vec))
        scored.append((rec["scene_id"], score, summ))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[: max(1, top_k)]


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="GraphRAG-Global eval on TIME-Lite")
    parser.add_argument("--dataset", type=str, default="time_lite")
    parser.add_argument("--split", type=str, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model_card", type=str, default=None)
    parser.add_argument("--base_url", type=str, default=None)
    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--k", type=int, default=3, help="top-k summaries")
    parser.add_argument("--rebuild_index", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--graph_path", type=str, default=str(DATA_DIR / "processed" / "timelite_graph.jsonl"))
    args = parser.parse_args(argv)

    configure_llm_env(args.model_card, args.base_url, args.api_key)
    set_seed(args.seed)
    ensure_dirs()

    data = load_timelite(args.split)
    limit = args.max_samples or args.limit
    if limit:
        data = data.select(range(limit))
    q_key, a_key, c_key = pick_fields(data[0])

    graphs = load_graph_snapshot(Path(args.graph_path))
    summary_path = DATA_DIR / "processed" / "graphrag_global_summaries.jsonl"
    summaries = build_global_summaries(graphs, summary_path, rebuild=args.rebuild_index)

    model_tag = model_tag_from_card(args.model_card)
    pred_path, metrics_path = build_output_paths("graphrag_global", model_tag, args.split, args.out_dir)
    done: Dict[str, Dict[str, Any]] = load_jsonl(pred_path) if args.resume else {}
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    out_f = pred_path.open("w", encoding="utf-8")

    correct = 0
    all_recs: List[Dict[str, Any]] = []
    total = len(data)
    pbar = tqdm(total=total, desc="graphrag_global")

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
        ranked = rank_summaries(question, summaries, args.k)
        if sid not in [r[0] for r in ranked] and sid in graphs:
            own_summary = next((s for s in summaries if s["scene_id"] == sid), None)
            if own_summary:
                ranked = ranked[:-1] + [(sid, 0.0, own_summary.get("summary", ""))]
        context_lines = [f"[Scene {scene_id}] {summary}" for scene_id, _, summary in ranked if summary]
        context = trim_context(question, "\n".join(context_lines))

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
                "method_note": METHOD_NOTE,
                "summaries": [{"scene_id": scene_id, "score": score} for scene_id, score, _ in ranked],
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
    print(f"\nGraphRAG-Global accuracy: {acc:.4f}")


if __name__ == "__main__":
    main()
