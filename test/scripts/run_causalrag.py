from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

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
from .graph_rag_utils import SceneGraph, event_token_overlap_score, load_graph_snapshot
from .rag_embeddings import tokenize
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
    "CausalRAG: 抽取因果/时序边，基于 seed + s 步扩展生成路径。"
    "可选 LLM 摘要（--enable_summary_llm）。"
)


def build_directed_edges(graph: SceneGraph) -> Dict[str, List[str]]:
    adjacency: Dict[str, List[str]] = {}
    for edge in graph.edges:
        rel = (edge.get("rel") or "").lower()
        if rel not in {"cause", "result", "precede", "next"}:
            continue
        adjacency.setdefault(edge["from_id"], []).append(edge["to_id"])
    return adjacency


def bfs_paths(adjacency: Dict[str, List[str]], seeds: List[str], steps: int) -> List[List[str]]:
    paths: List[List[str]] = []
    for seed in seeds:
        frontier = [[seed]]
        for _ in range(steps):
            next_frontier: List[List[str]] = []
            for path in frontier:
                last = path[-1]
                for nei in adjacency.get(last, []):
                    if nei in path:
                        continue
                    new_path = path + [nei]
                    next_frontier.append(new_path)
            frontier = next_frontier
            if not frontier:
                break
        paths.extend(frontier if frontier else [[seed]])
    # de-dup
    seen = set()
    uniq_paths = []
    for path in paths:
        key = tuple(path)
        if key in seen:
            continue
        seen.add(key)
        uniq_paths.append(path)
    return uniq_paths


def summarize_paths_with_llm(paths: List[List[str]], lookup: Dict[str, Dict[str, str]]) -> List[str]:
    summaries = []
    for path in paths:
        text = " -> ".join(lookup[eid]["text"] for eid in path if eid in lookup)
        messages = [
            {
                "role": "system",
                "content": "你是时间/因果链条摘要助手。输出 1 句话以内摘要，不要解释。",
            },
            {"role": "user", "content": f"请总结以下事件链：\n{text}"},
        ]
        try:
            resp = call_vllm(messages, max_tokens=128, temperature=0.0, enable_thinking=False)
        except Exception as exc:
            resp = f"ERROR: {exc}"
        summaries.append(strip_think(resp))
    return summaries


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="CausalRAG eval on TIME-Lite")
    parser.add_argument("--dataset", type=str, default="time_lite")
    parser.add_argument("--split", type=str, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model_card", type=str, default=None)
    parser.add_argument("--base_url", type=str, default=None)
    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--k", type=int, default=3, help="top-k seed events")
    parser.add_argument("--steps", type=int, default=2, help="expansion steps")
    parser.add_argument("--path_k", type=int, default=6, help="top-k paths to keep")
    parser.add_argument("--enable_summary_llm", action="store_true")
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

    model_tag = model_tag_from_card(args.model_card)
    pred_path, metrics_path = build_output_paths("causalrag", model_tag, args.split, args.out_dir)
    done: Dict[str, Dict[str, Any]] = load_jsonl(pred_path) if args.resume else {}
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    out_f = pred_path.open("w", encoding="utf-8")

    correct = 0
    all_recs: List[Dict[str, Any]] = []
    total = len(data)
    pbar = tqdm(total=total, desc="causalrag")

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
        graph = graphs.get(sid)
        context = ""
        extra: Dict[str, Any] = {
            "method_note": METHOD_NOTE,
            "k": args.k,
            "steps": args.steps,
            "path_k": args.path_k,
            "enable_summary_llm": args.enable_summary_llm,
        }
        if graph:
            q_tokens = tokenize(question)
            event_scores = [
                (event_token_overlap_score(q_tokens, ev.get("text", "")), ev["event_id"])
                for ev in graph.events
            ]
            event_scores = [item for item in event_scores if item[0] > 0]
            event_scores.sort(key=lambda x: x[0], reverse=True)
            seeds = [eid for _, eid in event_scores[: max(1, args.k)]]
            adjacency = build_directed_edges(graph)
            paths = bfs_paths(adjacency, seeds, args.steps)
            lookup = graph.event_lookup()
            path_scores: List[Tuple[float, List[str]]] = []
            for path in paths:
                score = sum(
                    event_token_overlap_score(q_tokens, lookup[eid]["text"])
                    for eid in path
                    if eid in lookup
                )
                path_scores.append((score, path))
            path_scores.sort(key=lambda x: x[0], reverse=True)
            top_paths = [p for _, p in path_scores[: max(1, args.path_k)]]
            if args.enable_summary_llm:
                summaries = summarize_paths_with_llm(top_paths, lookup)
                context_lines = [f"[Path {i+1}] {summaries[i]}" for i in range(len(summaries))]
            else:
                context_lines = [
                    "[Path {idx}] ".format(idx=i + 1)
                    + " -> ".join(lookup[eid]["text"] for eid in path if eid in lookup)
                    for i, path in enumerate(top_paths)
                ]
            context = "\n".join([line for line in context_lines if line.strip()])
            extra.update(
                {
                    "seed_events": seeds,
                    "paths": top_paths,
                }
            )
        context = trim_context(question, context)
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
            "extra": extra,
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
    print(f"\nCausalRAG accuracy: {acc:.4f}")


if __name__ == "__main__":
    main()
