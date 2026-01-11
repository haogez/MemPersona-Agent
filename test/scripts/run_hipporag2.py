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
    "HippoRAG2: seed events + 简化 triples/path ranking。"
    "先按关键词选 seed，再对邻接边构造 triples 并排序。"
)


def collect_triples(graph: SceneGraph) -> List[Tuple[str, str, str]]:
    lookup = graph.event_lookup()
    triples = []
    for edge in graph.edges:
        a = lookup.get(edge["from_id"])
        b = lookup.get(edge["to_id"])
        if not a or not b:
            continue
        triples.append((a.get("text", ""), edge.get("rel", ""), b.get("text", "")))
    return triples


def rank_triples(
    question_tokens: List[str],
    seed_ids: List[str],
    graph: SceneGraph,
    triples: List[Tuple[str, str, str]],
) -> List[Tuple[float, Tuple[str, str, str]]]:
    lookup = graph.event_lookup()
    seed_texts = {lookup[sid]["text"] for sid in seed_ids if sid in lookup}
    scored: List[Tuple[float, Tuple[str, str, str]]] = []
    for triple in triples:
        text = " ".join(triple)
        score = event_token_overlap_score(question_tokens, text)
        if score <= 0:
            continue
        seed_bonus = 0.0
        if triple[0] in seed_texts or triple[2] in seed_texts:
            seed_bonus = 0.5
        scored.append((score + seed_bonus, triple))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="HippoRAG2 eval on TIME-Lite")
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
    parser.add_argument("--path_k", type=int, default=8, help="top-k triples/paths")
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
    pred_path, metrics_path = build_output_paths("hipporag2", model_tag, args.split, args.out_dir)
    done: Dict[str, Dict[str, Any]] = load_jsonl(pred_path) if args.resume else {}
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    out_f = pred_path.open("w", encoding="utf-8")

    correct = 0
    all_recs: List[Dict[str, Any]] = []
    total = len(data)
    pbar = tqdm(total=total, desc="hipporag2")

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
        extra: Dict[str, Any] = {"method_note": METHOD_NOTE, "k": args.k, "path_k": args.path_k}
        if graph:
            q_tokens = tokenize(question)
            event_scores = [
                (event_token_overlap_score(q_tokens, ev.get("text", "")), ev["event_id"])
                for ev in graph.events
            ]
            event_scores = [item for item in event_scores if item[0] > 0]
            event_scores.sort(key=lambda x: x[0], reverse=True)
            seeds = [eid for _, eid in event_scores[: max(1, args.k)]]
            triples = collect_triples(graph)
            ranked = rank_triples(q_tokens, seeds, graph, triples)
            top_triples = ranked[: max(1, args.path_k)]
            context_lines = [f"{t[0]} --{t[1]}--> {t[2]}" for _, t in top_triples]
            context = "\n".join([line for line in context_lines if line.strip()])
            extra.update(
                {
                    "seed_events": seeds,
                    "triples": [{"score": score, "text": f"{t[0]} --{t[1]}--> {t[2]}"} for score, t in top_triples],
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
    print(f"\nHippoRAG2 accuracy: {acc:.4f}")


if __name__ == "__main__":
    main()
