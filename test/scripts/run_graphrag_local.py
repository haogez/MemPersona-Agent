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
from .graph_rag_utils import (
    SceneGraph,
    detail_token_overlap_score,
    event_token_overlap_score,
    expand_neighbors,
    load_graph_snapshot,
)
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
    "GraphRAG-Local: 基于 timelite_graph.jsonl 做局部子图扩展，"
    "seed=问题关键词匹配事件，邻居扩展后拼接为上下文。"
)


def select_seed_events(graph: SceneGraph, question_tokens: List[str], k: int) -> List[str]:
    scored = []
    for ev in graph.events:
        score = event_token_overlap_score(question_tokens, ev.get("text", ""))
        if score > 0:
            scored.append((score, ev["event_id"]))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [eid for _, eid in scored[: max(1, k)]]


def build_context(graph: SceneGraph, selected_ids: List[str]) -> str:
    lookup = graph.event_lookup()
    ordered = sorted(selected_ids, key=lambda eid: lookup.get(eid, {}).get("order_index", 1e9))
    lines = []
    for eid in ordered:
        ev = lookup.get(eid)
        if not ev:
            continue
        time_hint = ev.get("time") or ""
        prefix = f"[{time_hint}] " if time_hint else ""
        lines.append(f"{prefix}{ev.get('text','')}")
    return "\n".join([l for l in lines if l.strip()])


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="GraphRAG-Local eval on TIME-Lite")
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
    parser.add_argument("--neighbor", type=int, default=1, help="neighbor expansion depth")
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
    pred_path, metrics_path = build_output_paths("graphrag_local", model_tag, args.split, args.out_dir)
    done: Dict[str, Dict[str, Any]] = load_jsonl(pred_path) if args.resume else {}
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    out_f = pred_path.open("w", encoding="utf-8")

    correct = 0
    all_recs: List[Dict[str, Any]] = []
    total = len(data)
    pbar = tqdm(total=total, desc="graphrag_local")

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
        extra: Dict[str, Any] = {"method_note": METHOD_NOTE, "k": args.k, "neighbor": args.neighbor}
        if graph:
            q_tokens = tokenize(question)
            seeds = select_seed_events(graph, q_tokens, args.k)
            expanded = expand_neighbors(graph.edges, seeds, args.neighbor)
            selected = list(dict.fromkeys(seeds + expanded))
            context = build_context(graph, selected)
            detail_hits = []
            for det in graph.details:
                score = detail_token_overlap_score(q_tokens, det.get("text", ""))
                if score > 0:
                    detail_hits.append({"score": score, "text": det.get("text", "")})
            detail_hits.sort(key=lambda x: x["score"], reverse=True)
            extra.update(
                {
                    "seed_events": seeds,
                    "selected_events": selected,
                    "detail_hits": detail_hits[:5],
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
    print(f"\nGraphRAG-Local accuracy: {acc:.4f}")


if __name__ == "__main__":
    main()
