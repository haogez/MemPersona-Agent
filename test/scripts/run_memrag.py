from __future__ import annotations

import argparse
import json
import os
import re
from typing import Any, Dict, List

from concurrent.futures import ThreadPoolExecutor

from neo4j import GraphDatabase
from tqdm import tqdm

from .utils import (
    OUTPUT_DIR,
    call_vllm,
    ensure_dirs,
    judge_equiv,
    load_timelite,
    normalize_answer,
    pick_fields,
    scene_summary_from_sample,
    strip_think,  # 仍保留：你 utils 里已有也没关系，但本脚本会用自己的 remove_think_blocks
)

SCENE_LABEL = "TimeliteScene"
EVENT_LABEL = "TimeliteEvent"
DETAIL_LABEL = "TimeliteDetail"
MAX_MODEL_TOKENS = 40_960


# ----------------------------
# 0) Helper utils
# ----------------------------
def _scene_id_from_sample(sample: Any, idx: int) -> str:
    if isinstance(sample, dict):
        return str(sample.get("id") or sample.get("sample_id") or sample.get("idx") or f"S{idx:06d}")
    try:
        return str(sample)
    except Exception:
        return f"S{idx:06d}"


def _get_field(sample: Any, key: str, default: str = "") -> str:
    if isinstance(sample, dict):
        return str(sample.get(key, default))
    return default


def estimate_tokens(text: str) -> int:
    """粗估 token 数（约 4 字符 / token）。"""
    return max(1, len(text) // 4)


def trim_text_to_tokens(text: str, max_tokens: int) -> str:
    if estimate_tokens(text) <= max_tokens:
        return text
    return text[: max_tokens * 4]


def remove_think_blocks(text: str) -> str:
    """整段删除 <think>...</think> / <analysis>...</analysis>，并清理残留标签。"""
    if not text:
        return ""
    text = re.sub(r"(?is)<think>.*?</think>", "", text)
    text = re.sub(r"(?is)<analysis>.*?</analysis>", "", text)
    text = re.sub(r"(?is)</?think>", "", text)
    text = re.sub(r"(?is)</?analysis>", "", text)
    return text.strip()


def extract_final_answer(question: str, cleaned_text: str, *, max_chars: int = 650) -> str:
    if not cleaned_text:
        return ""

    ans = cleaned_text.strip()  

    # 选择题：抽 A/B/C/D（保持不变）
    if re.search(r"(?m)^\s*A\.", question) and re.search(r"(?m)^\s*B\.", question):
        m2 = re.search(r"\b([ABCD])\b", ans)
        if m2:
            return m2.group(1)

    # 非选择题：不要只取第一行，保留全文（去掉可能的前缀）
    ans = re.sub(r"(?i)^(answer|final answer)\s*:\s*", "", ans).strip()

    # 可选：压掉多余空白，但不强行变短
    ans = re.sub(r"\n{3,}", "\n\n", ans).strip()

    # 限制长度（你说最高约 600，就给 650 余量）
    if len(ans) > max_chars:
        ans = ans[:max_chars].rstrip()

    return ans


def tokenize(text: str) -> List[str]:
    if not text:
        return []
    return [t for t in re.findall(r"[A-Za-z0-9]+|[\u4e00-\u9fff]+", text.lower()) if t]


# ----------------------------
# 1) Neo4j retrieval
# ----------------------------
def query_events(driver, tokens: List[str], k: int, scene_id: str) -> List[Dict[str, Any]]:
    if not tokens:
        return []
    db = os.getenv("NEO4J_DB", "neo4j")
    with driver.session(database=db) as session:
        result = session.run(
            f"""
            WITH $tokens AS toks
            MATCH (s:{SCENE_LABEL} {{scene_id:$sid}})-[:HAS_EVENT]->(e:{EVENT_LABEL})
            OPTIONAL MATCH (d:{DETAIL_LABEL})-[:INVOLVES]->(e)
            WITH e,
                 REDUCE(s=0, t IN toks | s + CASE WHEN toLower(e.text) CONTAINS t THEN 1 ELSE 0 END) AS text_hits,
                 REDUCE(s=0, t IN toks | s + CASE WHEN toLower(coalesce(d.text,'')) CONTAINS t THEN 1 ELSE 0 END) AS detail_hits
            WHERE text_hits > 0 OR detail_hits > 0
            WITH e, text_hits, detail_hits, (detail_hits*2 + text_hits) AS score
            RETURN e.event_id AS event_id, e.scene_id AS scene_id, e.text AS text, e.order_index AS order_index,
                   text_hits, detail_hits, score
            ORDER BY detail_hits DESC, score DESC
            LIMIT $k
            """,
            {"tokens": tokens, "k": k, "sid": scene_id},
        )
        return [dict(record) for record in result]


def fetch_event_details(driver, event_ids: List[str]) -> Dict[str, List[Dict[str, str]]]:
    if not event_ids:
        return {}
    db = os.getenv("NEO4J_DB", "neo4j")
    with driver.session(database=db) as session:
        res = session.run(
            f"""
            MATCH (d:{DETAIL_LABEL})-[:INVOLVES]->(e:{EVENT_LABEL})
            WHERE e.event_id IN $ids
            RETURN e.event_id AS eid, d.text AS text, d.dtype AS dtype
            """,
            {"ids": event_ids},
        )
        out: Dict[str, List[Dict[str, str]]] = {}
        for r in res:
            eid = r["eid"]
            out.setdefault(eid, []).append({"text": r["text"], "dtype": r["dtype"]})
    return out


def fetch_scene_graph(driver, scene_id: str):
    """拉取该 scene 的事件及其 CAUSES/NARRATES 边。"""
    db = os.getenv("NEO4J_DB", "neo4j")
    with driver.session(database=db) as session:
        ev_res = session.run(
            f"""
            MATCH (s:{SCENE_LABEL} {{scene_id:$sid}})-[:HAS_EVENT]->(e:{EVENT_LABEL})
            RETURN e.event_id AS event_id, e.text AS text, e.order_index AS order_index, coalesce(e.time,"") AS time
            """,
            {"sid": scene_id},
        )
        events = {
            r["event_id"]: {
                "event_id": r["event_id"],
                "text": r["text"],
                "order_index": r["order_index"],
                "time": r["time"],
            }
            for r in ev_res
        }
        edge_res = session.run(
            f"""
            MATCH (a:{EVENT_LABEL})-[r:CAUSES]->(b:{EVENT_LABEL})
            WHERE a.scene_id=$sid AND b.scene_id=$sid
            RETURN a.event_id AS from_id, b.event_id AS to_id, coalesce(r.rel_type,'cause') AS rel
            UNION
            MATCH (a:{EVENT_LABEL})-[r:NARRATES]->(b:{EVENT_LABEL})
            WHERE a.scene_id=$sid AND b.scene_id=$sid
            RETURN a.event_id AS from_id, b.event_id AS to_id, coalesce(r.rel_type,'precede') AS rel
            """,
            {"sid": scene_id},
        )
        edges = [dict(r) for r in edge_res]
    return events, edges


def expand_neighbors(
    events: Dict[str, Dict[str, Any]],
    edges: List[Dict[str, str]],
    seed_ids: List[str],
    depth: int,
) -> List[str]:
    """BFS 按因果/顺序边扩展，返回包含 seed 的节点集合列表（保序去重）。"""
    if depth <= 0:
        return []
    graph: Dict[str, List[str]] = {}
    for e in edges:
        a = e["from_id"]
        b = e["to_id"]
        graph.setdefault(a, []).append(b)
        graph.setdefault(b, []).append(a)
    visited = set(seed_ids)
    frontier = list(seed_ids)
    for _ in range(depth):
        nxt = []
        for node in frontier:
            for nei in graph.get(node, []):
                if nei not in visited:
                    visited.add(nei)
                    nxt.append(nei)
        frontier = nxt
        if not frontier:
            break
    ordered = sorted(list(visited), key=lambda x: events.get(x, {}).get("order_index", 1e9))
    return ordered


# ----------------------------
# 2) Summaries (can think; we'll remove think anyway)
# ----------------------------
def summarize_events(events: List[Dict[str, Any]]) -> Dict[str, str]:
    """调用 LLM 对事件文本做简短摘要；允许 think，但最终抽取结果会去掉 think。"""
    summaries: Dict[str, str] = {}
    if not events:
        return summaries

    def _fallback(text: str) -> str:
        return trim_text_to_tokens(text or "", 64)

    def _build_messages(text: str):
        return [
            {
                "role": "system",
                "content": (
                    "You are a summarization agent.\n"
                    "You MAY think inside <think>...</think>.\n"
                    "Then output exactly one line: FINAL: <one short sentence summary>\n"
                    "No other text after FINAL."
                ),
            },
            {
                "role": "user",
                "content": f"Event:\n{text}\n\nReturn:\nFINAL: <one short sentence summary>",
            },
        ]

    with ThreadPoolExecutor(max_workers=8) as ex:
        futures = {}
        for ev in events:
            eid = ev.get("event_id")
            text = ev.get("text", "")
            if not eid or not text:
                continue
            # ✅ 注意：call_vllm 的 max_tokens/temperature 是 keyword-only
            fut = ex.submit(call_vllm, _build_messages(text), max_tokens=96, temperature=0.0)
            futures[fut] = (eid, text)

        for fut, (eid, text) in futures.items():
            try:
                raw = fut.result()
                no_think = remove_think_blocks(raw)
                summ = extract_final_answer("", no_think)  # 这里不需要选择题判断
                summaries[eid] = summ.strip() if summ else _fallback(text)
            except Exception:
                summaries[eid] = _fallback(text)

    return summaries


# ----------------------------
# 3) Prompt building (ALLOW THINK, require FINAL)
# ----------------------------
def build_prompt(
    question: str,
    top_events: List[Dict[str, Any]],
    neighbor_events: List[Dict[str, Any]],
    scene_summary: str,
    anchor_event_id: str | None,
    causal_edges: List[Dict[str, str]],
    event_details: Dict[str, List[Dict[str, str]]],
    summaries: Dict[str, str],
) -> List[Dict[str, str]]:
    max_tokens = MAX_MODEL_TOKENS - 512  # 留出回答空间

    top_lines = []
    for i, ev in enumerate(top_events):
        time_hint = ev.get("time") or ""
        summ = summaries.get(ev.get("event_id"), ev.get("text", ""))
        line = f"{i+1}. ({ev.get('event_id')}) [time={time_hint}] {summ}"
        dets = event_details.get(ev.get("event_id"), [])
        if dets:
            detail_txt = "; ".join([f"[{d.get('dtype','')}] {d.get('text','')}" for d in dets])
            line += f" | details: {detail_txt}"
        top_lines.append(line)

    neighbor_lines = []
    for ev in neighbor_events:
        time_hint = ev.get("time") or ""
        summ = summaries.get(ev.get("event_id"), ev.get("text", ""))
        neighbor_lines.append(f"({ev.get('event_id')}) [time={time_hint}] {summ}")

    causal_lines = [f"{edge['from_id']} -> {edge['to_id']}" for edge in causal_edges] if causal_edges else []
    anchor_line = f"Anchor event: {anchor_event_id or 'none'}; events are ordered by relevance."
    top_text = "\n".join(top_lines)
    neighbor_text = "\n".join(neighbor_lines)
    summary = scene_summary or ""

    base_parts = [
        "You are given TOP events (with details) and NEIGHBOR events (summaries only). Use ONLY these to answer.",
        "[TOP_EVENTS]",
        top_text,
        "[/TOP_EVENTS]",
        "[NEIGHBOR_EVENTS]",
        neighbor_text or "None",
        "[/NEIGHBOR_EVENTS]",
        "Causal links (cause/precede):",
        "\n".join(causal_lines) if causal_lines else "None",
        anchor_line,
        f"Scene summary: {summary}",
        f"Question: {question}",
        (
            "OUTPUT FORMAT (MUST FOLLOW):\n"
            "1) You MAY think, but put ALL reasoning inside <think>...</think>.\n"
            "2) Then output the answer DIRECTLY (no 'FINAL:' prefix).\n"
            "3) If multiple-choice: output exactly one letter A/B/C/D.\n"
            "4) Otherwise: output 1-6 sentences, up to ~600 characters.\n"
            "5) If cannot be determined from provided events: output EXACTLY 'UNKNOWN'.\n"
            "6) Do not output anything else."
        ),
    ]

    user_content = "\n".join(base_parts)

    # 长度保护：若超标，先截摘要，再截邻居，再截 top 文本
    if estimate_tokens(user_content) > max_tokens:
        summary = trim_text_to_tokens(summary, 3000)
        base_parts[10] = f"Scene summary: {summary}"
        user_content = "\n".join(base_parts)

    if estimate_tokens(user_content) > max_tokens:
        budget = max_tokens - estimate_tokens("\n".join(base_parts[:5] + base_parts[6:])) - 256
        if budget > 0:
            neighbor_text = trim_text_to_tokens(neighbor_text, budget)
            base_parts[5] = neighbor_text
            user_content = "\n".join(base_parts)

    if estimate_tokens(user_content) > max_tokens:
        budget = max_tokens - estimate_tokens("\n".join(base_parts[:2] + base_parts[3:])) - 256
        if budget > 0:
            top_text = trim_text_to_tokens(top_text, budget)
            base_parts[2] = top_text
            user_content = "\n".join(base_parts)

    if estimate_tokens(user_content) > max_tokens:
        user_content = trim_text_to_tokens(user_content, max_tokens)

    return [
        {
            "role": "system",
            "content": (
                "You are a QA agent.\n"
                "You MAY think and reason.\n\n"
                "MANDATORY OUTPUT FORMAT:\n"
                "- Put ALL reasoning inside <think>...</think>.\n"
                "- Then output the answer DIRECTLY (no 'FINAL:' prefix).\n"
                "- If multiple-choice: output exactly one letter A/B/C/D.\n"
                "- Otherwise: output up to ~600 characters.\n"
                "- If cannot be determined: output EXACTLY 'UNKNOWN'.\n"
                "- Do not output anything else."
            ),
        },
        {"role": "user", "content": user_content},
    ]


# ----------------------------
# 4) Main
# ----------------------------
def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Graph-RAG eval on TIME-Lite")
    parser.add_argument("--split", type=str, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--k", type=int, default=3, help="top-k events")
    parser.add_argument("--neighbor", type=int, default=0, help="neighbor hop depth along causal paths")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)

    ensure_dirs()
    data = load_timelite(args.split)
    if args.limit:
        data = data.select(range(args.limit))
    q_key, a_key, c_key = pick_fields(data[0])

    pred_path = OUTPUT_DIR / "predictions" / "memrag.jsonl"
    done: Dict[str, Dict[str, Any]] = {}
    if args.resume and pred_path.exists():
        for line in pred_path.open("r", encoding="utf-8"):
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            done[str(rec.get("id"))] = rec

    out_f = pred_path.open("w", encoding="utf-8")

    uri = os.getenv("NEO4J_URI")
    user = os.getenv("NEO4J_USER") or os.getenv("NEO4J_USERNAME")
    pwd = os.getenv("NEO4J_PASSWORD")
    if not uri or not user or not pwd:
        raise SystemExit("NEO4J_URI/NEO4J_USER/NEO4J_PASSWORD required")
    driver = GraphDatabase.driver(uri, auth=(user, pwd))

    correct = 0
    all_recs: List[Dict[str, Any]] = []
    total = len(data)
    pbar = tqdm(total=total, desc="memrag")

    for idx, sample in enumerate(data):
        sid = _scene_id_from_sample(sample, idx)

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
        if not question:
            pbar.update(1)
            continue

        tokens = tokenize(question)
        candidates = query_events(driver, tokens, args.k, sid)
        scene_summary = scene_summary_from_sample(sample)
        anchor_event_id = candidates[0]["event_id"] if candidates else None

        # scene graph
        scene_events_map, scene_edges = fetch_scene_graph(driver, sid)

        # top events list
        timeline: List[Dict[str, Any]] = []
        added = set()
        for cand in candidates:
            eid = cand["event_id"]
            if eid in added:
                continue
            added.add(eid)
            ev_obj = scene_events_map.get(eid, {})
            timeline.append(
                {
                    "event_id": eid,
                    "scene_id": cand["scene_id"],
                    "text": ev_obj.get("text", cand["text"]),
                    "order_index": ev_obj.get("order_index", cand.get("order_index", 0)),
                    "time": ev_obj.get("time", ""),
                }
            )

        event_ids = [ev["event_id"] for ev in timeline]

        # neighbor expansion
        neighbor_ids: List[str] = []
        if args.neighbor and scene_events_map:
            expanded = expand_neighbors(scene_events_map, scene_edges, event_ids, args.neighbor)
            neighbor_ids = [eid for eid in expanded if eid not in event_ids]

        neighbor_events = [
            {
                "event_id": eid,
                "scene_id": sid,
                "text": scene_events_map[eid]["text"],
                "order_index": scene_events_map[eid]["order_index"],
                "time": scene_events_map[eid].get("time", ""),
            }
            for eid in neighbor_ids
            if eid in scene_events_map
        ]

        # details only for top events
        event_details = fetch_event_details(driver, event_ids) if timeline else {}

        # causal edges only among included nodes
        include_ids = set(event_ids + neighbor_ids)
        causal_edges = [e for e in scene_edges if e["from_id"] in include_ids and e["to_id"] in include_ids]

        # summaries (top + neighbor)
        summaries = summarize_events(timeline + neighbor_events)

        messages = build_prompt(
            question,
            timeline,
            neighbor_events,
            scene_summary,
            anchor_event_id,
            causal_edges,
            event_details,
            summaries,
        )

        try:
            # ✅ 允许“思考很长”，但仍受服务端限制；这里给大一些
            pred_raw = call_vllm(messages, max_tokens=800, temperature=0.0, enable_thinking=False)
        except Exception as exc:
            pred_raw = f"ERROR: {exc}"

        # ✅ 评测/落盘：先整段删 think，再抽 FINAL
        pred_no_think = remove_think_blocks(pred_raw)
        pred_clean = extract_final_answer(question, pred_no_think, max_chars=650)


        gold_val = sample[a_key] if isinstance(sample, dict) else ""
        is_correct = judge_equiv(question, pred_clean, gold_val)
        correct += 1 if is_correct else 0

        rec = {
            "id": sid,
            "question": question,
            "pred": pred_clean,            # 用于 judge/metrics 的最终答案
            "pred_raw": pred_raw,          # 原始输出（可能包含<think>）
            "pred_no_think": pred_no_think,  # 删除 think 后文本（用于 debug）
            "gold": gold_val,
            "correct": is_correct,
            "timeline_used": [ev.get("event_id") for ev in timeline],
        }

        out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        all_recs.append(rec)
        pbar.update(1)
        pbar.set_postfix({"acc": f"{correct/max(1,len(all_recs)):.4f}"})

    out_f.close()

    acc = correct / total if total else 0.0
    metrics_path = OUTPUT_DIR / "metrics" / "memrag.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(
        json.dumps({"accuracy": acc, "total": total, "samples": all_recs}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\nGraph-RAG accuracy: {acc:.4f}")


if __name__ == "__main__":
    main()
