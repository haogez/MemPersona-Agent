from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Set, Tuple

from neo4j import GraphDatabase
from tqdm import tqdm

from .utils import DATA_DIR, call_vllm, ensure_dirs, hf_split_text, load_timelite, pick_fields


SCENE_LABEL = "TimeliteScene"
EVENT_LABEL = "TimeliteEvent"
DETAIL_LABEL = "TimeliteDetail"
MAX_MODEL_TOKENS = 40_960
CHUNK_TOKEN_LIMIT = 20_000  # 单段上限，估算 token 后分段
LOG_PATH = DATA_DIR / "processed" / "build_graph_calls.jsonl"
LOG_CALLS = os.getenv("BUILD_GRAPH_LOG_CALLS", "1") == "1"


def strip_think(content: str) -> str:
    """去除 <think>...</think> 包裹的内容，便于 JSON 解析。"""
    return re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL | re.IGNORECASE)


def estimate_tokens(text: str) -> int:
    # 粗估 token 数，避免超长（约 4 字符/token）
    return max(1, len(text) // 4)


def split_context_into_chunks(text: str, max_tokens: int = CHUNK_TOKEN_LIMIT) -> List[str]:
    """
    若上下文过长，按句号/换行切分，拼接成若干段，每段不超过 max_tokens（估算）。
    不截断句子，若单句超长则单独作为一段。
    """
    if estimate_tokens(text) <= max_tokens:
        return [text]
    # 按句子切分
    sentences = hf_split_text(text)
    if len(sentences) <= 1:
        return [text]
    chunks: List[str] = []
    cur: List[str] = []
    cur_tokens = 0
    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue
        sent_tokens = estimate_tokens(sent)
        if cur and cur_tokens + sent_tokens > max_tokens:
            chunks.append(" ".join(cur))
            cur = [sent]
            cur_tokens = sent_tokens
        else:
            cur.append(sent)
            cur_tokens += sent_tokens
    if cur:
        chunks.append(" ".join(cur))
    return chunks or [text]


def trim_text_to_tokens(text: str, max_tokens: int, reserve_tokens: int = 0) -> str:
    """按句子裁剪，使估算 token 不超过 max_tokens-reserve_tokens，保留句子完整。"""
    limit = max_tokens - reserve_tokens
    if estimate_tokens(text) <= limit:
        return text
    sentences = hf_split_text(text)
    acc: List[str] = []
    total = 0
    for s in sentences:
        t = estimate_tokens(s)
        if total + t > limit:
            break
        acc.append(s)
        total += t
    if acc:
        return " ".join(acc)
    # 若单句即超长，退化为硬截取
    return text[: limit * 4]


def sentence_events(text: str) -> List[str]:
    # split by sentence end and filter short
    return hf_split_text(text)


def dialogue_events(text: str) -> List[str]:
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if len(lines) <= 1:
        return sentence_events(text)
    return lines


def llm_extract_events(text: str, question: str | None, max_events: int = 8) -> Tuple[List[Dict[str, Any]], List[Tuple[int, int]]]:
    """
    Use vLLM to extract events with time order and causal links.
    Returns (events, causal_pairs) where events=[{text,time_hint}], causal_pairs=[(from_idx,to_idx)].
    """
    if not text or not text.strip():
        return [], []
    user_lines = [
        "请从下方上下文抽取按真实时间顺序排列的事件链，保留因果/先后关系。",
        f"至少 3 条、最多 {max_events} 条事件；每条直接截取原文中的句子或短语，不改写、不压缩，不得把整段原文合并成一条。",
        "事件必须源自原文，不要编造新信息；尽量复用原词/实体，保留关键动作/时间/地点。",
        "为每条事件生成 JSON 对象: {\"text\":\"...\",\"time\":\"可选时间点\"}，按时间顺序排列到 events 数组中。",
        "如果存在明确因果关系，输出 causal 数组，元素形如 {\"from\":事件序号(从1起),\"to\":事件序号,\"type\":\"cause/result\"}，没有则为空数组。",
        "严格禁止输出 <think> 或解释，最终只返回 JSON 对象: {\"events\":[...],\"causal\":[...]}。",
    ]
    if question:
        user_lines.append(f"问题: {question}")
    user_lines.append("上下文:\n" + text)
    user_lines.append('输出示例: {"events":[{"text":"[T1] 事件A","time":"2015年"},{"text":"[T2] 事件B","time":""}],"causal":[{"from":1,"to":2,"type":"cause"}]}')
    messages = [
        {
            "role": "system",
            "content": "你是时间顺序抽取助手。禁止输出 <think>、禁止解释，只输出 JSON 对象，保持时间/因果顺序。",
        },
        {"role": "user", "content": "\n".join(user_lines)},
    ]
    try:
        raw = call_vllm(messages, max_tokens=512, temperature=0.0, top_p=0.9)
    except Exception:
        return [], []
    events: List[Dict[str, Any]] = []
    causal: List[Tuple[int, int]] = []
    try:
        obj = json.loads(raw.strip())
        evs = obj.get("events") if isinstance(obj, dict) else None
        if isinstance(evs, list):
            for ev in evs:
                if not isinstance(ev, dict):
                    continue
                txt = str(ev.get("text") or "").strip()
                if not txt:
                    continue
                events.append({"text": txt, "time": str(ev.get("time") or "").strip()})
        c_list = obj.get("causal") if isinstance(obj, dict) else None
        if isinstance(c_list, list):
            for c in c_list:
                if not isinstance(c, dict):
                    continue
                try:
                    a = int(c.get("from"))
                    b = int(c.get("to"))
                    causal.append((a - 1, b - 1))
                except Exception:
                    continue
    except Exception:
        pass
    if not events:
        # 尝试解析为简单数组
        try:
            arr = json.loads(raw.strip())
            if isinstance(arr, list):
                events = [{"text": str(x), "time": ""} for x in arr if str(x).strip()]
        except Exception:
            pass
    return events, [(a, b, "cause") for a, b in causal]


def llm_extract_details_context(text: str, question: str | None, max_details: int = 40) -> List[Dict[str, str]]:
    """
    Deprecated stub (kept for backward compatibility). Use llm_extract_details_by_types instead.
    """
    return []


def llm_extract_details_by_types(text: str, question: str | None, per_type_limit: int = 12) -> List[Dict[str, str]]:
    """
    分类型提取细节：分别对 person/entity/dialogue/time/other 逐一发请求。
    每次只做一件事，输出 JSON 数组字符串列表，最后合并并去重。
    """
    if not text or not text.strip():
        return []
    types = ["person", "entity", "dialogue", "time", "other"]
    all_details: List[Dict[str, str]] = []
    seen = set()
    typed_seen = {"person": set(), "entity": set(), "dialogue": set(), "time": set(), "other": set()}

    def log_call(tag: str, messages: List[Dict[str, str]], response: str) -> None:
        if not LOG_CALLS:
            return
        try:
            LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
            with LOG_PATH.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"tag": tag, "messages": messages, "response": response}, ensure_ascii=False) + "\n")
        except Exception:
            pass
    payloads = []
    for dtype in types:
        user_lines = [
            f"只从下方上下文中抽取类型为 {dtype} 的细节片段。",
            "每个片段必须直接摘自原文，禁止编造、禁止解释。",
            "最终只输出 JSON 数组，每个元素是一个字符串（原文片段）。",
            f"最多返回 {per_type_limit} 个。",
            "严格禁止输出 <think>、思考过程或任何非 JSON 内容。",
        ]
        if question:
            user_lines.append(f"问题: {question}")
        user_lines.append("上下文:")
        user_lines.append(text)
        messages = [
            {
                "role": "system",
                "content": "你是信息抽取助手。禁止输出 <think> 或解释，最终只输出 JSON 数组（字符串列表）。",
            },
            {"role": "user", "content": "\n".join(user_lines)},
        ]
        payloads.append((dtype, messages))

    def run_call(payload):
        dtype, messages = payload
        raw = call_vllm(messages, max_tokens=None, temperature=0.0, top_p=0.9)
        raw = strip_think(raw)
        return dtype, messages, raw

    with ThreadPoolExecutor(max_workers=len(payloads)) as ex:
        futures = [ex.submit(run_call, p) for p in payloads]
        for fut in futures:
            try:
                dtype, messages, raw = fut.result()
                log_call(f"detail_{dtype}", messages, raw)
                arr = json.loads(raw.strip())
                if isinstance(arr, list):
                    for item in arr:
                        txt = str(item).strip()
                        if not txt:
                            continue
                        low_txt = txt.lower()
                        # 避免 entity 与已有 person/time/dialogue 重复
                        if dtype == "entity" and (
                            low_txt in typed_seen["person"] or low_txt in typed_seen["time"] or low_txt in typed_seen["dialogue"]
                        ):
                            continue
                        key = (dtype, low_txt)
                        if key in seen:
                            continue
                        seen.add(key)
                        typed_seen[dtype].add(low_txt)
                        all_details.append({"type": dtype, "text": txt})
            except Exception:
                continue
    return all_details


def llm_extract_events_with_details(
    text: str,
    details: List[Dict[str, str]],
    question: str | None,
    max_events: int = 8,
) -> Tuple[List[Dict[str, Any]], List[Tuple[int, int]]]:
    """
    基于已抽取的细节节点，从上下文抽取相关事件，并给出因果链。
    返回事件列表（包含 detail_refs 字段）和因果对。
    """
    if not text or not text.strip():
        return [], []
    detail_lines = []
    for i, d in enumerate(details):
        detail_lines.append(f"{i+1}. [{d.get('type','')}] {d.get('text','')}")
    user_lines = [
        "依据下方细节列表与原文，抽取按真实时间顺序排列的事件链，并标注关联的细节索引。",
        f"至少 3 条、最多 {max_events} 条事件；每条直接引用原文句子或短语，不改写、不合并整段。",
        "事件 JSON: {\"text\":\"...\",\"time\":\"可选时间\",\"details\":[细节索引列表] }（细节索引从1起，对应细节列表）。",
        "输出整体 JSON 对象: {\"events\":[...],\"causal\":[{\"from\":1,\"to\":2,\"type\":\"cause/result\"},...]}",
        "严格禁止输出 <think> 或思考过程，最终只输出 JSON 对象，不要附加解释。",
    ]
    if question:
        user_lines.append(f"问题: {question}")
    if detail_lines:
        user_lines.append("细节列表:")
        user_lines.extend(detail_lines)
    user_lines.append("上下文:")
    user_lines.append(text)
    messages = [
        {
            "role": "system",
            "content": "你是事件抽取助手。禁止输出 <think> 或思考过程，只输出 JSON 对象（events 与 causal），不要附加解释。",
        },
        {"role": "user", "content": "\n".join(user_lines)},
    ]
    try:
        raw = call_vllm(messages, max_tokens=None, temperature=0.0, top_p=0.9)
        raw = strip_think(raw)
        if LOG_CALLS:
            try:
                LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
                with LOG_PATH.open("a", encoding="utf-8") as f:
                    f.write(json.dumps({"tag": "events_with_details", "messages": messages, "response": raw}, ensure_ascii=False) + "\n")
            except Exception:
                pass
        obj = json.loads(raw.strip())
    except Exception:
        return [], []
    events: List[Dict[str, Any]] = []
    causal: List[Tuple[int, int]] = []
    if isinstance(obj, dict):
        evs = obj.get("events")
        if isinstance(evs, list):
            for ev in evs:
                if not isinstance(ev, dict):
                    continue
                txt = str(ev.get("text") or "").strip()
                if not txt:
                    continue
                time_hint = str(ev.get("time") or "").strip()
                refs = []
                ref_raw = ev.get("details", [])
                if isinstance(ref_raw, list):
                    for r in ref_raw:
                        try:
                            refs.append(int(r) - 1)
                        except Exception:
                            continue
                events.append({"text": txt, "time": time_hint, "details": refs})
        c_list = obj.get("causal")
        if isinstance(c_list, list):
            for c in c_list:
                if not isinstance(c, dict):
                    continue
                try:
                    a = int(c.get("from"))
                    b = int(c.get("to"))
                    rel = str(c.get("type") or "cause").lower()
                    causal.append((a - 1, b - 1, rel))
                except Exception:
                    continue
    return events, causal


def compress_event_text(text: str, max_chars: int = 20, max_words: int = 12) -> str:
    """Remove冗余前缀，保留原文内容，不再截断。"""
    t = text.strip()
    for prefix in ("Title:", "Content:", "Abstract:", "Summary:", "摘要：", "标题："):
        if t.lower().startswith(prefix.lower()):
            t = t[len(prefix):].strip()
            break
    return t


def parse_time_prefix(text: str) -> Tuple[str, int | None]:
    """
    解析类似 [T3] 前缀，返回 (去前缀文本, time_order)；若无则 time_order=None。
    """
    m = re.match(r"\s*\[t?(\d+)\]\s*(.*)", text, flags=re.IGNORECASE)
    if m:
        try:
            num = int(m.group(1))
        except ValueError:
            num = None
        return m.group(2).strip(), num
    return text.strip(), None


def add_missing_events_from_context(ctx: str, events: List[Dict[str, Any]], max_extra: int = 50) -> List[Dict[str, Any]]:
    """补充未覆盖的上下文句子为事件，确保上下文基本全覆盖。"""
    if not ctx:
        return events
    sentences = hf_split_text(ctx)
    existing = [e.get("text", "").lower() for e in events]
    extra: List[Dict[str, Any]] = []
    for sent in sentences:
        s = sent.strip()
        if not s:
            continue
        low = s.lower()
        if any((low in t) or (t in low) for t in existing):
            continue
        extra.append({"text": s, "time": "", "details": []})
        if len(extra) >= max_extra:
            break
    return events + extra


def infer_extra_relations(events: List[Dict[str, Any]], ctx: str, question: str | None) -> List[Tuple[int, int, str]]:
    """
    针对补充的事件，重新推断全局因果/顺序关系。允许模型返回空（无关系）。
    """
    if not events:
        return []
    rels = llm_infer_relations(events, ctx, question) if ctx else []
    # 去重并过滤非法索引
    seen = set()
    final = []
    for a_idx, b_idx, rel in rels:
        if not (0 <= a_idx < len(events) and 0 <= b_idx < len(events)):
            continue
        key = (a_idx, b_idx, rel)
        if key in seen:
            continue
        seen.add(key)
        final.append((a_idx, b_idx, rel))
    return final


def llm_infer_relations(events: List[Dict[str, Any]], ctx: str, question: str | None) -> List[Tuple[int, int, str]]:
    """
    让模型推断事件关系，若无明显因果则给出叙述前后关系。
    返回列表[(from_idx,to_idx,rel_type)]，rel_type in {"cause","result","precede"}。
    """
    if not events:
        return []
    lines = []
    for i, ev in enumerate(events):
        lines.append(f"{i+1}. {ev.get('text','')}")
    user_lines = [
        "根据事件列表推断事件关系，类型只能是 cause/result/precede（precede 表示真实时间/语义层面的前后顺序，而非仅文本顺序）。",
        "输出 JSON 数组，每个元素形如 {\"from\":1,\"to\":2,\"type\":\"cause|result|precede\"}，严格使用事件编号。",
        "禁止输出 <think> 或解释，只输出 JSON。",
        "事件列表：",
        "\n".join(lines),
    ]
    if question:
        user_lines.append(f"问题: {question}")
    if ctx:
        user_lines.append("原文上下文（可参考）：")
        ctx_snippet = trim_text_to_tokens(ctx, MAX_MODEL_TOKENS, reserve_tokens=1024)
        user_lines.append(ctx_snippet)
    messages = [
        {
            "role": "system",
            "content": "你是关系推断助手。禁止输出 <think>，只输出 JSON 数组，关系类型为 cause/result/precede（precede 需体现真实时间/语义顺序）。",
        },
        {"role": "user", "content": "\n".join(user_lines)},
    ]
    try:
        raw = call_vllm(messages, max_tokens=None, temperature=0.0, top_p=0.9)
        raw = strip_think(raw)
        try:
            LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
            with LOG_PATH.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"tag": "infer_relations", "messages": messages, "response": raw}, ensure_ascii=False) + "\n")
        except Exception:
            pass
        arr = json.loads(raw.strip())
    except Exception:
        return []
    rels: List[Tuple[int, int, str]] = []
    if isinstance(arr, list):
        for item in arr:
            if not isinstance(item, dict):
                continue
            try:
                a = int(item.get("from"))
                b = int(item.get("to"))
                rtype = str(item.get("type") or "precede").lower()
                rels.append((a - 1, b - 1, rtype))
            except Exception:
                continue
    return rels


def sample_to_events(
    sample: Dict[str, Any],
    scene_id: str,
    context_key: str | None,
    q_key: str,
    context_override: str | None = None,
) -> Tuple[List[Dict[str, Any]], List[Tuple[int, int, str]], List[Dict[str, str]]]:
    events: List[Dict[str, Any]] = []
    causal_pairs: List[Tuple[int, int, str]] = []
    details: List[Dict[str, str]] = []
    question = str(sample.get(q_key) or "")
    ctx = context_override if context_override is not None else ""
    if not ctx and context_key and sample.get(context_key):
        ctx = str(sample[context_key])
    if ctx:
        chunks = split_context_into_chunks(ctx, CHUNK_TOKEN_LIMIT)
        # 全局细节去重
        detail_map: Dict[Tuple[str, str], int] = {}
        details = []
        all_events: List[Dict[str, Any]] = []
        for chunk in chunks:
            chunk_details = llm_extract_details_by_types(chunk, question)
            chunk_idx_map: Dict[int, int] = {}
            for d in chunk_details:
                key = (d.get("type", "").lower(), (d.get("text") or "").strip().lower())
                if not key[1]:
                    continue
                if key in detail_map:
                    chunk_idx_map[len(chunk_idx_map)] = detail_map[key]
                else:
                    new_idx = len(details)
                    detail_map[key] = new_idx
                    chunk_idx_map[len(chunk_idx_map)] = new_idx
                    details.append(d)
            evs, _ = llm_extract_events_with_details(chunk, chunk_details, question)
            # 映射 detail refs 到全局索引
            for ev in evs:
                refs = ev.get("details") or []
                mapped = []
                for r in refs:
                    if isinstance(r, int) and 0 <= r < len(chunk_details):
                        mapped_idx = chunk_idx_map.get(r, None)
                        if mapped_idx is not None:
                            mapped.append(mapped_idx)
                ev["details"] = mapped
            if not evs:
                if "\n" in chunk and len(chunk.splitlines()) > 1:
                    evs = [{"text": e, "time": "", "details": []} for e in dialogue_events(chunk)]
                else:
                    evs = [{"text": e, "time": "", "details": []} for e in sentence_events(chunk)]
            if len(evs) < 3:
                extra = dialogue_events(chunk) if "\n" in chunk else sentence_events(chunk)
                extra_objs = [{"text": e, "time": "", "details": []} for e in extra if e not in [ev.get("text") for ev in evs]]
                evs = (evs or []) + extra_objs
                evs = evs[: max(3, len(evs))]
            all_events.extend(evs)
        events = add_missing_events_from_context(ctx, all_events)
    if not events:
        return [], [], details
    seen = set()
    out: List[Dict[str, Any]] = []
    parsed_events: List[Tuple[int, str, str]] = []
    ref_list: List[List[int]] = []
    for idx, ev in enumerate(events):
        text = ev.get("text", "")
        time_hint = ev.get("time", "")
        refs = ev.get("details", [])
        raw_text, t_order = parse_time_prefix(text)
        compressed = compress_event_text(raw_text)
        if compressed in seen:
            continue
        seen.add(compressed)
        parsed_events.append((t_order if t_order is not None else idx, compressed, time_hint))
        ref_list.append(refs if isinstance(refs, list) else [])
    # 按时间顺序排序
    combined = list(zip(parsed_events, ref_list))
    combined.sort(key=lambda x: x[0][0])
    for new_idx, (parsed, refs) in enumerate(combined):
        _, txt, time_hint = parsed
        out.append(
            {
                "event_id": f"{scene_id}#E{new_idx:03d}",
                "text": txt,
                "time": time_hint,
                "order_index": new_idx,
                "details": refs,
            }
        )
    # 重新推断关系；若模型未给出，则用叙述顺序关系
    rels = infer_extra_relations(out, ctx, question)
    if rels:
        causal_pairs = rels
    if not causal_pairs and len(out) >= 2:
        causal_pairs = [(i, i + 1, "precede") for i in range(len(out) - 1)]
    # 顺序叙述的关系仅保留相邻事件
    filtered: List[Tuple[int, int, str]] = []
    for a_idx, b_idx, rel in causal_pairs:
        rlow = (rel or "").lower()
        if rlow in ("cause", "result"):
            filtered.append((a_idx, b_idx, rlow))
        else:
            if abs(a_idx - b_idx) == 1:
                filtered.append((a_idx, b_idx, "precede"))
    # 确保所有事件至少有一条关系，否则按上下文顺序补充“precede”连接
    if out:
        related = set()
        for a_idx, b_idx, _ in filtered:
            related.add(a_idx)
            related.add(b_idx)
        if len(related) < len(out):
            for i in range(len(out)):
                if i in related:
                    continue
                # 优先连接前一条，否则后一条
                if i > 0:
                    if (i - 1, i, "precede") not in filtered:
                        filtered.append((i - 1, i, "precede"))
                        related.add(i - 1)
                        related.add(i)
                elif i + 1 < len(out):
                    if (i, i + 1, "precede") not in filtered:
                        filtered.append((i, i + 1, "precede"))
                        related.add(i)
                        related.add(i + 1)
    return out, filtered, details


def wipe_all(driver) -> None:
    # 清空整个数据库（仅用于测试构建，确保无残留）
    with driver.session(database=os.getenv("NEO4J_DB", "neo4j")) as session:
        session.run("MATCH (n) DETACH DELETE n")


def create_constraints(driver) -> None:
    with driver.session(database=os.getenv("NEO4J_DB", "neo4j")) as session:
        session.run(
            f"CREATE CONSTRAINT scene_id_unique IF NOT EXISTS FOR (s:{SCENE_LABEL}) REQUIRE s.scene_id IS UNIQUE;"
        )
        session.run(
            f"CREATE CONSTRAINT event_id_unique IF NOT EXISTS FOR (e:{EVENT_LABEL}) REQUIRE e.event_id IS UNIQUE;"
        )
        session.run(
            f"CREATE CONSTRAINT detail_id_unique IF NOT EXISTS FOR (d:{DETAIL_LABEL}) REQUIRE d.detail_id IS UNIQUE;"
        )


def insert_samples(
    driver,
    samples: List[Dict[str, Any]],
    q_key: str,
    a_key: str,
    c_key: str | None,
    snapshot_path: Path | None = None,
    existing_ids: Set[str] | None = None,
) -> None:
    db = os.getenv("NEO4J_DB", "neo4j")
    out_f = snapshot_path.open("a", encoding="utf-8") if snapshot_path else None
    batch_size = 20

    detail_label_map = {
        "person": "PersonDetail",
        "entity": "EntityDetail",
        "dialogue": "DialogueDetail",
        "time": "TimeDetail",
        "other": "OtherDetail",
    }

    def process_sample(idx_sample):
        idx, sample = idx_sample
        scene_id = str(sample.get("id") or sample.get("sample_id") or sample.get("idx") or f"S{idx:06d}")
        if existing_ids and scene_id in existing_ids:
            return None
        ctx = str(sample.get(c_key) or "") if c_key else ""
        events, causal_pairs, details = sample_to_events(sample, scene_id, c_key, q_key, context_override=ctx)
        if not events:
            return None
        return scene_id, events, causal_pairs, details

    with driver.session(database=db) as session:
        for batch_start in tqdm(range(0, len(samples), batch_size), desc="writing to neo4j"):
            batch = list(enumerate(samples[batch_start : batch_start + batch_size], start=batch_start))
            results = []
            with ThreadPoolExecutor(max_workers=batch_size) as ex:
                futures = [ex.submit(process_sample, item) for item in batch]
                for fut in futures:
                    res = fut.result()
                    if res:
                        results.append(res)
            for scene_id, events, causal_pairs, details in results:
                scene_props = {
                    "scene_id": scene_id,
                }
                session.run(
                    """
                    MERGE (s:%s {scene_id:$scene_id})
                    SET s += $props
                    WITH s UNWIND $events AS ev
                    MERGE (e:%s {event_id: ev.event_id})
                    SET e.text = ev.text, e.order_index = ev.order_index, e.time = coalesce(ev.time, "")
                    MERGE (s)-[:HAS_EVENT]->(e)
                    """
                    % (SCENE_LABEL, EVENT_LABEL),
                    {
                        "scene_id": scene_id,
                        "props": scene_props,
                        "events": events,
                    },
                )
                # NEXT edges
                for i in range(len(events) - 1):
                    session.run(
                        """
                        MATCH (a:%s {event_id:$a}), (b:%s {event_id:$b})
                        MERGE (a)-[:NEXT]->(b)
                        """
                        % (EVENT_LABEL, EVENT_LABEL),
                        {"a": events[i]["event_id"], "b": events[i + 1]["event_id"]},
                    )
                # causal edges
                for a_idx, b_idx, rel in causal_pairs:
                    if 0 <= a_idx < len(events) and 0 <= b_idx < len(events):
                        rel_lower = (rel or "").lower()
                        if rel_lower in ("cause", "result"):
                            session.run(
                                f"""
                                MATCH (a:{EVENT_LABEL} {{event_id:$a}}), (b:{EVENT_LABEL} {{event_id:$b}})
                                MERGE (a)-[:CAUSES {{rel_type:$rtype}}]->(b)
                                """,
                                {"a": events[a_idx]["event_id"], "b": events[b_idx]["event_id"], "rtype": rel_lower},
                            )
                        else:
                            session.run(
                                f"""
                                MATCH (a:{EVENT_LABEL} {{event_id:$a}}), (b:{EVENT_LABEL} {{event_id:$b}})
                                MERGE (a)-[:NARRATES {{rel_type:$rtype}}]->(b)
                                """,
                                {"a": events[a_idx]["event_id"], "b": events[b_idx]["event_id"], "rtype": rel_lower or "precede"},
                            )
                # Detail nodes
                for didx, det in enumerate(details):
                    dtype = str(det.get("type") or "").lower()
                    dtype_label = detail_label_map.get(dtype, detail_label_map["other"])
                    labels = f"{DETAIL_LABEL}:{dtype_label}"
                    detail_id = f"{scene_id}#D{didx:03d}"
                    session.run(
                        f"""
                        MERGE (d:{labels} {{detail_id:$did}})
                        SET d.text=$text, d.dtype=$dtype
                        """,
                        {"did": detail_id, "text": det.get("text", ""), "dtype": dtype},
                    )
                # INVOLVES edges based on detail refs
                for ev in events:
                    refs = ev.get("details") or []
                    for ridx in refs:
                        if ridx is None or ridx < 0 or ridx >= len(details):
                            continue
                        detail_id = f"{scene_id}#D{ridx:03d}"
                        dtype = str(details[ridx].get("type") or "").lower()
                        dtype_label = detail_label_map.get(dtype, detail_label_map["other"])
                        labels = f"{DETAIL_LABEL}:{dtype_label}"
                        session.run(
                            f"""
                            MATCH (e:{EVENT_LABEL} {{event_id:$eid}})
                            MATCH (d:{labels} {{detail_id:$did}})
                            MERGE (d)-[:INVOLVES]->(e)
                            """,
                            {"eid": ev["event_id"], "did": detail_id},
                        )
                # 额外基于文本匹配的 INVOLVES（防止未标 refs 但文本包含细节）
                for ridx, det in enumerate(details):
                    det_text = (det.get("text") or "").strip()
                    if not det_text:
                        continue
                    det_l = det_text.lower()
                    for ev in events:
                        if det_l and det_l in (ev.get("text") or "").lower():
                            dtype = str(det.get("type") or "").lower()
                            dtype_label = detail_label_map.get(dtype, detail_label_map["other"])
                            labels = f"{DETAIL_LABEL}:{dtype_label}"
                            detail_id = f"{scene_id}#D{ridx:03d}"
                            session.run(
                                f"""
                                MATCH (e:{EVENT_LABEL} {{event_id:$eid}})
                                MATCH (d:{labels} {{detail_id:$did}})
                                MERGE (d)-[:INVOLVES]->(e)
                                """,
                                {"eid": ev["event_id"], "did": detail_id},
                            )
                if out_f:
                    out_f.write(
                        json.dumps(
                            {
                                "scene": scene_props,
                                "events": events,
                                "causal": causal_pairs,
                                "details": details,
                                "source_sample": {"id": scene_id},
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
    if out_f:
        out_f.close()


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build Neo4j graph for TIME-Lite")
    parser.add_argument("--split", type=str, default=None, help="dataset split")
    parser.add_argument("--limit", type=int, default=None, help="limit samples")
    parser.add_argument("--wipe", action="store_true", help="delete existing timelite tag data before insert")
    parser.add_argument("--resume", action="store_true", help="resume from existing snapshot/graph without wiping")
    args = parser.parse_args(argv)

    ensure_dirs()
    data = load_timelite(args.split)
    if args.limit:
        data = data.select(range(args.limit))
    sample0 = data[0]
    q_key, a_key, c_key = pick_fields(sample0)

    uri = os.getenv("NEO4J_URI")
    user = os.getenv("NEO4J_USER") or os.getenv("NEO4J_USERNAME")
    pwd = os.getenv("NEO4J_PASSWORD")
    if not uri or not user or not pwd:
        raise SystemExit("NEO4J_URI/NEO4J_USER/NEO4J_PASSWORD required")
    driver = GraphDatabase.driver(uri, auth=(user, pwd))

    # 清空调用日志
    if LOG_CALLS:
        try:
            LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
            if LOG_PATH.exists():
                LOG_PATH.unlink()
        except Exception:
            pass

    existing_ids: Set[str] = set()
    snapshot_path = DATA_DIR / "processed" / "timelite_graph.jsonl"
    if args.resume and snapshot_path.exists():
        with snapshot_path.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    sid = obj.get("scene", {}).get("scene_id")
                    if sid:
                        existing_ids.add(str(sid))
                except Exception:
                    continue
    if not args.resume:
        wipe_all(driver)
        if snapshot_path.exists():
            snapshot_path.unlink()
        existing_ids = set()
    create_constraints(driver)
    # 过滤已完成的样本，避免批次空跑
    pending = []
    for idx, sample in enumerate(data):
        sid = str(sample.get("id") or sample.get("sample_id") or sample.get("idx") or f"S{idx:06d}")
        if sid in existing_ids:
            continue
        pending.append(sample)
    insert_samples(driver, pending, q_key, a_key, c_key, snapshot_path=snapshot_path, existing_ids=existing_ids)
    print(f"Snapshot saved to {snapshot_path}")
    print("Graph build done.")


if __name__ == "__main__":
    main()
