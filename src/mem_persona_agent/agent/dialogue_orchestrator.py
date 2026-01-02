from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass
from typing import Any, AsyncGenerator, Dict, List, Optional
import re

from mem_persona_agent.agent.base import build_stage_a_system_prompt, build_stage_b_system_prompt, build_user_message
from mem_persona_agent.agent.roleplay_agent import fast_keyword_hit
from mem_persona_agent.config import settings
from mem_persona_agent.llm import ChatClient
from mem_persona_agent.memory import MemoryRetriever
from mem_persona_agent.memory.gating import detect_open_slots, should_recall_scene
from mem_persona_agent.memory.prompt_compiler import compile_detail_context, compile_scene_context
from mem_persona_agent.utils.state_manager import TurnIDManager


@dataclass
class SSEEvent:
    event: str
    data: str


logger = logging.getLogger(__name__)


def _truncate_text(text: str | None, max_len: int = 800) -> str | None:
    if text is None:
        return None
    if len(text) <= max_len:
        return text
    return f"{text[:max_len]}...<truncated {len(text)} chars>"


def _log(stage: str, payload: Dict[str, Any]) -> None:
    logger.info("chat_stream.%s %s", stage, json.dumps(payload, ensure_ascii=False))


def _persona_summary(persona: Any) -> Dict[str, Any]:
    if persona is None:
        return {}
    if hasattr(persona, "model_dump"):
        data = persona.model_dump()
    elif isinstance(persona, dict):
        data = persona
    else:
        return {}
    return {k: data.get(k) for k in ("name", "age", "gender", "occupation", "hobby") if k in data}


def _summarize_history(history: List[Dict[str, Any]], limit: int = 3) -> List[Dict[str, Any]]:
    summaries: List[Dict[str, Any]] = []
    for item in history[-limit:]:
        content = item.get("content") or ""
        summaries.append(
            {
                "role": item.get("role"),
                "content": _truncate_text(str(content), max_len=80),
                "length": len(str(content)),
            }
        )
    return summaries


def _summarize_nodes(nodes: List[Dict[str, Any]], limit: int = 10) -> List[Dict[str, Any]]:
    summaries: List[Dict[str, Any]] = []
    for node in nodes[:limit]:
        summaries.append(
            {
                "id": node.get("id"),
                "type": node.get("type") or node.get("node_type"),
                "name": _truncate_text(str(node.get("name") or node.get("speaker") or ""), max_len=80),
                "text": _truncate_text(
                    str(node.get("text") or node.get("utterance") or node.get("summary") or ""), max_len=80
                ),
            }
        )
    return summaries


def _build_detail_context_text(detail_pack: Dict[str, Any]) -> str:
    if not detail_pack:
        return ""
    scene_gist = detail_pack.get("scene_gist") or ""
    characters = [c.get("name") for c in (detail_pack.get("characters") or []) if isinstance(c, dict)]
    places = [p.get("name") for p in (detail_pack.get("places") or []) if isinstance(p, dict)]
    events = detail_pack.get("events") or []
    utterances = detail_pack.get("utterances") or []
    lines = ["[DETAIL MEMORY | internal]"]
    if scene_gist:
        lines.append(f"Scene: {scene_gist}")
    if characters:
        lines.append(f"People: {', '.join([str(x) for x in characters[:3] if x])}")
    if places:
        lines.append(f"Places: {', '.join([str(x) for x in places[:2] if x])}")
    lines.append("Key events:")
    for event in events[:5]:
        if not isinstance(event, dict):
            continue
        text = (
            event.get("event_text")
            or event.get("summary")
            or event.get("description")
            or event.get("content")
            or event.get("action")
            or event.get("title")
        )
        if text:
            lines.append(f"- {text}")
        else:
            fallback_id = event.get("event_id") or event.get("id") or "event"
            lines.append(f"- Event {fallback_id}")
    if utterances:
        lines.append("Utterances:")
        for utt in utterances[:3]:
            if not isinstance(utt, dict):
                continue
            text = utt.get("text") or utt.get("content") or utt.get("dialogue_text")
            speaker = utt.get("speaker") or utt.get("name") or ""
            if text:
                prefix = f"{speaker}: " if speaker else ""
                lines.append(f"- {prefix}{text}")
    if len(lines) <= 2 and events:
        lines.append("(Events found but no displayable fields.)")
    lines.append("[/DETAIL MEMORY]")
    return chr(10).join(lines)

def _stream_delay_seconds(token: str) -> float:
    if not token:
        return 0.0
    if any(mark in token for mark in ("。", ".")):
        return 1.0
    if any(mark in token for mark in ("，", ",")):
        return 0.5
    return 0.0


def _needs_deep_recall(user_input: str) -> bool:
    text = user_input or ""
    if detect_open_slots(text):
        return True
    lowered = text.lower()
    trigger_hits = [k for k in settings.memory_trigger_keywords if k and k.lower() in lowered]
    return bool(trigger_hits)


def _build_decision_prompt(
    persona: Any,
    user_input: str,
    stage_a_text: str,
    scene_context: str,
    detail_pack: Dict[str, Any],
    open_slots: List[str],
) -> str:
    scene_gist = detail_pack.get("scene_gist") or ""
    event_cnt = len(detail_pack.get("events") or [])
    utter_cnt = len(detail_pack.get("utterances") or [])
    people = [c.get("name") for c in (detail_pack.get("characters") or []) if isinstance(c, dict)]
    places = [p.get("name") for p in (detail_pack.get("places") or []) if isinstance(p, dict)]
    summary = {
        "scene_gist": scene_gist,
        "event_cnt": event_cnt,
        "utterance_cnt": utter_cnt,
        "people": people[:5],
        "places": places[:3],
        "open_slots": open_slots,
    }
    persona_name = ""
    if hasattr(persona, "model_dump"):
        persona_name = persona.model_dump().get("name") or ""
    elif isinstance(persona, dict):
        persona_name = persona.get("name") or ""
    lines = [
        "You are a decision module. Output ONLY strict JSON, no extra text.",
        f"user_input: {user_input}",
        f"persona_name: {persona_name}",
        f"stage_a_text: {stage_a_text}",
        f"scene_context: {scene_context}",
        f"detail_pack_summary: {json.dumps(summary, ensure_ascii=False)}",
        "Decide if StageA should be interrupted to allow detail supplement. If memory is old or low importance, still allow a compressed supplement with reasons.",
        "Return JSON with fields:",
        '{'
        '"should_interrupt_stage_a": true/false, '
        '"interrupt_reason": "...", '
        '"transition_text": "...", '
        '"should_supplement_now": true/false, '
        '"supplement_budget_chars": 220, '
        '"selected_detail_keys": ["participants","event_text","place","actions","utterances","time","objects"], '
        '"defer_to_next_turn": false, '
        '"defer_prompt_hint": "", '
        '"memory_compression": {"level":"low|medium|high|extreme", "reason":"..."}, '
        '"memory_limitations": {"should_explain": true/false, "reasons": ["too_old","not_important","fragmented","emotion_avoidance"]}'
        '}',
    ]
    return chr(10).join(lines)


def _parse_decision_json(text: str) -> Dict[str, Any]:
    try:
        data = json.loads(text)
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _default_decision(detail_pack: Dict[str, Any]) -> Dict[str, Any]:
    has_detail = bool(detail_pack.get("events") or detail_pack.get("utterances"))
    return {
        "should_interrupt_stage_a": False,
        "interrupt_reason": "",
        "transition_text": "...Let me think, I only recall fragments.",
        "should_supplement_now": bool(has_detail),
        "supplement_budget_chars": 220,
        "selected_detail_keys": ["participants", "event_text", "place", "utterances"],
        "defer_to_next_turn": False,
        "defer_prompt_hint": "",
        "memory_compression": {"level": "medium", "reason": ""},
        "memory_limitations": {"should_explain": False, "reasons": []},
    }


def _filter_detail_pack(detail_pack: Dict[str, Any], keys: List[str]) -> Dict[str, Any]:
    if not detail_pack:
        return {}
    keyset = {k.lower() for k in keys if isinstance(k, str)}
    filtered = {
        "scene_id": detail_pack.get("scene_id"),
        "scene_gist": detail_pack.get("scene_gist"),
        "events": [],
        "places": [],
        "times": [],
        "objects": [],
        "characters": [],
        "utterances": [],
        "event_edges": detail_pack.get("event_edges") or [],
        "utter_edges": detail_pack.get("utter_edges") or [],
    }
    if {"participants", "people", "characters"} & keyset:
        filtered["characters"] = detail_pack.get("characters") or []
    if {"place", "places", "location"} & keyset:
        filtered["places"] = detail_pack.get("places") or []
    if {"time", "times"} & keyset:
        filtered["times"] = detail_pack.get("times") or []
    if {"objects", "object"} & keyset:
        filtered["objects"] = detail_pack.get("objects") or []
    if {"event_text", "events", "actions", "action"} & keyset:
        filtered["events"] = detail_pack.get("events") or []
    if {"utterances", "dialogue", "speech"} & keyset:
        filtered["utterances"] = detail_pack.get("utterances") or []
    return filtered


class DialogueOrchestrator:
    def __init__(
        self,
        retriever: MemoryRetriever,
        *,
        turn_manager: Optional[TurnIDManager] = None,
        stage_b_timeout: float = 5.0,
    ) -> None:
        self.retriever = retriever
        self.turn_manager = turn_manager or TurnIDManager()
        self.stage_b_timeout = stage_b_timeout

    async def run_chat_stream(self, body: Any, history: List[Dict[str, Any]]) -> AsyncGenerator[str, None]:
        request_id = str(uuid.uuid4())
        queue: asyncio.Queue[SSEEvent] = asyncio.Queue()
        stage_a_done = asyncio.Event()
        stage_a_first_event = asyncio.Event()
        interrupt_event = asyncio.Event()
        interrupt_payload: Dict[str, Any] = {}
        stage_a_buffer: List[str] = []
        loop = asyncio.get_running_loop()
        reply_a_future: asyncio.Future[str] = loop.create_future()
        context_pack_future: asyncio.Future[Optional[Dict[str, Any]]] = loop.create_future()
        need_recall_future: asyncio.Future[bool] = loop.create_future()
        timing: Dict[str, Optional[float]] = {"stage_a_ms": None, "stage_b_ms": None}
        errors: Dict[str, Optional[str]] = {"stage_b": None}
        decision: Dict[str, Any] = {
            "should_supplement": False,
            "gain_reason": "",
            "interrupted": False,
            "should_interrupt_stage_a": False,
            "should_supplement_now": False,
            "defer_to_next_turn": False,
            "supplement_budget_chars": None,
            "selected_detail_keys": [],
            "memory_compression": {},
            "memory_limitations": {},
            "interrupt_reason": "",
            "transition_text": "",
            "defer_prompt_hint": "",
        }
        thread_a_meta: Dict[str, Any] = {
            "used_scene_memory": False,
            "scene_id": None,
            "scene_source": "none",
            "scene_gist_chars": 0,
            "anchors_count": 0,
            "participants_count": 0,
        }
        thread_b_meta: Dict[str, Any] = {
            "should_deep_recall": False,
            "decision_reason": "",
            "selected_scene_id": None,
            "top_score": None,
            "vector_enabled": False,
            "vector_attempted": False,
        }
        detail_stats: Dict[str, Any] = {
            "event_cnt": 0,
            "utterance_cnt": 0,
            "place_cnt": 0,
            "time_cnt": 0,
            "object_cnt": 0,
            "character_cnt": 0,
            "speaker_cnt": 0,
            "event_edge_cnt": 0,
            "utter_edge_cnt": 0,
            "detail_context_chars": 0,
        }
        thread_c_meta: Dict[str, Any] = {
            "supplement_plan": "none",
            "need_interrupt": False,
            "gain_reason": "",
            "gain_score": None,
            "used_detail_context": False,
            "compression": {"mode": "none", "ratio": None, "reason": None},
        }
        stage_a_first_ms: Optional[float] = None
        stage_a_stream_delay_ms: float = 0.0
        default_transition = _default_decision({}).get("transition_text") or "...Let me think, I only recall fragments."
        decision_payload: Dict[str, Any] = {}

        turn_id = self.turn_manager.start_turn(body.character_id)
        user_input = body.user_input or ""
        facts_allowed = bool(body.inject_memory and fast_keyword_hit(body.character_id, user_input))
        stage_a_candidates: List[Dict[str, Any]] = []
        stage_a_retrieve_meta: Dict[str, Any] = {}
        stage_a_decision: Dict[str, Any] = {}
        stage_a_selected_scene: Optional[Dict[str, Any]] = None
        stage_a_scene_context: Optional[str] = None
        if body.inject_memory:
            stage_a_candidates, stage_a_retrieve_meta = await self.retriever.retrieve_scene_candidates(
                body.character_id,
                user_input,
                place=body.place,
                npc=body.npc,
                limit=body.memory_top_k,
                vector_enabled=False,
                debug=True,
                cache_only=True,
            )
            stage_a_decision = should_recall_scene(user_input, stage_a_candidates)
            if stage_a_decision.get("recall") and stage_a_candidates:
                stage_a_selected_scene = stage_a_candidates[0].get("scene")
                stage_a_scene_context = compile_scene_context(stage_a_selected_scene)
                scene_gist = str(
                    stage_a_selected_scene.get("scene_gist")
                    or stage_a_selected_scene.get("summary_7whr")
                    or ""
                )
                anchors = stage_a_selected_scene.get("anchors") or stage_a_selected_scene.get("keywords") or []
                participants = stage_a_selected_scene.get("participants") or stage_a_selected_scene.get("who") or []
                thread_a_meta.update(
                    {
                        "used_scene_memory": True,
                        "scene_id": stage_a_selected_scene.get("scene_id"),
                        "scene_source": "thread_b_selected",
                        "scene_gist_chars": len(scene_gist),
                        "anchors_count": len(anchors) if isinstance(anchors, list) else 0,
                        "participants_count": len(participants) if isinstance(participants, list) else 0,
                    }
                )

        _log(
            "request",
            {
                "request_id": request_id,
                "character_id": body.character_id,
                "persona": _persona_summary(body.persona),
                "inject_memory": body.inject_memory,
                "history_count": len(history),
                "history_last": _summarize_history(history),
                "user_input": _truncate_text(user_input, max_len=400),
                "facts_allowed": facts_allowed,
            },
        )

        client = ChatClient(model=settings.chat_model_name)
        decision_client = ChatClient(model=settings.chat_model_name)
        system_prompt_a = build_stage_a_system_prompt(
            body.persona, body.place, body.npc, scene_context=stage_a_scene_context, facts_allowed=facts_allowed
        )
        messages_a = [system_prompt_a, *history, build_user_message(user_input)]
        _log(
            "stage_a.prompt",
            {
                "request_id": request_id,
                "message_count": len(messages_a),
                "system_chars": len(system_prompt_a.get("content", "")),
                "user_input_len": len(user_input),
            },
        )

        async def task_a() -> None:
            nonlocal stage_a_first_ms, stage_a_stream_delay_ms
            total_a = ""
            transition_emitted = False
            started = time.perf_counter()
            _log("task_a.start", {"request_id": request_id})
            try:
                async for token in client.stream_chat(messages_a):
                    if interrupt_event.is_set():
                        if not transition_emitted:
                            transition_text = interrupt_payload.get("transition_text") or default_transition
                            if transition_text:
                                stage_a_buffer.append(transition_text)
                                total_a += transition_text
                                await queue.put(SSEEvent(event="delta", data=transition_text))
                                delay = _stream_delay_seconds(transition_text)
                                if delay:
                                    stage_a_stream_delay_ms += delay * 1000.0
                                    await asyncio.sleep(delay)
                            transition_emitted = True
                        break
                    if not token:
                        continue
                    if stage_a_first_ms is None:
                        stage_a_first_ms = round((time.perf_counter() - started) * 1000.0, 2)
                        stage_a_first_event.set()
                        _log(
                            "task_a.first_token",
                            {"request_id": request_id, "first_token_ms": stage_a_first_ms},
                        )
                    stage_a_buffer.append(token)
                    total_a += token
                    await queue.put(SSEEvent(event="delta", data=token))
                    delay = _stream_delay_seconds(token)
                    if delay:
                        stage_a_stream_delay_ms += delay * 1000.0
                        await asyncio.sleep(delay)
            except Exception as exc:  # pragma: no cover - defensive
                _log("task_a.error", {"request_id": request_id, "error": _truncate_text(str(exc))})
                await queue.put(SSEEvent(event="delta", data=str(exc)))
            finally:
                if stage_a_first_ms is None:
                    stage_a_first_event.set()
                if interrupt_event.is_set() and not transition_emitted:
                    transition_text = interrupt_payload.get("transition_text") or default_transition
                    if transition_text:
                        stage_a_buffer.append(transition_text)
                        total_a += transition_text
                        await queue.put(SSEEvent(event="delta", data=transition_text))
                        delay = _stream_delay_seconds(transition_text)
                        if delay:
                            stage_a_stream_delay_ms += delay * 1000.0
                            await asyncio.sleep(delay)
                timing["stage_a_ms"] = round((time.perf_counter() - started) * 1000.0, 2)
                if not reply_a_future.done():
                    reply_a_future.set_result(total_a)
                stage_a_done.set()
                _log(
                    "task_a.done",
                    {
                        "request_id": request_id,
                        "output_chars": len(total_a),
                        "stage_a_ms": timing["stage_a_ms"],
                        "first_token_ms": stage_a_first_ms,
                    },
                )

        async def _deep_recall() -> Dict[str, Any]:
            recall_started = time.perf_counter()
            open_slots = detect_open_slots(user_input)
            if stage_a_selected_scene:
                candidates = stage_a_candidates or [
                    {
                        "scene": stage_a_selected_scene,
                        "scene_id": stage_a_selected_scene.get("scene_id"),
                        "score": stage_a_decision.get("top_score") or 1.0,
                        "keyword_score": stage_a_decision.get("top_score") or 1.0,
                    }
                ]
                retrieve_meta = stage_a_retrieve_meta or {
                    "vector_count": 0,
                    "keyword_count": len(candidates),
                    "query_text": user_input,
                    "vector_enabled": False,
                    "vector_attempted": False,
                    "fallback_mode": "keyword_only",
                    "final_candidates_count": len(candidates),
                }
                decision_local = stage_a_decision or {
                    "recall": True,
                    "reason": "stage_a",
                    "top_score": 1.0,
                    "gate_trace": [],
                }
            else:
                candidates, retrieve_meta = await self.retriever.retrieve_scene_candidates(
                    body.character_id,
                    user_input,
                    place=body.place,
                    npc=body.npc,
                    limit=body.memory_top_k,
                    vector_enabled=False,
                    debug=True,
                )
                decision_local = should_recall_scene(user_input, candidates)
            thread_b_meta["should_deep_recall"] = bool(decision_local.get("recall"))
            thread_b_meta["decision_reason"] = decision_local.get("reason") or thread_b_meta.get("decision_reason") or ""
            thread_b_meta["top_score"] = decision_local.get("top_score")
            thread_b_meta["vector_enabled"] = bool(retrieve_meta.get("vector_enabled"))
            thread_b_meta["vector_attempted"] = bool(retrieve_meta.get("vector_attempted"))
            selected_scene = None
            scene_context = None
            if decision_local.get("recall") and candidates:
                selected_scene = candidates[0].get("scene")
                scene_context = compile_scene_context(selected_scene)
                thread_b_meta["selected_scene_id"] = selected_scene.get("scene_id") if selected_scene else None

            detail_nodes: List[Dict[str, Any]] = []
            detail_context = ""
            detail_pack: Dict[str, Any] = {}
            if selected_scene:
                scene_id = selected_scene.get("scene_id")
                detail_pack = await self.retriever.retrieve_detail_pack(body.character_id, scene_id)
                detail_nodes = self.retriever.detail_nodes_from_pack(detail_pack)
                detail_context = compile_detail_context(detail_nodes)
                if not detail_context and (detail_pack.get("events") or detail_pack.get("utterances")):
                    detail_context = _build_detail_context_text(detail_pack)
                event_cnt = detail_pack.get("event_cnt") or len(detail_pack.get("events") or [])
                pack_counts = {
                    "events": len(detail_pack.get("events") or []),
                    "utterances": len(detail_pack.get("utterances") or []),
                    "objects": len(detail_pack.get("objects") or []),
                    "places": len(detail_pack.get("places") or []),
                    "times": len(detail_pack.get("times") or []),
                    "characters": len(detail_pack.get("characters") or []),
                }
                detail_stats.update(
                    {
                        "event_cnt": event_cnt,
                        "utterance_cnt": pack_counts["utterances"],
                        "place_cnt": pack_counts["places"],
                        "time_cnt": pack_counts["times"],
                        "object_cnt": pack_counts["objects"],
                        "character_cnt": pack_counts["characters"],
                        "speaker_cnt": len(detail_pack.get("speakers") or []),
                        "event_edge_cnt": len(detail_pack.get("event_edges") or []),
                        "utter_edge_cnt": len(detail_pack.get("utter_edges") or []),
                        "detail_context_chars": len(detail_context or ""),
                    }
                )
                sample_event = ""
                sample_utt = ""
                if detail_pack.get("events"):
                    event0 = detail_pack.get("events")[0]
                    if isinstance(event0, dict):
                        sample_event = str(
                            event0.get("event_text")
                            or event0.get("summary")
                            or event0.get("description")
                            or event0.get("event_id")
                            or ""
                        )
                if detail_pack.get("utterances"):
                    utt0 = detail_pack.get("utterances")[0]
                    if isinstance(utt0, dict):
                        sample_utt = str(utt0.get("text") or utt0.get("content") or utt0.get("utt_id") or "")
                _log(
                    "task_b.pack",
                    {
                        "request_id": request_id,
                        "event_cnt": event_cnt,
                        "pack_counts": pack_counts,
                        "detail_context_len": len(detail_context or ""),
                        "sample": {
                            "event0": _truncate_text(sample_event, max_len=30),
                            "utt0": _truncate_text(sample_utt, max_len=30),
                        },
                    },
                )
                if event_cnt < 3:
                    _log(
                        "task_b.event_short",
                        {"request_id": request_id, "scene_id": scene_id, "owner_id": body.character_id},
                    )

            _log(
                "task_b.retrieval",
                {
                    "request_id": request_id,
                    "duration_ms": round((time.perf_counter() - recall_started) * 1000.0, 2),
                    "candidate_count": len(candidates),
                    "keyword_count": retrieve_meta.get("keyword_count"),
                    "vector_count": retrieve_meta.get("vector_count"),
                    "selected_scene_id": selected_scene.get("scene_id") if selected_scene else None,
                    "open_slots": open_slots,
                    "decision": decision_local,
                    "detail_nodes": len(detail_nodes),
                    "detail_context_len": len(detail_context or ""),
                },
            )
            _log(
                "task_b.retrieval_debug",
                {
                    "request_id": request_id,
                    "retrieval_meta": retrieve_meta.get("debug") or retrieve_meta,
                },
            )
            if selected_scene:
                _log(
                    "task_b.scene",
                    {
                        "request_id": request_id,
                        "scene_id": selected_scene.get("scene_id"),
                        "scene_gist": _truncate_text(str(selected_scene.get("scene_gist") or ""), max_len=200),
                        "place": selected_scene.get("place"),
                        "participants": selected_scene.get("participants"),
                        "anchors": selected_scene.get("anchors"),
                    },
                )
            if detail_nodes:
                _log(
                    "task_b.details_sample",
                    {
                        "request_id": request_id,
                        "detail_nodes_count": len(detail_nodes),
                        "detail_nodes_sample": _summarize_nodes(detail_nodes),
                    },
                )

            return {
                "candidates": candidates,
                "retrieval_meta": retrieve_meta,
                "open_slots": open_slots,
                "decision": decision_local,
                "selected_scene": selected_scene,
                "scene_context": scene_context,
                "detail_nodes": detail_nodes,
                "detail_context": detail_context,
                "detail_pack": detail_pack,
            }

        async def task_b() -> None:
            started = time.perf_counter()
            context_pack: Optional[Dict[str, Any]] = None
            need_recall = bool(body.inject_memory and _needs_deep_recall(user_input))
            thread_b_meta["should_deep_recall"] = bool(need_recall)
            if not body.inject_memory:
                thread_b_meta["decision_reason"] = "inject_memory_false"
            elif not need_recall:
                thread_b_meta["decision_reason"] = "need_recall_false"
            if not need_recall_future.done():
                need_recall_future.set_result(need_recall)
            if not body.inject_memory:
                timing["stage_b_ms"] = 0.0
                if not context_pack_future.done():
                    context_pack_future.set_result(None)
                _log("task_b.skip", {"request_id": request_id, "reason": "inject_memory_false"})
                return
            _log("task_b.start", {"request_id": request_id, "need_deep_recall": need_recall})
            try:
                context_pack = await asyncio.wait_for(_deep_recall(), timeout=self.stage_b_timeout)
                if context_pack:
                    vector_error = (context_pack.get("retrieval_meta") or {}).get("vector_error")
                    if vector_error and not errors["stage_b"]:
                        errors["stage_b"] = str(vector_error) or repr(vector_error)
            except Exception as exc:  # pragma: no cover - defensive
                errors["stage_b"] = str(exc) or repr(exc) or "stage_b_error"
                context_pack = None
                _log("task_b.error", {"request_id": request_id, "error": _truncate_text(str(exc))})
            finally:
                timing["stage_b_ms"] = round((time.perf_counter() - started) * 1000.0, 2)
                if not context_pack_future.done():
                    context_pack_future.set_result(context_pack)
                _log(
                    "task_b.done",
                    {"request_id": request_id, "stage_b_ms": timing["stage_b_ms"], "has_context": bool(context_pack)},
                )

        async def task_c() -> None:
            nonlocal decision_payload
            _log("task_c.start", {"request_id": request_id})
            need_recall = False
            try:
                need_recall = await need_recall_future
            except Exception:
                need_recall = False
            context_pack: Optional[Dict[str, Any]] = None
            if need_recall:
                try:
                    await asyncio.wait_for(stage_a_first_event.wait(), timeout=0.8)
                except asyncio.TimeoutError:
                    pass
                context_pack = await context_pack_future
                detail_pack = (context_pack or {}).get("detail_pack") or {}
                if context_pack and detail_pack:
                    stage_a_text = _truncate_text("".join(stage_a_buffer), max_len=800) or ""
                    scene_context = context_pack.get("scene_context") or ""
                    open_slots = context_pack.get("open_slots") or []
                    decision_prompt = _build_decision_prompt(
                        body.persona,
                        user_input,
                        stage_a_text,
                        scene_context,
                        detail_pack,
                        open_slots,
                    )
                    try:
                        decision_text = await decision_client.chat(
                            [
                                {"role": "system", "content": decision_prompt},
                                {"role": "user", "content": "Return JSON only."},
                            ],
                            temperature=0.2,
                        )
                        decision_payload = _parse_decision_json(decision_text)
                    except Exception as exc:  # pragma: no cover - defensive
                        decision_payload = {}
                        errors["stage_b"] = errors["stage_b"] or str(exc) or repr(exc) or "decision_error"
                        _log("task_c.decision_error", {"request_id": request_id, "error": _truncate_text(str(exc))})
                if not decision_payload:
                    decision_payload = _default_decision(detail_pack)
                if decision_payload.get("should_interrupt_stage_a"):
                    if not stage_a_done.is_set():
                        transition_text = decision_payload.get("transition_text") or default_transition
                        interrupt_payload["transition_text"] = transition_text
                        interrupt_event.set()
                        _log(
                            "task_c.interrupt",
                            {
                                "request_id": request_id,
                                "reason": decision_payload.get("interrupt_reason") or "decision",
                                "transition_text": _truncate_text(transition_text, max_len=120),
                            },
                        )
                    else:
                        decision_payload["should_interrupt_stage_a"] = False
                        _log(
                            "task_c.interrupt_skip",
                            {"request_id": request_id, "reason": "stage_a_done"},
                        )
                _log("task_c.decision", {"request_id": request_id, "decision": decision_payload})

            await stage_a_done.wait()
            reply_a = await reply_a_future
            if context_pack is None:
                context_pack = await context_pack_future
            interrupted = not self.turn_manager.is_active(body.character_id, turn_id)
            decision["interrupted"] = interrupted
            if interrupted:
                decision["should_supplement"] = False
                decision["gain_reason"] = "interrupted"
                thread_c_meta.update(
                    {
                        "supplement_plan": "none",
                        "need_interrupt": False,
                        "gain_reason": decision["gain_reason"],
                        "gain_score": None,
                        "used_detail_context": False,
                        "compression": {"mode": "none", "ratio": None, "reason": None},
                    }
                )
                _log("task_c.interrupted", {"request_id": request_id})
                return
            if not context_pack:
                decision["should_supplement"] = False
                decision["gain_reason"] = "no_context"
                thread_c_meta.update(
                    {
                        "supplement_plan": "none",
                        "need_interrupt": False,
                        "gain_reason": decision["gain_reason"],
                        "gain_score": None,
                        "used_detail_context": False,
                        "compression": {"mode": "none", "ratio": None, "reason": None},
                    }
                )
                _log("task_c.no_context", {"request_id": request_id})
                return

            detail_pack = context_pack.get("detail_pack") or {}
            if not decision_payload:
                decision_payload = _default_decision(detail_pack)

            decision["should_interrupt_stage_a"] = bool(decision_payload.get("should_interrupt_stage_a"))
            decision["interrupt_reason"] = decision_payload.get("interrupt_reason") or ""
            decision["transition_text"] = decision_payload.get("transition_text") or ""
            decision["should_supplement_now"] = bool(decision_payload.get("should_supplement_now"))
            decision["defer_to_next_turn"] = bool(decision_payload.get("defer_to_next_turn"))
            decision["defer_prompt_hint"] = decision_payload.get("defer_prompt_hint") or ""
            budget_value = decision_payload.get("supplement_budget_chars")
            try:
                decision["supplement_budget_chars"] = int(budget_value)
            except (TypeError, ValueError):
                decision["supplement_budget_chars"] = 0
            selected_keys = decision_payload.get("selected_detail_keys")
            decision["selected_detail_keys"] = selected_keys if isinstance(selected_keys, list) else []
            decision["memory_compression"] = decision_payload.get("memory_compression") or {}
            decision["memory_limitations"] = decision_payload.get("memory_limitations") or {}

            if decision["defer_to_next_turn"]:
                decision["should_supplement"] = False
                decision["gain_reason"] = "defer_to_next_turn"
                thread_c_meta.update(
                    {
                        "supplement_plan": "next_turn",
                        "need_interrupt": bool(decision["should_interrupt_stage_a"]),
                        "gain_reason": decision["gain_reason"],
                        "gain_score": None,
                        "used_detail_context": False,
                        "compression": {"mode": "none", "ratio": None, "reason": None},
                    }
                )
                _log("task_c.defer", {"request_id": request_id, "hint": decision["defer_prompt_hint"]})
                return
            if not decision["should_supplement_now"]:
                decision["should_supplement"] = False
                decision["gain_reason"] = "decision_skip"
                thread_c_meta.update(
                    {
                        "supplement_plan": "none",
                        "need_interrupt": bool(decision["should_interrupt_stage_a"]),
                        "gain_reason": decision["gain_reason"],
                        "gain_score": None,
                        "used_detail_context": False,
                        "compression": {"mode": "none", "ratio": None, "reason": None},
                    }
                )
                _log("task_c.skip", {"request_id": request_id})
                return

            selected_keys = decision["selected_detail_keys"]
            filtered_pack = _filter_detail_pack(detail_pack, selected_keys) if selected_keys else detail_pack
            detail_nodes = self.retriever.detail_nodes_from_pack(filtered_pack)
            detail_context = compile_detail_context(detail_nodes)
            if not detail_context:
                detail_context = _build_detail_context_text(filtered_pack)
            if not detail_context and filtered_pack is not detail_pack:
                detail_nodes = self.retriever.detail_nodes_from_pack(detail_pack)
                detail_context = compile_detail_context(detail_nodes) or _build_detail_context_text(detail_pack)
            if not detail_context:
                decision["should_supplement"] = False
                decision["gain_reason"] = "no_detail_context"
                detail_stats["detail_context_chars"] = 0
                thread_c_meta.update(
                    {
                        "supplement_plan": "none",
                        "need_interrupt": bool(decision["should_interrupt_stage_a"]),
                        "gain_reason": decision["gain_reason"],
                        "gain_score": None,
                        "used_detail_context": False,
                        "compression": {"mode": "none", "ratio": None, "reason": None},
                    }
                )
                _log("task_c.no_detail", {"request_id": request_id})
                return

            compression = decision.get("memory_compression") or {}
            limitations = decision.get("memory_limitations") or {}
            compression_level = str(compression.get("level") or "").lower()
            if compression_level in {"high", "extreme"} and limitations.get("should_explain"):
                reasons = limitations.get("reasons") or []
                reason_text = ", ".join([str(r) for r in reasons if r])
                if not reason_text:
                    reason_text = str(compression.get("reason") or "")
                if not reason_text:
                    reason_text = "memory limitations"
                limitation_note = f"[MEMORY LIMITATIONS]\nreasons: {reason_text}\n[/MEMORY LIMITATIONS]"
                detail_context = f"{limitation_note}\n{detail_context}"

            detail_context_len = len(detail_context or "")
            detail_stats["detail_context_chars"] = detail_context_len
            gain_score = None
            if detail_context_len > 0:
                gain_score = round(min(1.0, detail_context_len / 400.0), 3)
            if compression_level in {"high", "extreme"}:
                compression_mode = "heavy"
            elif compression_level in {"low", "medium"}:
                compression_mode = "light"
            else:
                compression_mode = "none"
            ratio = None
            budget_value = decision.get("supplement_budget_chars") or 0
            if detail_context_len > 0 and budget_value:
                try:
                    ratio = round(min(1.0, float(budget_value) / float(detail_context_len)), 3)
                except (TypeError, ValueError):
                    ratio = None
            thread_c_meta.update(
                {
                    "supplement_plan": "this_turn",
                    "need_interrupt": bool(decision["should_interrupt_stage_a"]),
                    "gain_reason": decision.get("interrupt_reason") or "detail_context",
                    "gain_score": gain_score,
                    "used_detail_context": detail_context_len > 0,
                    "compression": {
                        "mode": compression_mode,
                        "ratio": ratio,
                        "reason": compression.get("reason"),
                    },
                }
            )

            decision["should_supplement"] = True
            decision["gain_reason"] = decision.get("interrupt_reason") or "detail_context"
            prior_text = _truncate_text(reply_a, max_len=800) or reply_a
            system_prompt_b = build_stage_b_system_prompt(
                body.persona,
                body.place,
                body.npc,
                context_pack.get("scene_context") or "",
                detail_context,
                prior_response=prior_text,
            )
            messages_b = [
                system_prompt_b,
                *history,
                build_user_message(user_input),
                {"role": "assistant", "content": reply_a},
            ]
            _log(
                "stage_b.prompt",
                {
                    "request_id": request_id,
                    "message_count": len(messages_b),
                    "system_chars": len(system_prompt_b.get("content", "")),
                    "detail_chars": len(detail_context or ""),
                },
            )
            try:
                _log("task_c.supplement_start", {"request_id": request_id})
                total_b = ""
                budget = int(decision.get("supplement_budget_chars") or 0)
                if budget <= 0:
                    budget = 220
                async for token in client.stream_chat(messages_b):
                    if not token:
                        continue
                    if budget and len(total_b) >= budget:
                        break
                    if budget and len(total_b) + len(token) > budget:
                        token = token[: max(budget - len(total_b), 0)]
                    if not token:
                        break
                    total_b += token
                    await queue.put(SSEEvent(event="supplement", data=token))
                    delay = _stream_delay_seconds(token)
                    if delay:
                        await asyncio.sleep(delay)
                    if budget and len(total_b) >= budget:
                        break
                _log(
                    "task_c.supplement_done",
                    {"request_id": request_id, "output_chars": len(total_b), "budget": budget},
                )
            except Exception as exc:  # pragma: no cover - defensive
                errors["stage_b"] = errors["stage_b"] or str(exc) or repr(exc) or "stage_b_error"
                decision["should_supplement"] = False
                decision["gain_reason"] = "supplement_error"
                _log("task_c.error", {"request_id": request_id, "error": _truncate_text(str(exc))})

        task_a_handle = asyncio.create_task(task_a())
        task_b_handle = asyncio.create_task(task_b())
        task_c_handle = asyncio.create_task(task_c())
        queue_task = asyncio.create_task(queue.get())

        try:
            tasks = {task_a_handle, task_c_handle, queue_task}
            while True:
                done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                if queue_task in done:
                    event = queue_task.result()
                    if event.event in {"delta", "supplement"}:
                        yield event.data
                    else:
                        yield f"\n\nevent: {event.event}\ndata: {event.data}\n\n"
                    queue_task = asyncio.create_task(queue.get())
                    tasks = {task_a_handle, task_c_handle, queue_task}
                    continue
                if task_a_handle.done() and task_c_handle.done():
                    while not queue.empty():
                        event = queue.get_nowait()
                        if event.event in {"delta", "supplement"}:
                            yield event.data
                        else:
                            yield f"\n\nevent: {event.event}\ndata: {event.data}\n\n"
                    break
            meta = {
                "timing": {
                    "stage_a_ms": timing["stage_a_ms"],
                    "stage_b_ms": timing["stage_b_ms"],
                    "stage_a_first_ms": stage_a_first_ms,
                    "stage_a_stream_delay_ms": round(stage_a_stream_delay_ms, 2),
                },
                "thread_a": thread_a_meta,
                "thread_b": thread_b_meta,
                "detail_stats": detail_stats,
                "thread_c": thread_c_meta,
            }
            _log(
                "meta",
                {
                    "request_id": request_id,
                    "timing": meta["timing"],
                    "thread_a": thread_a_meta,
                    "thread_b": thread_b_meta,
                    "detail_stats": detail_stats,
                    "thread_c": thread_c_meta,
                },
            )
            yield f"\n\n\nevent: meta\ndata: {json.dumps(meta, ensure_ascii=False)}\n\n"
            yield "event: end\ndata: [DONE]\n\n"
        finally:
            for handle in (task_b_handle, task_c_handle):
                if not handle.done():
                    handle.cancel()
                    try:
                        await handle
                    except asyncio.CancelledError:
                        pass
                else:
                    try:
                        handle.result()
                    except Exception:
                        pass
            if not task_a_handle.done():
                task_a_handle.cancel()
                try:
                    await task_a_handle
                except asyncio.CancelledError:
                    pass
            else:
                try:
                    task_a_handle.result()
                except Exception:
                    pass
            if not queue_task.done():
                queue_task.cancel()
                try:
                    await queue_task
                except asyncio.CancelledError:
                    pass

