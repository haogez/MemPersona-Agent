from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from mem_persona_agent.agent.base import build_stage_a_system_prompt, build_stage_b_system_prompt, build_user_message
from mem_persona_agent.config import settings
from mem_persona_agent.llm import ChatClient
from mem_persona_agent.memory import MemoryRetriever
from mem_persona_agent.memory.gating import detect_open_slots, should_recall_scene
from mem_persona_agent.memory.prompt_compiler import compile_detail_context, compile_scene_context
from mem_persona_agent.message import MemoryBuffer
from mem_persona_agent.persona.schema import Persona


_SCENE_KEYWORD_CACHE: Dict[str, Dict[str, List[str]]] = {}


def _load_scene_keyword_cache() -> Dict[str, Dict[str, List[str]]]:
    path = Path(settings.scene_keyword_index_path)
    if not path.is_absolute():
        path = Path.cwd() / path
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return data
    except Exception:
        return {}
    return {}


_SCENE_KEYWORD_CACHE = _load_scene_keyword_cache()


def fast_keyword_hit(character_id: str, text: str) -> bool:
    cache = _SCENE_KEYWORD_CACHE.get(character_id) or {}
    if not cache or not text:
        return False
    tokens = re.findall(r"[A-Za-z0-9]+|[\u4e00-\u9fff]+", text.lower())
    for token in tokens:
        if len(token) < 2:
            continue
        if token in cache:
            return True
        token_upper = token.upper()
        if token_upper in cache:
            return True
    return False


class RoleplayAgent:
    def __init__(
        self,
        persona: Persona,
        character_id: str,
        retriever: MemoryRetriever,
        buffer: Optional[MemoryBuffer] = None,
        client: Optional[ChatClient] = None,
    ):
        self.persona = persona
        self.character_id = character_id
        self.retriever = retriever
        self.buffer = buffer or MemoryBuffer()
        self.client = client or ChatClient()
        self.last_memory_context: List[Dict[str, Any]] = []

    async def chat(self, user_input: str, place: Optional[str] = None, npc: Optional[str] = None) -> Tuple[str, Dict[str, Any]]:
        history = self.buffer.history()
        candidates, meta = await self.retriever.retrieve_scene_candidates(
            self.character_id,
            user_input,
            place=place,
            npc=npc,
            limit=8,
        )
        decision = should_recall_scene(user_input, candidates)
        selected_scene = None
        scene_context = None
        if decision.get("recall") and candidates:
            selected_scene = candidates[0].get("scene")
            scene_context = compile_scene_context(selected_scene)

        system_prompt_a = build_stage_a_system_prompt(self.persona, place, npc, scene_context, facts_allowed=False)
        user_message = build_user_message(user_input)
        reply_a = await self.client.chat([system_prompt_a] + history + [user_message])

        open_slots = detect_open_slots(user_input)
        detail_nodes: List[Dict[str, Any]] = []
        reply_b = ""
        if selected_scene and open_slots:
            scene_id = selected_scene.get("scene_id")
            detail_nodes = await self.retriever.retrieve_details(self.character_id, scene_id, user_input, limit=10)
            detail_context = compile_detail_context(detail_nodes)
            if detail_context:
                system_prompt_b = build_stage_b_system_prompt(
                    self.persona,
                    place,
                    npc,
                    scene_context or "",
                    detail_context,
                    prior_response=reply_a,
                )
                reply_b = await self.client.chat([system_prompt_b] + history + [user_message, {"role": "assistant", "content": reply_a}])

        reply = f"{reply_a}{reply_b}" if reply_b else reply_a
        self.last_memory_context = candidates

        self.buffer.add_user(user_input)
        self.buffer.add_assistant(reply)

        used_memory = {
            "selected_scene_id": selected_scene.get("scene_id") if selected_scene else None,
            "open_slots": open_slots,
            "scene": selected_scene,
            "detail_nodes": detail_nodes,
            "decision": decision,
            "candidates": candidates[:3],
            "retrieval_meta": meta,
        }

        return reply, used_memory

    def build_stage_a_messages(
        self,
        user_input: str,
        history: List[Dict[str, Any]],
        *,
        place: Optional[str],
        npc: Optional[str],
        scene_context: Optional[str],
        facts_allowed: bool,
    ) -> List[Dict[str, Any]]:
        system_prompt_a = build_stage_a_system_prompt(self.persona, place, npc, scene_context, facts_allowed=facts_allowed)
        return [system_prompt_a, *history, build_user_message(user_input)]

    async def stream_stage_a(
        self,
        messages: List[Dict[str, Any]],
        *,
        temperature: float = 0.7,
    ):
        self.last_streamed_reply = ""
        async for token in self.client.stream_chat(messages, temperature=temperature):
            if not token:
                continue
            self.last_streamed_reply += token
            yield token

