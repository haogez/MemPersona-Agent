from __future__ import annotations

import json
from typing import Any, Dict, List, Optional
import logging
import math

from mem_persona_agent.llm import ChatClient, embed, build_memory_prompt, build_memory_supervision_prompt
from mem_persona_agent.memory.graph_store import GraphStore
from mem_persona_agent.persona.related import RelatedCharacterGenerator

DEFAULT_MEMORY_CONFIG: Dict[str, Any] = {
    "total_events": 5,
    "age_start": 0,
    "timeline_mode": "strict",
    "include_dialogue_ratio": 0.45,
    "max_summary_chars": 120,
    "max_dialogue_chars": 260,
}


class MemoryWriter:
    def __init__(self, store: GraphStore, client: ChatClient | None = None):
        self.store = store
        self.client = client or ChatClient()
        self.related_generator = RelatedCharacterGenerator(client=self.client)
        self.logger = logging.getLogger(__name__)

    async def generate_and_store(
        self,
        character_id: str,
        persona: Dict[str, Any],
        *,
        seed: str | None = None,
        memory_config: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        config = {**DEFAULT_MEMORY_CONFIG, **(memory_config or {})}
        config["age_end"] = persona.get("age", config.get("age_end"))

        self.logger.info("MemoryWriter: start generate related characters cid=%s seed=%s", character_id, seed)
        related = await self.related_generator.generate(seed or "", persona)
        self.logger.info("MemoryWriter: related characters count=%s data=%s", len(related), related)
        self.store.write_related_characters(character_id, related)

        messages = build_memory_prompt(character_id, persona, related, config)
        self.logger.info("MemoryWriter: sending memory prompt cid=%s", character_id)
        content = await self.client.chat(messages, temperature=0.5)
        self.logger.info("MemoryWriter: LLM raw content length=%s preview=%r", len(content or ""), (content or "")[:200])
        episodes = await self._extract_episodes(content, persona, related, config, character_id, messages)

        processed = []
        for idx, ep in enumerate(episodes):
            ep_id = ep.get("memory_id") or ep.get("id") or f"evt_{idx:04d}"
            summary = ep.get("summary_text", "")
            dialogue = ep.get("dialogue_text", "")
            merged_text = f"{summary} {dialogue}".strip()

            # build graph-friendly fields
            ep["id"] = ep_id
            ep["owner_id"] = character_id
            ep.setdefault("people", [])
            ep.setdefault("objects", [])
            ep.setdefault("places", [])

            participants = ep.get("participants") or []
            roles = ep.get("participant_roles") or []
            people = []
            for i, name in enumerate(participants):
                role = roles[i] if i < len(roles) else "other"
                people.append({"name": name, "role": role})
            ep["people"] = people
            place_name = ep.get("place")
            if place_name:
                ep["places"] = [{"name": place_name}]

            # derive objects/values from links
            links = ep.get("links") or []
            objects = []
            places = list(ep.get("places", []))
            values = []
            for link in links:
                if link.get("kind") == "object":
                    objects.append({"name": link.get("value")})
                if link.get("kind") == "value":
                    values.append({"name": link.get("value")})
                if link.get("kind") == "place":
                    places.append({"name": link.get("value")})
            ep["objects"] = objects
            ep["places"] = places
            if values:
                ep["values"] = values

            ep["embedding"] = await embed(merged_text)
            processed.append(ep)

        if processed:
            self.store.write_static_episodes(character_id, processed)
        return processed

    def _safe_loads(self, content: str) -> Dict[str, Any]:
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            start = content.find("{")
            end = content.rfind("}")
            if start != -1 and end != -1 and end > start:
                return json.loads(content[start : end + 1])
            raise

    def _fallback_episodes(self, persona: Dict[str, Any], related: List[Dict[str, Any]], config: Dict[str, Any], character_id: str) -> List[Dict[str, Any]]:
        total = config.get("total_events", 5)
        past = persona.get("past_experience") or []
        name = persona.get("name", "主角")
        places = [persona.get("usual_place", ""), "校园", "家中"]
        episodes: List[Dict[str, Any]] = []
        for i in range(total):
            src = past[i % len(past)] if past else {}
            age = src.get("时间") or src.get("age") or max(1, math.floor(persona.get("age", 18) * (i + 1) / total))
            topic = src.get("冲突点") or src.get("起因") or "成长事件"
            summary = src.get("结果") or src.get("summary_text") or f"{name}的关键经历"
            participants = [name]
            if related:
                participants.append(related[i % len(related)].get("name", "朋友"))
            ep = {
                "memory_id": f"evt_{i:04d}",
                "type": "achievement_event",
                "time_age": float(age) if isinstance(age, (int, float)) else 0.0,
                "time_label": f"{age}岁",
                "place": places[i % len(places)],
                "participants": participants,
                "participant_roles": ["self"] + (["friend"] if len(participants) > 1 else []),
                "topic": topic,
                "background": src.get("起因") or src.get("背景") or "",
                "summary_text": summary,
                "dialogue_text": "",
                "emotion_tags": ["焦虑", "坚韧"],
                "importance": 0.6,
                "canonical_facts": [
                    f"{name}在{age}岁经历过{topic}",
                ],
                "links": [
                    {"kind": "person", "value": name},
                ],
            }
            episodes.append(ep)
        return episodes

    async def _extract_episodes(self, content: str, persona: Dict[str, Any], related: List[Dict[str, Any]], config: Dict[str, Any], character_id: str, prompt_used: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """Parse episodes with retries and supervision; fallback if needed."""
        episodes: List[Dict[str, Any]] = []
        previous_output: Dict[str, Any] = {}

        def parse_once(raw: str) -> List[Dict[str, Any]]:
            try:
                data: Dict[str, Any] = self._safe_loads(raw)
                eps = data.get("episodes", [])
                if not eps and data.get("static_memories"):
                    eps = data.get("static_memories", [])
                return eps or []
            except Exception:
                return []

        episodes = parse_once(content)
        if episodes:
            self.logger.info("MemoryWriter: parsed episodes count=%s", len(episodes))
        previous_output = {"raw": content}

        if len(episodes) != config.get("total_events", 5) or not self._dialogue_sufficient(episodes):
            feedback = (
                f"episodes数量={len(episodes)}, 需要={config.get('total_events', 5)}; "
                f"dialogue_text需要至少5轮，当前是否满足={self._dialogue_sufficient(episodes)}"
            )
            self.logger.info("MemoryWriter: triggering supervision: %s", feedback)
            sup_messages = build_memory_supervision_prompt(character_id, persona, related, config, previous_output, feedback)
            sup_content = await self.client.chat(sup_messages, temperature=0.4)
            self.logger.info("MemoryWriter: supervision raw length=%s preview=%r", len(sup_content or ""), (sup_content or "")[:200])
            episodes = parse_once(sup_content)
            self.logger.info("MemoryWriter: supervision parsed episodes count=%s", len(episodes))

        if len(episodes) != config.get("total_events", 5) or not self._dialogue_sufficient(episodes):
            self.logger.info("MemoryWriter: using fallback episodes")
            episodes = self._fallback_episodes(persona, related, config, character_id)
            self.logger.info("MemoryWriter: fallback episodes count=%s", len(episodes))
        return episodes

    def _dialogue_sufficient(self, episodes: List[Dict[str, Any]]) -> bool:
        """Check dialogue_text has >=5轮 by counting '：' or lines."""
        for ep in episodes:
            dlg = ep.get("dialogue_text") or ""
            # count occurrences of ： or speaker markers
            colon_count = dlg.count("：")
            line_count = dlg.count("\n")
            if colon_count >= 5 or line_count >= 4:
                return True
        return False
