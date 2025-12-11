from __future__ import annotations

import json
from typing import Any, Dict, List

from mem_persona_agent.llm import ChatClient, embed, build_episode_prompt
from mem_persona_agent.memory.graph_store import GraphStore


class MemoryWriter:
    def __init__(self, store: GraphStore, client: ChatClient | None = None):
        self.store = store
        self.client = client or ChatClient()

    async def generate_and_store(self, character_id: str, persona: Dict[str, Any]) -> List[Dict[str, Any]]:
        messages = build_episode_prompt(persona)
        content = await self.client.chat(messages, temperature=0.7)
        try:
            data: Dict[str, Any] = json.loads(content)
            episodes = data.get("episodes", [])
        except json.JSONDecodeError:
            episodes = []
        processed = []
        for ep in episodes:
            narrative = ep.get("narrative", "")
            summary = ep.get("short_summary", "")
            ep["owner_id"] = character_id
            ep["embedding"] = await embed(narrative + " " + summary)
            processed.append(ep)
        if processed:
            self.store.write_static_episodes(character_id, processed)
        return processed
