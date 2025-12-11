from __future__ import annotations

from typing import Any, Dict, List, Optional

from mem_persona_agent.llm import embed
from mem_persona_agent.memory.graph_store import GraphStore


class MemoryRetriever:
    def __init__(self, store: GraphStore):
        self.store = store

    async def retrieve(self, character_id: str, user_input: str, place: Optional[str] = None, npc: Optional[str] = None) -> List[Dict[str, Any]]:
        query = f"{user_input} {place or ''}".strip()
        query_emb = await embed(query)
        results = self.store.query_similar(character_id, query_emb, npc=npc)
        return results
