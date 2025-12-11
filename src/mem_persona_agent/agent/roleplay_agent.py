from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional, Tuple

from mem_persona_agent.agent.base import build_system_prompt, build_user_message
from mem_persona_agent.llm import ChatClient
from mem_persona_agent.memory import MemoryRetriever
from mem_persona_agent.message import MemoryBuffer
from mem_persona_agent.persona.schema import Persona


class RoleplayAgent:
    def __init__(self, persona: Persona, character_id: str, retriever: MemoryRetriever, buffer: Optional[MemoryBuffer] = None, client: Optional[ChatClient] = None):
        self.persona = persona
        self.character_id = character_id
        self.retriever = retriever
        self.buffer = buffer or MemoryBuffer()
        self.client = client or ChatClient()
        self.last_memory_context: List[Dict[str, Any]] = []

    async def chat(self, user_input: str, place: Optional[str] = None, npc: Optional[str] = None) -> Tuple[str, List[Dict[str, Any]]]:
        memory_task = asyncio.create_task(
            self.retriever.retrieve(self.character_id, user_input, place, npc)
        )

        done, _ = await asyncio.wait({memory_task}, timeout=0)
        maybe_memory = list(done)[0].result() if done else None

        system_prompt = build_system_prompt(self.persona, place, npc, maybe_memory)
        history = self.buffer.history()
        user_message = build_user_message(user_input)

        reply = await self.client.chat([system_prompt] + history + [user_message])

        full_memory = await memory_task
        self.last_memory_context = full_memory

        self.buffer.add_user(user_input)
        self.buffer.add_assistant(reply)

        return reply, full_memory
