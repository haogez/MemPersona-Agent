from __future__ import annotations

from typing import List, Dict, Any

from mem_persona_agent.agent.roleplay_agent import RoleplayAgent


async def run_dialogues(agent: RoleplayAgent, dialogues: List[Dict[str, Any]]):
    results = []
    for turn in dialogues:
        reply, memory = await agent.chat(
            turn.get("user_input", ""),
            place=turn.get("place"),
            npc=turn.get("npc"),
        )
        results.append({"reply": reply, "memory": memory})
    return results
