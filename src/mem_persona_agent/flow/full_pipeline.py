from __future__ import annotations

import asyncio
import json
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

from mem_persona_agent.agent import RoleplayAgent
from mem_persona_agent.llm import ChatClient
from mem_persona_agent.memory import GraphStore, MemoryRetriever, MemoryWriter
from mem_persona_agent.persona.generator import PersonaGenerator
from mem_persona_agent.persona.schema import Persona


async def run_full_pipeline(
    seed: str,
    user_input: str,
    *,
    timeline_mode: str = "strict",
    place: Optional[str] = None,
    npc: Optional[str] = None,
    save_dir: str | Path = "artifacts",
    client: Optional[ChatClient] = None,
    store: Optional[GraphStore] = None,
) -> Dict[str, Any]:
    """
    一句话执行：角色生成 -> 记忆生成存储 -> 对话，且落盘 JSON。
    返回 persona、episodes、reply、used_memory 以及保存路径。
    """
    store = store or GraphStore()
    store.ensure_schema()
    client = client or ChatClient()
    generator = PersonaGenerator(client=client)

    persona: Persona = await generator.generate(seed, timeline_mode=timeline_mode)
    character_id = str(uuid.uuid4())
    persona_dict = persona.model_dump()
    store.write_persona(character_id, persona_dict)

    writer = MemoryWriter(store, client=client)
    episodes = await writer.generate_and_store(character_id, persona_dict)

    retriever = MemoryRetriever(store)
    agent = RoleplayAgent(persona, character_id, retriever, client=client)
    reply, used_memory = await agent.chat(user_input, place=place, npc=npc)

    paths = _save_artifacts(save_dir, character_id, persona_dict, episodes, used_memory, reply, user_input)

    return {
        "character_id": character_id,
        "persona": persona_dict,
        "episodes": episodes,
        "reply": reply,
        "used_memory": used_memory,
        "paths": paths,
    }


def run_full_pipeline_sync(**kwargs) -> Dict[str, Any]:
    """同步便捷封装，方便脚本批量调用。"""
    return asyncio.run(run_full_pipeline(**kwargs))


def _save_artifacts(
    base_dir: str | Path,
    cid: str,
    persona: Dict[str, Any],
    episodes: list[Dict[str, Any]],
    memory: list[Dict[str, Any]],
    reply: str,
    user_input: str,
) -> Dict[str, str]:
    base = Path(base_dir) / cid
    base.mkdir(parents=True, exist_ok=True)

    persona_path = base / "persona.json"
    episodes_path = base / "episodes.json"
    chat_path = base / "chat_sample.json"

    persona_path.write_text(json.dumps(persona, ensure_ascii=False, indent=2), encoding="utf-8")
    episodes_path.write_text(json.dumps(episodes, ensure_ascii=False, indent=2), encoding="utf-8")
    chat_path.write_text(
        json.dumps({"user_input": user_input, "reply": reply, "used_memory": memory}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return {"persona": str(persona_path), "episodes": str(episodes_path), "chat": str(chat_path)}
