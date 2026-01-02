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
    One-shot pipeline: persona -> scene memory generation -> dialogue.
    Returns persona, scenes, reply, used_memory and saved paths.
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
    memory_bundle = await writer.generate_and_store(character_id, persona_dict, seed=seed)

    retriever = MemoryRetriever(store)
    agent = RoleplayAgent(persona, character_id, retriever, client=client)
    reply, used_memory = await agent.chat(user_input, place=place, npc=npc)

    paths = _save_artifacts(save_dir, character_id, persona_dict, memory_bundle, used_memory, reply, user_input)

    return {
        "character_id": character_id,
        "persona": persona_dict,
        "sequence": memory_bundle.get("sequence", []),
        "scenes": memory_bundle.get("scenes", []),
        "reply": reply,
        "used_memory": used_memory,
        "paths": paths,
    }


def run_full_pipeline_sync(**kwargs) -> Dict[str, Any]:
    """Sync helper."""
    return asyncio.run(run_full_pipeline(**kwargs))


def _save_artifacts(
    base_dir: str | Path,
    cid: str,
    persona: Dict[str, Any],
    memory_bundle: Dict[str, Any],
    used_memory: Dict[str, Any],
    reply: str,
    user_input: str,
) -> Dict[str, str]:
    base = Path(base_dir) / cid
    base.mkdir(parents=True, exist_ok=True)

    persona_path = base / "persona.json"
    sequence_path = base / "scene_sequence.json"
    scenes_path = base / "scenes.json"
    chat_path = base / "chat_sample.json"

    with persona_path.open("w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(persona, ensure_ascii=False, indent=2))
    with sequence_path.open("w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(memory_bundle.get("sequence", []), ensure_ascii=False, indent=2))
    with scenes_path.open("w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(memory_bundle.get("scenes", []), ensure_ascii=False, indent=2))
    with chat_path.open("w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps({"user_input": user_input, "reply": reply, "used_memory": used_memory}, ensure_ascii=False, indent=2))

    return {
        "persona": str(persona_path),
        "sequence": str(sequence_path),
        "scenes": str(scenes_path),
        "chat": str(chat_path),
    }

