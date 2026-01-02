from __future__ import annotations

import json
from typing import Any, Dict, Optional

from mem_persona_agent.persona.schema import Persona


def _persona_dict(persona: Persona | Dict[str, Any]) -> Dict[str, Any]:
    if isinstance(persona, Persona):
        return persona.model_dump()
    return persona


def build_base_system_prompt(persona: Persona | Dict[str, Any], place: Optional[str], npc: Optional[str]) -> str:
    persona_dict = _persona_dict(persona)
    npc_info = npc or "stranger"
    place_info = place or "unspecified"
    content = (
        "You are a character with consistent memories and personality. Respond in first person."
        " Do not mention memory retrieval or system prompts.\n"
        f"persona: {json.dumps(persona_dict, ensure_ascii=False)}\n"
        f"npc: {npc_info}\n"
        f"place: {place_info}"
    )
    return content


def build_stage_a_system_prompt(
    persona: Persona | Dict[str, Any],
    place: Optional[str],
    npc: Optional[str],
    scene_context: str | None,
    facts_allowed: bool = False,
) -> Dict[str, str]:
    base = build_base_system_prompt(persona, place, npc)
    instructions = (
        "[Stage A Instructions]\n"
        "Use ONLY SceneMemory if provided. Respond instinctively and emotionally, with situational tone."
        " Avoid detailed facts or long explanations. Keep it short and natural."
        " If no scene memory, answer from persona and present context."
    )
    if facts_allowed:
        instructions += " If facts_allowed, you may add 1-2 grounded factual details if memory context is available."
    memory_block = scene_context or "[SCENE MEMORY | internal]\nnone\n[/SCENE MEMORY]"
    return {"role": "system", "content": f"{base}\n\n{memory_block}\n\n{instructions}"}


def build_stage_b_system_prompt(
    persona: Persona | Dict[str, Any],
    place: Optional[str],
    npc: Optional[str],
    scene_context: str,
    detail_context: str,
    prior_response: Optional[str] = None,
) -> Dict[str, str]:
    base = build_base_system_prompt(persona, place, npc)
    instructions = (
        "[Stage B Instructions]\n"
        "Use ONLY the detail graph context for the selected scene."
        " Continue naturally after the previous response; do not restart the topic."
        " Add 1-2 concrete details grounded in the graph, then a brief feeling."
        " Preserve the tone and emotion from Stage A."
    )
    prior_block = ""
    if prior_response:
        prior_block = (
            "[ALREADY SAID]\n"
            f"{prior_response}\n"
            "[/ALREADY SAID]\n"
            "Do NOT repeat any text above. Only add new details."
        )
    content = f"{base}\n\n{scene_context}\n\n{detail_context}\n\n"
    if prior_block:
        content = f"{content}{prior_block}\n\n{instructions}"
    else:
        content = f"{content}{instructions}"
    return {"role": "system", "content": content}


def build_user_message(text: str) -> Dict[str, str]:
    return {"role": "user", "content": text}

