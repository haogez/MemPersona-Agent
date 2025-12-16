from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from mem_persona_agent.persona.schema import Persona


def build_system_prompt(persona: Persona, place: Optional[str], npc: Optional[str], memory_context: Optional[List[Dict[str, Any]]]) -> Dict[str, str]:
    persona_dict = persona.model_dump()
    memory_text = "\n".join([
        f"- {m.get('topic', m.get('title', ''))}: {m.get('summary_text', m.get('short_summary', ''))} (score={m.get('score', '')})"
        for m in (memory_context or [])
    ]) or "无可用记忆"
    npc_info = f"当前对话对象：{npc}" if npc else "当前对话对象：陌生人"
    place_info = f"地点：{place}" if place else "地点：未指定"
    content = (
        "你是一名具备真实人格和记忆的角色。"
        "请根据提供的人设、场景和记忆回复用户。"
        "回复必须表现为角色本人（assistant）第一人称。"
        "无需额外解释。"
        f"人设：{json.dumps(persona_dict, ensure_ascii=False)}\n"
        f"{npc_info}\n"
        f"{place_info}\n"
        f"可用记忆：\n{memory_text}"
    )
    return {"role": "system", "content": content}


def build_user_message(text: str) -> Dict[str, str]:
    return {"role": "user", "content": text}
