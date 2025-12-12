from __future__ import annotations

import json
from typing import Any, Dict

from mem_persona_agent.llm import ChatClient, build_persona_prompt
from mem_persona_agent.persona.schema import Persona


class PersonaGenerator:
    def __init__(self, client: ChatClient | None = None):
        self.client = client or ChatClient()

    async def generate(self, seed: str, timeline_mode: str = "strict") -> Persona:
        messages = build_persona_prompt(seed, timeline_mode=timeline_mode)
        content = await self.client.chat(messages, temperature=0.2)
        try:
            data: Any = json.loads(content)
        except json.JSONDecodeError:
            # fallback: generate minimal persona
            data = {}
        filled = self._ensure_fields(data, seed)
        return Persona.model_validate(filled)

    def _ensure_fields(self, data: Dict[str, Any], seed: str) -> Dict[str, Any]:
        """填充缺失字段并兜底 personality/past_experience，避免 LLM 输出缺项导致 500。"""
        defaults: Dict[str, Any] = {
            "name": seed,
            "age": 25,
            "gender": "未知",
            "occupation": "未知",
            "hobby": "",
            "skill": "",
            "values": "",
            "living_habit": "",
            "dislike": "",
            "language_style": "",
            "appearance": "",
            "family_status": "",
            "education": "",
            "social_pattern": "",
            "favorite_thing": "",
            "usual_place": "",
            "past_experience": [],
            "background": "",
            "speech_style": "",
            "personality": {
                "openness": 50,
                "conscientiousness": 50,
                "extraversion": 50,
                "agreeableness": 50,
                "neuroticism": 50,
            },
        }
        merged = {**defaults, **(data or {})}

        # personality 必须是 dict，缺失的五个维度填 50
        personality = merged.get("personality") if isinstance(merged.get("personality"), dict) else {}
        merged["personality"] = {
            "openness": personality.get("openness", 50),
            "conscientiousness": personality.get("conscientiousness", 50),
            "extraversion": personality.get("extraversion", 50),
            "agreeableness": personality.get("agreeableness", 50),
            "neuroticism": personality.get("neuroticism", 50),
        }

        # past_experience 需要是列表
        pe = merged.get("past_experience")
        merged["past_experience"] = pe if isinstance(pe, list) else []

        return merged
