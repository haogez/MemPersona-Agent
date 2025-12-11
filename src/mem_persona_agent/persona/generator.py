from __future__ import annotations

import json
from typing import Any

from mem_persona_agent.llm import ChatClient, build_persona_prompt
from mem_persona_agent.persona.schema import Persona


class PersonaGenerator:
    def __init__(self, client: ChatClient | None = None):
        self.client = client or ChatClient()

    async def generate(self, seed: str) -> Persona:
        messages = build_persona_prompt(seed)
        content = await self.client.chat(messages, temperature=0.8)
        try:
            data: Any = json.loads(content)
        except json.JSONDecodeError:
            # fallback: generate minimal persona
            data = {
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
        return Persona.model_validate(data)
