from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, Optional

from mem_persona_agent.llm import ChatClient, build_persona_prompt
from mem_persona_agent.config import settings
from mem_persona_agent.persona.schema import Persona

logger = logging.getLogger(__name__)


class PersonaGenerator:
    def __init__(self, client: ChatClient | None = None):
        self.client = client or ChatClient()

    async def generate(self, seed: str, timeline_mode: str = "strict") -> Persona:
        messages = build_persona_prompt(seed, timeline_mode=timeline_mode)
        logger.info("Generating persona seed=%s timeline_mode=%s model=%s", seed, timeline_mode, getattr(self.client, "model", None))
        logger.info("Seed repr=%r len=%s", seed, len(seed))
        print(f"[PersonaGenerator] seed={seed} timeline_mode={timeline_mode} model={getattr(self.client, 'model', None)}")
        print(f"[PersonaGenerator] seed repr={seed!r} len={len(seed)}")

        content = await self.client.chat(messages, temperature=0.7)
        logger.info("LLM raw content: %s", content)

        try:
            data: Any = json.loads(content)
        except json.JSONDecodeError:
            data = {}

        overrides = self._parse_seed(seed)
        filled = self._ensure_fields(data, seed, overrides)

        logger.info(
            "Generated persona name=%s age=%s gender=%s overrides=%s",
            filled.get("name"),
            filled.get("age"),
            filled.get("gender"),
            overrides,
        )
        print(f"[PersonaGenerator] generated name={filled.get('name')} age={filled.get('age')} gender={filled.get('gender')} overrides={overrides}")
        return Persona.model_validate(filled)

    def _parse_seed(self, seed: str) -> Dict[str, Any]:
        """Extract name/age/gender hints from seed to hard-enforce critical fields."""
        if not settings.enable_seed_overrides:
            return {}
        overrides: Dict[str, Any] = {}
        for raw_line in seed.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            m_name = re.match(r"^(?:姓名|名字)\s*[:：]\s*([\u4e00-\u9fff]{2,8}(?:·[\u4e00-\u9fff]{1,8})?)", line)
            if m_name and "name" not in overrides:
                overrides["name"] = m_name.group(1)
                continue
            m_gender = re.match(r"^性别\s*[:：]\s*(男|女)", line)
            if m_gender and "gender" not in overrides:
                overrides["gender"] = m_gender.group(1)
                continue
            m_age = re.match(r"^年龄\s*[:：]\s*(\d{1,3})", line)
            if m_age and "age" not in overrides:
                try:
                    overrides["age"] = int(m_age.group(1))
                except ValueError:
                    continue
        return overrides

    def _ensure_fields(self, data: Dict[str, Any], seed: str, overrides: Dict[str, Any]) -> Dict[str, Any]:
        """Fill missing fields, enforce personality/past_experience defaults, and override critical fields with seed hints."""
        fallback_name = "未知"
        for raw_line in seed.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if re.match(r"^(姓名|名字|性别|年龄)\s*[:：]", line):
                continue
            if re.fullmatch(r"[\u4e00-\u9fff]{2,12}", line):
                fallback_name = line
            break
        defaults: Dict[str, Any] = {
            "name": fallback_name,
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

        merged.update(overrides)

        personality = merged.get("personality") if isinstance(merged.get("personality"), dict) else {}
        merged["personality"] = {
            "openness": personality.get("openness", 50),
            "conscientiousness": personality.get("conscientiousness", 50),
            "extraversion": personality.get("extraversion", 50),
            "agreeableness": personality.get("agreeableness", 50),
            "neuroticism": personality.get("neuroticism", 50),
        }

        pe = merged.get("past_experience")
        merged["past_experience"] = pe if isinstance(pe, list) else []

        return merged
