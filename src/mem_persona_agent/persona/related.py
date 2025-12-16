from __future__ import annotations

import json
from typing import Any, Dict, List
import logging

from mem_persona_agent.llm import ChatClient, build_related_characters_prompt


class RelatedCharacterGenerator:
    def __init__(self, client: ChatClient | None = None):
        self.client = client or ChatClient()
        self.logger = logging.getLogger(__name__)

    async def generate(self, seed: str, persona: Dict[str, Any]) -> List[Dict[str, Any]]:
        messages = build_related_characters_prompt(seed, persona)
        self.logger.info("RelatedCharacterGenerator: start seed=%s", seed)
        content = await self.client.chat(messages, temperature=0.5)
        self.logger.info("RelatedCharacterGenerator: raw content length=%s preview=%r", len(content or ""), (content or "")[:200])
        try:
            data: Dict[str, Any] = self._safe_loads(content)
            related = data.get("related_characters", [])
        except Exception:
            self.logger.warning("RelatedCharacterGenerator: json parse failed")
            related = []
        if not related:
            # fallback 3人，保证 name 是人名形式
            base = persona.get("name", "角色")
            related = [
                {"name": f"{base}的朋友A", "relation": "朋友", "attitude": "信任但保持距离"},
                {"name": f"{base}的老师B", "relation": "老师", "attitude": "尊重并感激"},
                {"name": f"{base}的同学C", "relation": "同学", "attitude": "表面友好，内心防备"},
            ]
            self.logger.info("RelatedCharacterGenerator: using fallback related characters")
        related = self._normalize_names(persona, related)
        return related

    def _safe_loads(self, content: str) -> Dict[str, Any]:
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            start = content.find("{")
            end = content.rfind("}")
            if start != -1 and end != -1 and end > start:
                return json.loads(content[start : end + 1])
            raise

    def _normalize_names(self, persona: Dict[str, Any], related: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """确保 name 更像人名而非关系称谓。"""
        base = persona.get("name", "角色")
        relation_terms = {"父亲", "母亲", "爸爸", "妈妈", "老师", "同学", "朋友", "同桌"}
        normalized: List[Dict[str, Any]] = []
        for item in related:
            name = item.get("name", "")
            rel = item.get("relation", "")
            if name in relation_terms or len(name) <= 2 and name in {"爸", "妈"}:
                name = f"{base}的{rel or name}"
            normalized.append({**item, "name": name})
        # 如果 seed/persona 里包含明确的人名，确保纳入
        if base not in [r.get("name") for r in normalized]:
            normalized.append({"name": base, "relation": "主角", "attitude": "自我"})
        return normalized
