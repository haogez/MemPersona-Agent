from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List

from mem_persona_agent.llm import ChatClient, build_related_characters_stage1_prompt
from mem_persona_agent.utils import StatsCollector

logger = logging.getLogger(__name__)

MIDDLE_DOT = "·"
NAME_ALLOWED_RE = re.compile(r"^[\u4e00-\u9fff·]{2,6}$")
BAD_NAME_CHARS_RE = re.compile(r"[A-Za-z0-9_=;:]")
LATIN1_RE = re.compile(r"[\u00c0-\u00ff]")


def _has_garbled(text: str) -> bool:
    return "\ufffd" in (text or "") or bool(LATIN1_RE.search(text or ""))


def is_valid_cn_person_name(name: str) -> bool:
    if not name:
        return False
    name = name.strip()
    if _has_garbled(name):
        return False
    if BAD_NAME_CHARS_RE.search(name):
        return False
    if not NAME_ALLOWED_RE.fullmatch(name):
        return False
    if name.count(MIDDLE_DOT) > 1 or name.startswith(MIDDLE_DOT) or name.endswith(MIDDLE_DOT):
        return False
    core = name.replace(MIDDLE_DOT, "")
    return 2 <= len(core) <= 6


def _collect_text_fields(persona: Dict[str, Any]) -> List[str]:
    texts: List[str] = []
    for key, val in (persona or {}).items():
        if isinstance(val, str):
            texts.append(val)
        elif isinstance(val, list):
            for item in val:
                if isinstance(item, dict):
                    for v in item.values():
                        if isinstance(v, str):
                            texts.append(v)
                elif isinstance(item, str):
                    texts.append(item)
        elif isinstance(val, dict):
            for v in val.values():
                if isinstance(v, str):
                    texts.append(v)
    return texts


def _extract_names_from_text(text: str) -> List[str]:
    names: List[str] = []
    if not text:
        return names
    explicit_patterns = [
        r"(?:叫|名叫|名字叫|名字是|名为|姓名是|昵称|外号)([\u4e00-\u9fff·]{2,6})",
    ]
    for pat in explicit_patterns:
        for match in re.findall(pat, text):
            if match and match not in names:
                names.append(match)
    for match in re.findall(r"(?<![\u4e00-\u9fff·])([\u4e00-\u9fff·]{2,6})(?![\u4e00-\u9fff·])", text):
        if match not in names:
            names.append(match)
    return names


def extract_known_names(seed: str, persona: Dict[str, Any]) -> List[str]:
    persona_name = (persona or {}).get("name") or ""
    candidates: List[str] = []
    for name in _extract_names_from_text(seed or ""):
        if name not in candidates:
            candidates.append(name)
    for text in _collect_text_fields(persona or {}):
        for name in _extract_names_from_text(text):
            if name not in candidates:
                candidates.append(name)
    known = []
    for name in candidates:
        if name == persona_name:
            continue
        if not is_valid_cn_person_name(name):
            continue
        if name not in known:
            known.append(name)
    return known


class RelatedCharacterGenerator:
    def __init__(
        self,
        client: ChatClient | None = None,
        *,
        stats: StatsCollector | None = None,
        target_min: int = 8,
        target_max: int = 12,
    ):
        self.client = client or ChatClient(stats=stats)
        if stats and getattr(self.client, "stats", None) is None:
            self.client.stats = stats
        self.stats = stats
        self.target_min = target_min
        self.target_max = target_max

    async def generate(self, seed: str, persona: Dict[str, Any]) -> List[Dict[str, Any]]:
        known_names = extract_known_names(seed or "", persona)
        feedback = ""
        for attempt in range(3):
            if self.stats and attempt > 0:
                self.stats.record_retry("related.stage1")
            messages = build_related_characters_stage1_prompt(
                seed,
                persona,
                known_names,
                self.target_min,
                self.target_max,
                feedback=feedback,
            )
            if self.stats:
                with self.stats.stage("related.stage1"):
                    content = await self.client.chat(messages, temperature=0.4)
            else:
                content = await self.client.chat(messages, temperature=0.4)
            related = self._parse_related(content)
            if related:
                self._record_summary(len(related), known_names)
                return related
            feedback = "输出不是有效的 related_characters 列表"
        raise RuntimeError("RelatedCharacter generation failed: empty or invalid JSON output")

    def _record_summary(self, count: int, known_names: List[str]) -> None:
        if not self.stats:
            return
        self.stats.add_note("related_generation_stage", "stage1", stage="related.summary")
        self.stats.add_note("count", count, stage="related.summary")
        self.stats.add_note("known_names", known_names, stage="related.summary")

    def _parse_related(self, content: str) -> List[Dict[str, Any]]:
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            start = content.find("{")
            end = content.rfind("}")
            if start != -1 and end != -1 and end > start:
                data = json.loads(content[start : end + 1])
            else:
                return []
        if isinstance(data, dict):
            related = data.get("related_characters") or []
            if isinstance(related, list):
                return [x for x in related if isinstance(x, dict)]
        if isinstance(data, list):
            return [x for x in data if isinstance(x, dict)]
        return []
