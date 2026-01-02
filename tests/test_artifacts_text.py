from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from mem_persona_agent.persona.related import is_valid_cn_person_name
LATIN1_RE = re.compile(r"[\u00c0-\u00ff]")
GARBLED_RE = re.compile(r"\ufffd")


def _find_artifact_path(filename: str) -> Path | None:
    candidates = [Path("artifacts") / filename, Path("src") / "artifacts" / filename]
    for path in candidates:
        if path.exists():
            return path
    return None


def _read_jsonl_records(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        return []
    records: list[dict] = []
    for block in text.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        try:
            obj = json.loads(block)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            records.append(obj)
    if records:
        return records
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            records.append(obj)
    return records


def _find_persona_name() -> str | None:
    path = _find_artifact_path("personas.jsonl")
    if not path:
        return None
    records = _read_jsonl_records(path)
    if not records:
        return None
    persona = records[-1].get("persona") or {}
    name = persona.get("name")
    return name if isinstance(name, str) and name.strip() else None


def _assert_no_garbled(text: str) -> None:
    assert not GARBLED_RE.search(text)
    assert not LATIN1_RE.search(text)


def test_related_character_names_are_valid() -> None:
    path = _find_artifact_path("related_characters.jsonl")
    if not path:
        pytest.skip("related_characters.jsonl not found")
    text = path.read_text(encoding="utf-8")
    _assert_no_garbled(text)
    records = _read_jsonl_records(path)
    assert records
    record = records[-1]
    related = record.get("related_characters") or []
    for item in related:
        if not isinstance(item, dict):
            continue
        name = item.get("name") or ""
        assert is_valid_cn_person_name(name)


def test_scene_memories_protagonist_consistency() -> None:
    path = _find_artifact_path("scene_memories.jsonl")
    if not path:
        pytest.skip("scene_memories.jsonl not found")
    text = path.read_text(encoding="utf-8")
    _assert_no_garbled(text)
    persona_name = _find_persona_name()
    if not persona_name:
        pytest.skip("persona name not found")
    records = _read_jsonl_records(path)
    assert records
    record = records[-1]
    scenes = record.get("scenes") or []
    for scene in scenes:
        participants = scene.get("participants") or []
        assert persona_name in participants


def test_detail_graphs_text_clean() -> None:
    path = _find_artifact_path("scene_detail_graphs.jsonl")
    if not path:
        pytest.skip("scene_detail_graphs.jsonl not found")
    text = path.read_text(encoding="utf-8")
    _assert_no_garbled(text)
    records = _read_jsonl_records(path)
    if not records:
        pytest.skip("scene_detail_graphs.jsonl empty")
    record = records[-1]
    events = record.get("events") or []
    for event in events:
        if not isinstance(event, dict):
            continue
        for key in ["event_text", "time_point"]:
            val = event.get(key)
            if isinstance(val, str):
                _assert_no_garbled(val)
        place = event.get("place") or {}
        if isinstance(place, dict):
            for key in ["name", "type"]:
                val = place.get(key)
                if isinstance(val, str):
                    _assert_no_garbled(val)
        for val in event.get("participants") or []:
            if isinstance(val, str):
                _assert_no_garbled(val)
        for val in event.get("objects") or []:
            if isinstance(val, str):
                _assert_no_garbled(val)
        dialogue = event.get("dialogue") or []
        for d in dialogue:
            if not isinstance(d, dict):
                continue
            for key in ["speaker", "text"]:
                val = d.get(key)
                if isinstance(val, str):
                    _assert_no_garbled(val)
