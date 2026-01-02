from __future__ import annotations

from typing import Any, Dict, List


def compile_scene_context(scene: Dict[str, Any]) -> str:
    if not scene:
        return ""
    place = scene.get("place") or {}
    self_state = scene.get("self_state") or {}
    salience = scene.get("salience") or {}
    lines = [
        "[SCENE MEMORY | internal]",
        f"scene_id: {scene.get('scene_id')}",
        f"life_stage: {scene.get('life_stage')} | time_range: {scene.get('time_range')}",
        f"place: {place.get('name')} ({place.get('type')})",
        f"participants: {', '.join(scene.get('participants') or [])}",
        f"scene_gist: {scene.get('scene_gist')}",
        f"self_state: physical={self_state.get('physical')}; mental={self_state.get('mental')}",
        f"emotion: {', '.join(scene.get('emotion') or [])}",
        "salience: "
        f"importance={salience.get('importance')}; emotional_intensity={salience.get('emotional_intensity')}; recall_probability={salience.get('recall_probability')}",
        f"anchors: {', '.join(scene.get('anchors') or [])}",
        "[/SCENE MEMORY]",
    ]
    return "\n".join([l for l in lines if l])


def compile_detail_context(nodes: List[Dict[str, Any]], max_items: int = 10) -> str:
    if not nodes:
        return ""
    lines = [
        "[DETAIL MEMORY | internal]",
        "Use only after scene selection. Keep details concise and grounded.",
    ]
    for node in nodes[:max_items]:
        node_type = node.get("type") or node.get("node_type")
        if node_type == "Event":
            text = node.get("summary") or node.get("description") or node.get("label")
            if text:
                lines.append(f"- Event: {text}")
        elif node_type == "Action":
            verb = node.get("verb") or node.get("action") or "action"
            lines.append(f"- Action: {verb}")
        elif node_type == "Utterance":
            speaker = node.get("speaker") or node.get("name") or "someone"
            text = node.get("text") or node.get("content")
            if text:
                lines.append(f"- Utterance: {speaker}: {text}")
        elif node_type == "Person":
            name = node.get("name")
            role = node.get("role") or ""
            if name:
                lines.append(f"- Person: {name} {role}".strip())
        elif node_type == "Place":
            name = node.get("name")
            if name:
                lines.append(f"- Place: {name}")
        elif node_type == "Object":
            name = node.get("name") or node.get("value")
            if name:
                lines.append(f"- Object: {name}")
    lines.append("[/DETAIL MEMORY]")
    return "\n".join(lines)

