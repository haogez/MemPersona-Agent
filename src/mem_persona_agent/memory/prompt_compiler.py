from __future__ import annotations

from typing import Any, Dict, List


def compile_scene_context(scene: Dict[str, Any]) -> str:
    if not scene:
        return ""
    summary = (
        scene.get("scene_gist")
        or scene.get("summary_7whr")
        or scene.get("summary")
        or ""
    )
    lines = [
        "[SCENE MEMORY | internal]",
        f"scene_id: {scene.get('scene_id')}",
        f"summary: {summary}",
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
