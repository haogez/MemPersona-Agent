from __future__ import annotations

from typing import Any, Dict, List, Optional

from mem_persona_agent.persona.schema import Persona


def _text_join(*parts: Optional[str]) -> str:
    return " ".join([p for p in parts if p]).lower()


def decide_memory_disclosure(
    persona: Persona,
    user_input: str,
    history: List[Dict[str, Any]],
    scene: Optional[str],
    candidates: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Runtime heuristic to decide whether to inject memories, and at what level of detail.
    Does not change persona schema; infers willingness from existing fields.
    """
    # A) persona-based guardedness
    guarded = 0.0
    bonus = 0.0
    pi = persona.personality or {}
    neuroticism = getattr(pi, "neuroticism", None) or pi.get("neuroticism", 0)
    extraversion = getattr(pi, "extraversion", None) or pi.get("extraversion", 0)
    if neuroticism >= 70:
        guarded += 0.25
    if extraversion >= 65:
        bonus += 0.15

    persona_text = _text_join(
        persona.values,
        persona.dislike,
        persona.social_pattern,
        persona.language_style,
        persona.living_habit,
    )
    guarded_keywords = ["害怕暴露", "不易示人", "秘密", "保持距离", "防备", "不信任", "躲避", "伪装"]
    for kw in guarded_keywords:
        if kw in persona_text:
            guarded += 0.1
    if "保持距离" in persona.social_pattern:
        guarded += 0.2
    if "直白" in persona.language_style or "直接" in persona.language_style:
        bonus += 0.05

    base_willingness = max(0.0, min(1.0, 0.6 - guarded + bonus))

    # B) memory heaviness from candidates
    heavy_keywords = ["排挤", "误解", "父母冷淡", "羞耻", "创伤", "崩溃", "孤立", "抑郁", "暴露", "脆弱", "争吵"]
    hits = 0
    count = 0
    for c in candidates[:3]:
        text = _text_join(c.get("text"), c.get("raw", {}).get("summary_text"), c.get("raw", {}).get("dialogue_text"))
        if not text:
            continue
        count += 1
        hits += sum(1 for kw in heavy_keywords if kw in text)
    memory_weight = min(1.0, (hits / max(1, count)) / 3.0) if count else 0.0

    # C) trust / scene
    trust = 0.5
    if history and len(history) >= 6:
        trust += 0.1
    polite_kw = ["请", "能不能", "方便说说", "好奇"]
    if any(kw in user_input for kw in polite_kw):
        trust += 0.1
    scene_privacy = 0.5
    scene_text = _text_join(scene, user_input)
    if any(kw in scene_text for kw in ["房间", "夜晚", "日记", "小公园", "私聊", "独处"]):
        scene_privacy += 0.2
    if any(kw in scene_text for kw in ["学校", "社团", "操场", "同学", "公开"]):
        scene_privacy -= 0.15
    trust = max(0.0, min(1.0, trust))
    scene_privacy = max(0.0, min(1.0, scene_privacy))

    # Aggregate willingness
    willingness = base_willingness * 0.6 + trust * 0.3 - memory_weight * 0.3 + scene_privacy * 0.1
    willingness = max(0.0, min(1.0, willingness))

    allow = willingness >= 0.35
    # Mode and caps
    if memory_weight > 0.6 or willingness < 0.55:
        mode = "summary"
        selected_cap = 2
        detail_budget_chars = 0
    else:
        mode = "detail"
        selected_cap = 2
        detail_budget_chars = 160
    # tighter for heavy memories
    if memory_weight >= 0.75:
        selected_cap = 1
        detail_budget_chars = 0
        mode = "summary"

    reason = "ok" if allow else "low_willingness"
    return {
        "allow_inject": allow,
        "mode": mode,
        "willingness": willingness,
        "trust": trust,
        "scene_privacy": scene_privacy,
        "memory_weight": memory_weight,
        "reason": reason,
        "selected_cap": selected_cap,
        "detail_budget_chars": detail_budget_chars,
    }
