from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from mem_persona_agent.config import settings


def detect_open_slots(user_input: str) -> List[str]:
    text = user_input or ""
    slots: List[str] = []
    patterns = {
        "who": r"(who|\u8c01|\u54ea\u4f4d|\u662f\u8c01|\u54ea\u4e9b\u4eba)",
        "why": r"(why|\u4e3a\u4f55|\u4e3a\u4ec0\u4e48|\u56e0\u4e3a\u4ec0\u4e48|\u539f\u56e0)",
        "what": r"(what|\u4ec0\u4e48|\u53d1\u751f\u4e86\u4ec0\u4e48|\u600e\u4e48\u56de\u4e8b)",
        "how": r"(how|\u600e\u4e48|\u5982\u4f55|\u600e\u6837)",
        "detail": r"(\u7ec6\u8282|\u5177\u4f53|\u8be6\u7ec6|more detail|details)",
    }
    for slot, pat in patterns.items():
        if re.search(pat, text, re.IGNORECASE):
            slots.append(slot)
    return slots


def should_recall_scene(
    user_input: str,
    candidates: List[Dict[str, Any]],
    *,
    score_threshold: Optional[float] = None,
) -> Dict[str, Any]:
    text = (user_input or "").lower()
    threshold = settings.scene_score_threshold if score_threshold is None else score_threshold
    top_score = max((c.get("score", 0.0) for c in candidates), default=0.0)
    gate_trace: List[Dict[str, Any]] = []

    smalltalk_hits = [k for k in settings.smalltalk_keywords if k and k.lower() in text]
    gate_trace.append(
        {
            "rule_name": "smalltalk_keyword",
            "features": {"hits": smalltalk_hits, "text_len": len(text)},
            "score": top_score,
            "threshold": threshold,
            "fired": bool(smalltalk_hits),
        }
    )
    if smalltalk_hits:
        return {"recall": False, "reason": "smalltalk", "top_score": top_score, "gate_trace": gate_trace}

    trigger_hits = [k for k in settings.memory_trigger_keywords if k and k.lower() in text]
    gate_trace.append(
        {
            "rule_name": "memory_trigger_keyword",
            "features": {"hits": trigger_hits, "text_len": len(text)},
            "score": top_score,
            "threshold": threshold,
            "fired": bool(trigger_hits),
        }
    )
    if trigger_hits:
        return {"recall": True, "reason": "trigger_keyword", "top_score": top_score, "gate_trace": gate_trace}

    gate_trace.append(
        {
            "rule_name": "score_threshold",
            "features": {"candidate_count": len(candidates)},
            "score": top_score,
            "threshold": threshold,
            "fired": top_score >= threshold,
        }
    )
    if top_score >= threshold:
        return {"recall": True, "reason": "score_threshold", "top_score": top_score, "gate_trace": gate_trace}

    return {"recall": False, "reason": "low_score", "top_score": top_score, "gate_trace": gate_trace}

