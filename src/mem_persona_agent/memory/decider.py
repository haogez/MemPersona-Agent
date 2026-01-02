from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple


@dataclass
class DeciderResult:
    selected: List[Dict[str, Any]]
    rejected: List[Dict[str, Any]]
    strategy: str
    budget_chars: int
    selected_ids: List[str]


class MemoryDecider:
    """Rule-based decider for selecting memories to inject."""

    def __init__(self, top_k: int = 5, budget_chars: int = 600, min_score: float = 0.0):
        self.top_k = top_k
        self.budget_chars = budget_chars
        self.min_score = min_score

    def decide(
        self,
        character_id: str,
        user_input: str,
        candidates: List[Dict[str, Any]],
    ) -> DeciderResult:
        selected: List[Dict[str, Any]] = []
        rejected: List[Dict[str, Any]] = []
        remaining = self.budget_chars

        seen_ids = set()
        seen_text = set()

        # filter wrong owner defensively
        filtered = []
        for c in candidates:
            raw = c.get("raw", {}) or {}
            owner = raw.get("owner_id") or raw.get("character_id")
            cid_match = (not owner) or owner == character_id
            if not cid_match:
                rejected.append({"candidate": c, "reason": "owner_mismatch"})
                continue
            if c.get("score", 0.0) < self.min_score:
                rejected.append({"candidate": c, "reason": "low_score"})
                continue
            # dedup by id then text
            cid = raw.get("id")
            txt = c.get("text", "")
            if cid and cid in seen_ids:
                rejected.append({"candidate": c, "reason": "dup"})
                continue
            if not cid and txt in seen_text:
                rejected.append({"candidate": c, "reason": "dup"})
                continue
            filtered.append(c)
            if cid:
                seen_ids.add(cid)
            if txt:
                seen_text.add(txt)

        # sort by score desc
        filtered = sorted(filtered, key=lambda x: x.get("score", 0.0), reverse=True)
        for c in filtered[: self.top_k]:
            text = c.get("text", "") or ""
            cost = len(text)
            if cost > remaining:
                rejected.append({"candidate": c, "reason": "budget_exceeded"})
                continue
            selected.append(c)
            remaining -= cost

        strategy = "none" if not selected else ("short" if len(selected) == 1 else ("medium" if len(selected) <= 3 else "long"))
        selected_ids = [c.get("raw", {}).get("id") for c in selected if c.get("raw")]
        return DeciderResult(selected=selected, rejected=rejected, strategy=strategy, budget_chars=self.budget_chars, selected_ids=selected_ids)
