from __future__ import annotations

from typing import Any, Dict, List


def select_top_memories(memories: List[Dict[str, Any]], k: int = 5) -> List[Dict[str, Any]]:
    return sorted(memories, key=lambda m: m.get("score", 0), reverse=True)[:k]
