from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Dict


@dataclass
class TurnIDManager:
    """Tracks active turn ids per character_id (in-memory, single-process)."""

    _active: Dict[str, str] = field(default_factory=dict)

    def start_turn(self, character_id: str) -> str:
        turn_id = str(uuid.uuid4())
        self._active[character_id] = turn_id
        return turn_id

    def is_active(self, character_id: str, turn_id: str) -> bool:
        return self._active.get(character_id) == turn_id
