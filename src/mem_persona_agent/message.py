from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Dict


@dataclass
class Message:
    role: str
    content: str


class MemoryBuffer:
    def __init__(self, max_turns: int = 6):
        self.max_turns = max_turns
        self.buffer: Deque[Message] = deque(maxlen=max_turns * 2)

    def add_user(self, content: str):
        self.buffer.append(Message(role="user", content=content))

    def add_assistant(self, content: str):
        self.buffer.append(Message(role="assistant", content=content))

    def history(self) -> List[Dict[str, str]]:
        return [message.__dict__ for message in self.buffer]

    def reset(self):
        self.buffer.clear()
