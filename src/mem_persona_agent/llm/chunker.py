from __future__ import annotations

import time
from typing import Iterable, List

# Sentence-ending punctuation (ASCII + Chinese)
PUNCT = ("。", "！", "？", "!", "?", "\n", "…")


def chunk_tokens(tokens: Iterable[str], chunk_chars: int = 24) -> List[str]:
    """Deterministic chunking (no time) used in tests."""
    buf = ""
    chunks: List[str] = []
    for t in tokens:
        buf += t
        if len(buf) >= chunk_chars or any(p in buf for p in PUNCT):
            chunks.append(buf)
            buf = ""
    if buf:
        chunks.append(buf)
    return chunks


class LiveChunker:
    """Runtime chunker with time-based flushing."""

    def __init__(self, chunk_chars: int = 24, flush_ms: int = 100):
        self.chunk_chars = chunk_chars
        self.flush_ms = flush_ms
        self.buf = ""
        self.last_flush = time.perf_counter()

    def feed(self, token: str) -> str | None:
        self.buf += token
        now = time.perf_counter()
        should_flush = (
            len(self.buf) >= self.chunk_chars
            or any(p in self.buf for p in PUNCT)
            or (now - self.last_flush) * 1000 >= self.flush_ms
        )
        if should_flush and self.buf:
            out = self.buf
            self.buf = ""
            self.last_flush = now
            return out
        return None

    def flush_rest(self) -> str | None:
        if self.buf:
            out = self.buf
            self.buf = ""
            self.last_flush = time.perf_counter()
            return out
        return None
