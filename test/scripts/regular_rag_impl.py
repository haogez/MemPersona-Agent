from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from .eval_utils import scene_id_from_sample
from .rag_embeddings import hash_vectorize
from .utils import hf_split_text


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def chunk_text(text: str, max_tokens: int) -> List[str]:
    if not text:
        return []
    if estimate_tokens(text) <= max_tokens:
        return [text]
    sentences = hf_split_text(text)
    chunks: List[str] = []
    buf: List[str] = []
    buf_tokens = 0
    for sent in sentences:
        tok = estimate_tokens(sent)
        if buf and buf_tokens + tok > max_tokens:
            chunks.append(" ".join(buf))
            buf = [sent]
            buf_tokens = tok
        else:
            buf.append(sent)
            buf_tokens += tok
    if buf:
        chunks.append(" ".join(buf))
    return chunks


@dataclass
class RegularRagChunk:
    text: str
    vector: np.ndarray


class RegularRagIndex:
    def __init__(self, chunks_by_scene: Dict[str, List[RegularRagChunk]], dim: int, chunk_tokens: int) -> None:
        self.chunks_by_scene = chunks_by_scene
        self.dim = dim
        self.chunk_tokens = chunk_tokens

    @classmethod
    def load_or_build(
        cls,
        dataset: Any,
        context_key: str | None,
        cache_path: Path,
        *,
        dim: int,
        chunk_tokens: int,
        rebuild: bool = False,
    ) -> "RegularRagIndex":
        if cache_path.exists() and not rebuild:
            with cache_path.open("rb") as f:
                payload = pickle.load(f)
            if payload.get("dim") == dim and payload.get("chunk_tokens") == chunk_tokens:
                chunks_by_scene = {
                    sid: [RegularRagChunk(text=c["text"], vector=np.array(c["vector"], dtype=np.float32)) for c in chunks]
                    for sid, chunks in payload.get("data", {}).items()
                }
                return cls(chunks_by_scene, dim=dim, chunk_tokens=chunk_tokens)
        chunks_by_scene: Dict[str, List[RegularRagChunk]] = {}
        for idx, sample in enumerate(dataset):
            sid = scene_id_from_sample(sample, idx)
            if not isinstance(sample, dict):
                continue
            ctx = str(sample.get(context_key) or "") if context_key else ""
            if not ctx:
                continue
            chunks = chunk_text(ctx, max_tokens=chunk_tokens)
            chunk_objs = [RegularRagChunk(text=chunk, vector=hash_vectorize(chunk, dim)) for chunk in chunks]
            if chunk_objs:
                chunks_by_scene[sid] = chunk_objs
        payload = {
            "dim": dim,
            "chunk_tokens": chunk_tokens,
            "data": {
                sid: [{"text": chunk.text, "vector": chunk.vector.tolist()} for chunk in chunks]
                for sid, chunks in chunks_by_scene.items()
            },
        }
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with cache_path.open("wb") as f:
            pickle.dump(payload, f)
        return cls(chunks_by_scene, dim=dim, chunk_tokens=chunk_tokens)

    def search(self, question: str, scene_id: str, top_k: int) -> List[Tuple[int, float, str]]:
        chunks = self.chunks_by_scene.get(scene_id, [])
        if not chunks:
            return []
        q_vec = hash_vectorize(question, self.dim)
        scored: List[Tuple[int, float, str]] = []
        for idx, chunk in enumerate(chunks):
            score = float(np.dot(q_vec, chunk.vector))
            scored.append((idx, score, chunk.text))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[: max(1, top_k)]
