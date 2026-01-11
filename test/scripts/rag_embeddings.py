from __future__ import annotations

import hashlib
import re
from typing import Iterable, List

import numpy as np


def tokenize(text: str) -> List[str]:
    if not text:
        return []
    return [t for t in re.findall(r"[A-Za-z0-9]+|[\u4e00-\u9fff]+", text.lower()) if t]


def hash_token_to_index(token: str, dim: int) -> int:
    digest = hashlib.md5(token.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % dim


def hash_vectorize(text: str, dim: int = 1024) -> np.ndarray:
    vec = np.zeros(dim, dtype=np.float32)
    for tok in tokenize(text):
        idx = hash_token_to_index(tok, dim)
        vec[idx] += 1.0
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return 0.0
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def average_vectors(vectors: Iterable[np.ndarray]) -> np.ndarray:
    vectors = list(vectors)
    if not vectors:
        return np.zeros(0, dtype=np.float32)
    stacked = np.vstack(vectors)
    avg = stacked.mean(axis=0)
    norm = np.linalg.norm(avg)
    if norm > 0:
        avg = avg / norm
    return avg
