from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

from .rag_embeddings import tokenize


@dataclass
class SceneGraph:
    scene_id: str
    events: List[Dict[str, str]]
    details: List[Dict[str, str]]
    edges: List[Dict[str, str]]

    def event_lookup(self) -> Dict[str, Dict[str, str]]:
        return {ev["event_id"]: ev for ev in self.events}


def _build_edges(events: List[Dict[str, str]], causal: List[List]) -> List[Dict[str, str]]:
    edges: List[Dict[str, str]] = []
    for entry in causal or []:
        if not isinstance(entry, (list, tuple)) or len(entry) < 2:
            continue
        a_idx = int(entry[0])
        b_idx = int(entry[1])
        rel = str(entry[2]) if len(entry) > 2 else "precede"
        if 0 <= a_idx < len(events) and 0 <= b_idx < len(events):
            edges.append(
                {
                    "from_id": events[a_idx]["event_id"],
                    "to_id": events[b_idx]["event_id"],
                    "rel": rel,
                }
            )
    # add sequential edges
    for i in range(len(events) - 1):
        edges.append(
            {
                "from_id": events[i]["event_id"],
                "to_id": events[i + 1]["event_id"],
                "rel": "next",
            }
        )
    return edges


def load_graph_snapshot(path: Path) -> Dict[str, SceneGraph]:
    if not path.exists():
        raise FileNotFoundError(f"Graph snapshot not found: {path}")
    out: Dict[str, SceneGraph] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            scene = obj.get("scene") or {}
            scene_id = str(scene.get("scene_id") or "")
            events = obj.get("events") or []
            details = obj.get("details") or []
            causal = obj.get("causal") or []
            if not scene_id or not events:
                continue
            edges = _build_edges(events, causal)
            out[scene_id] = SceneGraph(scene_id=scene_id, events=events, details=details, edges=edges)
    return out


def event_token_overlap_score(question_tokens: List[str], event_text: str) -> int:
    if not question_tokens or not event_text:
        return 0
    ev_tokens = set(tokenize(event_text))
    return sum(1 for tok in question_tokens if tok in ev_tokens)


def detail_token_overlap_score(question_tokens: List[str], detail_text: str) -> int:
    if not question_tokens or not detail_text:
        return 0
    dt_tokens = set(tokenize(detail_text))
    return sum(1 for tok in question_tokens if tok in dt_tokens)


def expand_neighbors(edges: List[Dict[str, str]], seed_ids: List[str], depth: int) -> List[str]:
    if depth <= 0:
        return []
    graph: Dict[str, List[str]] = {}
    for edge in edges:
        a = edge["from_id"]
        b = edge["to_id"]
        graph.setdefault(a, []).append(b)
        graph.setdefault(b, []).append(a)
    visited = set(seed_ids)
    frontier = list(seed_ids)
    for _ in range(depth):
        nxt: List[str] = []
        for node in frontier:
            for nei in graph.get(node, []):
                if nei not in visited:
                    visited.add(nei)
                    nxt.append(nei)
        frontier = nxt
        if not frontier:
            break
    return list(visited)
