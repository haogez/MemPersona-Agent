from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

import logging

from mem_persona_agent.config import settings
from mem_persona_agent.llm import embed
from mem_persona_agent.llm.embedding import EmbedError
from mem_persona_agent.memory.graph_store import GraphStore

logger = logging.getLogger(__name__)


class MemoryRetriever:
    def __init__(self, store: GraphStore):
        self.store = store

    async def retrieve_scenes(
        self,
        character_id: str,
        user_input: str,
        place: Optional[str] = None,
        npc: Optional[str] = None,
        limit: int = 8,
    ) -> List[Dict[str, Any]]:
        candidates, _ = await self.retrieve_scene_candidates(
            character_id,
            user_input,
            place=place,
            npc=npc,
            limit=limit,
        )
        return candidates

    async def retrieve_scene_candidates(
        self,
        character_id: str,
        user_input: str,
        place: Optional[str] = None,
        npc: Optional[str] = None,
        limit: int = 8,
        *,
        vector_enabled: Optional[bool] = None,
        debug: bool = False,
        cache_only: bool = False,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        query_text = " ".join([p for p in [user_input, place, npc] if p]).strip()
        vector_results: List[Dict[str, Any]] = []
        vector_error: Optional[str] = None
        vector_error_detail: Optional[Dict[str, Any]] = None
        vector_attempted = False
        vector_enabled_flag = settings.enable_vector_retrieval if vector_enabled is None else vector_enabled
        if vector_enabled_flag:
            vector_attempted = True
            try:
                query_emb = await embed(query_text)
                vector_results = self.store.query_scene_vectors(character_id, query_emb, limit=limit)
            except EmbedError as exc:
                vector_error = str(exc)
                vector_error_detail = {
                    "where": "embeddings",
                    "status_code": exc.status_code,
                    "body": exc.response_text,
                    "message": vector_error,
                }
                logger.warning("scene vector embed failed: %s", vector_error)
            except Exception as exc:  # pragma: no cover - defensive
                vector_error = str(exc)
                logger.warning("scene vector retrieval failed: %s", vector_error)
        keyword_results = self.store.query_scene_keywords(character_id, query_text, limit=limit, cache_only=cache_only)
        merged = self._merge_scene_results(vector_results, keyword_results)
        fallback_mode = "keyword_only" if (not vector_enabled_flag or vector_error) else "hybrid"
        meta = {
            "vector_count": len(vector_results),
            "keyword_count": len(keyword_results),
            "query_text": query_text,
            "vector_error": vector_error,
            "vector_error_detail": vector_error_detail,
            "vector_enabled": vector_enabled_flag,
            "vector_attempted": vector_attempted,
            "fallback_mode": fallback_mode,
            "final_candidates_count": len(merged),
        }
        if debug:
            tokens = self._tokenize_keywords(query_text)
            meta["debug"] = {
                "summary": {
                    "vector_enabled": vector_enabled_flag,
                    "vector_attempted": vector_attempted,
                    "vector_error": vector_error,
                    "vector_error_detail": vector_error_detail,
                    "keyword_count": len(keyword_results),
                    "final_candidates_count": len(merged),
                    "fallback_mode": fallback_mode,
                },
                "query": {
                    "query_text": query_text,
                    "rewritten_query": None,
                    "used_query": query_text,
                    "tokens": tokens,
                    "fields": ["scene_gist", "anchors_text", "participants_text", "place_name", "life_stage"],
                },
                "keyword": {
                    "keyword_count": len(keyword_results),
                    "hits": self._keyword_hits(keyword_results, tokens),
                },
                "vector": {
                    "vector_count": len(vector_results),
                    "vector_enabled": vector_enabled_flag,
                    "vector_attempted": vector_attempted,
                    "vector_error": vector_error,
                    "topk": [
                        {
                            "scene_id": res.get("scene_id") or res.get("id"),
                            "score": float(res.get("score", 0.0)),
                            "vector_field": "embedding",
                        }
                        for res in vector_results
                    ],
                },
                "candidates": self._candidate_debug(merged, limit=20),
            }
        return merged, meta

    async def retrieve_details(
        self,
        character_id: str,
        scene_id: str,
        user_input: str,
        limit: int = 12,
    ) -> List[Dict[str, Any]]:
        return self.store.query_scene_detail(character_id, scene_id, user_input, limit=limit)

    async def retrieve_detail_pack(self, character_id: str, scene_id: str) -> Dict[str, Any]:
        return self.store.query_scene_detail_pack(character_id, scene_id)

    def detail_nodes_from_pack(self, pack: Dict[str, Any]) -> List[Dict[str, Any]]:
        return self.store._detail_nodes_from_pack(pack)

    async def retrieve(
        self,
        character_id: str,
        user_input: str,
        place: Optional[str] = None,
        npc: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        return await self.retrieve_scenes(character_id, user_input, place=place, npc=npc)

    def _merge_scene_results(self, vector_results: List[Dict[str, Any]], keyword_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        merged: Dict[str, Dict[str, Any]] = {}
        for res in vector_results:
            scene_id = res.get("scene_id") or res.get("id")
            if not scene_id:
                continue
            record = merged.setdefault(scene_id, {"scene": self._normalize_scene(res), "scene_id": scene_id})
            record["vector_score"] = float(res.get("score", 0.0))
        for res in keyword_results:
            scene_id = res.get("scene_id") or res.get("id")
            if not scene_id:
                continue
            record = merged.setdefault(scene_id, {"scene": self._normalize_scene(res), "scene_id": scene_id})
            record["keyword_score"] = float(res.get("score", 0.0))

        combined: List[Dict[str, Any]] = []
        for record in merged.values():
            vector_score = record.get("vector_score", 0.0)
            keyword_score = record.get("keyword_score", 0.0)
            score = vector_score * 0.7 + keyword_score * 0.3
            record["score"] = score
            combined.append(record)
        return sorted(combined, key=lambda x: x.get("score", 0.0), reverse=True)

    def _normalize_scene(self, scene: Dict[str, Any]) -> Dict[str, Any]:
        scene_copy = dict(scene)
        scene_copy.pop("score", None)
        if "scene_id" not in scene_copy and "id" in scene_copy:
            scene_copy["scene_id"] = scene_copy.get("id")
        scene_copy.pop("embedding", None)
        if "place" not in scene_copy:
            scene_copy["place"] = {
                "name": scene_copy.get("place_name"),
                "type": scene_copy.get("place_type"),
            }
        if "self_state" not in scene_copy:
            scene_copy["self_state"] = {
                "physical": scene_copy.get("self_state_physical"),
                "mental": scene_copy.get("self_state_mental"),
            }
        if "salience" not in scene_copy:
            scene_copy["salience"] = {
                "importance": scene_copy.get("salience_importance"),
                "emotional_intensity": scene_copy.get("salience_emotional_intensity"),
                "recall_probability": scene_copy.get("salience_recall_probability"),
            }
        for key in ["emotion", "anchors", "participants"]:
            val = scene_copy.get(key)
            if isinstance(val, str):
                scene_copy[key] = [val]
        return scene_copy

    def _tokenize_keywords(self, text: str) -> List[str]:
        if not text:
            return []
        tokens = re.findall(r"[A-Za-z0-9]+|[\u4e00-\u9fff]+", text.lower())
        return [t for t in tokens if len(t) >= 2]

    def _keyword_hits(self, results: List[Dict[str, Any]], tokens: List[str]) -> List[Dict[str, Any]]:
        hits: List[Dict[str, Any]] = []
        for res in results:
            scene_id = res.get("scene_id") or res.get("id")
            fields = self._scene_fields(res)
            matched_terms: List[str] = []
            matched_fields: List[str] = []
            for token in tokens:
                for field_name, field_val in fields.items():
                    if token and field_val and token in field_val:
                        if token not in matched_terms:
                            matched_terms.append(token)
                        if field_name not in matched_fields:
                            matched_fields.append(field_name)
            hits.append(
                {
                    "scene_id": scene_id,
                    "matched_terms": matched_terms,
                    "matched_fields": matched_fields,
                    "score": float(res.get("score", 0.0)),
                }
            )
        return hits

    def _scene_fields(self, scene: Dict[str, Any]) -> Dict[str, str]:
        place = scene.get("place") if isinstance(scene.get("place"), dict) else {}
        anchors = scene.get("anchors") or []
        participants = scene.get("participants") or []
        return {
            "scene_gist": str(scene.get("scene_gist") or ""),
            "anchors_text": " ".join([str(x) for x in anchors if x]),
            "participants_text": " ".join([str(x) for x in participants if x]),
            "place_name": str(place.get("name") or scene.get("place_name") or ""),
            "life_stage": str(scene.get("life_stage") or ""),
        }

    def _candidate_debug(self, merged: List[Dict[str, Any]], limit: int = 20) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for idx, record in enumerate(merged[:limit], start=1):
            vector_score = float(record.get("vector_score", 0.0))
            keyword_score = float(record.get("keyword_score", 0.0))
            if vector_score and keyword_score:
                reason_tag = "hybrid"
            elif vector_score:
                reason_tag = "vector"
            elif keyword_score:
                reason_tag = "keyword"
            else:
                reason_tag = "unknown"
            out.append(
                {
                    "scene_id": record.get("scene_id"),
                    "rank": idx,
                    "vector_score": vector_score,
                    "keyword_score": keyword_score,
                    "combined_score": float(record.get("score", 0.0)),
                    "reason_tag": reason_tag,
                }
            )
        return out

