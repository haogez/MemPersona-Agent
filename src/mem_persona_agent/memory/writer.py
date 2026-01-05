from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from mem_persona_agent.config import settings
from mem_persona_agent.llm import ChatClient
from mem_persona_agent.llm.prompts import (
    build_detail_pack_prompt,
    build_inspiration_prompt,
    build_scene_pack_prompt,
    build_worldrule_prompt,
)
from mem_persona_agent.memory.graph_store import GraphStore, normalize_anchor, normalize_text
from mem_persona_agent.persona.related import RelatedCharacterGenerator
from mem_persona_agent.utils import StatsCollector

logger = logging.getLogger(__name__)

SCENE_PACK_SIZE = settings.scene_count


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_json_loads(content: str) -> Any:
    try:
        return json.loads(content)
    except Exception:
        pass
    for start_char, end_char in [("{", "}"), ("[", "]")]:
        start = content.find(start_char)
        end = content.rfind(end_char)
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(content[start : end + 1])
            except Exception:
                continue
    return None


def _log_step(event: str, payload: Dict[str, Any]) -> None:
    logger.info(json.dumps({"event": f"memgen.step.{event}", **payload}, ensure_ascii=False))


class MemoryWriter:
    def __init__(self, store: GraphStore, client: ChatClient | None = None):
        self.store = store
        self.client = client or ChatClient()
        self.related_generator = RelatedCharacterGenerator(client=self.client)

    async def generate_and_store(
        self,
        character_id: str,
        persona: Dict[str, Any],
        *,
        seed: str | None = None,
        memory_config: Optional[Dict[str, Any]] = None,
        related_characters: Optional[List[Dict[str, Any]]] = None,
        stats: StatsCollector | None = None,
    ) -> Dict[str, Any]:
        stats = stats or StatsCollector()
        started = time.perf_counter()
        client = ChatClient(stats=stats)
        related_generator = RelatedCharacterGenerator(client=client, stats=stats)

        persona_name = normalize_text(str((persona or {}).get("name") or ""))
        short_id = character_id.replace("-", "")[:8]
        with stats.stage("persona"):
            stats.add_note("provided", True)

        logger.info("memory.generate.start cid=%s", character_id)
        logger.info("memory.generate.config cid=%s scene_pack=%s", character_id, SCENE_PACK_SIZE)

        # Step2: related_characters
        step_start = time.perf_counter()
        _log_step(
            "start",
            {
                "step_name": "related_characters",
                "owner_id": character_id,
                "refs": {"persona_name": persona_name},
                "inputs": {"seed_provided": bool(seed), "related_provided": bool(related_characters)},
            },
        )
        related_record = self.store.read_related_characters_record(character_id)
        if related_record and related_record.get("related_characters"):
            related_cleaned = self._normalize_related(related_record.get("related_characters") or [])
            rc_id = str(related_record.get("rc_id") or f"rc_{short_id}_{int(time.time())}")
            rc_created_at = str(related_record.get("created_at") or _now_iso())
            stats.add_note("skipped", True, stage="related_characters")
            skipped_related = True
        else:
            related = related_characters or []
            if not related:
                related = await related_generator.generate(seed or "", persona)
            related_cleaned = self._normalize_related(related)
            rc_id = f"rc_{short_id}_{int(time.time())}"
            rc_created_at = _now_iso()
            self.store.write_related_characters(character_id, related_cleaned, rc_id=rc_id, created_at=rc_created_at)
            skipped_related = False
        self.store.write_related_character_edges(character_id, persona_name, related_cleaned)
        _log_step(
            "end",
            {
                "step_name": "related_characters",
                "owner_id": character_id,
                "duration_ms": round((time.perf_counter() - step_start) * 1000.0, 2),
                "outputs": {"paths": [str(self.store.related_store_path)]},
                "counts": {"related": len(related_cleaned)},
                "skipped": skipped_related,
            },
        )

        # Step4: worldrule
        step_start = time.perf_counter()
        _log_step(
            "start",
            {"step_name": "worldrule", "owner_id": character_id, "refs": {"rc_id": rc_id}, "inputs": {}},
        )
        worldrule_record = self.store.read_worldrule_record(character_id)
        if worldrule_record and worldrule_record.get("worldrule"):
            worldrule = self._normalize_worldrule(worldrule_record.get("worldrule"))
            worldrule_id = str(worldrule_record.get("worldrule_id") or f"wr_{short_id}_{int(time.time())}")
            stats.add_note("skipped", True, stage="worldrule")
            skipped_worldrule = True
        else:
            with stats.stage("worldrule"):
                worldrule_raw = await self._call_json(
                    client,
                    build_worldrule_prompt(persona, related_cleaned),
                    temperature=0.3,
                    step_name="worldrule",
                )
            worldrule = self._normalize_worldrule(worldrule_raw)
            worldrule_id = f"wr_{short_id}_{int(time.time())}"
            worldrule_record = {
                "owner_id": character_id,
                "worldrule_id": worldrule_id,
                "created_at": _now_iso(),
                "worldrule": worldrule,
            }
            self.store.write_worldrule(character_id, worldrule_record)
            skipped_worldrule = False
        _log_step(
            "end",
            {
                "step_name": "worldrule",
                "owner_id": character_id,
                "duration_ms": round((time.perf_counter() - step_start) * 1000.0, 2),
                "outputs": {"paths": [str(self.store.worldrule_store_path)]},
                "counts": {"basic_rules": len(worldrule.get("basic_rules") or [])},
                "skipped": skipped_worldrule,
            },
        )

        # Step5: inspiration
        step_start = time.perf_counter()
        _log_step(
            "start",
            {
                "step_name": "inspiration",
                "owner_id": character_id,
                "refs": {"rc_id": rc_id, "worldrule_id": worldrule_id},
                "inputs": {},
            },
        )
        inspiration_record = self.store.read_inspiration_record(character_id)
        if inspiration_record and (inspiration_record.get("concept_pool") or inspiration_record.get("visual_fragments")):
            concept_pool, visual_fragments = self._normalize_inspiration(inspiration_record)
            insp_id = str(inspiration_record.get("insp_id") or f"insp_{short_id}_{int(time.time())}")
            stats.add_note("skipped", True, stage="inspiration")
            skipped_inspiration = True
        else:
            with stats.stage("inspiration"):
                inspiration_raw = await self._call_json(
                    client,
                    build_inspiration_prompt(persona, related_cleaned, worldrule),
                    temperature=0.4,
                    step_name="inspiration",
                )
            concept_pool, visual_fragments = self._normalize_inspiration(inspiration_raw)
            insp_id = f"insp_{short_id}_{int(time.time())}"
            inspiration_record = {
                "owner_id": character_id,
                "insp_id": insp_id,
                "created_at": _now_iso(),
                "refs": {"persona_id": character_id, "rc_id": rc_id, "worldrule_id": worldrule_id},
                "concept_pool": concept_pool,
                "visual_fragments": visual_fragments,
            }
            self.store.write_inspiration(character_id, inspiration_record)
            skipped_inspiration = False
        _log_step(
            "end",
            {
                "step_name": "inspiration",
                "owner_id": character_id,
                "duration_ms": round((time.perf_counter() - step_start) * 1000.0, 2),
                "outputs": {"paths": [str(self.store.inspiration_store_path)]},
                "counts": {"visual_fragments": len(visual_fragments)},
                "skipped": skipped_inspiration,
            },
        )

        # Step6: scene pack
        step_start = time.perf_counter()
        _log_step(
            "start",
            {
                "step_name": "scene_memories",
                "owner_id": character_id,
                "refs": {"rc_id": rc_id, "worldrule_id": worldrule_id, "insp_id": insp_id},
                "inputs": {"scene_count": SCENE_PACK_SIZE},
            },
        )
        scene_pack_record = self.store.read_scene_pack_record(character_id)
        if scene_pack_record and isinstance(scene_pack_record.get("scenes"), list) and scene_pack_record.get("scenes"):
            scenes = scene_pack_record.get("scenes") or []
            scene_pack_id = str(scene_pack_record.get("scene_pack_id") or f"scene_pack_{short_id}_{int(time.time())}")
            scene_pack = dict(scene_pack_record)
            scene_pack["scene_pack_id"] = scene_pack_id
            scene_pack["scenes"] = scenes
            self.store.write_scene_pack(character_id, scene_pack, append_record=False)
            stats.add_note("skipped", True, stage="scene_memories")
            skipped_scene_pack = True
        else:
            with stats.stage("scene_memories"):
                scene_raw = await self._call_json(
                    client,
                    build_scene_pack_prompt(
                        persona,
                        related_cleaned,
                        worldrule,
                        {"concept_pool": concept_pool, "visual_fragments": visual_fragments},
                        scene_count=SCENE_PACK_SIZE,
                    ),
                    temperature=0.4,
                    step_name="scene_memories",
                )
            scenes = self._normalize_scenes(scene_raw, persona_name, short_id)
            scene_pack_id = f"scene_pack_{short_id}_{int(time.time())}"
            scene_pack = {
                "owner_id": character_id,
                "scene_pack_id": scene_pack_id,
                "created_at": _now_iso(),
                "refs": {"rc_id": rc_id, "worldrule_id": worldrule_id, "insp_id": insp_id, "persona_ref": character_id},
                "scenes": scenes,
            }
            self.store.write_scene_pack(character_id, scene_pack)
            skipped_scene_pack = False
        _log_step(
            "end",
            {
                "step_name": "scene_memories",
                "owner_id": character_id,
                "duration_ms": round((time.perf_counter() - step_start) * 1000.0, 2),
                "outputs": {"paths": [str(self.store.scene_store_path)]},
                "counts": {"scenes": len(scenes)},
                "skipped": skipped_scene_pack,
            },
        )

        # Step7: detail graphs
        detail_records: List[Dict[str, Any]] = []
        for idx, scene in enumerate(scenes, start=1):
            scene_id = scene.get("scene_id")
            step_start = time.perf_counter()
            _log_step(
                "start",
                {
                    "step_name": "detail_graph",
                    "owner_id": character_id,
                    "refs": {"scene_pack_id": scene_pack_id, "scene_id": scene_id},
                    "inputs": {"scene_index": idx},
                },
            )
            detail_record = self.store.read_detail_pack_record(character_id, scene_id)
            if detail_record and isinstance(detail_record.get("events"), list) and detail_record.get("events"):
                detail_pack = detail_record
                stats.add_note("skipped", True, stage="detail_graph")
                skipped_detail = True
            else:
                with stats.stage("detail_graph"):
                    detail_raw = await self._call_json(
                        client,
                        build_detail_pack_prompt(
                            scene,
                            persona,
                            related_cleaned,
                            worldrule,
                            {"concept_pool": concept_pool, "visual_fragments": visual_fragments},
                        ),
                        temperature=0.4,
                        step_name="detail_graph",
                    )
                detail_pack = self._normalize_detail_pack(detail_raw, character_id, scene_id, scene_pack_id)
                skipped_detail = False
            detail_records.append(detail_pack)
            with stats.stage("neo4j_write"):
                self.store.write_detail_pack(
                    character_id,
                    detail_pack,
                    canonical_name=persona_name,
                    append_record=not skipped_detail,
                )
            _log_step(
                "end",
                {
                    "step_name": "detail_graph",
                    "owner_id": character_id,
                    "duration_ms": round((time.perf_counter() - step_start) * 1000.0, 2),
                    "outputs": {"paths": [str(self.store.detail_store_path)]},
                    "counts": {
                        "events": len(detail_pack.get("events") or []),
                        "causal_edges": len(detail_pack.get("causal_edges") or []),
                    },
                    "skipped": skipped_detail,
                },
            )

        stats.log_summary()
        totals = stats.totals()
        elapsed_ms = round((time.perf_counter() - started) * 1000.0, 2)
        stats_payload = {
            "total_scenes": len(scenes),
            "elapsed_ms": elapsed_ms,
            "llm_calls": totals.get("llm_calls", 0),
            "input_tokens": totals.get("prompt_tokens", 0),
            "output_tokens": totals.get("completion_tokens", 0),
            "total_tokens": totals.get("total_tokens", 0),
            "cost_usd": totals.get("cost_usd", 0.0),
            "stages": stats.to_dict(),
        }
        return {"sequence": [s.get("summary_7whr") or s.get("scene_gist") for s in scenes], "scenes": scenes, "stats": stats_payload}

    async def _call_json(
        self,
        client: ChatClient,
        messages: List[Dict[str, str]],
        *,
        temperature: float,
        step_name: str,
    ) -> Any:
        try:
            content = await client.chat(messages, temperature=temperature)
        except Exception as exc:
            logger.warning("memgen.%s chat failed: %s", step_name, exc)
            return None
        data = _safe_json_loads(content)
        if data is not None:
            return data
        retry_messages = [*messages, {"role": "user", "content": "只输出严格 JSON，不要解释。"}]
        try:
            content_retry = await client.chat(retry_messages, temperature=temperature)
        except Exception as exc:
            logger.warning("memgen.%s chat retry failed: %s", step_name, exc)
            return None
        return _safe_json_loads(content_retry)

    def _normalize_related(self, related: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        cleaned: List[Dict[str, Any]] = []
        for item in related or []:
            if not isinstance(item, dict):
                continue
            name = normalize_text(item.get("name") or "")
            if not name:
                continue
            relation = normalize_text(item.get("relation") or "")
            role = normalize_text(item.get("role") or item.get("description") or "")
            cleaned.append({"name": name, "relation": relation, "role": role})
        return cleaned

    def _normalize_worldrule(self, data: Any) -> Dict[str, Any]:
        if not isinstance(data, dict):
            return {"era_state": "", "society_state": "", "basic_rules": [], "geo_scope": "", "tech_level": ""}
        return {
            "era_state": normalize_text(data.get("era_state") or ""),
            "society_state": normalize_text(data.get("society_state") or ""),
            "basic_rules": [
                normalize_text(val) for val in (data.get("basic_rules") or []) if isinstance(val, str)
            ],
            "geo_scope": normalize_text(data.get("geo_scope") or ""),
            "tech_level": normalize_text(data.get("tech_level") or ""),
        }

    def _normalize_inspiration(self, data: Any) -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
        if not isinstance(data, dict):
            return {"places": [], "orgs": [], "events": [], "social_facts": []}, []
        pool = data.get("concept_pool") if isinstance(data.get("concept_pool"), dict) else {}
        concept_pool = {
            "places": [
                normalize_text(val) for val in (pool.get("places") or []) if isinstance(val, str)
            ],
            "orgs": [normalize_text(val) for val in (pool.get("orgs") or []) if isinstance(val, str)],
            "events": [normalize_text(val) for val in (pool.get("events") or []) if isinstance(val, str)],
            "social_facts": [
                normalize_text(val) for val in (pool.get("social_facts") or []) if isinstance(val, str)
            ],
        }
        visual_fragments: List[Dict[str, Any]] = []
        for idx, item in enumerate(data.get("visual_fragments") or [], start=1):
            if not isinstance(item, dict):
                continue
            vf_id = str(item.get("vf_id") or f"vf_{idx:02d}")
            text = normalize_text(item.get("text") or "")
            tags = [normalize_text(val) for val in (item.get("tags") or []) if isinstance(val, str)]
            visual_fragments.append({"vf_id": vf_id, "text": text, "tags": tags})
        return concept_pool, visual_fragments

    def _normalize_scenes(self, data: Any, persona_name: str, short_id: str) -> List[Dict[str, Any]]:
        scenes_raw = []
        if isinstance(data, dict) and isinstance(data.get("scenes"), list):
            scenes_raw = data.get("scenes") or []
        elif isinstance(data, list):
            scenes_raw = data
        scenes: List[Dict[str, Any]] = []
        for idx, item in enumerate(scenes_raw, start=1):
            if not isinstance(item, dict):
                continue
            scene_id = str(item.get("scene_id") or f"scene_{idx:03d}_{short_id}")
            summary = normalize_text(item.get("summary_7whr") or item.get("summary") or "")
            who = [
                normalize_text(val)
                for val in (item.get("who") or item.get("participants") or [])
                if isinstance(val, str) and normalize_text(val)
            ]
            if persona_name and persona_name not in who:
                who.insert(0, persona_name)
            when = item.get("when") if isinstance(item.get("when"), dict) else {"time_point": item.get("when") or ""}
            where = item.get("where") if isinstance(item.get("where"), dict) else {"name": item.get("where") or ""}
            keywords = [
                normalize_text(val)
                for val in (item.get("keywords") or [])
                if isinstance(val, str) and normalize_text(val)
            ]
            anchors = [normalize_anchor(normalize_text(val)) for val in (item.get("anchors") or keywords) if isinstance(val, str)]
            salience = item.get("salience") if isinstance(item.get("salience"), dict) else {}
            scene = {
                "scene_id": scene_id,
                "summary_7whr": summary,
                "who": who,
                "when": {
                    "time_point": normalize_text(when.get("time_point") or when.get("value") or ""),
                    "time_hint": normalize_text(when.get("time_hint") or ""),
                },
                "where": {
                    "name": normalize_text(where.get("name") or where.get("place_name") or ""),
                    "type": normalize_text(where.get("type") or where.get("place_type") or ""),
                },
                "keywords": keywords,
                "anchors": anchors,
                "salience": {
                    "importance": salience.get("importance"),
                    "emotional_intensity": salience.get("emotional_intensity"),
                },
                "scene_gist": summary,
                "participants": who,
                "place": {
                    "name": normalize_text(where.get("name") or where.get("place_name") or ""),
                    "type": normalize_text(where.get("type") or where.get("place_type") or ""),
                },
                "created_at": _now_iso(),
                "event_root_ref": scene_id,
            }
            scenes.append(scene)
            if len(scenes) >= SCENE_PACK_SIZE:
                break
        return scenes

    def _normalize_detail_pack(
        self, data: Any, owner_id: str, scene_id: str, scene_pack_id: str
    ) -> Dict[str, Any]:
        if not isinstance(data, dict):
            data = {}
        events_raw = data.get("events") if isinstance(data.get("events"), list) else []
        causal_raw = data.get("causal_edges") if isinstance(data.get("causal_edges"), list) else []
        events: List[Dict[str, Any]] = []
        for idx, item in enumerate(events_raw, start=1):
            if not isinstance(item, dict):
                continue
            event_id = str(item.get("event_id") or f"{scene_id}_event_{idx:02d}")
            dialogue_raw = item.get("dialogue") if isinstance(item.get("dialogue"), list) else []
            dialogue: List[Dict[str, Any]] = []
            for didx, d in enumerate(dialogue_raw, start=1):
                if not isinstance(d, dict):
                    continue
                dialogue.append(
                    {
                        "utt_id": str(d.get("utt_id") or f"{event_id}_utt_{didx:02d}"),
                        "order": int(d.get("order") or didx),
                        "speaker": normalize_text(d.get("speaker") or ""),
                        "text": normalize_text(d.get("text") or ""),
                    }
                )
            place = item.get("place") if isinstance(item.get("place"), dict) else {}
            place_name = normalize_text(place.get("name") or "")
            place_type = normalize_text(place.get("type") or "")
            events.append(
                {
                    "event_id": event_id,
                    "order": int(item.get("order") or idx),
                    "phase": normalize_text(item.get("phase") or ""),
                    "event_text": normalize_text(item.get("event_text") or ""),
                    "time_point": normalize_text(item.get("time_point") or ""),
                    "place": {"name": place_name, "type": place_type} if (place_name or place_type) else {},
                    "participants": [
                        normalize_text(val)
                        for val in (item.get("participants") or [])
                        if isinstance(val, str) and normalize_text(val)
                    ],
                    "objects": [
                        normalize_text(val)
                        for val in (item.get("objects") or [])
                        if isinstance(val, str) and normalize_text(val)
                    ],
                    "dialogue": dialogue,
                }
            )
        causal_edges: List[Dict[str, Any]] = []
        for item in causal_raw:
            if not isinstance(item, dict):
                continue
            causal_edges.append(
                {"from": item.get("from"), "to": item.get("to"), "type": item.get("type") or "CAUSES"}
            )
        return {
            "owner_id": owner_id,
            "scene_id": scene_id,
            "detail_id": f"detail_{scene_id}",
            "created_at": _now_iso(),
            "refs": {"scene_pack_id": scene_pack_id},
            "events": events,
            "causal_edges": causal_edges,
        }
