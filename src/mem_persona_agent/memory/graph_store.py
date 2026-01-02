from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from neo4j import GraphDatabase, Driver

from mem_persona_agent.config import settings

logger = logging.getLogger(__name__)


def _read_records(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    records: List[Dict[str, Any]] = []
    for block in text.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        try:
            records.append(json.loads(block))
        except json.JSONDecodeError:
            continue
    if not records:
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def _write_records(path: Path, records: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not records:
        with path.open("w", encoding="utf-8", newline="\n") as fh:
            fh.write("")
        return
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False))
            fh.write("\n")


def _append_record(path: Path, record: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(record, ensure_ascii=False))
        fh.write("\n")


def _tokenize_keywords(text: str) -> List[str]:
    if not text:
        return []
    tokens = re.findall(r"[A-Za-z0-9]+|[\u4e00-\u9fff]+", text.lower())
    return [t for t in tokens if len(t) >= 2]


def _safe_label(value: str, default: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_]", "_", value or "").strip("_")
    return cleaned or default


def normalize_text(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text:
        return text
    try:
        fixed = text.encode("latin1").decode("utf-8")
    except Exception:
        return text
    if re.search(r"[\u4e00-\u9fff]", fixed):
        return fixed
    return text


def normalize_anchor(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    return value.strip().upper()


@dataclass
class GraphStore:
    driver: Optional[Driver] = field(default=None)
    scene_cache: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    detail_cache: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    keyword_index_cache: Dict[str, Dict[str, List[str]]] = field(default_factory=dict)
    persona_cache: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    persona_store_path: Path = field(default_factory=lambda: Path(settings.persona_store_path))
    related_store_path: Path = field(default_factory=lambda: Path(settings.related_store_path))
    worldrule_store_path: Path = field(default_factory=lambda: Path(settings.worldrule_store_path))
    inspiration_store_path: Path = field(default_factory=lambda: Path(settings.inspiration_store_path))
    scene_store_path: Path = field(default_factory=lambda: Path(settings.scene_memory_store_path))
    keyword_index_path: Path = field(default_factory=lambda: Path(settings.scene_keyword_index_path))
    detail_store_path: Path = field(default_factory=lambda: Path(settings.detail_graph_store_path))
    _keyword_index_loaded: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        if settings.neo4j_available:
            try:
                self.driver = GraphDatabase.driver(
                    settings.neo4j_uri,
                    auth=(settings.neo4j_username, settings.neo4j_password),
                )
            except Exception as exc:  # pragma: no cover - safety
                logger.warning("Neo4j connection failed, using in-memory store: %s", exc)
                self.driver = None
        try:
            self._load_keyword_index()
            self._load_scene_cache()
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("Scene cache preload failed: %s", exc)

    def ensure_schema(self) -> None:
        if not self.driver:
            return
        expected_dims = settings.embed_dimensions if settings.embed_dimensions > 0 else 1536
        scene_embedding_index_query = f"""
                CREATE VECTOR INDEX scene_embedding_index IF NOT EXISTS
                FOR (s:SceneMemory) ON (s.embedding)
                OPTIONS {{
                  indexConfig: {{
                    `vector.dimensions`: {expected_dims},
                    `vector.similarity_function`: 'cosine'
                  }}
                }};
                """
        queries = [
            (
                "character_id_unique",
                """
                CREATE CONSTRAINT character_id_unique IF NOT EXISTS
                FOR (c:Character) REQUIRE c.id IS UNIQUE;
                """,
                None,
            ),
            (
                "character_owner_name_unique",
                """
                CREATE CONSTRAINT character_owner_name_unique IF NOT EXISTS
                FOR (c:Character) REQUIRE (c.owner_id, c.name) IS UNIQUE;
                """,
                """
                CREATE INDEX character_owner_name_idx IF NOT EXISTS
                FOR (c:Character) ON (c.owner_id, c.name);
                """,
            ),
            (
                "scene_id_unique",
                """
                CREATE CONSTRAINT scene_id_unique IF NOT EXISTS
                FOR (s:SceneMemory) REQUIRE s.scene_id IS UNIQUE;
                """,
                None,
            ),
            (
                "event_id_unique",
                """
                CREATE CONSTRAINT event_id_unique IF NOT EXISTS
                FOR (e:Event) REQUIRE e.event_id IS UNIQUE;
                """,
                None,
            ),
            (
                "utterance_id_unique",
                """
                CREATE CONSTRAINT utterance_id_unique IF NOT EXISTS
                FOR (u:Utterance) REQUIRE u.utt_id IS UNIQUE;
                """,
                None,
            ),
            (
                "person_owner_name_unique",
                """
                CREATE CONSTRAINT person_owner_name_unique IF NOT EXISTS
                FOR (p:Person) REQUIRE (p.owner_id, p.name) IS UNIQUE;
                """,
                """
                CREATE INDEX person_owner_name_idx IF NOT EXISTS
                FOR (p:Person) ON (p.owner_id, p.name);
                """,
            ),
            (
                "place_owner_name_unique",
                """
                CREATE CONSTRAINT place_owner_name_unique IF NOT EXISTS
                FOR (pl:Place) REQUIRE (pl.owner_id, pl.name) IS UNIQUE;
                """,
                """
                CREATE INDEX place_owner_name_idx IF NOT EXISTS
                FOR (pl:Place) ON (pl.owner_id, pl.name);
                """,
            ),
            (
                "timepoint_owner_value_unique",
                """
                CREATE CONSTRAINT timepoint_owner_value_unique IF NOT EXISTS
                FOR (t:TimePoint) REQUIRE (t.owner_id, t.value) IS UNIQUE;
                """,
                """
                CREATE INDEX timepoint_owner_value_idx IF NOT EXISTS
                FOR (t:TimePoint) ON (t.owner_id, t.value);
                """,
            ),
            (
                "object_owner_name_unique",
                """
                CREATE CONSTRAINT object_owner_name_unique IF NOT EXISTS
                FOR (o:Object) REQUIRE (o.owner_id, o.name) IS UNIQUE;
                """,
                """
                CREATE INDEX object_owner_name_idx IF NOT EXISTS
                FOR (o:Object) ON (o.owner_id, o.name);
                """,
            ),
            (
                "scene_embedding_index",
                scene_embedding_index_query,
                None,
            ),
            (
                "scene_keyword_index",
                """
                CREATE FULLTEXT INDEX scene_keyword_index IF NOT EXISTS
                FOR (s:SceneMemory) ON EACH [s.scene_gist, s.anchors_text, s.place_name, s.participants_text, s.life_stage];
                """,
                None,
            ),
            (
                "canonrule_owner_idx",
                """
                CREATE INDEX canonrule_owner_idx IF NOT EXISTS
                FOR (r:CanonRule) ON (r.owner_id);
                """,
                None,
            ),
        ]
        with self.driver.session(database=settings.neo4j_db) as session:
            try:
                record = session.run(
                    """
                    SHOW INDEXES YIELD name, type, options
                    WHERE name = $name
                    RETURN options
                    """,
                    {"name": "scene_embedding_index"},
                ).single()
                if record:
                    options = record.get("options") or {}
                    index_config = options.get("indexConfig") or {}
                    dims = index_config.get("vector.dimensions")
                    dims_val = None
                    try:
                        if dims is not None:
                            dims_val = int(dims)
                    except (TypeError, ValueError):
                        dims_val = None
                    if dims_val and dims_val != expected_dims:
                        logger.warning(
                            "Neo4j vector index dims %s != expected %s; dropping index",
                            dims_val,
                            expected_dims,
                        )
                        session.run("DROP INDEX scene_embedding_index IF EXISTS")
            except Exception as exc:
                logger.warning("Neo4j index check failed (scene_embedding_index): %s", exc)

            person_schema_exists = False
            try:
                record = session.run(
                    """
                    SHOW INDEXES YIELD labelsOrTypes, properties
                    WHERE $label IN labelsOrTypes
                      AND size(properties) = $prop_len
                      AND all(p IN $props WHERE p IN properties)
                    RETURN count(*) AS count
                    """,
                    {"label": "Person", "props": ["owner_id", "name"], "prop_len": 2},
                ).single()
                if record and (record.get("count") or 0) > 0:
                    person_schema_exists = True
                if not person_schema_exists:
                    record = session.run(
                        """
                        SHOW CONSTRAINTS YIELD labelsOrTypes, properties
                        WHERE $label IN labelsOrTypes
                          AND size(properties) = $prop_len
                          AND all(p IN $props WHERE p IN properties)
                        RETURN count(*) AS count
                        """,
                        {"label": "Person", "props": ["owner_id", "name"], "prop_len": 2},
                    ).single()
                    if record and (record.get("count") or 0) > 0:
                        person_schema_exists = True
            except Exception as exc:
                logger.debug("Neo4j Person(owner_id,name) schema check failed: %s", exc)

            for name, q, fallback in queries:
                if name == "person_owner_name_unique" and person_schema_exists:
                    logger.info("Neo4j schema exists: person_owner_name_unique, skipping")
                    continue
                try:
                    session.run(q)
                except Exception as exc:
                    logger.warning("Neo4j schema init failed (%s): %s", name, exc)
                    if fallback:
                        try:
                            session.run(fallback)
                        except Exception as fallback_exc:
                            logger.warning("Neo4j schema fallback failed (%s): %s", name, fallback_exc)

    def _load_keyword_index(self) -> None:
        if self._keyword_index_loaded:
            return
        if not self.keyword_index_path.exists():
            self.keyword_index_cache = {}
            self._keyword_index_loaded = True
            return
        try:
            data = json.loads(self.keyword_index_path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                self.keyword_index_cache = data
            else:
                self.keyword_index_cache = {}
        except Exception:
            self.keyword_index_cache = {}
        self._keyword_index_loaded = True

    def _load_scene_cache(self) -> None:
        if self.scene_cache:
            return
        if not self.scene_store_path.exists():
            return
        records = _read_records(self.scene_store_path)
        for record in records:
            if not isinstance(record, dict):
                continue
            owner_id = record.get("owner_id") or record.get("character_id")
            if not owner_id:
                continue
            scenes = record.get("scenes")
            if isinstance(scenes, list):
                for scene in scenes:
                    if not isinstance(scene, dict):
                        continue
                    scene_id = scene.get("scene_id")
                    if not scene_id:
                        continue
                    self.scene_cache.setdefault(owner_id, []).append(scene)
            else:
                scene_id = record.get("scene_id")
                if scene_id:
                    self.scene_cache.setdefault(owner_id, []).append(record)

    def _persist_keyword_index(self) -> None:
        self.keyword_index_path.parent.mkdir(parents=True, exist_ok=True)
        with self.keyword_index_path.open("w", encoding="utf-8", newline="\n") as fh:
            fh.write(json.dumps(self.keyword_index_cache, ensure_ascii=False, indent=2))

    def _update_keyword_index(self, character_id: str, scenes: List[Dict[str, Any]]) -> None:
        self._load_keyword_index()
        index = self.keyword_index_cache.setdefault(character_id, {})
        for scene in scenes:
            scene_id = scene.get("scene_id")
            if not scene_id:
                continue
            for kw in self._scene_keywords(scene):
                if not kw:
                    continue
                index.setdefault(kw, [])
                if scene_id not in index[kw]:
                    index[kw].append(scene_id)
        self._persist_keyword_index()

    def _scene_keywords(self, scene: Dict[str, Any]) -> List[str]:
        tokens: List[str] = []
        anchors = scene.get("anchors") or []
        participants = scene.get("participants") or []
        place = scene.get("place") or {}
        life_stage = scene.get("life_stage") or ""
        emotion = scene.get("emotion") or []

        for val in anchors:
            if isinstance(val, str):
                anchor = normalize_anchor(normalize_text(val))
                if anchor:
                    tokens.append(anchor)
        for val in participants + emotion:
            if isinstance(val, str):
                cleaned = normalize_text(val)
                if cleaned:
                    tokens.append(cleaned.lower())
        if isinstance(place, dict):
            place_name = place.get("name") or ""
            if place_name:
                cleaned = normalize_text(str(place_name))
                if cleaned:
                    tokens.append(cleaned.lower())
        if life_stage:
            cleaned = normalize_text(str(life_stage))
            if cleaned:
                tokens.append(cleaned.lower())
        seen: set[str] = set()
        output: List[str] = []
        for token in tokens:
            if not token:
                continue
            if token in seen:
                continue
            seen.add(token)
            output.append(token)
        return output

    def write_persona(self, character_id: str, persona: Dict[str, Any]) -> None:
        self.persona_cache[character_id] = persona
        _append_record(self.persona_store_path, {"character_id": character_id, "persona": persona})
        # Persona stage only persists to JSONL; Neo4j Character nodes are created by later stages if needed.
        return

    def write_ruleset(self, character_id: str, ruleset: Dict[str, Any]) -> None:
        if not ruleset or not isinstance(ruleset, dict):
            return
        if not self.driver:
            return
        cypher = """
        MERGE (c:Character {id: $cid})
        ON CREATE SET c.owner_id = $cid
        MERGE (r:CanonRule {id: $rid})
        SET r += $ruleset, r.owner_id = $cid
        MERGE (c)-[:HAS_RULESET]->(r)
        """
        with self.driver.session(database=settings.neo4j_db) as session:
            session.run(cypher, {"cid": character_id, "rid": ruleset.get("id"), "ruleset": ruleset})

    def write_related_characters(
        self,
        character_id: str,
        related: List[Dict[str, Any]],
        *,
        rc_id: str,
        created_at: str,
    ) -> None:
        cleaned_related: List[Dict[str, Any]] = []
        for item in related:
            if not isinstance(item, dict):
                continue
            name = normalize_text(item.get("name") or "")
            relation = normalize_text(item.get("relation") or "")
            role = normalize_text(item.get("role") or "")
            if not name:
                continue
            cleaned_related.append({"name": name, "relation": relation, "role": role})
        record = {
            "owner_id": character_id,
            "rc_id": rc_id,
            "created_at": created_at,
            "related_characters": cleaned_related,
        }
        _append_record(self.related_store_path, record)

    def read_related_characters(self, character_id: str) -> List[Dict[str, Any]]:
        record = self.read_related_characters_record(character_id)
        if not record:
            return []
        return record.get("related_characters") or []

    def _read_latest_record(self, path: Path, character_id: str) -> Optional[Dict[str, Any]]:
        records = _read_records(path)
        for record in reversed(records):
            owner_id = record.get("owner_id") or record.get("character_id")
            if owner_id == character_id:
                return record
        return None

    def read_related_characters_record(self, character_id: str) -> Optional[Dict[str, Any]]:
        return self._read_latest_record(self.related_store_path, character_id)

    def read_worldrule_record(self, character_id: str) -> Optional[Dict[str, Any]]:
        return self._read_latest_record(self.worldrule_store_path, character_id)

    def read_inspiration_record(self, character_id: str) -> Optional[Dict[str, Any]]:
        return self._read_latest_record(self.inspiration_store_path, character_id)

    def read_scene_pack_record(self, character_id: str) -> Optional[Dict[str, Any]]:
        return self._read_latest_record(self.scene_store_path, character_id)

    def read_detail_pack_record(self, character_id: str, scene_id: str) -> Optional[Dict[str, Any]]:
        records = _read_records(self.detail_store_path)
        for record in reversed(records):
            owner_id = record.get("owner_id") or record.get("character_id")
            if owner_id == character_id and record.get("scene_id") == scene_id:
                return record
        return None

    def write_worldrule(self, character_id: str, record: Dict[str, Any]) -> None:
        _append_record(self.worldrule_store_path, record)

    def write_inspiration(self, character_id: str, record: Dict[str, Any]) -> None:
        _append_record(self.inspiration_store_path, record)

    def write_scene_pack(self, character_id: str, scene_pack: Dict[str, Any], *, append_record: bool = True) -> None:
        scenes = scene_pack.get("scenes") if isinstance(scene_pack, dict) else []
        if not isinstance(scenes, list):
            scenes = []
        if append_record:
            _append_record(self.scene_store_path, scene_pack)
        self._write_scene_nodes(character_id, scenes)

    def _write_scene_nodes(self, character_id: str, scenes: List[Dict[str, Any]]) -> None:
        normalized_scenes: List[Dict[str, Any]] = []
        cleaned: List[Dict[str, Any]] = []
        for scene in scenes:
            place = scene.get("place") or scene.get("where") or {}
            self_state = scene.get("self_state") or {}
            salience = scene.get("salience") or {}
            anchors_raw = scene.get("anchors") or []
            participants_raw = scene.get("participants") or scene.get("who") or []

            anchors = [
                normalize_anchor(normalize_text(val))
                for val in anchors_raw
                if isinstance(val, str) and normalize_anchor(normalize_text(val))
            ]
            participants = [
                normalize_text(val)
                for val in participants_raw
                if isinstance(val, str) and normalize_text(val)
            ]
            place_name = normalize_text(place.get("name") or place.get("place_name") or "")
            place_type = normalize_text(place.get("type") or place.get("place_type") or "")
            life_stage = normalize_text(scene.get("life_stage") or "")
            time_range = normalize_text(scene.get("time_range") or "")
            scene_gist = normalize_text(scene.get("scene_gist") or scene.get("summary_7whr") or "")
            emotion = [
                normalize_text(val)
                for val in (scene.get("emotion") or [])
                if isinstance(val, str) and normalize_text(val)
            ]

            normalized_scene = dict(scene)
            normalized_scene["anchors"] = anchors
            normalized_scene["participants"] = participants
            if "who" in normalized_scene:
                normalized_scene["who"] = participants
            if "where" in normalized_scene and isinstance(normalized_scene.get("where"), dict):
                normalized_scene["where"] = {"name": place_name, "type": place_type}
            if "place" in normalized_scene and isinstance(normalized_scene.get("place"), dict):
                normalized_scene["place"] = {"name": place_name, "type": place_type}
            if life_stage:
                normalized_scene["life_stage"] = life_stage
            if time_range:
                normalized_scene["time_range"] = time_range
            if scene_gist:
                normalized_scene["scene_gist"] = scene_gist
            if emotion:
                normalized_scene["emotion"] = emotion

            normalized_scenes.append(normalized_scene)
            cleaned.append(
                {
                    "scene_id": normalized_scene.get("scene_id"),
                    "character_id": character_id,
                    "life_stage": normalized_scene.get("life_stage"),
                    "time_range": normalized_scene.get("time_range"),
                    "scene_gist": normalized_scene.get("scene_gist") or normalized_scene.get("summary_7whr"),
                    "self_state_physical": self_state.get("physical"),
                    "self_state_mental": self_state.get("mental"),
                    "emotion": normalized_scene.get("emotion") or [],
                    "salience_importance": salience.get("importance"),
                    "salience_emotional_intensity": salience.get("emotional_intensity"),
                    "salience_recall_probability": salience.get("recall_probability"),
                    "anchors": anchors,
                    "anchors_text": " ".join(anchors),
                    "place_name": place_name,
                    "place_type": place_type,
                    "participants": participants,
                    "participants_text": " ".join(participants),
                    "event_root_ref": normalized_scene.get("event_root_ref") or normalized_scene.get("scene_id"),
                    "created_at": normalized_scene.get("created_at"),
                    "embedding": normalized_scene.get("embedding"),
                    "owner_id": character_id,
                }
            )

        self._update_keyword_index(character_id, normalized_scenes)

        self.scene_cache.setdefault(character_id, []).extend(normalized_scenes)
        if not self.driver:
            return

        cypher = """
        UNWIND $scenes AS s
        MERGE (c:Character {id: $cid})
        ON CREATE SET c.owner_id = $cid
        SET c.owner_id = $cid
        MERGE (sm:SceneMemory {scene_id: s.scene_id})
        SET sm += s, sm.owner_id = $cid
        MERGE (c)-[:HAS_SCENE]->(sm)
        """
        with self.driver.session(database=settings.neo4j_db) as session:
            session.run(cypher, {"cid": character_id, "scenes": cleaned})

    def get_scene_by_id(self, character_id: str, scene_id: str) -> Optional[Dict[str, Any]]:
        if not scene_id:
            return None
        if not self.driver:
            for scene in self.scene_cache.get(character_id, []):
                if scene.get("scene_id") == scene_id:
                    return scene
            return None
        cypher = """
        MATCH (s:SceneMemory {scene_id: $scene_id, owner_id: $cid})
        RETURN s {.*} AS scene
        """
        with self.driver.session(database=settings.neo4j_db) as session:
            result = session.run(cypher, {"scene_id": scene_id, "cid": character_id})
            record = result.single()
            return record["scene"] if record else None

    def query_scene_vectors(self, character_id: str, query_emb: List[float], limit: int = 8) -> List[Dict[str, Any]]:
        expected_dims = settings.embed_dimensions if settings.embed_dimensions > 0 else 0
        if expected_dims and len(query_emb) != expected_dims:
            logger.error(
                "scene vector query skipped: query dims %s != expected %s",
                len(query_emb),
                expected_dims,
            )
            return []
        if not self.driver:
            return self._query_scene_cache(character_id, query_emb, limit)

        cypher = """
        CALL db.index.vector.queryNodes('scene_embedding_index', $limit, $query_emb)
        YIELD node, score
        WHERE node.owner_id = $cid
        RETURN node {.*, score: score} AS scene
        ORDER BY scene.score DESC
        LIMIT $limit
        """
        try:
            with self.driver.session(database=settings.neo4j_db) as session:
                result = session.run(cypher, {"cid": character_id, "query_emb": query_emb, "limit": limit})
                return [record["scene"] for record in result]
        except Exception:
            logger.exception("scene vector query failed; attempting fallback scan")
            try:
                fallback = self._query_scene_vectors_by_scan(character_id, query_emb, limit)
                if fallback:
                    return fallback
            except Exception:
                logger.exception("scene vector fallback scan failed; falling back to cache")
            return self._query_scene_cache(character_id, query_emb, limit)

    def query_scene_keywords(
        self,
        character_id: str,
        query_text: str,
        limit: int = 8,
        *,
        cache_only: bool = False,
    ) -> List[Dict[str, Any]]:
        tokens = _tokenize_keywords(query_text)
        if not tokens:
            return []
        if self.driver and not cache_only:
            query = " OR ".join(tokens)
            cypher = """
            CALL db.index.fulltext.queryNodes('scene_keyword_index', $query)
            YIELD node, score
            WHERE node.owner_id = $cid
            RETURN node {.*, score: score} AS scene
            ORDER BY scene.score DESC
            LIMIT $limit
            """
            try:
                with self.driver.session(database=settings.neo4j_db) as session:
                    result = session.run(cypher, {"cid": character_id, "query": query, "limit": limit})
                    return [record["scene"] for record in result]
            except Exception:
                logger.warning("scene keyword query failed; falling back to local index")

        self._load_keyword_index()
        index = self.keyword_index_cache.get(character_id, {})
        scores: Dict[str, float] = {}
        for token in tokens:
            for scene_id in index.get(token, []):
                scores[scene_id] = scores.get(scene_id, 0.0) + 1.0
        results: List[Dict[str, Any]] = []
        scenes_by_id = {s.get("scene_id"): s for s in self.scene_cache.get(character_id, [])}
        if scores:
            for scene_id, score in scores.items():
                scene = scenes_by_id.get(scene_id)
                if not scene:
                    continue
                scene_copy = dict(scene)
                scene_copy["score"] = score
                results.append(scene_copy)
            return sorted(results, key=lambda x: x.get("score", 0), reverse=True)[:limit]

        for scene in scenes_by_id.values():
            scene_gist = str(scene.get("scene_gist") or scene.get("summary_7whr") or "")
            anchors = scene.get("anchors") or []
            participants = scene.get("participants") or scene.get("who") or []
            place = scene.get("place") if isinstance(scene.get("place"), dict) else {}
            place_name = str(place.get("name") or scene.get("place_name") or "")
            field_text = " ".join(
                [
                    scene_gist,
                    place_name,
                    " ".join([str(x) for x in anchors if x]),
                    " ".join([str(x) for x in participants if x]),
                ]
            ).lower()
            score = 0
            for token in tokens:
                if token and token in field_text:
                    score += 1
            if score <= 0:
                continue
            scene_copy = dict(scene)
            scene_copy["score"] = float(score)
            results.append(scene_copy)
        return sorted(results, key=lambda x: x.get("score", 0), reverse=True)[:limit]

    def write_detail_graph(
        self,
        character_id: str,
        scene_id: str,
        graph: Dict[str, Any],
        *,
        canonical_name: Optional[str] = None,
    ) -> None:
        if not scene_id:
            return
        nodes = graph.get("nodes") or []
        edges = graph.get("edges") or []

        canonical_name = normalize_text(canonical_name or "")
        prepared_nodes: List[Dict[str, Any]] = []
        prepared_edges: List[Dict[str, Any]] = []
        detail_nodes: List[Dict[str, Any]] = []
        person_nodes: Dict[str, Dict[str, Any]] = {}
        place_nodes: Dict[str, Dict[str, Any]] = {}
        node_specs: Dict[str, Dict[str, Any]] = {}
        id_map: Dict[str, str] = {}
        event_ids: List[str] = []

        for idx, node in enumerate(nodes):
            if not isinstance(node, dict):
                continue
            node_copy = dict(node)
            node_type = node_copy.get("type") or node_copy.get("node_type")
            if not node_type:
                continue
            raw_id = node_copy.get("id") or f"{scene_id}_node_{idx:02d}"
            node_copy["type"] = node_type

            if node_type in {"Person", "Place"}:
                name = str(node_copy.get("name") or node_copy.get("label") or "").strip()
                if not name:
                    continue
                if node_type == "Person":
                    is_protagonist = bool(node_copy.get("is_protagonist")) or (canonical_name and name == canonical_name)
                    if is_protagonist:
                        node_specs[raw_id] = {"kind": "character"}
                        continue
                    key = re.sub(r"\s+", "", name)
                    if key not in person_nodes:
                        person_nodes[key] = {
                            "id": raw_id,
                            "type": "Person",
                            "node_type": "Person",
                            "name": name,
                            "owner_id": character_id,
                            "search_text": name,
                        }
                    node_specs[raw_id] = {"kind": "person", "name": person_nodes[key]["name"]}
                else:
                    place_type = str(node_copy.get("place_type") or node_copy.get("category") or "地点").strip() or "地点"
                    clean_name = re.sub(r"\s+", "", name)
                    key = f"{clean_name}|{place_type}"
                    if key not in place_nodes:
                        place_nodes[key] = {
                            "id": raw_id,
                            "type": "Place",
                            "node_type": "Place",
                            "name": name,
                            "place_type": place_type,
                            "owner_id": character_id,
                            "search_text": f"{name} {place_type}".strip(),
                        }
                    node_specs[raw_id] = {
                        "kind": "place",
                        "name": place_nodes[key]["name"],
                        "place_type": place_nodes[key]["place_type"],
                    }
                continue

            node_id = raw_id
            if not str(node_id).startswith(scene_id):
                node_id = f"{scene_id}_{node_id}"
            id_map[raw_id] = node_id
            node_copy["id"] = node_id
            node_copy["scene_id"] = scene_id
            node_copy["owner_id"] = character_id
            node_copy["node_type"] = node_type
            node_copy["search_text"] = self._build_search_text(node_copy)
            prepared_nodes.append(node_copy)
            detail_nodes.append(node_copy)
            node_specs[node_id] = {"kind": "detail"}
            if node_type == "Event":
                event_ids.append(node_id)

        prepared_nodes.extend(person_nodes.values())
        prepared_nodes.extend(place_nodes.values())

        for edge in edges:
            if not isinstance(edge, dict):
                continue
            src = edge.get("from")
            dst = edge.get("to")
            rel = edge.get("type") or edge.get("rel")
            if not src or not dst or not rel:
                continue
            src = id_map.get(src, src)
            dst = id_map.get(dst, dst)
            src_spec = node_specs.get(src)
            dst_spec = node_specs.get(dst)
            if not src_spec or not dst_spec:
                continue
            rel_type = _safe_label(rel, "RELATES_TO")
            edge_props = {k: v for k, v in edge.items() if k not in {"from", "to", "type", "rel"}}
            edge_props["owner_id"] = character_id
            edge_props["scene_id"] = scene_id
            prepared_edges.append({"from": src, "to": dst, "type": rel_type, **edge_props})

        graph_payload = {
            "owner_id": character_id,
            "scene_id": scene_id,
            "nodes": prepared_nodes,
            "edges": prepared_edges,
        }
        self.detail_cache[scene_id] = graph_payload
        _append_record(self.detail_store_path, {"character_id": character_id, "scene_id": scene_id, "graph": graph_payload})

        if not self.driver:
            return

        with self.driver.session(database=settings.neo4j_db) as session:
            session.run(
                """
                MATCH (n {scene_id: $scene_id, owner_id: $cid})
                WHERE NOT n:SceneMemory
                DETACH DELETE n
                """,
                {"scene_id": scene_id, "cid": character_id},
            )

            if detail_nodes:
                by_type: Dict[str, List[Dict[str, Any]]] = {}
                for node in detail_nodes:
                    node_type = _safe_label(node.get("node_type") or "Detail", "Detail")
                    node["node_type"] = node_type
                    by_type.setdefault(node_type, []).append(node)
                for node_type, nodes_list in by_type.items():
                    cypher = f"""
                    UNWIND $nodes AS n
                    MERGE (node:{node_type} {{id: n.id, owner_id: $cid, scene_id: $scene_id}})
                    SET node += n
                    """
                    session.run(cypher, {"nodes": nodes_list, "cid": character_id, "scene_id": scene_id})

            if person_nodes:
                session.run(
                    """
                    UNWIND $nodes AS n
                    MERGE (p:Person {owner_id: $cid, name: n.name})
                    SET p += n
                    """,
                    {"nodes": list(person_nodes.values()), "cid": character_id},
                )

            if place_nodes:
                session.run(
                    """
                    UNWIND $nodes AS n
                    MERGE (pl:Place {owner_id: $cid, name: n.name, place_type: n.place_type})
                    SET pl += n
                    """,
                    {"nodes": list(place_nodes.values()), "cid": character_id},
                )

            edges_by_combo: Dict[tuple[str, str, str], List[Dict[str, Any]]] = {}
            for edge in prepared_edges:
                rel_type = edge.get("type") or "RELATES_TO"
                src = edge.get("from")
                dst = edge.get("to")
                src_spec = node_specs.get(src)
                dst_spec = node_specs.get(dst)
                if not src_spec or not dst_spec:
                    continue
                combo = (src_spec["kind"], dst_spec["kind"], rel_type)
                record: Dict[str, Any] = {"props": {k: v for k, v in edge.items() if k not in {"from", "to", "type"}}}
                if src_spec["kind"] == "detail":
                    record["from_id"] = src
                elif src_spec["kind"] == "person":
                    record["from_name"] = src_spec["name"]
                elif src_spec["kind"] == "place":
                    record["from_name"] = src_spec["name"]
                    record["from_place_type"] = src_spec["place_type"]
                else:
                    record["from_character"] = True
                if dst_spec["kind"] == "detail":
                    record["to_id"] = dst
                elif dst_spec["kind"] == "person":
                    record["to_name"] = dst_spec["name"]
                elif dst_spec["kind"] == "place":
                    record["to_name"] = dst_spec["name"]
                    record["to_place_type"] = dst_spec["place_type"]
                else:
                    record["to_character"] = True
                edges_by_combo.setdefault(combo, []).append(record)

            for (from_kind, to_kind, rel_type), edges_list in edges_by_combo.items():
                if not edges_list:
                    continue
                match_a = ""
                match_b = ""
                if from_kind == "detail":
                    match_a = "MATCH (a {id: e.from_id, owner_id: $cid, scene_id: $scene_id})"
                elif from_kind == "person":
                    match_a = "MATCH (a:Person {owner_id: $cid, name: e.from_name})"
                elif from_kind == "place":
                    match_a = "MATCH (a:Place {owner_id: $cid, name: e.from_name, place_type: e.from_place_type})"
                else:
                    match_a = "MERGE (a:Character {id: $cid}) SET a.owner_id = $cid"

                if to_kind == "detail":
                    match_b = "MATCH (b {id: e.to_id, owner_id: $cid, scene_id: $scene_id})"
                elif to_kind == "person":
                    match_b = "MATCH (b:Person {owner_id: $cid, name: e.to_name})"
                elif to_kind == "place":
                    match_b = "MATCH (b:Place {owner_id: $cid, name: e.to_name, place_type: e.to_place_type})"
                else:
                    match_b = "MERGE (b:Character {id: $cid}) SET b.owner_id = $cid"

                cypher = f"""
                UNWIND $edges AS e
                {match_a}
                {match_b}
                MERGE (a)-[r:{_safe_label(rel_type, "RELATES_TO")}]->(b)
                SET r += e.props
                """
                session.run(cypher, {"edges": edges_list, "cid": character_id, "scene_id": scene_id})

            if event_ids:
                session.run(
                    """
                    MATCH (s:SceneMemory {scene_id: $scene_id, owner_id: $cid})
                    WITH s
                    UNWIND $event_ids AS eid
                    MATCH (e:Event {id: eid, owner_id: $cid, scene_id: $scene_id})
                    MERGE (e)-[:BELONGS_TO_SCENE]->(s)
                    """,
                    {"event_ids": event_ids, "scene_id": scene_id, "cid": character_id},
                )

    def write_related_character_edges(
        self,
        character_id: str,
        persona_name: str,
        related: List[Dict[str, Any]],
    ) -> None:
        persona_name = normalize_text(persona_name)
        if not self.driver or not persona_name:
            return
        with self.driver.session(database=settings.neo4j_db) as session:
            session.run(
                """
                MERGE (c:Character {owner_id: $cid, name: $name})
                SET c.id = $cid, c.node_type = 'Person', c.type = 'Person'
                """,
                {"cid": character_id, "name": persona_name},
            )
            if related:
                normalized_related: List[Dict[str, Any]] = []
                for item in related:
                    if not isinstance(item, dict):
                        continue
                    name = normalize_text(item.get("name") or "")
                    relation = normalize_text(item.get("relation") or "")
                    role = normalize_text(item.get("role") or "")
                    if not name:
                        continue
                    normalized_related.append({"name": name, "relation": relation, "role": role})
                session.run(
                    """
                    MATCH (c:Character {owner_id: $cid, name: $name})
                    UNWIND $related AS r
                    MERGE (rc:Character {owner_id: $cid, name: r.name})
                    SET rc.node_type = 'Person', rc.type = 'Person', rc.relation = r.relation, rc.role = r.role
                    MERGE (c)-[rel:REL]->(rc)
                    SET rel.relation = r.relation, rel.role = r.role
                    """,
                    {"cid": character_id, "name": persona_name, "related": normalized_related},
                )

    def write_detail_pack(
        self,
        character_id: str,
        detail_pack: Dict[str, Any],
        *,
        canonical_name: Optional[str] = None,
        append_record: bool = True,
    ) -> None:
        scene_id = detail_pack.get("scene_id")
        if not scene_id:
            return
        detail_pack = dict(detail_pack)
        events_raw = detail_pack.get("events") if isinstance(detail_pack.get("events"), list) else []
        normalized_events: List[Dict[str, Any]] = []
        event_id_map: Dict[str, str] = {}
        order_id_map: Dict[int, str] = {}
        used_orders: set[int] = set()

        def _coerce_order(value: Any, fallback: int, used: Optional[set[int]]) -> int:
            try:
                order_val = int(value)
            except (TypeError, ValueError):
                order_val = fallback
            if order_val <= 0:
                order_val = fallback
            if used is not None:
                if order_val in used:
                    order_val = fallback
                    while order_val in used:
                        order_val += 1
                used.add(order_val)
            return order_val

        def _event_id(order_val: int) -> str:
            return f"{scene_id}#E{order_val:03d}"

        def _utt_id(event_order: int, utt_order: int) -> str:
            return f"{scene_id}#E{event_order:03d}#U{utt_order:03d}"

        def _map_event_ref(value: Any) -> Optional[str]:
            if value is None:
                return None
            if isinstance(value, str):
                val = value.strip()
                if val.startswith(f"{scene_id}#E"):
                    return val
                if val in event_id_map:
                    return event_id_map[val]
                match = re.search(r"\d+", val)
                if match:
                    return order_id_map.get(int(match.group()))
                return None
            if isinstance(value, (int, float)):
                return order_id_map.get(int(value))
            val_str = str(value).strip()
            if val_str in event_id_map:
                return event_id_map[val_str]
            match = re.search(r"\d+", val_str)
            if match:
                return order_id_map.get(int(match.group()))
            return None

        sorted_events = sorted(
            [e for e in events_raw if isinstance(e, dict)], key=lambda e: int(e.get("order") or 0)
        )
        for idx, event in enumerate(sorted_events, start=1):
            order_val = _coerce_order(event.get("order"), idx, used_orders)
            event_id = _event_id(order_val)
            old_id = str(event.get("event_id") or event.get("id") or f"E{order_val:03d}").strip()
            if old_id:
                event_id_map[old_id] = event_id
            event_id_map[f"E{order_val:03d}"] = event_id
            order_id_map[order_val] = event_id
            new_event = dict(event)
            new_event["event_id"] = event_id
            new_event["order"] = order_val
            new_event["event_text"] = normalize_text(new_event.get("event_text") or "")
            new_event["time_point"] = normalize_text(new_event.get("time_point") or "")
            place = new_event.get("place") if isinstance(new_event.get("place"), dict) else {}
            place_name = normalize_text(place.get("name") or "")
            place_type = normalize_text(place.get("type") or "")
            new_event["place"] = {"name": place_name, "type": place_type} if (place_name or place_type) else {}
            participants_raw = new_event.get("participants") if isinstance(new_event.get("participants"), list) else []
            new_event["participants"] = [
                normalize_text(val)
                for val in participants_raw
                if isinstance(val, str) and normalize_text(val)
            ]
            objects_raw = new_event.get("objects") if isinstance(new_event.get("objects"), list) else []
            new_event["objects"] = [
                normalize_text(val)
                for val in objects_raw
                if isinstance(val, str) and normalize_text(val)
            ]
            dialogue_raw = new_event.get("dialogue") if isinstance(new_event.get("dialogue"), list) else []
            normalized_dialogue: List[Dict[str, Any]] = []
            used_utt_orders: set[int] = set()
            for didx, d in enumerate([x for x in dialogue_raw if isinstance(x, dict)], start=1):
                utt_order = _coerce_order(d.get("order"), didx, used_utt_orders)
                utt_id = _utt_id(order_val, utt_order)
                new_d = dict(d)
                new_d["utt_id"] = utt_id
                new_d["order"] = utt_order
                new_d["speaker"] = normalize_text(new_d.get("speaker") or "")
                new_d["text"] = normalize_text(new_d.get("text") or "")
                normalized_dialogue.append(new_d)
            new_event["dialogue"] = normalized_dialogue
            normalized_events.append(new_event)

        events = normalized_events
        detail_pack["events"] = normalized_events

        causal_raw = detail_pack.get("causal_edges") if isinstance(detail_pack.get("causal_edges"), list) else []
        causal_edges: List[Dict[str, Any]] = []
        for edge in causal_raw:
            if not isinstance(edge, dict):
                continue
            src = _map_event_ref(edge.get("from"))
            dst = _map_event_ref(edge.get("to"))
            if not src or not dst:
                continue
            causal_edges.append({"from": src, "to": dst, "type": edge.get("type") or "CAUSES"})
        detail_pack["causal_edges"] = causal_edges

        if append_record:
            _append_record(self.detail_store_path, detail_pack)
        nodes: List[Dict[str, Any]] = []
        edges: List[Dict[str, Any]] = []
        event_rows: List[Dict[str, Any]] = []
        event_ids: List[str] = []
        event_pairs: List[Dict[str, Any]] = []
        participant_links: List[Dict[str, Any]] = []
        place_links: List[Dict[str, Any]] = []
        time_links: List[Dict[str, Any]] = []
        object_links: List[Dict[str, Any]] = []
        utterances: List[Dict[str, Any]] = []
        utterance_pairs: List[Dict[str, Any]] = []
        utterance_speakers: List[Dict[str, Any]] = []

        person_nodes: Dict[str, Dict[str, Any]] = {}
        place_nodes: Dict[str, Dict[str, Any]] = {}
        time_nodes: Dict[str, Dict[str, Any]] = {}
        object_nodes: Dict[str, Dict[str, Any]] = {}

        sorted_events = sorted(events, key=lambda e: int(e.get("order") or 0))
        for idx, event in enumerate(sorted_events, start=1):
            if not isinstance(event, dict):
                continue
            event_id = str(event.get("event_id") or _event_id(int(event.get("order") or idx)))
            order = int(event.get("order") or idx)
            phase = normalize_text(event.get("phase") or "")
            event_text = normalize_text(event.get("event_text") or "")
            time_point = normalize_text(event.get("time_point") or "")
            place = event.get("place") if isinstance(event.get("place"), dict) else {}
            place_name = normalize_text(place.get("name") or "")
            place_type = normalize_text(place.get("type") or "")
            participants = [
                normalize_text(val)
                for val in (event.get("participants") or [])
                if isinstance(val, str) and normalize_text(val)
            ]
            objects = [
                normalize_text(val)
                for val in (event.get("objects") or [])
                if isinstance(val, str) and normalize_text(val)
            ]

            event_rows.append(
                {
                    "event_id": event_id,
                    "owner_id": character_id,
                    "scene_id": scene_id,
                    "node_type": "Event",
                    "type": "Event",
                    "order": order,
                    "phase": phase,
                    "event_text": event_text,
                    "summary": event_text,
                    "time_point": time_point,
                    "place_name": place_name,
                    "place_type": place_type,
                    "participants": participants,
                    "objects": objects,
                    "search_text": " ".join([event_text, time_point, place_name, " ".join(participants), " ".join(objects)]).strip(),
                }
            )
            event_ids.append(event_id)
            nodes.append(
                {
                    "id": event_id,
                    "type": "Event",
                    "node_type": "Event",
                    "summary": event_text,
                    "order": order,
                    "phase": phase,
                    "scene_id": scene_id,
                    "owner_id": character_id,
                    "search_text": event_text,
                }
            )
            if time_point:
                if time_point not in time_nodes:
                    time_nodes[time_point] = {
                        "id": f"{scene_id}_time_{len(time_nodes) + 1:02d}",
                        "type": "TimePoint",
                        "node_type": "TimePoint",
                        "value": time_point,
                        "owner_id": character_id,
                        "search_text": time_point,
                    }
                time_links.append({"event_id": event_id, "value": time_point})
                edges.append({"from": event_id, "to": time_nodes[time_point]["id"], "type": "AT_TIME"})
            if place_name:
                if place_name not in place_nodes:
                    place_nodes[place_name] = {
                        "id": f"{scene_id}_place_{len(place_nodes) + 1:02d}",
                        "type": "Place",
                        "node_type": "Place",
                        "name": place_name,
                        "place_type": place_type,
                        "owner_id": character_id,
                        "search_text": f"{place_name} {place_type}".strip(),
                    }
                place_links.append({"event_id": event_id, "name": place_name, "place_type": place_type})
                edges.append({"from": event_id, "to": place_nodes[place_name]["id"], "type": "AT_PLACE"})
            for name in participants:
                if not name:
                    continue
                if name not in person_nodes:
                    person_nodes[name] = {
                        "id": f"{scene_id}_person_{len(person_nodes) + 1:02d}",
                        "type": "Person",
                        "node_type": "Person",
                        "name": name,
                        "owner_id": character_id,
                        "search_text": name,
                    }
                participant_links.append({"event_id": event_id, "name": name})
                edges.append({"from": event_id, "to": person_nodes[name]["id"], "type": "INVOLVES"})
            for obj in objects:
                if not obj:
                    continue
                if obj not in object_nodes:
                    object_nodes[obj] = {
                        "id": f"{scene_id}_object_{len(object_nodes) + 1:02d}",
                        "type": "Object",
                        "node_type": "Object",
                        "name": obj,
                        "owner_id": character_id,
                        "search_text": obj,
                    }
                object_links.append({"event_id": event_id, "name": obj})
                edges.append({"from": event_id, "to": object_nodes[obj]["id"], "type": "USES_OBJECT"})

            dialogue_raw = event.get("dialogue") if isinstance(event.get("dialogue"), list) else []
            dialogue_sorted = sorted(
                [d for d in dialogue_raw if isinstance(d, dict)], key=lambda d: int(d.get("order") or 0)
            )
            prev_utt = None
            for didx, d in enumerate(dialogue_sorted, start=1):
                utt_id = str(d.get("utt_id") or f"{event_id}_utt_{didx:02d}")
                order_val = int(d.get("order") or didx)
                speaker = normalize_text(d.get("speaker") or "")
                text = normalize_text(d.get("text") or "")
                utterances.append(
                    {
                        "utt_id": utt_id,
                        "event_id": event_id,
                        "order": order_val,
                        "speaker": speaker,
                        "text": text,
                        "search_text": " ".join([speaker, text]).strip(),
                    }
                )
                nodes.append(
                    {
                        "id": utt_id,
                        "type": "Utterance",
                        "node_type": "Utterance",
                        "speaker": speaker,
                        "text": text,
                        "order": order_val,
                        "scene_id": scene_id,
                        "owner_id": character_id,
                        "search_text": " ".join([speaker, text]).strip(),
                    }
                )
                edges.append({"from": event_id, "to": utt_id, "type": "HAS_UTTERANCE"})
                if speaker:
                    utterance_speakers.append({"utt_id": utt_id, "name": speaker})
                if prev_utt:
                    utterance_pairs.append({"from": prev_utt, "to": utt_id})
                    edges.append({"from": prev_utt, "to": utt_id, "type": "NEXT"})
                prev_utt = utt_id

        for idx in range(len(event_ids) - 1):
            event_pairs.append({"from": event_ids[idx], "to": event_ids[idx + 1]})
            edges.append({"from": event_ids[idx], "to": event_ids[idx + 1], "type": "NEXT"})
        for edge in causal_edges:
            edges.append({"from": edge.get("from"), "to": edge.get("to"), "type": edge.get("type") or "CAUSES"})

        nodes.extend(list(person_nodes.values()))
        nodes.extend(list(place_nodes.values()))
        nodes.extend(list(time_nodes.values()))
        nodes.extend(list(object_nodes.values()))

        self.detail_cache[scene_id] = {"owner_id": character_id, "scene_id": scene_id, "nodes": nodes, "edges": edges}

        if not self.driver:
            return

        canonical_name = canonical_name or ""
        with self.driver.session(database=settings.neo4j_db) as session:
            session.run(
                """
                MATCH (n {scene_id: $scene_id, owner_id: $cid})
                WHERE NOT n:SceneMemory
                DETACH DELETE n
                """,
                {"scene_id": scene_id, "cid": character_id},
            )

            if canonical_name:
                session.run(
                    """
                    MERGE (c:Character {owner_id: $cid, name: $name})
                    SET c.id = $cid, c.node_type = 'Person', c.type = 'Person'
                    """,
                    {"cid": character_id, "name": canonical_name},
                )

            session.run(
                """
                MERGE (sm:SceneMemory {scene_id: $scene_id})
                SET sm.owner_id = $cid
                """,
                {"scene_id": scene_id, "cid": character_id},
            )

            if event_rows:
                session.run(
                    """
                    UNWIND $events AS e
                    MERGE (ev:Event {event_id: e.event_id})
                    SET ev += e
                    """,
                    {"events": event_rows},
                )

            if event_pairs:
                session.run(
                    """
                    UNWIND $pairs AS p
                    MATCH (a:Event {event_id: p.from})
                    MATCH (b:Event {event_id: p.to})
                    MERGE (a)-[:NEXT]->(b)
                    """,
                    {"pairs": event_pairs},
                )
            if causal_edges:
                session.run(
                    """
                    UNWIND $edges AS e
                    MATCH (a:Event {event_id: e.from, owner_id: $cid})
                    MATCH (b:Event {event_id: e.to, owner_id: $cid})
                    MERGE (a)-[r:CAUSES]->(b)
                    SET r.type = e.type
                    """,
                    {"edges": causal_edges, "cid": character_id},
                )

            if event_ids:
                session.run(
                    """
                    MATCH (sm:SceneMemory {scene_id: $scene_id, owner_id: $cid})
                    UNWIND $event_ids AS eid
                    MATCH (e:Event {event_id: eid, owner_id: $cid})
                    MERGE (sm)-[:HAS_EVENT]->(e)
                    """,
                    {"scene_id": scene_id, "cid": character_id, "event_ids": event_ids},
                )

            if canonical_name:
                session.run(
                    """
                    MATCH (c:Character {owner_id: $cid, name: $name})
                    MATCH (sm:SceneMemory {scene_id: $scene_id, owner_id: $cid})
                    MERGE (c)-[:HAS_SCENE]->(sm)
                    """,
                    {"cid": character_id, "name": canonical_name, "scene_id": scene_id},
                )

            if participant_links:
                session.run(
                    """
                    UNWIND $links AS link
                    MERGE (p:Character {owner_id: $cid, name: link.name})
                    SET p.node_type = 'Person', p.type = 'Person'
                    WITH p, link
                    MATCH (e:Event {event_id: link.event_id, owner_id: $cid})
                    MERGE (e)-[:INVOLVES]->(p)
                    """,
                    {"links": participant_links, "cid": character_id},
                )

            if place_links:
                session.run(
                    """
                    UNWIND $links AS link
                    MERGE (pl:Place {owner_id: $cid, name: link.name})
                    SET pl.place_type = link.place_type, pl.node_type = 'Place', pl.type = 'Place'
                    WITH pl, link
                    MATCH (e:Event {event_id: link.event_id, owner_id: $cid})
                    MERGE (e)-[:AT_PLACE]->(pl)
                    """,
                    {"links": place_links, "cid": character_id},
                )

            if time_links:
                session.run(
                    """
                    UNWIND $links AS link
                    MERGE (t:TimePoint {owner_id: $cid, value: link.value})
                    SET t.node_type = 'TimePoint', t.type = 'TimePoint'
                    WITH t, link
                    MATCH (e:Event {event_id: link.event_id, owner_id: $cid})
                    MERGE (e)-[:AT_TIME]->(t)
                    """,
                    {"links": time_links, "cid": character_id},
                )

            if object_links:
                session.run(
                    """
                    UNWIND $links AS link
                    MERGE (o:Object {owner_id: $cid, name: link.name})
                    SET o.node_type = 'Object', o.type = 'Object'
                    WITH o, link
                    MATCH (e:Event {event_id: link.event_id, owner_id: $cid})
                    MERGE (e)-[:USES_OBJECT]->(o)
                    """,
                    {"links": object_links, "cid": character_id},
                )

            if utterances:
                session.run(
                    """
                    UNWIND $utterances AS u
                    MERGE (ut:Utterance {utt_id: u.utt_id})
                    SET ut.owner_id = $cid, ut.scene_id = $scene_id, ut.order = u.order,
                        ut.speaker = u.speaker, ut.text = u.text, ut.search_text = u.search_text,
                        ut.node_type = 'Utterance', ut.type = 'Utterance'
                    WITH ut, u
                    MATCH (e:Event {event_id: u.event_id, owner_id: $cid})
                    MERGE (e)-[:HAS_UTTERANCE]->(ut)
                    """,
                    {"utterances": utterances, "cid": character_id, "scene_id": scene_id},
                )

            if utterance_pairs:
                session.run(
                    """
                    UNWIND $pairs AS p
                    MATCH (u1:Utterance {utt_id: p.from})
                    MATCH (u2:Utterance {utt_id: p.to})
                    MERGE (u1)-[:NEXT]->(u2)
                    """,
                    {"pairs": utterance_pairs},
                )

            if utterance_speakers:
                session.run(
                    """
                    UNWIND $rows AS r
                    MERGE (c:Character {owner_id: $cid, name: r.name})
                    SET c.node_type = 'Person', c.type = 'Person'
                    WITH c, r
                    MATCH (u:Utterance {utt_id: r.utt_id})
                    MERGE (u)-[:SPOKEN_BY]->(c)
                    """,
                    {"rows": utterance_speakers, "cid": character_id},
                )

    def query_scene_detail(self, character_id: str, scene_id: str, query_text: str = "", limit: int = 12) -> List[Dict[str, Any]]:
        if not scene_id:
            return []
        detail_pack = self.query_scene_detail_pack(character_id, scene_id)
        nodes = self._detail_nodes_from_pack(detail_pack)
        return nodes[:limit]

    def _node_to_dict(self, node: Any, node_type: str) -> Dict[str, Any]:
        data = dict(node) if node is not None else {}
        data["type"] = node_type
        data["node_type"] = node_type
        if "id" not in data:
            if node_type == "Event":
                data["id"] = data.get("event_id")
            elif node_type == "Utterance":
                data["id"] = data.get("utt_id")
            elif node_type in {"Person", "Place", "Object"}:
                data["id"] = data.get("name")
            elif node_type == "TimePoint":
                data["id"] = data.get("value")
        return data

    def _detail_nodes_from_pack(self, pack: Dict[str, Any]) -> List[Dict[str, Any]]:
        if not pack:
            return []
        nodes: List[Dict[str, Any]] = []
        for event in pack.get("events") or []:
            nodes.append(self._node_to_dict(event, "Event"))
        for utter in pack.get("utterances") or []:
            nodes.append(self._node_to_dict(utter, "Utterance"))
        for place in pack.get("places") or []:
            nodes.append(self._node_to_dict(place, "Place"))
        for obj in pack.get("objects") or []:
            nodes.append(self._node_to_dict(obj, "Object"))
        for t in pack.get("times") or []:
            nodes.append(self._node_to_dict(t, "TimePoint"))
        for c in pack.get("characters") or []:
            nodes.append(self._node_to_dict(c, "Person"))
        for sp in pack.get("speakers") or []:
            nodes.append(self._node_to_dict(sp, "Person"))
        return nodes

    def query_scene_detail_pack(self, character_id: str, scene_id: str) -> Dict[str, Any]:
        if not scene_id:
            return {}
        if not self.driver:
            graph = self.detail_cache.get(scene_id, {})
            nodes = graph.get("nodes") or []
            events = [n for n in nodes if (n.get("type") or n.get("node_type")) == "Event"]
            utterances = [n for n in nodes if (n.get("type") or n.get("node_type")) == "Utterance"]
            places = [n for n in nodes if (n.get("type") or n.get("node_type")) == "Place"]
            times = [n for n in nodes if (n.get("type") or n.get("node_type")) == "TimePoint"]
            objects = [n for n in nodes if (n.get("type") or n.get("node_type")) == "Object"]
            characters = [n for n in nodes if (n.get("type") or n.get("node_type")) in {"Person", "Character"}]
            return {
                "scene_id": scene_id,
                "scene_gist": "",
                "events": events,
                "event_edges": [e for e in (graph.get("edges") or []) if e.get("type") in {"NEXT", "CAUSES"}],
                "places": places,
                "times": times,
                "objects": objects,
                "characters": characters,
                "utterances": utterances,
                "utter_edges": [e for e in (graph.get("edges") or []) if e.get("type") == "NEXT"],
                "cross_scene_candidates": [],
                "event_cnt": len(events),
            }

        cypher_events = """
        MATCH (s:SceneMemory {scene_id:$scene_id, owner_id:$owner_id})-[:HAS_EVENT]->(e:Event)
        RETURN s.scene_id AS scene_id,
               s.scene_gist AS scene_gist,
               e {.*, element_id: elementId(e)} AS event
        ORDER BY coalesce(e.index, e.order, e.created_at) ASC
        LIMIT 50
        """
        cypher_pack = """
        UNWIND $event_ids AS eid
        MATCH (e:Event {owner_id:$owner_id, scene_id:$scene_id})
        WHERE e.event_id = eid
        OPTIONAL MATCH (e)-[:AT_PLACE]->(p:Place)
        OPTIONAL MATCH (e)-[:AT_TIME]->(t:TimePoint)
        OPTIONAL MATCH (e)-[:USES_OBJECT]->(o:Object)
        OPTIONAL MATCH (e)-[:INVOLVES]->(c:Character)
        OPTIONAL MATCH (e)-[:HAS_UTTERANCE]->(u:Utterance)
        OPTIONAL MATCH (u)-[:SPOKEN_BY]->(sp:Character)
        RETURN
          collect(DISTINCT p)[0..10] AS places,
          collect(DISTINCT t)[0..10] AS times,
          collect(DISTINCT o)[0..30] AS objects,
          collect(DISTINCT c)[0..30] AS involved_characters,
          collect(DISTINCT u)[0..40] AS utterances,
          collect(DISTINCT sp)[0..20] AS speakers
        """
        cypher_edges = """
        MATCH (s:SceneMemory {scene_id:$scene_id, owner_id:$owner_id})-[:HAS_EVENT]->(e:Event)
        OPTIONAL MATCH (e)-[r:NEXT|CAUSES]->(e2:Event)
        WHERE (s)-[:HAS_EVENT]->(e2)
        RETURN collect(DISTINCT {
          from: e.event_id,
          to: e2.event_id,
          type: type(r)
        }) AS event_edges
        """
        cypher_utter = """
        MATCH (s:SceneMemory {scene_id:$scene_id, owner_id:$owner_id})
              -[:HAS_EVENT]->(e:Event)
              -[:HAS_UTTERANCE]->(u:Utterance)
        OPTIONAL MATCH (u)-[:NEXT]->(u2:Utterance)
        RETURN collect(DISTINCT {
          from: u.utt_id,
          to: u2.utt_id
        }) AS utter_edges
        """
        cypher_cross = """
        MATCH (s:SceneMemory {scene_id:$scene_id, owner_id:$owner_id})
              -[:HAS_EVENT]->(e:Event)
              -[:INVOLVES]->(c:Character)
        WITH DISTINCT c LIMIT 5
        MATCH (c)<-[:SPOKEN_BY]-(u:Utterance)
              <-[:HAS_UTTERANCE]-(e2:Event)
              <-[:HAS_EVENT]-(s2:SceneMemory {owner_id:$owner_id})
        WHERE s2.scene_id <> $scene_id
        RETURN s2.scene_id AS scene_id,
               s2.scene_gist AS scene_gist
        LIMIT 5
        """
        with self.driver.session(database=settings.neo4j_db) as session:
            events: List[Dict[str, Any]] = []
            scene_gist = ""
            event_ids: List[str] = []
            event_records = session.run(
                cypher_events,
                {"scene_id": scene_id, "owner_id": character_id},
            )
            for record in event_records:
                if not scene_gist:
                    scene_gist = record.get("scene_gist") or ""
                event = record.get("event")
                if not event:
                    continue
                events.append(self._node_to_dict(event, "Event"))
                event_id = event.get("event_id") or event.get("id") or event.get("element_id")
                if event_id:
                    event_ids.append(str(event_id))
            event_cnt = len(events)
            logger.info(
                "detail_pack.events scene_id=%s owner_id=%s event_cnt=%s",
                scene_id,
                character_id,
                event_cnt,
            )
            if not event_ids:
                logger.info(
                    "detail_pack.empty_events scene_id=%s owner_id=%s",
                    scene_id,
                    character_id,
                )
                return {
                    "scene_id": scene_id,
                    "scene_gist": scene_gist,
                    "events": events,
                    "event_edges": [],
                    "places": [],
                    "times": [],
                    "objects": [],
                    "characters": [],
                    "utterances": [],
                    "utter_edges": [],
                    "cross_scene_candidates": [],
                    "event_cnt": event_cnt,
                }

            pack_record = session.run(
                cypher_pack,
                {"scene_id": scene_id, "owner_id": character_id, "event_ids": event_ids},
            ).single()
            places = [self._node_to_dict(n, "Place") for n in ((pack_record or {}).get("places") or []) if n]
            times = [self._node_to_dict(n, "TimePoint") for n in ((pack_record or {}).get("times") or []) if n]
            objects = [self._node_to_dict(n, "Object") for n in ((pack_record or {}).get("objects") or []) if n]
            characters = [
                self._node_to_dict(n, "Person") for n in ((pack_record or {}).get("involved_characters") or []) if n
            ]
            utterances = [
                self._node_to_dict(n, "Utterance") for n in ((pack_record or {}).get("utterances") or []) if n
            ]
            speakers = [self._node_to_dict(n, "Person") for n in ((pack_record or {}).get("speakers") or []) if n]
            logger.info(
                "detail_pack.nodes scene_id=%s owner_id=%s events=%s utterances=%s places=%s times=%s objects=%s characters=%s speakers=%s",
                scene_id,
                character_id,
                len(events),
                len(utterances),
                len(places),
                len(times),
                len(objects),
                len(characters),
                len(speakers),
            )
            edges_rec = session.run(
                cypher_edges,
                {"scene_id": scene_id, "owner_id": character_id},
            ).single()
            utter_rec = session.run(
                cypher_utter,
                {"scene_id": scene_id, "owner_id": character_id},
            ).single()
            cross_records = session.run(
                cypher_cross,
                {"scene_id": scene_id, "owner_id": character_id},
            )
            cross_candidates = [
                {"scene_id": row.get("scene_id"), "scene_gist": row.get("scene_gist")} for row in cross_records
            ]
            return {
                "scene_id": scene_id,
                "scene_gist": scene_gist,
                "events": events,
                "event_edges": edges_rec.get("event_edges") if edges_rec else [],
                "places": places,
                "times": times,
                "objects": objects,
                "characters": characters,
                "utterances": utterances,
                "utter_edges": utter_rec.get("utter_edges") if utter_rec else [],
                "cross_scene_candidates": cross_candidates,
                "speakers": speakers,
                "event_cnt": event_cnt,
            }

    def delete_persona(self, character_id: str) -> None:
        self.persona_cache.pop(character_id, None)
        self.scene_cache.pop(character_id, None)
        self.detail_cache = {k: v for k, v in self.detail_cache.items() if v.get("owner_id") != character_id}
        self._remove_record(self.persona_store_path, character_id)
        self._remove_record(self.related_store_path, character_id)
        self._remove_record(self.worldrule_store_path, character_id)
        self._remove_record(self.inspiration_store_path, character_id)
        self._remove_record(self.scene_store_path, character_id)
        self._remove_record(self.detail_store_path, character_id)
        self._remove_keyword_index(character_id)

        if not self.driver:
            return
        cypher = """
        MATCH (n {owner_id: $cid})
        DETACH DELETE n
        """
        with self.driver.session(database=settings.neo4j_db) as session:
            session.run(cypher, {"cid": character_id})

    def delete_all_personas(self) -> None:
        self.persona_cache.clear()
        self.scene_cache.clear()
        self.detail_cache.clear()
        self.keyword_index_cache.clear()
        self._keyword_index_loaded = True
        _write_records(self.persona_store_path, [])
        _write_records(self.related_store_path, [])
        _write_records(self.worldrule_store_path, [])
        _write_records(self.inspiration_store_path, [])
        _write_records(self.scene_store_path, [])
        _write_records(self.detail_store_path, [])
        if self.keyword_index_path.exists():
            with self.keyword_index_path.open("w", encoding="utf-8", newline="\n") as fh:
                fh.write("")
        for legacy_path in [
            Path("artifacts") / "scene_sequences.jsonl",
            Path("src") / "artifacts" / "scene_sequences.jsonl",
        ]:
            if legacy_path.exists():
                with legacy_path.open("w", encoding="utf-8", newline="\n") as fh:
                    fh.write("")

        if not self.driver:
            return
        cypher = """
        MATCH (n)
        WHERE n:Character OR n:SceneMemory OR n:Event OR n:Action OR n:Utterance OR n:Place OR n:Person OR n:Object OR n:CanonRule OR n:TimePoint OR n:Detail
        DETACH DELETE n
        """
        with self.driver.session(database=settings.neo4j_db) as session:
            session.run(cypher)

    def reset_all_data(self) -> Dict[str, Any]:
        files_deleted: List[str] = []
        files_cleared: List[str] = []
        paths = [
            self.persona_store_path,
            self.related_store_path,
            self.worldrule_store_path,
            self.inspiration_store_path,
            self.scene_store_path,
            self.detail_store_path,
            self.keyword_index_path,
            Path("artifacts") / "scene_sequences.jsonl",
            Path("src") / "artifacts" / "scene_sequences.jsonl",
        ]
        for path in paths:
            try:
                if path.exists():
                    path.unlink()
                    files_deleted.append(str(path))
            except Exception:
                try:
                    with path.open("w", encoding="utf-8", newline="\n") as fh:
                        fh.write("")
                    files_cleared.append(str(path))
                except Exception:
                    continue

        self.persona_cache.clear()
        self.scene_cache.clear()
        self.detail_cache.clear()
        self.keyword_index_cache.clear()
        self._keyword_index_loaded = False

        neo4j_deleted = {"nodes": 0, "relationships": 0, "available": bool(self.driver)}
        if not self.driver:
            return {
                "files_deleted": files_deleted,
                "files_cleared": files_cleared,
                "embedding_reset": {"keyword_index": True, "in_memory_cache": True, "vector_index": "neo4j_index_kept"},
                "neo4j_deleted": neo4j_deleted,
            }

        label_filter = " OR ".join(
            [
                "n:Character",
                "n:SceneMemory",
                "n:Event",
                "n:Action",
                "n:Utterance",
                "n:Place",
                "n:Person",
                "n:Object",
                "n:CanonRule",
                "n:TimePoint",
                "n:Detail",
            ]
        )
        with self.driver.session(database=settings.neo4j_db) as session:
            node_record = session.run(
                f"""
                MATCH (n)
                WHERE {label_filter}
                RETURN count(n) AS nodes
                """
            ).single()
            rel_record = session.run(
                f"""
                MATCH (n)
                WHERE {label_filter}
                MATCH (n)-[r]-()
                RETURN count(DISTINCT r) AS rels
                """
            ).single()
            if node_record:
                neo4j_deleted["nodes"] = node_record.get("nodes", 0)
            if rel_record:
                neo4j_deleted["relationships"] = rel_record.get("rels", 0)

            session.run(
                f"""
                MATCH (n)
                WHERE {label_filter}
                DETACH DELETE n
                """
            )

        return {
            "files_deleted": files_deleted,
            "files_cleared": files_cleared,
            "embedding_reset": {"keyword_index": True, "in_memory_cache": True, "vector_index": "neo4j_index_kept"},
            "neo4j_deleted": neo4j_deleted,
        }

    def list_personas_from_file(self, limit: int = 50) -> List[Dict[str, Any]]:
        records = _read_records(self.persona_store_path)
        return records[-limit:]

    def _remove_keyword_index(self, character_id: str) -> None:
        self._load_keyword_index()
        if character_id in self.keyword_index_cache:
            self.keyword_index_cache.pop(character_id, None)
            self._persist_keyword_index()

    def _remove_record(self, path: Path, character_id: str) -> None:
        records = _read_records(path)
        kept = [
            r
            for r in records
            if (r.get("character_id") or r.get("owner_id")) != character_id
        ]
        _write_records(path, kept)

    def _build_search_text(self, node: Dict[str, Any]) -> str:
        parts: List[str] = []
        for key in [
            "summary",
            "text",
            "verb",
            "name",
            "label",
            "description",
            "role",
            "content",
            "object",
            "value",
            "utterance",
            "place_type",
        ]:
            val = node.get(key)
            if val:
                parts.append(str(val))
        return " ".join(parts).strip()

    def _score_scene_embeddings(
        self,
        query_emb: List[float],
        scenes: List[Dict[str, Any]],
        limit: int,
    ) -> List[Dict[str, Any]]:
        import numpy as np

        expected_dims = settings.embed_dimensions if settings.embed_dimensions > 0 else 0
        if not scenes or not query_emb:
            return []
        if expected_dims and len(query_emb) != expected_dims:
            return []
        q = np.array(query_emb, dtype=float)
        if q.size == 0:
            return []
        scored: List[Dict[str, Any]] = []
        for scene in scenes:
            emb_raw = scene.get("embedding", [])
            if expected_dims and len(emb_raw) != expected_dims:
                continue
            emb = np.array(emb_raw, dtype=float)
            if emb.size == 0:
                continue
            score = float(np.dot(q, emb) / (np.linalg.norm(q) * np.linalg.norm(emb) + 1e-8))
            copy_scene = dict(scene)
            copy_scene["score"] = score
            scored.append(copy_scene)
        return sorted(scored, key=lambda x: x.get("score", 0), reverse=True)[:limit]

    def _query_scene_vectors_by_scan(self, character_id: str, query_emb: List[float], limit: int) -> List[Dict[str, Any]]:
        if not self.driver:
            return []
        cypher = """
        MATCH (s:SceneMemory {owner_id: $cid})
        WHERE s.embedding IS NOT NULL
        RETURN s {.*, embedding: s.embedding} AS scene
        """
        with self.driver.session(database=settings.neo4j_db) as session:
            result = session.run(cypher, {"cid": character_id})
            scenes = [record["scene"] for record in result]
        return self._score_scene_embeddings(query_emb, scenes, limit)

    def _query_scene_cache(self, character_id: str, query_emb: List[float], limit: int) -> List[Dict[str, Any]]:
        scenes = self.scene_cache.get(character_id, [])
        return self._score_scene_embeddings(query_emb, scenes, limit)
