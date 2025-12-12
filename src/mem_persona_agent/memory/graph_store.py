from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
import json

from neo4j import GraphDatabase, Driver

from mem_persona_agent.config import settings

logger = logging.getLogger(__name__)


@dataclass
class GraphStore:
    driver: Optional[Driver] = field(default=None)
    memory_cache: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    persona_cache: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    persona_store_path: Path = field(default_factory=lambda: Path(settings.persona_store_path))

    def __post_init__(self):
        if settings.neo4j_available:
            try:
                self.driver = GraphDatabase.driver(
                    settings.neo4j_uri,
                    auth=(settings.neo4j_username, settings.neo4j_password),
                )
            except Exception as exc:  # pragma: no cover - safety
                logger.warning("Neo4j connection failed, using in-memory store: %s", exc)
                self.driver = None

    def ensure_schema(self):
        if not self.driver:
            return
        queries = [
            """
            CREATE CONSTRAINT static_episode_id_unique IF NOT EXISTS
            FOR (e:StaticEpisode) REQUIRE e.id IS UNIQUE;
            """,
            """
            CREATE VECTOR INDEX episode_embedding_index IF NOT EXISTS
            FOR (e:StaticEpisode) ON (e.embedding)
            OPTIONS {
              indexConfig: {
                `vector.dimensions`: 1536,
                `vector.similarity_function`: 'cosine'
              }
            };
            """,
        ]
        with self.driver.session(database=settings.neo4j_db) as session:
            for q in queries:
                session.run(q)

    def write_persona(self, character_id: str, persona: Dict[str, Any]):
        """Persist persona definition; stores JSON string in Neo4j, local cache, and JSONL file."""
        self.persona_cache[character_id] = persona
        self._persist_persona_file(character_id, persona)
        if not self.driver:
            return

        cypher = """
        MERGE (c:Character {id: $cid})
        ON CREATE SET c.owner_id = $cid
        SET c.persona_json = $persona_json
        """
        with self.driver.session(database=settings.neo4j_db) as session:
            session.run(cypher, {"cid": character_id, "persona_json": json.dumps(persona, ensure_ascii=False)})

    def delete_persona(self, character_id: str):
        """Delete persona + related episodes from Neo4j and local caches/files."""
        self.persona_cache.pop(character_id, None)
        self.memory_cache.pop(character_id, None)
        self._delete_persona_file_record(character_id)

        if not self.driver:
            return
        cypher = """
        MATCH (c:Character {id: $cid})
        OPTIONAL MATCH (c)-[:HAS_STATIC_EPISODE]->(e:StaticEpisode)
        DETACH DELETE e, c
        """
        with self.driver.session(database=settings.neo4j_db) as session:
            session.run(cypher, {"cid": character_id})

    def write_static_episodes(self, character_id: str, episodes: List[Dict[str, Any]]):
        if not self.driver:
            self.memory_cache.setdefault(character_id, []).extend(episodes)
            return

        cypher = """
        UNWIND $episodes AS ep
        MERGE (c:Character {id: $cid})
        ON CREATE SET c.owner_id = $cid
        MERGE (e:StaticEpisode {id: ep.id})
        SET e += ep, e.owner_id = $cid
        MERGE (c)-[:HAS_STATIC_EPISODE]->(e)
        WITH e, ep
        UNWIND ep.people AS p
            MERGE (pr:PersonRef {name: p.name, owner_id: ep.owner_id})
            SET pr += p
            MERGE (e)-[:INVOLVES_PERSON]->(pr)
        WITH e, ep
        UNWIND ep.objects AS o
            MERGE (or: ObjectRef {name: o.name, owner_id: ep.owner_id})
            SET or += o
            MERGE (e)-[:INVOLVES_OBJECT]->(or)
        WITH e, ep
        UNWIND ep.places AS pl
            MERGE (plr: PlaceRef {name: pl.name, owner_id: ep.owner_id})
            SET plr += pl
            MERGE (e)-[:LOCATED_IN]->(plr)
        """
        with self.driver.session(database=settings.neo4j_db) as session:
            session.run(cypher, {"cid": character_id, "episodes": episodes})

    def query_similar(self, character_id: str, query_emb: List[float], npc: str | None = None, limit: int = 8) -> List[Dict[str, Any]]:
        if not self.driver:
            return self._query_cache(character_id, query_emb, npc, limit)

        cypher = """
        CALL db.index.vector.queryNodes('episode_embedding_index', $limit, $query_emb)
        YIELD node, score
        WHERE node.owner_id = $cid
        OPTIONAL MATCH (node)-[:INVOLVES_PERSON]->(p:PersonRef)
        WITH node, score, p, CASE WHEN p.name = $npc THEN 0.3 ELSE 0 END AS bonus
        RETURN node {.*, score: score + bonus} AS ep
        ORDER BY ep.score DESC
        LIMIT $limit
        """
        with self.driver.session(database=settings.neo4j_db) as session:
            result = session.run(cypher, {"cid": character_id, "query_emb": query_emb, "npc": npc, "limit": limit})
            return [record["ep"] for record in result]

    def _query_cache(self, character_id: str, query_emb: List[float], npc: str | None, limit: int) -> List[Dict[str, Any]]:
        # simple cosine similarity on cached episodes
        import numpy as np

        episodes = self.memory_cache.get(character_id, [])
        if not episodes:
            return []
        q = np.array(query_emb)
        scored: List[Dict[str, Any]] = []
        for ep in episodes:
            emb = np.array(ep.get("embedding", []))
            if emb.size == 0:
                continue
            score = float(np.dot(q, emb) / (np.linalg.norm(q) * np.linalg.norm(emb) + 1e-8))
            if npc:
                people = [p.get("name") for p in ep.get("people", [])]
                if npc in people:
                    score += 0.3
            copy_ep = dict(ep)
            copy_ep["score"] = score
            scored.append(copy_ep)
        return sorted(scored, key=lambda x: x.get("score", 0), reverse=True)[:limit]

    def list_personas_from_file(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Read recent personas from the local JSONL store."""
        if not self.persona_store_path.exists():
            return []
        records: List[Dict[str, Any]] = []
        with self.persona_store_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return records[-limit:]

    def _persist_persona_file(self, character_id: str, persona: Dict[str, Any]):
        self.persona_store_path.parent.mkdir(parents=True, exist_ok=True)
        record = {"character_id": character_id, "persona": persona}
        with self.persona_store_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _delete_persona_file_record(self, character_id: str):
        if not self.persona_store_path.exists():
            return
        lines = self.persona_store_path.read_text(encoding="utf-8").splitlines()
        kept = []
        for line in lines:
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if obj.get("character_id") != character_id:
                kept.append(line)
        self.persona_store_path.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")
