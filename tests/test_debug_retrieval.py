import pytest
from starlette.requests import Request

from mem_persona_agent.config import settings
from mem_persona_agent.memory.graph_store import GraphStore
from mem_persona_agent.memory.retriever import MemoryRetriever


def test_query_text_utf8():
    scope = {
        "type": "http",
        "method": "GET",
        "path": "/debug/jsonl/stats",
        "query_string": b"owner_id=cid&query=%E4%BA%89%E5%90%B5",
        "headers": [],
    }
    request = Request(scope)
    assert request.query_params.get("query", "") == "争吵"


@pytest.mark.asyncio
async def test_vector_retrieval_returns_candidates(monkeypatch):
    monkeypatch.setattr(settings, "neo4j_uri", "")
    monkeypatch.setattr(settings, "neo4j_username", "")
    monkeypatch.setattr(settings, "neo4j_password", "")
    monkeypatch.setattr(settings, "embed_dimensions", 3)
    store = GraphStore()
    store.scene_cache["cid"] = [
        {
            "scene_id": "s1",
            "scene_gist": "测试场景",
            "anchors": ["争吵"],
            "participants": ["甲"],
            "place": {"name": "家里", "type": "家"},
            "embedding": [0.1, 0.2, 0.3],
        }
    ]

    async def fake_embed(text, stats=None):
        return [0.1, 0.2, 0.3]

    monkeypatch.setattr("mem_persona_agent.memory.retriever.embed", fake_embed)
    retriever = MemoryRetriever(store)
    candidates, meta = await retriever.retrieve_scene_candidates("cid", user_input="争吵", limit=3)

    assert meta["vector_count"] > 0
    assert candidates[0]["scene_id"] == "s1"


def test_vector_dimension_guard(monkeypatch):
    monkeypatch.setattr(settings, "neo4j_uri", "")
    monkeypatch.setattr(settings, "neo4j_username", "")
    monkeypatch.setattr(settings, "neo4j_password", "")
    monkeypatch.setattr(settings, "embed_dimensions", 3)
    store = GraphStore()
    store.scene_cache["cid"] = [{"scene_id": "s1", "embedding": [0.1, 0.2, 0.3]}]
    assert store.query_scene_vectors("cid", [0.1, 0.2], limit=3) == []
