from mem_persona_agent.config import settings
from mem_persona_agent.memory.graph_store import GraphStore


def test_scene_keyword_query_uses_index():
    settings.neo4j_uri = ""
    settings.neo4j_username = ""
    settings.neo4j_password = ""
    store = GraphStore()
    store.scene_cache["a"] = [
        {
            "scene_id": "s1",
            "anchors": ["coffee"],
            "participants": ["A"],
            "place": {"name": "cafe", "type": "place"},
            "scene_gist": "a",
            "embedding": [0.1],
        }
    ]
    store.keyword_index_cache = {"a": {"coffee": ["s1"]}}
    store._keyword_index_loaded = True
    res = store.query_scene_keywords("a", "coffee", limit=5)
    assert res[0]["scene_id"] == "s1"

