from mem_persona_agent.config import settings
from mem_persona_agent.memory.graph_store import GraphStore
from mem_persona_agent.memory.retriever import MemoryRetriever


def test_merge_scene_results_scores():
    settings.neo4j_uri = ""
    settings.neo4j_username = ""
    settings.neo4j_password = ""
    retriever = MemoryRetriever(GraphStore())
    vector_results = [{"scene_id": "s1", "score": 1.0, "scene_gist": "a"}]
    keyword_results = [
        {"scene_id": "s1", "score": 0.5, "scene_gist": "a"},
        {"scene_id": "s2", "score": 1.0, "scene_gist": "b"},
    ]
    merged = retriever._merge_scene_results(vector_results, keyword_results)
    assert merged[0]["scene_id"] == "s1"
    assert merged[0]["score"] >= merged[1]["score"]

