from mem_persona_agent.memory.gating import detect_open_slots, should_recall_scene


def test_detect_open_slots():
    slots = detect_open_slots("为什么会这样？")
    assert "why" in slots


def test_should_recall_scene_smalltalk():
    res = should_recall_scene("你好", [])
    assert not res["recall"]


def test_should_recall_scene_by_score():
    res = should_recall_scene("tell me", [{"score": 0.2}])
    assert res["recall"]

