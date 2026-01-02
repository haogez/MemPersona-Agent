from mem_persona_agent.llm.chunker import chunk_tokens


def test_chunk_tokens_splits_on_punct_and_length():
    tokens = list("你好，这是一次测试。继续输出更多文字，不要过于碎片。")
    chunks = chunk_tokens(tokens, chunk_chars=10)
    assert len(chunks) >= 3
    assert any("。" in c for c in chunks)
