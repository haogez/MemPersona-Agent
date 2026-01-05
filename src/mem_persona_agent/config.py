from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List

from dotenv import load_dotenv

load_dotenv()


def _get_env(name: str, default: str | None = None) -> str:
    value = os.getenv(name, default)
    if value is None:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def _get_env_bool(name: str, default: str = "true") -> bool:
    value = os.getenv(name, default)
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


@dataclass
class Settings:
    llm_api_base: str = os.getenv("LLM_API_BASE_URL", "")
    llm_api_key: str = os.getenv("LLM_API_KEY", "")
    dmx_api_key: str = os.getenv("DMX_API_KEY", "") or os.getenv("LLM_API_KEY", "")
    llm_model_name: str = os.getenv("LLM_MODEL_NAME", "gpt-4.1-mini")
    chat_model_name: str = os.getenv("CHAT_MODEL_NAME", "gpt-5")
    embed_model_name: str = os.getenv("EMBED_MODEL_NAME", "text-embedding-3-small")
    embedding_model_name: str = os.getenv(
        "EMBEDDING_MODEL_NAME",
        os.getenv("EMBED_MODEL_NAME", "text-embedding-3-small"),
    )
    embed_dimensions: int = int(os.getenv("EMBED_DIMENSIONS", "1536"))
    enable_memory_pipeline: bool = _get_env_bool("ENABLE_MEMORY_PIPELINE", "true")
    enable_seed_overrides: bool = _get_env_bool("ENABLE_SEED_OVERRIDES", "false")
    debug_trace_chat: bool = _get_env_bool("DEBUG_TRACE_CHAT", "true")
    enable_vector_retrieval: bool = _get_env_bool("ENABLE_VECTOR_RETRIEVAL", "true")
    llm_timeout_seconds: float = float(os.getenv("LLM_TIMEOUT_SECONDS", "0"))  # 0/负数表示不限制
    scene_count: int = int(os.getenv("SCENE_COUNT", "20"))
    model_prices_json: str = os.getenv("MODEL_PRICES_JSON", "")

    neo4j_uri: str = os.getenv("NEO4J_URI", "")
    neo4j_username: str = os.getenv("NEO4J_USERNAME", "") or os.getenv("NEO4J_USER", "")
    neo4j_password: str = os.getenv("NEO4J_PASSWORD", "")
    neo4j_db: str = os.getenv("NEO4J_DB", "neo4j")

    message_history: int = int(os.getenv("MESSAGE_HISTORY", "6"))
    persona_store_path: str = os.getenv("PERSONA_STORE_PATH", "artifacts/personas.jsonl")
    related_store_path: str = os.getenv("RELATED_STORE_PATH", "artifacts/related_characters.jsonl")
    worldrule_store_path: str = os.getenv("WORLDRULE_STORE_PATH", "artifacts/worldrules.jsonl")
    inspiration_store_path: str = os.getenv("INSPIRATION_STORE_PATH", "artifacts/inspirations.jsonl")
    scene_memory_store_path: str = os.getenv("SCENE_MEMORY_STORE_PATH", "artifacts/scene_memories.jsonl")
    scene_keyword_index_path: str = os.getenv("SCENE_KEYWORD_INDEX_PATH", "artifacts/scene_keywords.json")
    detail_graph_store_path: str = os.getenv("DETAIL_GRAPH_STORE_PATH", "artifacts/scene_detail_graphs.jsonl")

    # gating keywords（小写/中文直接 contains 匹配）
    memory_trigger_keywords: List[str] = field(
        default_factory=lambda: os.getenv(
            "MEMORY_TRIGGER_KEYWORDS",
            "经历,那次,以前,过去,为什么会这样,讲讲,回忆,曾经,当时,小时候,初中,高中,大学,吵架,误会,日记",
        ).split(",")
    )
    smalltalk_keywords: List[str] = field(
        default_factory=lambda: os.getenv(
            "SMALLTALK_KEYWORDS",
            "你好,在吗,哈哈,呵呵,随便聊聊,最近怎么样,今天怎么样,天气,问候,无聊",
        ).split(",")
    )
    memory_score_threshold: float = float(os.getenv("MEMORY_SCORE_THRESHOLD", "0.0"))
    scene_score_threshold: float = float(os.getenv("SCENE_SCORE_THRESHOLD", "0.15"))
    fast_prefix_chars_default: int = int(os.getenv("FAST_PREFIX_CHARS", "240"))
    stream_chunk_chars_default: int = int(os.getenv("STREAM_CHUNK_CHARS", "24"))
    stream_flush_ms_default: int = int(os.getenv("STREAM_FLUSH_MS", "100"))
    stream_delay_period_seconds: float = float(os.getenv("STREAM_DELAY_PERIOD_SECONDS", "0"))
    stream_delay_comma_seconds: float = float(os.getenv("STREAM_DELAY_COMMA_SECONDS", "0"))
    qwen_visible_devices: str = os.getenv("QWEN_VISIBLE_DEVICES", "0,1")
    llm_temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.7"))
    llm_top_p: float = float(os.getenv("LLM_TOP_P", "0.9"))
    llm_max_tokens_dialog: int = int(os.getenv("LLM_MAX_TOKENS_DIALOG", "1024"))
    llm_max_tokens_persona: int = int(os.getenv("LLM_MAX_TOKENS_PERSONA", "4096"))
    llm_stream_default: bool = _get_env_bool("LLM_STREAM_DEFAULT", "false")

    @property
    def neo4j_available(self) -> bool:
        return bool(self.neo4j_uri and self.neo4j_username and self.neo4j_password)


settings = Settings()
