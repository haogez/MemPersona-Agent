from __future__ import annotations

import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


def _get_env(name: str, default: str | None = None) -> str:
    value = os.getenv(name, default)
    if value is None:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


@dataclass
class Settings:
    llm_api_base: str = os.getenv("LLM_API_BASE_URL", "")
    llm_api_key: str = os.getenv("LLM_API_KEY", "")
    llm_model_name: str = os.getenv("LLM_MODEL_NAME", "text-embedding-3-large")
    embed_model_name: str = os.getenv("EMBED_MODEL_NAME", "text-embedding-3-large")

    neo4j_uri: str = os.getenv("NEO4J_URI", "")
    neo4j_username: str = os.getenv("NEO4J_USERNAME", "")
    neo4j_password: str = os.getenv("NEO4J_PASSWORD", "")
    neo4j_db: str = os.getenv("NEO4J_DB", "neo4j")

    message_history: int = int(os.getenv("MESSAGE_HISTORY", "6"))

    @property
    def neo4j_available(self) -> bool:
        return bool(self.neo4j_uri and self.neo4j_username and self.neo4j_password)


settings = Settings()
