from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional
import logging
import json

from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel

from mem_persona_agent.agent.roleplay_agent import RoleplayAgent
from mem_persona_agent.config import settings
from mem_persona_agent.llm import ChatClient
from mem_persona_agent.memory import GraphStore, MemoryRetriever, MemoryWriter
from mem_persona_agent.persona.generator import PersonaGenerator
from mem_persona_agent.persona.schema import Persona

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", encoding="utf-8")
logger = logging.getLogger(__name__)

app = FastAPI(title="MemPersona-Agent")

store = GraphStore()
store.ensure_schema()
writer = MemoryWriter(store)
retriever = MemoryRetriever(store)


class PersonaRequest(BaseModel):
    seed: str


class PersonaResponse(BaseModel):
    character_id: str
    persona: Persona


class MemoryRequest(BaseModel):
    character_id: str
    persona: Persona
    seed: Optional[str] = None


class MemoryResponse(BaseModel):
    episodes: List[Dict[str, Any]]


class ChatHistoryItem(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    character_id: str
    persona: Persona
    place: Optional[str] = None
    npc: Optional[str] = None
    mode: str = "static_only"
    dialogue_history: List[ChatHistoryItem] = []
    user_input: str


class ChatResponse(BaseModel):
    reply: str
    used_memory: List[Dict[str, Any]]


class PersonaDeleteRequest(BaseModel):
    character_id: str


class PersonaListResponse(BaseModel):
    personas: List[Dict[str, Any]]


@app.post("/persona/delete_all")
async def delete_all_personas():
    store.delete_all_personas()
    return {"status": "ok", "deleted": "all"}


@app.post("/reset/all")
async def reset_all():
    store.delete_all_personas()
    return {"status": "ok", "reset": "all personas + memories + related"}


@app.post("/persona/generate", response_model=PersonaResponse)
async def generate_persona(request: Request):
    raw = await request.body()
    data, used_encoding = _decode_json_with_fallback(raw)
    seed = data.get("seed") if isinstance(data, dict) else None
    if not seed or not isinstance(seed, str):
        raise HTTPException(status_code=400, detail="Invalid payload: missing seed")
    logger.info("API receive seed repr=%r len=%s encoding=%s", seed, len(seed), used_encoding)
    generator = PersonaGenerator()
    persona = await generator.generate(seed)
    character_id = str(uuid.uuid4())
    store.write_persona(character_id, persona.model_dump())
    return PersonaResponse(character_id=character_id, persona=persona)


@app.post("/memory/static/generate", response_model=MemoryResponse)
async def generate_memory(body: MemoryRequest):
    logger.info("API /memory/static/generate cid=%s seed=%s", body.character_id, body.seed)
    episodes = await writer.generate_and_store(body.character_id, body.persona.model_dump(), seed=body.seed or "")
    return MemoryResponse(episodes=episodes)


@app.get("/persona/list", response_model=PersonaListResponse)
async def list_personas(limit: int = 50):
    return PersonaListResponse(personas=store.list_personas_from_file(limit=limit))


@app.post("/persona/delete")
async def delete_persona(body: PersonaDeleteRequest):
    store.delete_persona(body.character_id)
    return {"status": "ok", "deleted": body.character_id}


@app.post("/chat", response_model=ChatResponse)
async def chat(body: ChatRequest):
    # rebuild memory buffer from history
    agent = RoleplayAgent(body.persona, body.character_id, retriever, client=ChatClient())
    for item in body.dialogue_history:
        if item.role == "user":
            agent.buffer.add_user(item.content)
        else:
            agent.buffer.add_assistant(item.content)

    reply, memory = await agent.chat(body.user_input, place=body.place, npc=body.npc)
    return ChatResponse(reply=reply, used_memory=memory)


@app.get("/health")
async def health():
    return {"status": "ok", "neo4j": store.driver is not None, "neo4j_available": settings.neo4j_available}


def _decode_json_with_fallback(raw: bytes, encodings: tuple[str, ...] = ("utf-8", "gbk", "cp936")) -> tuple[Dict[str, Any], str]:
    """Decode JSON body, trying UTF-8 first, then common Windows encodings to tolerate cp936 inputs."""
    last_error: Exception | None = None
    for enc in encodings:
        try:
            return json.loads(raw.decode(enc)), enc
        except Exception as exc:  # pragma: no cover - defensive
            last_error = exc
    raise HTTPException(status_code=400, detail=f"Cannot decode payload; tried {encodings}; last_error={last_error}")
