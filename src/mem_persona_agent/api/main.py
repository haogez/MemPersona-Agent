from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional

from fastapi import FastAPI
from pydantic import BaseModel

from mem_persona_agent.agent.roleplay_agent import RoleplayAgent
from mem_persona_agent.config import settings
from mem_persona_agent.llm import ChatClient
from mem_persona_agent.memory import GraphStore, MemoryRetriever, MemoryWriter
from mem_persona_agent.persona.generator import PersonaGenerator
from mem_persona_agent.persona.schema import Persona

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


@app.post("/persona/generate", response_model=PersonaResponse)
async def generate_persona(body: PersonaRequest):
    generator = PersonaGenerator()
    persona = await generator.generate(body.seed)
    character_id = str(uuid.uuid4())
    store.write_persona(character_id, persona.model_dump())
    return PersonaResponse(character_id=character_id, persona=persona)


@app.post("/memory/static/generate", response_model=MemoryResponse)
async def generate_memory(body: MemoryRequest):
    episodes = await writer.generate_and_store(body.character_id, body.persona.model_dump())
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
