from __future__ import annotations

import json
import logging
import time
import uuid
from typing import Any, Dict, List, Optional
from pathlib import Path

import httpx
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel

from mem_persona_agent.agent.dialogue_orchestrator import DialogueOrchestrator
from mem_persona_agent.agent.roleplay_agent import RoleplayAgent
from mem_persona_agent.config import settings
from mem_persona_agent.llm import ChatClient
from mem_persona_agent.llm.client import ChatError
from mem_persona_agent.memory import GraphStore, MemoryRetriever, MemoryWriter
from mem_persona_agent.persona.generator import PersonaGenerator
from mem_persona_agent.persona.schema import Persona
from mem_persona_agent.utils import estimate_message_tokens, estimate_tokens
from mem_persona_agent.utils.state_manager import TurnIDManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", encoding="utf-8")
logger = logging.getLogger(__name__)
logging.getLogger("neo4j.notifications").setLevel(logging.WARNING)
logging.getLogger("neo4j").setLevel(logging.WARNING)

app = FastAPI(title="MemPersona-Agent")

store = GraphStore()
store.ensure_schema()
writer = MemoryWriter(store)
retriever = MemoryRetriever(store)
turn_manager = TurnIDManager()
orchestrator = DialogueOrchestrator(retriever, turn_manager=turn_manager)


def _to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False)
    except Exception:
        return str(value)


def _truncate_text(value: Any, max_len: int = 2000, head: int = 800, tail: int = 800) -> str:
    text = _to_text(value)
    if len(text) <= max_len:
        return text
    if tail <= 0:
        return f"{text[:head]}...<truncated {len(text)} chars>..."
    return f"{text[:head]}...<truncated {len(text)} chars>...{text[-tail:]}"


def _summarize_history(history: List[ChatHistoryItem], limit: int = 3) -> List[Dict[str, Any]]:
    items = history[-limit:] if limit else []
    summaries: List[Dict[str, Any]] = []
    for item in items:
        content = item.content or ""
        summaries.append(
            {
                "role": item.role,
                "content": _truncate_text(content, max_len=80, head=80, tail=0),
                "length": len(content),
            }
        )
    return summaries


def _summarize_nodes(nodes: List[Dict[str, Any]], limit: int = 10) -> List[Dict[str, Any]]:
    summaries: List[Dict[str, Any]] = []
    for node in nodes[:limit]:
        node_type = node.get("type") or node.get("node_type")
        text = node.get("text") or node.get("content") or node.get("utterance") or node.get("summary") or ""
        name = node.get("name") or node.get("label") or node.get("speaker") or ""
        summaries.append(
            {
                "id": node.get("id"),
                "type": node_type,
                "name": _truncate_text(name, max_len=80, head=80, tail=0),
                "text": _truncate_text(text, max_len=80, head=80, tail=0),
            }
        )
    return summaries


def _summarize_edges(edges: List[Dict[str, Any]], limit: int = 10) -> List[Dict[str, Any]]:
    summaries: List[Dict[str, Any]] = []
    for edge in edges[:limit]:
        summaries.append(
            {
                "from": edge.get("from"),
                "to": edge.get("to"),
                "type": edge.get("type") or edge.get("rel"),
            }
        )
    return summaries


def _log_debug(stage: str, payload: Dict[str, Any]) -> None:
    logger.info("chat_stream.%s %s", stage, json.dumps(payload, ensure_ascii=False))


def _llm_error_detail(exc: Exception, *, stage: str, model: str | None = None) -> Dict[str, Any]:
    detail: Dict[str, Any] = {"stage": stage, "type": exc.__class__.__name__, "message": _truncate_text(str(exc))}
    if model:
        detail["model"] = model
    if isinstance(exc, ChatError):
        detail["status_code"] = exc.status_code
        detail["body"] = exc.response_text
        return detail
    if isinstance(exc, httpx.HTTPStatusError):
        status_code = exc.response.status_code if exc.response else None
        body = None
        if exc.response is not None:
            try:
                body = _truncate_text(exc.response.text, max_len=800, head=800, tail=0)
            except Exception:
                body = None
        detail["status_code"] = status_code
        detail["body"] = body
    return detail


def load_persona_from_store(character_id: str) -> Optional[Persona]:
    path = Path(settings.persona_store_path)
    if not path.exists():
        return None
    latest: Optional[Persona] = None
    try:
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if record.get("character_id") != character_id:
                    continue
                persona_data = record.get("persona")
                if persona_data is None:
                    continue
                try:
                    latest = Persona.model_validate(persona_data)
                except Exception:
                    continue
    except Exception:
        return None
    return latest


class PersonaRequest(BaseModel):
    seed: str


class PersonaResponse(BaseModel):
    character_id: str
    persona: Persona


class MemoryRequest(BaseModel):
    character_id: str
    persona: Persona
    seed: Optional[str] = None
    related_characters: Optional[List[Dict[str, Any]]] = None


class MemoryResponse(BaseModel):
    sequence: List[str]
    scenes: List[Dict[str, Any]]
    stats: Optional[Dict[str, Any]] = None


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
    used_memory: Dict[str, Any]


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
    result = store.reset_all_data()
    return {"status": "ok", "reset": "all data", "details": result}


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
    result = await writer.generate_and_store(
        body.character_id,
        body.persona.model_dump(),
        seed=body.seed or "",
        related_characters=body.related_characters,
    )
    return MemoryResponse(
        sequence=result.get("sequence", []),
        scenes=result.get("scenes", []),
        stats=result.get("stats"),
    )


@app.get("/persona/list", response_model=PersonaListResponse)
async def list_personas(limit: int = 50):
    return PersonaListResponse(personas=store.list_personas_from_file(limit=limit))


@app.post("/persona/delete")
async def delete_persona(body: PersonaDeleteRequest):
    store.delete_persona(body.character_id)
    return {"status": "ok", "deleted": body.character_id}


@app.post("/chat", response_model=ChatResponse)
async def chat(body: ChatRequest):
    agent = RoleplayAgent(
        body.persona,
        body.character_id,
        retriever,
        client=ChatClient(model=settings.chat_model_name),
    )
    for item in body.dialogue_history:
        if item.role == "user":
            agent.buffer.add_user(item.content)
        else:
            agent.buffer.add_assistant(item.content)
    reply, memory = await agent.chat(body.user_input, place=body.place, npc=body.npc)
    return ChatResponse(reply=reply, used_memory=memory)


@app.get("/health")
async def health():
    connected = store.driver is not None
    return {"status": "ok", "neo4j": connected, "neo4j_available": connected}


@app.get("/debug/jsonl/stats")
async def debug_scene_stats(request: Request, owner_id: str, query: str = ""):
    query_text = request.query_params.get("query", "")
    logger.info("debug/jsonl/stats owner_id=%s query_text=%r", owner_id, query_text)
    candidates, meta = await retriever.retrieve_scene_candidates(owner_id, user_input=query_text, place=None, npc=None, limit=3)
    payload = {
        "owner_id": owner_id,
        "returned": len(candidates),
        "top_scene_id": candidates[0].get("scene_id") if candidates else None,
        "meta": meta,
    }
    return JSONResponse(content=payload, media_type="application/json; charset=utf-8")


class ChatStreamRequest(BaseModel):
    character_id: str
    persona: Optional[Persona] = None
    place: Optional[str] = None
    npc: Optional[str] = None
    history: List[ChatHistoryItem] = []
    user_input: str
    mode: Optional[str] = "stream_only"
    inject_memory: bool = True
    stream_chunk_chars: int = 24
    stream_flush_ms: int = 100
    fast_prefix_chars: int = 240
    memory_top_k: int = 5


@app.post("/chat/stream")
async def chat_stream(body: ChatStreamRequest):
    if body.persona is None:
        persona = load_persona_from_store(body.character_id)
        if persona is None:
            raise HTTPException(status_code=400, detail="persona_not_found_for_character_id")
        body.persona = persona
    max_turns = settings.message_history
    history_dicts = [item.model_dump() for item in body.history[-max_turns * 2 :]]
    stream = orchestrator.run_chat_stream(body, history_dicts)
    return StreamingResponse(stream, media_type="text/event-stream")


def _decode_json_with_fallback(raw: bytes, encodings: tuple[str, ...] = ("utf-8", "gbk", "cp936")) -> tuple[Dict[str, Any], str]:
    last_error: Exception | None = None
    for enc in encodings:
        try:
            return json.loads(raw.decode(enc)), enc
        except Exception as exc:
            last_error = exc
    raise HTTPException(status_code=400, detail=f"Cannot decode payload; tried {encodings}; last_error={last_error}")
