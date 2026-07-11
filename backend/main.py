import os
import uuid
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

import db
import orchestrator

app = FastAPI(title="Agentic System")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

db.init_db()

FRONTEND_DIR = Path(__file__).parent.parent / "frontend"


class ChatRequest(BaseModel):
    session_id: str | None = None
    message: str


class ChatResponse(BaseModel):
    session_id: str
    reply: str
    error: str | None = None


class AgentCreateRequest(BaseModel):
    name: str
    description: str
    system_prompt: str


class AgentUpdateRequest(BaseModel):
    description: str | None = None
    system_prompt: str | None = None


class TestAgentRequest(BaseModel):
    agent_name: str
    message: str


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    session_id = req.session_id or str(uuid.uuid4())
    try:
        reply = orchestrator.run_turn(session_id, req.message)
        return ChatResponse(session_id=session_id, reply=reply)
    except Exception as e:
        err_detail = f"{type(e).__name__}: {e}"
        return ChatResponse(
            session_id=session_id,
            reply=f"⚠️ خطای بک‌اند رخ داد:\n{err_detail}",
            error=err_detail,
        )


@app.get("/api/agents")
def list_agents():
    return db.list_agents()


@app.post("/api/agents")
def create_agent_endpoint(req: AgentCreateRequest):
    existing = db.get_agent(req.name)
    if existing:
        return {"ok": False, "error": "ایجنتی با این نام از قبل وجود دارد."}
    try:
        db.create_agent(req.name, req.description, req.system_prompt)
        return {"ok": True}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@app.put("/api/agents/{name}")
def update_agent_endpoint(name: str, req: AgentUpdateRequest):
    ok = db.update_agent(name, req.description, req.system_prompt)
    return {"ok": ok}


@app.delete("/api/agents/{name}")
def delete_agent_endpoint(name: str):
    ok = db.delete_agent(name)
    return {"ok": ok}


@app.post("/api/test-agent")
def test_agent_endpoint(req: TestAgentRequest):
    try:
        result = orchestrator.test_agent(req.agent_name, req.message)
        return {"ok": True, "result": result}
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}


@app.get("/api/history/{session_id}")
def history(session_id: str):
    return db.get_recent_messages(session_id, limit=100)


@app.get("/api/facts")
def facts():
    return db.get_all_facts()


@app.get("/api/health")
def health():
    groq_key_set = bool(os.environ.get("GROQ_API_KEY"))
    return {
        "status": "ok",
        "groq_key_set": groq_key_set,
        "agents_count": len(db.list_agents()),
    }


app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


@app.get("/")
def root():
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/manifest.json")
def manifest():
    return FileResponse(FRONTEND_DIR / "manifest.json")


@app.get("/sw.js")
def sw():
    return FileResponse(FRONTEND_DIR / "sw.js", media_type="application/javascript")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
