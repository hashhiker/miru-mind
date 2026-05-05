"""
Miru Mind – Web Server
=======================
Startet einen lokalen HTTP-Server, der das Browser-Frontend bedient
und als sicherer Proxy für den LLM-API-Call fungiert.

Start:
  pip install fastapi uvicorn
  python server.py
  → http://localhost:8000
"""

import json
import os
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

# Importiere die LLM-Kernfunktion und den System-Prompt aus main.py
from main import _call_llm, build_system_prompt, SYSTEM_PROMPT_BASE

app = FastAPI(title="Miru Mind API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)


# ─── Datenmodelle ─────────────────────────────────────────────────────────────

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: list[Message]
    history: dict = {}   # Optionale Memory-Daten (sessions, moods, user_profile)


# ─── API Endpoints ────────────────────────────────────────────────────────────

@app.post("/api/chat")
def chat(req: ChatRequest):
    """
    Nimmt den Gesprächsverlauf entgegen und gibt Mirus Antwort zurück.
    Der System-Prompt wird serverseitig aufgebaut – kein API-Key im Browser.
    """
    try:
        system_prompt = build_system_prompt(req.history) if req.history else SYSTEM_PROMPT_BASE
        messages = [{"role": "system", "content": system_prompt}]
        messages += [{"role": m.role, "content": m.content} for m in req.messages]
        response = _call_llm(messages, max_tokens=300, temperature=0.7)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/health")
def health():
    return {"status": "ok", "mode": os.getenv("MODE", "groq")}


# ─── Frontend ausliefern ──────────────────────────────────────────────────────

WEB_DIR = Path(__file__).parent / "web"

if WEB_DIR.exists():
    @app.get("/")
    def index():
        return FileResponse(WEB_DIR / "index.html")

    app.mount("/", StaticFiles(directory=WEB_DIR), name="web")


# ─── Start ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    print("\n🌿 Miru Mind Web\n")
    print("   http://localhost:8000\n")
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)