"""
Gspänli – Web Server
=====================
Start:  python server.py  →  http://localhost:8000
"""

import datetime
import os
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

from main import (
    _call_llm,
    build_system_prompt,
    summarize_session,
    update_user_profile,
    load_history,
    save_history,
    save_checkin,
)

app = FastAPI(title="Gspänli API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ─── Modelle ──────────────────────────────────────────────────────────────────

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: list[Message]
    history: dict = {}

class FinalizeRequest(BaseModel):
    session_messages: list[Message]

class CheckinRequest(BaseModel):
    mood: int
    sleep: str
    exercise: bool
    note: str = ""


# ─── Chat ─────────────────────────────────────────────────────────────────────

@app.post("/api/chat")
def chat(req: ChatRequest):
    try:
        system_prompt = build_system_prompt(req.history if req.history else {})
        messages = [{"role": "system", "content": system_prompt}]
        messages += [{"role": m.role, "content": m.content} for m in req.messages]
        response = _call_llm(messages, max_tokens=300, temperature=0.7)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─── History ──────────────────────────────────────────────────────────────────

@app.get("/api/history")
def get_history():
    return load_history()


@app.post("/api/finalize")
def finalize(req: FinalizeRequest):
    if not req.session_messages:
        return {"status": "skipped"}

    data = load_history()
    session_msgs = [{"role": m.role, "content": m.content} for m in req.session_messages]
    summary = summarize_session(session_msgs)

    session_entry: dict = {"date": datetime.datetime.now().isoformat(), "messages": session_msgs}
    if summary:
        session_entry["summary"]   = summary.get("summary", "")
        session_entry["themes"]    = summary.get("themes", [])
        session_entry["key_facts"] = summary.get("key_facts", [])
        if summary.get("mood_observed") is not None:
            session_entry["mood_observed"] = summary["mood_observed"]

        signals = summary.get("lifestyle_signals", {})
        if signals.get("sleep") or signals.get("exercise") is not None:
            data.setdefault("checkins", []).append({
                "date": datetime.datetime.now().isoformat(),
                "mood": summary.get("mood_observed"),
                "sleep": signals.get("sleep"),
                "exercise": signals.get("exercise"),
                "source": "conversation"
            })

        data["user_profile"] = update_user_profile(data, summary)

    data.setdefault("sessions", []).append(session_entry)
    save_history(data)
    return {"status": "ok", "summary": summary}


# ─── Check-in ─────────────────────────────────────────────────────────────────

@app.post("/api/checkin")
def checkin(req: CheckinRequest):
    if not 1 <= req.mood <= 10:
        raise HTTPException(status_code=422, detail="mood 1–10")
    if req.sleep not in ("gut", "mittel", "schlecht"):
        raise HTTPException(status_code=422, detail="sleep: gut|mittel|schlecht")
    data = load_history()
    save_checkin(data, req.mood, req.sleep, req.exercise, req.note)
    return {"status": "ok"}


@app.get("/api/health")
def health():
    return {"status": "ok", "mode": os.getenv("MODE", "groq")}


# ─── Frontend ─────────────────────────────────────────────────────────────────

WEB_DIR = Path(__file__).parent / "web"
if WEB_DIR.exists():
    @app.get("/")
    def index():
        return FileResponse(WEB_DIR / "index.html")
    app.mount("/", StaticFiles(directory=WEB_DIR), name="web")


if __name__ == "__main__":
    import uvicorn
    print("\n🌿 Gspänli  →  http://localhost:8000\n")
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)