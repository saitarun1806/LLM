"""
app.py

FastAPI service wrapping the RAG pipeline:
  POST /skill-gap        - one-shot skill-gap analysis (structured JSON)
  POST /chat              - conversational endpoint, keeps history per session

NOT RUN IN THIS SANDBOX (depends on rag_pipeline.py's stack + a Groq API
key — see rag_pipeline.py's GROQ_API_KEY env var).

On startup, importing rag_pipeline triggers an automatic download of
./chroma_store from Google Drive if it isn't already present locally
(see rag_pipeline.py's _ensure_chroma_store / SKIP_CHROMA_DOWNLOAD).

Install:
    pip install fastapi uvicorn gdown

Run:
    uvicorn app:app --reload --port 8000

Docs (auto-generated):
    http://localhost:8000/docs
"""

import uuid
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Fixed: rag_pipeline.py has no leading digit, so it's a normal importable
# module name — no need for importlib.import_module gymnastics.
import rag_pipeline as rag

app = FastAPI(title="India Job Market Skill-Gap API")

# In-memory session store for the chat endpoint. Fine for a demo/single-instance
# deployment; swap for Redis or a DB before running multiple app instances.
_sessions: dict[str, list[dict]] = {}


# ---------- /skill-gap ----------

class SkillGapRequest(BaseModel):
    current_skills: list[str]
    target_role: str
    location: Optional[str] = None
    top_k: int = 12


@app.post("/skill-gap")
def skill_gap(req: SkillGapRequest):
    result = rag.skill_gap_analysis(
        current_skills=req.current_skills,
        target_role=req.target_role,
        location=req.location,
        k=req.top_k,
    )
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return result


# ---------- /chat ----------

class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    message: str
    current_skills: Optional[list[str]] = None
    target_role: Optional[str] = None
    location: Optional[str] = None


@app.post("/chat")
def chat(req: ChatRequest):
    session_id = req.session_id or str(uuid.uuid4())
    history = _sessions.setdefault(session_id, [])
    history.append({"role": "user", "content": req.message})

    # If the user has given us enough profile info, ground the reply in
    # real retrieved postings; otherwise just have the model ask for it.
    if req.current_skills and req.target_role:
        hits = rag.retrieve_jobs(req.current_skills, req.target_role, req.location, k=8)
        context = "\n\n".join(h["text"] for h in hits)
        prompt = (
            f"Conversation so far:\n"
            + "\n".join(f"{m['role']}: {m['content']}" for m in history)
            + f"\n\nRelevant live job postings:\n{context}\n\n"
            "Reply to the user's last message, grounding any job-market claims "
            "in the postings above. Be conversational, concise."
        )
    else:
        prompt = (
            "You are a job-market assistant. The user hasn't given their current "
            "skills and target role yet — ask for them before giving specific advice.\n\n"
            + "\n".join(f"{m['role']}: {m['content']}" for m in history)
        )

    reply = rag.call_llm(prompt)
    history.append({"role": "assistant", "content": reply})
    return {"session_id": session_id, "reply": reply}


@app.get("/health")
def health():
    return {"status": "ok"}