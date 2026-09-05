"""
rag_pipeline.py

Retrieval (Chroma) + generation for skill-gap analysis. This is the core
logic both the CLI and the FastAPI app (app.py) call into.

Generation backend: Groq's hosted API (needs GROQ_API_KEY env var).

NOT RUN IN THIS SANDBOX (depends on 02's Chroma store + a Groq API key).
Written against the real schema from 01/02 and the documented Groq HTTP
API — run in your own environment.

Chroma store bootstrap:
    On import, if ./chroma_store doesn't exist locally (or is empty), this
    module downloads it automatically from the Google Drive folder below
    using `gdown`. This only works if the Drive folder is shared as
    "Anyone with the link" — gdown can't authenticate as you, so a
    private/restricted folder will fail to download. If you'd rather manage
    the store yourself, set SKIP_CHROMA_DOWNLOAD=1 and place the three
    Chroma files under ./chroma_store manually.

    Requires: pip install gdown

Setup (Groq):
    1. Get a free API key: https://console.groq.com
    2. export GROQ_API_KEY=your_key_here

Run (CLI test):
    python rag_pipeline.py

Install (add to your existing environment):
    pip install gdown
"""

import json
import os
import time

import requests
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

# Reads a .env file (if present) in the current working directory and loads
# its key=value pairs into os.environ — lets you keep GROQ_API_KEY etc. out
# of your shell history / process list. No-op if no .env file exists.
load_dotenv()

CHROMA_DIR = "./chroma_store"
COLLECTION_NAME = "india_jobs"
EMBED_MODEL = "all-MiniLM-L6-v2"

# ---------- Chroma store bootstrap (auto-download from Google Drive) ----------

# Folder ID parsed out of:
# https://drive.google.com/drive/folders/1EOJHVBLEcBZH3JLBoOd7D2WNnrJ9Cwtg
CHROMA_STORE_DRIVE_FOLDER_ID = os.environ.get(
    "CHROMA_STORE_DRIVE_FOLDER_ID", "1EOJHVBLEcBZH3JLBoOd7D2WNnrJ9Cwtg"
)


def _ensure_chroma_store(target_dir: str = CHROMA_DIR) -> None:
    """
    Makes sure the Chroma persistence directory exists and is populated
    before anything tries to open it. If it's missing or empty, pulls the
    three store files down from the shared Google Drive folder via `gdown`.

    Set SKIP_CHROMA_DOWNLOAD=1 to disable this and manage the store
    yourself (e.g. you already have it locally, or you're pointing
    CHROMA_DIR somewhere with its own setup step).
    """
    if os.environ.get("SKIP_CHROMA_DOWNLOAD") == "1":
        return

    already_populated = os.path.isdir(target_dir) and len(os.listdir(target_dir)) > 0
    if already_populated:
        return

    try:
        import gdown
    except ImportError as exc:
        raise RuntimeError(
            "chroma_store is missing locally and `gdown` isn't installed to "
            "fetch it automatically. Run `pip install gdown`, or set "
            "SKIP_CHROMA_DOWNLOAD=1 and place the Chroma files under "
            f"{target_dir} yourself."
        ) from exc

    os.makedirs(target_dir, exist_ok=True)
    print(f"[rag_pipeline] {target_dir} not found locally — downloading from Google Drive...")

    gdown.download_folder(
        id=CHROMA_STORE_DRIVE_FOLDER_ID,
        output=target_dir,
        quiet=False,
        use_cookies=False,
    )

    # gdown.download_folder() recreates the source folder's name as a
    # subdirectory inside `output` (e.g. ./chroma_store/chroma_store/...).
    # Flatten that one level so CHROMA_DIR itself is the persistence dir
    # chromadb.PersistentClient expects.
    nested = os.path.join(target_dir, "chroma_store")
    if os.path.isdir(nested):
        for name in os.listdir(nested):
            os.replace(os.path.join(nested, name), os.path.join(target_dir, name))
        os.rmdir(nested)

    if not os.listdir(target_dir):
        raise RuntimeError(
            f"Download from Google Drive folder {CHROMA_STORE_DRIVE_FOLDER_ID} "
            f"completed but {target_dir} is still empty — check that the folder "
            "is shared as 'Anyone with the link' and contains the Chroma files."
        )

    print(f"[rag_pipeline] chroma_store downloaded and placed at {target_dir}")


_ensure_chroma_store()

# ---------- LLM backend config (Groq only) ----------
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")  # good general-purpose instruct model

_client = chromadb.PersistentClient(path=CHROMA_DIR)
_embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)
_collection = _client.get_collection(name=COLLECTION_NAME, embedding_function=_embed_fn)


def retrieve_jobs(current_skills: list[str], target_role: str, location: str = None, k: int = 12) -> list[dict]:
    """Semantic search over Chroma, optionally pre-filtered by location."""
    query_text = (
        f"Target role: {target_role}. "
        f"Current skills: {', '.join(current_skills)}. "
        f"Location: {location or 'any'}."
    )

    where = None
    if location:
        # Chroma metadata filter is an exact match on the stored 'location' field;
        # for fuzzier matching, retrieve more (k*3) and re-filter with substring
        # matching in Python instead — real-world location strings are messy
        # ("Bengaluru" vs "Bangalore" vs "Hybrid - Bengaluru").
        where = {"location": {"$eq": location}}

    results = _collection.query(
        query_texts=[query_text],
        n_results=k,
        where=where,
    )

    hits = []
    for doc, meta, dist in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
        hits.append({"text": doc, "metadata": meta, "distance": dist})
    return hits


def call_groq(prompt: str, model: str = GROQ_MODEL, max_retries: int = 3) -> str:
    """Call Groq's OpenAI-compatible chat completions endpoint, with basic
    retry/backoff on 429 (rate limit) responses — Groq's free tier has
    fairly tight per-minute limits, so bursts of requests will hit this."""
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY environment variable is not set")

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
    }

    for attempt in range(max_retries):
        resp = requests.post(GROQ_URL, headers=headers, json=payload, timeout=120)

        if resp.status_code == 429:
            # Respect Retry-After if Groq sends one, otherwise back off exponentially
            wait = float(resp.headers.get("retry-after", 2 ** attempt))
            time.sleep(wait)
            continue

        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    raise RuntimeError(f"Groq rate limit exceeded after {max_retries} retries")


def call_llm(prompt: str) -> str:
    return call_groq(prompt)


def build_skill_gap_prompt(current_skills: list[str], target_role: str, location: str, hits: list[dict]) -> str:
    context_block = "\n\n".join(
        f"[Posting {i+1}] {h['text']}" for i, h in enumerate(hits)
    )
    return f"""You are a career advisor analyzing the current Indian job market.

USER PROFILE
- Current skills: {', '.join(current_skills)}
- Target role: {target_role}
- Preferred location: {location or 'Any'}

RELEVANT LIVE JOB POSTINGS (retrieved from a vector database of real postings):
{context_block}

TASK
Based ONLY on the postings above, produce:
1. "matched_skills": skills the user already has that these postings want.
2. "missing_skills": specific skills/technologies that appear repeatedly in these
   postings but are NOT in the user's current skill list — ranked by how often
   they appear.
3. "emerging_signals": any skill, tool, or technology in the postings that looks
   like a newer/rising requirement (not something you already knew was standard
   for this role — base this only on what's actually in the postings).
4. "recommendation": 2-3 sentences of concrete, prioritized next steps.

Respond ONLY with valid JSON in this shape:
{{
  "matched_skills": [...],
  "missing_skills": [...],
  "emerging_signals": [...],
  "recommendation": "..."
}}
"""


def skill_gap_analysis(current_skills: list[str], target_role: str, location: str = None, k: int = 12) -> dict:
    hits = retrieve_jobs(current_skills, target_role, location, k=k)
    if not hits:
        return {"error": "No matching postings found — try a broader role or location."}

    prompt = build_skill_gap_prompt(current_skills, target_role, location, hits)
    raw = call_llm(prompt)

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        # Models sometimes wrap JSON in prose or code fences despite
        # instructions — try stripping fences before giving up.
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            cleaned = cleaned.split("\n", 1)[-1] if cleaned.lower().startswith("json") else cleaned
        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError:
            parsed = {"raw_response": raw, "note": "Model did not return valid JSON"}

    parsed["_retrieved_postings"] = [h["metadata"] for h in hits]
    return parsed


if __name__ == "__main__":
    result = skill_gap_analysis(
        current_skills=["Python", "SQL", "Excel"],
        target_role="Data Analyst",
        location="Bengaluru",
    )
    print(json.dumps(result, indent=2))