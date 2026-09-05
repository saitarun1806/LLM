"""
rag_pipeline.py

Retrieval (Chroma) + generation for skill-gap analysis. This is the core
logic both the CLI and the FastAPI app (app.py) call into.

Generation backend: Groq's hosted API (needs GROQ_API_KEY env var).

Retrieval backend: an EXISTING Chroma store (CHROMA_DIR/COLLECTION_NAME
below) that's already populated from prior Kaggle + live-Naukri scraping
runs, with location/experience filtering already tuned against that data.
This file does not re-ingest or reshape that store — it only reads from
it — so the retrieval/filter logic (retrieve_jobs, EXPERIENCE_FIELD,
FRESHER_EXPERIENCE_VALUES) is unchanged from the working version.

What's new in this version, layered on top of retrieval:
  - Student profile now includes education and career_interests, not just
    current_skills/target_role/location/work_experience.
  - skill_gap_analysis flags missing skills PER POSTING (not just in
    aggregate), so the caller can see which specific job wants what.
  - AI-suggested courses to close each missing skill.
  - A personalized, AI-generated step-by-step plan toward the target role.
  - ensure_chroma_store(): if CHROMA_DIR isn't present locally (e.g. a
    fresh clone/environment), this downloads it from the shared Google
    Drive folder via `gdown` before the Chroma client tries to open it,
    instead of failing with "collection not found". No-op if the store is
    already on disk.

NOT RUN IN THIS SANDBOX (depends on the existing Chroma store + a Groq API
key). Written against the real schema from 01/02 and the documented Groq
HTTP API — run in your own environment.

Setup:
    1. Get a free API key: https://console.groq.com
    2. export GROQ_API_KEY=your_key_here
    3. (only if chroma_store/ isn't already present locally)
       pip install gdown

Run (CLI test):
    python rag_pipeline.py
"""

import json
import os
import time
from typing import Optional

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

# Shared Google Drive folder containing a pre-built chroma_store, for
# environments that don't already have it on disk (fresh clone, new
# machine, CI, etc). Only used as a fallback — see ensure_chroma_store().
CHROMA_STORE_DRIVE_FOLDER_URL = (
    "https://drive.google.com/drive/folders/1EOJHVBLEcBZH3JLBoOd7D2WNnrJ9Cwtg?usp=drive_link"
)

# ---------- Experience metadata ----------
# Unchanged from the working retrieval setup — these already match the
# metadata schema of the existing Chroma store. EDIT THESE TWO only if you
# repoint this at a differently-shaped store. Run this to check what's
# actually stored:
#
#   python -c "
#   import chromadb
#   c = chromadb.PersistentClient(path='./chroma_store')
#   col = c.get_collection('india_jobs')
#   print(col.get(limit=3, include=['metadatas'])['metadatas'])
#   "
EXPERIENCE_FIELD = "experience"          # metadata key holding the range string
FRESHER_EXPERIENCE_VALUES = ["0-1", "0-2"]  # values that count as entry-level

# ---------- LLM backend config ----------
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")  # good general-purpose instruct model


def ensure_chroma_store(
    chroma_dir: str = CHROMA_DIR,
    drive_folder_url: str = CHROMA_STORE_DRIVE_FOLDER_URL,
) -> None:
    """Make sure `chroma_dir` exists and is populated before anything tries
    to open it as a PersistentClient.

    If the directory is missing or empty, download it from the shared
    Google Drive folder using `gdown` (the standard tool for scripted
    Drive downloads — a plain `requests.get` on a Drive share link returns
    an HTML interstitial page, not the file/folder contents).

    Safe to call unconditionally at import/startup time: if the store is
    already present (the common case after the first run, or in an
    environment that already has it baked in), this is a no-op and does
    not touch the network.
    """
    if os.path.isdir(chroma_dir) and os.listdir(chroma_dir):
        return

    try:
        import gdown
    except ImportError as e:
        raise RuntimeError(
            f"'{chroma_dir}' was not found locally and the 'gdown' package "
            "isn't installed, so it can't be auto-downloaded.\n"
            "Fix with one of:\n"
            "  1. pip install gdown   (then just re-run this script)\n"
            "  2. Manually download the folder from:\n"
            f"     {drive_folder_url}\n"
            f"     and place its contents at: {os.path.abspath(chroma_dir)}"
        ) from e

    print(f"[rag_pipeline] '{chroma_dir}' not found locally — downloading from Google Drive...")
    os.makedirs(chroma_dir, exist_ok=True)
    try:
        gdown.download_folder(
            url=drive_folder_url,
            output=chroma_dir,
            quiet=False,
            use_cookies=False,
        )
    except Exception as e:
        raise RuntimeError(
            "Automatic download of chroma_store from Google Drive failed "
            f"({e}). This is usually a Drive rate limit or permissions "
            "issue on large/shared folders. Download it manually from:\n"
            f"  {drive_folder_url}\n"
            f"and place its contents at: {os.path.abspath(chroma_dir)}"
        ) from e

    if not os.listdir(chroma_dir):
        raise RuntimeError(
            f"Download from Google Drive completed but '{chroma_dir}' is "
            "still empty — check the folder link is shared as "
            "'Anyone with the link' and try again, or download it manually."
        )

    print(f"[rag_pipeline] chroma_store downloaded to '{chroma_dir}'.")


# Run the check once at import time, before the Chroma client below tries
# to open CHROMA_DIR. Both the CLI (`python rag_pipeline.py`) and app.py
# (`import rag_pipeline as rag`) go through this same import, so neither
# needs to remember to call it separately.
ensure_chroma_store()

# anonymized_telemetry=False silences a known, harmless ChromaDB/posthog
# version-mismatch bug ("capture() takes 1 positional argument but 3 were
# given") that otherwise logs a failed-telemetry line on every startup.
# It has no effect on retrieval/embeddings — purely cosmetic in the logs.
_client = chromadb.PersistentClient(
    path=CHROMA_DIR,
    settings=chromadb.Settings(anonymized_telemetry=False),
)
_embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)
_collection = _client.get_collection(name=COLLECTION_NAME, embedding_function=_embed_fn)


def retrieve_jobs(
    current_skills: list[str],
    target_role: str,
    location: str = None,
    k: int = 12,
    work_experience: Optional[int] = None,
) -> list[dict]:
    """Semantic search over the existing Chroma store, with progressive
    filter relaxation so a strict-but-empty filter never means "nothing to
    show the student."

    Freshers (work_experience == 0) are matched by experience level ONLY —
    current_skills is deliberately left out of both the query text and any
    matching logic here. A fresher's listed skills (coursework, personal
    projects) aren't a reliable signal for which postings suit them; what
    matters is that the posting itself is tagged entry-level. Skills are
    still shown to the LLM later (in build_skill_gap_prompt) for the
    "missing skills" / "recommendation" writeup — they're just not used to
    steer retrieval for freshers.

    Non-freshers keep the original behavior: skills + role feed the
    semantic query, no experience filter is applied.

    Filter relaxation, in order, stopping as soon as a step returns hits:
      1. location filter + fresher-experience filter (both, if applicable)
      2. fresher-experience filter only (drop location) — location is an
         EXACT string match against however the scraper wrote it
         ("Bengaluru" vs "Bangalore" vs "Hybrid - Bengaluru"), so a miss
         here usually means "no exact string match," not "no jobs there."
      3. no filters at all — plain semantic search on role (+ skills for
         non-freshers). A fresher-tagged posting is a nice-to-have, not
         worth returning zero results over.
    A caller only ever sees an empty list if step 3 also comes back empty,
    i.e. nothing in the whole store is semantically close to this role.
    """
    is_fresher = work_experience == 0

    if is_fresher:
        query_text = f"Target role: {target_role}. Entry-level / fresher position. Location: {location or 'any'}."
    else:
        query_text = (
            f"Target role: {target_role}. "
            f"Current skills: {', '.join(current_skills)}. "
            f"Location: {location or 'any'}."
        )

    def run_query(where):
        results = _collection.query(
            query_texts=[query_text],
            n_results=k,
            where=where,
        )
        hits = []
        for doc, meta, dist in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
            hits.append({"text": doc, "metadata": meta, "distance": dist})
        return hits

    fresher_filter = {EXPERIENCE_FIELD: {"$in": FRESHER_EXPERIENCE_VALUES}} if is_fresher else None
    location_filter = {"location": {"$eq": location}} if location else None

    # Step 1: every applicable filter.
    filters = [f for f in (location_filter, fresher_filter) if f]
    where = None if not filters else (filters[0] if len(filters) == 1 else {"$and": filters})
    hits = run_query(where)
    if hits:
        return hits

    # Step 2: drop location, keep fresher filter if applicable.
    if location_filter:
        hits = run_query(fresher_filter)
        if hits:
            return hits

    # Step 3: no filters — last resort before giving up.
    if fresher_filter or location_filter:
        hits = run_query(None)

    return hits


def call_groq(
    prompt: str,
    model: str = GROQ_MODEL,
    max_retries: int = 3,
    max_tokens: int = 4096,
    json_mode: bool = False,
) -> str:
    """Call Groq's OpenAI-compatible chat completions endpoint, with basic
    retry/backoff on 429 (rate limit) responses — Groq's free tier has
    fairly tight per-minute limits, so bursts of requests will hit this.

    max_tokens defaults to 4096 rather than Groq's own default cap — the
    skill-gap prompt asks for a job_matches entry per posting plus
    courses/roadmap, which was previously getting cut off mid-JSON.

    json_mode requests Groq's OpenAI-compatible JSON mode
    (response_format={"type": "json_object"}), which forces a
    syntactically valid JSON object back instead of relying on the model
    to follow "respond with only JSON" in the prompt text. Only pass
    json_mode=True for prompts that actually expect JSON — chat replies
    are plain text and must NOT set this.
    """
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
        "max_tokens": max_tokens,
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}

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


def call_llm(prompt: str, max_tokens: int = 4096, json_mode: bool = False) -> str:
    """Generation backend is Groq only."""
    return call_groq(prompt, max_tokens=max_tokens, json_mode=json_mode)


def _experience_text(work_experience: Optional[int]) -> str:
    """Render years-of-experience for the prompt. None = not disclosed;
    0 is deliberately distinguished as fresher/entry-level, and is NOT
    inferred from an empty current_skills list — someone can have zero
    work experience and still list skills from coursework/projects."""
    if work_experience is None:
        return "not specified"
    if work_experience == 0:
        return "0 years — fresher / entry-level candidate"
    return f"{work_experience} years"


def build_skill_gap_prompt(
    current_skills: list[str],
    target_role: str,
    location: str,
    hits: list[dict],
    work_experience: Optional[int] = None,
    education: Optional[str] = None,
    career_interests: Optional[list[str]] = None,
) -> str:
    # Number each posting so the model can reference them by index in
    # per-job breakdowns (job_matches) without repeating full text back.
    context_block = "\n\n".join(
        f"[Posting {i+1}] {h['text']}" for i, h in enumerate(hits)
    )
    skills_text = ", ".join(current_skills) if current_skills else "none listed"
    education_text = education if education else "not specified"
    interests_text = ", ".join(career_interests) if career_interests else "not specified"

    # Two different framings for the personalized_plan/roadmap step below,
    # depending on whether the student has any skills on file yet. Neither
    # path blocks analysis — a student with zero skills still gets a full
    # result, just angled at "how do I start" instead of "here's my gap".
    if not current_skills:
        skills_guidance = (
            'The user has NOT listed any current skills yet. Do not treat this '
            'as missing data to apologize for - treat it as a brand-new '
            'candidate. "matched_skills" should simply be empty, and '
            '"missing_skills" / each job_matches entry\'s "missing_skills" '
            'should list what these postings actually want. Lean '
            '"personalized_plan" into a foundational BEST PATH for someone '
            'starting from zero: which 2-3 skills to learn FIRST, in order, to '
            'become minimally competitive for this target role - not an '
            'exhaustive list of every skill every posting mentions.'
        )
    else:
        skills_guidance = (
            "The user has listed current skills above - ground "
            '"matched_skills" / "missing_skills" precisely in what they '
            'already have vs. what these postings additionally want, and make '
            '"personalized_plan" the BEST PATH from their current skill set to '
            'the target role (build on what they already have, don\'t restart '
            'them from zero).'
        )

    return f"""You are a career advisor analyzing the current Indian job market.

USER PROFILE
- Current skills: {skills_text}
- Education: {education_text}
- Career interests: {interests_text}
- Work experience: {_experience_text(work_experience)}
- Target role: {target_role}
- Preferred location: {location or 'Any'}

RELEVANT LIVE JOB POSTINGS (retrieved from a vector database of real postings,
numbered [Posting N] for reference below):
{context_block}

TASK
{skills_guidance}

Based ONLY on the postings above, produce:

1. "matched_skills": skills the user already has that these postings want.

2. "missing_skills": specific skills/technologies that appear repeatedly across
   ALL the postings but are NOT in the user's current skill list — ranked by
   how often they appear.

3. "emerging_signals": any skill, tool, or technology in the postings that looks
   like a newer/rising requirement (not something you already knew was standard
   for this role — base this only on what's actually in the postings).

4. "job_matches": a per-posting breakdown, ONE entry per posting above, each with:
   - "posting_number": the [Posting N] index (integer)
   - "title" and "company" (copy from the posting)
   - "matched_skills": skills from the user's list that THIS specific posting wants
   - "missing_skills": skills THIS specific posting wants that the user does NOT
     have — this is the per-job flag, distinct from the aggregate list in (2)
   - "fit_summary": one sentence on how good a fit this posting is right now

5. "suggested_courses": for each of the top 3-5 items in "missing_skills" (2),
   suggest one concrete course or learning resource to close that specific gap.
   Each entry: "skill", "course_title", "platform" (e.g. Coursera, YouTube,
   NPTEL, freeCodeCamp — use well-known, plausible platforms, don't invent
   obscure ones), and "level" ("beginner"/"intermediate"/"advanced").

6. "personalized_plan": a "recommendation" of 2-3 sentences (concrete,
   prioritized next steps — if the user is a fresher, prioritize foundational
   skills and how to break in even if they already have some listed skills)
   PLUS a "roadmap": an ordered list of 3-5 short phases (e.g. "Weeks 1-2: ...",
   "Month 2: ...") that ties together the missing skills, suggested courses,
   and the user's career interests into a concrete path toward the target role.

Respond ONLY with valid JSON in this shape:
{{
  "matched_skills": [...],
  "missing_skills": [...],
  "emerging_signals": [...],
  "job_matches": [
    {{
      "posting_number": 1,
      "title": "...",
      "company": "...",
      "matched_skills": [...],
      "missing_skills": [...],
      "fit_summary": "..."
    }}
  ],
  "suggested_courses": [
    {{"skill": "...", "course_title": "...", "platform": "...", "level": "..."}}
  ],
  "personalized_plan": {{
    "recommendation": "...",
    "roadmap": ["...", "...", "..."]
  }}
}}
"""


def build_general_advice_prompt(
    current_skills: list[str],
    target_role: str,
    location: str,
    work_experience: Optional[int] = None,
    education: Optional[str] = None,
    career_interests: Optional[list[str]] = None,
) -> str:
    """Same output shape as build_skill_gap_prompt, used when retrieve_jobs
    (even after relaxing its filters) found nothing in the store close to
    this role — e.g. a niche or newly-named title. Rather than dead-end the
    student with an error, ask the model to reason from its own general
    knowledge of the Indian job market instead of specific postings.
    job_matches is necessarily empty here (there's nothing to cite), but
    missing_skills / suggested_courses / personalized_plan are exactly as
    useful without them — that's the actual point of this screen."""
    skills_text = ", ".join(current_skills) if current_skills else "none listed"
    education_text = education if education else "not specified"
    interests_text = ", ".join(career_interests) if career_interests else "not specified"
    skills_guidance = (
        "The user has NOT listed any current skills yet — treat them as a "
        'brand-new candidate. "matched_skills" should be empty; focus '
        '"missing_skills" and "personalized_plan" on the 2-3 foundational '
        "skills to learn FIRST, in order, to become minimally competitive."
        if not current_skills
        else
        "The user has listed current skills above — ground "
        '"matched_skills" / "missing_skills" in what they already have vs. '
        "what this role typically needs, and make \"personalized_plan\" the "
        "best path from their current skills to the target role."
    )

    return f"""You are a career advisor for the current Indian job market.

No live job postings in our database currently matched the role below
closely enough to cite (this is a gap in this specific dataset right now,
not a claim that the role doesn't exist). Answer using your own general,
up-to-date knowledge of what Indian employers typically expect for this
role instead.

USER PROFILE
- Current skills: {skills_text}
- Education: {education_text}
- Career interests: {interests_text}
- Work experience: {_experience_text(work_experience)}
- Target role: {target_role}
- Preferred location: {location or 'Any'}

TASK
{skills_guidance}

Produce the same fields a postings-grounded analysis would, based on general
knowledge instead of specific listings:

1. "matched_skills": skills the user already has that are typically relevant
   to this role.
2. "missing_skills": the specific skills/technologies most commonly required
   for this role in the current Indian market, ranked by importance, that
   the user does NOT have.
3. "emerging_signals": 1-3 genuinely current trends or rising requirements
   for this role (only include something here if you're confident it's a
   real, current trend — an empty list is better than a guess).
4. "job_matches": always an empty array — there are no specific postings to
   cite, so do not invent any.
5. "suggested_courses": for each of the top 3-5 items in "missing_skills",
   one concrete course or learning resource ("skill", "course_title",
   "platform" — e.g. Coursera, YouTube, NPTEL, freeCodeCamp — and "level":
   "beginner"/"intermediate"/"advanced").
6. "personalized_plan": a "recommendation" of 2-3 sentences plus a
   "roadmap" of 3-5 ordered short phases toward the target role.

Respond ONLY with valid JSON in this exact shape:
{{
  "matched_skills": [...],
  "missing_skills": [...],
  "emerging_signals": [...],
  "job_matches": [],
  "suggested_courses": [
    {{"skill": "...", "course_title": "...", "platform": "...", "level": "..."}}
  ],
  "personalized_plan": {{
    "recommendation": "...",
    "roadmap": ["...", "...", "..."]
  }}
}}
"""


def _as_str_list(value) -> list:
    """Coerce a field the model may have returned as a string, null, or
    nested structure into a flat list of strings — the frontend always
    expects an array here and .map()s over it directly."""
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    if isinstance(value, str):
        # Model sometimes returns a comma-separated string instead of a
        # JSON array despite the schema — split it rather than treating
        # the whole sentence as one "skill".
        return [v.strip() for v in value.split(",") if v.strip()]
    return [str(value)]


def _normalize_job_match(entry: dict, index: int, hits: list) -> dict:
    fallback_meta = hits[index]["metadata"] if index < len(hits) else {}
    return {
        "posting_number": entry.get("posting_number", index + 1),
        "title": entry.get("title") or fallback_meta.get("title", "Unknown"),
        "company": entry.get("company") or fallback_meta.get("company", "Unknown"),
        "matched_skills": _as_str_list(entry.get("matched_skills")),
        "missing_skills": _as_str_list(entry.get("missing_skills")),
        "fit_summary": str(entry.get("fit_summary") or "").strip(),
    }


def _normalize_course(entry: dict) -> dict:
    return {
        "skill": str(entry.get("skill", "")).strip(),
        "course_title": str(entry.get("course_title", "")).strip(),
        "platform": str(entry.get("platform", "")).strip(),
        "level": entry.get("level") if entry.get("level") in ("beginner", "intermediate", "advanced") else "beginner",
    }


def normalize_skill_gap_result(parsed: dict, hits: list) -> dict:
    """Guarantee the exact response shape the frontend expects, regardless
    of how the model actually structured (or mis-structured) its answer.
    Missing keys get sensible empty defaults instead of the frontend
    hitting undefined; wrong types (a string where a list was expected,
    etc.) get coerced rather than left as-is. This runs on every
    successful JSON parse, not just the fallback path, since inconsistent
    field types are just as likely as a completely missing key."""
    job_matches_raw = parsed.get("job_matches")
    job_matches = (
        [_normalize_job_match(jm, i, hits) for i, jm in enumerate(job_matches_raw)]
        if isinstance(job_matches_raw, list)
        else []
    )

    courses_raw = parsed.get("suggested_courses")
    suggested_courses = (
        [_normalize_course(c) for c in courses_raw if isinstance(c, dict)]
        if isinstance(courses_raw, list)
        else []
    )

    plan_raw = parsed.get("personalized_plan") if isinstance(parsed.get("personalized_plan"), dict) else {}

    return {
        "matched_skills": _as_str_list(parsed.get("matched_skills")),
        "missing_skills": _as_str_list(parsed.get("missing_skills")),
        "emerging_signals": _as_str_list(parsed.get("emerging_signals")),
        "job_matches": job_matches,
        "suggested_courses": suggested_courses,
        "personalized_plan": {
            "recommendation": str(plan_raw.get("recommendation", "")).strip(),
            "roadmap": _as_str_list(plan_raw.get("roadmap")),
        },
    }


def skill_gap_analysis(
    current_skills: list[str],
    target_role: str,
    location: str = None,
    k: int = 12,
    work_experience: Optional[int] = None,
    education: Optional[str] = None,
    career_interests: Optional[list[str]] = None,
) -> dict:
    hits = retrieve_jobs(current_skills, target_role, location, k=k, work_experience=work_experience)

    # retrieve_jobs already relaxes location/experience filters internally
    # before giving up, so an empty result here means nothing in the whole
    # store is even semantically close to this role — not just a filter
    # mismatch. Rather than error out, fall back to a general-knowledge
    # answer so the student still gets a real "what to learn" recommendation.
    used_live_postings = bool(hits)
    if used_live_postings:
        prompt = build_skill_gap_prompt(
            current_skills,
            target_role,
            location,
            hits,
            work_experience=work_experience,
            education=education,
            career_interests=career_interests,
        )
    else:
        prompt = build_general_advice_prompt(
            current_skills,
            target_role,
            location,
            work_experience=work_experience,
            education=education,
            career_interests=career_interests,
        )

    raw = call_llm(prompt, max_tokens=4096, json_mode=True)

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
            parsed = {}

    result = normalize_skill_gap_result(parsed, hits)
    result["_retrieved_postings"] = [h["metadata"] for h in hits]
    result["used_live_postings"] = used_live_postings
    return result


if __name__ == "__main__":
    result = skill_gap_analysis(
        current_skills=["Python", "SQL", "Excel"],  # ignored for retrieval since work_experience=0
        target_role="Data Analyst",
        location="Bengaluru",
        work_experience=0,
        education="B.Sc. Computer Science, 2026",
        career_interests=["Data Analytics", "Business Intelligence"],
    )
    print(json.dumps(result, indent=2))
