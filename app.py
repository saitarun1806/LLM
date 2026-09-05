"""
app.py

FastAPI service wrapping the RAG pipeline:
  POST /skill-gap        - one-shot skill-gap analysis (structured JSON)
  POST /chat              - conversational endpoint, keeps history per session

Profile now covers skills, education, and career interests in one place
(not just current_skills/target_role), and /skill-gap's response includes
per-job missing-skill flags, AI-suggested courses, and a personalized plan
— see rag_pipeline.py's build_skill_gap_prompt for the full shape.

/skill-gap's narrative fields (matched_skills, missing_skills,
emerging_signals, suggested_courses, personalized_plan) come from the
pre-built Chroma store via rag_pipeline.skill_gap_analysis — that store is
for ANALYSIS ONLY (it has no apply_link in its metadata). Anything the
student can actually click "Apply" on (job_matches, _retrieved_postings)
is overwritten below with a fresh call to live_jobs.retrieve_live_jobs,
the same live, apply_link-bearing source /chat already uses, with
matched/missing skills computed deterministically in Python — never
LLM-guessed.

NOT RUN IN THIS SANDBOX (depends on rag_pipeline.py's stack + a Groq API
key — see rag_pipeline.py's GROQ_API_KEY env var).

Install:
    pip install fastapi uvicorn

Run:
    uvicorn app:app --reload --port 8000

Docs (auto-generated):
    http://localhost:8000/docs
"""

import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# rag_pipeline.py has no leading digit, so it's a normal importable
# module name — no need for importlib.import_module gymnastics.
import rag_pipeline as rag
# /chat uses this instead of rag.retrieve_jobs: it fetches the live
# GitHub-hosted job postings JSON at request time (short-cached) rather
# than reading the pre-built Chroma store, and computes matched/missing
# skills deterministically per posting — see live_jobs.py for why.
# /skill-gap now also uses this, for the same reason: it's the only
# source with a real, clickable apply_link per posting.
import live_jobs

app = FastAPI(title="India Job Market Skill-Gap API")

# In-memory session store for the chat endpoint. Fine for a demo/single-instance
# deployment; swap for Redis or a DB before running multiple app instances.
# Each session tracks both the message history AND an accumulated profile
# (current_skills/target_role/location/work_experience/education/
# career_interests), so the user only has to give their profile once —
# later messages in the same session don't need to repeat it.
_sessions: dict[str, dict] = {}


# ---------- /skill-gap ----------

class SkillGapRequest(BaseModel):
    current_skills: list[str]
    target_role: str
    location: Optional[str] = None
    top_k: int = 12
    # Years of work experience. 0 = fresher/entry-level. Deliberately
    # separate from current_skills: a fresher can still list real skills
    # (from coursework, personal projects, certifications) — "no work
    # experience yet" and "no skills yet" are different facts. Fresher
    # status should be driven by this field, not by current_skills == [].
    work_experience: Optional[int] = None
    # Highest/most relevant qualification, e.g. "B.Tech CSE, 2026" or
    # "12th Pass". Optional context that helps the LLM tailor course
    # suggestions and the personalized plan — not used for retrieval
    # filtering.
    education: Optional[str] = None
    # Broader interests beyond the single target_role, e.g. ["Data
    # Analytics", "Product Management"] — feeds the personalized plan and
    # course suggestions, doesn't change which postings get retrieved.
    career_interests: Optional[list[str]] = None


def _build_live_job_matches(req: "SkillGapRequest") -> tuple[list[dict], list[dict]]:
    """Fetches real, apply_link-bearing postings for this profile and
    shapes them into the two apply-able fields of the /skill-gap
    response: job_matches (for the LLM-analysis-shaped table) and
    _retrieved_postings (the raw list job_matches' posting_number
    indexes into — same convention build_skill_gap_prompt/_normalize_job_match
    used with Chroma hits, just backed by live data now).

    matched_skills/missing_skills here are the ones live_jobs.py already
    computed deterministically (exact set-difference against the user's
    current_skills) — never recomputed or guessed by the LLM.
    """
    live_hits = live_jobs.retrieve_live_jobs(
        req.current_skills,
        req.target_role,
        req.location,
        k=6,
        work_experience=req.work_experience,
    )

    job_matches = [
        {
            "posting_number": i + 1,
            "title": h["title"],
            "company": h["company"],
            "matched_skills": h["matched_skills"],
            "missing_skills": h["missing_skills"],
            "fit_summary": (
                f"Matches {len(h['matched_skills'])} of "
                f"{len(h['matched_skills']) + len(h['missing_skills'])} listed skills."
                if (h["matched_skills"] or h["missing_skills"])
                else "No specific skills listed on this posting."
            ),
        }
        for i, h in enumerate(live_hits)
    ]
    return job_matches, live_hits


@app.post("/skill-gap")
def skill_gap(req: SkillGapRequest):
    # rag.skill_gap_analysis (Chroma retrieval + one Groq LLM call for the
    # narrative fields) and _build_live_job_matches (a network fetch of the
    # live postings dataset, on a cache miss, plus a relevance-scoring pass
    # over it for the apply-able fields) are completely independent of each
    # other - neither's input depends on the other's output. Running them
    # back-to-back was adding their latencies together, and the LLM call in
    # particular (a 120B model asked for up to 4096 tokens of structured
    # JSON) is already the single slowest thing in this endpoint. Running
    # them concurrently instead brings total wait time down to roughly
    # whichever one is slower, not the sum of both.
    with ThreadPoolExecutor(max_workers=2) as pool:
        analysis_future = pool.submit(
            rag.skill_gap_analysis,
            current_skills=req.current_skills,
            target_role=req.target_role,
            location=req.location,
            k=req.top_k,
            work_experience=req.work_experience,
            education=req.education,
            career_interests=req.career_interests,
        )
        live_future = pool.submit(_build_live_job_matches, req)

        result = analysis_future.result()
        job_matches, live_postings = live_future.result()

    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])

    # Chroma (via rag.skill_gap_analysis above) is for the narrative
    # fields only — matched_skills, missing_skills, emerging_signals,
    # suggested_courses, personalized_plan. It has no apply_link in its
    # metadata, so anything apply-able must come from the live source
    # instead. Overwrite job_matches/_retrieved_postings with that live,
    # deterministic data — this also means posting_number in job_matches
    # now indexes into the live list, not the Chroma hits.
    result["job_matches"] = job_matches
    result["_retrieved_postings"] = live_postings

    return result


# ---------- /chat ----------

class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    message: str
    current_skills: Optional[list[str]] = None
    target_role: Optional[str] = None
    location: Optional[str] = None
    # Years of work experience. 0 = fresher/entry-level. Kept independent of
    # current_skills — see the note on SkillGapRequest.work_experience above.
    work_experience: Optional[int] = None
    # Same role as in SkillGapRequest — optional profile context, merged
    # into the session the same way as the other fields below.
    education: Optional[str] = None
    career_interests: Optional[list[str]] = None


@app.post("/chat")
def chat(req: ChatRequest):
    session_id = req.session_id or str(uuid.uuid4())
    session = _sessions.setdefault(session_id, {"history": [], "profile": {}})
    history = session["history"]
    profile = session["profile"]

    history.append({"role": "user", "content": req.message})

    # Merge any newly provided profile fields into what we already know for
    # this session — only overwrite a field if this message actually sent one.
    # NOTE: use "is not None" rather than truthiness for current_skills, so
    # an explicit empty list ([]) — someone with no skills listed yet — is
    # honored as a real answer instead of being ignored (an empty list is
    # falsy in Python, so `if req.current_skills:` would otherwise treat
    # "I have no skills" the same as "field not provided").
    if req.current_skills is not None:
        profile["current_skills"] = req.current_skills
    if req.target_role:
        profile["target_role"] = req.target_role
    if req.location:
        profile["location"] = req.location
    # Same "is not None" reasoning: an explicit 0 (fresher, zero years of
    # experience) must be honored as a real answer, not treated as unset.
    if req.work_experience is not None:
        profile["work_experience"] = req.work_experience
    if req.education:
        profile["education"] = req.education
    if req.career_interests is not None:
        profile["career_interests"] = req.career_interests

    postings = []

    # Fresher/entry-level status is driven purely by work_experience, not
    # by current_skills being empty — a fresher can still have listed
    # skills (bootcamp, coursework, personal projects). current_skills is
    # still required for retrieval matching for non-freshers, but for
    # freshers retrieve_jobs ignores skills entirely and matches on the
    # experience metadata field instead (see rag_pipeline.retrieve_jobs).
    #
    # education/career_interests are NOT required to unlock retrieval —
    # they're optional personalization context layered onto the prompt
    # once the core profile (skills, role, experience) is known.
    if (
        profile.get("current_skills") is not None
        and profile.get("target_role")
        and profile.get("work_experience") is not None
    ):
        # Live source (not the pre-built Chroma store): fetches the
        # GitHub-hosted postings JSON at request time (short-cached inside
        # live_jobs.py) and computes matched/missing skills per posting in
        # Python — so the "missing skills" facts below are exact, not an
        # LLM guess.
        live_hits = live_jobs.retrieve_live_jobs(
            profile["current_skills"],
            profile["target_role"],
            profile.get("location"),
            k=6,
            work_experience=profile["work_experience"],
        )
        postings = live_hits

        skills_text = ", ".join(profile["current_skills"]) if profile["current_skills"] else "none listed"
        is_fresher = profile["work_experience"] == 0
        experience_text = (
            "0 years — fresher / entry-level candidate"
            if is_fresher
            else f"{profile['work_experience']} years"
        )
        education_text = profile.get("education") or "not specified"
        interests_text = (
            ", ".join(profile["career_interests"]) if profile.get("career_interests") else "not specified"
        )

        if live_hits:
            # Hand the LLM the already-computed matched/missing skills per
            # posting as plain facts, rather than the raw postings, so it
            # states them rather than re-deriving (and potentially getting
            # them wrong).
            postings_block = "\n\n".join(
                f"[{h['title']} at {h['company']}, {h['location'] or 'location N/A'}, "
                f"requires {h['experience_required'] or 'N/A'} experience]\n"
                f"Skills you already have that match: {', '.join(h['matched_skills']) or 'none'}\n"
                f"Skills this job wants that you're missing: {', '.join(h['missing_skills']) or 'none'}\n"
                f"Apply link: {h['apply_link']}"
                for h in live_hits
            )
        else:
            postings_block = "(No closely matching live postings found for this role/location right now.)"

        prompt = (
            f"User profile — current skills: {skills_text}; "
            f"education: {education_text}; "
            f"career interests: {interests_text}; "
            f"work experience: {experience_text}; "
            f"target role: {profile['target_role']}; "
            f"location: {profile.get('location') or 'any'}.\n\n"
            f"Conversation so far:\n"
            + "\n".join(f"{m['role']}: {m['content']}" for m in history)
            + f"\n\nLive job postings, with matched/missing skills already computed "
            f"exactly against the user's profile:\n{postings_block}\n\n"
            "Reply to the user's last message. Ground any job-market claims in "
            "the postings above, and use the ALREADY-COMPUTED matched/missing "
            "skill lists exactly as given — do not recompute or guess them. "
            "Call out the specific missing skills for the most relevant "
            "posting(s) and suggest how to close each gap (a course, project, "
            "or certification). If the user is a fresher (0 years of work "
            "experience), treat them as entry-level and focus advice on "
            "foundational skills and how to break in — even if they already "
            "have some listed skills from coursework or personal projects. "
            "Be conversational, concise, and reply in 3-6 plain sentences. "
            "Do not use markdown formatting — no headers, no bullet points, "
            "no numbered lists, no asterisks for bold or italic. Write it "
            "as plain prose someone would read in a chat bubble, not a "
            "formatted document."
        )
    else:
        missing = []
        if profile.get("current_skills") is None:
            missing.append("current skills (or say you have none yet)")
        if not profile.get("target_role"):
            missing.append("target role")
        if profile.get("work_experience") is None:
            missing.append("years of work experience (0 if you're a fresher)")
        prompt = (
            "You are a job-market assistant. The user hasn't given their "
            f"{' and '.join(missing)} yet — ask for whichever is missing before "
            "giving specific advice. Reply in 1-3 plain sentences, no markdown "
            "formatting (no headers, bullets, or asterisks).\n\n"
            + "\n".join(f"{m['role']}: {m['content']}" for m in history)
        )

    # Chat replies are plain text, not JSON — json_mode stays off here.
    reply = rag.call_llm(prompt, max_tokens=600, json_mode=False)
    history.append({"role": "assistant", "content": reply})

    response = {"session_id": session_id, "reply": reply}
    if postings:
        response["retrieved_postings"] = postings
    return response


@app.get("/health")
def health():
    return {"status": "ok"}
