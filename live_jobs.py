"""
live_jobs.py

Fetches the live-scraped Naukri job postings JSON hosted on GitHub and does
lightweight keyword-based retrieval + DETERMINISTIC missing-skill
computation against it. Used by app.py's /chat endpoint as a live (as
opposed to the pre-built Chroma store) data source.

Why not just reuse rag_pipeline.retrieve_jobs()?
  - That reads from the pre-built local Chroma store (chroma_store/), which
    only reflects whatever was ingested into it ahead of time.
  - This module instead fetches JOBS_URL fresh over HTTP (with a short
    cache to avoid re-downloading on every message), so /chat always works
    off whatever's currently in the GitHub file — genuinely "live",
    trading semantic search for simpler keyword-overlap matching (no
    pre-built vector index needed for a file fetched at request time).

Why compute missing_skills in Python instead of asking the LLM?
  - "Which skills are missing" is a deterministic set-difference between
    the job's tagged skills and the user's current_skills. Computing it
    here means /chat's answer is always grounded in the real posting data
    - the LLM is then just asked to phrase the missing-skill fact
    naturally, not to invent it.

NOT RUN IN THIS SANDBOX (needs `requests` and network access to
raw.githubusercontent.com). Source dataset:
https://github.com/saitarun1806/nakuri-webscraping
"""

import re
import time
from typing import Optional

import requests

JOBS_URL = (
    "https://raw.githubusercontent.com/saitarun1806/nakuri-webscraping/"
    "refs/heads/main/naukri_jobs_all_india.json"
)

# Refetch at most this often; the underlying GitHub file doesn't change
# every second, and re-downloading it on every chat message would be slow
# and wasteful. Bump this down (or call retrieve_live_jobs with force=True
# from a manual /refresh route) if you need tighter freshness.
_CACHE_TTL_SECONDS = 15 * 60
_cache: dict = {"fetched_at": 0.0, "records": []}


# ---------- cleaning helpers ----------
# The scraper's "Skills" field mixes real key-skills with page chrome
# (related-job carousels, footer links, disclaimers) — cut at the first
# such marker before treating the rest as skill tokens.
_SKILLS_JUNK_MARKERS = [
    "About company", "About the company", "Beware of imposters",
    "Similar jobs", "Jobs you might be interested in", "Home ",
    "roles\nyou might be interested in", "Company Info", "Reviews\nView all",
]


def _clean_skills(raw: Optional[str]) -> list[str]:
    if not raw or raw == "N/A":
        return []
    text = raw
    cut_at = len(text)
    for marker in _SKILLS_JUNK_MARKERS:
        idx = text.find(marker)
        if idx != -1:
            cut_at = min(cut_at, idx)
    text = text[:cut_at]
    text = re.sub(r"Skills highlighted with.*?preferred keyskills,?", "", text)
    tokens = [t.strip(" .\n") for t in text.split(",")]
    return [t for t in tokens if t and t.lower() != "n/a"]


_EXP_RANGE_RE = re.compile(r"(\d+)\s*-\s*(\d+)\s*years?", re.IGNORECASE)


def _parse_experience_min(raw: Optional[str]) -> Optional[int]:
    if not raw or raw == "N/A":
        return None
    m = _EXP_RANGE_RE.search(raw.replace("Yrs", "years").replace("yrs", "years"))
    return int(m.group(1)) if m else None


def _fetch_records(force: bool = False) -> list[dict]:
    now = time.time()
    if not force and _cache["records"] and (now - _cache["fetched_at"]) < _CACHE_TTL_SECONDS:
        return _cache["records"]
    resp = requests.get(JOBS_URL, timeout=60)
    resp.raise_for_status()
    records = resp.json()
    _cache["records"] = records
    _cache["fetched_at"] = now
    return records


def _relevance_score(record: dict, target_role: str, location: Optional[str]) -> int:
    role_terms = [t.lower() for t in target_role.split() if len(t) > 2]
    haystack = " ".join([
        record.get("Title", ""), record.get("Role", ""),
        record.get("Skills", ""), record.get("Department", ""),
        record.get("RoleCategory", ""),
    ]).lower()
    score = sum(1 for t in role_terms if t in haystack)
    if location and (record.get("State") or "").lower() == location.lower():
        score += 2
    return score


def _skill_is_covered(job_skill: str, user_skills_lower: set[str]) -> bool:
    """Loose match: job-tagged skill counts as "known" if it substring-matches
    (either direction) something the user listed — job skill tags are often
    multi-word ("Data Analysis") while users type shorter forms ("data"),
    and vice versa ("Python" vs "Python programming")."""
    js = job_skill.lower()
    return any(js in us or us in js for us in user_skills_lower)


def retrieve_live_jobs(
    current_skills: list[str],
    target_role: str,
    location: Optional[str] = None,
    k: int = 6,
    work_experience: Optional[int] = None,
    force_refresh: bool = False,
) -> list[dict]:
    """Keyword-score the live GitHub dataset against target_role/location,
    return the top-k postings with matched_skills/missing_skills computed
    deterministically against current_skills."""
    records = _fetch_records(force=force_refresh)
    user_skills_lower = {s.strip().lower() for s in current_skills if s and s.strip()}

    scored = []
    for rec in records:
        title = rec.get("Title") or rec.get("Role")
        if not title or title == "N/A":
            continue
        if work_experience == 0:
            exp_min = _parse_experience_min(rec.get("Experience"))
            # Skip postings that clearly aren't entry-level for freshers
            # (min experience required is more than 1 year).
            if exp_min is not None and exp_min > 1:
                continue
        score = _relevance_score(rec, target_role, location)
        if score <= 0:
            continue
        scored.append((score, rec))

    scored.sort(key=lambda x: x[0], reverse=True)

    results = []
    for score, rec in scored[:k]:
        job_skills = _clean_skills(rec.get("Skills"))
        missing = [js for js in job_skills if not _skill_is_covered(js, user_skills_lower)]
        matched = [js for js in job_skills if js not in missing]
        results.append({
            "title": rec.get("Title"),
            "company": rec.get("Company"),
            "location": rec.get("State"),
            "experience_required": rec.get("Experience"),
            "apply_link": rec.get("ApplyLink"),
            "matched_skills": matched,
            "missing_skills": missing,
            "relevance_score": score,
        })
    return results
