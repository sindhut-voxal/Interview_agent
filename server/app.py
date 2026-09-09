import os
import uuid
import json
import pathlib
import asyncio
import tempfile
from copy import deepcopy
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse
from loguru import logger
from dotenv import load_dotenv

load_dotenv()

from interview.controller import InterviewController
from interview.state import InterviewState
from session_store import save_session, load_session, acquire_report_lock, release_report_lock

app = FastAPI(title="Voxal — 10-min Screening Interview")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "http://localhost:7860",
        "http://127.0.0.1:7860",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory store: session_id -> InterviewState
sessions: dict[str, InterviewState] = {}
controller = InterviewController()

# Dedupe: if same resume+JD received within 60s (e.g. client double-fire), reuse same generation
_generation_cache: dict[str, tuple[float, InterviewState]] = {}
_generation_locks: dict[str, asyncio.Lock] = {}
_report_locks: dict[str, asyncio.Lock] = {}

CLIENT_DIR = pathlib.Path(__file__).parent.parent / "client"
SERVER_DIR = pathlib.Path(__file__).parent

# ---------- helpers ----------
async def extract_text_from_upload(file: UploadFile) -> str:
    name = file.filename or ""
    data = await file.read()
    suffix = pathlib.Path(name).suffix.lower()

    if suffix == ".pdf":
        try:
            from pypdf import PdfReader
            import io
            reader = PdfReader(io.BytesIO(data))
            text = "\n".join([p.extract_text() or "" for p in reader.pages])
            stripped = text.strip()
            if stripped:
                return stripped
            raise ValueError("Could not extract text from PDF.")
        except Exception as e:
            logger.warning(f"pypdf failed for {name}: {e}")
            raise HTTPException(400, f"Could not extract text from PDF '{name}': {e}")

    if suffix == ".docx":
        try:
            import docx
            import io
            doc = docx.Document(io.BytesIO(data))
            text = "\n".join([p.text for p in doc.paragraphs])
            stripped = text.strip()
            if stripped:
                return stripped
            raise ValueError("Could not extract text from DOCX.")
        except HTTPException:
            raise
        except Exception as e:
            logger.warning(f"docx parse failed for {name}: {e}")
            raise HTTPException(400, f"Could not extract text from DOCX '{name}': {e}")

    if suffix == ".doc":
        raise HTTPException(400, f"Legacy .doc format is not supported for '{name}' — please upload .docx or PDF.")

    try:
        return data.decode("utf-8")
    except Exception:
        return data.decode("utf-8", errors="ignore")


def _get_session(session_id: str) -> InterviewState | None:
    """Try disk first (bot writes), fallback to in-memory."""
    state = load_session(session_id)
    if state is not None:
        sessions[session_id] = state
        return state
    return sessions.get(session_id)


def _write_latest_interview(resume: str, jd: str):
    """Write latest resume/JD so the Pipecat bot (voice mode) can pick it up. Debug only — session_id is authoritative."""
    payload = {"resume": resume, "job_description": jd}
    tmp_path = pathlib.Path(tempfile.gettempdir()) / "latest_interview.json"
    for p in [SERVER_DIR / "latest_interview.json", tmp_path]:
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            logger.debug(f"Wrote latest interview to {p}")
        except Exception as e:
            logger.warning(f"Failed to write {p}: {e}")

# ---------- API ----------
@app.get("/api/health")
async def health():
    return {"status": "ok", "sessions": len(sessions)}


@app.api_route("/api/offer", methods=["GET", "POST", "OPTIONS", "PATCH"])
async def offer_hint():
    """Hint when UI hits app server (8000) instead of voice runner (7860)."""
    return JSONResponse(
        {"detail": "Voice runner not running on this port. Start `python server/bot.py` (WebRTC on :7860). UI should POST to http://localhost:7860/api/offer"},
        status_code=503,
    )


@app.post("/api/extract")
async def extract(file: UploadFile = File(...)):
    """Extract text from an uploaded file (pdf/docx/txt) — used by voice UI before WebRTC."""
    try:
        text = await extract_text_from_upload(file)
        return {"text": text, "filename": file.filename}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Extract failed: {e}")


@app.post("/api/interview/create")
async def create_interview(
    resume_file: Optional[UploadFile] = File(None),
    jd_file: Optional[UploadFile] = File(None),
    resume_text: Optional[str] = Form(None),
    jd_text: Optional[str] = Form(None),
):
    """
    Create a new screening interview.

    Accepts either:
      - resume_file (pdf/docx/txt) OR resume_text
      - jd_file (pdf/docx/txt) OR jd_text

    At least one resume source and one JD source must be provided.
    """
    resume = ""
    jd = ""

    if resume_file and resume_file.filename:
        resume = await extract_text_from_upload(resume_file)
    elif resume_text and resume_text.strip():
        resume = resume_text.strip()

    if jd_file and jd_file.filename:
        jd = await extract_text_from_upload(jd_file)
    elif jd_text and jd_text.strip():
        jd = jd_text.strip()

    if not resume:
        raise HTTPException(400, "Resume is required — upload a file or paste text.")
    if not jd:
        raise HTTPException(400, "Job description is required — upload a file or paste text.")

    resume = resume[:15000]
    jd = jd[:15000]

    logger.info(f"Creating interview — resume {len(resume)} chars, JD {len(jd)} chars")

    import hashlib, time
    cache_key = hashlib.sha256(f"{resume}\n---\n{jd}".encode()).hexdigest()
    now = time.time()
    cached = _generation_cache.get(cache_key)
    if cached and (now - cached[0] < 60):
        logger.info("Dedup: reusing cached interview for same resume/JD within 60s")
        state = deepcopy(cached[1])
    else:
        lock = _generation_locks.setdefault(cache_key, asyncio.Lock())
        async with lock:
            cached2 = _generation_cache.get(cache_key)
            if cached2 and (time.time() - cached2[0] < 60):
                logger.info("Dedup (locked): reusing cached interview")
                state = deepcopy(cached2[1])
            else:
                state = await controller.create_interview(resume=resume, job_description=jd)
                if not state or not state.questions:
                    raise HTTPException(500, "LLM failed to generate interview questions")
                _generation_cache[cache_key] = (time.time(), state)
                state = deepcopy(state)
                for k, (ts, _) in list(_generation_cache.items()):
                    if time.time() - ts > 300:
                        _generation_cache.pop(k, None)
                        _generation_locks.pop(k, None)

    session_id = str(uuid.uuid4())
    sessions[session_id] = state
    save_session(session_id, state)

    _write_latest_interview(resume, jd)

    # Do NOT leak questions to candidate before interview. Voice runner loads them via session_id.
    return {
        "session_id": session_id,
        "total": len(state.questions),
    }


@app.post("/api/interview/{session_id}/answer")
async def submit_answer(session_id: str, payload: dict):
    """
    REAL-TIME: store answer and return next question. No LLM eval.
    Body: { "answer": "..." }
    Returns: { next_question, done, progress }
    """
    state = _get_session(session_id)
    if not state:
        raise HTTPException(404, "Session not found. Create a new interview first.")

    answer = (payload.get("answer") or "").strip()
    if not answer:
        raise HTTPException(400, "Answer must not be empty.")

    current = state.get_current_question()
    if current is None:
        return {
            "done": True,
            "message": "Interview already complete. Call /report to generate evaluation.",
        }

    next_q = await controller.submit_answer(state=state, answer=answer)
    save_session(session_id, state)

    if next_q is None:
        return {
            "done": True,
            "progress": {"current": len(state.questions), "total": len(state.questions)},
            "message": "Thank you. That was the last question. Generating your detailed report...",
            "answers": state.answers,
        }

    idx = state.current_question_index
    return {
        "done": False,
        "next_question": next_q,
        "progress": {"current": idx + 1, "total": len(state.questions)},
        "transition": "Thanks for your answer. Let's move to the next question.",
    }


def _report_payload(session_id: str, state: InterviewState) -> dict:
    strengths = []
    improvements = []
    for ev in state.evaluations or []:
        strengths.extend(ev.get("strengths") or [])
        improvements.extend(ev.get("improvements") or [])

    def dedupe(seq):
        seen = set()
        out = []
        for x in seq:
            if x not in seen:
                seen.add(x)
                out.append(x)
        return out

    return {
        "session_id": session_id,
        "status": getattr(state, "report_status", "idle") or "idle",
        "is_complete": state.is_interview_complete(),
        "answered": len(state.answers),
        "total": len(state.questions),
        "final_score": state.final_score,
        "evaluations": state.evaluations,
        "answers": state.answers,
        "questions": state.questions,
        "strengths": dedupe(strengths)[:5],
        "improvements": dedupe(improvements)[:5],
        "error": getattr(state, "report_error", None),
    }


def _report_is_ready(state: InterviewState) -> bool:
    return bool(
        state.evaluations
        and state.final_score is not None
        and len(state.evaluations) >= len(state.answers)
        and len(state.answers) >= len(state.questions)
    )


async def _run_report(session_id: str, state: InterviewState) -> dict:
    state.report_status = "running"
    state.report_error = None
    save_session(session_id, state)
    try:
        report = await controller.evaluate_interview(state)
        state.report_status = "ready"
        save_session(session_id, state)
        payload = _report_payload(session_id, state)
        payload["status"] = "ready"
        payload["final_score"] = report["final_score"]
        payload["evaluations"] = report["evaluations"]
        payload["strengths"] = report.get("strengths", [])
        payload["improvements"] = report.get("improvements", [])
        return payload
    except Exception as e:
        state.report_status = "error"
        state.report_error = str(e)
        save_session(session_id, state)
        raise


@app.get("/api/interview/{session_id}/report")
async def get_report(session_id: str):
    """Pollable report snapshot. Does not start evaluation."""
    state = _get_session(session_id)
    if not state:
        raise HTTPException(404, "Session not found")
    payload = _report_payload(session_id, state)
    if _report_is_ready(state):
        payload["status"] = "ready"
    elif not state.is_interview_complete():
        payload["status"] = "incomplete"
    elif getattr(state, "report_status", "idle") == "error":
        payload["status"] = "error"
    elif getattr(state, "report_status", "idle") == "running":
        payload["status"] = "running"
    else:
        payload["status"] = "pending"
    return payload


@app.post("/api/interview/{session_id}/report")
async def generate_report(session_id: str):
    """
    POST-INTERVIEW: batch Gemini evaluation for all stored answers.
    Call after is_complete. Returns detailed report, or 202 if already running.
    """
    state = _get_session(session_id)
    if not state:
        raise HTTPException(404, "Session not found")

    if not state.is_interview_complete():
        raise HTTPException(
            400,
            f"Interview not complete: {len(state.answers)}/{len(state.questions)} answered. Finish all questions first.",
        )

    if _report_is_ready(state):
        payload = _report_payload(session_id, state)
        payload["status"] = "ready"
        return payload

    lock = _report_locks.setdefault(session_id, asyncio.Lock())
    if lock.locked() or not acquire_report_lock(session_id):
        return JSONResponse(content={**_report_payload(session_id, state), "status": "running"}, status_code=202)

    async with lock:
        try:
            state = _get_session(session_id) or state
            if _report_is_ready(state):
                payload = _report_payload(session_id, state)
                payload["status"] = "ready"
                return payload
            logger.info(f"Generating batch report for {session_id} ({len(state.answers)} answers)")
            return await _run_report(session_id, state)
        except HTTPException:
            raise
        except Exception as e:
            logger.exception(f"Report generation failed: {e}")
            raise HTTPException(500, f"Report generation failed: {e}")
        finally:
            release_report_lock(session_id)


@app.get("/api/interview/{session_id}/progress")
async def get_interview_progress(session_id: str):
    """Live UI snapshot: current question only (does not leak remaining questions)."""
    state = _get_session(session_id)
    if not state:
        raise HTTPException(404, "Session not found")
    q = state.get_current_question()
    done = state.is_interview_complete()
    return {
        "type": "interview_progress",
        "done": done,
        "current_index": state.current_question_index,
        "total": len(state.questions),
        "question": None if done or q is None else q.get("question"),
        "question_id": None if not q else q.get("id"),
    }


@app.get("/api/interview/{session_id}")
async def get_interview(session_id: str):
    state = _get_session(session_id)
    if not state:
        raise HTTPException(404, "Session not found")
    return {
        "session_id": session_id,
        "questions": state.questions,
        "current_question": state.get_current_question(),
        "current_index": state.current_question_index,
        "answers": state.answers,
        "evaluations": state.evaluations,
        "final_score": state.final_score,
        "is_complete": state.is_interview_complete(),
    }


# ---------- Static client ----------
if CLIENT_DIR.exists():
    @app.get("/", include_in_schema=False)
    async def serve_index():
        return FileResponse(str(CLIENT_DIR / "index.html"))

    @app.get("/{path:path}", include_in_schema=False)
    async def serve_spa(path: str):
        if path.startswith("api/") or path.startswith("docs") or path.startswith("openapi") or path.startswith("redoc"):
            return JSONResponse({"detail": "Not found"}, status_code=404)
        candidate = CLIENT_DIR / path
        if candidate.is_file():
            return FileResponse(str(candidate))
        index = CLIENT_DIR / "index.html"
        if index.exists():
            return FileResponse(str(index))
        return JSONResponse({"detail": "Not found"}, status_code=404)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
