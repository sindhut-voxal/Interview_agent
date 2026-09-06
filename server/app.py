import os
import uuid
import json
import pathlib
import asyncio
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

app = FastAPI(title="Voxal — 10-min Screening Interview")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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
            return text.strip() or data.decode("utf-8", errors="ignore")
        except Exception as e:
            logger.warning(f"pypdf failed for {name}: {e}, falling back to utf-8")
            return data.decode("utf-8", errors="ignore")

    if suffix in (".docx", ".doc"):
        try:
            import docx
            import io
            doc = docx.Document(io.BytesIO(data))
            return "\n".join([p.text for p in doc.paragraphs])
        except Exception as e:
            logger.warning(f"docx parse failed for {name}: {e}")
            return data.decode("utf-8", errors="ignore")

    # txt, md, etc — try utf-8
    try:
        return data.decode("utf-8")
    except Exception:
        return data.decode("utf-8", errors="ignore")


def _write_latest_interview(resume: str, jd: str):
    """Write latest resume/JD so the Pipecat bot (voice mode) can pick it up."""
    import tempfile
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
        # file was uploaded — extract
        resume = await extract_text_from_upload(resume_file)
    elif resume_text and resume_text.strip():
        resume = resume_text.strip()

    if jd_file and jd_file.filename:
        # need to re-read because earlier if consumed? already handled
        jd = await extract_text_from_upload(jd_file)
    elif jd_text and jd_text.strip():
        jd = jd_text.strip()

    # Edge: both provided — file takes precedence already
    if not resume:
        raise HTTPException(400, "Resume is required — upload a file or paste text.")
    if not jd:
        raise HTTPException(400, "Job description is required — upload a file or paste text.")

    # Trim to avoid huge prompts
    resume = resume[:15000]
    jd = jd[:15000]

    logger.info(f"Creating interview — resume {len(resume)} chars, JD {len(jd)} chars")

    # Dedup key = hash of trimmed resume+jd
    import hashlib, time
    cache_key = hashlib.sha256(f"{resume}\n---\n{jd}".encode()).hexdigest()
    now = time.time()
    # Reuse if same request within 60s (double-click / client retry)
    cached = _generation_cache.get(cache_key)
    if cached and (now - cached[0] < 60):
        logger.info("Dedup: reusing cached interview for same resume/JD within 60s")
        state = cached[1]
    else:
        # Lock per key so concurrent duplicate requests coalesce to single LLM call
        lock = _generation_locks.setdefault(cache_key, asyncio.Lock())
        async with lock:
            # double-check after acquiring lock
            cached2 = _generation_cache.get(cache_key)
            if cached2 and (time.time() - cached2[0] < 60):
                logger.info("Dedup (locked): reusing cached interview")
                state = cached2[1]
            else:
                state = await controller.create_interview(resume=resume, job_description=jd)
                if not state or not state.questions:
                    raise HTTPException(500, "LLM failed to generate interview questions")
                _generation_cache[cache_key] = (time.time(), state)
                # prune old entries (>5 min)
                for k, (ts, _) in list(_generation_cache.items()):
                    if time.time() - ts > 300:
                        _generation_cache.pop(k, None)

    session_id = str(uuid.uuid4())
    sessions[session_id] = state

    _write_latest_interview(resume, jd)

    first = state.get_current_question()

    return {
        "session_id": session_id,
        "total": len(state.questions),
        "questions": state.questions,
        "current_question": first,
        "current_index": 0,
    }


@app.post("/api/interview/{session_id}/answer")
async def submit_answer(session_id: str, payload: dict):
    """
    Body: { "answer": "..." }
    Returns: { feedback, next_question, done, final_score?, progress }
    """
    state = sessions.get(session_id)
    if not state:
        raise HTTPException(404, "Session not found. Create a new interview first.")

    answer = (payload.get("answer") or "").strip()
    if not answer:
        raise HTTPException(400, "Answer must not be empty.")

    current = state.get_current_question()
    if current is None:
        return {
            "done": True,
            "final_score": state.final_score,
            "message": "Interview already complete.",
        }

    next_q = await controller.submit_answer(state=state, answer=answer)

    last_eval = state.evaluations[-1] if state.evaluations else None
    feedback = (last_eval or {}).get("feedback", "")
    score = (last_eval or {}).get("score", 0)

    if next_q is None:
        # done
        return {
            "done": True,
            "feedback": feedback,
            "score": score,
            "final_score": state.final_score,
            "evaluations": state.evaluations,
            "progress": {"current": len(state.questions), "total": len(state.questions)},
            "message": "Thank you. That was the last question. The interview is now complete.",
        }

    idx = state.current_question_index
    return {
        "done": False,
        "feedback": feedback,
        "score": score,
        "next_question": next_q,
        "progress": {"current": idx + 1, "total": len(state.questions)},
        # front-end will show: feedback + "Let's move to the next question." + next_question
        "transition": f"{feedback} Let's move to the next question." if feedback else "Thanks. Let's move to the next question.",
    }


@app.get("/api/interview/{session_id}")
async def get_interview(session_id: str):
    state = sessions.get(session_id)
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

    # Serve client files (css/js etc) — catch-all that does NOT shadow /api or /docs
    @app.get("/{path:path}", include_in_schema=False)
    async def serve_spa(path: str):
        if path.startswith("api/") or path.startswith("docs") or path.startswith("openapi") or path.startswith("redoc"):
            return JSONResponse({"detail": "Not found"}, status_code=404)
        candidate = CLIENT_DIR / path
        if candidate.is_file():
            return FileResponse(str(candidate))
        # SPA fallback
        index = CLIENT_DIR / "index.html"
        if index.exists():
            return FileResponse(str(index))
        return JSONResponse({"detail": "Not found"}, status_code=404)


if __name__ == "__main__":
    import uvicorn
    # When run as `python app.py` from server/ dir, app is importable as __main__:app
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
