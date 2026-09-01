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


def _fallback_questions(resume: str, jd: str):
    """Basic screening fallback when LLM is unavailable — always returns 6 short questions."""
    return [
        {"id": 1, "question": "Could you briefly introduce yourself and walk me through your background?", "skill": "Introduction", "criteria": ["Clear summary", "Relevant experience mentioned"], "weight": 15},
        {"id": 2, "question": f"Your resume mentions projects with {resume[:60].split(',')[0] if resume else 'your stack'} — can you briefly explain one project you enjoyed working on and your role in it?", "skill": "Project Experience", "criteria": ["Explains role clearly", "Shows understanding"], "weight": 20},
        {"id": 3, "question": "What are the core responsibilities you're most comfortable with for this role, and why?", "skill": "Role Fit", "criteria": ["Aligns with JD", "Shows motivation"], "weight": 15},
        {"id": 4, "question": "In Python (or your primary language), how would you explain a function versus a class to a junior developer?", "skill": "Fundamentals", "criteria": ["Clear definition", "Simple example"], "weight": 15},
        {"id": 5, "question": "Tell me about a time you debugged a tricky issue — what was the problem and how did you solve it?", "skill": "Problem Solving", "criteria": ["Structured story", "Shows approach"], "weight": 15},
        {"id": 6, "question": "What are you hoping to learn or grow in during your first few months in this role?", "skill": "Motivation / Culture", "criteria": ["Shows curiosity", "Growth mindset"], "weight": 20},
    ]

def _fallback_feedback(answer: str, weight: int):
    ans = answer.strip()
    if len(ans) < 20:
        return {"feedback": "Thanks — try adding a bit more detail next time.", "score": max(1, weight // 3)}
    if len(ans) < 80:
        return {"feedback": "Good start — you covered the basics clearly.", "score": int(weight * 0.6)}
    return {"feedback": "Nice — you explained it clearly and concisely.", "score": int(weight * 0.85)}


def _write_latest_interview(resume: str, jd: str):
    """Write latest resume/JD so the Pipecat bot (voice mode) can pick it up."""
    payload = {"resume": resume, "job_description": jd}
    for p in [SERVER_DIR / "latest_interview.json", pathlib.Path("/tmp/latest_interview.json")]:
        try:
            p.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        except Exception as e:
            logger.warning(f"Failed to write {p}: {e}")

# ---------- API ----------
@app.get("/api/health")
async def health():
    return {"status": "ok", "sessions": len(sessions)}


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

    state = None
    try:
        state = await controller.create_interview(resume=resume, job_description=jd)
    except Exception as e:
        logger.warning(f"LLM question generation failed, using fallback: {e}")
        # Build a fallback InterviewState directly so UI still works without API keys
        from interview.state import InterviewState as _IS
        fallback_qs = _fallback_questions(resume, jd)
        state = _IS(resume=resume, job_description=jd, questions=fallback_qs)
    if not state or not state.questions:
        from interview.state import InterviewState as _IS2
        state = _IS2(resume=resume, job_description=jd, questions=_fallback_questions(resume, jd))

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

    # Evaluate + advance — controller does scoring + moving pointer
    # Fallback if LLM evaluator fails (e.g. missing API key)
    next_q = None
    try:
        next_q = await controller.submit_answer(state=state, answer=answer)
    except Exception as e:
        logger.warning(f"LLM evaluation failed, using fallback: {e}")
        # Manual fallback: mimic what controller does but locally
        curr = state.get_current_question()
        if curr is not None:
            fb = _fallback_feedback(answer, int(curr.get("weight", 15)))
            state.add_answer(answer)
            state.add_evaluation({"question_id": curr["id"], "feedback": fb["feedback"], "score": fb["score"], "strengths": [], "improvements": []})
            state.move_to_next_question()
            if state.is_interview_complete():
                from interview.scoring import calculate_final_score
                calculate_final_score(state)
                next_q = None
            else:
                next_q = state.get_current_question()

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
