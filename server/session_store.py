"""Shared session persistence for app.py (:8000) and bot.py (:7860).

Both processes run separately, so in-memory `sessions` dict cannot be shared.
This module provides file-backed persistence so bot answers are visible to /report.
"""
import json
import os
import pathlib
import tempfile
import time

from loguru import logger

from interview.state import InterviewState

SESSION_DIR = pathlib.Path(tempfile.gettempdir()) / "voxal_sessions"
SESSION_DIR.mkdir(parents=True, exist_ok=True)


def session_path(session_id: str) -> pathlib.Path:
    safe = "".join(c for c in session_id if c.isalnum() or c in "-_")
    return SESSION_DIR / f"{safe}.json"


def report_lock_path(session_id: str) -> pathlib.Path:
    return session_path(session_id).with_suffix(".report.lock")


def acquire_report_lock(session_id: str, stale_s: float = 180.0) -> bool:
    """Cross-process lock so bot.py and app.py do not double-evaluate."""
    path = report_lock_path(session_id)
    try:
        fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        try:
            os.write(fd, str(time.time()).encode("utf-8"))
        finally:
            os.close(fd)
        return True
    except FileExistsError:
        try:
            age = time.time() - path.stat().st_mtime
            if age > stale_s:
                path.unlink(missing_ok=True)
                return acquire_report_lock(session_id, stale_s)
        except Exception:
            return False
        return False
    except Exception as e:
        logger.warning(f"Failed to acquire report lock for {session_id}: {e}")
        return False


def release_report_lock(session_id: str) -> None:
    path = report_lock_path(session_id)
    try:
        path.unlink(missing_ok=True)
    except Exception:
        pass


def save_session(session_id: str, state: InterviewState) -> None:
    payload = {
        "resume": state.resume,
        "job_description": state.job_description,
        "questions": state.questions,
        "current_question_index": state.current_question_index,
        "answers": state.answers,
        "evaluations": state.evaluations,
        "final_score": state.final_score,
        "report_status": getattr(state, "report_status", "idle") or "idle",
        "report_error": getattr(state, "report_error", None),
    }
    path = session_path(session_id)
    tmp = path.with_suffix(".tmp")
    try:
        tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, path)
        logger.debug(f"Saved session {session_id} -> {path}")
    except Exception as e:
        logger.warning(f"Failed to save session {session_id}: {e}")
        try:
            if tmp.exists():
                tmp.unlink()
        except Exception:
            pass


def load_session(session_id: str) -> InterviewState | None:
    path = session_path(session_id)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return InterviewState(
            resume=data["resume"],
            job_description=data["job_description"],
            questions=data["questions"],
            current_question_index=data.get("current_question_index", 0),
            answers=data.get("answers", []),
            evaluations=data.get("evaluations", []),
            final_score=data.get("final_score"),
            report_status=data.get("report_status", "idle") or "idle",
            report_error=data.get("report_error"),
        )
    except Exception as e:
        logger.warning(f"Failed to load session {session_id}: {e}")
        return None
