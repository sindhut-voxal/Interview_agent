"""Turn policy: stay on a question, clarify, or advance.

Rules run first. A small LLM is used only when the utterance is ambiguous.
"""
from __future__ import annotations

import re

from loguru import logger

from interview.llm_config import gemini_model
from interview.prompts import TURN_ROUTER_PROMPT
from interview.question_generator import _generate_once, parse_json_response

FILLERS = {
    "oh yeah",
    "yeah",
    "yes",
    "yep",
    "yup",
    "hello",
    "hello?",
    "hi",
    "hey",
    "okay",
    "ok",
    "thanks",
    "thank you",
    "mm",
    "mmm",
    "uh",
    "um",
    "huh",
    "hmm",
}

REPEAT_RE = re.compile(
    r"\b("
    r"repeat|say that again|come again|didn't (catch|hear|get)|did not (catch|hear|get)|"
    r"what was the question|what's the question|what is the question|"
    r"pardon|sorry\??$|can you (repeat|say that)|could you (repeat|say that)|"
    r"one more time|say it again|didn't understand|do not understand|don't understand"
    r")\b",
    re.IGNORECASE,
)

WAIT_RE = re.compile(
    r"\b(wait|hold on|hang on|give me a (sec|second|minute)|one second|one minute)\b",
    re.IGNORECASE,
)

DEFAULT_REPEAT_REPLY = "Sure, here it is again."
DEFAULT_WAIT_REPLY = "Take your time."

ACTIONS = {"advance", "stay", "clarify"}


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def _norm_key(text: str) -> str:
    return _norm(text).lower().strip(" .!?,")


def word_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9']+", text or ""))


def looks_like_speech(text: str) -> bool:
    cleaned = _norm(text)
    if not cleaned:
        return False
    return len(cleaned) >= 8 or word_count(cleaned) >= 2


def looks_like_interrupt(text: str) -> bool:
    """Enough signal to cut bot speech (including short wait/repeat)."""
    cleaned = _norm(text)
    if not cleaned:
        return False
    if REPEAT_RE.search(cleaned) or WAIT_RE.search(cleaned):
        return True
    return looks_like_speech(cleaned)


def classify_with_rules(
    utterance: str,
    *,
    interrupted: bool = False,
    playback_pct: float = 1.0,
) -> dict | None:
    """Return a decision dict, or None if the LLM should decide."""
    text = _norm(utterance)
    if not text:
        return {"action": "stay", "reason": "empty", "reply": None}

    key = _norm_key(text)
    words = word_count(text)
    chars = len(text)
    pct = max(0.0, min(1.0, float(playback_pct)))

    if REPEAT_RE.search(text):
        return {"action": "clarify", "reason": "clarification_request", "reply": DEFAULT_REPEAT_REPLY}

    if WAIT_RE.search(text) and words <= 10:
        return {"action": "clarify", "reason": "wait", "reply": DEFAULT_WAIT_REPLY}

    if key in FILLERS or (words <= 2 and key.replace(" ", "") in {f.replace(" ", "") for f in FILLERS}):
        if interrupted and pct < 0.85:
            return {"action": "clarify", "reason": "clarification_request", "reply": DEFAULT_REPEAT_REPLY}
        return {"action": "stay", "reason": "noise", "reply": None}

    # Talking over the last part of the question with a real-sized answer.
    if interrupted and pct >= 0.75 and (chars >= 50 or words >= 8):
        return {"action": "advance", "reason": "answered", "reply": None}

    # Early barge-in that is not clearly an answer — repeat the question.
    if interrupted and pct < 0.4 and words < 8:
        return {"action": "clarify", "reason": "clarification_request", "reply": DEFAULT_REPEAT_REPLY}

    # After the bot finished, a substantial utterance is an answer.
    if not interrupted and (chars >= 50 or words >= 8):
        return {"action": "advance", "reason": "answered", "reply": None}

    return None


def _fallback_decision(*, interrupted: bool, playback_pct: float, utterance: str) -> dict:
    words = word_count(utterance)
    if interrupted and playback_pct < 0.5:
        return {"action": "clarify", "reason": "clarification_request", "reply": DEFAULT_REPEAT_REPLY}
    if words >= 8 or len(_norm(utterance)) >= 50:
        return {"action": "advance", "reason": "answered", "reply": None}
    return {"action": "stay", "reason": "incomplete", "reply": None}


def _normalize_decision(data: dict, utterance: str, interrupted: bool, playback_pct: float) -> dict:
    action = str(data.get("action") or "").strip().lower()
    if action not in ACTIONS:
        return _fallback_decision(interrupted=interrupted, playback_pct=playback_pct, utterance=utterance)
    reason = str(data.get("reason") or "answered").strip()
    reply = data.get("reply")
    if action != "clarify":
        reply = None
    elif not isinstance(reply, str) or not reply.strip():
        reply = DEFAULT_REPEAT_REPLY
    else:
        reply = " ".join(reply.strip().split())
        if len(reply.split()) > 24:
            reply = DEFAULT_REPEAT_REPLY
    return {"action": action, "reason": reason, "reply": reply}


async def classify_with_llm(
    *,
    question: str,
    partial: str,
    utterance: str,
    interrupted: bool,
    playback_pct: float,
    timeout: int = 8,
) -> dict:
    prompt = TURN_ROUTER_PROMPT.format(
        question=question or "",
        partial=partial or "(empty)",
        utterance=utterance or "",
        interrupted="true" if interrupted else "false",
        playback_pct=f"{max(0.0, min(1.0, playback_pct)):.2f}",
    )
    model = gemini_model()
    raw = await _generate_once(prompt, model, timeout=timeout)
    data = parse_json_response(raw)
    if not isinstance(data, dict):
        raise ValueError("Turn router JSON must be an object")
    return _normalize_decision(data, utterance, interrupted, playback_pct)


async def decide_turn(
    *,
    question: str,
    partial: str,
    utterance: str,
    interrupted: bool,
    playback_pct: float,
) -> dict:
    ruled = classify_with_rules(
        utterance,
        interrupted=interrupted,
        playback_pct=playback_pct,
    )
    if ruled is not None:
        logger.info(f"TurnRouter rules → {ruled['action']} ({ruled['reason']})")
        return ruled
    try:
        decision = await classify_with_llm(
            question=question,
            partial=partial,
            utterance=utterance,
            interrupted=interrupted,
            playback_pct=playback_pct,
        )
        logger.info(f"TurnRouter LLM → {decision['action']} ({decision['reason']})")
        return decision
    except Exception as e:
        logger.warning(f"TurnRouter LLM failed, using fallback: {e}")
        return _fallback_decision(
            interrupted=interrupted,
            playback_pct=playback_pct,
            utterance=utterance,
        )
