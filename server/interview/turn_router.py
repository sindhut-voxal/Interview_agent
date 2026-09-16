"""Turn policy: stay, clarify, answer a candidate question, skip, or advance.

Rules run first. Gemini function tools are used only when the utterance is ambiguous.
"""
from __future__ import annotations

import asyncio
import os
import re

from loguru import logger

from interview.llm_config import gemini_model
from interview.prompts import TURN_ROUTER_PROMPT

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

SKIP_RE = re.compile(
    r"\b("
    r"skip( this| that| it| the question| this one| that one)?|"
    r"pass( on this)?|"
    r"(can|could) we skip|"
    r"let's (skip|move on)|lets (skip|move on)"
    r")\b",
    re.IGNORECASE,
)

IDK_RE = re.compile(
    r"\b(i (don't|do not|dont) know|no idea|not sure)\b",
    re.IGNORECASE,
)

END_RE = re.compile(
    r"\b("
    r"end (the )?interview|stop (the )?interview|"
    r"i('m| am) done|"
    r"that's all( for me)?|"
    r"i (want to|have to|need to) (stop|go|leave)|"
    r"(can|could) we (stop|end|finish)"
    r")\b",
    re.IGNORECASE,
)

# Candidate is asking the interviewer — not answering the question.
META_QUESTION_RE = re.compile(
    r"("
    r"\?$|"
    r"\bwhat do you mean\b|"
    r"\b(what|which) (do you|did you) mean\b|"
    r"\bcan you (explain|clarify|define)\b|"
    r"\bcould you (explain|clarify|define)\b|"
    r"\bdid you mean\b|"
    r"\bhow much time\b|"
    r"\bhow many questions\b|"
    r"\bis this (about|for)\b|"
    r"\bdo you want me to\b"
    r")",
    re.IGNORECASE,
)

DEFAULT_REPEAT_REPLY = "Sure, here it is again."
DEFAULT_WAIT_REPLY = "Take your time."
DEFAULT_ANSWER_REPLY = "Happy to clarify."
DEFAULT_SKIP_REPLY = "Alright, we'll skip that."
DEFAULT_END_REPLY = "Of course, we can stop here."
DEFAULT_ADVANCE_ACK = "Got it."
DEFAULT_NUDGE_REPLY = "I'm here whenever you're ready."
DEFAULT_FOLLOWUP_REPLY = "Could you give a brief example?"
DEFAULT_REPAIR_REPLY = "Sorry, I missed that. Could you say it again?"
SCREENING_LIMIT_S = 10 * 60

ACTIONS = {"advance", "stay", "clarify", "answer", "skip", "end", "follow_up"}

TOOL_TO_ACTION = {
    "stay_silent": "stay",
    "acknowledge_wait": "clarify",
    "repeat_or_rephrase": "clarify",
    "answer_candidate_question": "answer",
    "ask_follow_up": "follow_up",
    "submit_answer_and_advance": "advance",
    "skip_question": "skip",
    "end_interview": "end",
}

TOOL_REASON = {
    "stay_silent": "incomplete",
    "acknowledge_wait": "wait",
    "repeat_or_rephrase": "clarification_request",
    "answer_candidate_question": "candidate_question",
    "ask_follow_up": "follow_up",
    "submit_answer_and_advance": "answered",
    "skip_question": "skip",
    "end_interview": "end",
}

TURN_TOOL_NAMES = list(TOOL_TO_ACTION.keys())


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def _norm_key(text: str) -> str:
    return _norm(text).lower().strip(" .!?,")


def word_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9']+", text or ""))


def looks_like_stt_garbage(text: str) -> bool:
    cleaned = _norm(text)
    if not cleaned:
        return False
    letters = re.findall(r"[A-Za-z]", cleaned)
    if len(letters) >= 6 and not re.search(r"[aeiouAEIOU]", cleaned):
        return True
    if re.search(r"(.)\1{5,}", cleaned):
        return True
    if word_count(cleaned) <= 2 and re.search(r"[^A-Za-z0-9'?.!\s]", cleaned) and not re.search(r"[A-Za-z]{3,}", cleaned):
        return True
    return False


def looks_like_speech(text: str) -> bool:
    cleaned = _norm(text)
    if not cleaned:
        return False
    return len(cleaned) >= 8 or word_count(cleaned) >= 2


def looks_like_skip(text: str) -> bool:
    cleaned = _norm(text)
    if not cleaned or word_count(cleaned) > 12:
        return False
    if REPEAT_RE.search(cleaned) or WAIT_RE.search(cleaned):
        return False
    return bool(SKIP_RE.search(cleaned))


def looks_like_end(text: str) -> bool:
    cleaned = _norm(text)
    if not cleaned or word_count(cleaned) > 12:
        return False
    return bool(END_RE.search(cleaned))


def looks_like_idk(text: str) -> bool:
    """Short 'I don't know' is an answer, not a skip."""
    cleaned = _norm(text)
    if not cleaned or word_count(cleaned) > 10:
        return False
    if looks_like_skip(cleaned) or looks_like_end(cleaned):
        return False
    return bool(IDK_RE.search(cleaned))


def restate_question(question: str, max_words: int = 18) -> str:
    """Shorter restatement for repeats — first sentence, clipped."""
    text = _norm(question)
    if not text:
        return ""
    first = re.split(r"(?<=[.?!])\s+", text, maxsplit=1)[0].strip()
    words = first.split()
    if len(words) <= max_words:
        return first
    return " ".join(words[:max_words])


def looks_like_candidate_question(text: str) -> bool:
    cleaned = _norm(text)
    if not cleaned:
        return False
    if REPEAT_RE.search(cleaned):
        return False
    if looks_like_skip(cleaned) or looks_like_end(cleaned):
        return False
    if word_count(cleaned) > 28:
        return False
    return bool(META_QUESTION_RE.search(cleaned))


def looks_like_interrupt(text: str) -> bool:
    """Enough signal to cut bot speech (including short wait/repeat/skip)."""
    cleaned = _norm(text)
    if not cleaned:
        return False
    if REPEAT_RE.search(cleaned) or WAIT_RE.search(cleaned):
        return True
    if looks_like_skip(cleaned) or looks_like_end(cleaned) or looks_like_candidate_question(cleaned):
        return True
    return looks_like_speech(cleaned)


def decision_from_tool(name: str, args: dict | None = None) -> dict:
    args = args or {}
    action = TOOL_TO_ACTION.get(str(name or "").strip())
    if action is None:
        raise ValueError(f"Unknown turn tool: {name}")
    reason = str(args.get("reason") or TOOL_REASON.get(name, action)).strip()
    reply = args.get("reply")
    if action == "clarify" and reason != "wait":
        reason = str(args.get("reason") or "clarification_request")
    if name == "acknowledge_wait":
        reason = "wait"
    return _normalize_decision(
        {"action": action, "reason": reason, "reply": reply, "tool": name},
        utterance="",
        interrupted=False,
        playback_pct=1.0,
    )


def classify_with_rules(
    utterance: str,
    *,
    interrupted: bool = False,
    playback_pct: float = 1.0,
    followup_used: bool = False,
    remaining_min: float = 10.0,
) -> dict | None:
    """Return a decision dict, or None if tools should decide."""
    text = _norm(utterance)
    if not text:
        return {"action": "stay", "reason": "empty", "reply": None, "tool": None}

    key = _norm_key(text)
    words = word_count(text)
    chars = len(text)
    pct = max(0.0, min(1.0, float(playback_pct)))

    if looks_like_stt_garbage(text):
        return {
            "action": "clarify",
            "reason": "stt_repair",
            "reply": DEFAULT_REPAIR_REPLY,
            "tool": "repeat_or_rephrase",
        }

    if REPEAT_RE.search(text):
        return {
            "action": "clarify",
            "reason": "clarification_request",
            "reply": DEFAULT_REPEAT_REPLY,
            "tool": "repeat_or_rephrase",
        }

    if looks_like_end(text):
        return {
            "action": "end",
            "reason": "end",
            "reply": DEFAULT_END_REPLY,
            "tool": "end_interview",
        }

    if looks_like_skip(text):
        return {
            "action": "skip",
            "reason": "skip",
            "reply": DEFAULT_SKIP_REPLY,
            "tool": "skip_question",
        }

    if looks_like_idk(text):
        return {
            "action": "advance",
            "reason": "answered",
            "reply": DEFAULT_ADVANCE_ACK,
            "tool": "submit_answer_and_advance",
        }

    if looks_like_candidate_question(text):
        # Ambiguous meta-talk: tools decide how to answer without advancing.
        return None

    if WAIT_RE.search(text) and words <= 10:
        return {
            "action": "clarify",
            "reason": "wait",
            "reply": DEFAULT_WAIT_REPLY,
            "tool": "acknowledge_wait",
        }

    if key in FILLERS or (words <= 2 and key.replace(" ", "") in {f.replace(" ", "") for f in FILLERS}):
        if interrupted and pct < 0.85:
            return {
                "action": "clarify",
                "reason": "clarification_request",
                "reply": DEFAULT_REPEAT_REPLY,
                "tool": "repeat_or_rephrase",
            }
        return {"action": "stay", "reason": "noise", "reply": None, "tool": "stay_silent"}

    # Talking over the last part of the question with a real-sized answer.
    if interrupted and pct >= 0.75 and (chars >= 50 or words >= 8):
        return {
            "action": "advance",
            "reason": "answered",
            "reply": DEFAULT_ADVANCE_ACK,
            "tool": "submit_answer_and_advance",
        }

    # Early barge-in that is not clearly an answer — repeat the question.
    if interrupted and pct < 0.4 and words < 8:
        return {
            "action": "clarify",
            "reason": "clarification_request",
            "reply": DEFAULT_REPEAT_REPLY,
            "tool": "repeat_or_rephrase",
        }

    # After the bot finished, a substantial utterance is an answer.
    if not interrupted and (chars >= 50 or words >= 8):
        thin = words < 12 and chars < 80
        if thin and not followup_used and remaining_min > 1.0:
            return {
                "action": "follow_up",
                "reason": "thin_answer",
                "reply": DEFAULT_FOLLOWUP_REPLY,
                "tool": "ask_follow_up",
            }
        return {
            "action": "advance",
            "reason": "answered",
            "reply": DEFAULT_ADVANCE_ACK,
            "tool": "submit_answer_and_advance",
        }

    return None


def _fallback_decision(*, interrupted: bool, playback_pct: float, utterance: str) -> dict:
    if looks_like_candidate_question(utterance):
        return {
            "action": "answer",
            "reason": "candidate_question",
            "reply": DEFAULT_ANSWER_REPLY,
            "tool": "answer_candidate_question",
        }
    if looks_like_skip(utterance):
        return {
            "action": "skip",
            "reason": "skip",
            "reply": DEFAULT_SKIP_REPLY,
            "tool": "skip_question",
        }
    if looks_like_idk(utterance):
        return {
            "action": "advance",
            "reason": "answered",
            "reply": DEFAULT_ADVANCE_ACK,
            "tool": "submit_answer_and_advance",
        }
    words = word_count(utterance)
    if interrupted and playback_pct < 0.5:
        return {
            "action": "clarify",
            "reason": "clarification_request",
            "reply": DEFAULT_REPEAT_REPLY,
            "tool": "repeat_or_rephrase",
        }
    if words >= 8 or len(_norm(utterance)) >= 50:
        return {
            "action": "advance",
            "reason": "answered",
            "reply": DEFAULT_ADVANCE_ACK,
            "tool": "submit_answer_and_advance",
        }
    return {"action": "stay", "reason": "incomplete", "reply": None, "tool": "stay_silent"}


def _clip_reply(reply: str | None, max_words: int = 24) -> str | None:
    if not isinstance(reply, str) or not reply.strip():
        return None
    reply = " ".join(reply.strip().split())
    if len(reply.split()) > max_words:
        return None
    return reply


def _normalize_decision(data: dict, utterance: str, interrupted: bool, playback_pct: float) -> dict:
    action = str(data.get("action") or "").strip().lower()
    if action not in ACTIONS:
        return _fallback_decision(interrupted=interrupted, playback_pct=playback_pct, utterance=utterance)
    reason = str(data.get("reason") or "answered").strip()
    tool = data.get("tool")
    reply = _clip_reply(data.get("reply"))
    if action == "stay":
        reply = None
    elif action == "clarify":
        if reason == "wait":
            reply = reply or DEFAULT_WAIT_REPLY
        else:
            reply = reply or DEFAULT_REPEAT_REPLY
    elif action == "answer":
        reply = reply or DEFAULT_ANSWER_REPLY
    elif action == "skip":
        reply = reply or DEFAULT_SKIP_REPLY
    elif action == "end":
        reply = reply or DEFAULT_END_REPLY
    elif action == "follow_up":
        reply = reply or DEFAULT_FOLLOWUP_REPLY
    elif action == "advance":
        if reply and len(reply.split()) > 12:
            reply = None
        reply = reply or DEFAULT_ADVANCE_ACK
    return {"action": action, "reason": reason, "reply": reply, "tool": tool}


def _turn_tool_declarations():
    from google.genai import types

    reply_schema = types.Schema(
        type="STRING",
        description="One short spoken sentence, under 20 words. Empty string if the bot should not speak.",
    )
    reason_schema = types.Schema(type="STRING", description="Short label for logs.")
    params = types.Schema(
        type="OBJECT",
        properties={"reason": reason_schema, "reply": reply_schema},
        required=["reason"],
    )
    specs = [
        ("stay_silent", "They are thinking, filling, or still answering. Do not speak."),
        ("acknowledge_wait", "They asked you to wait or hold on. Speak a brief ack."),
        ("repeat_or_rephrase", "They did not hear or understand the question, or spoke too soon."),
        (
            "answer_candidate_question",
            "They asked YOU something (meaning, time, process). Answer in one sentence. Do not advance.",
        ),
        (
            "ask_follow_up",
            "The answer is real but thin. Ask one short follow-up. Do not advance. Only if follow-up was not already used.",
        ),
        (
            "submit_answer_and_advance",
            "This utterance is a real answer to the current interview question. Optional short ack in reply.",
        ),
        ("skip_question", "They want to skip or pass on this question and are not answering it."),
        ("end_interview", "They want to stop the interview."),
    ]
    return [
        types.FunctionDeclaration(name=name, description=desc, parameters=params) for name, desc in specs
    ]


def _function_call_from_response(response) -> tuple[str, dict]:
    cands = getattr(response, "candidates", None) or []
    for cand in cands:
        content = getattr(cand, "content", None)
        parts = getattr(content, "parts", None) or []
        for part in parts:
            fc = getattr(part, "function_call", None)
            if not fc or not getattr(fc, "name", None):
                continue
            raw_args = getattr(fc, "args", None) or {}
            try:
                args = dict(raw_args)
            except Exception:
                args = {}
            return str(fc.name), args
    raise ValueError("No function call in model response")


async def classify_with_tools(
    *,
    question: str,
    partial: str,
    utterance: str,
    interrupted: bool,
    playback_pct: float,
    followup_used: bool = False,
    remaining_min: float = 10.0,
    timeout: int = 8,
) -> dict:
    from google import genai
    from google.genai import types

    prompt = TURN_ROUTER_PROMPT.format(
        question=question or "",
        partial=partial or "(empty)",
        utterance=utterance or "",
        interrupted="true" if interrupted else "false",
        playback_pct=f"{max(0.0, min(1.0, playback_pct)):.2f}",
        followup_used="true" if followup_used else "false",
        remaining_min=f"{max(0.0, remaining_min):.1f}",
    )
    model = gemini_model()
    client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
    config = types.GenerateContentConfig(
        tools=[types.Tool(function_declarations=_turn_tool_declarations())],
        tool_config=types.ToolConfig(
            function_calling_config=types.FunctionCallingConfig(
                mode="ANY",
                allowed_function_names=TURN_TOOL_NAMES,
            )
        ),
        temperature=0.0,
    )

    def _call():
        return client.models.generate_content(model=model, contents=prompt, config=config)

    response = await asyncio.wait_for(asyncio.to_thread(_call), timeout=timeout)
    name, args = _function_call_from_response(response)
    decision = decision_from_tool(name, args)
    return _normalize_decision(decision, utterance, interrupted, playback_pct)


async def classify_with_llm(
    *,
    question: str,
    partial: str,
    utterance: str,
    interrupted: bool,
    playback_pct: float,
    followup_used: bool = False,
    remaining_min: float = 10.0,
    timeout: int = 8,
) -> dict:
    return await classify_with_tools(
        question=question,
        partial=partial,
        utterance=utterance,
        interrupted=interrupted,
        playback_pct=playback_pct,
        followup_used=followup_used,
        remaining_min=remaining_min,
        timeout=timeout,
    )


async def decide_turn(
    *,
    question: str,
    partial: str,
    utterance: str,
    interrupted: bool,
    playback_pct: float,
    followup_used: bool = False,
    remaining_min: float = 10.0,
) -> dict:
    ruled = classify_with_rules(
        utterance,
        interrupted=interrupted,
        playback_pct=playback_pct,
        followup_used=followup_used,
        remaining_min=remaining_min,
    )
    if ruled is not None:
        logger.info(f"TurnRouter rules → {ruled['action']} ({ruled['reason']}) tool={ruled.get('tool')}")
        return ruled
    try:
        decision = await classify_with_tools(
            question=question,
            partial=partial,
            utterance=utterance,
            interrupted=interrupted,
            playback_pct=playback_pct,
            followup_used=followup_used,
            remaining_min=remaining_min,
        )
        logger.info(f"TurnRouter tool → {decision['action']} ({decision['reason']}) tool={decision.get('tool')}")
        return decision
    except Exception as e:
        logger.warning(f"TurnRouter tools failed, using fallback: {e}")
        return _fallback_decision(
            interrupted=interrupted,
            playback_pct=playback_pct,
            utterance=utterance,
        )
