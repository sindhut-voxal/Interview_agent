"""Fast spoken-line composer for the live cascade (not scoring)."""
from __future__ import annotations

import asyncio
import os

from loguru import logger

from interview.llm_config import gemini_model

FOLLOWUP_PROMPT = """You are a friendly screening interviewer. The candidate gave a short answer.
Ask ONE brief spoken follow-up (under 16 words). Do not praise, score, or change topics.

QUESTION: {question}
ANSWER SO FAR: {answer}

Return only the spoken sentence.
"""

PARAPHRASE_PROMPT = """Rephrase this screening question in simpler spoken English.
Keep the same meaning. One sentence, under 18 words. No preamble.

QUESTION: {question}
"""


async def _compose(prompt: str, timeout: float = 5.0) -> str:
    from google import genai

    model = gemini_model()
    client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

    def _call():
        return client.models.generate_content(model=model, contents=prompt)

    response = await asyncio.wait_for(asyncio.to_thread(_call), timeout=timeout)
    text = (getattr(response, "text", None) or "").strip()
    text = " ".join(text.split())
    if text.startswith('"') and text.endswith('"'):
        text = text[1:-1].strip()
    if not text or len(text.split()) > 24:
        raise ValueError("Empty or too-long spoken line")
    return text


async def compose_followup(question: str, answer: str) -> str | None:
    try:
        return await _compose(FOLLOWUP_PROMPT.format(question=question, answer=answer or "(short)"))
    except Exception as e:
        logger.warning(f"compose_followup failed: {e}")
        return None


async def compose_paraphrase(question: str) -> str | None:
    try:
        return await _compose(PARAPHRASE_PROMPT.format(question=question))
    except Exception as e:
        logger.warning(f"compose_paraphrase failed: {e}")
        return None
