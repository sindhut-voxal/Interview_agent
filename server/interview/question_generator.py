import json
import os
import re

from dotenv import load_dotenv
from loguru import logger

from pipecat.frames.frames import (
    EndFrame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMTextFrame,
)

from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineWorker
from pipecat.workers.runner import WorkerRunner

from pipecat.processors.frame_processor import (
    FrameDirection,
    FrameProcessor,
)
from pipecat.processors.aggregators.llm_context import (
    LLMContext,
)

from pipecat.services.google.llm import GoogleLLMService

from interview.prompts import QUESTION_GENERATION_PROMPT


load_dotenv()

import asyncio


class ResponseCollector(FrameProcessor):

    def __init__(self):
        super().__init__()

        self.response = ""

        self.response_complete = asyncio.Event()

    async def process_frame(
        self,
        frame,
        direction: FrameDirection,
    ):
        await super().process_frame(
            frame,
            direction,
        )

        if isinstance(frame, LLMTextFrame):

            self.response += frame.text

        elif isinstance(
            frame,
            LLMFullResponseEndFrame,
        ):

            logger.info("LLM response generation completed")

            self.response_complete.set()

        await self.push_frame(
            frame,
            direction,
        )


def parse_json_response(response: str):
    """Robust JSON extraction: handles markdown fences, preamble, and extracts first JSON object."""
    raw = response.strip()
    # Try to extract fenced JSON block (case-insensitive)
    fence_match = re.search(r"```(?:json)?\s*(.*?)\s*```", raw, re.DOTALL | re.IGNORECASE)
    if fence_match:
        raw = fence_match.group(1).strip()
    else:
        # No fence: try to find first {...} JSON object
        # If response has preamble like "Here is the JSON:", extract from first { to last }
        first_brace = raw.find("{")
        last_brace = raw.rfind("}")
        if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
            candidate = raw[first_brace : last_brace + 1]
            try:
                return json.loads(candidate)
            except Exception:
                pass  # fall through to direct parse
    raw = raw.strip()
    # strip leftover fence markers
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?", "", raw, flags=re.IGNORECASE).strip()
    if raw.endswith("```"):
        raw = raw[:-3].strip()
    return json.loads(raw)


def validate_questions(data) -> list:
    if not isinstance(data, dict):
        raise ValueError("Question response must be an object")
    questions = data.get("questions")
    if not isinstance(questions, list):
        raise ValueError("Missing questions list")
    if len(questions) != 6:
        raise ValueError(f"Expected 6 questions, got {len(questions)}")
    ids = [q.get("id") for q in questions]
    if len(set(ids)) != 6:
        raise ValueError("Question IDs must be unique")
    # ensure ids are 1..6 (allow any 6 unique but warn if not 1..6)
    try:
        if set(int(x) for x in ids) != {1, 2, 3, 4, 5, 6}:
            logger.warning(f"Question IDs are not 1..6: {ids}")
    except Exception:
        pass
    required = {"id", "question", "skill", "criteria", "weight"}
    for q in questions:
        if not isinstance(q, dict) or not required.issubset(q):
            raise ValueError(f"Malformed question missing fields: {q}")
        if not isinstance(q["criteria"], list) or not (2 <= len(q["criteria"]) <= 4):
            raise ValueError(f"Question criteria must be 2-4 items: {q}")
        try:
            w = int(q["weight"])
            if w <= 0:
                raise ValueError
        except Exception:
            raise ValueError(f"Invalid weight: {q.get('weight')}")
    total = sum(int(q["weight"]) for q in questions)
    if total != 100:
        raise ValueError(f"Question weights must sum to 100, got {total}")
    return questions


async def _generate_once(prompt: str, model: str, timeout: int = 30) -> str:
    """Single-attempt generation for a given model."""
    llm = GoogleLLMService(
        api_key=os.getenv("GOOGLE_API_KEY"),
        settings=GoogleLLMService.Settings(model=model),
    )
    collector = ResponseCollector()
    pipeline = Pipeline([llm, collector])
    worker = PipelineWorker(pipeline)
    runner = WorkerRunner()
    context = LLMContext(messages=[{"role": "user", "content": prompt}])
    await worker.queue_frame(LLMContextFrame(context=context))
    await worker.queue_frame(EndFrame())
    await runner.add_workers(worker)
    await asyncio.wait_for(runner.run(), timeout=timeout)
    if not collector.response or not collector.response.strip():
        raise RuntimeError("Empty LLM response")
    return collector.response


async def generate_questions(
    resume: str,
    job_description: str,
):
    logger.info("Starting question generation")
    prompt = QUESTION_GENERATION_PROMPT.format(resume=resume, job_description=job_description)
    model = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")

    logger.info(f"QuestionGen model={model}")
    raw = await _generate_once(prompt, model, timeout=30)
    logger.info(f"Raw LLM response (model={model}):\n{raw}")
    data = parse_json_response(raw)
    questions = validate_questions(data)
    return questions
