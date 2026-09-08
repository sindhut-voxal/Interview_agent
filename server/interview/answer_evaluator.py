import asyncio
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

from interview.prompts import ANSWER_EVALUATION_PROMPT


load_dotenv()


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

            logger.info("LLM answer evaluation completed")

            self.response_complete.set()

        await self.push_frame(
            frame,
            direction,
        )


def parse_json_response(response: str):

    raw = response.strip()
    fence_match = re.search(r"```(?:json)?\s*(.*?)\s*```", raw, re.DOTALL | re.IGNORECASE)
    if fence_match:
        raw = fence_match.group(1).strip()
    else:
        first_brace = raw.find("{")
        last_brace = raw.rfind("}")
        if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
            candidate = raw[first_brace : last_brace + 1]
            try:
                return json.loads(candidate)
            except Exception:
                pass
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?", "", raw, flags=re.IGNORECASE).strip()
    if raw.endswith("```"):
        raw = raw[:-3].strip()
    return json.loads(raw)


async def _evaluate_once(prompt: str, model: str, timeout: int = 20) -> str:
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


async def evaluate_answer(
    question: dict,
    answer: str,
) -> dict:
    logger.info("Starting answer evaluation")
    prompt = ANSWER_EVALUATION_PROMPT.format(
        question_id=question["id"],
        question=question["question"],
        skill=question["skill"],
        criteria=json.dumps(question["criteria"], indent=2),
        weight=question["weight"],
        answer=answer,
    )
    model = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")

    logger.info(f"Evaluate model={model}")
    raw = await _evaluate_once(prompt, model, timeout=20)

    logger.info(f"Raw evaluation response:\n{raw}")

    evaluation = parse_json_response(raw)

    if not isinstance(evaluation, dict):
        raise ValueError(f"Malformed evaluation JSON: {evaluation}")
    for field in ("question_id", "score", "feedback", "strengths", "improvements"):
        if field not in evaluation:
            raise ValueError(f"Missing field '{field}' in evaluation: {evaluation}")
    # score validation + clamping
    try:
        score = int(evaluation["score"])
    except Exception:
        raise ValueError(f"Invalid score: {evaluation.get('score')}")
    weight = int(question.get("weight", 100))
    score = max(0, min(score, weight))
    evaluation["score"] = score
    # ensure strengths/improvements are lists
    if not isinstance(evaluation.get("strengths"), list):
        evaluation["strengths"] = []
    if not isinstance(evaluation.get("improvements"), list):
        evaluation["improvements"] = []
    if not isinstance(evaluation.get("feedback"), str):
        evaluation["feedback"] = str(evaluation.get("feedback", ""))
    evaluation["question_id"] = int(question["id"])
    return evaluation
