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
from interview.llm_config import gemini_model


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


def _parse_score(raw, weight: int) -> int:
    if isinstance(raw, bool):
        raise ValueError(f"Invalid score: {raw}")
    if isinstance(raw, (int, float)):
        score = int(round(float(raw)))
    else:
        text = str(raw).strip()
        match = re.match(r"(-?\d+(?:\.\d+)?)", text)
        if not match:
            raise ValueError(f"Invalid score: {raw}")
        score = int(round(float(match.group(1))))
    return max(0, min(score, weight))


async def _evaluate_once(prompt: str, model: str, timeout: int = 45) -> str:
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
    model = gemini_model()
    weight = int(question.get("weight", 100))
    last_error = None
    for attempt in range(3):
        try:
            logger.info(f"Evaluate model={model} attempt={attempt + 1} Q{question.get('id')}")
            raw = await _evaluate_once(prompt, model, timeout=45)
            logger.info(f"Raw evaluation response:\n{raw}")
            evaluation = parse_json_response(raw)
            if not isinstance(evaluation, dict):
                raise ValueError(f"Malformed evaluation JSON: {evaluation}")
            for field in ("question_id", "score", "feedback", "strengths", "improvements"):
                if field not in evaluation:
                    raise ValueError(f"Missing field '{field}' in evaluation: {evaluation}")
            evaluation["score"] = _parse_score(evaluation["score"], weight)
            if not isinstance(evaluation.get("strengths"), list):
                evaluation["strengths"] = []
            if not isinstance(evaluation.get("improvements"), list):
                evaluation["improvements"] = []
            if not isinstance(evaluation.get("feedback"), str):
                evaluation["feedback"] = str(evaluation.get("feedback", ""))
            evaluation["question_id"] = int(question["id"])
            evaluation.pop("error", None)
            return evaluation
        except Exception as e:
            last_error = e
            logger.warning(f"Evaluate attempt {attempt + 1} failed for Q{question.get('id')}: {e}")
            await asyncio.sleep(0.4 * (attempt + 1))
    raise last_error
