import asyncio
import json
import os

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

            logger.info(
                "LLM answer evaluation completed"
            )

            self.response_complete.set()

        await self.push_frame(
            frame,
            direction,
        )


def parse_json_response(response: str):

    response = response.strip()

    if response.startswith("```json"):
        response = response[len("```json"):]

    elif response.startswith("```"):
        response = response[len("```"):]

    if response.endswith("```"):
        response = response[:-3]

    response = response.strip()

    return json.loads(response)


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

    if not isinstance(evaluation, dict) or "score" not in evaluation or "feedback" not in evaluation:
        raise ValueError(f"Malformed evaluation JSON: {evaluation}")

    try:
        evaluation["score"] = max(0, min(int(evaluation["score"]), int(question.get("weight", 100))))
    except Exception:
        pass
    return evaluation
