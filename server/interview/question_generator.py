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

            logger.info(
                "LLM response generation completed"
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
    questions_data = parse_json_response(raw)
    return questions_data["questions"]
