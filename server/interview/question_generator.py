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
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask

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
async def generate_questions(
    resume: str,
    job_description: str,
):

    logger.info(
        "Starting question generation"
    )

    prompt = QUESTION_GENERATION_PROMPT.format(
        resume=resume,
        job_description=job_description,
    )

    llm = GoogleLLMService(
        api_key=os.getenv("GOOGLE_API_KEY"),
        settings=GoogleLLMService.Settings(
            model="gemma-4-26b-a4b-it",
        ),
    )

    collector = ResponseCollector()

    pipeline = Pipeline(
        [
            llm,
            collector,
        ]
    )

    task = PipelineTask(
        pipeline
    )

    runner = PipelineRunner()

    context = LLMContext(
    messages=[
        {
            "role": "user",
            "content": prompt,
        }
    ]
)
    logger.info(
        "Sending prompt to LLM"
    )

    await task.queue_frame(
    LLMContextFrame(
        context=context
    )
)

    await task.queue_frame(
        EndFrame()
    )

    await runner.run(task)

    logger.info(
        f"Raw LLM response:\n"
        f"{collector.response}"
    )

    if not collector.response or not collector.response.strip():
        raise RuntimeError("Empty LLM response for question generation")

    questions_data = parse_json_response(
    collector.response
)

    return questions_data["questions"]

