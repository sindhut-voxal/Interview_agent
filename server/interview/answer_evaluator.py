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


async def evaluate_answer(
    question: dict,
    answer: str,
) -> dict:

    logger.info(
        "Starting answer evaluation"
    )

    prompt = ANSWER_EVALUATION_PROMPT.format(
        question_id=question["id"],
        question=question["question"],
        skill=question["skill"],
        criteria=json.dumps(
            question["criteria"],
            indent=2,
        ),
        weight=question["weight"],
        answer=answer,
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
        "Sending evaluation prompt to LLM"
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
        f"Raw evaluation response:\n"
        f"{collector.response}"
    )

    evaluation = parse_json_response(
        collector.response
    )

    return evaluation