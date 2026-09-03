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


def _fallback_evaluation(question: dict, answer: str) -> dict:
    """Heuristic fallback when LLM is unavailable or returns invalid JSON."""
    weight = int(question.get("weight", 10))
    ans = (answer or "").strip()
    qid = question.get("id", 0)
    if len(ans) < 20:
        score = max(1, weight // 3)
        feedback = "Thanks — try adding a bit more detail next time."
    elif len(ans) < 80:
        score = int(weight * 0.6)
        feedback = "Good start — you covered the basics clearly."
    else:
        score = int(weight * 0.85)
        feedback = "Nice — you explained it clearly and concisely."
    return {
        "question_id": qid,
        "score": score,
        "feedback": feedback,
        "strengths": [],
        "improvements": ["Could add more detail"],
    }


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
            model="gemini-3.6-flash",
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

    try:
        await runner.run(task)
    except Exception as e:
        logger.warning(f"LLM runner failed for evaluation: {e}")
        return _fallback_evaluation(question, answer)

    logger.info(
        f"Raw evaluation response:\n"
        f"{collector.response}"
    )

    if not collector.response or not collector.response.strip():
        logger.warning("Empty LLM response for evaluation — using fallback")
        return _fallback_evaluation(question, answer)

    try:
        evaluation = parse_json_response(
            collector.response
        )
    except Exception as e:
        logger.warning(f"Failed to parse LLM evaluation JSON: {e} — using fallback. Raw: {collector.response[:500]}")
        return _fallback_evaluation(question, answer)

    # Validate required fields, fallback if malformed
    if not isinstance(evaluation, dict) or "score" not in evaluation or "feedback" not in evaluation:
        logger.warning(f"Malformed evaluation JSON — using fallback: {evaluation}")
        return _fallback_evaluation(question, answer)

    # Clamp score to weight
    try:
        evaluation["score"] = max(0, min(int(evaluation["score"]), int(question.get("weight", 100))))
    except Exception:
        pass

    return evaluation