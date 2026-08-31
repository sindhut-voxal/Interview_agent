import os

from dotenv import load_dotenv
from loguru import logger

from pipecat.frames.frames import LLMRunFrame

from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask

from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
)

from pipecat.services.google.gemini_live.llm import (
    GeminiLiveLLMService,
)

from pipecat.transports.base_transport import TransportParams
from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport

from pipecat.runner.types import (
    RunnerArguments,
    SmallWebRTCRunnerArguments,
)

from pipecat_whisker import WhiskerObserver


load_dotenv()


SYSTEM_INSTRUCTION = """
You are a professional AI interviewer.

You are conducting a technical interview for an AI Engineer Intern.

Be natural, conversational, and professional.

Ask one question at a time.
Keep your responses concise.
Do not use markdown.
Do not give long explanations.

For this initial test, simply have a natural conversation
with the candidate about their AI and ML experience.
"""


async def run_bot(transport):

    logger.info("Starting Gemini Live S2S test")

    gemini = GeminiLiveLLMService(
        api_key=os.getenv("GOOGLE_API_KEY"),
        settings=GeminiLiveLLMService.Settings(
            model="gemini-3.1-flash-live-preview",
            voice="Charon",
        ),
        system_instruction=SYSTEM_INSTRUCTION,
    )

    context = LLMContext(
        messages=[
            {
                "role": "user",
                "content": (
                    "Introduce yourself to the candidate "
                    "and ask them to tell you about themselves."
                ),
            }
        ]
    )

    context_aggregator = LLMContextAggregatorPair(
        context
    )

    pipeline = Pipeline(
        [
            transport.input(),

            context_aggregator.user(),

            gemini,

            transport.output(),

            context_aggregator.assistant(),
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            allow_interruptions=True,
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
    )

    task.add_observer(
        WhiskerObserver(task.pipeline)
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(
        transport,
        client,
    ):

        logger.info("Candidate connected")

        await task.queue_frame(
            LLMRunFrame()
        )

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(
        transport,
        client,
    ):

        logger.info("Candidate disconnected")

    runner = PipelineRunner()

    await runner.run(task)


async def bot(
    runner_args: RunnerArguments,
):

    if isinstance(
        runner_args,
        SmallWebRTCRunnerArguments,
    ):

        transport = SmallWebRTCTransport(
            params=TransportParams(
                audio_in_enabled=True,
                audio_out_enabled=True,
            ),
            webrtc_connection=(
                runner_args.webrtc_connection
            ),
        )

    else:

        logger.error(
            f"Unsupported runner arguments: "
            f"{type(runner_args)}"
        )

        return

    await run_bot(transport)


if __name__ == "__main__":

    from pipecat.runner.run import main

    main()