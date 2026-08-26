import os

from dotenv import load_dotenv
from loguru import logger


from prompt import SYSTEM_PROMPT_TEMPLATE
from document_loader import (
    load_resume,
    load_job_description,
)
from question_generator import generate_questions
from interview_manager import InterviewManager

from pipecat.frames.frames import TTSSpeakFrame

from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import (
    PipelineParams,
    PipelineTask,
)

from pipecat.processors.aggregators.llm_context import (
    LLMContext,
)
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
)

from pipecat.services.google.llm import GoogleLLMService

from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.deepgram.tts import DeepgramTTSService

from pipecat.transports.base_transport import (
    TransportParams,
)
from pipecat.transports.smallwebrtc.transport import (
    SmallWebRTCTransport,
)

from pipecat.runner.types import (
    RunnerArguments,
    SmallWebRTCRunnerArguments,
)

from pipecat_whisker import WhiskerObserver

from pipecat.services.tts_service import (
    TextAggregationMode,
)

load_dotenv()

from interview_processor import (
    InterviewProcessor,
)

async def run_bot(transport):

    logger.info(
        "Starting AI Interview Agent..."
    )

    # --------------------------------
    # Load documents
    # --------------------------------

    resume = load_resume()

    job_description = (
        load_job_description()
    )

    logger.info(
        "Resume and Job Description loaded"
    )

    # --------------------------------
    # Generate interview questions
    # --------------------------------

    logger.info(
        "Generating interview questions..."
    )

    questions = generate_questions(
        resume=resume,
        job_description=job_description,
    )

    logger.info(
        f"Generated {len(questions)} questions"
    )

    for question in questions:

        logger.info(
            f"Q{question['id']}: "
            f"{question['question']}"
        )

    # --------------------------------
    # Create Interview Manager
    # --------------------------------

    interview_manager = InterviewManager(
        questions
    )
    interview_processor = InterviewProcessor(
    interview_manager
    )
    # --------------------------------
    # Services
    # --------------------------------

    stt = DeepgramSTTService(
        api_key=os.getenv(
            "DEEPGRAM_API_KEY"
        ),
    )

    system_prompt = (
        SYSTEM_PROMPT_TEMPLATE.render()
    )

    llm = GoogleLLMService(
        api_key=os.getenv(
            "GOOGLE_API_KEY"
        ),
        settings=GoogleLLMService.Settings(
            model="gemma-4-26b-a4b-it",
            system_instruction=system_prompt,
        ),
    )

    tts = DeepgramTTSService(
        api_key=os.getenv(
            "DEEPGRAM_API_KEY"
        ),
        text_aggregation_mode=(
            TextAggregationMode.TOKEN
        ),
        settings=DeepgramTTSService.Settings(
            voice="aura-asteria-en",
        ),
    )

    # --------------------------------
    # Conversation Context
    # --------------------------------

    context = LLMContext(
        messages=[]
    )

    context_aggregator = (
        LLMContextAggregatorPair(
            context
        )
    )

    # --------------------------------
    # Pipeline
    # --------------------------------
    pipeline = Pipeline(
    [
        # Browser microphone
        transport.input(),

        # Speech -> Text
        stt,

        # Controls interview progression
        interview_processor,

        # Receives controlled LLM messages
        context_aggregator.user(),

        # LLM
        llm,

        # Text -> Speech
        tts,

        # Send audio to browser
        transport.output(),

        # Save assistant response
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
        WhiskerObserver(
            task.pipeline
        )
    )

    # --------------------------------
    # Client Connected
    # --------------------------------

    @transport.event_handler(
        "on_client_connected"
    )
    async def on_client_connected(
        transport,
        client,
    ):

        logger.info(
            "Candidate connected"
        )

        first_question = (
            interview_manager
            .get_current_question()
        )

        if first_question is None:

            logger.error(
                "No interview questions generated"
            )

            return

        intro = (
            "Hello. Welcome to your interview. "
            "I will ask you a series of questions "
            "based on your resume and the job description. "
            "Please answer each question before we move on. "
            "Let's begin. "
        )

        await task.queue_frame(
            TTSSpeakFrame(
                intro
                + first_question[
                    "question"
                ]
            )
        )

    # --------------------------------
    # Client Disconnected
    # --------------------------------

    @transport.event_handler(
        "on_client_disconnected"
    )
    async def on_client_disconnected(
        transport,
        client,
    ):

        logger.info(
            "Candidate disconnected"
        )

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