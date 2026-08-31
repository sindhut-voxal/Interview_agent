import os
from dotenv import load_dotenv
load_dotenv()

#python library for logs
from loguru import logger

from prompt import SYSTEM_PROMPT_TEMPLATE

#handles state
from interview.controller import InterviewController
#controls what goes to the LLM
from interview_processor import InterviewProcessor

from pipecat.frames.frames import TTSSpeakFrame

#connects different frame processors together
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import (PipelineParams,PipelineTask)

#stores context for LLM
from pipecat.processors.aggregators.llm_context import (LLMContext)

from pipecat.processors.aggregators.llm_response_universal import (LLMContextAggregatorPair)

#STT,LLM,TTS
from pipecat.services.google.llm import GoogleLLMService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.deepgram.tts import DeepgramTTSService

from pipecat.transports.base_transport import (TransportParams)
from pipecat.transports.smallwebrtc.transport import (SmallWebRTCTransport)

from pipecat.runner.types import (RunnerArguments, SmallWebRTCRunnerArguments)

#Debugger for pipecat-visualise the flow of each frame
from pipecat_whisker import WhiskerObserver

from pipecat.services.tts_service import (TextAggregationMode)
from text_normaliser import (TextNormalizerProcessor)

RESUME = """
AI Engineer Intern

Skills:
Python, Machine Learning, Deep Learning,
Generative AI, FastAPI, Docker

Projects:

1. AI Language Tutor
Built a real-time AI language tutor using
Pipecat, Deepgram, Gemini and WebRTC.

2. Interview Agent
Built an AI-powered interview system capable
of asking questions and evaluating candidate
responses.
"""


JOB_DESCRIPTION = """
We are looking for an AI Engineer.

Required skills:

- Strong Python programming
- Experience with Large Language Models
- Machine Learning fundamentals
- API development
- FastAPI
- Docker
"""


async def run_bot(transport):
    logger.info("Starting AI Interview Agent")
    logger.info("Creating interview...")

    controller = InterviewController()

    state = await controller.create_interview(
        resume=RESUME,
        job_description=JOB_DESCRIPTION,
    )

    logger.info(f"Generated {len(state.questions)} questions")

    for question in state.questions:
        logger.info(
            f"Q{question['id']}: "
            f"{question['question']}"
        )

    interview_processor = InterviewProcessor(controller=controller, state=state)

    stt = DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY"))

    system_prompt = (SYSTEM_PROMPT_TEMPLATE.render())

    llm = GoogleLLMService(
        api_key=os.getenv("GOOGLE_API_KEY"),
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
            voice="aura-2-juno-en",
        ),
    )
    context = LLMContext(messages=[])
    context_aggregator = (LLMContextAggregatorPair(context))

    text_normalizer = (TextNormalizerProcessor())

    pipeline = Pipeline(
    [
        transport.input(),
        stt,
        interview_processor,
        context_aggregator.user(),
        llm,
        text_normalizer,
        tts,
        transport.output(),
        context_aggregator.assistant(),
    ]
)
    task = PipelineTask(pipeline,params=PipelineParams(allow_interruptions=True,enable_metrics=True,enable_usage_metrics=True))

    task.add_observer(WhiskerObserver(task.pipeline))


    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport,client):
        logger.info("Candidate connected")

        first_question = (state.get_current_question())
        if first_question is None:
            logger.error("No interview questions generated")
            return
        intro = (
            "Hello. Welcome to your interview. "
            "I will ask you a series of questions "
            "based on your resume and the job description. "
            "Please answer each question before we move on. "
            "Let's begin. "
        )
    #this skips the LLM and speaks the text given using the TTS service.
        await task.queue_frame(TTSSpeakFrame(intro + first_question["question"]))

    @transport.event_handler(
        "on_client_disconnected"
    )
    async def on_client_disconnected(transport,client):
        logger.info("Candidate disconnected")
    runner = PipelineRunner()
    await runner.run(task)

async def bot(runner_args: RunnerArguments):
    if isinstance(runner_args,SmallWebRTCRunnerArguments):
        transport = SmallWebRTCTransport(
            params=TransportParams(audio_in_enabled=True,audio_out_enabled=True),
            webrtc_connection=(runner_args.webrtc_connection),
)
    else:
        logger.error( f"Unsupported runner arguments: "f"{type(runner_args)}")
        return
    await run_bot(transport)


if __name__ == "__main__":
    from pipecat.runner.run import main
    main()