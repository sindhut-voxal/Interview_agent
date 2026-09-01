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

DEFAULT_RESUME = """
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


DEFAULT_JOB_DESCRIPTION = """
We are looking for an AI Engineer.

Required skills:

- Strong Python programming
- Experience with Large Language Models
- Machine Learning fundamentals
- API development
- FastAPI
- Docker
"""

# Optional: if API server wrote a latest session file, use it for dynamic resume/JD
import json
import pathlib

def _load_dynamic_docs():
    candidate_paths = [
        pathlib.Path(__file__).parent / "latest_interview.json",
        pathlib.Path("/tmp/latest_interview.json"),
        pathlib.Path(__file__).parent / ".." / "latest_interview.json",
    ]
    for p in candidate_paths:
        try:
            if p.exists():
                data = json.loads(p.read_text(encoding="utf-8"))
                r = data.get("resume")
                j = data.get("job_description")
                if r and j:
                    logger.info(f"Loaded dynamic resume/JD from {p}")
                    return r, j
        except Exception as e:
            logger.warning(f"Failed to load dynamic docs from {p}: {e}")
    # Also check env overrides
    env_resume = os.getenv("INTERVIEW_RESUME")
    env_jd = os.getenv("INTERVIEW_JD")
    if env_resume and env_jd:
        return env_resume, env_jd
    return DEFAULT_RESUME, DEFAULT_JOB_DESCRIPTION


async def run_bot(transport, resume: str | None = None, job_description: str | None = None):
    logger.info("Starting AI Interview Agent — 10-min Screening (voice) Mode")
    logger.info("Creating interview...")

    controller = InterviewController()

    # Priority: 1) per-connection body (from WebRTC offer), 2) latest file, 3) defaults
    if resume is None or job_description is None:
        file_resume, file_jd = _load_dynamic_docs()
        resume = resume or file_resume
        job_description = job_description or file_jd

    # Safety trim for prompt
    resume = (resume or "")[:15000]
    job_description = (job_description or "")[:15000]

    # Fallback if LLM unavailable — still produce voice flow with static questions
    try:
        state = await controller.create_interview(
            resume=resume,
            job_description=job_description,
        )
    except Exception as e:
        logger.warning(f"LLM generation failed, using fallback voice questions: {e}")
        from interview.state import InterviewState
        fallback = [
            {"id": 1, "question": "Could you briefly introduce yourself and walk me through your background?", "skill": "Introduction", "criteria": ["Clear summary"], "weight": 15},
            {"id": 2, "question": "You mentioned a project on your resume — could you briefly explain one you enjoyed and your role in it?", "skill": "Project Experience", "criteria": ["Explains role"], "weight": 20},
            {"id": 3, "question": "What core skills from the job description are you most comfortable with and why?", "skill": "Role Fit", "criteria": ["Aligns with JD"], "weight": 15},
            {"id": 4, "question": "In your own words, how would you explain a function versus a class to a junior developer?", "skill": "Fundamentals", "criteria": ["Clear definition"], "weight": 15},
            {"id": 5, "question": "Tell me about a time you debugged a tricky issue — what was the problem and how did you solve it?", "skill": "Problem Solving", "criteria": ["Structured story"], "weight": 15},
            {"id": 6, "question": "What would you most like to learn in your first months in this role?", "skill": "Motivation", "criteria": ["Growth mindset"], "weight": 20},
        ]
        state = InterviewState(resume=resume, job_description=job_description, questions=fallback)

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
            "Hello. Welcome to your screening interview. "
            "This is a quick 10 minute first round to get to know you. "
            "I'll ask about 6 basic questions based on your resume and the job description. "
            "After each answer I'll share a quick thought and we'll move to the next question. "
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
        # Extract per-connection resume/JD sent by the custom UI via offer request_data
        body = getattr(runner_args, "body", None) or {}
        # body may be dict with resume/jd or nested inside
        if isinstance(body, dict):
            b_resume = body.get("resume") or body.get("resume_text") or body.get("resumeText")
            b_jd = body.get("job_description") or body.get("jd") or body.get("jd_text") or body.get("jdText")
        else:
            b_resume = b_jd = None
        # Also support nested body key (Pipecat runner nests under body)
        if isinstance(body, dict) and "body" in body and isinstance(body["body"], dict):
            inner = body["body"]
            b_resume = b_resume or inner.get("resume")
            b_jd = b_jd or inner.get("job_description") or inner.get("jd")

        transport = SmallWebRTCTransport(
            params=TransportParams(audio_in_enabled=True,audio_out_enabled=True),
            webrtc_connection=(runner_args.webrtc_connection),
        )
        await run_bot(transport, resume=b_resume, job_description=b_jd)
        return
    else:
        logger.error( f"Unsupported runner arguments: "f"{type(runner_args)}")
        return


# Serve custom UI + extract API on the runner's FastAPI app (so 7860 is self-contained)
try:
    from pipecat.runner.run import app as _runner_app
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import FileResponse
    from fastapi import UploadFile, File
    import pathlib as _pl
    _CLIENT_DIR = _pl.Path(__file__).parent.parent / "client"
    if _CLIENT_DIR.exists() and _CLIENT_DIR.joinpath("index.html").exists():
        try:
            _runner_app.add_middleware(
                CORSMiddleware,
                allow_origins=["*"],
                allow_credentials=True,
                allow_methods=["*"],
                allow_headers=["*"],
            )
        except Exception:
            pass  # middleware may already be added
        @_runner_app.get("/app", include_in_schema=False)
        async def _serve_custom():
            return FileResponse(str(_CLIENT_DIR / "index.html"))
        @_runner_app.post("/api/extract", include_in_schema=False)
        async def _extract_runner(file: UploadFile = File(...)):
            from pypdf import PdfReader
            import io, pathlib as _p
            data = await file.read()
            suffix = _p.Path(file.filename or "").suffix.lower()
            if suffix == ".pdf":
                try:
                    r = PdfReader(io.BytesIO(data))
                    text = "\n".join([p.extract_text() or "" for p in r.pages])
                    return {"text": text.strip(), "filename": file.filename}
                except Exception:
                    pass
            return {"text": data.decode("utf-8", errors="ignore"), "filename": file.filename}
        @_runner_app.get("/api/health", include_in_schema=False)
        async def _health_runner():
            return {"status": "ok", "mode": "voice-pipeline", "pipeline": "SmallWebRTC→DeepgramSTT→InterviewProcessor(feedback)→Gemini→DeepgramTTS"}
except Exception as _e:
    logger.warning(f"Could not mount custom UI on runner: {_e}")

if __name__ == "__main__":
    from pipecat.runner.run import main
    main()