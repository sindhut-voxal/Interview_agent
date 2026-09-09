import os
import sys
from dotenv import load_dotenv
load_dotenv()

from loguru import logger

logger.remove()
logger.add(
    sys.stderr,
    level="INFO",
    filter=lambda record: "unable to append audio to context" not in record["message"]
    and "Data channel not established" not in record["message"],
)

from interview.controller import InterviewController
from interview_processor import InterviewProcessor

from pipecat.frames.frames import TTSSpeakFrame, OutputTransportMessageUrgentFrame

from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineWorker, PipelineParams
from pipecat.workers.runner import WorkerRunner

from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.deepgram.tts import DeepgramTTSService

from pipecat.transports.base_transport import TransportParams
from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport

from pipecat.runner.types import RunnerArguments, SmallWebRTCRunnerArguments
from pipecat.services.tts_service import TextAggregationMode

try:
    from pipecat_whisker import WhiskerObserver
except ImportError:
    WhiskerObserver = None  # type: ignore
    logger.warning("pipecat_whisker not installed — WhiskerObserver disabled (pip install pipecat-ai-whisker)")

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

import json
import pathlib


def _load_dynamic_docs():
    import tempfile
    candidate_paths = [
        pathlib.Path(__file__).parent / "latest_interview.json",
        pathlib.Path(tempfile.gettempdir()) / "latest_interview.json",
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
    env_resume = os.getenv("INTERVIEW_RESUME")
    env_jd = os.getenv("INTERVIEW_JD")
    if env_resume and env_jd:
        return env_resume, env_jd
    return DEFAULT_RESUME, DEFAULT_JOB_DESCRIPTION


async def run_bot(transport, resume: str | None = None, job_description: str | None = None, session_id: str | None = None):
    logger.info("Starting AI Interview Agent — 10-min Screening (voice) Mode — deterministic pipeline (STT→Processor→TTS, no live LLM)")

    controller = InterviewController()

    state = None
    if session_id:
        try:
            from session_store import load_session

            loaded = load_session(session_id)
            if loaded and loaded.questions:
                state = loaded
                logger.info(f"Loaded existing session {session_id} with {len(state.questions)} questions, {len(state.answers)} answers already stored")
            else:
                logger.info(f"No persisted session for {session_id}, will create new interview and save under that id")
        except Exception as e:
            logger.warning(f"Failed to load session {session_id}: {e}")

    if state is None:
        logger.info("Creating interview...")
        if resume is None or job_description is None:
            file_resume, file_jd = _load_dynamic_docs()
            resume = resume or file_resume
            job_description = job_description or file_jd

        resume = (resume or "")[:15000]
        job_description = (job_description or "")[:15000]

        state = await controller.create_interview(
            resume=resume,
            job_description=job_description,
        )
        if session_id:
            try:
                from session_store import save_session as _save2

                _save2(session_id, state)
                logger.info(f"Saved new interview under supplied session_id {session_id}")
            except Exception as e:
                logger.warning(f"Failed to save new session {session_id}: {e}")

    logger.info(f"Generated {len(state.questions)} questions")

    for question in state.questions:
        logger.info(f"Q{question['id']}: {question['question']}")

    interview_processor = InterviewProcessor(controller=controller, state=state, session_id=session_id)

    stt = DeepgramSTTService(
        api_key=os.getenv("DEEPGRAM_API_KEY"),
        settings=DeepgramSTTService.Settings(
            model="nova-3",
            language="en",
            interim_results=True,
            endpointing=400,
            smart_format=True,
            punctuate=True,
        ),
    )

    tts = DeepgramTTSService(
        api_key=os.getenv("DEEPGRAM_API_KEY"),
        text_aggregation_mode=TextAggregationMode.SENTENCE,
        settings=DeepgramTTSService.Settings(
            voice="aura-2-juno-en",
        ),
    )

    # Deterministic pipeline: no live Gemini, no LLMContext, no aggregators, no text normaliser frame.
    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            interview_processor,
            tts,
            transport.output(),
        ]
    )
    worker = PipelineWorker(pipeline, params=PipelineParams(allow_interruptions=False, enable_metrics=True, enable_usage_metrics=True))

    if WhiskerObserver is not None:
        try:
            worker.add_observer(WhiskerObserver(worker.pipeline))
            logger.info("WhiskerObserver attached at ws://localhost:9090")
        except Exception as e:
            logger.warning(f"Failed to attach WhiskerObserver: {e}")
    else:
        logger.info("WhiskerObserver skipped (not installed)")

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("Candidate connected")

        first_question = state.get_current_question()
        if first_question is None:
            logger.error("No interview questions generated")
            return
        # Q1: Speak ONLY the Q1 text as ONE atomic utterance via TTSSpeakFrame.
        # No intro filler, no prepended transition. Natural pause before answer
        # is provided by TTS completion + InterviewProcessor's 200ms guard.
        try:
            from text_normaliser import normalize_for_tts as _norm
        except Exception:
            _norm = lambda x: x.strip()
        q1_text = _norm(first_question["question"])
        logger.info(f"Queueing Q1 as atomic TTSSpeakFrame ({len(q1_text)} chars)")
        await worker.queue_frame(
            OutputTransportMessageUrgentFrame(
                message={
                    "type": "interview_progress",
                    "done": False,
                    "current_index": state.current_question_index,
                    "total": len(state.questions),
                    "question": first_question.get("question"),
                    "question_id": first_question.get("id"),
                }
            )
        )
        await worker.queue_frame(TTSSpeakFrame(text=q1_text, append_to_context=False))

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Candidate disconnected")

    runner = WorkerRunner()
    await runner.add_workers(worker)
    await runner.run()


async def bot(runner_args: RunnerArguments):
    if isinstance(runner_args, SmallWebRTCRunnerArguments):
        body = getattr(runner_args, "body", None) or {}
        request_data = body.get("request_data") if isinstance(body, dict) and "request_data" in body else body
        if isinstance(body, dict) and "body" in body and isinstance(body["body"], dict):
            inner_body = body["body"]
            if isinstance(request_data, dict):
                for k, v in inner_body.items():
                    request_data.setdefault(k, v)
            else:
                request_data = inner_body

        if isinstance(request_data, dict):
            b_resume = request_data.get("resume") or request_data.get("resume_text") or request_data.get("resumeText")
            b_jd = request_data.get("job_description") or request_data.get("jd") or request_data.get("jd_text") or request_data.get("jdText")
            b_session_id = request_data.get("session_id") or request_data.get("sessionId")
        elif isinstance(body, dict):
            b_resume = body.get("resume") or body.get("resume_text") or body.get("resumeText")
            b_jd = body.get("job_description") or body.get("jd") or body.get("jd_text") or body.get("jdText")
            b_session_id = body.get("session_id") or body.get("sessionId")
        else:
            b_resume = b_jd = b_session_id = None
        if isinstance(body, dict) and "body" in body and isinstance(body["body"], dict):
            inner = body["body"]
            b_resume = b_resume or inner.get("resume")
            b_jd = b_jd or inner.get("job_description") or inner.get("jd")
            b_session_id = b_session_id or inner.get("session_id") or inner.get("sessionId")

        logger.info(f"Offer request_data session_id={b_session_id} resume_present={bool(b_resume)} jd_present={bool(b_jd)}")

        transport = SmallWebRTCTransport(
            params=TransportParams(audio_in_enabled=True, audio_out_enabled=True),
            webrtc_connection=(runner_args.webrtc_connection),
        )
        await run_bot(transport, resume=b_resume, job_description=b_jd, session_id=b_session_id)
        return
    else:
        logger.error(f"Unsupported runner arguments: {type(runner_args)}")
        return


try:
    from pipecat.runner.run import app as _runner_app
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import FileResponse
    from fastapi import UploadFile, File
    import pathlib as _pl

    try:
        _runner_app.add_middleware(
            CORSMiddleware,
            allow_origins=[
                "http://localhost:8000",
                "http://127.0.0.1:8000",
                "http://localhost:3000",
                "http://127.0.0.1:3000",
            ],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    except Exception:
        pass

    try:
        from fastapi.responses import JSONResponse

        @_runner_app.options("/api/offer", include_in_schema=False)
        async def _offer_options():
            return JSONResponse(content={}, status_code=200)
    except Exception:
        pass

    _CLIENT_DIR = _pl.Path(__file__).parent.parent / "client"
    if _CLIENT_DIR.exists() and _CLIENT_DIR.joinpath("index.html").exists():

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
                    if text.strip():
                        return {"text": text.strip(), "filename": file.filename}
                    return {"text": "", "filename": file.filename, "error": "Could not extract text from PDF."}
                except Exception as e:
                    return {"text": "", "filename": file.filename, "error": f"Could not extract text from PDF: {e}"}
            if suffix == ".docx":
                try:
                    import docx

                    doc = docx.Document(io.BytesIO(data))
                    text = "\n".join([p.text for p in doc.paragraphs])
                    return {"text": text.strip(), "filename": file.filename}
                except Exception as e:
                    return {"text": "", "filename": file.filename, "error": f"Could not extract text from DOCX: {e}"}
            if suffix == ".doc":
                return {"text": "", "filename": file.filename, "error": ".doc (legacy Word) is not supported — please upload .docx or PDF."}
            try:
                return {"text": data.decode("utf-8"), "filename": file.filename}
            except Exception:
                return {"text": data.decode("utf-8", errors="ignore"), "filename": file.filename}

        @_runner_app.get("/api/health", include_in_schema=False)
        async def _health_runner():
            return {"status": "ok", "mode": "voice-pipeline", "pipeline": "SmallWebRTC→DeepgramSTT(700ms)→InterviewProcessor→DeepgramTTS"}

except Exception as _e:
    logger.warning(f"Could not mount custom UI on runner: {_e}")

if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
