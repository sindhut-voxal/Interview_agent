import asyncio
import re

from loguru import logger

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    InterimTranscriptionFrame,
    TextFrame,
    TranscriptionFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    TTSSpeakFrame,
    OutputTransportMessageUrgentFrame,
)

from pipecat.processors.frame_processor import (
    FrameDirection,
    FrameProcessor,
)

from interview.controller import InterviewController
from interview.state import InterviewState
from session_store import save_session

# Inline lightweight normalization so we don't depend on LLMTextFrame path.
# Reuse text_normaliser.normalize_for_tts if available.
try:
    from text_normaliser import normalize_for_tts  # type: ignore
except Exception:
    def normalize_for_tts(text: str) -> str:  # fallback no-op
        return text.strip()


class InterviewProcessor(FrameProcessor):

    def __init__(
        self,
        controller: InterviewController,
        state: InterviewState,
        session_id: str | None = None,
        debounce_s: float = 1.8,
        min_chars: int = 3,
    ):
        super().__init__()

        self.controller = controller
        self.state = state
        self.session_id = session_id

        self.interview_finished = False
        # Buffer accumulates ONLY final transcripts. Interim is used solely to reset debounce.
        self._buffer: str = ""
        self._debounce_s = debounce_s
        self._min_chars = min_chars
        self._debounce_task: asyncio.Task | None = None
        self._processing: bool = False
        # Speaking/Listening state: question delivery must be atomic
        self._is_speaking: bool = True  # start speaking (intro Q1) until first TTSStoppedFrame + guard
        self._guard_task: asyncio.Task | None = None
        self._report_triggered: bool = False

    # ---------- debounce helpers ----------

    def _reset_debounce(self):
        """Reset the silence timer because candidate is still speaking (interim detected)."""
        if self._debounce_task and not self._debounce_task.done():
            self._debounce_task.cancel()
        self._debounce_task = asyncio.create_task(self._debounced_flush())

    async def _debounced_flush(self):
        try:
            await asyncio.sleep(self._debounce_s)
            await self._flush_buffer()
        except asyncio.CancelledError:
            pass

    async def _flush_buffer(self):
        """Store answer and advance. No LLM in live path."""
        if self.interview_finished:
            return
        answer = self._buffer.strip()
        self._buffer = ""
        if not answer:
            logger.info("STT buffer empty — ignoring")
            return
        if len(answer) < self._min_chars:
            logger.info(f"STT buffer below threshold ({len(answer)} chars) — still storing: '{answer}'")
        # Filler filter: single-word filler utterances should not advance interview
        _filler = answer.lower().strip(" .!?,")
        if _filler in {"oh yeah", "oh yeah.", "yeah", "yes", "hello", "hello?", "hi", "hey", "okay", "ok", "thanks", "thank you"}:
            logger.info(f"STT filler ignored: '{answer}'")
            return
        q_idx = self.state.current_question_index + 1
        logger.info(f"STT final (debounced) → storing answer for Q{q_idx}: '{answer[:120]}...' ({len(answer)} chars)")
        try:
            next_question = await self.controller.submit_answer(state=self.state, answer=answer)
            if self.session_id:
                try:
                    save_session(self.session_id, self.state)
                except Exception as e:
                    logger.warning(f"Failed to persist session {self.session_id}: {e}")
            await self._push_result(next_question)
        except Exception as e:
            logger.exception(f"Error in _flush_buffer: {e}")
            try:
                curr = self.state.get_current_question()
                if curr is not None and len(self.state.answers) > self.state.current_question_index:
                    self.state.move_to_next_question()
                nxt = self.state.get_current_question() if not self.state.is_interview_complete() else None
                await self._push_result(nxt)
            except Exception as rec_e:
                logger.error(f"Recovery also failed: {rec_e}")

    def _clear_stale_stt(self):
        """Clear buffer/debounce leaked from previous turn."""
        if self._buffer:
            logger.info(f"Clearing stale STT buffer: '{self._buffer[:80]}'")
            self._buffer = ""
        if self._debounce_task and not self._debounce_task.done():
            self._debounce_task.cancel()
            self._debounce_task = None

    def _progress_payload(self, next_question):
        total = len(self.state.questions)
        if next_question is None:
            return {
                "type": "interview_progress",
                "done": True,
                "current_index": total,
                "total": total,
                "question": None,
                "question_id": None,
            }
        return {
            "type": "interview_progress",
            "done": False,
            "current_index": self.state.current_question_index,
            "total": total,
            "question": next_question.get("question"),
            "question_id": next_question.get("id"),
        }

    async def _push_ui_progress(self, next_question):
        """Notify the browser of the current question over the WebRTC data channel."""
        await self.push_frame(
            OutputTransportMessageUrgentFrame(message=self._progress_payload(next_question)),
            FrameDirection.DOWNSTREAM,
        )

    async def _push_result(self, next_question):
        """Push next question or closing message as ONE atomic TTS utterance.

        Uses TTSSpeakFrame which creates a dedicated audio context per utterance
        in Pipecat 1.4.0. This guarantees the TTS service synthesizes the entire
        text as a single logical utterance instead of splitting a TextFrame into
        multiple sentence-level aggregations (which Deepgram would speak as
        independent chunks, causing awkward pauses and merged transition+question
        audio).
        """
        self._clear_stale_stt()
        await self._push_ui_progress(next_question)
        if next_question is None:
            self.interview_finished = True
            final_message = "Thank you. That concludes the interview."
            final_message = normalize_for_tts(final_message)
            self._is_speaking = True
            if self._guard_task and not self._guard_task.done():
                self._guard_task.cancel()
                self._guard_task = None
            logger.info("SPEAKING -> pushing final message (atomic TTSSpeakFrame)")
            await self.push_frame(TTSSpeakFrame(text=final_message), FrameDirection.DOWNSTREAM)
            # Persist completion so /report can batch-evaluate; trigger report once.
            if self.session_id:
                try:
                    save_session(self.session_id, self.state)
                except Exception as e:
                    logger.warning(f"Failed to persist session {self.session_id} on completion: {e}")
            # Trigger post-interview report generation in background (once, non-blocking).
            if not getattr(self, "_report_triggered", False):
                self._report_triggered = True
                try:
                    asyncio.create_task(self._trigger_report())
                except Exception as e:
                    logger.warning(f"Failed to schedule report generation: {e}")
            return
        # Q2-Q6: speak ONLY the next question text — no filler.
        question_text = next_question["question"]
        question_text = normalize_for_tts(question_text)
        self._is_speaking = True
        if self._guard_task and not self._guard_task.done():
            self._guard_task.cancel()
            self._guard_task = None
        logger.info(f"SPEAKING -> pushing Q{self.state.current_question_index+1} atomic TTSSpeakFrame ({len(question_text)} chars)")
        await self.push_frame(TTSSpeakFrame(text=question_text), FrameDirection.DOWNSTREAM)

    async def _trigger_report(self):
        """Background batch evaluation after interview completes. Idempotent."""
        try:
            # Avoid duplicate generation if evaluations already exist.
            if self.state.evaluations and self.state.final_score is not None:
                logger.info("Report already generated — skipping background evaluation")
                return
            logger.info(f"Starting post-interview batch evaluation for session {self.session_id or 'unknown'}")
            await self.controller.evaluate_interview(self.state)
            if self.session_id:
                try:
                    save_session(self.session_id, self.state)
                except Exception as e:
                    logger.warning(f"Failed to persist session after report: {e}")
            logger.info(f"Batch report generated — final_score={self.state.final_score}")
        except Exception as e:
            logger.exception(f"Background report generation failed: {e}")

    async def _on_tts_started(self):
        if self._guard_task and not self._guard_task.done():
            self._guard_task.cancel()
            self._guard_task = None
        if not self._is_speaking:
            logger.info("SPEAKING started (TTSStartedFrame)")
        self._is_speaking = True

    async def _on_tts_stopped(self):
        # Guard 200ms before transitioning to LISTENING — handles multi-sentence splits + filters bot echo
        if self._guard_task and not self._guard_task.done():
            self._guard_task.cancel()

        async def guard():
            try:
                await asyncio.sleep(0.2)
                if self.interview_finished:
                    return
                self._clear_stale_stt()
                self._is_speaking = False
                logger.info("TTSStoppedFrame + 200ms guard -> LISTENING (buffer cleared, fresh turn)")
            except asyncio.CancelledError:
                return

        self._guard_task = asyncio.create_task(guard())

    async def _cancel_debounce(self):
        if self._debounce_task and not self._debounce_task.done():
            self._debounce_task.cancel()
            try:
                await self._debounce_task
            except asyncio.CancelledError:
                pass
            self._debounce_task = None

    async def process_frame(
        self,
        frame,
        direction: FrameDirection,
    ):
        await super().process_frame(frame, direction)

        if isinstance(frame, (CancelFrame, EndFrame)):
            self.interview_finished = True
            await self._cancel_debounce()
            if self._guard_task and not self._guard_task.done():
                self._guard_task.cancel()
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, (TTSStartedFrame, BotStartedSpeakingFrame)):
            await self._on_tts_started()
            await self.push_frame(frame, direction)
            return
        if isinstance(frame, (TTSStoppedFrame, BotStoppedSpeakingFrame)):
            await self._on_tts_stopped()
            await self.push_frame(frame, direction)
            return

        # Interim: used ONLY to detect continued speech and reset debounce.
        # Never append interim text to buffer (interim is an evolving hypothesis).
        if isinstance(frame, InterimTranscriptionFrame):
            txt = frame.text.strip()
            if not txt:
                return
            if self._is_speaking:
                logger.info(f"STT interim during SPEAKING — dropped: '{txt[:60]}'")
                return
            # LISTENING: if we already have final text buffered, interim means candidate still speaking
            if self._buffer:
                logger.info(f"STT interim (LISTENING, buffer={len(self._buffer)}) — resetting debounce: '{txt[:60]}'")
                self._reset_debounce()
            else:
                logger.info(f"STT interim (LISTENING, no buffer yet): '{txt[:60]}'")
            # Do NOT forward downstream (no LLM aggregator in deterministic pipeline)
            return

        if isinstance(frame, TranscriptionFrame):
            if self.interview_finished:
                return
            if self._is_speaking:
                logger.info(f"STT final during SPEAKING — dropped: '{frame.text.strip()[:60]}'")
                return

            answer = frame.text.strip()
            logger.info(f"STT transcript received: '{answer[:80]}' (buffer len before={len(self._buffer)})")

            if not answer:
                return

            # Accumulate ONLY final transcripts
            if self._buffer:
                self._buffer += " " + answer
            else:
                self._buffer = answer

            # Reset debounce timer (candidate may still be speaking; interim will extend it)
            if self._debounce_task and not self._debounce_task.done():
                self._debounce_task.cancel()
            self._debounce_task = asyncio.create_task(self._debounced_flush())
            return

        await self.push_frame(frame, direction)
