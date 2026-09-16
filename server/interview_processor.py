import asyncio
import time

from loguru import logger

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    InterruptionFrame,
    InterimTranscriptionFrame,
    TranscriptionFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    TTSSpeakFrame,
    OutputTransportMessageUrgentFrame,
    UserStartedSpeakingFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)

from pipecat.processors.frame_processor import (
    FrameDirection,
    FrameProcessor,
)

from interview.controller import InterviewController
from interview.state import InterviewState
from interview.turn_router import (
    DEFAULT_ADVANCE_ACK,
    DEFAULT_FOLLOWUP_REPLY,
    DEFAULT_NUDGE_REPLY,
    DEFAULT_REPEAT_REPLY,
    SCREENING_LIMIT_S,
    decide_turn,
    looks_like_interrupt,
    restate_question,
    word_count,
)
from interview.speaker import compose_followup
from session_store import save_session, acquire_report_lock, release_report_lock

try:
    from text_normaliser import normalize_for_tts  # type: ignore
except Exception:
    def normalize_for_tts(text: str) -> str:  # fallback no-op
        return text.strip()

SPEAKING = "SPEAKING"
LISTENING = "LISTENING"
DECIDING = "DECIDING"
CLARIFYING = "CLARIFYING"

ECHO_WINDOW_S = 0.45
CHARS_PER_SEC = 14.0
SILENCE_NUDGE_S = 12.0


class InterviewProcessor(FrameProcessor):

    def __init__(
        self,
        controller: InterviewController,
        state: InterviewState,
        session_id: str | None = None,
        debounce_s: float = 0.6,
        min_chars: int = 3,
        silence_nudge_s: float = SILENCE_NUDGE_S,
    ):
        super().__init__()

        self.controller = controller
        self.state = state
        self.session_id = session_id

        self.interview_finished = False
        self._buffer: str = ""
        self._partial_answer: str = ""
        self._debounce_s = debounce_s
        self._min_chars = min_chars
        self._debounce_task: asyncio.Task | None = None
        self._processing: bool = False
        self._turn_state: str = SPEAKING
        self._is_speaking: bool = True
        self._guard_task: asyncio.Task | None = None
        self._report_triggered: bool = False
        self._interrupted: bool = False
        self._barge_in_this_turn: bool = False
        self._tts_started_at: float | None = None
        self._tts_expected_s: float = 1.0
        self._tts_text: str = ""
        self._silence_nudge_s = float(silence_nudge_s)
        self._silence_task: asyncio.Task | None = None
        self._nudged_this_question: bool = False
        self._followup_used: bool = False
        self._started_at: float = time.time()

    def _playback_pct(self) -> float:
        if not self._tts_started_at:
            return 1.0 if self._turn_state != SPEAKING else 0.0
        elapsed = time.monotonic() - self._tts_started_at
        expected = max(0.8, self._tts_expected_s)
        return max(0.0, min(1.0, elapsed / expected))

    def _merge_text(self, left: str, right: str) -> str:
        left = (left or "").strip()
        right = (right or "").strip()
        if not left:
            return right
        if not right:
            return left
        return f"{left} {right}"

    def _reset_debounce(self):
        if self._debounce_task and not self._debounce_task.done():
            self._debounce_task.cancel()
        self._debounce_task = asyncio.create_task(self._debounced_flush())

    async def _debounced_flush(self):
        try:
            await asyncio.sleep(self._debounce_s)
            await self._flush_buffer()
        except asyncio.CancelledError:
            pass

    def _clear_stale_stt(self):
        if self._buffer:
            logger.info(f"Clearing stale STT buffer: '{self._buffer[:80]}'")
            self._buffer = ""
        if self._debounce_task and not self._debounce_task.done():
            self._debounce_task.cancel()
            self._debounce_task = None

    async def _stop_bot_speech(self):
        """Cut in-progress TTS so the candidate's interrupt is heard."""
        self._interrupted = True
        self._barge_in_this_turn = True
        self._is_speaking = False
        self._turn_state = LISTENING
        if self._guard_task and not self._guard_task.done():
            self._guard_task.cancel()
            self._guard_task = None
        try:
            await self.broadcast_interruption()
            logger.info("BARGE-IN -> broadcast InterruptionFrame, now LISTENING")
        except Exception as e:
            logger.warning(f"broadcast_interruption failed: {e}")

    async def _maybe_barge_in(self, text: str) -> bool:
        if self._turn_state not in {SPEAKING, CLARIFYING} and not self._is_speaking:
            return False
        if self._tts_started_at and (time.monotonic() - self._tts_started_at) < ECHO_WINDOW_S:
            logger.info(f"STT during echo window — dropped: '{text[:60]}'")
            return True
        if not looks_like_interrupt(text):
            logger.info(f"STT during SPEAKING too short for barge-in: '{text[:60]}'")
            return True
        await self._stop_bot_speech()
        return False

    async def _flush_buffer(self):
        if self.interview_finished or self._processing:
            return
        if self._turn_state in {SPEAKING, CLARIFYING} and self._is_speaking:
            return
        answer = self._buffer.strip()
        self._buffer = ""
        if not answer:
            logger.info("STT buffer empty — ignoring")
            return
        if len(answer) < self._min_chars:
            logger.info(f"STT buffer below threshold ({len(answer)} chars): '{answer}'")

        self._processing = True
        self._turn_state = DECIDING
        self._cancel_silence()
        interrupted = self._barge_in_this_turn
        playback_pct = self._playback_pct()
        current = self.state.get_current_question() or {}
        question_text = current.get("question") or ""
        try:
            decision = await decide_turn(
                question=question_text,
                partial=self._partial_answer,
                utterance=answer,
                interrupted=interrupted,
                playback_pct=playback_pct,
                followup_used=self._followup_used,
                remaining_min=self._remaining_min(),
            )
            await self._apply_decision(
                decision,
                answer,
                question_text,
                interrupted=interrupted,
                playback_pct=playback_pct,
            )
        except Exception as e:
            logger.exception(f"Error in _flush_buffer: {e}")
            combined = self._merge_text(self._partial_answer, answer)
            if len(combined) >= 50:
                await self._advance_with_answer(combined)
            else:
                self._partial_answer = combined
                self._turn_state = LISTENING
                self._is_speaking = False
        finally:
            self._processing = False
            self._barge_in_this_turn = False

    def _remaining_min(self) -> float:
        elapsed = time.time() - self._started_at
        return max(0.0, (SCREENING_LIMIT_S - elapsed) / 60.0)

    def _time_up(self) -> bool:
        return (time.time() - self._started_at) >= (SCREENING_LIMIT_S - 30)

    def _preface(self, text) -> str | None:
        if not isinstance(text, str):
            return None
        spoken = " ".join(text.strip().split())
        if not spoken or len(spoken.split()) > 12:
            return None
        return spoken

    def _cancel_silence(self):
        if self._silence_task and not self._silence_task.done():
            self._silence_task.cancel()
        self._silence_task = None

    def _arm_silence(self):
        self._cancel_silence()
        if (
            self.interview_finished
            or self._nudged_this_question
            or self._silence_nudge_s <= 0
        ):
            return
        self._silence_task = asyncio.create_task(self._silence_nudge())

    async def _silence_nudge(self):
        try:
            await asyncio.sleep(self._silence_nudge_s)
            if (
                self.interview_finished
                or self._processing
                or self._is_speaking
                or self._turn_state != LISTENING
            ):
                return
            self._nudged_this_question = True
            self._turn_state = CLARIFYING
            logger.info("Silence nudge — prompting candidate")
            await self._speak(DEFAULT_NUDGE_REPLY, progress_question=self.state.get_current_question())
        except asyncio.CancelledError:
            return

    def _clarify_speech(self, reply: str, question_text: str) -> str:
        restated = restate_question(question_text)
        cleaned = (reply or "").strip()
        if not cleaned or cleaned == DEFAULT_REPEAT_REPLY:
            return restated or question_text
        if word_count(cleaned) >= 8:
            return cleaned
        if restated and restated.lower() not in cleaned.lower():
            return f"{cleaned} {restated}".strip()
        return cleaned or restated or question_text

    async def _apply_decision(
        self,
        decision: dict,
        utterance: str,
        question_text: str,
        *,
        interrupted: bool = False,
        playback_pct: float = 1.0,
    ):
        action = decision.get("action")
        reason = decision.get("reason")
        logger.info(
            f"Turn decision={action} reason={reason} tool={decision.get('tool')} barge_in={self._barge_in_this_turn}"
        )

        if action == "advance":
            combined = self._merge_text(self._partial_answer, utterance)
            self._partial_answer = ""
            ack = self._preface(decision.get("reply")) or DEFAULT_ADVANCE_ACK
            await self._advance_with_answer(combined, preface=ack)
            return

        if action == "follow_up":
            self._partial_answer = self._merge_text(self._partial_answer, utterance)
            self._followup_used = True
            spoken = (decision.get("reply") or "").strip() or DEFAULT_FOLLOWUP_REPLY
            if spoken == DEFAULT_FOLLOWUP_REPLY:
                composed = await compose_followup(question_text, self._partial_answer)
                if composed:
                    spoken = composed
            self._turn_state = CLARIFYING
            await self._speak(spoken, progress_question=self.state.get_current_question())
            return

        if action == "stay":
            self._partial_answer = self._merge_text(self._partial_answer, utterance)
            self._turn_state = LISTENING
            self._is_speaking = False
            logger.info(f"STAY on Q{self.state.current_question_index + 1} (partial={len(self._partial_answer)} chars)")
            self._arm_silence()
            return

        if action == "skip":
            self._partial_answer = ""
            await self._advance_with_answer("[skipped]", preface=self._preface(decision.get("reply")))
            return

        if action == "end":
            self._partial_answer = ""
            await self._push_result(None, preface=self._preface(decision.get("reply")))
            return

        if action == "answer":
            reply = (decision.get("reply") or "").strip() or "Happy to clarify."
            spoken = reply
            if interrupted and playback_pct < 0.85 and question_text:
                restated = restate_question(question_text)
                spoken = f"{reply} {restated}".strip() if restated.lower() not in reply.lower() else reply
            self._turn_state = CLARIFYING
            await self._speak(spoken, progress_question=self.state.get_current_question())
            return

        reply = (decision.get("reply") or "").strip()
        if reason == "wait":
            spoken = reply or "Take your time."
        elif reason == "stt_repair":
            spoken = reply or "Sorry, I missed that. Could you say it again?"
        else:
            spoken = self._clarify_speech(reply, question_text)
        self._turn_state = CLARIFYING
        await self._speak(spoken, progress_question=self.state.get_current_question())

    async def _advance_with_answer(self, answer: str, preface: str | None = None):
        q_idx = self.state.current_question_index + 1
        logger.info(f"ADVANCE Q{q_idx}: '{answer[:120]}...' ({len(answer)} chars)")
        try:
            next_question = await self.controller.submit_answer(state=self.state, answer=answer)
            if self._time_up() and next_question is not None:
                next_question = None
                preface = "We're at time."
            if self.session_id:
                try:
                    save_session(self.session_id, self.state)
                except Exception as e:
                    logger.warning(f"Failed to persist session {self.session_id}: {e}")
            self._partial_answer = ""
            await self._push_result(next_question, preface=preface)
        except Exception as e:
            logger.exception(f"Error advancing question: {e}")
            try:
                curr = self.state.get_current_question()
                if curr is not None and len(self.state.answers) > self.state.current_question_index:
                    self.state.move_to_next_question()
                nxt = self.state.get_current_question() if not self.state.is_interview_complete() else None
                await self._push_result(nxt, preface=preface)
            except Exception as rec_e:
                logger.error(f"Recovery also failed: {rec_e}")

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
        await self.push_frame(
            OutputTransportMessageUrgentFrame(message=self._progress_payload(next_question)),
            FrameDirection.DOWNSTREAM,
        )

    async def _speak(self, text: str, *, progress_question=None):
        spoken = normalize_for_tts(text)
        self._cancel_silence()
        self._clear_stale_stt()
        self._is_speaking = True
        self._interrupted = False
        self._tts_text = spoken
        self._tts_expected_s = max(0.8, len(spoken) / CHARS_PER_SEC)
        self._tts_started_at = time.monotonic()
        if self._guard_task and not self._guard_task.done():
            self._guard_task.cancel()
            self._guard_task = None
        if progress_question is not None:
            await self._push_ui_progress(progress_question)
        logger.info(f"{self._turn_state} -> TTSSpeakFrame ({len(spoken)} chars)")
        await self.push_frame(TTSSpeakFrame(text=spoken), FrameDirection.DOWNSTREAM)

    async def _push_result(self, next_question, preface: str | None = None):
        """Push next question or closing message as ONE atomic TTS utterance."""
        self._partial_answer = ""
        self._barge_in_this_turn = False
        self._nudged_this_question = False
        self._followup_used = False
        lead = self._preface(preface)
        if next_question is None:
            self.interview_finished = True
            self._turn_state = SPEAKING
            closing = "Thank you. That concludes the interview."
            if lead:
                await self._speak(lead, progress_question=None)
            await self._speak(closing, progress_question=None)
            await self._push_ui_progress(None)
            if self.session_id:
                try:
                    save_session(self.session_id, self.state)
                except Exception as e:
                    logger.warning(f"Failed to persist session {self.session_id} on completion: {e}")
            if not getattr(self, "_report_triggered", False):
                self._report_triggered = True
                try:
                    asyncio.create_task(self._trigger_report())
                except Exception as e:
                    logger.warning(f"Failed to schedule report generation: {e}")
            return
        self._turn_state = SPEAKING
        q_text = next_question["question"]
        if lead:
            await self._speak(lead, progress_question=None)
        await self._speak(q_text, progress_question=next_question)

    async def _trigger_report(self):
        sid = self.session_id
        locked = False
        try:
            if self.state.evaluations and self.state.final_score is not None and len(self.state.evaluations) >= len(self.state.answers):
                logger.info("Report already generated — skipping background evaluation")
                return
            if sid and not acquire_report_lock(sid):
                logger.info(f"Report lock held for {sid} — skipping duplicate background evaluation")
                return
            locked = bool(sid)
            logger.info(f"Starting post-interview batch evaluation for session {sid or 'unknown'}")
            self.state.report_status = "running"
            if sid:
                save_session(sid, self.state)
            await self.controller.evaluate_interview(self.state)
            self.state.report_status = "ready"
            if sid:
                try:
                    save_session(sid, self.state)
                except Exception as e:
                    logger.warning(f"Failed to persist session after report: {e}")
            logger.info(f"Batch report generated — final_score={self.state.final_score}")
        except Exception as e:
            self.state.report_status = "error"
            self.state.report_error = str(e)
            if sid:
                try:
                    save_session(sid, self.state)
                except Exception:
                    pass
            logger.exception(f"Background report generation failed: {e}")
        finally:
            if locked and sid:
                release_report_lock(sid)

    async def _on_tts_started(self):
        if self._guard_task and not self._guard_task.done():
            self._guard_task.cancel()
            self._guard_task = None
        self._is_speaking = True
        if self._turn_state not in {CLARIFYING, SPEAKING}:
            self._turn_state = SPEAKING
        self._tts_started_at = time.monotonic()
        logger.info("SPEAKING started (TTSStartedFrame)")

    async def _on_tts_stopped(self):
        if self._interrupted:
            self._is_speaking = False
            self._turn_state = LISTENING
            self._interrupted = False
            logger.info("TTS stopped after barge-in — keeping STT buffer")
            return
        if self._guard_task and not self._guard_task.done():
            self._guard_task.cancel()

        async def guard():
            try:
                await asyncio.sleep(0.2)
                if self.interview_finished:
                    return
                if self._interrupted or self._turn_state == DECIDING:
                    return
                self._clear_stale_stt()
                self._is_speaking = False
                self._turn_state = LISTENING
                self._barge_in_this_turn = False
                logger.info("TTSStoppedFrame + 200ms guard -> LISTENING")
                if not self.interview_finished:
                    self._arm_silence()
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
            self._cancel_silence()
            if self._guard_task and not self._guard_task.done():
                self._guard_task.cancel()
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, InterruptionFrame):
            self._is_speaking = False
            if self._turn_state in {SPEAKING, CLARIFYING}:
                self._turn_state = LISTENING
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, (VADUserStartedSpeakingFrame, UserStartedSpeakingFrame)):
            self._cancel_silence()
            if (self._turn_state in {SPEAKING, CLARIFYING} or self._is_speaking) and not (
                self._tts_started_at and (time.monotonic() - self._tts_started_at) < ECHO_WINDOW_S
            ):
                await self._stop_bot_speech()
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, VADUserStoppedSpeakingFrame):
            if self._buffer.strip() and self._turn_state == LISTENING and not self._processing:
                if self._debounce_task and not self._debounce_task.done():
                    self._debounce_task.cancel()

                async def _vad_flush():
                    try:
                        await asyncio.sleep(0.15)
                        await self._flush_buffer()
                    except asyncio.CancelledError:
                        pass

                self._debounce_task = asyncio.create_task(_vad_flush())
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

        if isinstance(frame, InterimTranscriptionFrame):
            txt = frame.text.strip()
            if not txt or self.interview_finished:
                return
            dropped = await self._maybe_barge_in(txt)
            if dropped:
                return
            self._cancel_silence()
            if self._buffer:
                logger.info(f"STT interim (buffer={len(self._buffer)}) — resetting debounce: '{txt[:60]}'")
                self._reset_debounce()
            return

        if isinstance(frame, TranscriptionFrame):
            if self.interview_finished:
                return
            answer = frame.text.strip()
            if not answer:
                return
            dropped = await self._maybe_barge_in(answer)
            if dropped:
                return
            self._cancel_silence()

            logger.info(f"STT transcript received: '{answer[:80]}' (buffer len before={len(self._buffer)})")
            if self._buffer:
                self._buffer += " " + answer
            else:
                self._buffer = answer
            if self._debounce_task and not self._debounce_task.done():
                self._debounce_task.cancel()
            self._debounce_task = asyncio.create_task(self._debounced_flush())
            return

        await self.push_frame(frame, direction)
