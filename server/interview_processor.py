import asyncio

from loguru import logger

from pipecat.frames.frames import (
    InterimTranscriptionFrame,
    TextFrame,
    TranscriptionFrame,
)

from pipecat.processors.frame_processor import (
    FrameDirection,
    FrameProcessor,
)

from interview.controller import InterviewController
from interview.state import InterviewState


class InterviewProcessor(FrameProcessor):

    def __init__(
        self,
        controller: InterviewController,
        state: InterviewState,
        debounce_s: float = 3.0,
        min_chars: int = 8,
    ):
        super().__init__()

        self.controller = controller
        self.state = state

        self.interview_finished = False
        # Buffer + debounce to avoid treating every 1-2 word Deepgram final as a full answer
        self._buffer: str = ""
        self._debounce_s = debounce_s
        self._min_chars = min_chars
        self._debounce_task: asyncio.Task | None = None
        self._processing: bool = False

    async def _flush_buffer(self):
        """Wait debounce then evaluate accumulated buffer as one answer."""
        try:
            await asyncio.sleep(self._debounce_s)
            if self.interview_finished:
                return
            if self._processing:
                return
            answer = self._buffer.strip()
            self._buffer = ""
            if not answer or len(answer) < self._min_chars:
                logger.info(f"STT buffer below threshold ({len(answer)} chars) — ignoring: '{answer}'")
                return
            logger.info(f"STT final transcript (debounced) → evaluating answer for Q{self.state.current_question_index+1}: '{answer}'")
            self._processing = True
            next_question = (
                await self.controller.submit_answer(
                    state=self.state,
                    answer=answer,
                )
            )
            self._processing = False
            # need to push feedback/transition via queue to avoid blocking pipeline
            await self._push_result(next_question)
        except asyncio.CancelledError:
            return
        except Exception as e:
            self._processing = False
            logger.exception(f"Error in _flush_buffer: {e}")

    async def _push_result(self, next_question):
        last_evaluation = (
            self.state.evaluations[-1]
            if self.state.evaluations
            else None
        )
        feedback_text = ""
        if last_evaluation and last_evaluation.get("feedback"):
            feedback_text = last_evaluation["feedback"].strip()
            if feedback_text and not feedback_text.endswith("."):
                feedback_text += "."

        if next_question is None:
            self.interview_finished = True
            if feedback_text:
                final_message = (
                    f"{feedback_text} "
                    "Thank you. That was the last question. "
                    "The interview is now complete. "
                    f"Your final score is "
                    f"{self.state.final_score} out of 100."
                )
            else:
                final_message = (
                    "Thank you. The interview is now complete. "
                    f"Your final score is "
                    f"{self.state.final_score} out of 100."
                )
            await self.push_frame(TextFrame(final_message), FrameDirection.DOWNSTREAM)
            return

        if feedback_text:
            combined = (
                f"{feedback_text} "
                "Let's move to the next question. "
                f"{next_question['question']}"
            )
        else:
            combined = (
                "Thanks for your answer. "
                "Let's move to the next question. "
                f"{next_question['question']}"
            )
        await self.push_frame(TextFrame(combined), FrameDirection.DOWNSTREAM)

    async def process_frame(
        self,
        frame,
        direction: FrameDirection,
    ):
        await super().process_frame(
            frame,
            direction,
        )

        # Log interim for debugging and RESET debounce (user still speaking)
        if isinstance(frame, InterimTranscriptionFrame):
            txt = frame.text.strip()
            if txt:
                logger.debug(f"STT interim: '{txt}'")
                # keep debounce alive while user is still speaking
                if self._buffer and self._debounce_task and not self._debounce_task.done():
                    self._debounce_task.cancel()
                    self._debounce_task = asyncio.create_task(self._flush_buffer())
            await self.push_frame(frame, direction)
            return

        # Only process candidate speech from STT
        if isinstance(frame, TranscriptionFrame):

            # Ignore anything after interview completion
            if self.interview_finished:

                return

            answer = frame.text.strip()
            logger.info(f"STT transcript received: '{answer}' (buffer len before={len(self._buffer)})")

            if not answer:
                return

            # Accumulate into buffer instead of evaluating immediately
            if self._buffer:
                self._buffer += " " + answer
            else:
                self._buffer = answer

            # Reset debounce timer
            if self._debounce_task and not self._debounce_task.done():
                self._debounce_task.cancel()
            self._debounce_task = asyncio.create_task(self._flush_buffer())
            return

        await self.push_frame(
            frame,
            direction,
        )