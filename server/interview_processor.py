from pipecat.frames.frames import (
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
    ):
        super().__init__()

        self.controller = controller
        self.state = state

        self.interview_finished = False


    async def process_frame(
        self,
        frame,
        direction: FrameDirection,
    ):
        await super().process_frame(
            frame,
            direction,
        )

        # Only process candidate speech from STT
        if isinstance(frame, TranscriptionFrame):

            # Ignore anything after interview completion
            if self.interview_finished:

                return

            answer = frame.text.strip()

            if not answer:

                return

            next_question = (
                await self.controller.submit_answer(
                    state=self.state,
                    answer=answer,
                )
            )

            # Grab the evaluation for the answer just submitted
            last_evaluation = (
                self.state.evaluations[-1]
                if self.state.evaluations
                else None
            )
            feedback_text = ""
            if last_evaluation and last_evaluation.get("feedback"):
                feedback_text = last_evaluation["feedback"].strip()
                # Ensure it ends with a period for TTS
                if feedback_text and not feedback_text.endswith("."):
                    feedback_text += "."

            # Interview complete
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

                await self.push_frame(
                    TextFrame(final_message),
                    direction,
                )

                return

            # Send small feedback + transition + next question
            # This TextFrame flows via context_aggregator.user() -> LLM -> TTS
            # The LLM is instructed to acknowledge briefly then ask the next question.
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

            await self.push_frame(
                TextFrame(combined),
                direction,
            )

            return

        await self.push_frame(
            frame,
            direction,
        )