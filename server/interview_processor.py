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

            # Interview complete
            if next_question is None:

                self.interview_finished = True

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

            # Send only the next predefined question
            await self.push_frame(
                TextFrame(
                    next_question["question"]
                ),
                direction,
            )

            return

        await self.push_frame(
            frame,
            direction,
        )