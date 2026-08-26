from interview.state import InterviewState
from interview.question_generator import generate_questions


class InterviewController:

    async def create_interview(
        self,
        resume: str,
        job_description: str,
    ) -> InterviewState:

        questions = await generate_questions(
            resume=resume,
            job_description=job_description,
        )

        return InterviewState(
            resume=resume,
            job_description=job_description,
            questions=questions,
        )


    def submit_answer(
        self,
        state: InterviewState,
        answer: str,
    ):

        # Store answer for current question
        state.add_answer(answer)

        # Move to the next question
        state.move_to_next_question()

        # Return next question
        return state.get_current_question()