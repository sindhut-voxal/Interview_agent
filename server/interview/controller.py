from interview.state import InterviewState
from interview.question_generator import generate_questions
from interview.answer_evaluator import evaluate_answer
from interview.scoring import (
    calculate_final_score,
)

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


    async def submit_answer(
    self,
    state: InterviewState,
    answer: str,
):

    # 1. Get current question
        current_question = (
            state.get_current_question()
        )

        if current_question is None:
            return None

        # 2. Store answer
        state.add_answer(answer)

        # 3. Evaluate answer
        evaluation = await evaluate_answer(
            question=current_question,
            answer=answer,
        )

        # 4. Store evaluation
        state.add_evaluation(
            evaluation
        )

        # 5. Move to next question
        state.move_to_next_question()

        # 6. Check whether interview is complete
        if state.is_interview_complete():

            calculate_final_score(
                state
            )

            return None

        # 7. Return next question
        return state.get_current_question()