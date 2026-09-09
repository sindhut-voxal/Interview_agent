import asyncio
import json
import os
from loguru import logger

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
        """REAL-TIME path: store answer and advance, no LLM eval."""
        current_question = state.get_current_question()
        if current_question is None:
            return None
        state.add_answer(answer)
        state.move_to_next_question()
        if state.is_interview_complete():
            return None
        return state.get_current_question()

    async def submit_answer_with_eval(
        self,
        state: InterviewState,
        answer: str,
    ):
        """Legacy path with per-answer LLM eval (for offline tests)."""
        current_question = state.get_current_question()
        if current_question is None:
            return None
        state.add_answer(answer)
        evaluation = await evaluate_answer(question=current_question, answer=answer)
        state.add_evaluation(evaluation)
        state.move_to_next_question()
        if state.is_interview_complete():
            calculate_final_score(state)
            return None
        return state.get_current_question()

    async def evaluate_interview(self, state: InterviewState, concurrency: int = 3) -> dict:
        """POST-INTERVIEW batch evaluation. Runs after is_complete."""
        if not state.answers:
            state.report_status = "ready"
            state.final_score = 0
            return {"evaluations": [], "final_score": 0, "strengths": [], "improvements": []}

        def qid(value):
            try:
                return int(value)
            except Exception:
                return value

        q_by_id = {qid(q["id"]): q for q in state.questions}
        sem = asyncio.Semaphore(concurrency)

        async def eval_one(index: int, ans_entry: dict):
            async with sem:
                raw_qid = ans_entry.get("question_id")
                q = q_by_id.get(qid(raw_qid))
                if q is None:
                    q = state.questions[index] if index < len(state.questions) else None
                if q is None:
                    return {"question_id": raw_qid, "score": 0, "feedback": "Evaluation unavailable.", "strengths": [], "improvements": [], "error": "No matching question"}
                try:
                    ev = await evaluate_answer(question=q, answer=ans_entry.get("answer", ""))
                    ev["question_id"] = qid(q["id"])
                    return ev
                except Exception as e:
                    logger.warning(f"eval failed for Q{raw_qid or q['id']}: {e}")
                    return {"question_id": qid(q["id"]), "score": 0, "feedback": "Evaluation unavailable.", "strengths": [], "improvements": [], "error": str(e)}

        results = await asyncio.gather(*(eval_one(i, a) for i, a in enumerate(state.answers)))
        state.evaluations = list(results)
        total = calculate_final_score(state)
        strengths = []
        improvements = []
        for ev in results:
            strengths.extend(ev.get("strengths") or [])
            improvements.extend(ev.get("improvements") or [])

        def dedupe(seq):
            seen = set()
            out = []
            for x in seq:
                if x not in seen:
                    seen.add(x)
                    out.append(x)
            return out

        failed = sum(1 for ev in results if ev.get("error"))
        state.report_status = "ready"
        state.report_error = f"{failed} question(s) could not be evaluated" if failed else None
        return {
            "evaluations": state.evaluations,
            "final_score": total,
            "strengths": dedupe(strengths)[:5],
            "improvements": dedupe(improvements)[:5],
        }
