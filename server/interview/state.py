from dataclasses import dataclass, field


@dataclass
class InterviewState:
    resume: str
    job_description: str

    questions: list = field(default_factory=list)

    current_question_index: int = 0

    answers: list = field(default_factory=list)

    evaluations: list = field(default_factory=list)

    final_score: float | None = None


    def get_current_question(self):
        if self.current_question_index >= len(self.questions):
            return None

        return self.questions[
            self.current_question_index
        ]


    def add_answer(self, answer: str):

        current_question = self.get_current_question()

        if current_question is None:
            return

        self.answers.append(
            {
                "question_id": current_question["id"],
                "answer": answer,
            }
        )


    def move_to_next_question(self):
        self.current_question_index += 1


    def is_interview_complete(self):
        return (
            self.current_question_index
            >= len(self.questions)
        )