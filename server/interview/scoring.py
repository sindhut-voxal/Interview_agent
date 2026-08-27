from interview.state import InterviewState


def calculate_final_score(
    state: InterviewState,
) -> float:

    total_score = 0.0

    for evaluation in state.evaluations:

        score = evaluation.get(
            "score",
            0,
        )

        total_score += score

    state.final_score = total_score

    return total_score