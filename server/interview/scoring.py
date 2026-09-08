from interview.state import InterviewState


def calculate_final_score(
    state: InterviewState,
) -> float:

    total_score = 0.0

    for evaluation in state.evaluations:
        try:
            score = float(evaluation.get("score", 0) or 0)
        except Exception:
            score = 0
        score = max(0, score)
        total_score += score

    # Clamp to sum of weights so score never exceeds 100 (or custom total)
    try:
        max_possible = sum(int(q.get("weight", 0)) for q in state.questions) or 100
        total_score = min(total_score, max_possible)
    except Exception:
        pass

    state.final_score = total_score

    return total_score