import asyncio
import json

from interview.answer_evaluator import evaluate_answer


QUESTION = {
    "id": 1,
    "question": (
        "How would you use Python and FastAPI to implement "
        "an asynchronous endpoint that handles a long-running "
        "Machine Learning inference task without blocking the server?"
    ),
    "skill": "FastAPI & Python",
    "criteria": [
        "Understanding of async def and asynchronous programming in Python",
        "Knowledge of how FastAPI handles concurrency",
        "Ability to prevent event loop blocking during heavy computations",
    ],
    "weight": 15,
}


ANSWER = """
I would create an async FastAPI endpoint using async def.

However, if the machine learning inference is CPU-intensive,
I would avoid running it directly on the event loop because it
could block other requests.

I could move the inference to a background worker or separate
service. For I/O operations, async and await would allow the
server to handle other requests while waiting for the operation
to complete.
"""


async def main():

    print("Starting answer evaluation...\n")

    evaluation = await evaluate_answer(
        question=QUESTION,
        answer=ANSWER,
    )

    print("\n--- EVALUATION ---\n")

    print(
        json.dumps(
            evaluation,
            indent=4,
        )
    )


if __name__ == "__main__":
    asyncio.run(main())