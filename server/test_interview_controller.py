import asyncio
import json

from interview.controller import InterviewController


RESUME = """
AI Engineer Intern

Skills:
Python, Machine Learning, Deep Learning,
Generative AI, FastAPI, Docker

Projects:

1. AI Language Tutor
Built a real-time AI language tutor using
Pipecat, Deepgram, Gemini and WebRTC.

2. Interview Agent
Built an AI-powered interview system.
"""


JOB_DESCRIPTION = """
We are looking for an AI Engineer.

Required skills:

- Python
- Large Language Models
- Machine Learning
- FastAPI
- Docker
"""


async def main():

    controller = InterviewController()

    print("Creating interview...\n")

    state = await controller.create_interview(
        resume=RESUME,
        job_description=JOB_DESCRIPTION,
    )

    # -------------------------
    # Question 1
    # -------------------------

    current_question = state.get_current_question()

    print("QUESTION 1:")
    print(current_question["question"])

    answer_1 = """
    I used Python extensively in my AI projects.
    In my AI Language Tutor, I used Python with
    Pipecat to build the backend pipeline.
    """

    next_question = controller.submit_answer(
        state,
        answer_1,
    )

    print("\nAnswer stored.")

    print(
        "Current question index:",
        state.current_question_index,
    )

    print("\nNEXT QUESTION:")

    print(next_question["question"])


    # -------------------------
    # Question 2
    # -------------------------

    answer_2 = """
    I have worked with LLMs such as Gemini
    for generating responses in AI applications.
    """

    next_question = controller.submit_answer(
        state,
        answer_2,
    )

    print("\nAnswer stored.")

    print(
        "Current question index:",
        state.current_question_index,
    )

    print("\nNEXT QUESTION:")

    print(next_question["question"])


    # -------------------------
    # Stored answers
    # -------------------------

    print("\n--- STORED ANSWERS ---\n")

    print(
        json.dumps(
            state.answers,
            indent=4,
        )
    )


if __name__ == "__main__":
    asyncio.run(main())