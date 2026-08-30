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
Built an AI-powered interview system capable
of asking questions and evaluating candidate
responses.
"""


JOB_DESCRIPTION = """
We are looking for an AI Engineer.

Required skills:

- Strong Python programming
- Experience with Large Language Models
- Machine Learning fundamentals
- API development
- FastAPI
- Docker
"""


async def main():

    print("1. Creating InterviewController...")

    controller = InterviewController()

    print("2. Creating interview...")

    state = await controller.create_interview(
        resume=RESUME,
        job_description=JOB_DESCRIPTION,
    )

    print("\n3. Interview created successfully")

    print("\nGenerated questions:")

    print(
        json.dumps(
            state.questions,
            indent=4,
        )
    )

    print("\nCurrent question:")

    current_question = state.get_current_question()

    print(current_question["question"])


if __name__ == "__main__":
    asyncio.run(main())