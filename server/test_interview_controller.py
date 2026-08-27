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

    answers = [
        """
        I have used Python extensively in my AI projects.
        In my AI Language Tutor, I used Python and Pipecat
        to build the backend pipeline.
        """,

        """
        I have worked with Docker to manage application
        dependencies and ensure consistent environments.
        I would define dependencies in a Dockerfile and
        run the application inside a container.
        """,

        """
        In my AI Language Tutor, latency was an important
        challenge. I measured the different stages of the
        pipeline such as speech-to-text, LLM processing,
        and text-to-speech to identify bottlenecks.
        """,

        """
        For managing LLM context, I would avoid continuously
        sending the entire conversation. I could use strategies
        such as summarization or a sliding context window.
        """,

        """
        To scale the Interview Agent, I would containerize
        the application using Docker and deploy multiple
        FastAPI instances behind a load balancer. Heavy AI
        processing could be moved to separate services.
        """
    ]

    while not state.is_interview_complete():

        current_question = state.get_current_question()

        print("\n" + "=" * 60)

        print(
            f"QUESTION {current_question['id']}:"
        )

        print(
            current_question["question"]
        )

        answer = answers[
            state.current_question_index
        ]

        print("\nANSWER:")

        print(answer.strip())

        await controller.submit_answer(
            state,
            answer,
        )

        evaluation = state.evaluations[-1]

        print("\nEVALUATION:")

        print(
            json.dumps(
                evaluation,
                indent=4,
            )
        )

    print("\n" + "=" * 60)

    print("\n--- INTERVIEW COMPLETE ---\n")

    print(
        "Final Score:",
        state.final_score,
    )

    print("\n--- ALL ANSWERS ---\n")

    print(
        json.dumps(
            state.answers,
            indent=4,
        )
    )

    print("\n--- ALL EVALUATIONS ---\n")

    print(
        json.dumps(
            state.evaluations,
            indent=4,
        )
    )


if __name__ == "__main__":
    asyncio.run(main())