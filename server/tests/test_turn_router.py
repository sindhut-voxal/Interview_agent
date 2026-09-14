"""Turn-router rules and interview barge-in (no live Gemini)."""
import asyncio
import sys
import time
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from interview.turn_router import classify_with_rules, looks_like_interrupt
from interview.state import InterviewState
from interview.controller import InterviewController
from interview_processor import InterviewProcessor, SPEAKING, LISTENING, CLARIFYING
from pipecat.frames.frames import TTSSpeakFrame


def make_state(num=6):
    questions = [
        {"id": i + 1, "question": f"Question {i + 1} text?", "skill": "test", "criteria": ["a", "b"], "weight": 15}
        for i in range(num)
    ]
    questions[-1]["weight"] = 25
    return InterviewState(resume="r", job_description="jd", questions=questions)


class CapturingProcessor(InterviewProcessor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.captured = []

    async def push_frame(self, frame, direction=None):
        from pipecat.processors.frame_processor import FrameDirection
        if direction is None:
            direction = FrameDirection.DOWNSTREAM
        self.captured.append((frame, direction))


def test_rules_repeat():
    d = classify_with_rules("sorry can you repeat that", interrupted=True, playback_pct=0.2)
    assert d["action"] == "clarify"
    assert d["reason"] == "clarification_request"


def test_rules_wait():
    d = classify_with_rules("wait one second", interrupted=False, playback_pct=1.0)
    assert d["action"] == "clarify"
    assert d["reason"] == "wait"


def test_rules_filler_listening():
    d = classify_with_rules("yeah", interrupted=False, playback_pct=1.0)
    assert d["action"] == "stay"
    assert d["reason"] == "noise"


def test_rules_filler_mid_question():
    d = classify_with_rules("ok", interrupted=True, playback_pct=0.3)
    assert d["action"] == "clarify"


def test_rules_long_answer_after_listen():
    d = classify_with_rules(
        "I used FastAPI for the backend and Docker to package the service for deploy.",
        interrupted=False,
        playback_pct=1.0,
    )
    assert d["action"] == "advance"


def test_rules_barge_in_late_long_answer():
    d = classify_with_rules(
        "I used FastAPI for the backend and Docker to package the service for deploy.",
        interrupted=True,
        playback_pct=0.9,
    )
    assert d["action"] == "advance"


def test_rules_early_short_barge_in():
    d = classify_with_rules("so about docker", interrupted=True, playback_pct=0.15)
    assert d["action"] == "clarify"


def test_interrupt_signal():
    assert looks_like_interrupt("wait")
    assert looks_like_interrupt("can you repeat")
    assert not looks_like_interrupt("uh")


async def test_barge_in_does_not_drop():
    proc = CapturingProcessor(InterviewController(), make_state(), session_id="barge-1")
    proc._turn_state = SPEAKING
    proc._is_speaking = True
    proc._tts_started_at = time.monotonic() - 1.0
    dropped = await proc._maybe_barge_in("can you repeat the question")
    assert dropped is False
    assert proc._turn_state == LISTENING
    assert proc._barge_in_this_turn is True
    assert proc._is_speaking is False


async def test_echo_window_drops():
    proc = CapturingProcessor(InterviewController(), make_state(), session_id="echo-1")
    proc._turn_state = SPEAKING
    proc._is_speaking = True
    proc._tts_started_at = time.monotonic()
    dropped = await proc._maybe_barge_in("hello there candidate")
    assert dropped is True
    assert proc._turn_state == SPEAKING


async def test_clarify_does_not_advance():
    state = make_state()
    proc = CapturingProcessor(InterviewController(), state, session_id="clarify-1")
    q = state.get_current_question()["question"]
    await proc._apply_decision(
        {"action": "clarify", "reason": "clarification_request", "reply": "Sure, here it is again."},
        "repeat please",
        q,
    )
    assert state.current_question_index == 0
    assert state.answers == []
    tts = [f for f, _ in proc.captured if isinstance(f, TTSSpeakFrame)]
    assert len(tts) == 1
    assert "Sure, here it is again." in tts[0].text
    assert q in tts[0].text
    assert proc._turn_state == CLARIFYING


async def test_advance_stores_answer():
    state = make_state()
    proc = CapturingProcessor(InterviewController(), state, session_id="adv-1")
    answer = "I used Python and FastAPI to build the API and Docker to ship it."
    await proc._apply_decision({"action": "advance", "reason": "answered", "reply": None}, answer, "Q?")
    assert len(state.answers) == 1
    assert state.current_question_index == 1
    assert answer in state.answers[0]["answer"]


async def test_stay_keeps_partial():
    state = make_state()
    proc = CapturingProcessor(InterviewController(), state, session_id="stay-1")
    await proc._apply_decision({"action": "stay", "reason": "incomplete", "reply": None}, "so first I", "Q?")
    assert state.current_question_index == 0
    assert proc._partial_answer.startswith("so first I")
    assert proc._turn_state == LISTENING


async def main():
    test_rules_repeat()
    test_rules_wait()
    test_rules_filler_listening()
    test_rules_filler_mid_question()
    test_rules_long_answer_after_listen()
    test_rules_barge_in_late_long_answer()
    test_rules_early_short_barge_in()
    test_interrupt_signal()
    await test_barge_in_does_not_drop()
    await test_echo_window_drops()
    await test_clarify_does_not_advance()
    await test_advance_stores_answer()
    await test_stay_keeps_partial()
    print("PASS: turn router + barge-in")


if __name__ == "__main__":
    asyncio.run(main())
