"""Turn-router rules and interview barge-in (no live Gemini)."""
import asyncio
import sys
import time
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from interview.turn_router import (
    classify_with_rules,
    decision_from_tool,
    looks_like_candidate_question,
    looks_like_interrupt,
    looks_like_stt_garbage,
)
from interview.state import InterviewState
from interview.controller import InterviewController
from interview_processor import InterviewProcessor, SPEAKING, LISTENING, CLARIFYING
from pipecat.frames.frames import TTSSpeakFrame, VADUserStartedSpeakingFrame


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
    tts = [f for f, _ in proc.captured if isinstance(f, TTSSpeakFrame)]
    assert tts[0].text == "Got it."
    assert state.questions[1]["question"] in tts[1].text


async def test_stay_keeps_partial():
    state = make_state()
    proc = CapturingProcessor(InterviewController(), state, session_id="stay-1")
    await proc._apply_decision({"action": "stay", "reason": "incomplete", "reply": None}, "so first I", "Q?")
    assert state.current_question_index == 0
    assert proc._partial_answer.startswith("so first I")
    assert proc._turn_state == LISTENING


def test_rules_skip():
    d = classify_with_rules("can we skip this question", interrupted=True, playback_pct=0.4)
    assert d["action"] == "skip"
    assert d["tool"] == "skip_question"


def test_rules_end():
    d = classify_with_rules("I want to stop the interview", interrupted=False, playback_pct=1.0)
    assert d["action"] == "end"


def test_rules_candidate_question_does_not_advance():
    d = classify_with_rules(
        "wait what do you mean by REST?",
        interrupted=True,
        playback_pct=0.3,
    )
    assert d is None


def test_rules_long_answer_not_treated_as_skip():
    d = classify_with_rules(
        "I don't know Kubernetes well but I used Docker in production for the language tutor.",
        interrupted=False,
        playback_pct=1.0,
    )
    assert d["action"] == "advance"


def test_meta_question_helper():
    assert looks_like_candidate_question("what do you mean by that")
    assert not looks_like_candidate_question("can you repeat the question")


def test_decision_from_tool_mapping():
    skip = decision_from_tool("skip_question", {"reason": "skip", "reply": "We'll skip it."})
    assert skip["action"] == "skip"
    ans = decision_from_tool("answer_candidate_question", {"reason": "candidate_question", "reply": "I mean REST APIs."})
    assert ans["action"] == "answer"
    assert "REST" in ans["reply"]
    stay = decision_from_tool("stay_silent", {"reason": "incomplete", "reply": "should be dropped"})
    assert stay["action"] == "stay"
    assert stay["reply"] is None


async def test_wait_does_not_replay_question():
    state = make_state()
    proc = CapturingProcessor(InterviewController(), state, session_id="wait-1")
    q = state.get_current_question()["question"]
    await proc._apply_decision(
        {"action": "clarify", "reason": "wait", "reply": "Take your time."},
        "hold on",
        q,
    )
    tts = [f for f, _ in proc.captured if isinstance(f, TTSSpeakFrame)]
    assert len(tts) == 1
    assert tts[0].text == "Take your time."
    assert q not in tts[0].text
    assert state.current_question_index == 0


async def test_answer_candidate_question_stays():
    state = make_state()
    proc = CapturingProcessor(InterviewController(), state, session_id="meta-1")
    q = state.get_current_question()["question"]
    await proc._apply_decision(
        {"action": "answer", "reason": "candidate_question", "reply": "I mean how you used Docker."},
        "what do you mean",
        q,
        interrupted=False,
        playback_pct=1.0,
    )
    assert state.current_question_index == 0
    assert state.answers == []
    tts = [f for f, _ in proc.captured if isinstance(f, TTSSpeakFrame)]
    assert tts[0].text == "I mean how you used Docker."
    assert q not in tts[0].text


async def test_answer_mid_question_restates():
    state = make_state()
    proc = CapturingProcessor(InterviewController(), state, session_id="meta-2")
    q = state.get_current_question()["question"]
    await proc._apply_decision(
        {"action": "answer", "reason": "candidate_question", "reply": "Happy to clarify."},
        "what do you mean",
        q,
        interrupted=True,
        playback_pct=0.2,
    )
    tts = [f for f, _ in proc.captured if isinstance(f, TTSSpeakFrame)]
    assert "Happy to clarify." in tts[0].text
    assert q in tts[0].text
    assert state.current_question_index == 0


async def test_skip_does_not_store_utterance():
    state = make_state()
    proc = CapturingProcessor(InterviewController(), state, session_id="skip-1")
    q2 = state.questions[1]["question"]
    await proc._apply_decision(
        {"action": "skip", "reason": "skip", "reply": "Alright, we'll skip that."},
        "can we skip this",
        "Q?",
    )
    assert state.current_question_index == 1
    assert state.answers[0]["answer"] == "[skipped]"
    tts = [f for f, _ in proc.captured if isinstance(f, TTSSpeakFrame)]
    assert tts[0].text == "Alright, we'll skip that."
    assert q2 in tts[1].text


async def test_end_does_not_store_stop_phrase():
    state = make_state()
    proc = CapturingProcessor(InterviewController(), state, session_id="end-1")
    await proc._apply_decision(
        {"action": "end", "reason": "end", "reply": "Of course, we can stop here."},
        "please end the interview",
        "Q?",
    )
    assert state.answers == []
    assert proc.interview_finished is True
    tts = [f for f, _ in proc.captured if isinstance(f, TTSSpeakFrame)]
    assert tts[0].text == "Of course, we can stop here."
    assert "That concludes the interview." in tts[1].text


def test_rules_idk_is_answer_not_skip():
    d = classify_with_rules("I don't know", interrupted=False, playback_pct=1.0)
    assert d["action"] == "advance"
    assert d["reply"] == "Got it."


def test_restate_clips_long_question():
    from interview.turn_router import restate_question

    long_q = "Can you walk through how you used Docker in production including networking volumes and CI? Then also mention Kubernetes."
    restated = restate_question(long_q)
    assert "Kubernetes" not in restated
    assert restated.startswith("Can you walk")
    assert len(restated.split()) <= 18


async def test_clarify_uses_short_restatement():
    from interview.turn_router import restate_question

    state = make_state()
    long_q = "Can you walk through how you used Docker in production including networking volumes and CI? Then also mention Kubernetes."
    state.questions[0]["question"] = long_q
    proc = CapturingProcessor(InterviewController(), state, session_id="restate-1")
    await proc._apply_decision(
        {"action": "clarify", "reason": "clarification_request", "reply": "Sure, here it is again."},
        "repeat please",
        long_q,
    )
    tts = [f for f, _ in proc.captured if isinstance(f, TTSSpeakFrame)]
    assert tts[0].text == restate_question(long_q)
    assert "Kubernetes" not in tts[0].text
    assert "Sure, here it is again." not in tts[0].text


async def test_silence_nudge_once():
    state = make_state()
    proc = CapturingProcessor(
        InterviewController(), state, session_id="nudge-1", silence_nudge_s=0.05
    )
    proc._turn_state = LISTENING
    proc._is_speaking = False
    proc._arm_silence()
    await asyncio.sleep(0.12)
    tts = [f for f, _ in proc.captured if isinstance(f, TTSSpeakFrame)]
    assert len(tts) == 1
    assert "whenever you're ready" in tts[0].text
    proc._turn_state = LISTENING
    proc._is_speaking = False
    proc._arm_silence()
    await asyncio.sleep(0.12)
    tts = [f for f, _ in proc.captured if isinstance(f, TTSSpeakFrame)]
    assert len(tts) == 1


def test_rules_thin_answer_follow_up():
    d = classify_with_rules(
        "I used Docker a lot in production deploy jobs.",
        interrupted=False,
        playback_pct=1.0,
        followup_used=False,
    )
    assert d["action"] == "follow_up"
    d2 = classify_with_rules(
        "I used Docker a lot in production deploy jobs.",
        interrupted=False,
        playback_pct=1.0,
        followup_used=True,
    )
    assert d2["action"] == "advance"


def test_rules_garbage_stt():
    assert looks_like_stt_garbage("zzzzzzzz kkkkkk")
    d = classify_with_rules("zzzzzzzz kkkkkk", interrupted=False, playback_pct=1.0)
    assert d["action"] == "clarify"
    assert d["reason"] == "stt_repair"


async def test_follow_up_does_not_advance():
    state = make_state()
    proc = CapturingProcessor(InterviewController(), state, session_id="fu-1")
    await proc._apply_decision(
        {"action": "follow_up", "reason": "thin_answer", "reply": "Can you share a brief example?"},
        "I used Docker for deploys.",
        "How have you used Docker?",
    )
    assert state.current_question_index == 0
    assert state.answers == []
    assert proc._followup_used is True
    tts = [f for f, _ in proc.captured if isinstance(f, TTSSpeakFrame)]
    assert tts[0].text == "Can you share a brief example?"


async def test_vad_barge_in_cuts_speech():
    proc = CapturingProcessor(InterviewController(), make_state(), session_id="vad-1")
    proc._turn_state = SPEAKING
    proc._is_speaking = True
    proc._tts_started_at = time.monotonic() - 1.0
    from pipecat.processors.frame_processor import FrameDirection

    await proc.process_frame(VADUserStartedSpeakingFrame(), FrameDirection.DOWNSTREAM)
    assert proc._turn_state == LISTENING
    assert proc._barge_in_this_turn is True


async def test_time_up_closes():
    state = make_state()
    proc = CapturingProcessor(InterviewController(), state, session_id="time-1")
    proc._started_at = time.time() - 9.6 * 60
    answer = "I used Python and FastAPI to build the API and Docker to ship it."
    await proc._apply_decision({"action": "advance", "reason": "answered", "reply": "Got it."}, answer, "Q?")
    assert proc.interview_finished is True
    tts = [f for f, _ in proc.captured if isinstance(f, TTSSpeakFrame)]
    assert any("We're at time." in f.text for f in tts)
    assert any("concludes the interview" in f.text for f in tts)


async def test_llm_tools_optional():
    import os
    from dotenv import load_dotenv
    from interview.turn_router import classify_with_tools

    load_dotenv()
    if not os.getenv("GOOGLE_API_KEY"):
        print("SKIP: live Gemini tool call (no GOOGLE_API_KEY)")
        return
    decision = await classify_with_tools(
        question="How have you used Docker in your projects?",
        partial="",
        utterance="what do you mean by used, like in production?",
        interrupted=True,
        playback_pct=0.25,
    )
    print(f"LIVE tool decision: {decision}")
    assert decision["action"] in {"answer", "clarify"}
    assert decision["action"] != "advance"


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
    test_rules_skip()
    test_rules_end()
    test_rules_candidate_question_does_not_advance()
    test_rules_long_answer_not_treated_as_skip()
    test_meta_question_helper()
    test_decision_from_tool_mapping()
    await test_wait_does_not_replay_question()
    await test_answer_candidate_question_stays()
    await test_answer_mid_question_restates()
    await test_skip_does_not_store_utterance()
    await test_end_does_not_store_stop_phrase()
    test_rules_idk_is_answer_not_skip()
    test_restate_clips_long_question()
    await test_clarify_uses_short_restatement()
    await test_silence_nudge_once()
    test_rules_thin_answer_follow_up()
    test_rules_garbage_stt()
    await test_follow_up_does_not_advance()
    await test_vad_barge_in_cuts_speech()
    await test_time_up_closes()
    await test_llm_tools_optional()
    print("PASS: turn router + barge-in + tools")


if __name__ == "__main__":
    asyncio.run(main())
