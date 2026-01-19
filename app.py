# InteReview AI

from __future__ import annotations

import threading
import time
from typing import Any, Dict, List

import numpy as np
import av
import streamlit as st
from streamlit_mic_recorder import speech_to_text
from streamlit.components.v1 import html

from streamlit_webrtc import (
    webrtc_streamer,
    WebRtcMode,
    RTCConfiguration,
    AudioProcessorBase,
    VideoProcessorBase,
)

from interview.config import settings, get_openai_client
from interview.questions import generate_next_question
from interview.engine import (
    tts_autoplay_html,
    score_full_interview,
)
from interview.vision import VisionAggregator, vision_available



# Constants
SR = 16000
SAMPLE_WIDTH = 2  # int16
BYTES_PER_SEC = SR * SAMPLE_WIDTH

# WebRTC config
RTC_CONFIG = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# Legacy cosntants
CHUNK_SECONDS = 2.0
OVERLAP_SECONDS = 0.5
CHUNK_BYTES = int(BYTES_PER_SEC * CHUNK_SECONDS)
OVERLAP_BYTES = int(BYTES_PER_SEC * OVERLAP_SECONDS)

MAX_ANSWER_SECONDS = 90
MAX_ANSWER_BYTES = BYTES_PER_SEC * MAX_ANSWER_SECONDS


# Helpers
def _jsonable(x: Any) -> Any:
    try:
        import numpy as _np

        if isinstance(x, (_np.integer, _np.floating)):
            return x.item()
    except Exception:
        pass
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    return str(x)


def section_title(title: str, emoji: str = "") -> None:
    st.markdown(f"### {emoji} {title}".strip())


def pill(label: str, value: str) -> None:
    st.markdown(
        f"""
        <div style="display:inline-block;padding:6px 10px;border-radius:999px;
        border:1px solid rgba(255,255,255,0.15);margin-right:8px;margin-bottom:8px;
        background:rgba(255,255,255,0.03);font-size:14px;">
        <b>{label}:</b> {value}
        </div>
        """,
        unsafe_allow_html=True,
    )


# Video processor (camera)
class SimpleVideoProcessor(VideoProcessorBase):
    def __init__(self) -> None:
        self._latest = None

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        self._latest = img
        return frame

    def get_latest(self):
        return self._latest


# State init
def init_state() -> None:
    ss = st.session_state
    ss.setdefault("stage", "setup")  # setup, media, question, finished
    ss.setdefault("profile", {})
    ss.setdefault("question_idx", 0)
    ss.setdefault("current_question", "")
    ss.setdefault("qa", [])
    ss.setdefault("stt_nonce", {})  # Reset counter per question for mic component

    ss.setdefault("vision", VisionAggregator())
    ss.setdefault("final_result", None)

    ss.setdefault("tts_cache", {})
    ss.setdefault("audio_enabled", False)

    # Mdia + timer state
    ss.setdefault("media_mode", None)          # "mic" or "mic+face"
    ss.setdefault("timer_seconds", 30)         # 30 or 60
    ss.setdefault("timer_start", None)         # float timestamp
    ss.setdefault("timer_question_idx", None)  # to reset per question
    ss.setdefault("timer_auto_submitted", False)


init_state()


# --------------------------
# Setup page
# --------------------------
def render_setup() -> None:
    st.title("🤖 Mock Interview Robot")
    st.caption("Mic + Camera interview practice with employer-style feedback.")

    c1, c2 = st.columns([1.25, 1], gap="large")
    with c1:
        section_title("Interview setup", "🧾")
        with st.form("setup_form"):
            job_title = st.text_input("Target role / job title", placeholder="e.g., Robotics Software Intern")
            job_field = st.selectbox("Job field", options=settings.job_fields, index=0)
            company_size = st.selectbox("Company size", ["Startup (1–10)", "Small (11–50)", "Mid (51–300)", "Large (301+)"])
            interview_style = st.selectbox(
                "Interview style",
                ["Balanced (Behavioral + Technical)", "Behavioral-heavy", "Technical-heavy"],
            )
            personality = st.selectbox(
                "Interviewer personality",
                ["Friendly", "Neutral", "Fast-paced & high standards", "Skeptical (but fair)"],
                index=1,
            )
            experience_level = st.selectbox(
                "Your level",
                ["Student / New Grad", "Junior (1–2 yrs)", "Intermediate (3–5 yrs)", "Senior (6+ yrs)"],
            )
            resume_notes = st.text_area("Anything the interviewer should know (optional)", height=120)
            n_questions = st.slider("Number of questions", 3, 8, 5)
            submitted = st.form_submit_button("🚀 Start interview", use_container_width=True)

        if submitted:
            st.session_state.profile = {
                "job_title": (job_title.strip() or "Intern"),
                "job_field": job_field,
                "company_size": company_size,
                "interview_style": interview_style,
                "personality": personality,
                "experience_level": experience_level,
                "resume_notes": resume_notes.strip(),
                "n_questions": int(n_questions),
            }
            st.session_state.stage = "media"
            st.session_state.question_idx = 0
            st.session_state.qa = []
            st.session_state.stt_nonce = {}

            st.session_state.final_result = None
            st.session_state.tts_cache = {}
            st.session_state.audio_enabled = False
            st.session_state.vision = VisionAggregator()

            # Pre-generate first question; actual interview starts after media setup
            st.session_state.current_question = generate_next_question(
                client=get_openai_client(),
                profile=st.session_state.profile,
                qa_history=st.session_state.qa,
                question_idx=0,
                n_questions=st.session_state.profile["n_questions"],
            )
            st.rerun()


# Media setup page
def render_media_setup() -> None:
    ss = st.session_state

    st.title("🎧 Recording setup")
    section_title("Recording options", "🎚️")

    media_choice = st.radio(
        "Choose how you want to be recorded:",
        options=["Microphone only", "Microphone + Camera"],
        key="media_choice_radio",
    )

    if media_choice == "Microphone only":
        ss.media_mode = "mic"
    elif media_choice == "Microphone + Camera":
        ss.media_mode = "mic+cam"

    timer_label = st.radio(
        "Answer time limit per question:",
        options=["30 seconds", "60 seconds"],
        index=0,
        key="timer_choice_radio",
    )
    ss.timer_seconds = 30 if timer_label.startswith("30") else 60

    st.caption(
        "Your microphone (and camera, if selected) will be used during each question. "
        "Please allow browser permissions if/when prompted."
    )

    if st.button("Start Interview", type="primary", use_container_width=True):
        if ss.media_mode is None:
            st.error("Please choose a recording mode before starting the interview.")
            return

        # Reset timer state at the start of the interview
        ss.timer_start = None
        ss.timer_question_idx = None
        ss.timer_auto_submitted = False

        ss.stage = "question"
        st.rerun()

# Submit helper that is reused by the button + timer
def submit_current_answer() -> None:
    ss = st.session_state

    profile = ss.profile
    q_idx = ss.question_idx
    n_questions = int(profile.get("n_questions", 5))
    question = ss.current_question

    answer_key = f"answer_text_{q_idx}"
    answer_text = (ss.get(answer_key, "") or "").strip()
    if not answer_text:
        st.error("No transcript captured yet. Click Start, speak, then click Stop.")
        return

    # Keep a lightweight "voice" payload for downstream scoring 
    # Note: Do not store audio
    voice_stats = {
        "stt_engine": "streamlit_mic_recorder",
        "words": int(len(answer_text.split())),
        "chars": int(len(answer_text)),
    }

    if vision_available:
        try:
            snap = ss.vision.snapshot_and_reset()
            face_stats = _jsonable(snap.to_dict() if hasattr(snap, "to_dict") else snap)
        except Exception:
            face_stats = {"vision_enabled": True, "error": "snapshot failed"}
    else:
        face_stats = {"vision_enabled": False}

    ss.qa.append(
        {"q": question, "a": answer_text, "voice": _jsonable(voice_stats), "face": _jsonable(face_stats)}
    )

    next_idx = q_idx + 1
    ss.question_idx = next_idx

    if next_idx >= n_questions:
        ss.stage = "finished"
        st.rerun()

    ss.current_question = generate_next_question(
        client=get_openai_client(),
        profile=profile,
        qa_history=ss.qa,
        question_idx=next_idx,
        n_questions=n_questions,
    )
    st.rerun()

# Interview page
def render_question() -> None:
    profile = st.session_state.profile
    q_idx = st.session_state.question_idx
    n_questions = int(profile.get("n_questions", 5))
    question = st.session_state.current_question

    st.title("🎤 Interview in progress")

    top = st.columns([1.2, 1, 1], gap="small")
    with top[0]:
        pill("Role", profile.get("job_title", ""))
    with top[1]:
        pill("Field", profile.get("job_field", ""))
    with top[2]:
        pill("Question", f"{q_idx + 1} / {n_questions}")

    st.divider()

    section_title("Question", "🗣️")
    st.markdown(
        f"""
        <div style="padding:18px;border-radius:18px;background:rgba(255,255,255,0.04);
        border:1px solid rgba(255,255,255,0.08);font-size:18px;line-height:1.35;">
        <b>Interviewer:</b> {question}
        </div>
        """,
        unsafe_allow_html=True,
    )

    # TTS cache per question
    q_key = f"q_{q_idx}"
    if q_key not in st.session_state.tts_cache:
        try:
            st.session_state.tts_cache[q_key] = tts_autoplay_html(get_openai_client(), question)
        except Exception as e:
            st.session_state.tts_cache[q_key] = f"<div style='color:#ffb4b4'>TTS error: {e}</div>"

    if st.session_state.audio_enabled:
        html(st.session_state.tts_cache[q_key], height=120)
    else:
        html(st.session_state.tts_cache[q_key].replace("autoplay", ""), height=120)

    if st.button("🔊 Replay question", use_container_width=True):
        html(st.session_state.tts_cache[q_key], height=120)

    st.divider()

    colL, colR = st.columns([1.1, 0.9], gap="large")

    # ---- MIC (Start/Stop Speech-to-Text) ----
    with colL:
        section_title("Mic (start/stop speech-to-text)", "🎙️")
        st.write("1. Click **Start talking**.")
        st.write("2. Allow microphone access if prompted.")
        st.write("3. Speak your full answer, then click **Stop**.")
        st.write("4. The transcript will be inserted into the box below.")

        answer_key = f"answer_text_{q_idx}"
        if answer_key not in st.session_state:
            st.session_state[answer_key] = ""

        # Per-question reset counter so the mic component can be restarted reliably.
        nonce = int(st.session_state.stt_nonce.get(q_idx, 0))
        stt_key = f"stt_q{q_idx}_{nonce}"

        if st.button("🔁 Record again (reset)", use_container_width=True, key=f"reset_stt_{q_idx}"):
            st.session_state.stt_nonce[q_idx] = nonce + 1
            st.session_state[answer_key] = ""
            st.rerun()

        stt_text = speech_to_text(
            language="en",
            start_prompt="🎙️ Start talking",
            stop_prompt="⏹️ Stop",
            just_once=True,
            use_container_width=True,
            key=stt_key,
        )

        # When the user clicks Stop, speech_to_text returns the transcript once.
        if stt_text:
            st.session_state[answer_key] = (stt_text or "").strip()

        st.text_area("Transcript", key=answer_key, height=220)

        if not (st.session_state.get(answer_key) or "").strip():
            st.caption("No transcript yet. Record your answer, then click Stop.")

    # CAMERA (gated by media_mode)
    with colR:
        section_title("Camera preview", "📷")
        media_mode = st.session_state.get("media_mode") or "mic"
        if media_mode == "mic+face":
            video_ctx = webrtc_streamer(
                key=f"cam_{q_idx}",
                mode=WebRtcMode.SENDRECV,
                rtc_configuration=RTC_CONFIG,
                media_stream_constraints={"video": True, "audio": False},
                async_processing=True,
                video_processor_factory=SimpleVideoProcessor,
            )

            # Feed frames into vision aggregator
            if vision_available and video_ctx and video_ctx.state.playing and video_ctx.video_processor:
                frame = video_ctx.video_processor.get_latest()
                if frame is not None:
                    try:
                        st.session_state.vision.update(frame)
                    except Exception:
                        pass
        else:
            st.caption("Camera disabled (microphone-only mode).")

    # TIMER (per question, auto-submit if it runs out)
    ss = st.session_state
    current_q_idx = q_idx

    if ss.timer_question_idx != current_q_idx:
        ss.timer_question_idx = current_q_idx
        ss.timer_start = time.time()
        ss.timer_auto_submitted = False

    if ss.timer_start is not None:
        elapsed = time.time() - ss.timer_start
        remaining = int(ss.timer_seconds - elapsed)
        if remaining < 0:
            remaining = 0

        col_time, col_bar = st.columns([1, 3])
        with col_time:
            st.metric("Time left (s)", remaining)
        with col_bar:
            st.progress(
                max(0.0, min(1.0, remaining / float(ss.timer_seconds))),
                text="Answer time remaining",
            )

        if remaining <= 0 and not ss.timer_auto_submitted:
            ss.timer_auto_submitted = True
            submit_current_answer()
            return

    st.divider()

    if st.button("✅ Submit Answer", type="primary", use_container_width=True):
        submit_current_answer()

# Finished page + scoring
def render_finished() -> None:
    st.title("🏁 Interview complete")

    if st.session_state.final_result is None:
        with st.spinner("Scoring your interview..."):
            try:
                st.session_state.final_result = score_full_interview(
                    client=get_openai_client(),
                    profile=st.session_state.profile,
                    qa_history=st.session_state.qa,
                )
            except Exception as e:
                st.session_state.final_result = {"error": str(e)}

    result = st.session_state.final_result or {}
    if "error" in result:
        st.error(f"Scoring failed: {result['error']}")
        return

    section_title("Results", "📊")
    st.json(result)

    st.divider()
    if st.button("↩️ Start a new interview", use_container_width=True):
        st.session_state.stage = "setup"
        st.session_state.question_idx = 0
        st.session_state.qa = []
        st.session_state.current_question = ""
        st.session_state.final_result = None
        st.session_state.tts_cache = {}
        st.session_state.stt_nonce = {}
        st.session_state.vision = VisionAggregator()
        st.rerun()

# Router
stage = st.session_state.stage
if stage == "setup":
    render_setup()
elif stage == "media":
    render_media_setup()
elif stage == "question":
    render_question()
elif stage == "finished":
    render_finished()
else:
    st.session_state.stage = "setup"
    st.rerun()
