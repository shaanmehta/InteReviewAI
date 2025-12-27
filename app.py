"""
Streamlit AI Mock Interview Robot (web app phase)

Hard fixes:
- Live transcript is stable: chunking + overlap + no audio loss
- Audio processor uses recv_queued (no dropped-frame warning)
- No runaway memory: cap stored audio per answer
- Camera remains separate & stable
"""

from __future__ import annotations

import threading
import time
from typing import Any, Dict, List

import numpy as np
import av
import streamlit as st
from streamlit.components.v1 import html
from streamlit_autorefresh import st_autorefresh

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
    transcribe_audio_bytes,
    summarize_voice_stats,
    score_full_interview,
)
from interview.vision import VisionAggregator, vision_available


# --------------------------
# Constants (tune once)
# --------------------------
SR = 16000
BYTES_PER_SEC = SR * 2  # 16-bit mono
CHUNK_SECONDS = 2.5
OVERLAP_SECONDS = 0.5
CHUNK_BYTES = int(BYTES_PER_SEC * CHUNK_SECONDS)
OVERLAP_BYTES = int(BYTES_PER_SEC * OVERLAP_SECONDS)

# prevent runaway memory per answer (e.g., 90s max)
MAX_ANSWER_SECONDS = 90
MAX_ANSWER_BYTES = BYTES_PER_SEC * MAX_ANSWER_SECONDS


# --------------------------
# Helpers
# --------------------------
def _jsonable(x: Any) -> Any:
    try:
        from dataclasses import is_dataclass, asdict

        if is_dataclass(x):
            return asdict(x)
    except Exception:
        pass
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    return str(x)


def _merge_transcript(existing: str, new: str) -> str:
    existing = (existing or "").strip()
    new = (new or "").strip()
    if not new:
        return existing
    if not existing:
        return new
    # simple merge; avoids flicker
    return (existing + " " + new).strip()


def _pcm16_rms(pcm16: bytes) -> float:
    if not pcm16:
        return 0.0
    a = np.frombuffer(pcm16, dtype=np.int16).astype(np.float32)
    if a.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(a * a)))


# --------------------------
# WebRTC processors
# --------------------------
class BufferedAudioProcessor(AudioProcessorBase):
    """
    Robust mic buffer using recv_queued (prevents dropped-frame warning).
    Produces 16kHz mono s16 PCM bytes.
    """
    def __init__(self):
        self._lock = threading.Lock()
        self._pcm16 = bytearray()
        self._resampler = av.audio.resampler.AudioResampler(
            format="s16",
            layout="mono",
            rate=SR,
        )

    def _append_frame(self, frame: av.AudioFrame) -> None:
        frames = self._resampler.resample(frame)
        for f in frames:
            arr = f.to_ndarray()
            if arr.ndim == 2:
                arr = arr[0]
            self._pcm16.extend(arr.astype(np.int16).tobytes())

    def recv_queued(self, frames: List[av.AudioFrame]) -> List[av.AudioFrame]:
        try:
            with self._lock:
                for fr in frames:
                    self._append_frame(fr)
        except Exception:
            pass
        return frames

    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        try:
            with self._lock:
                self._append_frame(frame)
        except Exception:
            pass
        return frame

    def pop_all(self) -> bytes:
        with self._lock:
            data = bytes(self._pcm16)
            self._pcm16.clear()
        return data

    def size(self) -> int:
        with self._lock:
            return len(self._pcm16)


class LatestVideoProcessor(VideoProcessorBase):
    def __init__(self):
        self._lock = threading.Lock()
        self._latest_bgr = None

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        try:
            img = frame.to_ndarray(format="bgr24")
            with self._lock:
                self._latest_bgr = img
        except Exception:
            pass
        return frame

    def get_latest(self):
        with self._lock:
            return None if self._latest_bgr is None else self._latest_bgr.copy()


# --------------------------
# Streamlit config
# --------------------------
st.set_page_config(page_title="Mock Interview Robot", page_icon="🤖", layout="wide")

RTC_CONFIG = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302", "stun:stun1.l.google.com:19302"]}]}
)


# --------------------------
# Session state init
# --------------------------
def _init_state() -> None:
    ss = st.session_state
    ss.setdefault("stage", "setup")  # setup | question | finished
    ss.setdefault("profile", {})
    ss.setdefault("question_idx", 0)
    ss.setdefault("current_question", "")
    ss.setdefault("qa", [])

    ss.setdefault("live_transcript", "")
    ss.setdefault("answer_audio_bytes", b"")  # full answer audio (capped)
    ss.setdefault("pending_pcm", b"")         # rolling buffer for STT (never lost)
    ss.setdefault("last_stt_ts", 0.0)

    ss.setdefault("vision", VisionAggregator())
    ss.setdefault("final_result", None)

    ss.setdefault("tts_cache", {})
    ss.setdefault("audio_enabled", False)

_init_state()


# --------------------------
# UI helpers
# --------------------------
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


# --------------------------
# Sidebar
# --------------------------
with st.sidebar:
    st.markdown("## 🤖 Mock Interview Robot")
    st.caption(f"LLM: `{settings.model}`")
    st.caption(f"TTS: `{settings.tts_model}` | STT: `{settings.stt_model}`")

    if st.button("🔁 Reset"):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()


# --------------------------
# Setup
# --------------------------
def render_setup() -> None:
    st.title("🤖 Mock Interview Robot")
    st.caption("Voice + camera interview practice with employer-style feedback.")

    c1, c2 = st.columns([1.25, 1], gap="large")
    with c1:
        section_title("Interview setup", "🧾")
        with st.form("setup_form"):
            job_title = st.text_input("Target role / job title", placeholder="e.g., Robotics Software Intern")
            job_field = st.selectbox("Job field", options=settings.job_fields, index=0)
            company_size = st.selectbox("Company size", ["Startup (1–10)", "Small (11–50)", "Mid (51–300)", "Large (301+)"])
            interview_style = st.selectbox("Interview style", ["Mixed (Behavioral + Technical)", "Behavioral-heavy", "Technical-heavy"])
            personality = st.selectbox("Interviewer personality", ["Warm & supportive", "Neutral & professional", "Fast-paced & high standards", "Skeptical (but fair)"], index=1)
            experience_level = st.selectbox("Your level", ["Student / New Grad", "Junior (1–2 yrs)", "Intermediate (3–5 yrs)", "Senior (6+ yrs)"])
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
            st.session_state.stage = "question"
            st.session_state.question_idx = 0
            st.session_state.qa = []

            st.session_state.live_transcript = ""
            st.session_state.answer_audio_bytes = b""
            st.session_state.pending_pcm = b""
            st.session_state.last_stt_ts = 0.0

            st.session_state.final_result = None
            st.session_state.tts_cache = {}
            st.session_state.audio_enabled = False
            st.session_state.vision = VisionAggregator()

            st.session_state.current_question = generate_next_question(
                client=get_openai_client(),
                profile=st.session_state.profile,
                qa_history=st.session_state.qa,
                question_idx=0,
                n_questions=st.session_state.profile["n_questions"],
            )
            st.rerun()

    with c2:
        section_title("Notes", "ℹ️")
        st.markdown(
            """
- **Mic is audio-only WebRTC** (more reliable on Mac)
- Live transcript updates every ~2–3 seconds (stable, not spammy)
- Camera is separate and stays smooth
            """.strip()
        )


# --------------------------
# Interview page
# --------------------------
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

    # Autoplay requires a user gesture in many browsers.
    if not st.session_state.audio_enabled:
        if st.button("🔓 Enable Audio (one-time)", type="primary", use_container_width=True):
            st.session_state.audio_enabled = True
            st.rerun()
        st.info("Browsers often block autoplay until you click once.")

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

    # ---- MIC (audio-only) ----
    with colL:
        section_title("Mic (for live transcript)", "🎙️")
        audio_ctx = webrtc_streamer(
            key=f"mic_{q_idx}",
            mode=WebRtcMode.SENDONLY,
            rtc_configuration=RTC_CONFIG,
            media_stream_constraints={"audio": True, "video": False},
            async_processing=True,
            audio_processor_factory=BufferedAudioProcessor,
        )

        # refresh while playing
        if audio_ctx and audio_ctx.state.playing:
            st_autorefresh(interval=700, key=f"refresh_mic_{q_idx}")

        mic_buf = 0
        if audio_ctx and audio_ctx.state.playing and audio_ctx.audio_processor:
            proc: BufferedAudioProcessor = audio_ctx.audio_processor  # type: ignore

            # Pull everything accumulated since last rerun
            new_bytes = proc.pop_all()
            if new_bytes:
                # append to full answer (cap)
                st.session_state.answer_audio_bytes += new_bytes
                if len(st.session_state.answer_audio_bytes) > MAX_ANSWER_BYTES:
                    st.session_state.answer_audio_bytes = st.session_state.answer_audio_bytes[-MAX_ANSWER_BYTES:]

                # append to pending STT buffer (this is the important part)
                st.session_state.pending_pcm += new_bytes

                # cap pending too (keep last ~30s max)
                max_pending = BYTES_PER_SEC * 30
                if len(st.session_state.pending_pcm) > max_pending:
                    st.session_state.pending_pcm = st.session_state.pending_pcm[-max_pending:]

            mic_buf = proc.size() + len(st.session_state.pending_pcm)

            # Transcribe only when we have enough pending audio
            now = time.time()
            can_call = (now - float(st.session_state.last_stt_ts)) >= 1.4

            # While we have enough bytes, do at most ONE STT per rerun (keeps UI responsive)
            if len(st.session_state.pending_pcm) >= CHUNK_BYTES and can_call:
                st.session_state.last_stt_ts = now

                chunk = st.session_state.pending_pcm[:CHUNK_BYTES]

                # VAD-ish gate: ignore silence chunks
                if _pcm16_rms(chunk) >= 160:
                    try:
                        text = transcribe_audio_bytes(get_openai_client(), chunk)
                    except Exception:
                        text = ""
                    if text:
                        st.session_state.live_transcript = _merge_transcript(st.session_state.live_transcript, text)

                # Keep overlap to avoid cutting words
                st.session_state.pending_pcm = st.session_state.pending_pcm[CHUNK_BYTES - OVERLAP_BYTES :]

        section_title("Live transcript", "📝")
        st.markdown(
            f"""
            <div style="padding:14px;border-radius:14px;background:rgba(255,255,255,0.03);
            border:1px dashed rgba(255,255,255,0.18);min-height:110px;">
            <b>Live transcript:</b><br/>
            {st.session_state.live_transcript or "<span style='opacity:0.7'>Click Start and speak…</span>"}
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.caption(f"(debug) mic buffer: {mic_buf} bytes | saved answer bytes: {len(st.session_state.answer_audio_bytes)}")

    # ---- CAMERA ----
    with colR:
        section_title("Camera preview", "📷")
        video_ctx = webrtc_streamer(
            key=f"cam_{q_idx}",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTC_CONFIG,
            media_stream_constraints={"audio": False, "video": True},
            async_processing=True,
            video_processor_factory=LatestVideoProcessor,
            sendback_audio=False,
        )

        if vision_available() and video_ctx and video_ctx.state.playing and video_ctx.video_processor:
            frame = video_ctx.video_processor.get_latest()
            if frame is not None:
                try:
                    st.session_state.vision.update(frame)
                except Exception:
                    pass
        elif not vision_available():
            st.caption("Vision unavailable (cv2/mediapipe missing) — camera preview still works.")

    st.divider()

    if st.button("✅ Submit Answer", type="primary", use_container_width=True):
        # one last STT pass on whatever is pending (best effort)
        if len(st.session_state.pending_pcm) >= int(BYTES_PER_SEC * 1.0):
            tail = st.session_state.pending_pcm
            if _pcm16_rms(tail) >= 160:
                try:
                    tail_text = transcribe_audio_bytes(get_openai_client(), tail)
                except Exception:
                    tail_text = ""
                if tail_text:
                    st.session_state.live_transcript = _merge_transcript(st.session_state.live_transcript, tail_text)

        answer_text = (st.session_state.live_transcript or "").strip()
        if not answer_text:
            st.error("I didn't capture any transcript yet. Speak for a moment, then try again.")
            return

        voice_stats = summarize_voice_stats(st.session_state.answer_audio_bytes)

        if vision_available():
            try:
                snap = st.session_state.vision.snapshot_and_reset()
                face_stats = _jsonable(snap.to_dict() if hasattr(snap, "to_dict") else snap)
            except Exception:
                face_stats = {"vision_enabled": True, "error": "snapshot failed"}
        else:
            face_stats = {"vision_enabled": False}

        st.session_state.qa.append(
            {"q": question, "a": answer_text, "voice": _jsonable(voice_stats), "face": _jsonable(face_stats)}
        )

        # reset for next question
        st.session_state.live_transcript = ""
        st.session_state.answer_audio_bytes = b""
        st.session_state.pending_pcm = b""
        st.session_state.last_stt_ts = 0.0

        next_idx = q_idx + 1
        st.session_state.question_idx = next_idx

        if next_idx >= n_questions:
            st.session_state.stage = "finished"
            st.rerun()

        st.session_state.current_question = generate_next_question(
            client=get_openai_client(),
            profile=profile,
            qa_history=st.session_state.qa,
            question_idx=next_idx,
            n_questions=n_questions,
        )
        st.rerun()


# --------------------------
# Finished page + scoring
# --------------------------
def render_finished() -> None:
    st.title("🏁 Interview complete")

    section_title("Transcript", "🧾")
    for i, item in enumerate(st.session_state.qa, start=1):
        with st.expander(f"Q{i}: {item['q']}", expanded=(i == 1)):
            st.markdown(f"**Your answer:** {item['a']}")
            st.json({"voice": item.get("voice", {}), "face": item.get("face", {})})

    st.divider()

    if st.button("🏆 Get Final Score", type="primary", use_container_width=True):
        with st.spinner("Scoring interview…"):
            qa_safe = [{"q": x["q"], "a": x["a"], "voice": x["voice"], "face": x["face"]} for x in st.session_state.qa]
            result = score_full_interview(
                client=get_openai_client(),
                profile=st.session_state.profile,
                qa=qa_safe,
            )
        st.session_state.final_result = result
        st.rerun()

    if st.session_state.final_result:
        result = st.session_state.final_result
        score = result.get("overall_score")
        if score is not None:
            st.markdown(
                f"""
                <div style="padding:18px;border-radius:20px;background:rgba(0,0,0,0.2);
                border:1px solid rgba(255,255,255,0.12);">
                    <div style="font-size:46px;font-weight:800;line-height:1;">
                        {int(round(score))} / 100
                    </div>
                    <div style="opacity:0.8;margin-top:6px;">Overall interview score</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.error("Scoring failed or returned no overall_score.")
            st.write(result)

        st.divider()
        section_title("Actionable feedback", "🧠")
        st.write(result.get("summary", ""))


# --------------------------
# Router
# --------------------------
stage = st.session_state.stage
if stage == "setup":
    render_setup()
elif stage == "question":
    render_question()
elif stage == "finished":
    render_finished()
else:
    st.session_state.stage = "setup"
    st.rerun()
