# interview/engine.py
from __future__ import annotations

import base64
import io
import json
import wave
from typing import Any, Dict, List, Optional

from openai import OpenAI

from interview.config import settings
from interview.prompts import SCORER_SYSTEM_PROMPT


# -----------------------
# TTS (AI speaks)
# -----------------------
def tts_autoplay_html(client: OpenAI, text: str) -> str:
    """
    Returns audio HTML with autoplay + controls.
    Controls are critical because Safari often blocks autoplay unless user interacts.
    """
    def _read_audio_bytes(obj: Any) -> bytes:
        if hasattr(obj, "read"):
            return obj.read()
        if isinstance(obj, (bytes, bytearray)):
            return bytes(obj)
        for attr in ("content", "data"):
            if hasattr(obj, attr):
                v = getattr(obj, attr)
                if isinstance(v, (bytes, bytearray)):
                    return bytes(v)
        raise TypeError("Could not read audio bytes from TTS response.")

    audio_obj = None
    last_err: Optional[Exception] = None

    # Try response_format first (newer SDKs)
    try:
        audio_obj = client.audio.speech.create(
            model=settings.tts_model,
            voice="alloy",
            input=text,
            response_format="mp3",
        )
    except Exception as e:
        last_err = e

    # Fallback without response_format
    if audio_obj is None:
        try:
            audio_obj = client.audio.speech.create(
                model=settings.tts_model,
                voice="alloy",
                input=text,
            )
        except Exception as e:
            last_err = e

    if audio_obj is None:
        raise RuntimeError(f"TTS failed: {last_err}")

    mp3_bytes = _read_audio_bytes(audio_obj)
    b64 = base64.b64encode(mp3_bytes).decode("utf-8")

    # controls + autoplay + JS play attempt
    # (JS helps on Chrome; Safari still may require user click, but controls are visible)
    return f"""
    <audio id="q_audio" autoplay controls style="width:100%; max-width:560px;">
      <source src="data:audio/mpeg;base64,{b64}" type="audio/mpeg">
    </audio>
    <script>
      (function() {{
        const a = document.getElementById("q_audio");
        if (!a) return;
        const playTry = () => a.play().catch(() => {{}});
        // Try immediately and after a tick
        setTimeout(playTry, 50);
        setTimeout(playTry, 300);
      }})();
    </script>
    """


# -----------------------
# STT (user speaks)
# -----------------------
def transcribe_audio_bytes(client: OpenAI, pcm16_bytes: bytes) -> str:
    """
    Takes 16kHz mono PCM16 bytes and transcribes reliably on modern OpenAI SDK:
    We wrap into WAV BytesIO and pass as file-like object with a .name.
    """
    if not pcm16_bytes or len(pcm16_bytes) < int(16000 * 2 * 0.6):  # <0.6s
        return ""

    wav_buf = io.BytesIO()
    with wave.open(wav_buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(pcm16_bytes)
    wav_buf.seek(0)

    # OpenAI SDK expects file-like; name helps content-type inference
    wav_buf.name = "audio.wav"  # type: ignore[attr-defined]

    try:
        resp = client.audio.transcriptions.create(
            model=settings.stt_model,
            file=wav_buf,
        )
        if isinstance(resp, str):
            return resp.strip()
        if hasattr(resp, "text"):
            return str(resp.text).strip()
        return str(resp).strip()
    except Exception:
        return ""


def summarize_voice_stats(pcm16_bytes: bytes) -> Dict[str, Any]:
    duration_s = len(pcm16_bytes) / (16000 * 2) if pcm16_bytes else 0.0
    return {
        "duration_s": round(duration_s, 2),
        "audio_bytes": int(len(pcm16_bytes)),
        "note": "Heuristics only (add VAD/pitch later).",
    }


def score_full_interview(client: OpenAI, profile: Dict[str, Any], qa: List[Dict[str, Any]]) -> Dict[str, Any]:
    payload = {
        "candidate_profile": profile,
        "transcript": qa,
        "note": "voice/face stats are heuristics; treat cautiously.",
    }

    # Don't force response_format here; some accounts/models reject it.
    resp = client.chat.completions.create(
        model=settings.model,
        messages=[
            {"role": "system", "content": SCORER_SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
        temperature=0.2,
    )

    content = (resp.choices[0].message.content or "").strip()
    try:
        return json.loads(content)
    except Exception:
        return {"overall_score": None, "summary": content, "strengths": [], "improvements": [], "rubric": {}}
