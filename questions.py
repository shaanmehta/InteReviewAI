# interview/questions.py
from __future__ import annotations

from typing import Dict, Any, List
from openai import OpenAI

from interview.prompts import INTERVIEWER_SYSTEM_PROMPT
from interview.config import settings


def _build_interviewer_messages(
    profile: Dict[str, Any],
    qa_history: List[Dict[str, Any]],
    question_idx: int,
    n_questions: int,
) -> list[dict]:
    # Provide context compactly
    history_lines = []
    for i, item in enumerate(qa_history, start=1):
        q = item.get("q", "").strip()
        a = item.get("a", "").strip()
        history_lines.append(f"Q{i}: {q}\nA{i}: {a}")

    profile_block = {
        "job_title": profile.get("job_title", ""),
        "job_field": profile.get("job_field", ""),
        "company_size": profile.get("company_size", ""),
        "interview_style": profile.get("interview_style", ""),
        "interviewer_personality": profile.get("personality", ""),
        "candidate_experience_level": profile.get("experience_level", ""),
        "resume_notes": profile.get("resume_notes", ""),
        "question_index": question_idx,
        "total_questions": n_questions,
    }

    user_content = (
        "CANDIDATE_PROFILE:\n"
        f"{profile_block}\n\n"
        "Q/A HISTORY (most recent last):\n"
        + ("\n\n".join(history_lines) if history_lines else "(none yet)")
        + "\n\n"
        "Now generate the NEXT single interview question."
    )

    return [
        {"role": "system", "content": INTERVIEWER_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def generate_next_question(
    client: OpenAI,
    profile: Dict[str, Any],
    qa_history: List[Dict[str, Any]],
    question_idx: int,
    n_questions: int,
) -> str:
    messages = _build_interviewer_messages(profile, qa_history, question_idx, n_questions)
    resp = client.chat.completions.create(
        model=profile.get("model_override") or settings.model,
        messages=messages,
        temperature=0.7,
    )
    text = resp.choices[0].message.content.strip()

    # Defensive cleanup
    text = text.strip().strip('"').strip()
    if len(text) > 240:
        # keep it spoken-friendly
        text = text[:240].rsplit(" ", 1)[0].strip() + "…"
    return text
