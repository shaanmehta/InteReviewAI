from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from openai import OpenAI


def ensure_env_loaded() -> None:

    # Load .env early
    project_root = Path(__file__).resolve().parent.parent  # repo root
    env_path = project_root / ".env"
    if env_path.exists():
        load_dotenv(dotenv_path=env_path, override=False)
    else:
        load_dotenv(override=False)


# Load env first so settings reads correct model names
ensure_env_loaded()


@dataclass(frozen=True)
class Settings:
    model: str
    tts_model: str
    stt_model: str
    job_fields: List[str]


def _read_job_fields() -> List[str]:
    return [
        "Accounting","Advanced Computing","Advertising","Aeronautical Engineering","Aerospace","Agriculture","Animation",
        "Applications Design and Data Analysis","Applied Computing","Applied Electronics","Applied Informatics",
        "Applied Mathematics","Applied Science","Applied Statistics","Architecture","Art","Artificial Intelligence",
        "Automation","Automotive Engineering","Big Data","Algorithms","Biochemistry","BioComputational Physics",
        "Bio-Electrical Engineering","Bioengineering","Bioinformatics","Biology","Biomedical Applications",
        "Biomedical Engineering","Business Administration","Business Analytics","Business Management","Catalysis Science",
        "Chartered Accountant","Chemical Engineering","Chemistry","Civil Engineering","Cognitive Science",
        "Communications","Complexity Science & Engineering","Computational Fluid Dynamics","Computational Mathematics",
        "Computational Science & Engineering","Computer & Info Science","Computer Application","Computer Engineering",
        "Computer Games","Computer Information Systems","Computer Networking","Computer Science",
        "Computer Science & Engineering","Computer System Design","Computer Vision and Machine Learning",
        "Control & Instrumentation","Control Science and Engineering","Data Science","Deep Learning","Design Technology",
        "Ecommerce","Economics","Education","Electrical & Computer Engineering","Electrical and Electronics Engineering",
        "Electrical Engineering","Electrical Engineering and Computer Science","Electronic Engineering","Electronics",
        "Electronics & Communication","Embedded System Design","Energy and Environmental Systems Engineering","English",
        "Entrepreneurship","Finance","Game Design","Geomatics Engineering","Graphic Design","High Performance Computing",
        "History","HumanComputer Interaction","Humanities","Human Resources","Images","Industrial Arts",
        "Industrial Engineering","Information systems & Technologies","Information Technology",
        "Innovation and Research Results Transfer","Interactive Telecommunications Program","Interdisciplinary Studies",
        "Law","Logic and Methodology of Science","Machine Learning","Manufacturing Engineering","Materials Science",
        "Mathematics","Mechanical Engineering","Mechatronics Engineering","Medicine","Microbiology","Nanotechnology",
        "Neuroscience","Operations Research","Philosophy","Physics","Political Science","Product Management",
        "Project Management","Psychology","Robotics","Software Engineering","Statistics","Systems Engineering",
        "UX / UI Design",
    ]


settings = Settings(
    model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
    tts_model=os.getenv("OPENAI_TTS_MODEL", "gpt-4o-mini-tts"),
    stt_model=os.getenv("OPENAI_STT_MODEL", "gpt-4o-mini-transcribe"),
    job_fields=_read_job_fields(),
)


def get_openai_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY not found. Put it in a .env file at the repo root:\n"
            "OPENAI_API_KEY=sk-...\n"
        )
    return OpenAI(api_key=api_key)
