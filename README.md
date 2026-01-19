# InteReviewAI

Mock interviewer that simulates job interviews and provides feedback/scoring.

## What’s in this repo
- `app.py` (entry point)
- Core modules: `engine.py`, `questions.py`, `prompts.py`, `scoring.py`
- Optional vision module: `vision.py`
- Configuration: `config.py` 

## Features
- Position-specific interview question generation + follow-ups
- Automated scoring / rubric-style feedback
- Optional camera/vision analysis

## Getting started
1. Clone the repository (git clone https://github.com/shaanmehta/AI-Mock-Interviewer.git)
2. Install dependencies (requirements.txt)
3. Create .env file containing OpenAPI key
4. Ensure .venv is created correctly within the project directory
5. In the project root, "streamlit run app.py"
