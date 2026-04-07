# AAI Chatbot

An intelligent college assistant chatbot designed as a portfolio-grade AI systems project.

This project demonstrates how to combine traditional NLP, LLM orchestration, and product-focused UX to build a reliable, domain-aware assistant instead of a generic chatbot.

## Why This Project Stands Out

- Built a guided LLM system, not just a prompt wrapper
- Combined ensemble intent classification + emotion detection + scope control
- Added production-minded reliability: retries, API key rotation, fallback model path
- Implemented conversation context, logging, and analytics for observability
- Shipped both Streamlit product UI and CLI workflow for fast testing

## What Problem It Solves

Students ask repetitive but high-stakes questions (fees, exams, placements, hostel, faculty), often in mixed language styles and unclear phrasing.

AAI Chatbot addresses this by:

- grounding answers in local college data
- detecting low-confidence queries and asking for clarification
- adapting tone using emotion detection
- rejecting out-of-domain requests safely and consistently

## Core Engineering Highlights

### 1. Guided Response Pipeline

Instead of directly sending user text to an LLM, the request flows through:

1. Intent classification (ensemble)
2. Emotion detection
3. Scope detection
4. Prompt engineering with context + grounded knowledge
5. LLM generation (Groq primary, Gemini fallback)
6. SQLite interaction logging

### 2. Reliability Features

- Groq key rotation support (`GROQ_API_KEY_1`, `GROQ_API_KEY_2`, ...)
- Exponential backoff + jitter retry strategy
- Request throttling and concurrent request control
- Fallback path to Gemini when primary fails

### 3. Product and UX Focus

- Modern Streamlit chat UI
- Debug panel for intent/emotion/source visibility
- Time-aware greeting and session handling
- End-to-end test scripts for realistic prompt scenarios

## Tech Stack

- Python 3.8+
- Streamlit
- scikit-learn, sentence-transformers, transformers
- Groq API (`llama-3.1-8b-instant`)
- Google GenAI fallback (`gemini-2.5-flash`)
- SQLite

## Repository Map

Application:

- `app.py` - Streamlit interface
- `main.py` - CLI mode, training, log viewing

AI Pipeline:

- `intent_model.py` - intent classifier (ensemble)
- `emotion_detector.py` - emotion analysis
- `scope_detector.py` - in-scope/out-of-scope logic
- `prompt_engineering.py` - structured prompts
- `llm_handler.py` - model calls, fallback, retry, key rotation
- `context_manager.py` - multi-turn memory

Data + Persistence:

- `data/intents.json`
- `data/college_data.json`
- `database.py`

Testing + Diagnostics:

- `tests/`
- `enhanced_test_suite.py`
- `run_test_suite.py`
- `tests/run_all_tests.py`
- `view_database.py`

## Quick Demo Setup

1. Create environment and install dependencies:

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Configure keys:

```bash
copy .env.example .env
```

Set at least one:

- `GROQ_API_KEY` or `GROQ_API_KEY_1`

Optional:

- `GEMINI_API_KEY`

3. Run app:

```bash
streamlit run app.py
```

Open `http://localhost:8501`

## CLI Commands

```bash
# interactive chatbot
python main.py

# train intent model
python main.py --train

# inspect recent logs
python main.py --logs
python main.py --logs 50
```

## Testing

```bash
pytest
python tests/run_all_tests.py
python enhanced_test_suite.py
python run_test_suite.py
```

## Resume-Friendly Contributions

- Engineered a domain-aware AI assistant using a multi-stage NLP + LLM pipeline
- Improved robustness through confidence gating, scope control, and fallback orchestration
- Built observability via SQLite logging and analytics-friendly structured records
- Delivered a complete prototype with user UI, CLI tooling, and automated test suites

## Role-Specific Pitch (All 3)

### ML Engineer

- Designed an ensemble intent classifier (semantic + TF-IDF) to improve robustness on noisy user inputs.
- Integrated emotion detection and confidence-aware clarification to reduce low-quality model outputs.
- Built a controllable inference pipeline where classical NLP signals guide LLM generation behavior.
- Created repeatable evaluation flow using pytest suites and scenario-driven stress tests.

### Backend Engineer

- Implemented resilient model-serving orchestration with retries, throttling, and fallback routing.
- Added API key rotation and concurrency safeguards for stable behavior under burst traffic.
- Built structured SQLite logging and session-level persistence for operational visibility.
- Delivered dual runtime surfaces (Streamlit + CLI) with modular, testable service components.

### Applied AI Engineer

- Built an end-to-end domain assistant that combines intent, emotion, scope, context, and grounded prompts.
- Reduced hallucination risk through scope filtering and data-grounded prompt construction.
- Tuned user experience via emotion-aware tone and low-confidence clarification loops.
- Shipped a practical AI product with production-minded reliability patterns and measurable test coverage.

## Notes

- Generated artifacts like `intent_model.pkl` and `chatbot.db` are created during usage/training.
- No explicit repository license file is currently present.