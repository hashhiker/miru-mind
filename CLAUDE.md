# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Miru Mind is a German-language mental health companion chatbot running in the terminal. It maintains persistent memory of past sessions and mood ratings to provide personalized, context-aware conversations.

## Running the App

```bash
pip install -r requirements.txt
python main.py
```

Requires a `.env` file with `GROQ_API_KEY=your_key`.

## Architecture

The entire application is a single file: `main.py`. There is no build step.

**LLM backends** — controlled by `MODE` at line 36:
- `"groq"` — Groq API (default, requires `GROQ_API_KEY`)
- `"local"` — Ollama running at `OLLAMA_HOST`

**Persistence** — `data/history.json` stores two collections:
- `sessions`: list of past conversations (date + messages array)
- `moods`: list of mood logs (timestamp, value 1–10, optional note)

**System prompt construction** — `build_system_prompt(data)` enriches the base German-language persona with live context injected at runtime: mood trend over the last `MOOD_TREND_DAYS` days and summaries of the last `RECENT_SESSIONS_COUNT` sessions.

**In-app commands** (typed by the user during chat):
- `/beenden` — quit
- `/neu` — start a new session
- `/verlauf` — display mood history chart
- `/stimmung` — log a mood rating (1–10)

## Key Configuration Constants (main.py ~line 31–49)

| Constant | Purpose |
|---|---|
| `MODE` | `"groq"` or `"local"` |
| `GROQ_MODEL` | Groq model ID |
| `OLLAMA_MODEL` | Ollama model ID |
| `RECENT_SESSIONS_COUNT` | How many past sessions to inject into the system prompt |
| `MOOD_TREND_DAYS` | Lookback window for mood trend calculation |
| `DATA_FILE` | Path to the JSON persistence file |

## Language & Tone

The system prompt and all in-app text is in **German**. Miru's persona is warm, therapeutic, and designed for users with light depression or mood swings. Safety protocols are included for crisis situations (reference to emergency hotline). Keep any changes to the system prompt consistent with this tone and language.
