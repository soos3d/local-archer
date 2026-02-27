# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Archer** is a fully local voice assistant that chains three AI models: OpenAI Whisper (STT) → Ollama (LLM) → Pocket TTS/ChatterBox (TTS). All inference runs offline. The entry point is `app.py`.

## Commands

```bash
# Install
pip install -e .

# Install dev dependencies (pytest, pre-commit, etc.)
pip install -e ".[dev]"  # or: uv sync --group dev

# Download required NLTK data (one-time)
python -c "import nltk; nltk.download('punkt_tab')"

# Run the assistant (Ollama must be running: ollama serve)
python app.py                                              # press-to-talk (default)
python app.py --vad                                        # wake word mode ("hey archer")
python app.py --continuous                                 # always-on listening (implies --vad)
python app.py --config config/custom.yaml
python app.py --model llama3 --exaggeration 0.7 --save-voice
python app.py --voice path/to/sample.wav
archer --vad --model gemma3                                # installed CLI entry point

# Run all tests (enforces 80% coverage via pytest-cov)
pytest

# Run a single test file
pytest tests/unit/test_config.py

# Run a specific test
pytest tests/unit/test_config.py::test_function_name -v

# Lint (ruff + mypy + gitleaks via pre-commit)
pre-commit run --all-files
```

## Architecture

The pipeline is strictly sequential per turn: **record → transcribe → generate → synthesize → play**.

### Provider Pattern

Each layer (STT, TTS, LLM) follows the same pattern:
- Abstract base class in `archer/<layer>/base.py` defines the interface
- Concrete implementations in sibling files (e.g., `whisper_stt.py`, `pocket_tts.py`, `ollama_llm.py`)
- Factory functions (`create_stt`, `create_tts`, `create_llm`) in `archer/core/assistant.py` instantiate providers by name from config

To add a new provider: subclass `Base*`, implement abstract methods, add a branch in the corresponding `create_*` factory.

### Configuration

`archer/core/config.py` defines dataclasses (`ArcherConfig`, `STTConfig`, `TTSConfig`, `LLMConfig`, `PersonalityConfig`, `EmotionConfig`). `load_config()` merges `config/default.yaml` with `config/personalities/<name>.yaml`. CLI args override config values via direct attribute mutation in `app.py`.

### Key Behaviors

- **Emotion control**: `Assistant.analyze_emotion()` scans LLM output for keywords defined in personality YAML and adjusts the TTS `exaggeration` parameter (range 0.3–0.9).
- **Long TTS**: Text is split into sentences via NLTK `sent_tokenize`, synthesized individually, then concatenated with 250ms silence gaps before playback.
- **Conversation memory**: LangChain `InMemoryChatMessageHistory` keyed by `session_id`; resets on process restart.
- **Device acceleration**: Whisper and TTS models auto-detect CUDA/MPS.

### TTS Providers

- **Pocket TTS** (default, `provider: "pocket_tts"`): Fast (~6x real-time on CPU). Supports 8 built-in voices and voice cloning (requires HuggingFace login).
- **ChatterBox** (legacy, `provider: "chatterbox"`): Slower, supports `exaggeration` and `cfg_weight` controls.

## Testing

Tests live in `tests/unit/` and `tests/integration/`. pytest is configured in `pyproject.toml` with `--cov-fail-under=80`. Shared fixtures are in `tests/conftest.py`.

## Pre-commit Hooks

Configured in `.pre-commit-config.yaml`: ruff (lint + autofix), mypy (type checking, excludes tests), standard file fixers (trailing whitespace, EOF), and gitleaks (secret scanning).
