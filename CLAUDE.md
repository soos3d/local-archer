# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Archer** is a fully local voice assistant that chains three AI models: OpenAI Whisper (STT) → Ollama (LLM) → ChatterBox (TTS). It runs offline; all inference is local. The entry point is `app.py`.

## Commands

```bash
# Install (prefer uv, or pip)
pip install -e .

# Download required NLTK data (one-time setup)
python -c "import nltk; nltk.download('punkt_tab')"

# Run the assistant
python app.py
python app.py --config config/custom.yaml
python app.py --model llama3 --exaggeration 0.7 --save-voice
python app.py --voice path/to/sample.wav   # voice cloning

# Lint (runs ruff + mypy + pre-commit hooks)
make lint
# or directly:
pre-commit run --all-files

# Ollama must be running before starting the assistant
ollama serve
ollama pull gemma3
```

## Architecture

The pipeline is strictly sequential per turn: record → transcribe → generate → synthesize → play.

```
app.py          CLI entry point; parses args, loads config, creates Assistant
archer/
  core/
    config.py       Dataclasses (ArcherConfig, STTConfig, TTSConfig, LLMConfig,
                    PersonalityConfig, EmotionConfig) + YAML loader.
                    Config merges config/default.yaml + config/personalities/<name>.yaml
    assistant.py    Assistant class: orchestrates the pipeline; factory functions
                    create_stt/create_tts/create_llm for provider selection
  audio/
    recorder.py     AudioRecorder — starts a background thread, stops on Enter key
    player.py       AudioPlayer — plays numpy audio arrays via sounddevice
  stt/
    base.py         BaseSTT abstract interface
    whisper_stt.py  WhisperSTT — wraps openai-whisper; call load_model() before use
  tts/
    base.py         BaseTTS abstract interface (synthesize + synthesize_long)
    chatterbox_tts.py  ChatterboxTTS — splits text into sentences with NLTK,
                       synthesizes each, concatenates audio arrays
  llm/
    base.py         BaseLLM abstract interface (generate, clear_history, set_system_prompt)
    ollama_llm.py   OllamaLLM — wraps langchain-ollama; maintains per-session
                    InMemoryChatMessageHistory; uses RunnableWithMessageHistory
  tools/            Empty; reserved for future API integrations
config/
  default.yaml                Main config (assistant name, provider settings)
  personalities/archer.yaml   System prompt + emotional_keywords + emotion params
```

## Extending Providers

Each layer (STT, TTS, LLM) uses an abstract base class. To add a new provider:
1. Create a new file in the relevant `archer/<layer>/` directory
2. Subclass the `Base*` class and implement its abstract methods
3. Add a new branch in the corresponding `create_*` factory function in `archer/core/assistant.py`

## Adding a Personality

Create `config/personalities/<name>.yaml` mirroring `archer.yaml`, then set `assistant.personality: <name>` in `config/default.yaml` (or pass `--config`).

## Key Behaviors

- **Emotion control**: `Assistant.analyze_emotion()` scans LLM output for keywords and adjusts ChatterBox's `exaggeration` parameter dynamically (range 0.3–0.9).
- **Long TTS**: `synthesize_long()` splits text into sentences via NLTK and concatenates audio to avoid context-length limits in ChatterBox.
- **Conversation memory**: LangChain `InMemoryChatMessageHistory` is keyed by `session_id`; clears on process restart.
- **Device acceleration**: Whisper and ChatterBox auto-detect CUDA/MPS; no manual configuration needed.

## Pre-commit Hooks

Configured in `.pre-commit-config.yaml`: ruff (linting + autofix), mypy (type checking), standard file fixers, and gitleaks (secret scanning).
