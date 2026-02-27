# Archer Voice Assistant

A modular, local voice assistant powered by Ollama, Whisper, and Kyutai Pocket TTS. Runs entirely offline on your computer.

## Features

- 🎤 **Voice Input** - OpenAI Whisper for speech-to-text
- 🤖 **Local LLM** - Ollama for offline language model inference
- 🔊 **Fast Speech** - Kyutai Pocket TTS (~6x real-time on CPU, ~200ms first-audio latency)
- 🎭 **Voice Cloning** - Clone any voice from a WAV/MP3 sample (requires HuggingFace login)
- 👂 **Hands-Free Listening** - Silero VAD with wake word detection ("hey archer") or continuous mode
- 🔀 **Streaming Playback** - TTS synthesis and audio playback run concurrently for lower latency
- ⚙️ **Configurable** - YAML-based configuration for easy customization
- 🔌 **Modular** - Abstract interfaces for easy provider swapping

## Quick Start

### Prerequisites

- Python 3.11+
- [Ollama](https://ollama.ai) installed and running
- A microphone

### Installation

```bash
# Clone the repository
git clone https://github.com/soos3d/local-archer

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e .

# Download NLTK data (for sentence tokenization)
python -c "import nltk; nltk.download('punkt_tab')"
```

### Pull an Ollama Model

```bash
ollama pull gemma3  # or any model you prefer
```

### Run Archer

```bash
# Default: press-Enter-to-talk mode
python app.py

# Or use the installed CLI entry point
archer
```

## Usage

### Listening Modes

Archer supports three ways to capture your voice:

| Mode | How to start | How it works |
|---|---|---|
| **Press-to-talk** (default) | `python app.py` | Press Enter to start recording, press Enter again to stop. |
| **Wake word** | `python app.py --vad` | Say "hey archer" to activate, then speak naturally. Conversation stays active until a configurable timeout (default 30s of silence). |
| **Continuous** | `python app.py --continuous` | Always listening — every detected utterance is processed. No wake word needed. |

### Command Line Options

```bash
python app.py [OPTIONS]
```

| Flag | Description |
|---|---|
| `--config PATH` | Path to a YAML config file (default: `config/default.yaml`) |
| `--model NAME` | Override the LLM model (e.g. `llama3`, `gemma3`) |
| `--voice PATH` | Override voice sample for cloning (WAV/MP3) |
| `--exaggeration FLOAT` | Override base emotion exaggeration (0.0–1.0) |
| `--save-voice` | Save generated voice responses to the `voices/` directory |
| `--vad` | Enable VAD-based listening with wake word detection |
| `--continuous` | Enable continuous listening (always-on, implies `--vad`) |

### Examples

```bash
# Use a custom config
python app.py --config config/custom.yaml

# Voice cloning with a different model (requires HF login — see below)
python app.py --voice path/to/sample.wav --model llama3

# Hands-free with wake word, save all responses
python app.py --vad --save-voice

# Always-on listening with higher emotion
python app.py --continuous --exaggeration 0.7

# Combine everything
python app.py --vad --model gemma3 --voice my_voice.wav --save-voice
```

## Configuration

### Main Config (`config/default.yaml`)

```yaml
assistant:
  name: "Archer"
  personality: "archer" # References config/personalities/archer.yaml

stt:
  provider: "whisper"
  model: "base.en"            # Options: tiny, base, small, medium, large
  min_audio_duration: 0.5     # Reject clips shorter than this (seconds)
  silence_threshold: 0.001    # RMS below this is treated as silence

tts:
  provider: "pocket_tts"      # Or "chatterbox" for legacy provider
  voice_sample: null           # Path to WAV/MP3 for voice cloning (overrides voice_name)
  voice_name: null             # Built-in voice: alba, marius, javert, jean,
                               #   fantine, cosette, eponine, azelma
  cfg_weight: 0.5             # Pacing control (chatterbox only)
  save_responses: false
  output_dir: "voices"

llm:
  provider: "ollama"
  model: "gemma3"
  base_url: "http://localhost:11434"

listening:
  vad_enabled: false
  mode: "wake_word"              # "wake_word" or "continuous"
  wake_word: "hey archer"
  vad_threshold: 0.5             # VAD confidence (0.0-1.0)
  silence_duration_s: 1.2        # Seconds of silence before ending an utterance
  pre_buffer_ms: 300             # Audio kept before speech onset (avoids clipping)
  max_utterance_s: 30.0          # Maximum single utterance length
  conversation_timeout_s: 30.0   # Silence before wake word mode re-engages
  post_speech_delay_s: 0.8       # Mic suppression after TTS to avoid echo
```

### Personality Config (`config/personalities/archer.yaml`)

Customize Archer's behavior:

```yaml
name: "Archer"
system_prompt: |
  You are Archer, a helpful and intelligent AI assistant...

emotional_keywords:
  positive: ["amazing", "wonderful", "excited"]
  negative: ["terrible", "sad", "frustrated"]

emotion:
  base_exaggeration: 0.5
  max_exaggeration: 0.9
```

## Project Structure

```
archer/
├── core/
│   ├── config.py          # Configuration dataclasses & YAML loading
│   └── assistant.py       # Main orchestrator & conversation loop
├── audio/
│   ├── recorder.py        # Microphone input (press-to-talk & VAD streaming)
│   ├── player.py          # Audio output
│   ├── vad.py             # Silero VAD wrapper
│   └── pipeline.py        # Streaming TTS + playback pipeline
├── stt/
│   ├── base.py            # Abstract STT interface
│   └── whisper_stt.py     # Whisper implementation
├── tts/
│   ├── base.py            # Abstract TTS interface
│   ├── pocket_tts.py      # Kyutai Pocket TTS (default)
│   └── chatterbox_tts.py  # ChatterBox (legacy, GPU recommended)
├── llm/
│   ├── base.py            # Abstract LLM interface
│   └── ollama_llm.py      # Ollama implementation
└── tools/                 # Reserved for future API integrations
```

## TTS Providers

### Pocket TTS (default)

Kyutai's 100M-parameter model. Fast enough to run in real-time on CPU.

- ~6x real-time on Apple Silicon CPU
- ~200ms to first audio chunk
- 8 built-in voices; voice cloning via audio file (requires HuggingFace access)

**Voice cloning setup:**

1. Accept terms at [huggingface.co/kyutai/pocket-tts](https://huggingface.co/kyutai/pocket-tts)
2. Log in locally:
   ```bash
   uvx hf auth login
   ```
3. Set your voice sample in `config/default.yaml`:
   ```yaml
   tts:
     voice_sample: "path/to/sample.wav"
   ```

**Speed up repeated startups** by exporting the voice embedding once:

```python
python -c "
from archer.core.config import TTSConfig
from archer.tts.pocket_tts import PocketTTS
tts = PocketTTS(TTSConfig(voice_sample='path/to/sample.wav'))
tts.export_voice_state('path/to/sample.safetensors')
"
```

Then point `voice_sample` at the `.safetensors` file — startup skips audio processing entirely.

### ChatterBox (legacy)

Set `provider: "chatterbox"` in config. Slower but supports `exaggeration` (emotion intensity) and `cfg_weight` (pacing).

### Streaming Pipeline

Regardless of provider, Archer uses a streaming pipeline that synthesizes and plays audio concurrently. The first sentence plays while later sentences are still being generated, significantly reducing perceived latency.

## Listening Modes In Depth

### Press-to-Talk (default)

The simplest mode — no VAD model is loaded. Press Enter to start recording, press Enter again to stop.

### Wake Word Mode (`--vad`)

Loads Silero VAD and continuously listens for speech. When speech is detected, it's transcribed and checked for the wake word (default: "hey archer"). Once activated:

- Archer enters a **conversation** where follow-up utterances are processed directly without repeating the wake word.
- The conversation times out after `conversation_timeout_s` seconds of silence (default 30s), returning to wake word detection.
- The wake word match is flexible — punctuation and extra whitespace between words are tolerated (e.g. "hey, archer!" works).

### Continuous Mode (`--continuous`)

Every detected utterance is processed immediately with no wake word required. Useful for dedicated assistant setups where only the intended user is near the microphone.

Both VAD modes include echo suppression: the microphone is briefly muted after TTS playback (`post_speech_delay_s`) to prevent Archer from hearing its own voice.

## Tips

### Voice Cloning

- Use a clear 10–30 second audio sample
- Minimal background noise works best
- WAV and MP3 formats are both supported

### Performance

- Pocket TTS runs fast on CPU — no GPU needed
- Smaller Whisper models (`tiny.en`, `base.en`) reduce transcription latency
- Export voice state to `.safetensors` to avoid re-processing on each startup
- The streaming pipeline overlaps TTS synthesis with playback — first audio plays while later sentences are still being generated

### VAD Tuning

- Raise `vad_threshold` (e.g. 0.7) if Archer triggers on background noise
- Lower `silence_duration_s` for snappier turn-taking, raise it if Archer cuts you off
- Increase `conversation_timeout_s` if you want longer pauses between turns in wake word mode
- Adjust `post_speech_delay_s` if Archer picks up its own voice through the speakers

### CLI Entry Point

After `pip install -e .`, the `archer` command is available globally in your environment:

```bash
archer --vad --model llama3
```

This is equivalent to `python app.py` with the same flags.

## Troubleshooting

### Microphone not working

- Check system permissions for microphone access
- Verify your default audio input device

### Ollama connection failed

- Ensure Ollama is running: `ollama serve`
- Check the `base_url` in config matches Ollama's address

### Voice cloning raises `ValueError: VOICE_CLONING_UNSUPPORTED`

- Accept the license at [huggingface.co/kyutai/pocket-tts](https://huggingface.co/kyutai/pocket-tts)
- Log in: `uvx hf auth login`

### Import errors

- Ensure virtual environment is activated
- Run `pip install -e .` to reinstall the package

## License

MIT License - See [LICENSE](LICENSE) for details.

## Acknowledgments

- [OpenAI Whisper](https://github.com/openai/whisper) for speech recognition
- [Ollama](https://ollama.ai) for local LLM serving
- [Kyutai Pocket TTS](https://github.com/kyutai-labs/pocket-tts) for fast, local TTS with voice cloning
- [ChatterBox](https://github.com/resemble-ai/chatterbox) for the original TTS backend
- Original concept from [vndee/local-talking-llm](https://github.com/vndee/local-talking-llm)
