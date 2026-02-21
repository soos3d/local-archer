# Archer Voice Assistant

A modular, local voice assistant powered by Ollama, Whisper, and Kyutai Pocket TTS. Runs entirely offline on your computer.

## Features

- 🎤 **Voice Input** - OpenAI Whisper for speech-to-text
- 🤖 **Local LLM** - Ollama for offline language model inference
- 🔊 **Fast Speech** - Kyutai Pocket TTS (~6x real-time on CPU, ~200ms first-audio latency)
- 🎭 **Voice Cloning** - Clone any voice from a WAV/MP3 sample (requires HuggingFace login)
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
git clone https://github.com/your-username/archer-assistant.git
cd archer-assistant

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
python app.py
```

## Usage

### Basic Interaction

1. Run `python app.py`
2. Press Enter to start recording
3. Speak your message
4. Press Enter to stop recording
5. Archer transcribes, thinks, and responds with voice

### Command Line Options

```bash
# Use a different config file
python app.py --config config/custom.yaml

# Voice cloning (use a 10-30 second audio sample; requires HF login — see below)
python app.py --voice path/to/voice_sample.wav

# Use a different LLM model
python app.py --model llama3

# Save voice responses to files
python app.py --save-voice
```

## Configuration

### Main Config (`config/default.yaml`)

```yaml
assistant:
  name: "Archer"
  personality: "archer" # References config/personalities/archer.yaml

stt:
  provider: "whisper"
  model: "base.en" # Options: tiny, base, small, medium, large

tts:
  provider: "pocket_tts"         # Default. Use "chatterbox" to switch back.
  voice_sample: null             # Path to WAV/MP3 for voice cloning (overrides voice_name)
  voice_name: null               # Built-in voice: alba, marius, javert, jean,
                                 #   fantine, cosette, eponine, azelma
  cfg_weight: 0.5                # Pacing control (chatterbox only)

llm:
  provider: "ollama"
  model: "gemma3"
  base_url: "http://localhost:11434"
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
│   ├── config.py          # Configuration loading
│   └── assistant.py       # Main orchestrator
├── audio/
│   ├── recorder.py        # Microphone input
│   └── player.py          # Audio output
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

## Tips

### Voice Cloning

- Use a clear 10–30 second audio sample
- Minimal background noise works best
- WAV and MP3 formats are both supported

### Performance

- Pocket TTS runs fast on CPU — no GPU needed
- Smaller Whisper models (`tiny.en`, `base.en`) reduce transcription latency
- Export voice state to `.safetensors` to avoid re-processing on each startup

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
