"""Configuration loading and dataclasses for Archer Voice Assistant."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class AudioConfig:
    """Audio recording/playback configuration."""

    sample_rate: int = 16000
    channels: int = 1


@dataclass
class STTConfig:
    """Speech-to-Text configuration."""

    provider: str = "whisper"
    model: str = "base.en"
    fp16: bool = False
    min_audio_duration: float = 0.5  # seconds; shorter clips are rejected
    silence_threshold: float = 0.001  # RMS amplitude below this is treated as silence


@dataclass
class TTSConfig:
    """Text-to-Speech configuration."""

    provider: str = "pocket_tts"
    voice_sample: str | None = None  # Path to audio file for voice cloning
    voice_name: str | None = None  # Built-in Pocket TTS voice (alba, marius, etc.)
    cfg_weight: float = 0.5
    save_responses: bool = False
    output_dir: str = "voices"


@dataclass
class LLMConfig:
    """Language Model configuration."""

    provider: str = "ollama"
    model: str = "gemma3"
    base_url: str = "http://localhost:11434"


@dataclass
class EmotionConfig:
    """Emotion control settings for TTS."""

    base_exaggeration: float = 0.5
    max_exaggeration: float = 0.9
    min_exaggeration: float = 0.3
    keyword_boost: float = 0.1


@dataclass
class PersonalityConfig:
    """Personality configuration."""

    name: str = "Archer"
    system_prompt: str = "You are a helpful AI assistant."
    emotional_keywords: dict[str, list[str]] = field(default_factory=dict)
    emotion: EmotionConfig = field(default_factory=EmotionConfig)


@dataclass
class ArcherConfig:
    """Top-level configuration for Archer Voice Assistant."""

    assistant_name: str = "Archer"
    personality_name: str = "archer"
    audio: AudioConfig = field(default_factory=AudioConfig)
    stt: STTConfig = field(default_factory=STTConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    personality: PersonalityConfig = field(default_factory=PersonalityConfig)


def _load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML file and return its contents."""
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _load_personality(config_dir: Path, personality_name: str) -> PersonalityConfig:
    """Load personality configuration from file."""
    personality_path = config_dir / "personalities" / f"{personality_name}.yaml"

    if not personality_path.exists():
        print(f"Warning: Personality file not found: {personality_path}")
        return PersonalityConfig()

    data = _load_yaml(personality_path)

    emotion_data = data.get("emotion", {})
    emotion_config = EmotionConfig(
        base_exaggeration=emotion_data.get("base_exaggeration", 0.5),
        max_exaggeration=emotion_data.get("max_exaggeration", 0.9),
        min_exaggeration=emotion_data.get("min_exaggeration", 0.3),
        keyword_boost=emotion_data.get("keyword_boost", 0.1),
    )

    return PersonalityConfig(
        name=data.get("name", "Archer"),
        system_prompt=data.get("system_prompt", "You are a helpful AI assistant."),
        emotional_keywords=data.get("emotional_keywords", {}),
        emotion=emotion_config,
    )


def load_config(config_path: str = "config/default.yaml") -> ArcherConfig:
    """
    Load Archer configuration from YAML file.

    Args:
        config_path: Path to the main configuration file.

    Returns:
        ArcherConfig with all settings loaded.
    """
    config_file = Path(config_path)

    if not config_file.exists():
        print(f"Warning: Config file not found: {config_path}, using defaults")
        return ArcherConfig()

    data = _load_yaml(config_file)
    config_dir = config_file.parent

    # Parse assistant section
    assistant_data = data.get("assistant", {})
    assistant_name = assistant_data.get("name", "Archer")
    personality_name = assistant_data.get("personality", "archer")

    # Parse audio section
    audio_data = data.get("audio", {})
    audio_config = AudioConfig(
        sample_rate=audio_data.get("sample_rate", 16000),
        channels=audio_data.get("channels", 1),
    )

    # Parse STT section
    stt_data = data.get("stt", {})
    stt_config = STTConfig(
        provider=stt_data.get("provider", "whisper"),
        model=stt_data.get("model", "base.en"),
        fp16=stt_data.get("fp16", False),
        min_audio_duration=stt_data.get("min_audio_duration", 0.5),
        silence_threshold=stt_data.get("silence_threshold", 0.001),
    )

    # Parse TTS section
    tts_data = data.get("tts", {})
    tts_config = TTSConfig(
        provider=tts_data.get("provider", "pocket_tts"),
        voice_sample=tts_data.get("voice_sample"),
        voice_name=tts_data.get("voice_name"),
        cfg_weight=tts_data.get("cfg_weight", 0.5),
        save_responses=tts_data.get("save_responses", False),
        output_dir=tts_data.get("output_dir", "voices"),
    )

    # Parse LLM section
    llm_data = data.get("llm", {})
    llm_config = LLMConfig(
        provider=llm_data.get("provider", "ollama"),
        model=llm_data.get("model", "gemma3"),
        base_url=llm_data.get("base_url", "http://localhost:11434"),
    )

    # Load personality
    personality_config = _load_personality(config_dir, personality_name)

    return ArcherConfig(
        assistant_name=assistant_name,
        personality_name=personality_name,
        audio=audio_config,
        stt=stt_config,
        tts=tts_config,
        llm=llm_config,
        personality=personality_config,
    )
