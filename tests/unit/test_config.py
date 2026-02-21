"""Unit tests for archer.core.config."""

import yaml
import pytest

from archer.core.config import (
    ArcherConfig,
    AudioConfig,
    EmotionConfig,
    LLMConfig,
    PersonalityConfig,
    STTConfig,
    TTSConfig,
    _load_personality,
    _load_yaml,
    load_config,
)


class TestAudioConfig:
    def test_defaults(self):
        config = AudioConfig()
        assert config.sample_rate == 16000
        assert config.channels == 1

    def test_custom(self):
        config = AudioConfig(sample_rate=44100, channels=2)
        assert config.sample_rate == 44100
        assert config.channels == 2


class TestSTTConfig:
    def test_defaults(self):
        config = STTConfig()
        assert config.provider == "whisper"
        assert config.model == "base.en"
        assert config.fp16 is False
        assert config.min_audio_duration == 0.5
        assert config.silence_threshold == 0.001


class TestTTSConfig:
    def test_defaults(self):
        config = TTSConfig()
        assert config.provider == "pocket_tts"
        assert config.voice_sample is None
        assert config.voice_name is None
        assert config.cfg_weight == 0.5
        assert config.save_responses is False
        assert config.output_dir == "voices"


class TestLLMConfig:
    def test_defaults(self):
        config = LLMConfig()
        assert config.provider == "ollama"
        assert config.model == "gemma3"
        assert config.base_url == "http://localhost:11434"


class TestEmotionConfig:
    def test_defaults(self):
        config = EmotionConfig()
        assert config.base_exaggeration == 0.5
        assert config.max_exaggeration == 0.9
        assert config.min_exaggeration == 0.3
        assert config.keyword_boost == 0.1


class TestPersonalityConfig:
    def test_defaults(self):
        config = PersonalityConfig()
        assert config.name == "Archer"
        assert config.system_prompt == "You are a helpful AI assistant."
        assert config.emotional_keywords == {}
        assert isinstance(config.emotion, EmotionConfig)


class TestArcherConfig:
    def test_defaults(self):
        config = ArcherConfig()
        assert config.assistant_name == "Archer"
        assert config.personality_name == "archer"
        assert isinstance(config.audio, AudioConfig)
        assert isinstance(config.stt, STTConfig)
        assert isinstance(config.tts, TTSConfig)
        assert isinstance(config.llm, LLMConfig)
        assert isinstance(config.personality, PersonalityConfig)


class TestLoadYaml:
    def test_valid_file(self, tmp_path):
        yaml_file = tmp_path / "test.yaml"
        yaml_file.write_text("key: value\nnested:\n  a: 1\n")
        result = _load_yaml(yaml_file)
        assert result == {"key": "value", "nested": {"a": 1}}

    def test_empty_file(self, tmp_path):
        yaml_file = tmp_path / "empty.yaml"
        yaml_file.write_text("")
        result = _load_yaml(yaml_file)
        assert result == {}

    def test_list_values(self, tmp_path):
        yaml_file = tmp_path / "list.yaml"
        yaml_file.write_text("items:\n  - a\n  - b\n")
        result = _load_yaml(yaml_file)
        assert result == {"items": ["a", "b"]}


class TestLoadPersonality:
    def test_valid_personality(self, tmp_config_dir):
        personality_data = {
            "name": "TestBot",
            "system_prompt": "You are a test bot.",
            "emotional_keywords": {
                "positive": ["great"],
                "negative": ["bad"],
            },
            "emotion": {
                "base_exaggeration": 0.6,
                "max_exaggeration": 0.95,
                "min_exaggeration": 0.2,
                "keyword_boost": 0.15,
            },
        }
        p_file = tmp_config_dir / "personalities" / "testbot.yaml"
        p_file.write_text(yaml.dump(personality_data))

        result = _load_personality(tmp_config_dir, "testbot")

        assert result.name == "TestBot"
        assert result.system_prompt == "You are a test bot."
        assert result.emotional_keywords == {"positive": ["great"], "negative": ["bad"]}
        assert result.emotion.base_exaggeration == 0.6
        assert result.emotion.max_exaggeration == 0.95
        assert result.emotion.min_exaggeration == 0.2
        assert result.emotion.keyword_boost == 0.15

    def test_missing_personality_returns_default(self, tmp_config_dir):
        result = _load_personality(tmp_config_dir, "nonexistent")
        assert isinstance(result, PersonalityConfig)
        assert result.name == "Archer"
        assert result.system_prompt == "You are a helpful AI assistant."

    def test_partial_yaml_uses_defaults(self, tmp_config_dir):
        personality_data = {"name": "Partial", "system_prompt": "Partial bot."}
        p_file = tmp_config_dir / "personalities" / "partial.yaml"
        p_file.write_text(yaml.dump(personality_data))

        result = _load_personality(tmp_config_dir, "partial")

        assert result.name == "Partial"
        assert result.emotion.base_exaggeration == 0.5
        assert result.emotional_keywords == {}


class TestLoadConfig:
    def test_missing_file_returns_defaults(self):
        result = load_config("nonexistent_config_file_xyz.yaml")
        assert isinstance(result, ArcherConfig)
        assert result.assistant_name == "Archer"

    def test_full_config(self, tmp_config_dir):
        personality_data = {
            "name": "TestBot",
            "system_prompt": "Hello.",
            "emotional_keywords": {},
        }
        (tmp_config_dir / "personalities" / "mybot.yaml").write_text(
            yaml.dump(personality_data)
        )

        main_data = {
            "assistant": {"name": "MyBot", "personality": "mybot"},
            "audio": {"sample_rate": 44100, "channels": 2},
            "stt": {"provider": "whisper", "model": "small", "fp16": True},
            "tts": {"provider": "chatterbox", "cfg_weight": 0.7},
            "llm": {"provider": "ollama", "model": "llama3"},
        }
        config_file = tmp_config_dir / "config.yaml"
        config_file.write_text(yaml.dump(main_data))

        result = load_config(str(config_file))

        assert result.assistant_name == "MyBot"
        assert result.personality_name == "mybot"
        assert result.audio.sample_rate == 44100
        assert result.audio.channels == 2
        assert result.stt.model == "small"
        assert result.stt.fp16 is True
        assert result.tts.provider == "chatterbox"
        assert result.tts.cfg_weight == 0.7
        assert result.llm.model == "llama3"
        assert result.personality.name == "TestBot"

    def test_partial_yaml_uses_defaults(self, tmp_config_dir):
        main_data = {"assistant": {"name": "Custom"}}
        config_file = tmp_config_dir / "config.yaml"
        config_file.write_text(yaml.dump(main_data))

        result = load_config(str(config_file))

        assert result.assistant_name == "Custom"
        assert result.stt.provider == "whisper"
        assert result.tts.cfg_weight == 0.5
        assert result.llm.model == "gemma3"
