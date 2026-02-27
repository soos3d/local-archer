"""Unit tests for archer.core.assistant."""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from archer.core.config import (
    ArcherConfig,
    EmotionConfig,
    ListeningConfig,
    LLMConfig,
    PersonalityConfig,
    STTConfig,
    TTSConfig,
)
from archer.core.assistant import create_llm, create_stt, create_tts, Assistant


def _make_config_with_emotions(**emotion_kwargs) -> ArcherConfig:
    """Build an ArcherConfig with emotional keywords."""
    config = ArcherConfig()
    config.personality = PersonalityConfig(
        name="Test",
        system_prompt="Test prompt.",
        emotional_keywords={
            "positive": ["amazing", "wonderful", "great"],
            "negative": ["terrible", "awful"],
            "emphasis": ["obviously", "absolutely"],
        },
        emotion=EmotionConfig(**emotion_kwargs) if emotion_kwargs else EmotionConfig(),
    )
    return config


def _make_assistant(config: ArcherConfig) -> Assistant:
    """Create an Assistant with all providers mocked."""
    with (
        patch("archer.core.assistant.create_stt", return_value=MagicMock()),
        patch("archer.core.assistant.create_tts", return_value=MagicMock()),
        patch("archer.core.assistant.create_llm", return_value=MagicMock()),
        patch("archer.core.assistant.AudioRecorder"),
        patch("archer.core.assistant.AudioPlayer"),
        patch("archer.core.assistant.SileroVAD"),
        patch("os.makedirs"),
    ):
        return Assistant(config)


class TestAnalyzeEmotion:
    def test_base_score_when_no_keywords_match(self):
        config = _make_config_with_emotions()
        assistant = _make_assistant(config)
        result = assistant.analyze_emotion("This is a neutral statement.")
        assert result == pytest.approx(0.5)

    def test_single_positive_keyword(self):
        config = _make_config_with_emotions()
        assistant = _make_assistant(config)
        result = assistant.analyze_emotion("That is amazing work!")
        assert result == pytest.approx(0.6)

    def test_multiple_keywords_same_category(self):
        config = _make_config_with_emotions()
        assistant = _make_assistant(config)
        result = assistant.analyze_emotion("That is amazing and wonderful, great job!")
        assert result == pytest.approx(0.8)

    def test_keywords_across_categories(self):
        config = _make_config_with_emotions()
        assistant = _make_assistant(config)
        # positive("amazing") + negative("terrible") + emphasis("obviously") = +3 boosts
        result = assistant.analyze_emotion("That is amazing but obviously terrible.")
        assert result == pytest.approx(0.8)

    def test_clamp_to_max(self):
        config = _make_config_with_emotions(
            base_exaggeration=0.5, max_exaggeration=0.7, keyword_boost=0.1
        )
        assistant = _make_assistant(config)
        # 7 keywords would push score far above max
        result = assistant.analyze_emotion(
            "amazing wonderful great obviously absolutely terrible awful"
        )
        assert result == pytest.approx(0.7)

    def test_clamp_to_min(self):
        config = _make_config_with_emotions(
            base_exaggeration=0.1, min_exaggeration=0.3, keyword_boost=0.1
        )
        assistant = _make_assistant(config)
        result = assistant.analyze_emotion("neutral text with no keywords")
        assert result == pytest.approx(0.3)

    def test_case_insensitive_matching(self):
        config = _make_config_with_emotions()
        assistant = _make_assistant(config)
        result = assistant.analyze_emotion("OBVIOUSLY this is case insensitive.")
        assert result == pytest.approx(0.6)

    def test_empty_keywords_config(self):
        config = ArcherConfig()
        assistant = _make_assistant(config)
        result = assistant.analyze_emotion("amazing wonderful text")
        assert result == pytest.approx(0.5)

    def test_empty_text(self):
        config = _make_config_with_emotions()
        assistant = _make_assistant(config)
        result = assistant.analyze_emotion("")
        assert result == pytest.approx(0.5)


class TestProcessInput:
    def test_process_input_uses_streaming_pipeline(self):
        config = _make_config_with_emotions()
        assistant = _make_assistant(config)

        audio = np.ones(16000, dtype=np.float32)
        assistant.stt.transcribe.return_value = "Hello"
        assistant.llm.stream.return_value = iter(["Hi there.", "How are you?"])

        with patch("archer.core.assistant.StreamingPipeline") as MockPipeline:
            mock_pipeline = MagicMock()
            mock_pipeline.run.return_value = np.ones(100, dtype=np.float32)
            MockPipeline.return_value = mock_pipeline

            assistant._process_input(audio)

            MockPipeline.assert_called_once_with(assistant.tts)
            mock_pipeline.run.assert_called_once()
            call_kwargs = mock_pipeline.run.call_args
            assert call_kwargs[0][0] == ["Hi there.", "How are you?"]

    def test_process_input_empty_audio_returns_early(self):
        config = _make_config_with_emotions()
        assistant = _make_assistant(config)

        audio = np.array([], dtype=np.float32)
        assistant._process_input(audio)
        assistant.stt.transcribe.assert_not_called()

    def test_process_input_empty_transcription_returns_early(self):
        config = _make_config_with_emotions()
        assistant = _make_assistant(config)

        audio = np.ones(16000, dtype=np.float32)
        assistant.stt.transcribe.return_value = ""

        assistant._process_input(audio)
        assistant.llm.stream.assert_not_called()

    def test_process_input_calls_stream_not_generate(self):
        config = _make_config_with_emotions()
        assistant = _make_assistant(config)

        audio = np.ones(16000, dtype=np.float32)
        assistant.stt.transcribe.return_value = "Hello"
        assistant.llm.stream.return_value = iter(["Response."])

        with patch("archer.core.assistant.StreamingPipeline") as MockPipeline:
            mock_pipeline = MagicMock()
            mock_pipeline.run.return_value = np.ones(100, dtype=np.float32)
            MockPipeline.return_value = mock_pipeline

            assistant._process_input(audio)

        assistant.llm.stream.assert_called_once()
        assistant.llm.generate.assert_not_called()

    def test_process_input_strips_stage_directions(self):
        config = _make_config_with_emotions()
        assistant = _make_assistant(config)

        audio = np.ones(16000, dtype=np.float32)
        assistant.stt.transcribe.return_value = "Hello"
        assistant.llm.stream.return_value = iter([
            "(A long dramatic sigh)",
            "Hello there.",
            "(chuckles) How are you?",
        ])

        with patch("archer.core.assistant.StreamingPipeline") as MockPipeline:
            mock_pipeline = MagicMock()
            mock_pipeline.run.return_value = np.ones(100, dtype=np.float32)
            MockPipeline.return_value = mock_pipeline

            assistant._process_input(audio)

            tts_sentences = mock_pipeline.run.call_args[0][0]
            # Pure stage direction "(A long dramatic sigh)" should be dropped entirely
            # "(chuckles) How are you?" should become "How are you?"
            assert tts_sentences == ["Hello there.", "How are you?"]


class TestCreateSTT:
    def test_creates_whisper_stt(self):
        config = STTConfig(provider="whisper")
        with patch("archer.core.assistant.WhisperSTT") as MockWhisper:
            mock_instance = MagicMock()
            MockWhisper.return_value = mock_instance
            result = create_stt(config)
            MockWhisper.assert_called_once_with(config)
            mock_instance.load_model.assert_called_once()
            assert result is mock_instance

    def test_raises_for_unknown_provider(self):
        config = STTConfig(provider="unknown_stt")
        with pytest.raises(ValueError, match="Unknown STT provider"):
            create_stt(config)


class TestCreateTTS:
    def test_creates_chatterbox_tts(self):
        config = TTSConfig(provider="chatterbox")
        with patch("archer.core.assistant.ChatterboxTTS") as MockChatterbox:
            mock_instance = MagicMock()
            MockChatterbox.return_value = mock_instance
            result = create_tts(config)
            MockChatterbox.assert_called_once_with(config)
            assert result is mock_instance

    def test_creates_pocket_tts(self):
        config = TTSConfig(provider="pocket_tts")
        with patch("archer.core.assistant.PocketTTS") as MockPocket:
            mock_instance = MagicMock()
            MockPocket.return_value = mock_instance
            result = create_tts(config)
            MockPocket.assert_called_once_with(config)
            assert result is mock_instance

    def test_raises_for_unknown_provider(self):
        config = TTSConfig(provider="unknown_tts")
        with pytest.raises(ValueError, match="Unknown TTS provider"):
            create_tts(config)


class TestCreateLLM:
    def test_creates_ollama_llm(self):
        config = LLMConfig(provider="ollama")
        personality = PersonalityConfig()
        with patch("archer.core.assistant.OllamaLLM") as MockOllama:
            mock_instance = MagicMock()
            MockOllama.return_value = mock_instance
            result = create_llm(config, personality)
            MockOllama.assert_called_once_with(config, personality)
            assert result is mock_instance

    def test_raises_for_unknown_provider(self):
        config = LLMConfig(provider="unknown_llm")
        personality = PersonalityConfig()
        with pytest.raises(ValueError, match="Unknown LLM provider"):
            create_llm(config, personality)


class TestProcessText:
    def test_process_text_calls_llm_and_tts(self):
        config = _make_config_with_emotions()
        assistant = _make_assistant(config)

        assistant.llm.stream.return_value = iter(["Hello!", "How are you?"])

        with patch("archer.core.assistant.StreamingPipeline") as MockPipeline:
            mock_pipeline = MagicMock()
            MockPipeline.return_value = mock_pipeline

            assistant._process_text("What is the weather?")

        assistant.llm.stream.assert_called_once()
        mock_pipeline.run.assert_called_once()

    def test_process_text_does_not_call_stt(self):
        config = _make_config_with_emotions()
        assistant = _make_assistant(config)

        assistant.llm.stream.return_value = iter(["Response."])

        with patch("archer.core.assistant.StreamingPipeline") as MockPipeline:
            MockPipeline.return_value = MagicMock()

            assistant._process_text("hello")

        assistant.stt.transcribe.assert_not_called()


class TestRunDispatch:
    def test_run_uses_legacy_loop_when_vad_disabled(self):
        config = ArcherConfig()
        assistant = _make_assistant(config)
        assistant.recorder.record_until_enter.side_effect = KeyboardInterrupt

        assistant.run()

        assistant.recorder.record_until_enter.assert_called_once()

    def test_run_uses_vad_loop_when_vad_enabled(self):
        config = ArcherConfig(
            listening=ListeningConfig(vad_enabled=True, mode="continuous"),
        )
        assistant = _make_assistant(config)

        with patch.object(assistant, "_run_vad_loop", side_effect=KeyboardInterrupt):
            assistant.run()

    def test_vad_loop_continuous_processes_utterance(self):
        config = ArcherConfig(
            listening=ListeningConfig(vad_enabled=True, mode="continuous"),
        )
        assistant = _make_assistant(config)

        utterance = np.ones(16000, dtype=np.float32)
        call_count = 0

        def fake_listen_continuous(vad, listening, paused):
            nonlocal call_count
            call_count += 1
            if call_count > 1:
                raise KeyboardInterrupt
            yield utterance

        assistant.recorder.listen_continuous = fake_listen_continuous

        with patch.object(assistant, "_process_input") as mock_process:
            try:
                assistant._run_vad_loop()
            except KeyboardInterrupt:
                pass

        mock_process.assert_called_once()

    def test_vad_loop_wake_word_strips_prefix(self):
        config = ArcherConfig(
            listening=ListeningConfig(
                vad_enabled=True,
                mode="wake_word",
                wake_word="hey archer",
            ),
        )
        assistant = _make_assistant(config)
        assistant.stt.transcribe.return_value = "hey archer what is the weather"

        utterance = np.ones(16000, dtype=np.float32)
        call_count = 0

        def fake_listen_continuous(vad, listening, paused):
            nonlocal call_count
            call_count += 1
            if call_count > 1:
                raise KeyboardInterrupt
            yield utterance

        assistant.recorder.listen_continuous = fake_listen_continuous

        with patch.object(assistant, "_process_text") as mock_process:
            try:
                assistant._run_vad_loop()
            except KeyboardInterrupt:
                pass

        mock_process.assert_called_once_with("what is the weather")

    def test_vad_loop_wake_word_only_waits_for_next(self):
        config = ArcherConfig(
            listening=ListeningConfig(
                vad_enabled=True,
                mode="wake_word",
                wake_word="hey archer",
            ),
        )
        assistant = _make_assistant(config)
        assistant.stt.transcribe.return_value = "hey archer"

        call_count = 0

        def fake_listen_continuous(vad, listening, paused):
            nonlocal call_count
            call_count += 1
            if call_count > 1:
                raise KeyboardInterrupt
            yield np.ones(16000, dtype=np.float32)

        assistant.recorder.listen_continuous = fake_listen_continuous

        with patch.object(assistant, "_process_text") as mock_process:
            try:
                assistant._run_vad_loop()
            except KeyboardInterrupt:
                pass

        mock_process.assert_not_called()

    def test_vad_loop_ignores_non_wake_word_speech(self):
        config = ArcherConfig(
            listening=ListeningConfig(
                vad_enabled=True,
                mode="wake_word",
                wake_word="hey archer",
            ),
        )
        assistant = _make_assistant(config)
        assistant.stt.transcribe.return_value = "hello world"

        call_count = 0

        def fake_listen_continuous(vad, listening, paused):
            nonlocal call_count
            call_count += 1
            if call_count > 1:
                raise KeyboardInterrupt
            yield np.ones(16000, dtype=np.float32)

        assistant.recorder.listen_continuous = fake_listen_continuous

        with patch.object(assistant, "_process_text") as mock_process:
            try:
                assistant._run_vad_loop()
            except KeyboardInterrupt:
                pass

        mock_process.assert_not_called()
