"""Integration tests for the full Archer pipeline (_process_input and run loop)."""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from archer.core.config import ArcherConfig, EmotionConfig, PersonalityConfig, TTSConfig


def _make_assistant(config=None):
    """Create a fully mocked Assistant for integration testing."""
    from archer.core.assistant import Assistant

    if config is None:
        config = ArcherConfig()

    mock_stt = MagicMock()
    mock_tts = MagicMock()
    mock_llm = MagicMock()
    mock_recorder = MagicMock()
    mock_player = MagicMock()

    fake_audio = np.zeros(16000, dtype=np.float32)
    mock_tts.sample_rate = 16000
    mock_tts.synthesize.return_value = (16000, fake_audio)

    with (
        patch("archer.core.assistant.create_stt", return_value=mock_stt),
        patch("archer.core.assistant.create_tts", return_value=mock_tts),
        patch("archer.core.assistant.create_llm", return_value=mock_llm),
        patch("archer.core.assistant.AudioRecorder", return_value=mock_recorder),
        patch("archer.core.assistant.AudioPlayer", return_value=mock_player),
        patch("os.makedirs"),
    ):
        assistant = Assistant(config)

    return assistant, mock_stt, mock_tts, mock_llm, mock_recorder, mock_player


class TestPipelineProcessInput:
    def test_full_pipeline_happy_path(self, sample_audio):
        assistant, mock_stt, mock_tts, mock_llm, _, mock_player = _make_assistant()
        mock_stt.transcribe.return_value = "What is the weather?"
        mock_llm.generate.return_value = "It is sunny today."

        assistant._process_input(sample_audio)

        mock_stt.transcribe.assert_called_once_with(sample_audio)
        mock_llm.generate.assert_called_once_with("What is the weather?", "archer_session")
        mock_tts.synthesize.assert_called_once()
        mock_player.play.assert_called_once()

    def test_empty_audio_skips_pipeline(self):
        assistant, mock_stt, mock_tts, mock_llm, _, mock_player = _make_assistant()

        assistant._process_input(np.array([], dtype=np.float32))

        mock_stt.transcribe.assert_not_called()
        mock_llm.generate.assert_not_called()
        mock_player.play.assert_not_called()

    def test_empty_transcription_skips_llm(self, sample_audio):
        assistant, mock_stt, mock_tts, mock_llm, _, mock_player = _make_assistant()
        mock_stt.transcribe.return_value = ""

        assistant._process_input(sample_audio)

        mock_stt.transcribe.assert_called_once()
        mock_llm.generate.assert_not_called()
        mock_player.play.assert_not_called()

    def test_emotion_boosts_exaggeration_for_keywords(self, sample_audio):
        config = ArcherConfig()
        config.personality = PersonalityConfig(
            emotional_keywords={"positive": ["amazing"]},
            emotion=EmotionConfig(base_exaggeration=0.5, keyword_boost=0.1),
        )
        assistant, mock_stt, mock_tts, mock_llm, _, _ = _make_assistant(config)
        mock_stt.transcribe.return_value = "test input"
        mock_llm.generate.return_value = "That is amazing!"

        assistant._process_input(sample_audio)

        call_kwargs = mock_tts.synthesize.call_args[1]
        assert call_kwargs["exaggeration"] == pytest.approx(0.6)

    def test_high_emotion_reduces_cfg_weight(self, sample_audio):
        config = ArcherConfig()
        config.tts = TTSConfig(cfg_weight=1.0)
        config.personality = PersonalityConfig(
            emotional_keywords={"positive": ["amazing", "wonderful", "incredible"]},
            emotion=EmotionConfig(base_exaggeration=0.5, keyword_boost=0.1),
        )
        assistant, mock_stt, mock_tts, mock_llm, _, _ = _make_assistant(config)
        mock_stt.transcribe.return_value = "test"
        # 3 positive keywords push exaggeration to 0.8 (> 0.6 threshold)
        mock_llm.generate.return_value = "That is amazing, wonderful, and incredible!"

        assistant._process_input(sample_audio)

        call_kwargs = mock_tts.synthesize.call_args[1]
        assert call_kwargs["cfg_weight"] == pytest.approx(0.8)  # 1.0 * 0.8

    def test_multi_sentence_response_calls_synthesize_per_sentence(self, sample_audio):
        assistant, mock_stt, mock_tts, mock_llm, _, mock_player = _make_assistant()
        mock_stt.transcribe.return_value = "tell me a story"
        mock_llm.generate.return_value = "Once upon a time. There was a dragon. The end."

        assistant._process_input(sample_audio)

        assert mock_tts.synthesize.call_count == 3
        mock_player.play.assert_called_once()

    def test_multi_sentence_player_receives_concatenated_audio(self, sample_audio):
        assistant, mock_stt, mock_tts, mock_llm, _, mock_player = _make_assistant()
        mock_stt.transcribe.return_value = "tell me"
        mock_llm.generate.return_value = "First sentence. Second sentence."

        assistant._process_input(sample_audio)

        played_audio = mock_player.play.call_args[0][0]
        # 2 sentences (16000 each) + 1 silence gap (4000) = 36000
        silence_len = int(0.25 * 16000)  # 4000
        expected_len = 16000 + silence_len + 16000
        assert len(played_audio) == expected_len

    def test_save_responses_calls_save_audio(self, sample_audio):
        config = ArcherConfig()
        config.tts = TTSConfig(save_responses=True, output_dir="test_voices")
        assistant, mock_stt, mock_tts, mock_llm, _, _ = _make_assistant(config)
        mock_stt.transcribe.return_value = "test"
        mock_llm.generate.return_value = "Hello."

        assistant._process_input(sample_audio)

        mock_tts.save_audio.assert_called_once()
        call_args = mock_tts.save_audio.call_args[0]
        assert "response_001.wav" in call_args[1]

    def test_response_count_increments_on_each_save(self, sample_audio):
        config = ArcherConfig()
        config.tts = TTSConfig(save_responses=True, output_dir="test_voices")
        assistant, mock_stt, mock_tts, mock_llm, _, _ = _make_assistant(config)
        mock_stt.transcribe.return_value = "test"
        mock_llm.generate.return_value = "Hello."

        assistant._process_input(sample_audio)
        assistant._process_input(sample_audio)

        assert assistant.response_count == 2
        second_call = mock_tts.save_audio.call_args_list[1][0]
        assert "response_002.wav" in second_call[1]


class TestRunLoop:
    def test_keyboard_interrupt_exits_gracefully(self):
        assistant, _, _, _, mock_recorder, _ = _make_assistant()
        mock_recorder.record_until_enter.side_effect = KeyboardInterrupt

        # Should not raise
        assistant.run()

        mock_recorder.record_until_enter.assert_called_once()
