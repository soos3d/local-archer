"""Unit tests for archer.stt.whisper_stt."""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from archer.core.config import STTConfig
from archer.stt.whisper_stt import WhisperSTT


def _make_stt(
    fp16: bool = False,
    min_duration: float = 0.5,
    threshold: float = 0.001,
) -> WhisperSTT:
    config = STTConfig(
        provider="whisper",
        model="base.en",
        fp16=fp16,
        min_audio_duration=min_duration,
        silence_threshold=threshold,
    )
    return WhisperSTT(config)


class TestWhisperSTTLoadModel:
    def test_load_model(self):
        stt = _make_stt()
        with patch("archer.stt.whisper_stt.whisper") as mock_whisper:
            mock_model = MagicMock()
            mock_whisper.load_model.return_value = mock_model
            stt.load_model()
            mock_whisper.load_model.assert_called_once_with("base.en")
            assert stt.model is mock_model

    def test_model_starts_as_none(self):
        stt = _make_stt()
        assert stt.model is None


class TestWhisperSTTTranscribe:
    def test_transcribe_normal_audio(self, sample_audio):
        stt = _make_stt()
        mock_model = MagicMock()
        mock_model.transcribe.return_value = {"text": "  hello world  "}
        stt.model = mock_model

        result = stt.transcribe(sample_audio)

        assert result == "hello world"
        mock_model.transcribe.assert_called_once()

    def test_transcribe_strips_whitespace(self, sample_audio):
        stt = _make_stt()
        mock_model = MagicMock()
        mock_model.transcribe.return_value = {"text": "\n  trimmed \t"}
        stt.model = mock_model

        result = stt.transcribe(sample_audio)
        assert result == "trimmed"

    def test_transcribe_too_short_returns_empty(self, short_audio):
        stt = _make_stt(min_duration=0.5)
        stt.model = MagicMock()

        result = stt.transcribe(short_audio)

        assert result == ""
        stt.model.transcribe.assert_not_called()

    def test_transcribe_silent_returns_empty(self, silent_audio):
        stt = _make_stt()
        stt.model = MagicMock()

        result = stt.transcribe(silent_audio)

        assert result == ""
        stt.model.transcribe.assert_not_called()

    def test_transcribe_auto_loads_model_when_none(self, sample_audio):
        stt = _make_stt()
        assert stt.model is None

        with patch("archer.stt.whisper_stt.whisper") as mock_whisper:
            mock_model = MagicMock()
            mock_model.transcribe.return_value = {"text": "hello"}
            mock_whisper.load_model.return_value = mock_model
            stt.transcribe(sample_audio)
            mock_whisper.load_model.assert_called_once()

    def test_transcribe_passes_fp16_true(self, sample_audio):
        stt = _make_stt(fp16=True)
        mock_model = MagicMock()
        mock_model.transcribe.return_value = {"text": "test"}
        stt.model = mock_model

        stt.transcribe(sample_audio)

        call_kwargs = mock_model.transcribe.call_args[1]
        assert call_kwargs["fp16"] is True

    def test_transcribe_passes_fp16_false(self, sample_audio):
        stt = _make_stt(fp16=False)
        mock_model = MagicMock()
        mock_model.transcribe.return_value = {"text": "test"}
        stt.model = mock_model

        stt.transcribe(sample_audio)

        call_kwargs = mock_model.transcribe.call_args[1]
        assert call_kwargs["fp16"] is False
