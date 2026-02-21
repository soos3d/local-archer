"""Unit tests for archer.audio.pipeline."""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from archer.audio.pipeline import StreamingPipeline


def _make_tts_mock(sample_rate: int = 22050) -> MagicMock:
    """Create a mock TTS provider."""
    tts = MagicMock()
    tts.sample_rate = sample_rate

    def fake_synthesize(text, **kwargs):
        audio = np.ones(100, dtype=np.float32) * 0.5
        return (sample_rate, audio)

    tts.synthesize.side_effect = fake_synthesize
    return tts


class TestStreamingPipelineRun:
    @patch("archer.audio.pipeline.sd")
    def test_returns_none_for_empty_sentences(self, mock_sd):
        tts = _make_tts_mock()
        pipeline = StreamingPipeline(tts)
        result = pipeline.run([])
        assert result is None
        mock_sd.play.assert_not_called()

    @patch("archer.audio.pipeline.sd")
    def test_single_sentence(self, mock_sd):
        tts = _make_tts_mock()
        pipeline = StreamingPipeline(tts)
        result = pipeline.run(["Hello world."])
        assert result is not None
        assert isinstance(result, np.ndarray)
        assert len(result) == 100
        tts.synthesize.assert_called_once()
        mock_sd.play.assert_called()

    @patch("archer.audio.pipeline.sd")
    def test_multiple_sentences_include_silence_gaps(self, mock_sd):
        tts = _make_tts_mock()
        pipeline = StreamingPipeline(tts)
        result = pipeline.run(["Hello.", "World.", "Test."])
        assert result is not None
        # 3 audio chunks + 2 silence gaps
        assert tts.synthesize.call_count == 3
        # Result should be longer than 3 * 100 due to silence gaps
        assert len(result) > 300

    @patch("archer.audio.pipeline.sd")
    def test_passes_tts_parameters(self, mock_sd):
        tts = _make_tts_mock()
        pipeline = StreamingPipeline(tts)
        pipeline.run(
            ["Hello."],
            voice_sample="voice.wav",
            exaggeration=0.8,
            cfg_weight=0.3,
        )
        tts.synthesize.assert_called_once_with(
            "Hello.",
            voice_sample="voice.wav",
            exaggeration=0.8,
            cfg_weight=0.3,
        )

    @patch("archer.audio.pipeline.sd")
    def test_tts_error_raises_runtime_error(self, mock_sd):
        tts = _make_tts_mock()
        tts.synthesize.side_effect = ValueError("TTS failed")
        pipeline = StreamingPipeline(tts)
        with pytest.raises(RuntimeError, match="Streaming pipeline error"):
            pipeline.run(["Hello."])

    @patch("archer.audio.pipeline.sd")
    def test_playback_error_raises_runtime_error(self, mock_sd):
        tts = _make_tts_mock()
        mock_sd.play.side_effect = RuntimeError("Audio device error")
        pipeline = StreamingPipeline(tts)
        with pytest.raises(RuntimeError, match="Streaming pipeline error"):
            pipeline.run(["Hello."])

    @patch("archer.audio.pipeline.sd")
    def test_playback_called_per_chunk(self, mock_sd):
        tts = _make_tts_mock()
        pipeline = StreamingPipeline(tts)
        pipeline.run(["First.", "Second."])
        # Each sentence gets a play call; second also gets a silence play
        assert mock_sd.play.call_count >= 2
        assert mock_sd.wait.call_count >= 2
