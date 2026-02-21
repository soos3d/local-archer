"""Unit tests for archer.tts.pocket_tts."""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from archer.core.config import TTSConfig


def _make_mock_pocket_model(sample_rate: int = 24000) -> MagicMock:
    """Create a mock TTSModel instance."""
    mock_model = MagicMock()
    mock_model.sample_rate = sample_rate
    fake_audio = np.zeros(sample_rate, dtype=np.float32)
    mock_tensor = MagicMock()
    mock_tensor.numpy.return_value = fake_audio
    mock_model.generate_audio.return_value = mock_tensor
    mock_model.get_state_for_audio_prompt.return_value = MagicMock()
    return mock_model


def _make_pocket_tts(voice_sample=None, voice_name=None, sample_rate=24000):
    """Create a PocketTTS with TTSModel mocked."""
    from archer.tts.pocket_tts import PocketTTS

    config = TTSConfig(provider="pocket_tts", voice_sample=voice_sample, voice_name=voice_name)
    mock_model = _make_mock_pocket_model(sample_rate=sample_rate)

    with patch("archer.tts.pocket_tts.TTSModel") as MockTTSModel:
        MockTTSModel.load_model.return_value = mock_model
        tts = PocketTTS(config)

    return tts


class TestPocketTTSInit:
    def test_init_default_voice(self):
        from archer.tts.pocket_tts import PocketTTS

        config = TTSConfig(provider="pocket_tts")
        mock_model = _make_mock_pocket_model()

        with patch("archer.tts.pocket_tts.TTSModel") as MockTTSModel:
            MockTTSModel.load_model.return_value = mock_model
            PocketTTS(config)
            mock_model.get_state_for_audio_prompt.assert_called_once_with("alba")

    def test_init_with_voice_name(self):
        from archer.tts.pocket_tts import PocketTTS

        config = TTSConfig(provider="pocket_tts", voice_name="marius")
        mock_model = _make_mock_pocket_model()

        with patch("archer.tts.pocket_tts.TTSModel") as MockTTSModel:
            MockTTSModel.load_model.return_value = mock_model
            PocketTTS(config)
            mock_model.get_state_for_audio_prompt.assert_called_once_with("marius")

    def test_init_with_voice_sample(self, tmp_path):
        from archer.tts.pocket_tts import PocketTTS

        voice_file = tmp_path / "voice.wav"
        voice_file.write_bytes(b"fake wav data")
        config = TTSConfig(provider="pocket_tts", voice_sample=str(voice_file))
        mock_model = _make_mock_pocket_model()

        with patch("archer.tts.pocket_tts.TTSModel") as MockTTSModel:
            MockTTSModel.load_model.return_value = mock_model
            PocketTTS(config)
            mock_model.get_state_for_audio_prompt.assert_called_once_with(str(voice_file))

    def test_init_voice_sample_not_found(self):
        from archer.tts.pocket_tts import PocketTTS

        config = TTSConfig(provider="pocket_tts", voice_sample="/nonexistent/voice.wav")
        mock_model = _make_mock_pocket_model()

        with patch("archer.tts.pocket_tts.TTSModel") as MockTTSModel:
            MockTTSModel.load_model.return_value = mock_model
            with pytest.raises(FileNotFoundError, match="Voice sample not found"):
                PocketTTS(config)

    def test_init_model_load_failure(self):
        from archer.tts.pocket_tts import PocketTTS

        config = TTSConfig(provider="pocket_tts")

        with patch("archer.tts.pocket_tts.TTSModel") as MockTTSModel:
            MockTTSModel.load_model.side_effect = Exception("Model not found")
            with pytest.raises(RuntimeError, match="Failed to load Pocket TTS model"):
                PocketTTS(config)


class TestPocketTTSSynthesize:
    def test_synthesize_returns_sample_rate_and_array(self):
        tts = _make_pocket_tts()
        sample_rate, audio = tts.synthesize("Hello world.")
        assert sample_rate == 24000
        assert isinstance(audio, np.ndarray)

    def test_synthesize_uses_preloaded_voice_state(self):
        tts = _make_pocket_tts()
        voice_state = tts._voice_state
        tts.synthesize("Hello.")
        tts.model.generate_audio.assert_called_once_with(voice_state, "Hello.")

    def test_sample_rate_property(self):
        tts = _make_pocket_tts(sample_rate=16000)
        assert tts.sample_rate == 16000


class TestPocketTTSSynthesizeLong:
    def test_synthesize_long_splits_sentences(self):
        tts = _make_pocket_tts()
        text = "First sentence. Second sentence. Third sentence."
        tts.synthesize_long(text)
        assert tts.model.generate_audio.call_count == 3

    def test_synthesize_long_silence_between_sentences_only(self):
        tts = _make_pocket_tts(sample_rate=24000)
        text = "First sentence. Second sentence."
        sample_rate, audio = tts.synthesize_long(text)
        # PocketTTS adds silence BEFORE each sentence (except the first)
        # Result: audio1 + silence + audio2
        silence_len = int(0.25 * 24000)  # 6000
        expected_len = 24000 + silence_len + 24000
        assert len(audio) == expected_len

    def test_synthesize_long_returns_correct_sample_rate(self):
        tts = _make_pocket_tts(sample_rate=24000)
        sample_rate, _ = tts.synthesize_long("Single sentence.")
        assert sample_rate == 24000


class TestPocketTTSSaveAudio:
    def test_save_audio_calls_soundfile_write(self):
        tts = _make_pocket_tts()
        with patch("archer.tts.pocket_tts.sf") as mock_sf:
            tts.save_audio("Hello.", "output.wav")
            mock_sf.write.assert_called_once()
            call_args = mock_sf.write.call_args[0]
            assert call_args[0] == "output.wav"
            assert call_args[2] == tts.sample_rate
