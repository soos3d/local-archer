"""Unit tests for archer.tts.chatterbox_tts."""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from archer.core.config import TTSConfig


def _make_mock_model(sample_rate: int = 24000) -> MagicMock:
    """Create a mock ChatterboxModel instance."""
    mock_model = MagicMock()
    mock_model.sr = sample_rate
    fake_audio = np.zeros(sample_rate, dtype=np.float32)
    mock_tensor = MagicMock()
    mock_tensor.squeeze.return_value.cpu.return_value.numpy.return_value = fake_audio
    mock_model.generate.return_value = mock_tensor
    return mock_model


def _make_tts(voice_sample=None, cfg_weight=0.5, sample_rate=24000):
    """Create a ChatterboxTTS with all external deps mocked."""
    from archer.tts.chatterbox_tts import ChatterboxTTS

    config = TTSConfig(provider="chatterbox", voice_sample=voice_sample, cfg_weight=cfg_weight)
    mock_model = _make_mock_model(sample_rate=sample_rate)

    with (
        patch("archer.tts.chatterbox_tts.ChatterboxModel") as MockModel,
        patch("archer.tts.chatterbox_tts.torch") as mock_torch,
    ):
        MockModel.from_pretrained.return_value = mock_model
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = False
        mock_torch.device.return_value = MagicMock()
        tts = ChatterboxTTS(config)

    # tts.model already references mock_model; _sample_rate is set from mock_model.sr
    return tts


class TestChatterboxTTSDeviceDetection:
    def test_detect_cuda(self):
        from archer.tts.chatterbox_tts import ChatterboxTTS

        config = TTSConfig(provider="chatterbox")
        mock_model = _make_mock_model()

        with (
            patch("archer.tts.chatterbox_tts.ChatterboxModel") as MockModel,
            patch("archer.tts.chatterbox_tts.torch") as mock_torch,
        ):
            MockModel.from_pretrained.return_value = mock_model
            mock_torch.cuda.is_available.return_value = True
            mock_torch.backends.mps.is_available.return_value = False
            mock_torch.device.return_value = MagicMock()
            tts = ChatterboxTTS(config)

        assert tts.device == "cuda"

    def test_detect_mps(self):
        from archer.tts.chatterbox_tts import ChatterboxTTS

        config = TTSConfig(provider="chatterbox")
        mock_model = _make_mock_model()

        with (
            patch("archer.tts.chatterbox_tts.ChatterboxModel") as MockModel,
            patch("archer.tts.chatterbox_tts.torch") as mock_torch,
        ):
            MockModel.from_pretrained.return_value = mock_model
            mock_torch.cuda.is_available.return_value = False
            mock_torch.backends.mps.is_available.return_value = True
            mock_torch.device.return_value = MagicMock()
            tts = ChatterboxTTS(config)

        assert tts.device == "mps"

    def test_detect_cpu(self):
        from archer.tts.chatterbox_tts import ChatterboxTTS

        config = TTSConfig(provider="chatterbox")
        mock_model = _make_mock_model()

        with (
            patch("archer.tts.chatterbox_tts.ChatterboxModel") as MockModel,
            patch("archer.tts.chatterbox_tts.torch") as mock_torch,
        ):
            MockModel.from_pretrained.return_value = mock_model
            mock_torch.cuda.is_available.return_value = False
            mock_torch.backends.mps.is_available.return_value = False
            mock_torch.device.return_value = MagicMock()
            tts = ChatterboxTTS(config)

        assert tts.device == "cpu"


class TestChatterboxTTSSynthesize:
    def test_synthesize_returns_sample_rate_and_array(self):
        tts = _make_tts()
        sample_rate, audio = tts.synthesize("Hello world.")
        assert sample_rate == 24000
        assert isinstance(audio, np.ndarray)

    def test_synthesize_passes_voice_sample(self):
        tts = _make_tts()
        tts.synthesize("Hello.", voice_sample="path/to/voice.wav")
        call_kwargs = tts.model.generate.call_args[1]
        assert call_kwargs["audio_prompt_path"] == "path/to/voice.wav"

    def test_synthesize_passes_exaggeration(self):
        tts = _make_tts()
        tts.synthesize("Hello.", exaggeration=0.8)
        call_kwargs = tts.model.generate.call_args[1]
        assert call_kwargs["exaggeration"] == 0.8

    def test_synthesize_passes_cfg_weight(self):
        tts = _make_tts()
        tts.synthesize("Hello.", cfg_weight=0.3)
        call_kwargs = tts.model.generate.call_args[1]
        assert call_kwargs["cfg_weight"] == 0.3

    def test_sample_rate_property(self):
        tts = _make_tts(sample_rate=22050)
        assert tts.sample_rate == 22050


class TestChatterboxTTSSynthesizeLong:
    def test_synthesize_long_splits_sentences(self):
        tts = _make_tts()
        text = "First sentence. Second sentence. Third sentence."
        tts.synthesize_long(text)
        assert tts.model.generate.call_count == 3

    def test_synthesize_long_includes_silence_after_each_sentence(self):
        tts = _make_tts(sample_rate=24000)
        text = "First sentence. Second sentence."
        sample_rate, audio = tts.synthesize_long(text)
        # synthesize_long appends [audio, silence] for each sentence
        # 2 * (24000 audio + 6000 silence) = 60000 samples
        silence_len = int(0.25 * 24000)  # 6000
        expected_len = 2 * (24000 + silence_len)
        assert len(audio) == expected_len

    def test_synthesize_long_returns_correct_sample_rate(self):
        tts = _make_tts(sample_rate=24000)
        sample_rate, _ = tts.synthesize_long("Single sentence.")
        assert sample_rate == 24000


class TestChatterboxTTSSaveAudio:
    def test_save_audio_calls_torchaudio_save(self):
        tts = _make_tts()
        with patch("archer.tts.chatterbox_tts.ta") as mock_ta:
            tts.save_audio("Hello.", "output.wav")
            mock_ta.save.assert_called_once()
            call_args = mock_ta.save.call_args[0]
            assert call_args[0] == "output.wav"
            assert call_args[2] == tts.sample_rate

    def test_save_audio_passes_voice_sample(self):
        tts = _make_tts()
        with patch("archer.tts.chatterbox_tts.ta"):
            tts.save_audio("Hello.", "output.wav", voice_sample="voice.wav")
            call_kwargs = tts.model.generate.call_args[1]
            assert call_kwargs["audio_prompt_path"] == "voice.wav"
