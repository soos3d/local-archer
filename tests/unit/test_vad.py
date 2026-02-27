"""Unit tests for archer.audio.vad."""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from archer.core.config import ListeningConfig


class TestSileroVAD:
    def _make_vad(self, threshold: float = 0.5) -> "SileroVAD":
        """Create a SileroVAD with mocked torch.hub.load."""
        config = ListeningConfig(vad_threshold=threshold)
        mock_model = MagicMock()
        mock_utils = MagicMock()

        with patch("archer.audio.vad.torch") as mock_torch:
            mock_torch.hub.load.return_value = (mock_model, mock_utils)
            mock_torch.from_numpy = lambda x: MagicMock(
                float=MagicMock(return_value=x)
            )

            from archer.audio.vad import SileroVAD

            vad = SileroVAD(config)

        vad._model = mock_model
        return vad

    def test_loads_model_from_torch_hub(self):
        config = ListeningConfig()
        with patch("archer.audio.vad.torch") as mock_torch:
            mock_torch.hub.load.return_value = (MagicMock(), MagicMock())

            from archer.audio.vad import SileroVAD

            SileroVAD(config)

            mock_torch.hub.load.assert_called_once_with(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                trust_repo=True,
            )

    def test_is_speech_returns_float(self):
        vad = self._make_vad()
        chunk = np.zeros(1536, dtype=np.float32)
        mock_result = MagicMock()
        mock_result.item.return_value = 0.75
        vad._model.return_value = mock_result

        with patch("archer.audio.vad.torch") as mock_torch:
            mock_tensor = MagicMock()
            mock_tensor.float.return_value = mock_tensor
            mock_torch.from_numpy.return_value = mock_tensor

            score = vad.is_speech(chunk)

        assert score == 0.75

    def test_speech_detected_above_threshold(self):
        vad = self._make_vad(threshold=0.5)
        chunk = np.zeros(1536, dtype=np.float32)
        mock_result = MagicMock()
        mock_result.item.return_value = 0.8
        vad._model.return_value = mock_result

        with patch("archer.audio.vad.torch") as mock_torch:
            mock_tensor = MagicMock()
            mock_tensor.float.return_value = mock_tensor
            mock_torch.from_numpy.return_value = mock_tensor

            assert vad.speech_detected(chunk) is True

    def test_speech_detected_below_threshold(self):
        vad = self._make_vad(threshold=0.5)
        chunk = np.zeros(1536, dtype=np.float32)
        mock_result = MagicMock()
        mock_result.item.return_value = 0.3
        vad._model.return_value = mock_result

        with patch("archer.audio.vad.torch") as mock_torch:
            mock_tensor = MagicMock()
            mock_tensor.float.return_value = mock_tensor
            mock_torch.from_numpy.return_value = mock_tensor

            assert vad.speech_detected(chunk) is False

    def test_speech_detected_at_threshold(self):
        vad = self._make_vad(threshold=0.5)
        chunk = np.zeros(1536, dtype=np.float32)
        mock_result = MagicMock()
        mock_result.item.return_value = 0.5
        vad._model.return_value = mock_result

        with patch("archer.audio.vad.torch") as mock_torch:
            mock_tensor = MagicMock()
            mock_tensor.float.return_value = mock_tensor
            mock_torch.from_numpy.return_value = mock_tensor

            assert vad.speech_detected(chunk) is True

    def test_custom_threshold(self):
        vad = self._make_vad(threshold=0.9)
        chunk = np.zeros(1536, dtype=np.float32)
        mock_result = MagicMock()
        mock_result.item.return_value = 0.85
        vad._model.return_value = mock_result

        with patch("archer.audio.vad.torch") as mock_torch:
            mock_tensor = MagicMock()
            mock_tensor.float.return_value = mock_tensor
            mock_torch.from_numpy.return_value = mock_tensor

            assert vad.speech_detected(chunk) is False
