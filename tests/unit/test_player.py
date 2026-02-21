"""Unit tests for archer.audio.player."""

import numpy as np
import pytest
from unittest.mock import patch

from archer.audio.player import AudioPlayer


class TestAudioPlayer:
    def test_play_calls_sounddevice_play_and_wait(self):
        player = AudioPlayer()
        audio = np.zeros(16000, dtype=np.float32)
        with patch("archer.audio.player.sd") as mock_sd:
            player.play(audio, 16000)
            mock_sd.play.assert_called_once_with(audio, 16000)
            mock_sd.wait.assert_called_once()

    def test_play_passes_correct_sample_rate(self):
        player = AudioPlayer()
        audio = np.zeros(44100, dtype=np.float32)
        with patch("archer.audio.player.sd") as mock_sd:
            player.play(audio, 44100)
            mock_sd.play.assert_called_once_with(audio, 44100)

    def test_play_blocks_until_done(self):
        """Verify sd.play() is called before sd.wait()."""
        player = AudioPlayer()
        audio = np.zeros(100, dtype=np.float32)
        call_order = []

        with patch("archer.audio.player.sd") as mock_sd:
            mock_sd.play.side_effect = lambda *args: call_order.append("play")
            mock_sd.wait.side_effect = lambda: call_order.append("wait")
            player.play(audio, 16000)

        assert call_order == ["play", "wait"]
