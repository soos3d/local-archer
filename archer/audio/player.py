"""Audio playback functionality."""

import numpy as np
import sounddevice as sd


class AudioPlayer:
    """Plays audio through the speakers."""

    def play(self, audio_array: np.ndarray, sample_rate: int) -> None:
        """
        Play audio synchronously (blocks until complete).

        Args:
            audio_array: Audio data as numpy array.
            sample_rate: Sample rate in Hz.
        """
        sd.play(audio_array, sample_rate)
        sd.wait()

