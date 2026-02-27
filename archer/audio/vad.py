"""Silero Voice Activity Detection wrapper."""

import numpy as np
import torch

from archer.core.config import ListeningConfig


class SileroVAD:
    """Voice Activity Detection using Silero VAD model."""

    def __init__(self, config: ListeningConfig):
        """
        Initialize Silero VAD.

        Args:
            config: Listening configuration with VAD settings.
        """
        self.threshold = config.vad_threshold
        self._model, self._utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            trust_repo=True,
        )

    def reset(self) -> None:
        """Reset the model's internal RNN state between utterances."""
        self._model.reset_states()

    def is_speech(self, chunk: np.ndarray) -> float:
        """
        Score an audio chunk for speech probability.

        Args:
            chunk: Audio chunk as float32 numpy array (512 samples = 32ms @ 16kHz).

        Returns:
            Speech probability between 0.0 and 1.0.
        """
        tensor = torch.from_numpy(chunk).float()
        score: float = self._model(tensor, 16000).item()
        return score

    def speech_detected(self, chunk: np.ndarray) -> bool:
        """
        Check if speech is detected in an audio chunk.

        Args:
            chunk: Audio chunk as float32 numpy array.

        Returns:
            True if speech probability >= threshold.
        """
        return self.is_speech(chunk) >= self.threshold
