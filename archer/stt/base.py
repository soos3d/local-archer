"""Abstract base class for Speech-to-Text providers."""

from abc import ABC, abstractmethod

import numpy as np


class BaseSTT(ABC):
    """Abstract interface for speech-to-text providers."""

    @abstractmethod
    def load_model(self) -> None:
        """Load the STT model into memory."""
        pass

    @abstractmethod
    def transcribe(self, audio: np.ndarray) -> str:
        """
        Transcribe audio to text.

        Args:
            audio: Audio data as numpy array (float32, normalized).

        Returns:
            Transcribed text.
        """
        pass
