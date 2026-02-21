"""Abstract base class for Text-to-Speech providers."""

from abc import ABC, abstractmethod

import numpy as np


class BaseTTS(ABC):
    """Abstract interface for text-to-speech providers."""

    @property
    @abstractmethod
    def sample_rate(self) -> int:
        """Return the audio sample rate."""
        pass

    @abstractmethod
    def synthesize(
        self,
        text: str,
        voice_sample: str | None = None,
        exaggeration: float = 0.5,
        cfg_weight: float = 0.5,
    ) -> tuple[int, np.ndarray]:
        """
        Synthesize speech from text.

        Args:
            text: Text to synthesize.
            voice_sample: Optional path to voice sample for cloning.
            exaggeration: Emotion intensity (0.0-1.0).
            cfg_weight: Pacing control (0.0-1.0).

        Returns:
            Tuple of (sample_rate, audio_array).
        """
        pass

    @abstractmethod
    def synthesize_long(
        self,
        text: str,
        voice_sample: str | None = None,
        exaggeration: float = 0.5,
        cfg_weight: float = 0.5,
    ) -> tuple[int, np.ndarray]:
        """
        Synthesize speech from long-form text (multiple sentences).

        Args:
            text: Text to synthesize.
            voice_sample: Optional path to voice sample for cloning.
            exaggeration: Emotion intensity (0.0-1.0).
            cfg_weight: Pacing control (0.0-1.0).

        Returns:
            Tuple of (sample_rate, audio_array).
        """
        pass
