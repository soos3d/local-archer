"""Whisper Speech-to-Text implementation."""

import numpy as np
import whisper
from rich.console import Console

from archer.core.config import STTConfig
from archer.stt.base import BaseSTT

console = Console()


class WhisperSTT(BaseSTT):
    """OpenAI Whisper speech-to-text provider."""

    def __init__(self, config: STTConfig):
        """
        Initialize Whisper STT.

        Args:
            config: STT configuration.
        """
        self.config = config
        self.model = None

    def load_model(self) -> None:
        """Load the Whisper model."""
        self.model = whisper.load_model(self.config.model)

    def transcribe(self, audio: np.ndarray) -> str:
        """
        Transcribe audio to text using Whisper.

        Args:
            audio: Audio data as numpy array (float32, normalized, 16 kHz).

        Returns:
            Transcribed text, or empty string if audio is too short or silent.
        """
        if self.model is None:
            self.load_model()

        # Reject clips that are too short (Whisper hallucinates on them)
        duration = len(audio) / 16000
        if duration < self.config.min_audio_duration:
            console.print(
                f"[yellow]Audio too short ({duration:.2f}s). "
                "Please hold the recording longer.[/yellow]"
            )
            return ""

        # Reject silent audio
        rms = float(np.sqrt(np.mean(audio**2)))
        if rms < self.config.silence_threshold:
            console.print(
                f"[yellow]No speech detected (RMS={rms:.5f}, threshold={self.config.silence_threshold}). "
                "Please speak closer to the microphone, or lower silence_threshold in config.[/yellow]"
            )
            return ""

        result = self.model.transcribe(
            audio,
            fp16=self.config.fp16,
            language="en",
            condition_on_previous_text=False,
        )
        return result["text"].strip()
