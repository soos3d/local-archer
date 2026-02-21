"""Kyutai Pocket TTS implementation."""

from pathlib import Path

import nltk
import numpy as np
import soundfile as sf
from pocket_tts import TTSModel

from archer.core.config import TTSConfig
from archer.tts.base import BaseTTS

_DEFAULT_VOICE = "alba"


class PocketTTS(BaseTTS):
    """Kyutai Pocket TTS provider with voice cloning support.

    Runs ~6x real-time on CPU; first audio latency ~200ms.
    Voice cloning is supported via audio file path or pre-built voice names
    (alba, marius, javert, jean, fantine, cosette, eponine, azelma).
    """

    def __init__(self, config: TTSConfig):
        self.config = config
        try:
            self.model = TTSModel.load_model()
        except Exception as e:
            raise RuntimeError(
                "Failed to load Pocket TTS model. "
                "Ensure 'pocket-tts' is installed: pip install pocket-tts"
            ) from e
        self._sample_rate: int = self.model.sample_rate
        self._voice_state = self._load_voice_state()

    def _load_voice_state(self):
        """Load voice state from file, built-in name, or default."""
        if self.config.voice_sample:
            path = Path(self.config.voice_sample)
            if not path.is_file():
                raise FileNotFoundError(f"Voice sample not found: {path}")
            try:
                return self.model.get_state_for_audio_prompt(str(path))
            except Exception as e:
                raise ValueError(f"Failed to load voice sample '{path}': {e}") from e

        voice = self.config.voice_name or _DEFAULT_VOICE
        try:
            return self.model.get_state_for_audio_prompt(voice)
        except Exception as e:
            raise ValueError(f"Failed to load voice '{voice}': {e}") from e

    @property
    def sample_rate(self) -> int:
        """Return the audio sample rate."""
        return self._sample_rate

    def synthesize(
        self,
        text: str,
        voice_sample: str | None = None,
        exaggeration: float = 0.5,
        cfg_weight: float = 0.5,
    ) -> tuple[int, np.ndarray]:
        """Synthesize speech from text.

        Args:
            text: Text to synthesize.
            voice_sample: Ignored — voice state is pre-loaded at init for performance.
            exaggeration: Ignored — not supported by Pocket TTS.
            cfg_weight: Ignored — not supported by Pocket TTS.

        Returns:
            Tuple of (sample_rate, audio_array).
        """
        audio_tensor = self.model.generate_audio(self._voice_state, text)
        return self._sample_rate, audio_tensor.numpy()

    def synthesize_long(
        self,
        text: str,
        voice_sample: str | None = None,
        exaggeration: float = 0.5,
        cfg_weight: float = 0.5,
    ) -> tuple[int, np.ndarray]:
        """Synthesize speech from long-form text with natural pauses.

        Splits text into sentences via NLTK and concatenates with 250ms silence
        between each sentence, matching the ChatterBox behaviour.

        Args:
            text: Text to synthesize.
            voice_sample: Ignored — voice state is pre-loaded at init.
            exaggeration: Ignored — not supported by Pocket TTS.
            cfg_weight: Ignored — not supported by Pocket TTS.

        Returns:
            Tuple of (sample_rate, audio_array).
        """
        sentences = nltk.sent_tokenize(text)
        silence = np.zeros(int(0.25 * self._sample_rate), dtype=np.float32)
        pieces: list[np.ndarray] = []

        for i, sentence in enumerate(sentences):
            _, chunk = self.synthesize(sentence)
            if i > 0:
                pieces.append(silence.copy())
            pieces.append(chunk)

        return self._sample_rate, np.concatenate(pieces)

    def save_audio(self, text: str, output_path: str, voice_sample: str | None = None) -> None:
        """Save synthesized audio to a WAV file.

        Args:
            text: Text to synthesize.
            output_path: Path to save the audio file.
            voice_sample: Ignored — voice state is pre-loaded at init.
        """
        _, audio = self.synthesize_long(text)
        sf.write(output_path, audio, self._sample_rate)

