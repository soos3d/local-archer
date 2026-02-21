"""ChatterBox Text-to-Speech implementation."""

import warnings

import nltk
import numpy as np
import torch
import torchaudio as ta
from chatterbox.tts import ChatterboxTTS as ChatterboxModel

from archer.core.config import TTSConfig
from archer.tts.base import BaseTTS

warnings.filterwarnings(
    "ignore",
    message="torch.nn.utils.weight_norm is deprecated in favor of torch.nn.utils.parametrizations.weight_norm.",
)


class ChatterboxTTS(BaseTTS):
    """ChatterBox text-to-speech provider with voice cloning support."""

    def __init__(self, config: TTSConfig):
        """
        Initialize ChatterBox TTS.

        Args:
            config: TTS configuration.
        """
        self.config = config
        self.device = self._detect_device()
        self._patch_torch_load()
        self.model = ChatterboxModel.from_pretrained(device=self.device)
        self._sample_rate = self.model.sr
        self._warmup()

    def _detect_device(self) -> str:
        """Detect the best available compute device."""
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _patch_torch_load(self) -> None:
        """Patch torch.load for device compatibility."""
        map_location = torch.device(self.device)

        if not hasattr(torch, "_original_load"):
            torch._original_load = torch.load

        def patched_torch_load(*args, **kwargs):
            if "map_location" not in kwargs:
                kwargs["map_location"] = map_location
            return torch._original_load(*args, **kwargs)

        torch.load = patched_torch_load

    def _warmup(self) -> None:
        """Run a silent synthesis to pre-compile the MPS/CUDA compute graph."""
        try:
            self.model.generate("Hello.", exaggeration=0.5, cfg_weight=0.5)
        except Exception:
            pass  # Warmup is best-effort; failures should not block startup

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
        wav = self.model.generate(
            text,
            audio_prompt_path=voice_sample,
            exaggeration=exaggeration,
            cfg_weight=cfg_weight,
        )
        audio_array = wav.squeeze().cpu().numpy()
        return self._sample_rate, audio_array

    def synthesize_long(
        self,
        text: str,
        voice_sample: str | None = None,
        exaggeration: float = 0.5,
        cfg_weight: float = 0.5,
    ) -> tuple[int, np.ndarray]:
        """
        Synthesize speech from long-form text with natural pauses.

        Args:
            text: Text to synthesize.
            voice_sample: Optional path to voice sample for cloning.
            exaggeration: Emotion intensity (0.0-1.0).
            cfg_weight: Pacing control (0.0-1.0).

        Returns:
            Tuple of (sample_rate, audio_array).
        """
        sentences = nltk.sent_tokenize(text)
        pieces = []
        silence = np.zeros(int(0.25 * self._sample_rate))

        for sentence in sentences:
            _, audio_array = self.synthesize(
                sentence,
                voice_sample=voice_sample,
                exaggeration=exaggeration,
                cfg_weight=cfg_weight,
            )
            pieces.extend([audio_array, silence.copy()])

        return self._sample_rate, np.concatenate(pieces)

    def save_audio(self, text: str, output_path: str, voice_sample: str | None = None) -> None:
        """
        Save synthesized audio to file.

        Args:
            text: Text to synthesize.
            output_path: Path to save the audio file.
            voice_sample: Optional path to voice sample for cloning.
        """
        wav = self.model.generate(text, audio_prompt_path=voice_sample)
        ta.save(output_path, wav, self._sample_rate)
