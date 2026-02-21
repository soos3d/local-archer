"""Streaming TTS + playback pipeline for reduced latency."""

import threading
from queue import Queue

import numpy as np
import sounddevice as sd

from archer.tts.base import BaseTTS


class StreamingPipeline:
    """Concurrent TTS synthesis and audio playback pipeline.

    Uses two threads and two queues to overlap TTS synthesis with playback,
    so the user hears the first sentence while later sentences are still
    being synthesized.
    """

    def __init__(self, tts: BaseTTS, *, queue_size: int = 2):
        self._tts = tts
        self._queue_size = queue_size

    def run(
        self,
        sentences: list[str],
        *,
        voice_sample: str | None = None,
        exaggeration: float = 0.5,
        cfg_weight: float = 0.5,
    ) -> np.ndarray | None:
        """Run the streaming pipeline: synthesize and play concurrently.

        Args:
            sentences: List of sentences to synthesize and play.
            voice_sample: Optional path to voice sample for cloning.
            exaggeration: Emotion intensity (0.0-1.0).
            cfg_weight: Pacing control (0.0-1.0).

        Returns:
            Full concatenated audio as numpy array, or None if no sentences.

        Raises:
            RuntimeError: If an error occurs in the TTS or playback thread.
        """
        if not sentences:
            return None

        audio_queue: Queue[np.ndarray | None] = Queue(maxsize=self._queue_size)
        all_chunks: list[np.ndarray] = []
        error_holder: list[Exception] = []
        sample_rate = self._tts.sample_rate
        silence_gap = np.zeros(int(0.25 * sample_rate), dtype=np.float32)

        def tts_worker() -> None:
            try:
                for i, sentence in enumerate(sentences):
                    _, audio_chunk = self._tts.synthesize(
                        sentence,
                        voice_sample=voice_sample,
                        exaggeration=exaggeration,
                        cfg_weight=cfg_weight,
                    )
                    if i > 0:
                        all_chunks.append(silence_gap)
                    all_chunks.append(audio_chunk)
                    audio_queue.put(audio_chunk)
            except Exception as exc:
                error_holder.append(exc)
            finally:
                audio_queue.put(None)

        def playback_worker() -> None:
            try:
                first = True
                while True:
                    chunk = audio_queue.get()
                    if chunk is None:
                        break
                    if not first:
                        silence = np.zeros(
                            int(0.25 * sample_rate), dtype=np.float32
                        )
                        sd.play(silence, sample_rate)
                        sd.wait()
                    sd.play(chunk, sample_rate)
                    sd.wait()
                    first = False
            except Exception as exc:
                error_holder.append(exc)

        tts_thread = threading.Thread(target=tts_worker, daemon=True)
        playback_thread = threading.Thread(target=playback_worker, daemon=True)

        tts_thread.start()
        playback_thread.start()

        tts_thread.join()
        playback_thread.join()

        if error_holder:
            raise RuntimeError(
                f"Streaming pipeline error: {error_holder[0]}"
            ) from error_holder[0]

        if not all_chunks:
            return None

        return np.concatenate(all_chunks)
