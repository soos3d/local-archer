"""Audio recording functionality."""

from __future__ import annotations

import collections
import threading
import time
from collections.abc import Generator
from queue import Empty, Queue

import numpy as np
import sounddevice as sd
from rich.console import Console

from archer.audio.vad import SileroVAD
from archer.core.config import ListeningConfig

console = Console()

# Silero VAD expects 32ms chunks at 16kHz (512 samples)
VAD_CHUNK_SAMPLES = 512


class AudioRecorder:
    """Records audio from the microphone."""

    def __init__(self, sample_rate: int = 16000, channels: int = 1):
        """
        Initialize the audio recorder.

        Args:
            sample_rate: Audio sample rate in Hz.
            channels: Number of audio channels.
        """
        self.sample_rate = sample_rate
        self.channels = channels
        self._chunks: Queue[np.ndarray] = Queue()
        self._stop_event: threading.Event | None = None
        self._recording_thread: threading.Thread | None = None

    def _audio_callback(self, indata: np.ndarray, _frames: int, _time_info: dict, status: sd.CallbackFlags) -> None:
        """Callback for audio stream — receives float32 numpy arrays."""
        if status:
            console.print(f"[yellow]Audio status: {status}[/yellow]")
        self._chunks.put(indata.copy())

    def _record_loop(self) -> None:
        """Recording loop that runs in a separate thread."""
        with sd.InputStream(
            samplerate=self.sample_rate,
            dtype="float32",
            channels=self.channels,
            callback=self._audio_callback,
        ):
            while self._stop_event and not self._stop_event.is_set():
                time.sleep(0.1)

    def start_recording(self) -> None:
        """Start recording audio in a background thread."""
        self._chunks = Queue()
        self._stop_event = threading.Event()
        self._recording_thread = threading.Thread(target=self._record_loop)
        self._recording_thread.start()

    def stop_recording(self) -> None:
        """Stop the recording."""
        if self._stop_event:
            self._stop_event.set()
        if self._recording_thread:
            self._recording_thread.join()

    def get_audio_array(self) -> np.ndarray:
        """
        Get the recorded audio as a mono float32 numpy array.

        Returns:
            Normalized float32 audio array.
        """
        chunks = list(self._chunks.queue)
        if not chunks:
            return np.array([], dtype=np.float32)
        audio = np.concatenate(chunks, axis=0)
        # Flatten to mono
        if audio.ndim > 1:
            audio = audio[:, 0]
        return audio.astype(np.float32)

    def record_until_enter(self) -> np.ndarray:
        """
        Record audio until user presses Enter.

        Returns:
            Recorded audio as numpy array.
        """
        console.input("🎤 Press Enter to start recording, then press Enter again to stop.")
        self.start_recording()
        input()  # Wait for user to press Enter again
        self.stop_recording()
        return self.get_audio_array()

    def listen_continuous(
        self,
        vad: SileroVAD,
        listening_config: ListeningConfig,
        paused: threading.Event,
        stop: threading.Event | None = None,
    ) -> Generator[np.ndarray, None, None]:
        """
        Continuously listen using VAD, yielding complete utterances.

        Uses a pre-buffer ring to avoid clipping speech onset. Yields audio
        arrays when silence is detected after speech. Discards audio while
        *paused* is set (e.g. during TTS playback).

        Args:
            vad: Silero VAD instance.
            listening_config: Listening configuration.
            paused: Event that suppresses capture when set.
            stop: Optional event to signal clean shutdown.

        Yields:
            Complete utterance audio as float32 numpy arrays.
        """
        # At least 1 chunk for pre-buffer and silence detection
        pre_buffer_chunks = max(
            1,
            int(listening_config.pre_buffer_ms / 1000 * self.sample_rate / VAD_CHUNK_SAMPLES),
        )
        silence_chunks_needed = max(
            1,
            int(listening_config.silence_duration_s * self.sample_rate / VAD_CHUNK_SAMPLES),
        )
        max_chunks = int(
            listening_config.max_utterance_s * self.sample_rate / VAD_CHUNK_SAMPLES
        )

        ring: collections.deque[np.ndarray] = collections.deque(maxlen=pre_buffer_chunks)
        utterance_chunks: list[np.ndarray] = []
        silence_counter = 0
        in_speech = False
        was_paused = False

        chunk_queue: Queue[np.ndarray] = Queue()

        def _vad_callback(
            indata: np.ndarray, _frames: int, _time_info: object, status: sd.CallbackFlags,
        ) -> None:
            if status:
                console.print(f"[yellow]Audio status: {status}[/yellow]")
            chunk_queue.put(indata[:, 0].copy() if indata.ndim > 1 else indata.copy().flatten())

        with sd.InputStream(
            samplerate=self.sample_rate,
            dtype="float32",
            channels=self.channels,
            blocksize=VAD_CHUNK_SAMPLES,
            callback=_vad_callback,
        ):
            while stop is None or not stop.is_set():
                try:
                    chunk = chunk_queue.get(timeout=0.1)
                except Empty:
                    continue

                if paused.is_set():
                    ring.clear()
                    utterance_chunks.clear()
                    silence_counter = 0
                    in_speech = False
                    was_paused = True
                    continue

                if was_paused:
                    vad.reset()
                    was_paused = False

                is_speech = vad.speech_detected(chunk)

                if not in_speech:
                    ring.append(chunk)
                    if is_speech:
                        in_speech = True
                        utterance_chunks = list(ring)
                        ring.clear()
                        silence_counter = 0
                else:
                    utterance_chunks.append(chunk)
                    if is_speech:
                        silence_counter = 0
                    else:
                        silence_counter += 1

                    if silence_counter >= silence_chunks_needed or len(utterance_chunks) >= max_chunks:
                        yield np.concatenate(utterance_chunks).astype(np.float32)

                        # Generator resumes here after the caller finishes
                        # processing (TTS playback, etc.).  The sd.InputStream
                        # callback kept filling chunk_queue the whole time, so
                        # flush all stale audio to prevent echo/self-listening.
                        while True:
                            try:
                                chunk_queue.get_nowait()
                            except Empty:
                                break

                        utterance_chunks.clear()
                        ring.clear()
                        silence_counter = 0
                        in_speech = False
                        vad.reset()
