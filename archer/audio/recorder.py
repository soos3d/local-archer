"""Audio recording functionality."""

import threading
import time
from queue import Queue

import numpy as np
import sounddevice as sd
from rich.console import Console

console = Console()


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
