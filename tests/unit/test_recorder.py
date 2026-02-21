"""Unit tests for archer.audio.recorder."""

import numpy as np
from unittest.mock import patch

from archer.audio.recorder import AudioRecorder


class TestAudioRecorderInit:
    def test_defaults(self):
        recorder = AudioRecorder()
        assert recorder.sample_rate == 16000
        assert recorder.channels == 1

    def test_custom_params(self):
        recorder = AudioRecorder(sample_rate=44100, channels=2)
        assert recorder.sample_rate == 44100
        assert recorder.channels == 2


class TestGetAudioArray:
    def test_empty_queue_returns_empty_array(self):
        recorder = AudioRecorder()
        result = recorder.get_audio_array()
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32
        assert len(result) == 0

    def test_single_mono_chunk(self):
        recorder = AudioRecorder()
        chunk = np.array([[0.1], [0.2], [0.3]], dtype=np.float32)
        recorder._chunks.put(chunk)
        result = recorder.get_audio_array()
        np.testing.assert_array_almost_equal(result, [0.1, 0.2, 0.3])

    def test_multiple_chunks_concatenated(self):
        recorder = AudioRecorder()
        chunk1 = np.array([[0.1], [0.2]], dtype=np.float32)
        chunk2 = np.array([[0.3], [0.4]], dtype=np.float32)
        recorder._chunks.put(chunk1)
        recorder._chunks.put(chunk2)
        result = recorder.get_audio_array()
        np.testing.assert_array_almost_equal(result, [0.1, 0.2, 0.3, 0.4])

    def test_stereo_chunk_uses_first_channel(self):
        recorder = AudioRecorder(channels=2)
        chunk = np.array([[0.5, 0.6], [0.7, 0.8]], dtype=np.float32)
        recorder._chunks.put(chunk)
        result = recorder.get_audio_array()
        np.testing.assert_array_almost_equal(result, [0.5, 0.7])

    def test_returns_float32(self):
        recorder = AudioRecorder()
        chunk = np.array([[0.1], [0.2]], dtype=np.float64)
        recorder._chunks.put(chunk)
        result = recorder.get_audio_array()
        assert result.dtype == np.float32


class TestAudioCallback:
    def test_callback_stores_copy_of_data(self):
        recorder = AudioRecorder()
        indata = np.array([[0.1], [0.2]], dtype=np.float32)
        recorder._audio_callback(indata, 2, {}, None)
        result = recorder._chunks.get()
        np.testing.assert_array_equal(result, indata)
        assert result is not indata


class TestStartStopRecording:
    def test_start_stop_recording(self):
        recorder = AudioRecorder()
        with patch("archer.audio.recorder.sd"):
            recorder.start_recording()
            recorder.stop_recording()
            assert recorder._stop_event.is_set()
            assert not recorder._recording_thread.is_alive()

    def test_start_recording_resets_chunks(self):
        recorder = AudioRecorder()
        recorder._chunks.put(np.array([[0.1]], dtype=np.float32))
        with patch("archer.audio.recorder.sd"):
            recorder.start_recording()
            recorder.stop_recording()
        assert recorder._chunks.empty()
