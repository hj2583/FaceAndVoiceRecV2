import threading
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from audio_core import RealtimeVAD


class _DummyVad:
    def __init__(self, aggressiveness):
        self.aggressiveness = aggressiveness

    def is_speech(self, data, sample_rate):
        return False


def test_realtime_vad_start_waits_for_open_stream():
    release_worker = threading.Event()
    stream = Mock()
    audio = Mock()
    audio.open.return_value = stream

    def stop_after_start(*args, **kwargs):
        release_worker.wait(timeout=1.0)
        raise RuntimeError("stop worker")

    stream.read.side_effect = stop_after_start

    with patch("audio_core.pyaudio", new=SimpleNamespace(PyAudio=Mock(return_value=audio), paInt16=8)):
        with patch("audio_core.webrtcvad", new=SimpleNamespace(Vad=_DummyVad)):
            vad = RealtimeVAD()
            vad.start(startup_timeout=1.0)
            assert vad.available is True
            release_worker.set()
            vad.stop()


def test_realtime_vad_start_raises_when_input_stream_cannot_open():
    audio = Mock()
    audio.open.side_effect = OSError("no microphone")

    with patch("audio_core.pyaudio", new=SimpleNamespace(PyAudio=Mock(return_value=audio), paInt16=8)):
        with patch("audio_core.webrtcvad", new=SimpleNamespace(Vad=_DummyVad)):
            vad = RealtimeVAD()
            with pytest.raises(RuntimeError, match="no microphone"):
                vad.start(startup_timeout=1.0)
