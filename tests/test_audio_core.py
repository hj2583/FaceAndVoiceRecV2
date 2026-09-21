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


def test_realtime_vad_start_raises_when_pyaudio_constructor_fails():
    pyaudio_mock = SimpleNamespace(
        PyAudio=Mock(side_effect=OSError("pyaudio constructor failed")),
        paInt16=8,
    )

    with patch("audio_core.pyaudio", new=pyaudio_mock):
        with patch("audio_core.webrtcvad", new=SimpleNamespace(Vad=_DummyVad)):
            vad = RealtimeVAD()
            with pytest.raises(RuntimeError, match="pyaudio constructor failed"):
                vad.start(startup_timeout=1.0)


def test_realtime_vad_worker_exit_emits_terminal_false_after_speaking():
    callback = Mock()
    stream = Mock()
    stream.read.side_effect = [b"frame1", RuntimeError("worker exit")]

    audio = Mock()
    audio.open.return_value = stream

    class _TwoStepVad:
        def __init__(self, aggressiveness):
            self._calls = 0

        def is_speech(self, data, sample_rate):
            self._calls += 1
            return self._calls == 1

    with patch("audio_core.REALTIME_SPEECH_CONFIRM_FRAMES", 1):
        with patch("audio_core.pyaudio", new=SimpleNamespace(PyAudio=Mock(return_value=audio), paInt16=8)):
            with patch("audio_core.webrtcvad", new=SimpleNamespace(Vad=_TwoStepVad)):
                vad = RealtimeVAD(callback=callback)
                vad.start(startup_timeout=1.0)
                vad.thread.join(timeout=1.0)

    observed_states = [args[0][0] for args in callback.call_args_list]
    assert observed_states[-2:] == [True, False]
    assert vad.available is False


def test_realtime_vad_stop_keeps_thread_when_join_times_out():
    thread = Mock()
    thread.is_alive.return_value = True

    vad = RealtimeVAD()
    vad.thread = thread

    with patch("audio_core.logging.warning") as warning:
        vad.stop()

    thread.join.assert_called_once_with(timeout=2)
    assert vad.thread is thread
    warning.assert_called_once()
