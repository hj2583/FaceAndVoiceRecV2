import threading
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from audio_core import FRAME_SAMPLES, RealtimeVAD


def _silence_frame():
    return (b"\x00\x00" * FRAME_SAMPLES, False)


def test_realtime_vad_start_waits_for_open_stream():
    release_worker = threading.Event()
    stream = Mock()

    def stop_after_start(*args, **kwargs):
        release_worker.wait(timeout=1.0)
        raise RuntimeError("stop worker")

    stream.read.side_effect = stop_after_start

    fake_sd = SimpleNamespace(RawInputStream=Mock(return_value=stream))

    with patch("audio_core.sd", new=fake_sd):
        with patch("audio_core.load_silero_vad", new=Mock(return_value=lambda *_a, **_k: 0.0)):
            vad = RealtimeVAD()
            vad.start(startup_timeout=1.0)
            assert vad.startup_succeeded is True
            assert vad.available is True
            release_worker.set()
            vad.stop()


def test_realtime_vad_start_raises_when_input_stream_cannot_open():
    fake_sd = SimpleNamespace(RawInputStream=Mock(side_effect=OSError("no microphone")))

    with patch("audio_core.sd", new=fake_sd):
        with patch("audio_core.load_silero_vad", new=Mock(return_value=lambda *_a, **_k: 0.0)):
            vad = RealtimeVAD()
            with pytest.raises(RuntimeError, match="no microphone"):
                vad.start(startup_timeout=1.0)


def test_realtime_vad_start_raises_when_model_fails_to_load():
    fake_sd = SimpleNamespace(RawInputStream=Mock())

    with patch("audio_core.sd", new=fake_sd):
        with patch("audio_core.load_silero_vad", new=Mock(side_effect=OSError("model load failed"))):
            vad = RealtimeVAD()
            with pytest.raises(RuntimeError, match="model load failed"):
                vad.start(startup_timeout=1.0)


def test_realtime_vad_worker_exit_emits_terminal_false_after_speaking():
    callback = Mock()
    speaking_emitted = threading.Event()

    def _callback(state):
        callback(state)
        if state:
            speaking_emitted.set()

    continue_after_speaking = threading.Event()
    stream = Mock()

    def _read(*_args, **_kwargs):
        if not speaking_emitted.is_set():
            return _silence_frame()
        continue_after_speaking.wait(timeout=1.0)
        raise RuntimeError("worker exit")

    stream.read.side_effect = _read

    class _TwoStepModel:
        def __init__(self):
            self._calls = 0

        def __call__(self, _tensor, _sample_rate):
            self._calls += 1
            return 1.0 if self._calls == 1 else 0.0

    fake_sd = SimpleNamespace(RawInputStream=Mock(return_value=stream))

    with patch("audio_core.REALTIME_SPEECH_CONFIRM_FRAMES", 1):
        with patch("audio_core.sd", new=fake_sd):
            with patch("audio_core.load_silero_vad", new=Mock(return_value=_TwoStepModel())):
                vad = RealtimeVAD(callback=_callback)
                vad.start(startup_timeout=1.0)
                assert speaking_emitted.wait(timeout=1.0)
                continue_after_speaking.set()
                vad.thread.join(timeout=1.0)

    observed_states = [args[0][0] for args in callback.call_args_list]
    assert observed_states[-2:] == [True, False]
    assert vad.available is False


def test_realtime_vad_start_raises_when_worker_is_already_alive():
    running_thread = Mock()
    running_thread.is_alive.return_value = True

    vad = RealtimeVAD()
    vad.thread = running_thread

    fake_sd = SimpleNamespace(RawInputStream=Mock())

    with patch("audio_core.sd", new=fake_sd):
        with patch("audio_core.load_silero_vad", new=Mock(return_value=lambda *_a, **_k: 0.0)):
            with pytest.raises(RuntimeError, match="already running"):
                vad.start(startup_timeout=1.0)


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
