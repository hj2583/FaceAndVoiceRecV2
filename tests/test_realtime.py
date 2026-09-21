from unittest.mock import Mock

from realtime import start_realtime_vad


def test_start_realtime_vad_returns_available_started_instance():
    vad = Mock()
    factory = Mock(return_value=vad)

    result, available = start_realtime_vad(Mock(), vad_factory=factory)

    assert result is vad
    assert available is True
    vad.start.assert_called_once_with()


def test_start_realtime_vad_contains_startup_failure():
    vad = Mock()
    vad.start.side_effect = RuntimeError("no microphone")

    result, available = start_realtime_vad(Mock(), vad_factory=Mock(return_value=vad))

    assert result is None
    assert available is False
