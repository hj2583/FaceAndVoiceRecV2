import logging
import pytest
from realtime_runtime import DetectionScheduler, RollingFps


class DummyDetector:
    def __init__(self, raises_on=None, detections=None):
        self.calls = []
        self.raises_on = raises_on or set()
        self.detections = detections or [(1, 2, 3, 4, 0.9)]

    def __call__(self, frame):
        self.calls.append(frame)
        if frame in self.raises_on:
            raise RuntimeError("detector failed")
        return list(self.detections)


def test_interval_reuses_detections():
    sched = DetectionScheduler(interval=3)
    det = DummyDetector()

    out1 = sched.update(1, 1, det)
    assert out1 == det.detections

    out2 = sched.update(2, 2, det)
    assert out2 == det.detections

    out3 = sched.update(3, 3, det)
    assert out3 == det.detections

    out4 = sched.update(4, 4, det)
    assert out4 == det.detections


def test_detector_exception_clears_state(tmp_path, caplog):
    caplog.set_level(logging.ERROR)
    sched = DetectionScheduler(interval=3)
    det = DummyDetector(raises_on={4})

    out1 = sched.update(1, 1, det)
    assert out1 == det.detections

    # frames 2/3 reuse
    out2 = sched.update(2, 2, det)
    assert out2 == det.detections

    out3 = sched.update(3, 3, det)
    assert out3 == det.detections

    # frame 4 raises
    out4 = sched.update(4, 4, det)
    assert out4 == []
    assert "Realtime face detection failed" in caplog.text

    out5 = sched.update(5, 5, det)
    # since 5 is not a scheduled detection frame (interval=3 -> frames 1,4,7..), and last state was cleared,
    # it should remain empty
    assert out5 == []


def test_rolling_fps_basic():
    r = RollingFps(1.0)
    # drive timestamps up to 1.0
    r.tick(0.0)
    r.tick(0.25)
    r.tick(0.5)
    # at t=1.0 we have timestamps [0.0,0.25,0.5,1.0] -> (4-1)/1.0 = 3.0
    assert pytest.approx(r.tick(1.0), rel=1e-6) == 3.0
    # tick 1.5: window is (0.5,1.5] -> timestamps [0.5,1.0,1.5] -> (3-1)/1.0 = 2.0
    assert pytest.approx(r.tick(1.5), rel=1e-6) == 2.0


def test_validation_errors():
    with pytest.raises(ValueError):
        DetectionScheduler(0)
    with pytest.raises(ValueError):
        RollingFps(0)
