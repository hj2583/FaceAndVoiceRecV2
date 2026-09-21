import logging
from collections import deque
from typing import Callable, Deque, List, Any

logger = logging.getLogger(__name__)

# small epsilon to avoid absurd FPS values when duration is tiny but > 0
_DURATION_EPSILON = 1e-6

class DetectionScheduler:
    def __init__(self, interval: int):
        if interval < 1:
            raise ValueError("interval must be >= 1")
        self.interval = int(interval)
        self.last_detections: List = []

    def update(self, frame_no: int, frame: Any, detector: Callable[[Any], List]):
        try:
            # schedule detection on frames 1, 1+interval, 1+2*interval, ...
            if (frame_no - 1) % self.interval == 0:
                detections = detector(frame)
                self.last_detections = list(detections) if detections is not None else []
            return list(self.last_detections)
        except Exception:
            logger.exception("Realtime face detection failed")
            self.last_detections = []
            return []


class RollingFps:
    def __init__(self, window_seconds: float):
        if window_seconds <= 0:
            raise ValueError("window_seconds must be > 0")
        self.window_seconds = float(window_seconds)
        self._q: Deque[float] = deque()

    def tick(self, timestamp: float) -> float:
        ts = float(timestamp)
        self._q.append(ts)
        cutoff = ts - self.window_seconds
        # remove timestamps strictly older than cutoff
        while self._q and self._q[0] < cutoff:
            self._q.popleft()

        count = len(self._q)
        if count < 2:
            return 0.0

        duration = self._q[-1] - self._q[0]
        # guard against zero or extremely small durations which would
        # produce an absurdly large FPS value due to floating point
        # and input noise. Use a small epsilon and return 0.0 in those
        # edge cases to preserve scheduling semantics.
        if duration == 0 or duration < _DURATION_EPSILON:
            return 0.0

        return (count - 1) / duration
