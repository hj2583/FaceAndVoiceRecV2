Status: Completed

Commits:
- feat: add lightweight realtime face detector

Files changed:
- config.py
- detection_core.py
- tests/test_config.py (appended test already present)
- tests/test_detection_core.py (updated and added OpenCV tests)

Test summary:
- Command: `python -m pytest tests/test_config.py tests/test_detection_core.py -q -v`
- Result: 25 passed in 0.16s

Self-review:
- Implemented `REALTIME_*` configuration constants in `config.py` as requested.
- Added a public `detect_faces_opencv()` that uses a cached OpenCV `CascadeClassifier` with a lock.
- Kept the existing private `_detect_faces_opencv()` as a fallback; `detect_faces_opencv()` mirrors its behavior.
- Updated tests to avoid importing the real `deepface` package by injecting a fake module and to assert cascade caching without relying on system `cv2` internals.
- Preserved existing detection logic and applied `min_confidence` consistently to both full-frame and tiled detections per supplemental context.

Concerns:
- Tests mock `cv2` and `deepface` to avoid heavy native dependencies; behavior on real hardware/OpenCV builds should be verified in an integration environment.
- Default confidence value for OpenCV detections is set to 0.7 arbitrarily; this may need tuning.

End of report.
