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

Fix details performed:

- Tests: Replaced Mock-based module injection of `deepface` with `types.ModuleType` module-like doubles in [tests/test_detection_core.py](tests/test_detection_core.py#L96-L114) and [tests/test_detection_core.py](tests/test_detection_core.py#L129-L148) to more accurately simulate import-time module behavior. This prevents importing the real `deepface` package while allowing `DeepFace.extract_faces` to be called normally.
- Code: Made Haar cascade path construction portable in `detection_core.py` by using `os.path.join` and safe attribute access (`getattr(cv2, "data", None)` / `getattr(..., "haarcascades", "")`) so tests can override `cv2.data.haarcascades` when monkeypatching `cv2`. Change location: [detection_core.py](detection_core.py#L24-L32).

Test run (exact):

```
python -m pytest tests/test_config.py tests/test_detection_core.py -q -q

25 passed
```

Additional runs for regression fix and focused suite:

```
python -m pytest tests/test_config.py::test_config_exists tests/test_detection_core.py::test_get_opencv_cascade_handles_none_haarcascades -q -q

2 passed
```

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
