### Task 1: Cached Lightweight Detector And Configuration

**Files:**
- Modify: `config.py`
- Modify: `detection_core.py`
- Test: `tests/test_config.py`
- Test: `tests/test_detection_core.py`

**Interfaces:**
- Consumes: OpenCV `CascadeClassifier.detectMultiScale()` and the existing `Detection = Tuple[float, float, float, float, float]` contract.
- Produces: `detect_faces_opencv(frame: np.ndarray, scale_factor: float = 1.1, min_neighbors: int = 5, min_size: tuple[int, int] = (20, 20)) -> List[Detection]`.
- Produces: `REALTIME_DETECTION_INTERVAL = 3`, `REALTIME_HAAR_SCALE_FACTOR = 1.1`, `REALTIME_HAAR_MIN_NEIGHBORS = 5`, and `REALTIME_FPS_WINDOW_SECONDS = 1.0`.

- [ ] **Step 1: Add failing configuration tests**

Append to `tests/test_config.py`:

```python
def test_realtime_detection_defaults_prioritize_responsiveness():
    assert config.REALTIME_DETECTION_INTERVAL == 3
    assert config.REALTIME_HAAR_SCALE_FACTOR > 1.0
    assert config.REALTIME_HAAR_MIN_NEIGHBORS >= 1
    assert config.REALTIME_FPS_WINDOW_SECONDS == 1.0
```

- [ ] **Step 2: Run the configuration test and verify it fails**

Run: `python -m pytest tests/test_config.py::test_realtime_detection_defaults_prioritize_responsiveness -v`

Expected: FAIL because the `REALTIME_*` names do not exist.

- [ ] **Step 3: Add the realtime configuration block**

Add below the tracking settings in `config.py`, keeping the existing long-range block unchanged:

```python
# Realtime mode favors display responsiveness over long-range detection.
REALTIME_DETECTION_INTERVAL = 3
REALTIME_HAAR_SCALE_FACTOR = 1.1
REALTIME_HAAR_MIN_NEIGHBORS = 5
REALTIME_FPS_WINDOW_SECONDS = 1.0
```

- [ ] **Step 4: Run the configuration test and verify it passes**

Run: `python -m pytest tests/test_config.py::test_realtime_detection_defaults_prioritize_responsiveness -v`

Expected: PASS.

- [ ] **Step 5: Add failing detector contract and cache tests**

Update the imports in `tests/test_detection_core.py` to include `detect_faces_opencv`, then add:

```python
def test_detect_faces_opencv_returns_detection_contract():
    frame = np.zeros((80, 100, 3), dtype=np.uint8)
    cascade = Mock()
    cascade.empty.return_value = False
    cascade.detectMultiScale.return_value = [(4, 6, 20, 24)]

    with patch("detection_core._get_opencv_cascade", return_value=cascade):
        detections = detect_faces_opencv(frame)

    assert detections == [(4.0, 6.0, 20.0, 24.0, 0.7)]


def test_detect_faces_opencv_reuses_cached_cascade():
    import detection_core

    detection_core._OPENCV_CASCADE = None
    cascade = Mock()
    cascade.empty.return_value = False
    cascade.detectMultiScale.return_value = []

    with patch("detection_core.cv2.CascadeClassifier", return_value=cascade) as factory:
        detect_faces_opencv(np.zeros((40, 40, 3), dtype=np.uint8))
        detect_faces_opencv(np.zeros((40, 40, 3), dtype=np.uint8))

    factory.assert_called_once()
```

Also import `Mock` from `unittest.mock`. Reset `_OPENCV_CASCADE` in a `finally` block or fixture so the test cannot leak cache state.

- [ ] **Step 6: Add failing empty-frame and exception tests**

Add to `tests/test_detection_core.py`:

```python
def test_detect_faces_opencv_returns_empty_for_invalid_frame():
    assert detect_faces_opencv(None) == []
    assert detect_faces_opencv(np.empty((0, 0, 3), dtype=np.uint8)) == []


def test_detect_faces_opencv_contains_detection_failure():
    cascade = Mock()
    cascade.empty.return_value = False
    cascade.detectMultiScale.side_effect = RuntimeError("cascade failed")

    with patch("detection_core._get_opencv_cascade", return_value=cascade):
        assert detect_faces_opencv(np.zeros((40, 40, 3), dtype=np.uint8)) == []
```

- [ ] **Step 7: Run the detector tests and verify they fail**

Run: `python -m pytest tests/test_detection_core.py -k "opencv" -v`

Expected: FAIL because the public detector/cache interface does not exist or reconstructs the cascade for each call.

- [ ] **Step 8: Implement the cached detector**

In `detection_core.py`, add `threading`, `_OPENCV_CASCADE = None`, a lock, and this public boundary:

```python
_OPENCV_CASCADE = None
_OPENCV_CASCADE_LOCK = threading.Lock()


def _get_opencv_cascade():
    global _OPENCV_CASCADE
    if _OPENCV_CASCADE is None:
        with _OPENCV_CASCADE_LOCK:
            if _OPENCV_CASCADE is None:
                cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
                _OPENCV_CASCADE = cv2.CascadeClassifier(cascade_path)
    return _OPENCV_CASCADE


def detect_faces_opencv(
    frame: np.ndarray,
    scale_factor: float = 1.1,
    min_neighbors: int = 5,
    min_size: tuple[int, int] = (20, 20),
) -> List[Detection]:
    if frame is None or frame.size == 0:
        return []

    try:
        cascade = _get_opencv_cascade()
        if cascade.empty():
            logger.warning("OpenCV face cascade is unavailable")
            return []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(
            gray,
            scaleFactor=scale_factor,
            minNeighbors=min_neighbors,
            minSize=min_size,
        )
        return [
            (float(x), float(y), float(width), float(height), 0.7)
            for x, y, width, height in faces
        ]
    except Exception as error:
        logger.warning("OpenCV face detection failed: %s", error)
        return []
```

Make the existing private `_detect_faces_opencv()` fallback delegate to `detect_faces_opencv()` so both paths share the cache. Preserve the current lazy backend/import edits already present in the worktree.

- [ ] **Step 9: Run focused and full detector tests**

Run: `python -m pytest tests/test_config.py tests/test_detection_core.py -v`

Expected: PASS, including all pre-existing tiled detection tests.

- [ ] **Step 10: Commit Task 1**

```powershell
git add config.py detection_core.py tests/test_config.py tests/test_detection_core.py
git commit -m "feat: add lightweight realtime face detector"
```

Before staging, inspect `git diff` and confirm the commit retains the user's pre-existing changes in `detection_core.py` and `tests/test_detection_core.py`.

---

