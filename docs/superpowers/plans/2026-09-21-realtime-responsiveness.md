# Realtime Responsiveness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace synchronous tiled RetinaFace work in realtime mode with scheduled lightweight detection, preserve recognition behavior, tolerate unavailable audio, and report immediate launch failures.

**Architecture:** Keep the quality-oriented tiled detector unchanged for uploaded videos and add a cached OpenCV detector for realtime use. Put detection scheduling, rolling FPS calculation, and process-launch status into small dependency-light modules so they can be unit tested without importing the camera loop or Streamlit application.

**Tech Stack:** Python 3.12, OpenCV, NumPy, MediaPipe, Streamlit, PyAudio/WebRTC VAD, pytest, unittest.mock

## Global Constraints

- Target at least 24 displayed FPS after warm-up at 1280x720 on the current machine; this is a manual acceptance target, not a cross-machine guarantee.
- Run lightweight realtime detection every three captured frames by default.
- Calculate displayed FPS over a rolling one-second window.
- Clear cached detections after a scheduled detector failure; do not retain stale boxes indefinitely.
- Keep uploaded-video tiled RetinaFace detection unchanged.
- Keep the native OpenCV camera window and pump `imshow`/`waitKey(1)` every captured frame.
- Treat microphone/VAD startup failure as nonfatal and display `MIC: UNAVAILABLE`.
- Treat a child process still running after one second as successfully started.
- Capture child output in a unique file under `LOG_DIR`; display at most the final 4 KiB after immediate startup failure.
- Preserve existing uncommitted changes in `detection_core.py` and `tests/test_detection_core.py`; integrate with them and never overwrite or revert them.
- Do not add GPU inference, multiprocessing, browser video, live process supervision, log retention management, or unrelated refactors.

---

## File Structure

- Modify `config.py`: define realtime detector and FPS defaults separately from long-range video settings.
- Modify `detection_core.py`: expose cached, failure-contained OpenCV detection while preserving tiled detection.
- Modify `tests/test_detection_core.py`: test the public lightweight detector and cascade caching alongside existing uncommitted tests.
- Create `realtime_runtime.py`: own frame-based detection scheduling and rolling FPS state without camera/model imports.
- Create `tests/test_realtime_runtime.py`: test scheduling, reuse, failure clearing, and FPS calculation deterministically.
- Modify `audio_core.py`: make asynchronous microphone startup report success or failure to its caller.
- Create `tests/test_audio_core.py`: test microphone worker startup without physical audio hardware.
- Modify `realtime.py`: wire lightweight scheduling, FPS display, graceful VAD startup, cleanup, and actionable fatal logging into the existing loop.
- Create `tests/test_realtime.py`: test the isolated VAD-start helper without physical devices.
- Create `realtime_launcher.py`: launch the child with file-backed output and classify its one-second startup result.
- Create `tests/test_realtime_launcher.py`: test running and immediate-failure process outcomes without Streamlit.
- Modify `app.py`: call the launcher and render accurate startup success/error feedback.
- Modify `README.md`: document lightweight realtime behavior, native window controls, audio degradation, and the performance check.

---

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

### Task 2: Realtime Scheduling And FPS State

**Files:**
- Create: `realtime_runtime.py`
- Create: `tests/test_realtime_runtime.py`

**Interfaces:**
- Consumes: detector callable `(frame) -> list[Detection]` and monotonic timestamps supplied by the camera loop.
- Produces: `DetectionScheduler(interval: int)` with `update(frame_no: int, frame, detector) -> list`.
- Produces: `RollingFps(window_seconds: float)` with `tick(timestamp: float) -> float`.

- [ ] **Step 1: Write failing scheduler tests**

Create `tests/test_realtime_runtime.py`:

```python
from unittest.mock import Mock

from realtime_runtime import DetectionScheduler, RollingFps


def test_detection_scheduler_runs_every_third_frame_and_reuses_results():
    detector = Mock(return_value=[(1.0, 2.0, 3.0, 4.0, 0.7)])
    scheduler = DetectionScheduler(interval=3)

    assert scheduler.update(1, object(), detector) == detector.return_value
    assert scheduler.update(2, object(), detector) == detector.return_value
    assert scheduler.update(3, object(), detector) == detector.return_value
    assert scheduler.update(4, object(), detector) == detector.return_value
    assert detector.call_count == 2


def test_detection_scheduler_clears_results_after_detector_exception():
    detector = Mock(side_effect=[[(1.0, 2.0, 3.0, 4.0, 0.7)], RuntimeError("failed")])
    scheduler = DetectionScheduler(interval=3)

    assert scheduler.update(1, object(), detector)
    assert scheduler.update(4, object(), detector) == []
    assert scheduler.update(5, object(), detector) == []
```

- [ ] **Step 2: Write failing rolling-FPS test**

Append:

```python
def test_rolling_fps_uses_only_the_configured_window():
    counter = RollingFps(window_seconds=1.0)

    assert counter.tick(0.0) == 0.0
    counter.tick(0.25)
    counter.tick(0.5)
    fps = counter.tick(1.0)

    assert fps == 3.0
    assert counter.tick(1.5) == 2.0
```

- [ ] **Step 3: Run the runtime tests and verify they fail**

Run: `python -m pytest tests/test_realtime_runtime.py -v`

Expected: collection FAIL because `realtime_runtime` does not exist.

- [ ] **Step 4: Implement minimal dependency-light runtime state**

Create `realtime_runtime.py`:

```python
import logging
from collections import deque


logger = logging.getLogger(__name__)


class DetectionScheduler:
    def __init__(self, interval):
        if interval < 1:
            raise ValueError("interval must be at least 1")
        self.interval = interval
        self.detections = []

    def update(self, frame_no, frame, detector):
        if frame_no == 1 or (frame_no - 1) % self.interval == 0:
            try:
                self.detections = detector(frame)
            except Exception:
                logger.exception("Realtime face detection failed")
                self.detections = []
        return self.detections


class RollingFps:
    def __init__(self, window_seconds):
        if window_seconds <= 0:
            raise ValueError("window_seconds must be positive")
        self.window_seconds = window_seconds
        self.timestamps = deque()

    def tick(self, timestamp):
        self.timestamps.append(timestamp)
        cutoff = timestamp - self.window_seconds
        while self.timestamps and self.timestamps[0] < cutoff:
            self.timestamps.popleft()
        if len(self.timestamps) < 2:
            return 0.0
        duration = self.timestamps[-1] - self.timestamps[0]
        return (len(self.timestamps) - 1) / duration if duration > 0 else 0.0
```

- [ ] **Step 5: Run the runtime tests and verify they pass**

Run: `python -m pytest tests/test_realtime_runtime.py -v`

Expected: PASS.

- [ ] **Step 6: Commit Task 2**

```powershell
git add realtime_runtime.py tests/test_realtime_runtime.py
git commit -m "feat: add realtime frame scheduling"
```

---

### Task 3: Wire Responsive Realtime Processing And Audio Degradation

**Files:**
- Modify: `audio_core.py`
- Modify: `realtime.py`
- Create: `tests/test_audio_core.py`
- Create: `tests/test_realtime.py`

**Interfaces:**
- Consumes: Task 1 `detect_faces_opencv(...)`, Task 2 `DetectionScheduler` and `RollingFps`, and Task 1 `REALTIME_*` constants.
- Produces: `RealtimeVAD.start(startup_timeout: float = 2.0)`, which returns only after the worker opens its input stream and raises `RuntimeError` if startup fails or times out.
- Produces: `start_realtime_vad(callback, vad_factory=RealtimeVAD) -> tuple[RealtimeVAD | None, bool]`, where the boolean is audio availability.
- Preserves: `run(camera=0, width=1280, height=720)` CLI behavior.

- [ ] **Step 1: Write failing asynchronous audio startup tests**

Create `tests/test_audio_core.py`:

```python
import threading
from unittest.mock import Mock, patch

import pytest

from audio_core import RealtimeVAD


def test_realtime_vad_start_waits_for_open_stream():
    release_worker = threading.Event()
    stream = Mock()
    audio = Mock()
    audio.open.return_value = stream

    def stop_after_start(*args, **kwargs):
        release_worker.wait(timeout=1.0)
        raise RuntimeError("stop worker")

    stream.read.side_effect = stop_after_start

    with patch("audio_core.pyaudio.PyAudio", return_value=audio):
        vad = RealtimeVAD()
        vad.start(startup_timeout=1.0)
        assert vad.available is True
        release_worker.set()
        vad.stop()


def test_realtime_vad_start_raises_when_input_stream_cannot_open():
    audio = Mock()
    audio.open.side_effect = OSError("no microphone")

    with patch("audio_core.pyaudio.PyAudio", return_value=audio):
        vad = RealtimeVAD()
        with pytest.raises(RuntimeError, match="no microphone"):
            vad.start(startup_timeout=1.0)
```

- [ ] **Step 2: Run the audio startup tests and verify they fail**

Run: `python -m pytest tests/test_audio_core.py -v`

Expected: FAIL because `RealtimeVAD.start()` does not accept `startup_timeout`, does not wait for `audio.open()`, and does not expose `available`.

- [ ] **Step 3: Report worker startup through `RealtimeVAD.start()`**

In `audio_core.py`, initialize this state in `RealtimeVAD.__init__`:

```python
self.startup_event = threading.Event()
self.available = False
self.startup_error = None
```

Change `start()` to clear that state, launch the existing worker, and wait for startup:

```python
def start(self, startup_timeout=2.0):
    check_audio_dependencies()
    self.stop_event.clear()
    self.startup_event.clear()
    self.available = False
    self.startup_error = None
    self.speaking = False
    self.speech_frames = 0
    self.silence_frames = 0
    self.thread = threading.Thread(target=self._run, daemon=True)
    self.thread.start()

    if not self.startup_event.wait(timeout=startup_timeout):
        self.stop()
        raise RuntimeError("Microphone startup timed out")
    if not self.available:
        message = str(self.startup_error) if self.startup_error else "Microphone startup failed"
        self.stop()
        raise RuntimeError(message) from self.startup_error
```

In `_run()`, set `available = True` and signal `startup_event` immediately after `audio.open()` succeeds. In its exception handler, store the exception in `startup_error`. In `finally`, always signal `startup_event` so `start()` cannot wait until timeout after an immediate failure, and set `available = False` during shutdown.

- [ ] **Step 4: Run the audio tests and verify they pass**

Run: `python -m pytest tests/test_audio_core.py -v`

Expected: PASS; the fake worker may log its intentional `stop worker` exception but must terminate cleanly.

- [ ] **Step 5: Write failing VAD degradation tests**

Create `tests/test_realtime.py`:

```python
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
```

- [ ] **Step 6: Run the VAD degradation tests and verify they fail**

Run: `python -m pytest tests/test_realtime.py -v`

Expected: collection FAIL because `start_realtime_vad` does not exist.

- [ ] **Step 7: Add the VAD helper and guarded cleanup**

Add near `SpeakingState` in `realtime.py`:

```python
def start_realtime_vad(callback, vad_factory=RealtimeVAD):
    try:
        vad = vad_factory(callback=callback)
        vad.start()
        return vad, True
    except Exception:
        logging.exception("Realtime microphone is unavailable")
        return None, False
```

Replace direct VAD construction/start with this helper. Before the initialization `try`, set `index = None`, `tracker = None`, `vad = None`, `mesh = None`, `last_speaker_id = None`, and `last_speech_start = None`. Move FaceIndex, tracker, VAD, and MediaPipe construction inside that `try` so its `finally` always releases the already-open camera. Guard final speaker lookup/logging with `tracker is not None`; call `vad.stop()` and `mesh.close()` only when each resource is initialized.

- [ ] **Step 8: Run the VAD tests and verify they pass**

Run: `python -m pytest tests/test_realtime.py -v`

Expected: PASS without opening a physical microphone or camera.

- [ ] **Step 9: Replace tiled detection with scheduled lightweight detection**

Update `realtime.py` imports to use:

```python
from config import (
    REALTIME_DETECTION_INTERVAL,
    REALTIME_FPS_WINDOW_SECONDS,
    REALTIME_HAAR_MIN_NEIGHBORS,
    REALTIME_HAAR_SCALE_FACTOR,
)
from detection_core import detect_faces_opencv
from realtime_runtime import DetectionScheduler, RollingFps
```

Remove realtime imports and usage of `TILE_GRID`, `TILE_OVERLAP`, `TILE_UPSCALE`, `TILED_DETECTION_INTERVAL`, `NMS_IOU_THRESHOLD`, and `detect_faces_tiled`. Initialize:

```python
detection_scheduler = DetectionScheduler(REALTIME_DETECTION_INTERVAL)
fps_counter = RollingFps(REALTIME_FPS_WINDOW_SECONDS)
```

Replace the tiled detection block with:

```python
last_tiled_detections = detection_scheduler.update(
    frame_no,
    frame,
    lambda image: detect_faces_opencv(
        image,
        scale_factor=REALTIME_HAAR_SCALE_FACTOR,
        min_neighbors=REALTIME_HAAR_MIN_NEIGHBORS,
    ),
)
```

Rename `last_tiled_detections` to `face_detections` throughout the local loop because the values are no longer tiled.

- [ ] **Step 10: Add FPS, unavailable-audio, and window lifecycle handling**

After computing `timestamp`, call `display_fps = fps_counter.tick(timestamp)`. Before `imshow`, render:

```python
cv2.putText(
    frame,
    f"FPS: {display_fps:.1f}",
    (20, 30),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.7,
    (255, 255, 255),
    2,
)
```

Move the existing speaking banner below this line to prevent overlap. Change the microphone label expression to show `MIC: UNAVAILABLE` when `audio_available` is false, otherwise retain `MIC: SPEECH`/`MIC: SILENT`. Gate active-speaker candidate selection and audio event logging on `audio_available`.

Define the window title once and use it for both display and closure detection:

```python
window_name = "AI Face + Active Speaker Recognition"
cv2.imshow(window_name, frame)
key = cv2.waitKey(1) & 0xFF
if key == ord("q") or key == 27:
    break
try:
    if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
        break
except cv2.error:
    break
```

- [ ] **Step 11: Make fatal CLI errors actionable**

Wrap the entry-point `run(camera=args.camera)` call:

```python
try:
    run(camera=args.camera)
except Exception:
    logging.exception("Realtime recognition failed")
    raise SystemExit(1)
```

This preserves a nonzero exit code and sends the traceback to the launcher log.

- [ ] **Step 12: Run realtime unit tests and syntax diagnostics**

Run: `python -m pytest tests/test_audio_core.py tests/test_realtime.py tests/test_realtime_runtime.py -v`

Run: `python -m py_compile realtime.py realtime_runtime.py`

Expected: all tests PASS and compilation exits 0.

- [ ] **Step 13: Commit Task 3**

```powershell
git add audio_core.py realtime.py tests/test_audio_core.py tests/test_realtime.py
git commit -m "fix: keep realtime camera loop responsive"
```

---

### Task 4: Report Child-Process Startup Failures In Streamlit

**Files:**
- Create: `realtime_launcher.py`
- Create: `tests/test_realtime_launcher.py`
- Modify: `app.py`

**Interfaces:**
- Consumes: Python executable path, realtime script path, camera index, working directory, and `LOG_DIR`.
- Produces: immutable `LaunchResult(started: bool, pid: int | None, error: str | None, log_path: Path)`.
- Produces: `launch_realtime(script: Path, camera: int, cwd: Path, log_dir: Path, startup_timeout: float = 1.0, popen=subprocess.Popen) -> LaunchResult`.

- [ ] **Step 1: Write failing launcher tests**

Create `tests/test_realtime_launcher.py`:

```python
from pathlib import Path
import subprocess
from unittest.mock import Mock

from realtime_launcher import launch_realtime


def test_launch_realtime_reports_running_child(tmp_path):
    process = Mock(pid=1234)
    process.wait.side_effect = subprocess.TimeoutExpired(
        cmd=["python", "realtime.py"],
        timeout=1.0,
    )
    popen = Mock(return_value=process)

    result = launch_realtime(
        Path("realtime.py"), 0, tmp_path, tmp_path, popen=popen
    )

    assert result.started is True
    assert result.pid == 1234
    assert result.error is None


def test_launch_realtime_returns_tail_for_immediate_failure(tmp_path):
    process = Mock(pid=1234)
    process.wait.return_value = 1

    def fake_popen(*args, **kwargs):
        kwargs["stdout"].write(b"x" * 5000 + b"camera failed")
        kwargs["stdout"].flush()
        return process

    result = launch_realtime(
        Path("realtime.py"), 9, tmp_path, tmp_path, popen=fake_popen
    )

    assert result.started is False
    assert result.error.endswith("camera failed")
    assert len(result.error.encode("utf-8")) <= 4096
```

- [ ] **Step 2: Run launcher tests and verify they fail**

Run: `python -m pytest tests/test_realtime_launcher.py -v`

Expected: collection FAIL because `realtime_launcher` does not exist.

- [ ] **Step 3: Implement the standalone launcher**

Create `realtime_launcher.py` with:

```python
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import os
import subprocess
import sys


@dataclass(frozen=True)
class LaunchResult:
    started: bool
    pid: int | None
    error: str | None
    log_path: Path


def _read_tail(path, max_bytes=4096):
    with path.open("rb") as log_file:
        log_file.seek(0, os.SEEK_END)
        size = log_file.tell()
        log_file.seek(max(0, size - max_bytes))
        return log_file.read(max_bytes).decode("utf-8", errors="replace").strip()


def launch_realtime(script, camera, cwd, log_dir, startup_timeout=1.0, popen=subprocess.Popen):
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    log_path = Path(log_dir) / f"realtime-{stamp}.log"
    command = [sys.executable, str(script), "--camera", str(camera)]
    creationflags = subprocess.CREATE_NEW_CONSOLE if os.name == "nt" else 0

    with log_path.open("wb") as output:
        process = popen(
            command,
            cwd=str(cwd),
            stdout=output,
            stderr=subprocess.STDOUT,
            creationflags=creationflags,
        )
        try:
            exit_code = process.wait(timeout=startup_timeout)
        except subprocess.TimeoutExpired:
            return LaunchResult(True, process.pid, None, log_path)

    error = _read_tail(log_path) or f"Realtime process exited with code {exit_code}."
    return LaunchResult(False, process.pid, error, log_path)
```

Ensure `log_dir.mkdir(parents=True, exist_ok=True)` runs before opening the file. On non-Windows, omit `creationflags` from the `Popen` keyword arguments rather than passing Windows-only flags.

- [ ] **Step 4: Run launcher tests and verify they pass**

Run: `python -m pytest tests/test_realtime_launcher.py -v`

Expected: PASS.

- [ ] **Step 5: Replace direct `Popen` use in Streamlit**

In `app.py`, remove direct `os` and `subprocess` imports if unused elsewhere, import `launch_realtime`, and replace the button body with:

```python
with st.spinner("Starting realtime recognition..."):
    result = launch_realtime(
        script=Path(__file__).parent / "realtime.py",
        camera=int(camera),
        cwd=Path(__file__).parent,
        log_dir=LOG_DIR,
    )

if result.started:
    st.success(
        "Realtime recognition started. A camera window should appear. "
        "Press Q/ESC in that window to stop."
    )
else:
    st.error(f"Realtime recognition failed to start:\n\n{result.error}")
```

- [ ] **Step 6: Run launcher tests and compile the Streamlit entry point**

Run: `python -m pytest tests/test_realtime_launcher.py -v`

Run: `python -m py_compile app.py realtime_launcher.py`

Expected: tests PASS and compilation exits 0.

- [ ] **Step 7: Commit Task 4**

```powershell
git add app.py realtime_launcher.py tests/test_realtime_launcher.py
git commit -m "fix: report realtime startup failures"
```

---

### Task 5: Regression And Manual Acceptance Validation

**Files:**
- Modify: `README.md`

**Interfaces:**
- Consumes: completed detector, runtime, audio, and launcher behavior from Tasks 1-4.
- Produces: documented operation and recorded validation evidence; no new runtime API.

- [ ] **Step 1: Run the complete automated suite**

Run: `python -m pytest -v`

Expected: all repository tests PASS. If an unrelated pre-existing test fails, record its exact name and output and continue with only task-scoped verification; do not change unrelated code.

- [ ] **Step 2: Run static syntax validation for all touched Python files**

Run:

```powershell
python -m py_compile app.py config.py detection_core.py realtime.py realtime_launcher.py realtime_runtime.py
```

Expected: exit code 0 with no output.

- [ ] **Step 3: Document the realtime operating model**

Update the realtime section of `README.md` to state:

```markdown
Realtime mode uses lightweight OpenCV face detection to keep the native camera window responsive. Uploaded-video mode retains the heavier long-range detector. The window displays rolling FPS; Q or Escape stops it. If microphone initialization fails, face detection continues with `MIC: UNAVAILABLE` and active-speaker estimation disabled.
```

Also document that immediate startup errors appear in Streamlit and detailed child output is stored under `logs/realtime-*.log`.

- [ ] **Step 4: Start Streamlit for manual validation**

Run: `streamlit run app.py`

Expected: Streamlit prints a local URL and remains running. Open the Realtime page and start camera index 0.

- [ ] **Step 5: Verify the approved manual acceptance criteria**

Observe and record:

1. Rolling FPS reaches at least 24.0 after a five-second warm-up at 1280x720.
2. The native window can be moved and remains responsive.
3. A normal-distance known face is recognized.
4. Unknown-face sampling and database logging still occur.
5. Q and Escape each close a fresh run promptly.
6. With the microphone unavailable or mocked to fail, video continues and shows `MIC: UNAVAILABLE`.
7. An invalid camera index reports an actionable startup error in Streamlit.

If FPS remains below 24, record the measured rate and stop completion of this plan. Amend the approved design and this plan with a measured landmark or ArcFace scheduling change before editing further; do not introduce threading or remove recognition without that review.

- [ ] **Step 6: Commit documentation and any test-only acceptance notes**

```powershell
git add README.md
git commit -m "docs: explain responsive realtime mode"
```

- [ ] **Step 7: Inspect final scope**

Run: `git status --short`

Run: `git log --oneline -5`

Expected: only intentional files are changed, the original uncommitted detector work was preserved in Task 1, and Tasks 1-5 appear as focused commits.