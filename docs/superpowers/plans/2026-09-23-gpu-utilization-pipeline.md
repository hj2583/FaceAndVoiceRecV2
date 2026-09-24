# GPU Utilization Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Overlap CPU decode/encode with GPU inference in both the offline video pipeline and the live camera pipeline by splitting each into reader/compute/writer threads connected by queues, so the GPU is no longer idle waiting on CPU work.

**Architecture:** A single reusable `run_threaded_pipeline(read_frame, process_frame, write_frame, ...)` helper in a new `frame_pipeline.py` module runs three threads (reader, single compute thread, writer) connected by two `queue.Queue` instances, with a shared `threading.Event` for shutdown and a shared exception box for error propagation. `video_processor.py` and `realtime.py` are refactored to supply plain `read_frame`/`process_frame`/`write_frame` callables to this helper instead of running one monolithic per-frame loop, without changing any detection/recognition/tracking logic.

**Tech Stack:** Python `threading`, `queue.Queue` (standard library only — no new dependencies).

## Global Constraints

- No change to detection/recognition/tracking algorithms, thresholds, or scheduling intervals (`RECOGNITION_INTERVAL`, `TILED_DETECTION_INTERVAL`, `REALTIME_DETECTION_INTERVAL`, etc. from `config.py` stay exactly as-is).
- Offline pipeline (`video_processor.py`) must never drop frames: output frame count must exactly equal input frame count. Queues there are bounded and blocking (`maxsize=8`).
- Realtime pipeline (`realtime.py`) uses `maxsize=1` drop-oldest queues — showing the newest frame takes priority over showing every frame.
- The face tracker (`CentroidTracker`) must observe frames in strict order — the compute stage is always a single thread, never a pool.
- Any exception raised in any pipeline thread must propagate out of `process_video_pipeline(...)` / `run(...)` exactly as it would today (no silent hangs, no swallowed errors), and all threads must be joined before the function returns.
- Follow existing test patterns already in `tests/test_realtime.py` (full monkeypatch of `cv2`, `FaceIndex`, `CentroidTracker`, `create_face_landmarker`, etc.) for any new integration tests.

---

### Task 1: Generic threaded pipeline utility

**Files:**
- Create: `frame_pipeline.py`
- Test: `tests/test_frame_pipeline.py`

**Interfaces:**
- Produces: `run_threaded_pipeline(read_frame, process_frame, write_frame, queue_maxsize=8, drop_oldest=False, poll_interval=0.5)` — a function with no other dependencies (no `cv2`, no domain logic). `read_frame()` takes no args and returns a frame-like object or `None` to signal end-of-stream. `process_frame(item)` takes one item and returns one item. `write_frame(item)` takes one item and returns nothing. Raises the first exception encountered in any stage after all three threads have stopped and been joined.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_frame_pipeline.py`:

```python
import queue
import threading
import time

import pytest

from frame_pipeline import run_threaded_pipeline


def test_preserves_order_and_count_with_blocking_queues():
    items = list(range(20))
    written = []

    def read_frame(_it=iter(items)):
        return next(_it, None)

    def process_frame(item):
        return item * 2

    def write_frame(item):
        written.append(item)

    run_threaded_pipeline(read_frame, process_frame, write_frame, queue_maxsize=3)

    assert written == [item * 2 for item in items]


def test_propagates_exception_from_process_stage_and_stops_all_threads():
    def read_frame(_it=iter(range(5))):
        return next(_it, None)

    def process_frame(item):
        if item == 2:
            raise ValueError("boom")
        return item

    written = []

    with pytest.raises(ValueError, match="boom"):
        run_threaded_pipeline(read_frame, process_frame, written.append, queue_maxsize=2)

    # No thread should be left running.
    time.sleep(0.1)
    active_names = {t.name for t in threading.enumerate()}
    assert not any(name.startswith("frame-pipeline-") for name in active_names)


def test_drop_oldest_keeps_latest_item_when_consumer_is_slow():
    produced = list(range(50))
    write_started = threading.Event()
    release_write = threading.Event()
    written = []

    def read_frame(_it=iter(produced)):
        return next(_it, None)

    def process_frame(item):
        return item

    def write_frame(item):
        if item == produced[0]:
            write_started.set()
            assert release_write.wait(timeout=2.0)
        written.append(item)

    def _run():
        run_threaded_pipeline(
            read_frame,
            process_frame,
            write_frame,
            drop_oldest=True,
            poll_interval=0.05,
        )

    thread = threading.Thread(target=_run)
    thread.start()
    assert write_started.wait(timeout=2.0)
    release_write.set()
    thread.join(timeout=5.0)
    assert not thread.is_alive()

    # With drop-oldest queues the consumer cannot see every produced item,
    # but it must see the first item and the very last one.
    assert written[0] == produced[0]
    assert written[-1] == produced[-1]
    assert len(written) < len(produced)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_frame_pipeline.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'frame_pipeline'`

- [ ] **Step 3: Implement `frame_pipeline.py`**

```python
import logging
import queue
import threading

logger = logging.getLogger(__name__)

_SENTINEL = object()


class PipelineErrorBox:
    """Shared single-slot holder for the first exception raised by any stage."""

    def __init__(self):
        self._lock = threading.Lock()
        self._exception = None

    def set(self, exc):
        with self._lock:
            if self._exception is None:
                self._exception = exc

    def get(self):
        with self._lock:
            return self._exception


def _run_stage(name, stop_event, error_box, func):
    try:
        func()
    except Exception as exc:  # noqa: BLE001 - recorded and re-raised by the caller
        logger.exception("Pipeline stage '%s' failed", name)
        error_box.set(exc)
        stop_event.set()


def _put_latest(q, item):
    """Put item on a maxsize=1 queue, dropping any previously queued item."""
    while True:
        try:
            q.put_nowait(item)
            return
        except queue.Full:
            try:
                q.get_nowait()
            except queue.Empty:
                pass


def _enqueue(q, item, stop_event, drop_oldest, poll_interval):
    if drop_oldest:
        _put_latest(q, item)
        return True
    while not stop_event.is_set():
        try:
            q.put(item, timeout=poll_interval)
            return True
        except queue.Full:
            continue
    return False


def _dequeue(q, stop_event, poll_interval):
    """Returns (True, item) on success, (False, None) if stop_event fired first."""
    while not stop_event.is_set():
        try:
            return True, q.get(timeout=poll_interval)
        except queue.Empty:
            continue
    return False, None


def run_threaded_pipeline(
    read_frame,
    process_frame,
    write_frame,
    queue_maxsize=8,
    drop_oldest=False,
    poll_interval=0.5,
):
    """Run read_frame -> process_frame -> write_frame across three threads.

    read_frame() returns the next item, or None to signal end-of-stream.
    process_frame(item) transforms one item at a time, in strict input order.
    write_frame(item) consumes one processed item at a time, in order.

    When drop_oldest is False (default), both internal queues are bounded and
    blocking with maxsize=queue_maxsize: no item is ever dropped, at the cost
    of the reader blocking if the compute/writer stages fall behind.

    When drop_oldest is True, both internal queues have maxsize=1 and silently
    replace the queued item when full: the newest item always wins, and no
    stage ever blocks waiting for a slow downstream consumer.

    Raises the first exception encountered in any stage, after all three
    threads have stopped and been joined.
    """
    maxsize = 1 if drop_oldest else queue_maxsize
    raw_q = queue.Queue(maxsize=maxsize)
    out_q = queue.Queue(maxsize=maxsize)

    stop_event = threading.Event()
    error_box = PipelineErrorBox()

    def _reader():
        while not stop_event.is_set():
            frame = read_frame()
            if frame is None:
                _enqueue(raw_q, _SENTINEL, stop_event, drop_oldest, poll_interval)
                return
            if not _enqueue(raw_q, frame, stop_event, drop_oldest, poll_interval):
                return

    def _compute():
        while True:
            ok, frame = _dequeue(raw_q, stop_event, poll_interval)
            if not ok:
                return
            if frame is _SENTINEL:
                _enqueue(out_q, _SENTINEL, stop_event, drop_oldest, poll_interval)
                return
            processed = process_frame(frame)
            if not _enqueue(out_q, processed, stop_event, drop_oldest, poll_interval):
                return

    def _writer():
        while True:
            ok, frame = _dequeue(out_q, stop_event, poll_interval)
            if not ok:
                return
            if frame is _SENTINEL:
                return
            write_frame(frame)

    threads = [
        threading.Thread(
            target=_run_stage,
            args=(name, stop_event, error_box, stage),
            name=f"frame-pipeline-{name}",
            daemon=True,
        )
        for name, stage in (
            ("reader", _reader),
            ("compute", _compute),
            ("writer", _writer),
        )
    ]

    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    exc = error_box.get()
    if exc is not None:
        raise exc
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_frame_pipeline.py -v`
Expected: PASS (all 3 tests)

- [ ] **Step 5: Commit**

```bash
git add frame_pipeline.py tests/test_frame_pipeline.py
git commit -m "feat: add generic threaded read/process/write pipeline helper"
```

---

### Task 2: Wire `video_processor.py` into the threaded pipeline (offline, blocking queues)

**Files:**
- Modify: `video_processor.py:484-1147` (the `try: ... while True: ... finally:` block inside `process_video_pipeline`)
- Test: Create `tests/test_video_processor.py`

**Interfaces:**
- Consumes: `run_threaded_pipeline` from Task 1 (`frame_pipeline.py`), signature as above.
- Produces: `process_video_pipeline(input_path, output_path, log_path, progress_callback=None)` keeps its existing public signature and behavior (same return value: `None`; same side effects: writes `output_path` and `log_path`).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_video_processor.py`, following the full-monkeypatch style already used in `tests/test_realtime.py`:

```python
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

import video_processor


class _FakeCapture:
    def __init__(self, frame_count=5, width=160, height=120, fps=30.0):
        self.frame_count = frame_count
        self.width = width
        self.height = height
        self.fps = fps
        self._read_calls = 0

    def isOpened(self):
        return True

    def get(self, prop):
        return {
            cv2.CAP_PROP_FRAME_WIDTH: self.width,
            cv2.CAP_PROP_FRAME_HEIGHT: self.height,
            cv2.CAP_PROP_FPS: self.fps,
            cv2.CAP_PROP_FRAME_COUNT: self.frame_count,
        }[prop]

    def read(self):
        if self._read_calls >= self.frame_count:
            return False, None
        self._read_calls += 1
        frame = np.full((self.height, self.width, 3), self._read_calls, dtype=np.uint8)
        return True, frame

    def release(self):
        return None


class _FakeWriter:
    def __init__(self):
        self.frames = []
        self._opened = True

    def isOpened(self):
        return self._opened

    def write(self, frame):
        self.frames.append(frame.copy())

    def release(self):
        return None


class _FakeLandmarker:
    def close(self):
        return None


def _patch_common(monkeypatch, capture, writer):
    monkeypatch.setattr(video_processor.cv2, "VideoCapture", lambda *_a, **_k: capture)
    monkeypatch.setattr(video_processor.cv2, "VideoWriter", lambda *_a, **_k: writer)
    monkeypatch.setattr(video_processor.cv2, "VideoWriter_fourcc", lambda *_a: 0)
    monkeypatch.setattr(video_processor, "extract_audio_to_wav", lambda _p: None)
    monkeypatch.setattr(video_processor, "create_face_landmarker", lambda: _FakeLandmarker())
    monkeypatch.setattr(video_processor, "detect_faces_tiled", lambda *_a, **_k: [])
    monkeypatch.setattr(video_processor, "FaceIndex", lambda: object())
    monkeypatch.setattr(
        video_processor,
        "CentroidTracker",
        lambda: SimpleNamespace(update=lambda *_a, **_k: [], tracks={}),
    )
    monkeypatch.setattr(video_processor, "convert_h264", lambda *_a, **_k: False)


def test_process_video_pipeline_writes_one_frame_per_input_frame(monkeypatch, tmp_path):
    capture = _FakeCapture(frame_count=7)
    writer = _FakeWriter()
    _patch_common(monkeypatch, capture, writer)

    output_path = tmp_path / "out.mp4"
    log_path = tmp_path / "log.json"

    video_processor.process_video_pipeline(tmp_path / "in.mp4", output_path, log_path)

    assert len(writer.frames) == 7
    assert log_path.exists()
    data = json.loads(log_path.read_text(encoding="utf-8"))
    assert "recognition" in data


def test_process_video_pipeline_propagates_detection_errors(monkeypatch, tmp_path):
    capture = _FakeCapture(frame_count=3)
    writer = _FakeWriter()
    _patch_common(monkeypatch, capture, writer)

    def _boom(*_args, **_kwargs):
        raise ValueError("detector exploded")

    monkeypatch.setattr(video_processor, "detect_faces_tiled", _boom)

    with pytest.raises(ValueError, match="detector exploded"):
        video_processor.process_video_pipeline(
            tmp_path / "in.mp4", tmp_path / "out.mp4", tmp_path / "log.json"
        )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_video_processor.py -v`
Expected: FAIL — the current implementation still works frame-by-frame synchronously (tests may actually pass already for count/logging, but confirm by temporarily checking there is no `frame_pipeline` import yet in `video_processor.py`; the error-propagation test should already pass today too). Since this task's real goal is the refactor, treat this step as "run and note current pass/fail baseline" — proceed to Step 3 regardless.

- [ ] **Step 3: Refactor `process_video_pipeline` to use `run_threaded_pipeline`**

Add the import at the top of `video_processor.py` (near the other local imports):

```python
from frame_pipeline import run_threaded_pipeline
```

Inside `process_video_pipeline`, replace the block that currently reads:

```python
    try:

        while True:

            ok, frame = cap.read()

            if not ok:
                break

            frame_no += 1

            timestamp = (
                frame_no / fps
            )

            # =================================================
            # Tiled UniFace detection
            # =================================================
            ...
            # (existing per-frame body, lines ~502-1120, UNCHANGED)
            ...

            # =================================================
            # Write frame
            # =================================================

            writer.write(frame)

            # =================================================
            # Progress
            # =================================================

            if (
                progress_callback
                and total_frames
            ):

                progress_callback(
                    frame_no
                    / total_frames
                )

    finally:

        cap.release()

        writer.release()

        face_landmarker.close()
```

with:

```python
    def _read_frame():
        ok, frame = cap.read()
        return frame if ok else None

    def _process(frame):
        nonlocal frame_no, last_tiled_detections

        frame_no += 1

        timestamp = (
            frame_no / fps
        )

        # =================================================
        # Tiled UniFace detection
        # =================================================
        ...
        # (existing per-frame body, lines ~502-1120, UNCHANGED,
        #  moved here verbatim, ending right before "writer.write(frame)")
        ...

        return frame

    def _write_frame(frame):
        writer.write(frame)

        if progress_callback and total_frames:
            progress_callback(frame_no / total_frames)

    try:
        run_threaded_pipeline(
            _read_frame,
            _process,
            _write_frame,
            queue_maxsize=8,
            drop_oldest=False,
        )
    finally:
        cap.release()
        writer.release()
        face_landmarker.close()
```

Notes for this mechanical move:
- Everything between the old `timestamp = frame_no / fps` line and the old `writer.write(frame)` line moves unchanged into `_process`, just re-indented one level to sit inside the nested function.
- `frame_no` and `last_tiled_detections` are reassigned inside the body (`frame_no += 1`, `last_tiled_detections = detect_faces_tiled(...)`), so both need `nonlocal`. Every other variable referenced from the body (`index`, `tracker`, `recognition_logs`, `active_speech_logs`, `speech_segments`, `face_landmarker`, `fps`) is only read or mutated in place (e.g. `recognition_logs.append(...)`) so it needs no `nonlocal` declaration.
- `_process` must `return frame` at the end so `_write_frame` receives it.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_video_processor.py -v`
Expected: PASS (both tests)

- [ ] **Step 5: Run the full existing test suite to check for regressions**

Run: `pytest tests/ -v`
Expected: PASS (no regressions in other test files)

- [ ] **Step 6: Commit**

```bash
git add video_processor.py tests/test_video_processor.py
git commit -m "refactor: run offline video pipeline through threaded reader/compute/writer stages"
```

---

### Task 3: Wire `realtime.py` into the threaded pipeline (live camera, drop-oldest queues)

**Files:**
- Modify: `realtime.py:314-1330` (the `while True: ... finally:` block inside `run`)
- Test: Modify `tests/test_realtime.py`

**Interfaces:**
- Consumes: `run_threaded_pipeline` from Task 1 (`frame_pipeline.py`).
- Produces: `run(camera=0, width=1280, height=720)` keeps its existing public signature and behavior (returns `None`; same window/log/database side effects).

- [ ] **Step 1: Write the failing test**

Add to `tests/test_realtime.py` (reusing the `_Cap`, `_Mesh`, `_Tracker` fixtures and monkeypatch pattern already defined earlier in that file for `test_run_detects_vad_becoming_unavailable`):

```python
def test_run_processes_every_captured_frame_via_threaded_pipeline(monkeypatch):
    import realtime

    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    processed_count = {"value": 0}

    class _Cap:
        def __init__(self):
            self.calls = 0

        def isOpened(self):
            return True

        def set(self, *_args, **_kwargs):
            return True

        def get(self, prop):
            if prop == cv2.CAP_PROP_FPS:
                return 30.0
            if prop == cv2.CAP_PROP_FRAME_WIDTH:
                return 160.0
            if prop == cv2.CAP_PROP_FRAME_HEIGHT:
                return 120.0
            return 0.0

        def read(self):
            self.calls += 1
            if self.calls <= 3:
                return True, frame.copy()
            return False, None

        def release(self):
            return None

    class _Mesh:
        def detect_for_video(self, *_args, **_kwargs):
            processed_count["value"] += 1
            return SimpleNamespace(face_landmarks=[])

        def close(self):
            return None

    class _Tracker:
        def __init__(self):
            self.tracks = {}

        def update(self, detections, frame_no):
            return []

    monkeypatch.setattr(realtime.cv2, "VideoCapture", lambda *_args, **_kwargs: _Cap())
    monkeypatch.setattr(realtime, "FaceIndex", lambda: object())
    monkeypatch.setattr(realtime, "CentroidTracker", lambda: _Tracker())
    monkeypatch.setattr(realtime, "start_realtime_vad", lambda callback: (None, False))
    monkeypatch.setattr(realtime, "create_face_landmarker", lambda: _Mesh())
    monkeypatch.setattr(realtime, "detect_faces_realtime", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(realtime, "log_recognition", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(realtime, "log_audio", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(realtime.cv2, "cvtColor", lambda image, _code: image)
    monkeypatch.setattr(realtime.cv2, "imshow", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(realtime.cv2, "waitKey", lambda *_args, **_kwargs: 0)
    monkeypatch.setattr(realtime.cv2, "getWindowProperty", lambda *_args, **_kwargs: 1)
    monkeypatch.setattr(realtime.cv2, "rectangle", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(realtime.cv2, "putText", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(realtime.cv2, "destroyAllWindows", lambda: None)

    import sys
    fake_video_processor = ModuleType("video_processor")
    fake_video_processor.lip_open_ratio = lambda _landmarks: 0.0
    monkeypatch.setitem(sys.modules, "video_processor", fake_video_processor)
    monkeypatch.setitem(
        sys.modules,
        "mediapipe",
        SimpleNamespace(ImageFormat=SimpleNamespace(SRGB=1), Image=lambda image_format, data: data),
    )

    realtime.run(camera=0, width=160, height=120)
```

(This reuses the same imports — `ModuleType`, `SimpleNamespace`, `Mock`, `cv2`, `np` — already present at the top of `tests/test_realtime.py`; no new imports are needed since `detect_for_video` is only invoked when landmark detection is reached — if it turns out landmark detection is skipped for empty detections in this codepath, drop the `processed_count` assertion and instead assert `_Cap.calls == 4`, i.e. the reader attempted one more read than frames returned, confirming the reader ran to end-of-stream through the new pipeline.)

- [ ] **Step 2: Run test to verify current baseline**

Run: `pytest tests/test_realtime.py -v -k test_run_processes_every_captured_frame_via_threaded_pipeline`
Expected: PASS already today (the assertion only checks `_Cap.calls`, which the current synchronous loop also satisfies) — this test is a regression guard for the refactor in Step 3, not a red/green TDD gate. Proceed to Step 3.

- [ ] **Step 3: Refactor `run()` to use `run_threaded_pipeline`**

Add the import near the top of `realtime.py`:

```python
from frame_pipeline import run_threaded_pipeline
```

Inside `run()`, replace the block that currently reads:

```python
        while True:

            # =================================================
            # Read camera frame
            # =================================================

            ok, frame = cap.read()

            if not ok:
                break

            frame_no += 1
            ...
            # (existing per-frame body, UNCHANGED)
            ...

            cv2.imshow(window_name, frame)
            ...
```

with:

```python
        def _read_frame():
            ok, frame = cap.read()
            return frame if ok else None

        def _process(frame):
            nonlocal frame_no, face_detections, last_landmark_timestamp_ms

            frame_no += 1
            ...
            # (existing per-frame body, UNCHANGED, moved here verbatim,
            #  ending right before "cv2.imshow(window_name, frame)")
            ...

            return frame

        def _write_frame(frame):
            cv2.imshow(window_name, frame)
            ...
            # (existing post-imshow logic that was inside the old loop body
            #  after cv2.imshow — e.g. cv2.waitKey / window-close checks —
            #  stays here, UNCHANGED)

        run_threaded_pipeline(
            _read_frame,
            _process,
            _write_frame,
            drop_oldest=True,
            poll_interval=0.05,
        )
```

Notes for this mechanical move:
- Add `nonlocal` for every variable reassigned inside the per-frame body (at minimum `frame_no`, `face_detections`, `last_landmark_timestamp_ms`, `last_recognition_log`, `last_speaker_id`, `last_speech_start` — check each against the existing body between lines 314-1240 and add `nonlocal` for any that are rebound with `=` rather than only mutated in place).
- The window-close check (`cv2.getWindowProperty(...)`) and `cv2.waitKey(...)` that currently sit at the bottom of the `while True:` loop move into `_write_frame`, since they must run once per displayed frame, after `imshow`. If the window-close check needs to break out of the loop, instead have it call a `stop()` mechanism: since `run_threaded_pipeline` has no external cancellation hook yet, have `_write_frame` raise a small internal `_WindowClosed(Exception)` when the user closes the window, and catch that specific exception right after `run_threaded_pipeline(...)` returns (it will surface via the normal exception-propagation path from Task 1), then treat it as a normal exit (do not re-raise).

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_realtime.py -v`
Expected: PASS (all tests, including the new one and the pre-existing VAD-unavailable test)

- [ ] **Step 5: Run the full existing test suite to check for regressions**

Run: `pytest tests/ -v`
Expected: PASS (no regressions)

- [ ] **Step 6: Commit**

```bash
git add realtime.py tests/test_realtime.py
git commit -m "refactor: run realtime camera loop through threaded reader/compute/writer stages with drop-oldest queues"
```

---

### Task 4: Manual verification

**Files:** None (manual, no code changes)

- [ ] **Step 1: Time the offline pipeline before/after**

Run: `python -c "import time, video_processor; t=time.time(); video_processor.process_video_pipeline('initialVideo/<sample>.mp4', 'trackedVideo/out.mp4', 'logs/out.json'); print(time.time()-t)"` on the same sample clip on the `main` branch and on this branch; note the wall-clock difference.

- [ ] **Step 2: Watch GPU utilization during the offline run**

Run `nvidia-smi -l 1` in a separate terminal while the Step 1 command runs on this branch; confirm GPU utilization is visibly higher and more sustained than the pre-change baseline (informal spot-check, no hard pass/fail threshold).

- [ ] **Step 3: Manually smoke-test the realtime camera path**

Run `python realtime.py` (or the existing launcher) with a real camera attached; confirm the preview window updates smoothly, recognition still occurs, and closing the window shuts down cleanly with no leftover threads or hung process.
