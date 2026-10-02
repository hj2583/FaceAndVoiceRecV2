# Unified Video Transcript and Face-Based Speaker Labels Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make processing a video produce both its recognized/annotated output and a transcript whose speaker labels are assigned from the same face-attribution events, while reporting Torch and ONNX face runtime devices separately.

**Architecture:** Keep face processing, VAD, Whisper, and diarization independent. A small orchestration module will run the existing face/video pipeline, read its timestamped speaker events, and pass them through the existing transcription `diarize` callback. Word/segment assignment will require the configured minimum event-overlap coverage; ambiguous or uncovered spans become `UNKNOWN`. MP3 remains audio-only. A separate runtime-status helper will report PyTorch and ONNX providers independently.

**Tech Stack:** Python 3.14, existing Streamlit/OpenCV/Whisper/Silero/SpeechBrain/ONNX Runtime stack, SQLite, pytest. No new dependencies.

## Global Constraints

- Do not rewrite face detection, tracking, ArcFace, VAD, Whisper, or voice-embedding internals. Add only event metadata, speaker-interval mapping, workflow orchestration, and backend-status reporting.
- Use timestamps for transcript-to-speaker alignment; never infer alignment from frame counts.
- Use `config.SPEAKER_FACE_OVERLAP_THRESHOLD` for minimum coverage; keep existing callers' transcription behavior unchanged by default.
- If face events do not cover a speech interval sufficiently, assign `UNKNOWN`; the combined video workflow must not silently fall back to voice-only diarization.
- Keep audio-only MP3 transcription working as it does today.
- Do not claim the face models use CUDA unless ONNX Runtime reports `CUDAExecutionProvider`; PyTorch and ONNX status are displayed separately.
- Preserve a successfully completed tracked-video output if the subsequent transcript stage fails.
- Do not delete or overwrite existing meeting transcripts as a side effect beyond the current same-video reprocessing behavior.
- Reference spec: `docs/superpowers/specs/2026-10-02-unified-video-transcription-design.md`.

---

### Task 1: Face-event interval adapter and overlap-gated assignment

**Files:**
- Modify: `transcription_core.py` (`_speaker_for_interval`, `_split_segment_by_words`, `transcribe_with_diarization`, `process_meeting_transcription`)
- Test: `tests/test_transcription_core.py`

**Interfaces:**
- Produces: `build_face_event_diarizer(speech_events) -> callable`; callback signature remains `(audio_path, speech_regions) -> iterable[(speaker_label, start_seconds, end_seconds)]`.
- Produces: optional `minimum_speaker_overlap: float = 0.0` parameter on `transcribe_with_diarization()` and `process_meeting_transcription()`. Existing callers retain current behavior with default `0.0`; the combined-video path passes `config.SPEAKER_FACE_OVERLAP_THRESHOLD`.
- Uses overlap coverage `overlap_duration / transcript_interval_duration`. If no speaker covers the configured fraction, use `"UNKNOWN"`.

- [ ] **Step 1: Write failing tests for coverage and gaps**

Add `build_face_event_diarizer` to the existing imports at the top of `tests/test_transcription_core.py`, then add these tests:

```python
from transcription_core import build_face_event_diarizer


def test_face_event_diarizer_fills_uncovered_speech_with_unknown():
    diarize = build_face_event_diarizer([
        {"speaker": "Alice", "start_time": 1.0, "end_time": 2.0},
    ])

    assert diarize(Path("audio.wav"), [(0.0, 3.0)]) == [
        ("UNKNOWN", 0.0, 1.0),
        ("Alice", 1.0, 2.0),
        ("UNKNOWN", 2.0, 3.0),
    ]


def test_speaker_interval_requires_minimum_overlap_coverage():
    from transcription_core import _speaker_for_interval

    events = [("Alice", 0.0, 0.6), ("Bob", 0.6, 1.0)]

    assert _speaker_for_interval(0.0, 1.0, events, 0.8) == "UNKNOWN"
    assert _speaker_for_interval(0.0, 0.7, events, 0.8) == "Alice"
```

- [ ] **Step 2: Run the focused tests and verify they fail**

Run: `.\directmlvenv\Scripts\python.exe -m pytest tests/test_transcription_core.py::test_face_event_diarizer_fills_uncovered_speech_with_unknown tests/test_transcription_core.py::test_speaker_interval_requires_minimum_overlap_coverage -q`

Expected: collection fails because `build_face_event_diarizer` is not defined and `_speaker_for_interval` has no coverage parameter.

- [ ] **Step 3: Add the timestamped event adapter**

In `transcription_core.py`, add this helper. It clips face events to each VAD region, inserts `UNKNOWN` spans for uncovered parts, and returns the callback shape already consumed by `transcribe_with_diarization`:

```python
def build_face_event_diarizer(speech_events):
    events = sorted(
        [
            (
                str(event.get("speaker") or event.get("person_name") or "UNKNOWN"),
                float(event["start_time"]),
                float(event["end_time"]),
            )
            for event in speech_events
            if float(event.get("end_time", 0.0)) > float(event.get("start_time", 0.0))
        ],
        key=lambda event: event[1],
    )

    def diarize(_audio_path, speech_regions):
        if speech_regions is None:
            return events
        intervals = []
        for region_start, region_end in speech_regions:
            cursor = float(region_start)
            region_end = float(region_end)
            for speaker, event_start, event_end in events:
                start = max(cursor, float(region_start), event_start)
                end = min(region_end, event_end)
                if end <= start:
                    continue
                if start > cursor:
                    intervals.append(("UNKNOWN", cursor, start))
                intervals.append((speaker, start, end))
                cursor = end
            if cursor < region_end:
                intervals.append(("UNKNOWN", cursor, region_end))
        return intervals

    return diarize
```

Extend `_speaker_for_interval(start_seconds, end_seconds, diarization, minimum_overlap=0.0)`. For the existing default `0.0`, preserve current behavior, including its existing `"Unknown Speaker"` fallback when no interval overlaps. For a positive threshold, return `"UNKNOWN"` when `best_overlap / (end_seconds - start_seconds) < minimum_overlap`. Pass this parameter through `_split_segment_by_words()` and `transcribe_with_diarization()` so timed words receive the same coverage rule. This keeps voice-only callers backward-compatible while the combined face-attribution path uses uppercase `UNKNOWN` below its configured coverage threshold.

Add `minimum_speaker_overlap: float = 0.0` to `transcribe_with_diarization()` and `process_meeting_transcription()`. Forward it from the latter into the former. Existing call sites omit it and retain previous behavior. The video workflow in Task 3 passes `config.SPEAKER_FACE_OVERLAP_THRESHOLD`.

- [ ] **Step 4: Run the focused tests and verify they pass**

Run: `.\directmlvenv\Scripts\python.exe -m pytest tests/test_transcription_core.py -q`

Expected: all transcription tests pass, including the two new adapter/coverage tests.

- [ ] **Step 5: Commit**

```powershell
git add transcription_core.py tests/test_transcription_core.py
git commit -m "feat(transcription): map face speaker events onto Whisper intervals"
```

---

### Task 2: Include identity metadata in video speaker events

**Files:**
- Modify: `video_processor.py` (`_log_speaker_events` near the `active_speech_logs.append` block)
- Test: `tests/test_video_processor.py`

**Interfaces:**
- Produces JSON event keys `person_id` and `track_id` beside `person_name`, `start_time`, `end_time`, `confidence`, and `source`. Consumed by Task 3.

- [ ] **Step 1: Add assertions to the existing unknown-event integration test**

In `test_process_video_pipeline_logs_unknown_speaker_events`, after the existing `speech = json.loads(...)["speech"]` statement, assert the one event contains `person_id is None`, its original `track_id`, and `speaker == "UNKNOWN"`. Retain the existing `person_name == "UNKNOWN"` assertion for old log consumers.

- [ ] **Step 2: Run the focused test and verify it fails**

Run: `.\directmlvenv\Scripts\python.exe -m pytest tests/test_video_processor.py::test_process_video_pipeline_logs_unknown_speaker_events -q`

Expected: fail because the serialized speech event does not contain `speaker`, `person_id`, or `track_id` yet.

- [ ] **Step 3: Add the metadata to each JSON event**

In `video_processor.py`, extend the existing event dictionary with:

```python
"person_id": event["person_id"],
"track_id": event["track_id"],
"speaker": event["speaker"],
```

Set `event["speaker"]` to the same label as `event["person_name"]`. Keep `person_name` for existing JSON consumers. Do not change detection, tracking, or the `audio_logs` DB write.

- [ ] **Step 4: Run the focused test**

Run: `.\directmlvenv\Scripts\python.exe -m pytest tests/test_video_processor.py -q`

Expected: all tests in the file pass.

- [ ] **Step 5: Commit**

```powershell
git add video_processor.py tests/test_video_processor.py
git commit -m "feat(video): include identity in speaker event output"
```

---

### Task 3: Orchestrate face processing and face-attributed transcription

**Files:**
- Create: `video_workflow.py`
- Create: `tests/test_video_workflow.py`

**Interfaces:**
- Produces `process_video_and_transcribe(video_path, output_path, log_path, progress_callback=None) -> int`, returning the meeting ID.
- Produces `TranscriptStageError`, which carries `video_output_path` and the original error when video processing succeeded but transcript generation failed. Used by Task 4.

- [ ] **Step 1: Write a failing workflow test**

Add `tests/test_video_workflow.py`:

```python
import json

import config
import video_workflow


def test_workflow_passes_face_events_to_transcription(tmp_path, monkeypatch):
    video_path = tmp_path / "meeting.mp4"
    video_path.write_bytes(b"video")
    output_path = tmp_path / "tracked.mp4"
    log_path = tmp_path / "video.json"
    expected_events = [{
        "speaker": "Alice",
        "person_id": 7,
        "track_id": 3,
        "start_time": 1.0,
        "end_time": 2.0,
        "confidence": 0.9,
        "source": "video",
    }]

    def fake_process(*_args, **_kwargs):
        log_path.write_text(json.dumps({"speech": expected_events}), encoding="utf-8")
        return True

    calls = []

    def fake_transcribe(path, diarize=None, minimum_speaker_overlap=0.0):
        calls.append((path, diarize, minimum_speaker_overlap))
        return 14

    monkeypatch.setattr(video_workflow, "process_video_pipeline", fake_process)
    monkeypatch.setattr(video_workflow, "process_meeting_transcription", fake_transcribe)

    meeting_id = video_workflow.process_video_and_transcribe(video_path, output_path, log_path)

    assert meeting_id == 14
    assert calls[0][0] == video_path
    assert calls[0][2] == config.SPEAKER_FACE_OVERLAP_THRESHOLD
    assert calls[0][1](tmp_path / "audio.wav", [(0.0, 3.0)]) == [
        ("UNKNOWN", 0.0, 1.0),
        ("Alice", 1.0, 2.0),
        ("UNKNOWN", 2.0, 3.0),
    ]


def test_transcript_stage_failure_preserves_video_output(tmp_path, monkeypatch):
    video_path = tmp_path / "meeting.mp4"
    video_path.write_bytes(b"video")
    output_path = tmp_path / "tracked.mp4"
    log_path = tmp_path / "video.json"

    def fake_process(*_args, **_kwargs):
        output_path.write_bytes(b"tracked video")
        log_path.write_text(json.dumps({"speech": []}), encoding="utf-8")
        return True

    monkeypatch.setattr(video_workflow, "process_video_pipeline", fake_process)
    monkeypatch.setattr(
        video_workflow,
        "process_meeting_transcription",
        lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("Whisper failed")),
    )

    try:
        video_workflow.process_video_and_transcribe(video_path, output_path, log_path)
    except video_workflow.TranscriptStageError as error:
        assert error.video_output_path == output_path
        assert "Whisper failed" in str(error)
    else:
        raise AssertionError("expected TranscriptStageError")

    assert output_path.read_bytes() == b"tracked video"
```

- [ ] **Step 2: Run the tests and verify they fail**

Run: `.\directmlvenv\Scripts\python.exe -m pytest tests/test_video_workflow.py -q`

Expected: import failure because `video_workflow.py` is not yet present.

- [ ] **Step 3: Implement the workflow**

Create `video_workflow.py`:

```python
import json
from pathlib import Path

import config
from transcription_core import build_face_event_diarizer, process_meeting_transcription
from video_processor import process_video_pipeline


class TranscriptStageError(RuntimeError):
    def __init__(self, video_output_path, cause):
        self.video_output_path = Path(video_output_path)
        self.cause = cause
        super().__init__(str(cause))


def process_video_and_transcribe(video_path, output_path, log_path, progress_callback=None):
    process_video_pipeline(video_path, output_path, log_path, progress_callback=progress_callback)
    try:
        events = json.loads(Path(log_path).read_text(encoding="utf-8")).get("speech", [])
        diarize = build_face_event_diarizer(events)
        return process_meeting_transcription(
            video_path,
            diarize=diarize,
            minimum_speaker_overlap=config.SPEAKER_FACE_OVERLAP_THRESHOLD,
        )
    except Exception as error:
        raise TranscriptStageError(output_path, error) from error
```

- [ ] **Step 4: Run the focused tests**

Run: `.\directmlvenv\Scripts\python.exe -m pytest tests/test_video_workflow.py -q`

Expected: both workflow and partial-failure tests pass.

- [ ] **Step 5: Commit**

```powershell
git add video_workflow.py tests/test_video_workflow.py
git commit -m "feat(video): orchestrate face processing with aligned transcript"
```

---

### Task 4: Make the video UI action produce the transcript

**Files:**
- Modify: `app.py` (`render_video`)

**Interfaces:**
- Consumes `video_workflow.process_video_and_transcribe()` and `TranscriptStageError` from Task 3.
- Video gets one `Process Video + Transcript` action. MP3 keeps `Recognize Speakers & Transcribe Audio`.

- [ ] **Step 1: Update the video action**

Import `process_video_and_transcribe` and `TranscriptStageError` from `video_workflow`. Replace the video `Process Video` callback with a call to `process_video_and_transcribe`; on full success show the meeting ID and tracked-video output path. Catch `TranscriptStageError` separately and show that the tracked video exists at `error.video_output_path` while transcription failed. Leave the MP3 branch unchanged. Remove the video-only `Transcribe Selected Video` button and its separate `process_meeting_transcription` call.

- [ ] **Step 2: Compile and run focused tests**

Run: `.\directmlvenv\Scripts\python.exe -m py_compile app.py video_workflow.py`

Run: `.\directmlvenv\Scripts\python.exe -m pytest tests/test_video_workflow.py tests/test_transcription_core.py -q`

Expected: compile succeeds and focused tests pass.

- [ ] **Step 3: Commit**

```powershell
git add app.py
git commit -m "feat(app): combine video processing and face-attributed transcript"
```

---

### Task 5: Report Torch and ONNX face runtime separately

**Files:**
- Create: `runtime_status.py`
- Create: `tests/test_runtime_status.py`
- Modify: `app.py` (`get_runtime_mode` and `main` runtime banner)

**Interfaces:**
- Produces `get_runtime_status() -> dict[str, str]` with exactly `{"torch": ..., "face": ...}`.
- Torch reports `CUDA (<device name>)` only when PyTorch CUDA is available; otherwise `CPU`.
- Face reports `CUDA (ONNX Runtime)` only when `CUDAExecutionProvider` is active; reports `CPU (CUDA provider unavailable)` for `CPUExecutionProvider`; reports `Unavailable` if provider detection raises.

- [ ] **Step 1: Write failing runtime-status tests**

Add tests for the mixed state seen in this machine (`torch` CUDA, face CPU), both CPU, both CUDA, and ONNX provider failure. Use a helper signature accepting injected fake modules/providers so no GPU hardware is required by unit tests:

```python
from types import SimpleNamespace

from runtime_status import get_runtime_status


def _fake_torch(available, name="Test GPU"):
    return SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda: available,
            get_device_name=lambda _index: name,
        )
    )


def test_reports_torch_gpu_and_face_cpu_as_mixed_runtime():
    status = get_runtime_status(
        torch_module=_fake_torch(True, "RTX Test"),
        face_provider_getter=lambda: ["CPUExecutionProvider"],
    )
    assert status == {
        "torch": "CUDA (RTX Test)",
        "face": "CPU (CUDA provider unavailable)",
    }


def test_reports_both_cpu():
    status = get_runtime_status(
        torch_module=_fake_torch(False),
        face_provider_getter=lambda: ["CPUExecutionProvider"],
    )
    assert status == {"torch": "CPU", "face": "CPU (CUDA provider unavailable)"}


def test_reports_cuda_face_provider():
    status = get_runtime_status(
        torch_module=_fake_torch(True),
        face_provider_getter=lambda: ["CUDAExecutionProvider"],
    )
    assert status["face"] == "CUDA (ONNX Runtime)"


def test_reports_face_runtime_unavailable_when_provider_detection_fails():
    def fail():
        raise RuntimeError("provider load failed")

    status = get_runtime_status(torch_module=_fake_torch(False), face_provider_getter=fail)
    assert status["face"] == "Unavailable"
```

- [ ] **Step 2: Run tests and verify they fail**

Run: `.\directmlvenv\Scripts\python.exe -m pytest tests/test_runtime_status.py -q`

Expected: import failure because `runtime_status.py` does not exist.

- [ ] **Step 3: Implement runtime detection**

Create `runtime_status.py` with `get_runtime_status(torch_module=None, face_provider_getter=None)`. Import Torch lazily if no module is injected; obtain the face provider through `face_backend.get_cuda_providers` if no getter is injected. Catch each backend independently so failure to import one does not hide the other backend's status. Do not report GPU utilization; this helper reports provider/device availability only.

Replace `app.get_runtime_mode()` with the new status helper and render two separate lines in `main()`:

```python
runtime_status = get_runtime_status()
st.caption(f"Audio models (PyTorch): {runtime_status['torch']}")
st.caption(f"Face models (ONNX Runtime): {runtime_status['face']}")
```

- [ ] **Step 4: Run tests**

Run: `.\directmlvenv\Scripts\python.exe -m pytest tests/test_runtime_status.py tests/test_face_backend.py -q`

Expected: all tests pass and existing provider-selection behavior remains intact.

- [ ] **Step 5: Commit**

```powershell
git add runtime_status.py tests/test_runtime_status.py app.py
git commit -m "fix(app): report audio and face inference backends separately"
```

---

### Task 6: Full regression and manual sample validation

**Files:**
- No additional production files unless a specific test exposes a regression.

- [ ] **Step 1: Compile changed Python files**

Run: `.\directmlvenv\Scripts\python.exe -m py_compile app.py video_workflow.py runtime_status.py transcription_core.py video_processor.py`

- [ ] **Step 2: Run the full suite**

Run: `.\directmlvenv\Scripts\python.exe -m pytest tests -q`

Expected: zero failures. Record the exact count and output; do not report accuracy improvements from mocked tests.

- [ ] **Step 3: Verify an existing multi-person clip manually**

In Streamlit, select a video with visible faces and run `Process Video + Transcript`. Confirm both the tracked video and meeting transcript are produced. Compare `logs/<video>.json` speaker events to the transcript rows around a speaker change. Confirm uncovered transcript spans are `UNKNOWN` and `person_id` is populated for recognized names. This is a functional smoke check only; it is not WER or speaker-attribution accuracy measurement.

- [ ] **Step 4: Verify current runtime reporting**

Launch the app and confirm Torch and ONNX statuses are independent. In the current environment expected display is Torch CUDA (RTX 5000 Ada) and face CPU because `CUDAExecutionProvider` is not currently available. If ONNX provider availability changes, report the observed value instead.

- GPU provider installation/remediation is explicitly outside this plan; if runtime status still reports face CPU, retain the honest status and create a separate dependency/DLL diagnosis task.