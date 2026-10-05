# Burned-In Subtitles and Source Audio Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Export tracked videos with the original audio and short, speaker-labeled Whisper subtitles burned into the picture, while leaving audio-only MP3 transcription unchanged.

**Architecture:** Keep face/video processing and Whisper independent. Capture speaker-attributed word times from the existing Whisper pass, group them into short ASS subtitle cues, then FFmpeg-burn those captions into the annotated video while mapping audio from the original source. On transcript/subtitle failure, preserve an annotated output with original audio if mux succeeds.

**Tech Stack:** Python 3.14, existing Whisper, FFmpeg/libass, Streamlit, pytest. No new packages.

## Global Constraints

- Do not rewrite face detection, tracking, ArcFace, VAD, Whisper, or `SpeakerAttributor`.
- Use the existing Whisper result and face-event intervals; do not run a second ASR pass.
- Keep transcript API return values and MP3 behavior backward-compatible when the new subtitle-word callback is omitted.
- Subtitle labels come from face-attributed word speakers: recognized name or literal `UNKNOWN`.
- Keep the original source audio stream and timestamps; never map audio from the OpenCV-generated silent intermediate.
- Prefer audio stream-copy; retry AAC only if the source codec cannot be muxed into MP4.
- Keep temporary files out of tracked output folders and clean them after success or failure.
- Do not stage or overwrite unrelated existing changes in `app.py`, `audio_log_utils.py`, or `tests/test_audio_log_utils.py` (the Arrow-compatible Track-column fix). For the Task 5 `app.py` commit, stage only its `render_video` hunk.
- Reference spec: `docs/superpowers/specs/2026-10-05-tracked-video-subtitles-design.md`.

---

### Task 1: Subtitle cue data and ASS serialization

**Files:** Create `subtitle_renderer.py`, create `tests/test_subtitle_renderer.py`, modify `config.py`.

**Interfaces:**
- `SubtitleWord(speaker_label: str, start_ms: int, end_ms: int, text: str)`
- `SubtitleCue(speaker_label: str, start_ms: int, end_ms: int, lines: tuple[str, ...])`
- `build_subtitle_cues(words, max_gap_ms=None, max_chars_per_line=None, max_lines=None) -> list[SubtitleCue]`
- `write_ass_subtitles(cues, output_path) -> Path`

- [ ] **Step 1: Write cue-builder and ASS tests**

Add to `tests/test_subtitle_renderer.py`:

```python
from subtitle_renderer import SubtitleCue, SubtitleWord, build_subtitle_cues, write_ass_subtitles


def test_cues_split_on_speaker_change_and_long_gap():
    words = [
        SubtitleWord("Alice", 0, 250, "Hello"),
        SubtitleWord("Alice", 260, 500, "there."),
        SubtitleWord("Bob", 600, 850, "Hi."),
        SubtitleWord("Bob", 2000, 2200, "Later."),
        SubtitleWord("Alice", 2300, 2300, "invalid"),
    ]
    cues = build_subtitle_cues(words, max_gap_ms=800, max_chars_per_line=42, max_lines=2)
    assert [(c.speaker_label, c.start_ms, c.end_ms) for c in cues] == [
        ("Alice", 0, 500), ("Bob", 600, 850), ("Bob", 2000, 2200),
    ]


def test_cue_wraps_words_into_two_lines_with_speaker_prefix():
    words = [
        SubtitleWord("Alice", i * 250, i * 250 + 200, word)
        for i, word in enumerate(["We", "can", "begin", "the", "review"])
    ]
    cue, = build_subtitle_cues(words, max_gap_ms=800, max_chars_per_line=18, max_lines=2)
    assert cue.lines == ("Alice: We can", "begin the review")
    assert all(len(line) <= 18 for line in cue.lines)


def test_ass_writer_formats_unknown_caption_times(tmp_path):
    path = tmp_path / "subtitles.ass"
    write_ass_subtitles([
        SubtitleCue("UNKNOWN", 100, 1230, ("UNKNOWN: Hello there.",)),
    ], path)

    contents = path.read_text(encoding="utf-8")
    assert "[Script Info]" in contents and "[Events]" in contents
    assert "0:00:00.10,0:00:01.23" in contents
    assert "UNKNOWN: Hello there." in contents
```

- [ ] **Step 2: Verify RED**

Run: `.\directmlvenv\Scripts\python.exe -m pytest tests/test_subtitle_renderer.py -q`

Expected: collection fails because `subtitle_renderer` does not exist.

- [ ] **Step 3: Implement cue builder and ASS writer**

Add to `config.py`:

```python
SUBTITLE_MAX_GAP_MS = 800
SUBTITLE_MAX_CHARS_PER_LINE = 42
SUBTITLE_MAX_LINES = 2
```

`build_subtitle_cues` sorts by start time; discards empty text and `end_ms <= start_ms`; groups only same-speaker words whose gap is within `max_gap_ms`; and starts a new cue before a word would exceed the configured line count. Prefix line 1 with `speaker_label + ": "`. Wrap only at word boundaries; never split long words. Cue times are the first word start and last word end. A single unusually long word may exceed the character target but must not create a third line.

`write_ass_subtitles` writes a UTF-8 ASS file with a two-line-safe, bottom-aligned white-on-dark outlined style, one Dialogue row per cue, and millisecond times formatted `H:MM:SS.cc`. Escape ASS control characters in speaker/text values so transcript text cannot inject formatting. `UNKNOWN` stays literal.

- [ ] **Step 4: Verify GREEN**

Run: `.\directmlvenv\Scripts\python.exe -m pytest tests/test_subtitle_renderer.py -q`

Expected: all cue and ASS tests pass.

- [ ] **Step 5: Commit**

```powershell
git add subtitle_renderer.py tests/test_subtitle_renderer.py config.py
git commit -m "feat(subtitles): build speaker-labeled short caption cues"
```

---

### Task 2: Capture speaker-attributed Whisper word timings

**Files:** Modify `transcription_core.py`, test `tests/test_transcription_core.py`.

**Interface:** Add optional `subtitle_word_callback: Optional[Callable[[SubtitleWord], None]] = None` to `transcribe_with_diarization()` and `process_meeting_transcription()`. Existing return values, database turns, and audio-only callers remain unchanged when it is omitted.

- [ ] **Step 1: Write failing tests**

Add this first test in `tests/test_transcription_core.py`; import `SubtitleWord` from `subtitle_renderer` and use the existing pytest `tmp_path`/`monkeypatch` pattern:

```python
def test_subtitle_word_callback_receives_face_attributed_word_timestamps(tmp_path, monkeypatch):
    from transcription_core import SubtitleWord, transcribe_with_diarization

    wav_path = tmp_path / "audio.wav"
    wav_path.write_bytes(b"fake wav")
    monkeypatch.setattr("transcription_core.detect_speech_segments", lambda *_a: [(0.0, 1.0)])

    class FakeModel:
        def transcribe(self, _path, **_kwargs):
            return {"segments": [{
                "start": 0.0, "end": 1.0, "text": " hello there", "avg_logprob": -0.1,
                "words": [
                    {"word": " hello", "start": 0.1, "end": 0.4},
                    {"word": " there", "start": 0.5, "end": 0.8},
                ],
            }]}

    fake_whisper = type("FakeWhisperModule", (), {
        "load_model": staticmethod(lambda *_a, **_k: FakeModel())
    })
    monkeypatch.setitem(__import__("sys").modules, "whisper", fake_whisper)
    observed = []
    segments = transcribe_with_diarization(
        wav_path,
        diarize=lambda *_a: [("Alice", 0.0, 0.45), ("Bob", 0.45, 1.0)],
        minimum_speaker_overlap=0.8,
        subtitle_word_callback=observed.append,
    )

    assert observed == [
        SubtitleWord("Alice", 100, 400, "hello"),
        SubtitleWord("Bob", 500, 800, "there"),
    ]
    assert [(segment.speaker_label, segment.text) for segment in segments] == [
        ("Alice", "hello"), ("Bob", "there"),
    ]
```

Add this missing-timing fallback test too:

```python
def test_subtitle_callback_uses_whole_segment_when_word_timings_are_missing(tmp_path, monkeypatch):
    from transcription_core import SubtitleWord, transcribe_with_diarization

    wav_path = tmp_path / "audio.wav"
    wav_path.write_bytes(b"fake wav")
    monkeypatch.setattr("transcription_core.detect_speech_segments", lambda *_a: [(0.0, 1.0)])

    class FakeModel:
        def transcribe(self, _path, **_kwargs):
            return {"segments": [{
                "start": 0.0, "end": 1.0, "text": " hello world", "avg_logprob": -0.1,
            }]}

    fake_whisper = type("FakeWhisperModule", (), {
        "load_model": staticmethod(lambda *_a, **_k: FakeModel())
    })
    monkeypatch.setitem(__import__("sys").modules, "whisper", fake_whisper)
    observed = []

    transcribe_with_diarization(
        wav_path,
        diarize=lambda *_a: [("Alice", 0.0, 1.0)],
        subtitle_word_callback=observed.append,
    )

    assert observed == [SubtitleWord("Alice", 0, 1000, "hello world")]
```

- [ ] **Step 2: Verify RED**

Run: `.\directmlvenv\Scripts\python.exe -m pytest tests/test_transcription_core.py -k subtitle_word_callback -q`

Expected: fails because the callback argument is not accepted or no records are emitted.

- [ ] **Step 3: Thread the optional callback through one ASR pass**

Import `SubtitleWord` from `subtitle_renderer`. Add the callback to both functions and forward it from `process_meeting_transcription()` to `transcribe_with_diarization()`. When `_timed_words_if_reliable(raw)` succeeds, invoke the callback once per valid timed word using `_speaker_for_interval(word_start, word_end, diarization, minimum_speaker_overlap)`. When timings are missing/unreliable, invoke once for the full raw segment interval and its segment-level speaker. Do not change the returned `TranscriptionSegment` list or `save_transcripts()` schema. MP3 uses no callback by default.

- [ ] **Step 4: Verify transcription tests**

Run: `.\directmlvenv\Scripts\python.exe -m pytest tests/test_transcription_core.py -q`

Expected: existing tests and new callback tests pass.

- [ ] **Step 5: Commit**

```powershell
git add transcription_core.py tests/test_transcription_core.py
git commit -m "feat(transcription): expose attributed word timings for subtitles"
```

---

### Task 3: FFmpeg subtitle burn-in and source-audio mux

**Files:** Modify `subtitle_renderer.py`, test `tests/test_subtitle_renderer.py`.

**Interface:** `mux_tracked_video(annotated_video, source_video, output_path, subtitle_path=None, ffmpeg=None) -> Path`; raise `VideoMuxError` if safe finalization fails.

- [ ] **Step 1: Write failing mux tests**

Use `monkeypatch` on `subtitle_renderer.subprocess.run`. Have the fake runner record `args`/`cwd`, return code 1 on the first invocation (audio copy unsupported), then create `Path(args[-1])` and return code 0 on the AAC invocation. Assert:
- Inputs are annotated video first, source video second.
- Video map is `0:v:0`; audio map is optional `1:a:0?`.
- Subtitle run uses `-vf` with `subtitles=filename=subtitles.ass`, with `cwd=subtitle_path.parent` and no shell.
- First audio codec is `copy`; retry codec is `aac`.
- Final output is atomically created only after success.

Add a second fake-runner test where both runs fail: assert `VideoMuxError` and that a pre-existing output file remains unchanged. Add a no-subtitle test: `subtitle_path=None` omits `-vf` but still maps optional source audio.

- [ ] **Step 2: Verify RED**

Run: `.\directmlvenv\Scripts\python.exe -m pytest tests/test_subtitle_renderer.py -k mux -q`

Expected: `mux_tracked_video`/`VideoMuxError` do not exist.

- [ ] **Step 3: Implement muxing**

Use `shutil.which("ffmpeg")` when `ffmpeg` is not injected. Create a unique temporary sibling output. Build an argument list (never `shell=True`) with input 0 = annotated intermediate, input 1 = original, `-map 0:v:0`, optional `-map 1:a:0?`, H.264 video, and `-c:a copy`. When subtitles are enabled, set `cwd` to the ASS directory and use its simple generated basename (`subtitles.ass`) in `-vf subtitles=filename=subtitles.ass`, avoiding Windows drive-letter escaping. If copy fails, retry with AAC. On successful FFmpeg exit and a created temp output, atomically replace the final path. On failure, delete only the temp output and raise `VideoMuxError` with stderr.

Build each command through one helper so the copy/AAC retry cannot drift:

```python
import os
import shutil
import subprocess
import uuid
from pathlib import Path


def unique_temp_sibling(output_path):
    output_path = Path(output_path)
    return output_path.with_name(
        f".{output_path.stem}.{uuid.uuid4().hex}.tmp{output_path.suffix}"
    )


def _mux_args(ffmpeg, annotated_video, source_video, temp_output, subtitle_path, audio_codec):
    args = [ffmpeg, "-y", "-i", str(annotated_video), "-i", str(source_video)]
    if subtitle_path is not None:
        args.extend(["-vf", f"subtitles=filename={Path(subtitle_path).name}"])
    args.extend([
        "-map", "0:v:0",
        "-map", "1:a:0?",
        "-c:v", "libx264",
        "-c:a", audio_codec,
    ])
    if audio_codec == "aac":
        args.extend(["-b:a", "192k"])
    args.extend(["-movflags", "+faststart", str(temp_output)])
    return args


def mux_tracked_video(annotated_video, source_video, output_path, subtitle_path=None, ffmpeg=None):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = ffmpeg or shutil.which("ffmpeg")
    if ffmpeg is None:
        raise VideoMuxError("FFmpeg was not found in PATH")
    codecs = ("copy", "aac")
    last_error = ""
    for codec in codecs:
        temp_output = unique_temp_sibling(output_path)
        result = subprocess.run(
            _mux_args(ffmpeg, annotated_video, source_video, temp_output, subtitle_path, codec),
            cwd=str(Path(subtitle_path).parent) if subtitle_path is not None else None,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0 and Path(temp_output).is_file():
            os.replace(temp_output, output_path)
            return output_path
        last_error = result.stderr
        Path(temp_output).unlink(missing_ok=True)
    raise VideoMuxError(last_error)
```

`unique_temp_sibling(output_path)` creates a collision-free temporary MP4 in `output_path.parent`. If `ffmpeg` is missing or an output attempt fails, raise `VideoMuxError`; the workflow handles the audio-only or silent-video fallback.

- [ ] **Step 4: Verify renderer tests**

Run: `.\directmlvenv\Scripts\python.exe -m pytest tests/test_subtitle_renderer.py -q`

Expected: cue generation, ASS output, audio mapping, codec retry, and failed-output preservation all pass.

- [ ] **Step 5: Commit**

```powershell
git add subtitle_renderer.py tests/test_subtitle_renderer.py
git commit -m "feat(video): burn subtitles and mux source audio"
```

---

### Task 4: Integrate cues and partial-success audio preservation

**Files:** Modify `video_workflow.py`, test `tests/test_video_workflow.py`.

**Interfaces:** Add `stage: str` to `TranscriptStageError`, preserving `.video_output_path` and `.cause`. Keep `process_video_and_transcribe()` positional arguments compatible; append optional `stage_callback: Callable[[str], None] = None`.

- [ ] **Step 1: Write failing workflow tests**

Add three tests to `tests/test_video_workflow.py`. Use the existing module imports and add `json`, `Path`, `config`, `video_workflow`, and `SubtitleWord` if not already imported. The success path test should follow this contract:

```python
def test_workflow_runs_one_transcription_then_burns_cues(tmp_path, monkeypatch):
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    final = tmp_path / "tracked.mp4"
    log = tmp_path / "events.json"
    calls = {"video": 0, "transcribe": 0, "mux": []}

    def fake_video(_source, annotated, log_path, **_kwargs):
        calls["video"] += 1
        Path(annotated).write_bytes(b"annotated")
        Path(log_path).write_text(json.dumps({"speech": [{
            "speaker": "Alice", "start_time": 0.0, "end_time": 1.0,
        }]}), encoding="utf-8")
        return True

    def fake_transcribe(_source, diarize, minimum_speaker_overlap, subtitle_word_callback, **_kwargs):
        calls["transcribe"] += 1
        assert minimum_speaker_overlap == config.SPEAKER_FACE_OVERLAP_THRESHOLD
        assert diarize(source, [(0.0, 1.0)]) == [("Alice", 0.0, 1.0)]
        subtitle_word_callback(SubtitleWord("Alice", 100, 400, "Hello."))
        return 7

    def fake_write(cues, path):
        assert cues[0].lines == ("Alice: Hello.",)
        Path(path).write_text("ASS", encoding="utf-8")
        return Path(path)

    def fake_mux(annotated, original, output, subtitle_path=None, **_kwargs):
        calls["mux"].append((Path(annotated), original, Path(output), subtitle_path))
        Path(output).write_bytes(b"final")
        return Path(output)

    monkeypatch.setattr(video_workflow, "process_video_pipeline", fake_video)
    monkeypatch.setattr(video_workflow, "process_meeting_transcription", fake_transcribe)
    monkeypatch.setattr(video_workflow, "write_ass_subtitles", fake_write)
    monkeypatch.setattr(video_workflow, "mux_tracked_video", fake_mux)

    assert video_workflow.process_video_and_transcribe(source, final, log) == 7
    assert calls["video"] == calls["transcribe"] == len(calls["mux"]) == 1
    assert calls["mux"][0][1:3] == (source, final)
    assert calls["mux"][0][3] is not None
    assert final.read_bytes() == b"final"
```

Also add one transcription-failure test: fake the video stage to create its annotated output/log, make transcription raise, make the no-subtitle mux write an audio-bearing output, then assert `TranscriptStageError.stage == "transcription"` and the final file exists. Add one subtitle-failure test: transcription emits a word, ASS writing succeeds, the subtitle mux raises `VideoMuxError`, the `subtitle_path=None` retry writes an audio-bearing output, and `TranscriptStageError.stage == "subtitle"` is raised. All three tests must assert only one transcription call. Keep the existing MP3 test unchanged.

- [ ] **Step 2: Verify RED**

Run: `.\directmlvenv\Scripts\python.exe -m pytest tests/test_video_workflow.py -q`

Expected: callback capture, subtitle mux, staged errors, and/or fallback expectations fail against current workflow.

- [ ] **Step 3: Implement workflow finalization**

Use a temporary directory adjacent to `output_path` for `annotated.mp4` and `subtitles.ass`. Run face processing to the annotated path, load speaker events, create the face-event diarizer, then call `process_meeting_transcription()` once with the overlap threshold and `subtitle_words.append` callback. Build cues, write ASS, and call `mux_tracked_video(annotated, source, final, ass_path)`.

Report stages through `stage_callback`: `Processing faces and speaker events`, `Transcribing and timing subtitles`, `Burning subtitles and restoring audio`.

If transcription or cue generation fails, call `mux_tracked_video(annotated, source, final, subtitle_path=None)` before raising a transcription-stage error. If subtitle mux fails, attempt the same audio-only mux before raising a subtitle-stage error. If audio-only mux also fails, copy the annotated silent video to `final` and report `stage="audio_mux"` with both failure causes. Clean all temporary files in `finally`; do not delete a successfully produced final output before replacement succeeds.

- [ ] **Step 4: Verify workflow, transcription, and mux tests**

Run: `.\directmlvenv\Scripts\python.exe -m pytest tests/test_video_workflow.py tests/test_transcription_core.py tests/test_subtitle_renderer.py -q`

Expected: all focused tests pass.

- [ ] **Step 5: Commit**

```powershell
git add video_workflow.py tests/test_video_workflow.py
git commit -m "feat(video): mux speaker subtitles and original audio"
```

---

### Task 5: Show stage status in Streamlit
**Files:** Modify `app.py` (`render_video`).

Use this callback shape in `render_video`:

```python
def update_stage(message):
    status.write(message)

meeting_id = process_video_and_transcribe(
    input_path,
    output_path,
    log_path,
    progress_callback=update_progress,
    stage_callback=update_stage,
)
```

**Worktree constraint:** `app.py` currently has an unrelated uncommitted Arrow fix; keep its changes out of this commit. Stage only the new `render_video` hunk with `git add -p app.py`. Do not stage `audio_log_utils.py` or `tests/test_audio_log_utils.py`.

- [ ] **Step 1: Update the existing combined action**

Keep the `Process Video + Transcript` button and MP3 branch. Pass `stage_callback` to update the existing status area and retain the face-processing progress callback. On success show the meeting ID and final path, noting the MP4 has original audio and burned-in subtitles. In the `TranscriptStageError` handler include `exc.stage`, `exc.video_output_path`, and the exception message. Do not change the unrelated Arrow helper import, `Track` nullable dtype, or audio-log tables; leave the MP3 transcription call unchanged.

- [ ] **Step 2: Compile and run focused tests**

Run: `.\directmlvenv\Scripts\python.exe -m py_compile app.py video_workflow.py subtitle_renderer.py`

Run: `.\directmlvenv\Scripts\python.exe -m pytest tests/test_video_workflow.py tests/test_subtitle_renderer.py -q`

Expected: compile and focused tests pass.

- [ ] **Step 3: Commit only the video-action hunk**

```powershell
git add -p app.py
# Select only the render_video subtitle/audio status hunk; reject Arrow Track/log hunks.
git diff --cached --stat
git diff --cached -- app.py
git commit -m "feat(app): show subtitle and audio finalization status"
```

---

### Task 6: Full regression and real-media smoke validation

- [ ] **Step 1: Compile and run the full suite**

Run: `.\directmlvenv\Scripts\python.exe -m py_compile app.py subtitle_renderer.py video_workflow.py transcription_core.py video_processor.py`

Run: `.\directmlvenv\Scripts\python.exe -m pytest tests -q`

Expected: zero failures; record the actual count.

- [ ] **Step 2: Run a bounded real-media smoke test**

Use an existing short speech clip and temporary DB/output/log folders. Verify final tracked MP4 exists, contains video and source audio streams, and has visible burned captions. Compare one cue’s start/end and speaker label to the saved face-event JSON. If the clip yields only `UNKNOWN`, state that; this is a functional smoke test, not a recognition-accuracy claim.

Use `ffprobe -v error -show_entries stream=codec_type -of csv=p=0 <tracked.mp4>` to verify the final container has both `video` and `audio` streams for an audio-bearing source.

- [ ] **Step 3: Verify MP3 regression**

Run the existing audio-only MP3 action/test and confirm no video or subtitle mux is attempted.

- [ ] **Step 4: Record limitations**

Record the clip tested, input audio presence, output streams, caption visibility, and speaker labels. State that subtitle correctness and identity accuracy are not measured without labelled references.
