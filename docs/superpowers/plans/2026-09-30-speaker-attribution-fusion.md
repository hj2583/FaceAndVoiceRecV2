# Speaker Attribution / Fusion Layer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a shared `speaker_attribution.py` fusion module that scores every tracked face against synchronized voice-activity/lip-motion/face-confidence signals, smooths the decision over time, and produces `active_speaker`/`track_id`/`confidence`/`reason` results plus speaker events — replacing the duplicated naive "max lip_open" logic in `video_processor.py` and `realtime.py`.

**Architecture:** One `SpeakerAttributor` class ingests timestamped audio samples (`update_audio`) and per-frame face observations (`update_faces`), returning a structured attribution result and internally queuing closed speaker events (`pop_closed_events`) for `database.log_audio()`. Both the offline and realtime pipelines call the same class; no face/audio processing code changes.

**Tech Stack:** Python 3.14, stdlib only (`dataclasses`, `collections.deque`, `statistics.stdev`) — no new dependencies.

## Global Constraints

- Do not modify face detection, tracking, ArcFace recognition, or VAD/voice-embedding code. Only the speaker-selection block in each pipeline is replaced.
- All thresholds/weights live in `config.py`; no magic numbers in `speaker_attribution.py`.
- `track_id`, `person_id`, and the attribution result's speaker label stay distinct; nothing is written back onto `Track` face-recognition fields.
- Existing tests must keep passing: `tests/test_video_processor.py`, `tests/test_realtime.py`, `tests/test_realtime_runtime.py`, `tests/test_database_*`, except where this plan explicitly changes logging behavior (unknown-speaker events now logged instead of skipped).
- Reference spec: `docs/superpowers/specs/2026-09-30-speaker-attribution-fusion-design.md`.

---

### Task 1: Config parameters

**Files:**
- Modify: `config.py` (append to the "Active speaker settings" section, ~line 193)
- Test: `tests/test_config.py`

**Interfaces:**
- Produces: `config.SPEAKER_WINDOW_MS`, `config.SPEAKER_SWITCH_THRESHOLD`, `config.SPEAKER_MIN_CONFIDENCE`, `config.SPEAKER_GRACE_PERIOD_MS`, `config.AUDIO_SYNC_TOLERANCE_MS`, `config.VOICE_ACTIVITY_WEIGHT`, `config.LIP_MOTION_WEIGHT`, `config.FACE_CONFIDENCE_WEIGHT`, `config.TEMPORAL_WEIGHT`, `config.LIP_MOTION_NORM` — all consumed by Task 3.

- [ ] **Step 1: Write the failing test**

```python
def test_speaker_attribution_weights_are_configurable_and_positive():
    assert config.SPEAKER_WINDOW_MS > 0
    assert config.SPEAKER_SWITCH_THRESHOLD > 0
    assert 0.0 < config.SPEAKER_MIN_CONFIDENCE < 1.0
    assert config.SPEAKER_GRACE_PERIOD_MS > 0
    assert config.AUDIO_SYNC_TOLERANCE_MS > 0
    assert config.LIP_MOTION_NORM > 0
    weights = (
        config.VOICE_ACTIVITY_WEIGHT,
        config.LIP_MOTION_WEIGHT,
        config.FACE_CONFIDENCE_WEIGHT,
        config.TEMPORAL_WEIGHT,
    )
    assert all(w >= 0.0 for w in weights)
    assert abs(sum(weights) - 1.0) < 1e-6
```

Append this to `tests/test_config.py`.

- [ ] **Step 2: Run test to verify it fails**

Run: `.\directmlvenv\Scripts\python.exe -m pytest tests/test_config.py::test_speaker_attribution_weights_are_configurable_and_positive -v`
Expected: FAIL with `AttributeError: module 'config' has no attribute 'SPEAKER_WINDOW_MS'`

- [ ] **Step 3: Add the config parameters**

Append to `config.py` after the existing `SPEECH_CONFIRM_FRAMES`/`SPEECH_RELEASE_FRAMES` lines in the "Active speaker settings" section:

```python
# ============================================================
# Speaker attribution / fusion
# ============================================================

# Rolling window used both to smooth per-track scores (for switch decisions)
# and to compute mouth_motion_score from recent lip_open_ratio history.
SPEAKER_WINDOW_MS = 800

# A challenger must beat the incumbent's smoothed score by this margin
# before the active speaker changes. Does not apply to the very first
# assignment (no incumbent yet) or to the incumbent retaining its own track.
SPEAKER_SWITCH_THRESHOLD = 0.15

# Below this smoothed score, the result is "UNKNOWN" rather than a
# low-confidence guessed name.
SPEAKER_MIN_CONFIDENCE = 0.35

# How long to keep reporting the current speaker after voice activity
# drops, before falling back to active_speaker=None.
SPEAKER_GRACE_PERIOD_MS = 600

# Maximum gap allowed when matching a face observation's timestamp to the
# nearest audio sample. Beyond this, no audio data is treated as available.
AUDIO_SYNC_TOLERANCE_MS = 250

# Speaker-scoring weights; must sum to 1.0.
VOICE_ACTIVITY_WEIGHT = 0.40
LIP_MOTION_WEIGHT = 0.30
FACE_CONFIDENCE_WEIGHT = 0.20
TEMPORAL_WEIGHT = 0.10

# Standard deviation of a track's recent lip_open_ratio history is divided
# by this constant, then clamped to [0, 1], to produce mouth_motion_score.
LIP_MOTION_NORM = 0.02
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.\directmlvenv\Scripts\python.exe -m pytest tests/test_config.py -v`
Expected: PASS (all tests in the file)

- [ ] **Step 5: Commit**

```powershell
git add config.py tests/test_config.py
git commit -m "feat(config): add speaker-attribution fusion parameters"
```

---

### Task 2: Database — `track_id` on `audio_logs`

**Files:**
- Modify: `database.py:88-100` (schema), `database.py:732` (`log_audio`), `database.py:754` (`fetch_audio_logs`)
- Create: `tests/test_database_audio_logs.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `database.log_audio(start_time, end_time, person_id, person_name, confidence, source, transcript=None, track_id=None)`; `database.fetch_audio_logs(source=None)` rows now `(log_id, start_time, end_time, person_name, confidence, source, transcript, track_id)`. Consumed by Tasks 6, 7, 8.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_database_audio_logs.py
import database


def _setup_database(tmp_path, monkeypatch):
    monkeypatch.setattr(database, "DB_PATH", tmp_path / "faces.db")
    database.init_db()


def test_log_audio_stores_and_returns_track_id(tmp_path, monkeypatch):
    _setup_database(tmp_path, monkeypatch)
    person_id = database.create_person("Alice")

    database.log_audio(1.0, 2.5, person_id, "Alice", 0.9, "video", track_id=3)

    rows = database.fetch_audio_logs("video")
    assert len(rows) == 1
    _log_id, start, end, name, confidence, source, transcript, track_id = rows[0]
    assert (start, end, name, confidence, source, transcript, track_id) == (
        1.0, 2.5, "Alice", 0.9, "video", None, 3,
    )


def test_log_audio_defaults_track_id_to_none(tmp_path, monkeypatch):
    _setup_database(tmp_path, monkeypatch)

    database.log_audio(0.0, 1.0, None, "UNKNOWN", 0.4, "realtime")

    rows = database.fetch_audio_logs()
    assert rows[0][-1] is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.\directmlvenv\Scripts\python.exe -m pytest tests/test_database_audio_logs.py -v`
Expected: FAIL — `log_audio() got an unexpected keyword argument 'track_id'`

- [ ] **Step 3: Add the column and update the functions**

In `database.py`, add an additive migration next to the existing `sentence_type` migration pattern (immediately after the `audio_logs` table's `CREATE TABLE IF NOT EXISTS` block, before the `meetings` table):

```python
        audio_log_columns = {
            row[1] for row in conn.execute("PRAGMA table_info(audio_logs)")
        }
        if "track_id" not in audio_log_columns:
            conn.execute("ALTER TABLE audio_logs ADD COLUMN track_id INTEGER")
```

Update `log_audio`:

```python
def log_audio(
    start_time, end_time, person_id, person_name, confidence, source,
    transcript=None, track_id=None,
):
    with get_conn() as conn:
        conn.execute(
            """
            INSERT INTO audio_logs(
                start_time, end_time, person_id, person_name,
                confidence, source, transcript, track_id
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                float(start_time),
                float(end_time),
                person_id,
                person_name,
                float(confidence),
                source,
                transcript,
                track_id,
            ),
        )
```

Update `fetch_audio_logs`:

```python
def fetch_audio_logs(source=None):
    with get_conn() as conn:
        if source:
            return conn.execute(
                """
                SELECT log_id, start_time, end_time, person_name,
                       confidence, source, transcript, track_id
                FROM audio_logs
                WHERE source=?
                ORDER BY start_time DESC
                """,
                (source,),
            ).fetchall()

        return conn.execute(
            """
            SELECT log_id, start_time, end_time, person_name,
                   confidence, source, transcript, track_id
            FROM audio_logs
            ORDER BY start_time DESC
            """
        ).fetchall()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.\directmlvenv\Scripts\python.exe -m pytest tests/test_database_audio_logs.py -v`
Expected: PASS

- [ ] **Step 5: Run the full suite to check for regressions from the row-shape change**

Run: `.\directmlvenv\Scripts\python.exe -m pytest tests -q`
Expected: PASS. If `app.py`'s two `fetch_audio_logs()` unpacking sites (`_id, start, end, name, confidence, source, transcript = row`) raise a `ValueError: too many values to unpack`, that is expected and fixed in Task 8 — do not fix it here.

- [ ] **Step 6: Commit**

```powershell
git add database.py tests/test_database_audio_logs.py
git commit -m "feat(database): add track_id column to audio_logs"
```

---

### Task 3: `speaker_attribution.py` — data structures, audio sync, per-candidate scoring

**Files:**
- Create: `speaker_attribution.py`
- Create: `tests/test_speaker_attribution.py`

**Interfaces:**
- Produces: `FaceObservation` dataclass; `SpeakerAttributor.update_audio(timestamp, voice_active, voice_confidence=None)`; `SpeakerAttributor._score_candidates(timestamp, voice_active, voice_confidence, face_observations)` returning `list[dict]` with keys `observation`, `score`, `motion_score`. Consumed by Task 4.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_speaker_attribution.py
import config
from speaker_attribution import FaceObservation, SpeakerAttributor


def _observation(track_id, person_id=1, person_name="Alice", face_confidence=0.9, lip_open_ratio=0.05):
    return FaceObservation(
        track_id=track_id,
        person_id=person_id,
        person_name=person_name,
        face_confidence=face_confidence,
        mouth_open=lip_open_ratio >= config.LIP_OPEN_THRESHOLD,
        lip_open_ratio=lip_open_ratio,
        face_visible=True,
    )


def test_score_is_zero_when_voice_not_active():
    attributor = SpeakerAttributor()
    scored = attributor._score_candidates(1.0, False, 0.0, [_observation(1)])
    assert scored[0]["score"] == 0.0


def test_score_combines_voice_face_confidence_when_voice_active():
    attributor = SpeakerAttributor()
    scored = attributor._score_candidates(1.0, True, 1.0, [_observation(1, face_confidence=0.9)])
    expected = config.VOICE_ACTIVITY_WEIGHT * 1.0 + config.FACE_CONFIDENCE_WEIGHT * 0.9
    assert abs(scored[0]["score"] - expected) < 1e-6
    assert scored[0]["motion_score"] == 0.0  # fewer than 2 samples so far


def test_nearest_audio_sample_within_tolerance_is_used():
    attributor = SpeakerAttributor()
    attributor.update_audio(1.00, True, 0.8)
    scored = attributor._score_candidates_from_timestamp(1.05, [_observation(1, face_confidence=0.0)])
    expected = config.VOICE_ACTIVITY_WEIGHT * 0.8
    assert abs(scored[0]["score"] - expected) < 1e-6


def test_audio_sample_outside_tolerance_is_ignored():
    attributor = SpeakerAttributor()
    attributor.update_audio(1.00, True, 0.8)
    tolerance_s = config.AUDIO_SYNC_TOLERANCE_MS / 1000.0
    scored = attributor._score_candidates_from_timestamp(1.00 + tolerance_s + 0.05, [_observation(1)])
    assert scored[0]["score"] == 0.0


def test_mouth_motion_score_rises_with_lip_ratio_variance():
    attributor = SpeakerAttributor()
    attributor._score_candidates(1.00, True, 0.0, [_observation(1, lip_open_ratio=0.0)])
    scored = attributor._score_candidates(1.10, True, 0.0, [_observation(1, lip_open_ratio=0.05)])
    assert scored[0]["motion_score"] > 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.\directmlvenv\Scripts\python.exe -m pytest tests/test_speaker_attribution.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'speaker_attribution'`

- [ ] **Step 3: Implement the module**

```python
# speaker_attribution.py
"""Fusion layer combining face-tracking and voice-activity signals to
determine who is speaking. Contains no face detection or audio capture
code; it consumes normalized observations from the existing pipelines."""

from collections import deque
from dataclasses import dataclass, field
from statistics import stdev
from typing import Optional

import config


@dataclass(frozen=True)
class FaceObservation:
    track_id: int
    person_id: Optional[int]
    person_name: str
    face_confidence: float
    mouth_open: bool
    lip_open_ratio: float
    face_visible: bool


@dataclass
class _AudioSample:
    timestamp: float
    voice_active: bool
    voice_confidence: float


@dataclass
class _TrackHistory:
    lip_ratios: deque = field(default_factory=lambda: deque(maxlen=64))
    scores: deque = field(default_factory=lambda: deque(maxlen=64))


class SpeakerAttributor:
    """Stateful fusion engine; one instance per pipeline session (one per
    realtime.run() call, one per process_video_pipeline() call)."""

    def __init__(self, debug: bool = False):
        self.debug = debug

        self._window_s = config.SPEAKER_WINDOW_MS / 1000.0
        self._switch_threshold = config.SPEAKER_SWITCH_THRESHOLD
        self._min_confidence = config.SPEAKER_MIN_CONFIDENCE
        self._grace_period_s = config.SPEAKER_GRACE_PERIOD_MS / 1000.0
        self._sync_tolerance_s = config.AUDIO_SYNC_TOLERANCE_MS / 1000.0
        self._voice_weight = config.VOICE_ACTIVITY_WEIGHT
        self._lip_weight = config.LIP_MOTION_WEIGHT
        self._face_weight = config.FACE_CONFIDENCE_WEIGHT
        self._temporal_weight = config.TEMPORAL_WEIGHT
        self._lip_motion_norm = config.LIP_MOTION_NORM

        self._audio_samples: deque = deque(maxlen=64)
        self._track_history: dict[int, _TrackHistory] = {}

        self._current_speaker: Optional[str] = None
        self._current_track_id: Optional[int] = None
        self._current_open_since: Optional[float] = None
        self._current_confidence: float = 0.0
        self._silence_started_at: Optional[float] = None
        self._closed_events: list[dict] = []

    # ------------------------------------------------------------------
    # Audio ingestion
    # ------------------------------------------------------------------

    def update_audio(self, timestamp, voice_active, voice_confidence=None):
        confidence = 1.0 if voice_confidence is None else float(voice_confidence)
        self._audio_samples.append(_AudioSample(float(timestamp), bool(voice_active), confidence))

    def _nearest_audio(self, timestamp):
        best = None
        best_gap = None
        for sample in self._audio_samples:
            gap = abs(sample.timestamp - timestamp)
            if gap <= self._sync_tolerance_s and (best_gap is None or gap < best_gap):
                best = sample
                best_gap = gap
        return best

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def _mouth_motion_score(self, track_id, lip_open_ratio, timestamp):
        history = self._track_history.setdefault(track_id, _TrackHistory())
        history.lip_ratios.append((timestamp, lip_open_ratio))
        window_start = timestamp - self._window_s
        recent = [ratio for ts, ratio in history.lip_ratios if ts >= window_start]
        if len(recent) < 2:
            return 0.0
        spread = stdev(recent)
        return min(spread / self._lip_motion_norm, 1.0)

    def _record_score(self, track_id, timestamp, score):
        history = self._track_history.setdefault(track_id, _TrackHistory())
        history.scores.append((timestamp, score))

    def _smoothed_score(self, track_id, timestamp):
        history = self._track_history.get(track_id)
        if history is None:
            return 0.0
        window_start = timestamp - self._window_s
        recent = [score for ts, score in history.scores if ts >= window_start]
        return sum(recent) / len(recent) if recent else 0.0

    def _score_candidates(self, timestamp, voice_active, voice_confidence, face_observations):
        scored = []
        for observation in face_observations:
            motion_score = self._mouth_motion_score(observation.track_id, observation.lip_open_ratio, timestamp)
            is_incumbent = observation.track_id == self._current_track_id
            if voice_active:
                score = (
                    self._voice_weight * voice_confidence
                    + self._lip_weight * motion_score
                    + self._face_weight * observation.face_confidence
                    + self._temporal_weight * (1.0 if is_incumbent else 0.0)
                )
            else:
                score = 0.0
            self._record_score(observation.track_id, timestamp, score)
            scored.append({"observation": observation, "score": score, "motion_score": motion_score})
        return scored

    def _score_candidates_from_timestamp(self, timestamp, face_observations):
        """Test/debug helper: look up audio the same way update_faces does."""
        audio = self._nearest_audio(timestamp)
        voice_active = bool(audio and audio.voice_active)
        voice_confidence = audio.voice_confidence if audio else 0.0
        return self._score_candidates(timestamp, voice_active, voice_confidence, face_observations)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.\directmlvenv\Scripts\python.exe -m pytest tests/test_speaker_attribution.py -v`
Expected: PASS (all 5 tests)

- [ ] **Step 5: Commit**

```powershell
git add speaker_attribution.py tests/test_speaker_attribution.py
git commit -m "feat(speaker_attribution): add data structures, audio sync, and scoring"
```

---

### Task 4: `SpeakerAttributor.update_faces` — decision logic with hysteresis

**Files:**
- Modify: `speaker_attribution.py`
- Modify: `tests/test_speaker_attribution.py`

**Interfaces:**
- Consumes: `FaceObservation`, `_score_candidates`, `_smoothed_score` from Task 3.
- Produces: `SpeakerAttributor.update_faces(timestamp, face_observations) -> dict` with keys `timestamp`, `active_speaker`, and (when `active_speaker is not None`) `track_id`, `confidence`, `reason`. Consumed by Tasks 6, 7.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_speaker_attribution.py`:

```python
def test_no_speech_returns_none():
    attributor = SpeakerAttributor()
    attributor.update_audio(1.0, False)
    result = attributor.update_faces(1.0, [_observation(1)])
    assert result == {"timestamp": 1.0, "active_speaker": None}


def test_speech_with_no_faces_returns_unknown():
    attributor = SpeakerAttributor()
    attributor.update_audio(1.0, True, 0.9)
    result = attributor.update_faces(1.0, [])
    assert result["active_speaker"] == "UNKNOWN"
    assert result["track_id"] is None


def test_first_assignment_is_immediate_no_incumbent_wait():
    attributor = SpeakerAttributor()
    attributor.update_audio(1.0, True, 1.0)
    result = attributor.update_faces(1.0, [_observation(1, person_name="Alice", face_confidence=0.9)])
    assert result["active_speaker"] == "Alice"
    assert result["track_id"] == 1


def test_low_confidence_candidate_is_unknown_not_a_guessed_name():
    attributor = SpeakerAttributor()
    attributor.update_audio(1.0, True, 0.0)
    result = attributor.update_faces(1.0, [_observation(1, face_confidence=0.1)])
    assert result["active_speaker"] == "UNKNOWN"


def test_does_not_flip_on_single_frame_fluctuation():
    attributor = SpeakerAttributor()
    attributor.update_audio(1.0, True, 1.0)
    attributor.update_faces(1.0, [_observation(1, person_name="Alice", face_confidence=0.9)])

    attributor.update_audio(1.05, True, 1.0)
    result = attributor.update_faces(
        1.05,
        [
            _observation(1, person_name="Alice", face_confidence=0.9),
            _observation(2, person_name="Bob", face_confidence=0.91),  # barely higher, one frame only
        ],
    )
    assert result["active_speaker"] == "Alice"


def test_switches_after_sustained_margin():
    attributor = SpeakerAttributor()
    for t in (1.0, 1.1, 1.2, 1.3):
        attributor.update_audio(t, True, 1.0)
        attributor.update_faces(
            t,
            [
                _observation(1, person_name="Alice", face_confidence=0.5),
                _observation(2, person_name="Bob", face_confidence=0.95),
            ],
        )
    result = attributor.update_faces(
        1.3,
        [
            _observation(1, person_name="Alice", face_confidence=0.5),
            _observation(2, person_name="Bob", face_confidence=0.95),
        ],
    )
    assert result["active_speaker"] == "Bob"


def test_grace_period_keeps_speaker_through_brief_silence():
    attributor = SpeakerAttributor()
    attributor.update_audio(1.0, True, 1.0)
    attributor.update_faces(1.0, [_observation(1, person_name="Alice", face_confidence=0.9)])

    attributor.update_audio(1.1, False)
    result = attributor.update_faces(1.1, [_observation(1, person_name="Alice", face_confidence=0.9)])
    assert result["active_speaker"] == "Alice"


def test_speaker_becomes_none_after_grace_period_expires():
    attributor = SpeakerAttributor()
    attributor.update_audio(1.0, True, 1.0)
    attributor.update_faces(1.0, [_observation(1, person_name="Alice", face_confidence=0.9)])

    grace_s = config.SPEAKER_GRACE_PERIOD_MS / 1000.0
    attributor.update_audio(1.0 + grace_s + 0.1, False)
    result = attributor.update_faces(1.0 + grace_s + 0.1, [_observation(1, person_name="Alice", face_confidence=0.9)])
    assert result["active_speaker"] is None


def test_unknown_face_can_be_the_active_speaker():
    attributor = SpeakerAttributor()
    attributor.update_audio(1.0, True, 1.0)
    result = attributor.update_faces(
        1.0,
        [_observation(12, person_id=None, person_name="UNKNOWN", face_confidence=0.9)],
    )
    assert result["active_speaker"] == "UNKNOWN"
    assert result["track_id"] == 12
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.\directmlvenv\Scripts\python.exe -m pytest tests/test_speaker_attribution.py -v`
Expected: FAIL — `AttributeError: 'SpeakerAttributor' object has no attribute 'update_faces'`

- [ ] **Step 3: Implement `update_faces` and its helpers**

Append to the `SpeakerAttributor` class in `speaker_attribution.py`:

```python
    # ------------------------------------------------------------------
    # Decision / hysteresis
    # ------------------------------------------------------------------

    def update_faces(self, timestamp, face_observations):
        audio = self._nearest_audio(timestamp)
        voice_active = bool(audio and audio.voice_active)
        voice_confidence = audio.voice_confidence if audio else 0.0

        scored = self._score_candidates(timestamp, voice_active, voice_confidence, face_observations)

        if self.debug:
            self._log_debug(scored)

        if not voice_active:
            return self._handle_silence(timestamp)

        self._silence_started_at = None

        if not scored:
            if self._current_speaker is not None:
                self._close_current_event(timestamp)
            return self._result(timestamp, "UNKNOWN", None, 0.0, {})

        return self._decide(timestamp, voice_confidence, scored)

    def _handle_silence(self, timestamp):
        if self._current_speaker is None:
            return self._result(timestamp, None, None, 0.0, {})

        if self._silence_started_at is None:
            self._silence_started_at = timestamp
        elif timestamp - self._silence_started_at >= self._grace_period_s:
            self._close_current_event(timestamp)
            return self._result(timestamp, None, None, 0.0, {})

        return self._result(
            timestamp, self._current_speaker, self._current_track_id,
            self._current_confidence, {"grace_period": True},
        )

    def _decide(self, timestamp, voice_confidence, scored):
        best = max(scored, key=lambda item: item["score"])
        observation = best["observation"]
        label = observation.person_name if observation.person_id is not None else "UNKNOWN"
        smoothed = self._smoothed_score(observation.track_id, timestamp)

        if self._current_track_id is None:
            switch = best["score"] >= self._min_confidence
        elif observation.track_id == self._current_track_id:
            switch = True
        else:
            incumbent_smoothed = self._smoothed_score(self._current_track_id, timestamp)
            switch = (
                smoothed >= self._min_confidence
                and smoothed - incumbent_smoothed >= self._switch_threshold
            )

        if not switch:
            if self._current_speaker is not None:
                return self._result(
                    timestamp, self._current_speaker, self._current_track_id,
                    self._current_confidence, {"held_incumbent": True},
                )
            label = "UNKNOWN"

        if label != self._current_speaker or observation.track_id != self._current_track_id:
            if self._current_speaker is not None:
                self._close_current_event(timestamp)
            self._current_speaker = label
            self._current_track_id = observation.track_id
            self._current_open_since = timestamp

        self._current_confidence = best["score"]
        reason = {
            "voice_activity": voice_confidence,
            "mouth_motion": best["motion_score"],
            "face_confidence": observation.face_confidence,
            "temporal_consistency": smoothed,
        }
        return self._result(timestamp, label, observation.track_id, best["score"], reason)

    def _close_current_event(self, timestamp):
        if self._current_speaker is not None and self._current_open_since is not None:
            self._closed_events.append({
                "speaker": self._current_speaker,
                "track_id": self._current_track_id,
                "start_time": self._current_open_since,
                "end_time": timestamp,
                "confidence": self._current_confidence,
            })
        self._current_speaker = None
        self._current_track_id = None
        self._current_open_since = None
        self._current_confidence = 0.0
        self._silence_started_at = None

    def _result(self, timestamp, speaker, track_id, confidence, reason):
        result = {"timestamp": timestamp, "active_speaker": speaker}
        if speaker is not None:
            result["track_id"] = track_id
            result["confidence"] = confidence
            result["reason"] = reason
        return result

    def _log_debug(self, scored):
        import logging
        logger = logging.getLogger(__name__)
        for item in sorted(scored, key=lambda entry: entry["score"], reverse=True):
            observation = item["observation"]
            logger.debug(
                "Track %s -> %s -> speaker score %.2f",
                observation.track_id, observation.person_name, item["score"],
            )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.\directmlvenv\Scripts\python.exe -m pytest tests/test_speaker_attribution.py -v`
Expected: PASS (all tests, including Task 3's)

- [ ] **Step 5: Commit**

```powershell
git add speaker_attribution.py tests/test_speaker_attribution.py
git commit -m "feat(speaker_attribution): add hysteresis, grace period, and decision logic"
```

---

### Task 5: Speaker events — `pop_closed_events`

**Files:**
- Modify: `speaker_attribution.py` (already implements `_close_current_event`; this task adds the public accessor and covers boundary cases)
- Modify: `tests/test_speaker_attribution.py`

**Interfaces:**
- Produces: `SpeakerAttributor.pop_closed_events() -> list[dict]` with keys `speaker`, `track_id`, `start_time`, `end_time`, `confidence`. Consumed by Tasks 6, 7.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_speaker_attribution.py`:

```python
def test_pop_closed_events_returns_empty_list_initially():
    attributor = SpeakerAttributor()
    assert attributor.pop_closed_events() == []


def test_switching_speaker_closes_previous_event():
    attributor = SpeakerAttributor()
    for t in (1.0, 1.1, 1.2, 1.3):
        attributor.update_audio(t, True, 1.0)
        attributor.update_faces(
            t,
            [
                _observation(1, person_name="Alice", face_confidence=0.5),
                _observation(2, person_name="Bob", face_confidence=0.95),
            ],
        )
    attributor.update_faces(
        1.3,
        [
            _observation(1, person_name="Alice", face_confidence=0.5),
            _observation(2, person_name="Bob", face_confidence=0.95),
        ],
    )

    events = attributor.pop_closed_events()
    assert len(events) == 1
    assert events[0]["speaker"] == "Alice"
    assert events[0]["start_time"] == 1.0
    assert events[0]["end_time"] == 1.3

    # Draining again returns nothing new until another switch/close happens.
    assert attributor.pop_closed_events() == []


def test_event_closes_when_speech_ends_after_grace_period():
    attributor = SpeakerAttributor()
    attributor.update_audio(1.0, True, 1.0)
    attributor.update_faces(1.0, [_observation(1, person_name="Alice", face_confidence=0.9)])

    grace_s = config.SPEAKER_GRACE_PERIOD_MS / 1000.0
    end_t = 1.0 + grace_s + 0.1
    attributor.update_audio(end_t, False)
    attributor.update_faces(end_t, [_observation(1, person_name="Alice", face_confidence=0.9)])

    events = attributor.pop_closed_events()
    assert events == [{
        "speaker": "Alice", "track_id": 1, "start_time": 1.0, "end_time": end_t, "confidence": events[0]["confidence"],
    }]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.\directmlvenv\Scripts\python.exe -m pytest tests/test_speaker_attribution.py -v`
Expected: FAIL — `AttributeError: 'SpeakerAttributor' object has no attribute 'pop_closed_events'`

- [ ] **Step 3: Add the accessor**

Append to the `SpeakerAttributor` class:

```python
    # ------------------------------------------------------------------
    # Events
    # ------------------------------------------------------------------

    def pop_closed_events(self):
        events, self._closed_events = self._closed_events, []
        return events
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.\directmlvenv\Scripts\python.exe -m pytest tests/test_speaker_attribution.py -v`
Expected: PASS (full file)

- [ ] **Step 5: Commit**

```powershell
git add speaker_attribution.py tests/test_speaker_attribution.py
git commit -m "feat(speaker_attribution): add pop_closed_events for speaker-run logging"
```

---

### Task 6: Integrate into `video_processor.py`

**Files:**
- Modify: `video_processor.py:586` (init), `video_processor.py:940-966` (per-track candidate building), `video_processor.py:1037-1100` (speaker selection + logging)
- Modify: `tests/test_video_processor.py`

**Interfaces:**
- Consumes: `speaker_attribution.FaceObservation`, `SpeakerAttributor` (Tasks 3-5); `database.log_audio(..., track_id=...)` (Task 2).

- [ ] **Step 1: Write the failing test**

Append to `tests/test_video_processor.py` (reusing this file's existing `_patch_common`/`_FakeCapture` fixtures):

```python
def test_process_video_pipeline_logs_unknown_speaker_events(tmp_path, monkeypatch):
    """An unknown-but-speaking track must now produce an audio_logs event,
    where the old `if speaker.person_id is not None` guard used to drop it."""
    capture = _FakeCapture(frame_count=1)
    writer = _FakeWriter()
    _patch_common(monkeypatch, capture, writer)
    monkeypatch.setattr(video_processor, "detect_speech_segments", lambda *_a, **_k: [(0.0, 10.0)])

    class _Track:
        def __init__(self):
            self.track_id = 1
            self.person_id = None
            self.person_name = "Unknown"
            self.confidence = 0.0
            self.embedding = None
            self.lip_open = 1.0
            self.speech_frames = 0
            self.silent_frames = 0

    track = _Track()
    monkeypatch.setattr(
        video_processor, "CentroidTracker",
        lambda: SimpleNamespace(update=lambda dets, _n: [(dets[0], track)] if dets else [], tracks={1: track}),
    )
    monkeypatch.setattr(video_processor, "detect_faces_tiled", lambda *_a, **_k: [(0, 0, 60, 60, 0.9)])

    logged = []
    monkeypatch.setattr(video_processor, "log_audio", lambda *args, **kwargs: logged.append((args, kwargs)))

    video_processor.process_video_pipeline(
        Path("dummy.mp4"), tmp_path / "out.mp4", tmp_path / "log.json",
    )

    assert logged, "an UNKNOWN speaking event should have been logged, not skipped"
    args, kwargs = logged[0]
    # log_audio(start, end, person_id, person_name, confidence, source, track_id=...)
    assert args[3] == "UNKNOWN"
    assert kwargs["track_id"] == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.\directmlvenv\Scripts\python.exe -m pytest tests/test_video_processor.py::test_process_video_pipeline_logs_unknown_speaker_events -v`
Expected: FAIL (no event logged, because the current code only logs when `person_id is not None`)

- [ ] **Step 3: Replace the speaker-selection block**

At the top of `video_processor.py`, add the import:

```python
from speaker_attribution import FaceObservation, SpeakerAttributor
```

Where `speaking_candidates = []` is initialized per-frame (around `video_processor.py:586`), also build face observations instead:

```python
        speaking_candidates = []
        face_observations = []
```

Where each track currently appends to `speaking_candidates` (around `video_processor.py:960-966`), replace:

```python
            if (
                audio_is_speech
                and
                track.lip_open is not None
                and
                track.lip_open
                >= LIP_OPEN_THRESHOLD
            ):

                speaking_candidates.append(
                    track
                )
```

with:

```python
            if track.lip_open is not None:
                face_observations.append(FaceObservation(
                    track_id=track.track_id,
                    person_id=track.person_id,
                    person_name=track.person_name if track.person_id is not None else "UNKNOWN",
                    face_confidence=track.confidence,
                    mouth_open=track.lip_open >= LIP_OPEN_THRESHOLD,
                    lip_open_ratio=track.lip_open,
                    face_visible=True,
                ))
```

Before the frame loop begins (near `frame_no = 0` at `video_processor.py:~479`), instantiate the attributor once per pipeline run:

```python
    attributor = SpeakerAttributor()
```

Replace the `# Select ONE active speaker` block (`video_processor.py:1037-1100`) — which today does:

```python
        if (
            audio_is_speech
            and
            speaking_candidates
        ):

            speaker = max(
                speaking_candidates,
                key=lambda t:
                    t.lip_open,
            )

            speaker.speech_frames += 1
            speaker.silent_frames = 0

            if (
                speaker.speech_frames
                >= 2
            ):

                start = max(
                    0.0,
                    timestamp - 0.1,
                )

                end = timestamp

                if (
                    speaker.person_id
                    is not None
                ):

                    log_audio(
                        start,
                        end,
                        speaker.person_id,
                        speaker.person_name,
                        speaker.confidence,
                        "video",
                    )

                    active_speech_logs.append(
                        {
                            "start_time": round(
                                start,
                                2,
                            ),
                            "end_time": round(
                                end,
                                2,
                            ),
                            "person_name": (
                                speaker.person_name
                            ),
                            "confidence": round(
                                speaker.confidence,
                                4,
                            ),
                            "source": "video",
                        }
                    )
```

with:

```python
        attributor.update_audio(timestamp, audio_is_speech)
        result = attributor.update_faces(timestamp, face_observations)

        for event in attributor.pop_closed_events():
            person_id = None
            if event["speaker"] != "UNKNOWN":
                match = next(
                    (t for t in tracker.tracks.values() if t.track_id == event["track_id"]),
                    None,
                )
                person_id = match.person_id if match is not None else None

            log_audio(
                event["start_time"],
                event["end_time"],
                person_id,
                event["speaker"],
                event["confidence"],
                "video",
                track_id=event["track_id"],
            )

            active_speech_logs.append({
                "start_time": round(event["start_time"], 2),
                "end_time": round(event["end_time"], 2),
                "person_name": event["speaker"],
                "confidence": round(event["confidence"], 4),
                "source": "video",
            })
```

Note the existing `else:` branch immediately following (the one incrementing `track.silent_frames`) stays as-is — it is unrelated to speaker *selection* and still tracks per-track silence bookkeeping used elsewhere.

- [ ] **Step 4: Run test to verify it passes**

Run: `.\directmlvenv\Scripts\python.exe -m pytest tests/test_video_processor.py -v`
Expected: PASS (new test and all pre-existing ones in the file)

- [ ] **Step 5: Run the full suite**

Run: `.\directmlvenv\Scripts\python.exe -m pytest tests -q`
Expected: PASS

- [ ] **Step 6: Commit**

```powershell
git add video_processor.py tests/test_video_processor.py
git commit -m "feat(video_processor): use speaker_attribution fusion layer, log UNKNOWN events"
```

---

### Task 7: Integrate into `realtime.py`

**Files:**
- Modify: `realtime.py:294-350` (init), `realtime.py:990-1006` (candidate building), `realtime.py:1133-1260` (speaker selection + logging), `realtime.py:1367-1380` (final flush on exit)
- Modify: `tests/test_realtime.py`

**Interfaces:**
- Consumes: same as Task 6.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_realtime.py`, modeled on the existing `test_speaking_banner_drawn_below_fps_line` fixture in the same file:

```python
def test_run_logs_unknown_speaking_track(monkeypatch):
    """Mirrors test_speaking_banner_drawn_below_fps_line but with an
    unresolved track, asserting the new code path still logs it."""
    import realtime

    frame = np.zeros((120, 160, 3), dtype=np.uint8)

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
            if self.calls == 1:
                return True, frame.copy()
            return False, None

        def release(self):
            return None

    class _Track:
        def __init__(self):
            self.track_id = 9
            self.person_id = None
            self.person_name = "Unknown"
            self.confidence = 0.0
            self.embedding = None
            self.center_x = 60
            self.center_y = 60
            self.lip_open = 1.0

    class _Tracker:
        def __init__(self):
            self.tracks = {9: _Track()}

        def update(self, detections, frame_no):
            return [(detections[0], self.tracks[9])] if detections else []

    class _Mesh:
        def detect_for_video(self, *_args, **_kwargs):
            return SimpleNamespace(face_landmarks=[SimpleNamespace()])

        def close(self):
            return None

    class _Vad:
        def __init__(self):
            self.available = True

        def stop(self):
            return None

    logged = []

    monkeypatch.setattr(realtime.cv2, "VideoCapture", lambda *_args, **_kwargs: _Cap())
    monkeypatch.setattr(realtime, "FaceIndex", lambda: SimpleNamespace(search=lambda *_args, **_kwargs: None))
    monkeypatch.setattr(realtime, "CentroidTracker", lambda: _Tracker())

    def _start_realtime_vad(callback):
        callback(True)
        return _Vad(), True

    monkeypatch.setattr(realtime, "start_realtime_vad", _start_realtime_vad)
    monkeypatch.setattr(realtime, "create_face_landmarker", lambda: _Mesh())
    monkeypatch.setattr(realtime, "detect_faces_realtime", lambda *_args, **_kwargs: [(10, 10, 60, 60, 0.9)])
    monkeypatch.setattr(realtime, "extract_embedding", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(realtime, "face_quality", lambda *_args, **_kwargs: 0.0)
    monkeypatch.setattr(realtime, "log_recognition", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(realtime, "log_audio", lambda *args, **kwargs: logged.append((args, kwargs)))
    monkeypatch.setattr(realtime.cv2, "cvtColor", lambda image, _code: image)
    monkeypatch.setattr(realtime.cv2, "imshow", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(realtime.cv2, "waitKey", lambda *_args, **_kwargs: 0)
    monkeypatch.setattr(realtime.cv2, "getWindowProperty", lambda *_args, **_kwargs: 1)
    monkeypatch.setattr(realtime.cv2, "rectangle", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(realtime.cv2, "putText", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(realtime.cv2, "destroyAllWindows", lambda: None)

    dummy_mp = SimpleNamespace(ImageFormat=SimpleNamespace(SRGB=1), Image=lambda image_format, data: data)
    import sys
    fake_video_processor = ModuleType("video_processor")
    fake_video_processor.lip_open_ratio = lambda _landmarks: 1.0
    monkeypatch.setitem(sys.modules, "video_processor", fake_video_processor)
    monkeypatch.setitem(sys.modules, "mediapipe", dummy_mp)

    realtime.run(camera=0, width=160, height=120)

    assert logged, "an UNKNOWN speaking track should be logged on realtime exit, not skipped"
    args, kwargs = logged[0]
    # log_audio(start, end, person_id, person_name, confidence, source, track_id=...)
    assert args[3] == "UNKNOWN"
    assert kwargs["track_id"] == 9
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.\directmlvenv\Scripts\python.exe -m pytest tests/test_realtime.py::test_run_logs_unknown_speaking_track -v`
Expected: FAIL (today's code drops this event; `logged` stays empty until the pipeline exits without ever calling `log_audio` for an unresolved `person_id`)

- [ ] **Step 3: Replace the speaker-selection block**

Add the import at the top of `realtime.py`:

```python
from speaker_attribution import FaceObservation, SpeakerAttributor
```

Near the other per-run state initialization (`realtime.py:294-296`, alongside `last_speaker_id = None`), add:

```python
    attributor = SpeakerAttributor()
```

Where `candidates.append(track)` happens today (`realtime.py:990-1006`):

```python
                if (
                    audio_available
                    and audio_state.value
                    and track.lip_open is not None
                    and
                    track.lip_open
                    >= LIP_OPEN_THRESHOLD
                ):

                    candidates.append(
                        track
                    )
```

replace with building a `FaceObservation` unconditionally (voice-gating now happens inside the attributor, not here) and feeding the VAD state as an audio sample once per frame:

```python
                if track.lip_open is not None:
                    face_observations.append(FaceObservation(
                        track_id=track.track_id,
                        person_id=track.person_id,
                        person_name=track.person_name if track.person_id is not None else "UNKNOWN",
                        face_confidence=track.confidence,
                        mouth_open=track.lip_open >= LIP_OPEN_THRESHOLD,
                        lip_open_ratio=track.lip_open,
                        face_visible=True,
                    ))
```

Declare `face_observations = []` alongside the existing `candidates = []` line before the per-track loop.

Replace the `# Select active speaker` block (`realtime.py:1133-1260`):

```python
            if (
                audio_available
                and audio_state.value
                and
                candidates
            ):

                speaker = max(
                    candidates,
                    key=lambda t:
                        t.lip_open,
                )
                ... (voice fallback, drawing, last_speaker_* bookkeeping) ...

            else:

                if (
                    last_speaker_id
                    is not None
                    ...
                ):

                    log_audio(
                        last_speech_start,
                        time.time(),
                        last_speaker_id,
                        last_speaker_name,
                        last_speaker_confidence,
                        "realtime",
                    )

                last_speaker_id = None
                last_speech_start = None
```

with:

```python
            attributor.update_audio(timestamp, bool(audio_available and audio_state.value))
            result = attributor.update_faces(timestamp, face_observations)

            speaker_track = None
            if result["active_speaker"] is not None:
                speaker_track = tracker.tracks.get(result["track_id"])

            if speaker_track is not None:
                speaker_id = speaker_track.person_id
                speaker_name = result["active_speaker"]
                speaker_confidence = result["confidence"]

                # Voice-identity fallback: only overrides display/logging for
                # this frame, exactly as before — never written onto the track.
                if speaker_track.person_id is None or speaker_track.confidence < RECOGNITION_THRESHOLD:
                    voice_match = voice_fallback_match(speaker_track, frame_no, vad)
                    if voice_match is not None:
                        speaker_id = voice_match["person_id"]
                        speaker_name = voice_match["person_name"]
                        speaker_confidence = voice_match["similarity"]

                cv2.putText(
                    frame,
                    f"SPEAKING: {speaker_name}",
                    (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 0),
                    2,
                )
                cv2.rectangle(
                    frame,
                    (int(speaker_track.center_x - 80), int(speaker_track.center_y - 100)),
                    (int(speaker_track.center_x + 80), int(speaker_track.center_y + 100)),
                    (0, 255, 0),
                    3,
                )

            for event in attributor.pop_closed_events():
                log_audio(
                    event["start_time"],
                    event["end_time"],
                    None if event["speaker"] == "UNKNOWN" else tracker.tracks.get(event["track_id"], SimpleNamespace(person_id=None)).person_id,
                    event["speaker"],
                    event["confidence"],
                    "realtime",
                    track_id=event["track_id"],
                )
```

At the `finally:` block's final-event flush (`realtime.py:1367-1380`), replace the `last_speaker_id`-based flush with:

```python
        try:
            for event in attributor.pop_closed_events():
                log_audio(
                    event["start_time"],
                    event["end_time"],
                    None if event["speaker"] == "UNKNOWN" else tracker.tracks.get(event["track_id"], SimpleNamespace(person_id=None)).person_id,
                    event["speaker"],
                    event["confidence"],
                    "realtime",
                    track_id=event["track_id"],
                )
        except Exception:
            logging.exception("Failed to save final realtime audio event")
```

Remove the now-unused `last_speaker_id`, `last_speaker_name`, `last_speaker_confidence`, `last_speech_start` variables and their remaining references once nothing else reads them.

- [ ] **Step 4: Run test to verify it passes**

Run: `.\directmlvenv\Scripts\python.exe -m pytest tests/test_realtime.py -v`
Expected: PASS, including the pre-existing `test_speaking_banner_drawn_below_fps_line` (default weights place a single confident, incumbent-less candidate — `VOICE_ACTIVITY_WEIGHT*1.0 + FACE_CONFIDENCE_WEIGHT*0.99 = 0.598 >= SPEAKER_MIN_CONFIDENCE=0.35` — above the immediate-assignment floor even with `mouth_motion_score=0.0` on the very first frame)

- [ ] **Step 5: Run the full suite**

Run: `.\directmlvenv\Scripts\python.exe -m pytest tests -q`
Expected: PASS

- [ ] **Step 6: Commit**

```powershell
git add realtime.py tests/test_realtime.py
git commit -m "feat(realtime): use speaker_attribution fusion layer, log UNKNOWN events"
```

---

### Task 8: Surface `track_id` in the Streamlit audio-log tables

**Files:**
- Modify: `app.py` (two `fetch_audio_logs()` call sites: the "Recent Audio Logs" table near `app.py:148`, and `render_audio_logs()` near `app.py:851`)

**Interfaces:**
- Consumes: `database.fetch_audio_logs()`'s new 8-tuple row shape from Task 2.

- [ ] **Step 1: Update the unpacking and displayed columns**

In both places currently doing:

```python
        _id, start, end, name, confidence, source, transcript = row
```

change to:

```python
        _id, start, end, name, confidence, source, transcript, track_id = row
```

and add `"Track": track_id if track_id is not None else ""` to each row's dict literal (next to the existing `"Person"` key), in both the "Recent Audio Logs" block (`app.py:~142-150`) and `render_audio_logs()` (`app.py:~851-862`).

- [ ] **Step 2: Verify manually**

Run: `.\directmlvenv\Scripts\python.exe -m py_compile app.py`
Expected: no errors

Run: `.\directmlvenv\Scripts\python.exe -m pytest tests -q`
Expected: PASS (full suite — this task has no dedicated test file since `app.py`'s Streamlit rendering has no existing test harness in this repo; correctness here is compile-check + full-suite regression only)

- [ ] **Step 3: Commit**

```powershell
git add app.py
git commit -m "feat(app): display track_id in audio-log tables"
```

---

## Manual verification (after all tasks)

1. Offline: `streamlit run app.py` → Upload Video Recognition → process one of the existing `initialVideo/` clips with multiple visible faces → open Audio Logs → confirm events show plausible speaker switches (not one row per frame) and that any unresolved face appears as `UNKNOWN` with a `track_id` instead of being silently missing.
2. Realtime: launch realtime mode with webcam + mic, have two people alternate speaking → confirm the on-screen `SPEAKING: <name>` label follows the correct person without flickering on brief pauses, and check `Recent Audio Logs` afterward for a clean event timeline.
3. Debug mode: instantiate `SpeakerAttributor(debug=True)` in a throwaway script against a recorded clip's observations and confirm per-track score lines are logged at `DEBUG` level.
