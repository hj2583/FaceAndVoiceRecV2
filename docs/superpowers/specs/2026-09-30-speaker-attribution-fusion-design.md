# Speaker Attribution / Fusion Layer

## Problem

Face recognition (detection, tracking, ArcFace identity) and audio processing (Silero VAD) are both fully working and independent. Neither system, nor anything combining them, can currently answer "who is speaking right now?" in a principled way.

Active-speaker selection exists today, but it is duplicated in two places and both copies do the exact naive thing this feature must avoid:

```python
speaking_candidates = [t for t in tracks if voice_active and t.lip_open >= LIP_OPEN_THRESHOLD]
speaker = max(speaking_candidates, key=lambda t: t.lip_open)   # no scoring, no hysteresis
```

- `video_processor.py` (offline video pipeline), ~L900-1110
- `realtime.py` (live camera/mic pipeline), ~L990-1260

Concrete problems found by reading both implementations:
- No temporal smoothing: `speech_frames`/`silent_frames` counters exist on `Track` but only gate *whether to log an event*, not *whether to switch the selected speaker*. The winning candidate can flip every frame.
- Unknown speakers are silently dropped: `video_processor.py` only calls `log_audio()` when `speaker.person_id is not None`, so a clearly-speaking unknown face never produces an event today.
- `config.SPEAKER_FACE_OVERLAP_THRESHOLD` is dead: defined, never read anywhere.
- Realtime and offline compute "is voice active" differently (async boolean callback vs. precomputed `(start, end)` segment membership test), so any fusion logic must treat audio as timestamped samples rather than assuming a shared per-frame clock.

## Goals

- Add a single, shared speaker-attribution/fusion module used by both the realtime and offline pipelines — not two separate implementations.
- Score every visible tracked face each time observations arrive; never assume the first detected face is speaking.
- Support `active_speaker = None` (no speech) and `active_speaker = "UNKNOWN"` (speech with no confidently-associated face), in addition to a resolved person name.
- Smooth speaker selection over time (hysteresis) so it does not flip on single-frame fluctuations.
- Synchronize face and audio observations by timestamp, tolerant of the two pipelines producing observations at different rates.
- Emit speaker *events* (start/end runs), not one row per frame.
- Keep `track_id`, `person_id`, and `speaker_id` conceptually distinct in the module's output.

## Non-Goals

- Replacing face recognition, tracking, or VAD/voice-embedding code. All of it stays as-is.
- Building a new UI framework. The existing realtime overlay and Streamlit tables get minimal additions only.
- Requiring voice-identity recognition. It is used opportunistically (already exists as `voice_core.match_voice_embedding`) but VAD-only operation is fully supported.
- New database tables. `audio_logs` already stores exactly the event shape needed.

## Architecture

```mermaid
flowchart LR
    subgraph Face pipeline (unchanged)
        FD[Detection] --> FT[CentroidTracker]
        FT --> FR[ArcFace recognition]
        FR --> TR[Track: track_id, person_id, person_name, confidence, lip_open]
    end
    subgraph Audio pipeline (unchanged)
        VAD[Silero VAD] --> VA["voice_active(t)"]
        VC[voice_core.match_voice_embedding] --> VI[optional voice identity]
    end
    TR --> FO[FaceObservation per track, per frame]
    VA --> AO[AudioObservation: timestamp, voice_active, voice_confidence]
    FO --> SA[speaker_attribution.SpeakerAttributor]
    AO --> SA
    VI -.optional signal.-> SA
    SA --> RES["update_faces() -> {active_speaker, track_id, confidence, reason}"]
    SA --> EVENTS["pop_closed_events() -> [{speaker, start_time, end_time, confidence}]"]
    RES --> UI[Realtime overlay / Streamlit]
    EVENTS --> LOG[database.log_audio - existing table, no schema change to rows]
```

`speaker_attribution.py` contains no face detection and no audio capture/VAD code. It consumes normalized observations built by the two existing pipelines from data they already have (`Track` fields, VAD state), and it is the only place the scoring/smoothing/event logic lives.

## Data structures / interfaces

```python
# speaker_attribution.py

@dataclass(frozen=True)
class FaceObservation:
    track_id: int
    person_id: Optional[int]       # None => unresolved/unknown identity
    person_name: str                # "UNKNOWN" when person_id is None
    face_confidence: float          # track.confidence (0 if never recognized)
    mouth_open: bool                # track.lip_open >= config.LIP_OPEN_THRESHOLD
    lip_open_ratio: float           # track.lip_open, raw value (module derives motion from this)
    face_visible: bool              # detection succeeded this frame for this track


class SpeakerAttributor:
    def __init__(self, config_overrides: dict | None = None): ...

    def update_audio(
        self,
        timestamp: float,
        voice_active: bool,
        voice_confidence: Optional[float] = None,
    ) -> None:
        """Record one timestamped VAD sample. Called every frame offline;
        called from the RealtimeVAD callback in realtime mode."""

    def update_faces(
        self,
        timestamp: float,
        face_observations: list[FaceObservation],
    ) -> dict:
        """Score all observations, apply hysteresis, return the current
        attribution result. Also enqueues event start/end transitions
        internally for pop_closed_events()."""
        # Returns one of:
        # {"timestamp": t, "active_speaker": None}
        # {"timestamp": t, "active_speaker": "UNKNOWN", "track_id": 12, "confidence": 0.41, "reason": {...}}
        # {"timestamp": t, "active_speaker": "alice", "track_id": 3, "confidence": 0.89, "reason": {...}}

    def pop_closed_events(self) -> list[dict]:
        """Drain finished speaker runs since the last call:
        [{"speaker": "alice", "track_id": 3, "start_time": 12.50, "end_time": 15.20, "confidence": 0.89}, ...]
        speaker is a person name or "UNKNOWN"."""
```

`track_id`, `person_id`, and `speaker_id` stay distinct end-to-end: `speaker_id` in the result is derived (`person_name` if resolved, else `"UNKNOWN"`), `track_id` is always the tracker's stable id, and neither `update_faces` nor any caller writes attribution results back onto `Track` face-recognition fields (mirrors the existing realtime voice-fallback pattern, which already avoids that for the same reason).

## Scoring algorithm

```
score(candidate) =
      VOICE_ACTIVITY_WEIGHT * voice_confidence          # gated: candidate is not eligible at all if voice_active is False
    + LIP_MOTION_WEIGHT      * mouth_motion_score
    + FACE_CONFIDENCE_WEIGHT * face_confidence
    + TEMPORAL_WEIGHT        * (1.0 if candidate.track_id == current_speaker_track_id else 0.0)
```

- `voice_confidence` defaults to `1.0` when the audio source only provides a boolean (current VAD), so the term degrades gracefully without voice-identity recognition.
- `mouth_motion_score` is new and derived, not reused verbatim: `lip_open_ratio()` in `video_processor.py` is instantaneous, not a motion signal. The attribution module keeps a short per-`track_id` history of recent `lip_open_ratio` values (only as long as `SPEAKER_WINDOW_MS`) and computes `mouth_motion_score = min(stdev(history) / LIP_MOTION_NORM, 1.0)` — the standard deviation of that window, divided by a configurable normalization constant and clamped to `[0.0, 1.0]`. Fewer than 2 samples in the history yields `0.0` (no motion signal yet). `lip_open_ratio()` itself is not modified.
- If no candidate is eligible (no voice activity, or nobody scores above `SPEAKER_MIN_CONFIDENCE`) while `voice_active` is true, the result is `"UNKNOWN"`. If `voice_active` is false, the result is `None` (after the grace period below).

## Timestamp synchronization

Both pipelines already have a way to know "is voice active at time T" — offline via `(start, end)` segment membership, realtime via an async VAD callback — but they differ in shape. Rather than writing two matching implementations, `SpeakerAttributor` takes timestamped audio *samples* pushed via `update_audio()` and, inside `update_faces()`, selects the most recent audio sample within `AUDIO_SYNC_TOLERANCE_MS` of the query timestamp (falling back to "no voice data" beyond that tolerance, which behaves like `voice_active=False`). This one mechanism serves:
- **Offline:** call `update_audio(frame_timestamp, is_speech_at(frame_timestamp))` once per frame, immediately followed by `update_faces(frame_timestamp, ...)` — trivially within tolerance.
- **Realtime:** the VAD callback calls `update_audio(time.monotonic(), state)` whenever Silero's smoothed state changes; the frame loop calls `update_faces(time.monotonic(), ...)` independently at its own rate, looking up the nearest prior audio sample.

## Temporal smoothing (hysteresis)

- `SPEAKER_WINDOW_MS`: rolling window of recent per-candidate scores used to compute a smoothed score (not just the instantaneous frame).
- `SPEAKER_SWITCH_THRESHOLD`: a challenger must exceed the current incumbent's smoothed score by this margin before the active speaker changes.
- `SPEAKER_MIN_CONFIDENCE`: floor below which the result is `"UNKNOWN"` instead of naming a low-confidence guess.
- `SPEAKER_GRACE_PERIOD_MS`: how long to keep the current speaker selected after voice activity drops, before falling back to `None`. Prevents brief pauses mid-sentence from ending the speaker run.

## Multiple people / unknown faces

`update_faces()` scores every `FaceObservation` passed in, unconditionally — there is no "pick the first detection" path anywhere in the new module. Unknown faces (`person_id is None`) are scored identically to known ones; if one wins, the result's `active_speaker` is `"UNKNOWN"` and `track_id` is preserved in the result so the caller (and, per the behavior change below, `audio_logs`) can link back to that track for later identification via the existing `unknown_tracks` flow.

## Event / logging behavior

`pop_closed_events()` returns closed `{speaker, track_id, start_time, end_time, confidence}` runs. Both pipelines call this once per frame (or per audio update) and feed each closed event into the **existing** `database.log_audio()` — same table, same columns as today, no new table.

**Behavior change (approved):** both pipelines currently call `log_audio()` only `if speaker.person_id is not None`, silently dropping unknown-but-speaking events. This changes to always logging, with `person_id=None, person_name="UNKNOWN"` for unresolved speakers — `audio_logs.person_id` is already nullable and `person_name` is already `TEXT`, so this requires no schema change, only removing the `is not None` guard at both call sites.

**Additive schema change (approved):** `ALTER TABLE audio_logs ADD COLUMN track_id INTEGER` (nullable, backward-compatible — same migration pattern already used for `transcription_segments.sentence_type` in `database.py`). This directly satisfies "preserve the track ID so the unknown person can potentially be identified later" from `audio_logs` itself, in addition to the existing `unknown_tracks` linkage.

## Configuration (all in `config.py`, no hard-coded values in the module)

```python
SPEAKER_WINDOW_MS = 800
SPEAKER_SWITCH_THRESHOLD = 0.15
SPEAKER_MIN_CONFIDENCE = 0.35
SPEAKER_GRACE_PERIOD_MS = 600
AUDIO_SYNC_TOLERANCE_MS = 250

VOICE_ACTIVITY_WEIGHT = 0.40
LIP_MOTION_WEIGHT = 0.30
FACE_CONFIDENCE_WEIGHT = 0.20
TEMPORAL_WEIGHT = 0.10

# Standard deviation of a track's recent lip_open_ratio history is divided by
# this constant, then clamped to [0, 1], to produce mouth_motion_score.
LIP_MOTION_NORM = 0.02
```

Starting points only, matching this codebase's existing convention (e.g. `VOICE_MATCH_THRESHOLD`'s comment) of "validate against real recordings." The unused `SPEAKER_FACE_OVERLAP_THRESHOLD` is left in place but out of scope for this feature (it predates this work and nothing here depends on it).

## New / changed files

- `speaker_attribution.py` (new): `FaceObservation`, `SpeakerAttributor`, scoring, hysteresis, event tracking. No face/audio I/O.
- `tests/test_speaker_attribution.py` (new): scoring, hysteresis/no-flip, sync tolerance, unknown-speaker eligibility, multi-candidate selection, event open/close boundaries.
- `video_processor.py`: replace the `speaking_candidates`/`max(...)` block with `SpeakerAttributor` calls; remove the `person_id is not None` guard before `log_audio()`.
- `realtime.py`: same replacement; `RealtimeVAD` callback feeds `update_audio()`; remove the same guard.
- `config.py`: add the parameters above.
- `database.py`: additive `track_id` column on `audio_logs` (same `PRAGMA table_info` + `ALTER TABLE` pattern already used for `sentence_type`); `log_audio()` gains an optional `track_id` parameter.
- `app.py`: minimal display addition — show `Active Speaker: <name>` / `Confidence: <pct>` where the realtime page already reports state; no new page or framework.

## Debug mode

An optional `debug=True` path on `SpeakerAttributor` (or a module-level logger at `DEBUG` level) that logs, per call to `update_faces()`, every candidate's component scores and total, e.g. `Track 1 -> John -> speaker score 0.82`. Off by default; no UI changes required for this, existing Python logging is sufficient.

## Testing

- Unit tests in `tests/test_speaker_attribution.py` cover: single/multiple candidates, unknown-speaker eligibility, no-speech → `None`, speech-but-no-match → `"UNKNOWN"`, hysteresis (no flip on a one-frame blip, switch after sustained margin), grace period, and sync-tolerance edge cases (audio sample just inside/outside tolerance).
- Existing test suites (`tests/test_video_processor.py`, `tests/test_realtime.py`, `tests/test_realtime_runtime.py`, `tests/test_database_*`) must continue to pass unmodified in behavior except where this spec explicitly changes logging behavior (unknown-speaker events now logged).
- Manual verification procedure using existing sample media in `initialVideo/` for the offline path, plus a live webcam/mic smoke test for the realtime path, comparing before/after event timelines.

## Risks / open questions

- `mouth_motion_score`'s variance-window derivation is a new heuristic with no ground truth in this codebase yet; like the existing `VOICE_MATCH_THRESHOLD`/`VOICE_CLUSTER_DISTANCE_THRESHOLD` comments, its weight is a starting point to be tuned against real recordings, not a validated value.
- `AUDIO_SYNC_TOLERANCE_MS` interacts with `RealtimeVAD`'s own internal smoothing (`REALTIME_SPEECH_CONFIRM_FRAMES`/`REALTIME_SILENCE_CONFIRM_FRAMES`); if realtime speaker switching still feels laggy after implementation, the fix is tuning these two independently, not adding a second synchronization mechanism.
- This feature does not attempt overlapping-speaker (cross-talk) attribution — one active speaker at a time, matching the Non-Goals section.
