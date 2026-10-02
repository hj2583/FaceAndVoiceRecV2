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
    lip_ratios: deque = field(default_factory=lambda: deque(maxlen=config.SPEAKER_HISTORY_MAXLEN))
    scores: deque = field(default_factory=lambda: deque(maxlen=config.SPEAKER_HISTORY_MAXLEN))
    last_seen: float = 0.0


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
        self._min_mouth_motion_score = config.SPEAKER_MIN_MOUTH_MOTION_SCORE
        self._track_history_ttl_s = config.SPEAKER_TRACK_HISTORY_TTL_MS / 1000.0

        self._audio_samples: deque = deque(maxlen=config.SPEAKER_HISTORY_MAXLEN)
        self._track_history: dict[int, _TrackHistory] = {}

        self._current_speaker: Optional[str] = None
        self._current_track_id: Optional[int] = None
        self._current_person_id: Optional[int] = None
        self._current_open_since: Optional[float] = None
        self._current_last_face_at: Optional[float] = None
        self._current_confidence: float = 0.0
        self._current_run_sum: float = 0.0
        self._current_run_count: int = 0
        self._last_voice_at: Optional[float] = None
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
        history.last_seen = timestamp
        history.lip_ratios.append((timestamp, lip_open_ratio))
        window_start = timestamp - self._window_s
        recent = [ratio for ts, ratio in history.lip_ratios if ts >= window_start]
        if len(recent) < 2:
            return 0.0
        spread = stdev(recent)
        return min(spread / self._lip_motion_norm, 1.0)

    def _record_score(self, track_id, timestamp, score):
        history = self._track_history.setdefault(track_id, _TrackHistory())
        history.last_seen = timestamp
        history.scores.append((timestamp, score))

    def _prune_track_history(self, timestamp):
        window_start = timestamp - self._window_s
        stale_cutoff = timestamp - self._track_history_ttl_s
        stale_ids = []

        for track_id, history in self._track_history.items():
            while history.lip_ratios and history.lip_ratios[0][0] < window_start:
                history.lip_ratios.popleft()
            while history.scores and history.scores[0][0] < window_start:
                history.scores.popleft()
            if history.last_seen < stale_cutoff:
                stale_ids.append(track_id)

        for track_id in stale_ids:
            self._track_history.pop(track_id, None)

    def _has_mouth_evidence(self, observation, motion_score):
        if not observation.face_visible:
            return False
        return observation.mouth_open or motion_score >= self._min_mouth_motion_score

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
            has_mouth_evidence = self._has_mouth_evidence(observation, motion_score)
            is_incumbent = observation.track_id == self._current_track_id
            if voice_active and has_mouth_evidence:
                score = (
                    self._voice_weight * voice_confidence
                    + self._lip_weight * motion_score
                    + self._face_weight * observation.face_confidence
                    + self._temporal_weight * (1.0 if is_incumbent else 0.0)
                )
                # Silent or ineligible frames must not poison voiced smoothing.
                self._record_score(observation.track_id, timestamp, score)
            else:
                score = 0.0
            scored.append({
                "observation": observation,
                "score": score,
                "motion_score": motion_score,
                "eligible": has_mouth_evidence,
            })
        return scored

    def _score_candidates_from_timestamp(self, timestamp, face_observations):
        """Test/debug helper: look up audio the same way update_faces does."""
        audio = self._nearest_audio(timestamp)
        voice_active = bool(audio and audio.voice_active)
        voice_confidence = audio.voice_confidence if audio else 0.0
        return self._score_candidates(timestamp, voice_active, voice_confidence, face_observations)

    # ------------------------------------------------------------------
    # Decision / hysteresis
    # ------------------------------------------------------------------

    def update_faces(self, timestamp, face_observations):
        audio = self._nearest_audio(timestamp)
        voice_active = bool(audio and audio.voice_active)
        voice_confidence = audio.voice_confidence if audio else 0.0

        self._prune_track_history(timestamp)

        scored = self._score_candidates(timestamp, voice_active, voice_confidence, face_observations)

        if self.debug:
            self._log_debug(scored)

        if not voice_active:
            return self._handle_silence(timestamp, scored)

        self._last_voice_at = timestamp

        eligible_scored = [entry for entry in scored if entry["eligible"]]
        if not eligible_scored:
            return self._handle_voice_with_no_eligible_face(timestamp, voice_confidence, scored)

        return self._decide(timestamp, voice_confidence, eligible_scored)

    def _handle_silence(self, timestamp, scored):
        if self._current_speaker is None:
            return self._result(timestamp, None, None, 0.0, {})

        if self._last_voice_at is None or timestamp - self._last_voice_at >= self._grace_period_s:
            # The run ended when voice stopped, not when the grace window expired.
            self._close_current_event(self._last_voice_at if self._last_voice_at is not None else timestamp)
            return self._result(timestamp, None, None, 0.0, {})

        reason = self._build_reason(
            timestamp, self._current_track_id, 0.0, scored, {"grace_period": True},
        )
        return self._result(
            timestamp, self._current_speaker, self._current_track_id,
            self._current_confidence, reason,
        )

    def _open_or_continue_unknown_run(self, timestamp, voice_confidence, scored, extra_reason=None):
        frame_confidence = self._voice_weight * voice_confidence

        if self._current_speaker == "UNKNOWN" and self._current_track_id is None:
            self._current_run_sum += frame_confidence
            self._current_run_count += 1
            self._current_confidence = frame_confidence
            reason = self._build_reason(timestamp, None, voice_confidence, scored, extra_reason)
            return self._result(timestamp, "UNKNOWN", None, frame_confidence, reason)

        if self._current_speaker is not None:
            close_at = self._current_last_face_at if self._current_last_face_at is not None else timestamp
            self._close_current_event(close_at)

        self._current_speaker = "UNKNOWN"
        self._current_track_id = None
        self._current_person_id = None
        self._current_open_since = timestamp
        self._current_last_face_at = None
        self._current_run_sum = frame_confidence
        self._current_run_count = 1
        self._current_confidence = frame_confidence
        reason = self._build_reason(timestamp, None, voice_confidence, scored, extra_reason)
        return self._result(timestamp, "UNKNOWN", None, frame_confidence, reason)

    def _handle_voice_with_no_eligible_face(self, timestamp, voice_confidence, scored):
        if (
            self._current_track_id is not None
            and self._current_speaker is not None
            and self._current_last_face_at is not None
            and timestamp - self._current_last_face_at < self._grace_period_s
        ):
            reason = self._build_reason(
                timestamp,
                self._current_track_id,
                voice_confidence,
                scored,
                {"grace_period": True, "missing_face": True},
            )
            return self._result(
                timestamp,
                self._current_speaker,
                self._current_track_id,
                self._current_confidence,
                reason,
            )

        return self._open_or_continue_unknown_run(
            timestamp,
            voice_confidence,
            scored,
            {"no_eligible_face": True},
        )

    def _decide(self, timestamp, voice_confidence, scored):
        best = max(scored, key=lambda item: item["score"])
        observation = best["observation"]
        label = observation.person_name if observation.person_id is not None else "UNKNOWN"
        smoothed = self._smoothed_score(observation.track_id, timestamp)

        incumbent_id = self._current_track_id
        incumbent_smoothed = self._smoothed_score(incumbent_id, timestamp) if incumbent_id is not None else 0.0

        if incumbent_id is None:
            switch = best["score"] >= self._min_confidence
        elif observation.track_id == incumbent_id:
            switch = True
        else:
            switch = (
                smoothed >= self._min_confidence
                and smoothed - incumbent_smoothed >= self._switch_threshold
            )

        challenger_switching = switch and incumbent_id is not None and observation.track_id != incumbent_id

        if incumbent_id is not None and not challenger_switching and incumbent_smoothed < self._min_confidence:
            # Weak incumbent: demote on the smoothed score, not this frame's flicker.
            self._close_current_event(timestamp)
            reason = self._build_reason(timestamp, None, voice_confidence, scored)
            return self._result(timestamp, "UNKNOWN", None, 0.0, reason)

        if not switch:
            if self._current_speaker is not None:
                current_score = self._find_score(incumbent_id, scored)
                reported_confidence = current_score if current_score is not None else incumbent_smoothed
                if current_score is not None:
                    self._current_run_sum += current_score
                    self._current_run_count += 1
                self._current_confidence = reported_confidence
                if incumbent_id is not None and self._find_entry(incumbent_id, scored) is not None:
                    self._current_last_face_at = timestamp
                reason = self._build_reason(
                    timestamp, incumbent_id, voice_confidence, scored, {"held_incumbent": True},
                )
                return self._result(
                    timestamp, self._current_speaker, self._current_track_id,
                    reported_confidence, reason,
                )
            return self._open_or_continue_unknown_run(timestamp, voice_confidence, scored)

        if label != self._current_speaker or observation.track_id != self._current_track_id:
            if self._current_speaker is not None:
                self._close_current_event(timestamp)
            self._current_speaker = label
            self._current_track_id = observation.track_id
            self._current_person_id = observation.person_id if label != "UNKNOWN" else None
            self._current_open_since = timestamp
            self._current_last_face_at = timestamp
            self._current_run_sum = best["score"]
            self._current_run_count = 1
        else:
            self._current_run_sum += best["score"]
            self._current_run_count += 1
            self._current_last_face_at = timestamp

        self._current_confidence = best["score"]
        reason = self._build_reason(timestamp, observation.track_id, voice_confidence, scored)
        return self._result(timestamp, label, observation.track_id, best["score"], reason)

    def _find_entry(self, track_id, scored):
        for item in scored:
            if item["observation"].track_id == track_id:
                return item
        return None

    def _find_score(self, track_id, scored):
        entry = self._find_entry(track_id, scored)
        return entry["score"] if entry is not None else None

    def _build_reason(self, timestamp, track_id, voice_activity, scored, extra=None):
        entry = self._find_entry(track_id, scored)
        mouth_motion = entry["motion_score"] if entry is not None else 0.0
        face_confidence = entry["observation"].face_confidence if entry is not None else 0.0
        reason = {
            "voice_activity": voice_activity,
            "mouth_motion": mouth_motion,
            "face_confidence": face_confidence,
            "temporal_consistency": self._smoothed_score(track_id, timestamp),
        }
        if extra:
            reason.update(extra)
        return reason

    def _close_current_event(self, timestamp):
        if self._current_speaker is not None and self._current_open_since is not None:
            if self._current_run_count > 0:
                confidence = self._current_run_sum / self._current_run_count
            else:
                confidence = self._current_confidence
            if timestamp > self._current_open_since:
                self._closed_events.append({
                    "speaker": self._current_speaker,
                    "track_id": self._current_track_id,
                    "person_id": self._current_person_id,
                    "start_time": self._current_open_since,
                    "end_time": timestamp,
                    "confidence": confidence,
                })
        self._current_speaker = None
        self._current_track_id = None
        self._current_person_id = None
        self._current_open_since = None
        self._current_last_face_at = None
        self._current_confidence = 0.0
        self._current_run_sum = 0.0
        self._current_run_count = 0

    def _result(self, timestamp, speaker, track_id, confidence, reason):
        result = {"timestamp": timestamp, "active_speaker": speaker}
        if speaker is not None:
            result["track_id"] = track_id
            result["confidence"] = confidence
            result["run_start_time"] = self._current_open_since
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

    # ------------------------------------------------------------------
    # Events
    # ------------------------------------------------------------------

    def pop_closed_events(self):
        events, self._closed_events = self._closed_events, []
        return events

    def finish(self):
        """Close any still-open speaker run; call once when a pipeline ends."""
        if self._current_speaker is not None:
            self._close_current_event(self._last_voice_at)
