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

        self._audio_samples: deque = deque(maxlen=config.SPEAKER_HISTORY_MAXLEN)
        self._track_history: dict[int, _TrackHistory] = {}

        self._current_speaker: Optional[str] = None
        self._current_track_id: Optional[int] = None
        self._current_open_since: Optional[float] = None
        self._current_confidence: float = 0.0
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

        self._last_voice_at = timestamp

        if not scored:
            if self._current_speaker is not None:
                self._close_current_event(timestamp)
            return self._result(timestamp, "UNKNOWN", None, 0.0, {})

        return self._decide(timestamp, voice_confidence, scored)

    def _handle_silence(self, timestamp):
        if self._current_speaker is None:
            return self._result(timestamp, None, None, 0.0, {})

        if timestamp - self._last_voice_at >= self._grace_period_s:
            # The run ended when voice stopped, not when the grace window expired.
            self._close_current_event(self._last_voice_at)
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
