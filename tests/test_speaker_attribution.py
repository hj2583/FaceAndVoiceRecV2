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
