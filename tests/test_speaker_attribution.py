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


def _run_alice_then_bob(attributor):
    """Alice speaks alone at t=1.0; from t=1.1 Bob appears with a moving
    mouth while Alice's mouth is still. Returns the first time Bob wins."""
    attributor.update_audio(1.0, True, 1.0)
    attributor.update_faces(1.0, [_observation(1, person_name="Alice", face_confidence=0.9)])
    switch_time = None
    for i in range(1, 11):
        t = round(1.0 + 0.1 * i, 2)
        attributor.update_audio(t, True, 1.0)
        result = attributor.update_faces(
            t,
            [
                _observation(1, person_name="Alice", face_confidence=0.9, lip_open_ratio=0.05),
                _observation(2, person_id=2, person_name="Bob", face_confidence=0.9,
                             lip_open_ratio=0.0 if i % 2 else 0.06),
            ],
        )
        if result["active_speaker"] == "Bob" and switch_time is None:
            switch_time = t
    return switch_time


def test_switches_after_sustained_margin_not_immediately():
    attributor = SpeakerAttributor()
    switch_time = _run_alice_then_bob(attributor)
    assert switch_time is not None
    assert switch_time > 1.1


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
    # Grace is measured from the last voiced timestamp, so a single silent
    # update past the grace window must already end the run.
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


_REASON_KEYS = ("voice_activity", "mouth_motion", "face_confidence", "temporal_consistency")


def test_reason_has_all_four_keys_on_held_incumbent_path():
    attributor = SpeakerAttributor()
    attributor.update_audio(1.0, True, 1.0)
    attributor.update_faces(
        1.0,
        [
            _observation(1, person_name="Alice", face_confidence=0.9, lip_open_ratio=0.05),
            _observation(2, person_id=2, person_name="Bob", face_confidence=0.1, lip_open_ratio=0.0),
        ],
    )

    attributor.update_audio(1.1, True, 1.0)
    result = attributor.update_faces(
        1.1,
        [
            _observation(1, person_name="Alice", face_confidence=0.9, lip_open_ratio=0.05),
            # Raw score briefly beats Alice's, but not by enough on the smoothed score to switch.
            _observation(2, person_id=2, person_name="Bob", face_confidence=1.0, lip_open_ratio=0.5),
        ],
    )
    assert result["active_speaker"] == "Alice"
    assert result["reason"]["held_incumbent"] is True
    for key in _REASON_KEYS:
        assert key in result["reason"]


def test_reason_has_all_four_keys_on_grace_path():
    attributor = SpeakerAttributor()
    attributor.update_audio(1.0, True, 1.0)
    attributor.update_faces(1.0, [_observation(1, person_name="Alice", face_confidence=0.9)])

    attributor.update_audio(1.1, False)
    result = attributor.update_faces(1.1, [_observation(1, person_name="Alice", face_confidence=0.9)])
    assert result["active_speaker"] == "Alice"
    assert result["reason"]["grace_period"] is True
    assert result["reason"]["voice_activity"] == 0.0
    for key in _REASON_KEYS:
        assert key in result["reason"]


def test_reason_has_all_four_keys_on_no_faces_path():
    attributor = SpeakerAttributor()
    attributor.update_audio(1.0, True, 0.9)
    result = attributor.update_faces(1.0, [])
    assert result["active_speaker"] == "UNKNOWN"
    assert result["reason"]["voice_activity"] == 0.9
    assert result["reason"]["mouth_motion"] == 0.0
    assert result["reason"]["face_confidence"] == 0.0
    for key in _REASON_KEYS:
        assert key in result["reason"]


def test_weak_incumbent_is_demoted_using_smoothed_score():
    attributor = SpeakerAttributor()
    attributor.update_audio(1.0, True, 1.0)
    attributor.update_faces(1.0, [_observation(1, person_name="Alice", face_confidence=0.9)])

    result = None
    t = 1.0
    for _ in range(30):
        t = round(t + 0.05, 2)
        attributor.update_audio(t, True, 0.0)
        result = attributor.update_faces(t, [_observation(1, person_name="Alice", face_confidence=0.05)])
        if result["active_speaker"] == "UNKNOWN":
            break

    assert result["active_speaker"] == "UNKNOWN"
    assert result["track_id"] is None
    assert any(event["speaker"] == "Alice" for event in attributor._closed_events)


def test_closed_event_confidence_is_mean_of_run_scores():
    attributor = SpeakerAttributor()
    attributor.update_audio(1.0, True, 1.0)
    attributor.update_faces(1.0, [_observation(1, person_name="Alice", face_confidence=0.9)])

    attributor.update_audio(1.1, True, 1.0)
    attributor.update_faces(1.1, [_observation(1, person_name="Alice", face_confidence=0.9)])

    grace_s = config.SPEAKER_GRACE_PERIOD_MS / 1000.0
    silent_t = round(1.1 + grace_s + 0.1, 2)
    attributor.update_audio(silent_t, False)
    attributor.update_faces(silent_t, [_observation(1, person_name="Alice", face_confidence=0.9)])

    assert len(attributor._closed_events) == 1
    event = attributor._closed_events[0]

    # Frame 1: Alice is not yet the incumbent (no temporal bonus).
    score_1 = config.VOICE_ACTIVITY_WEIGHT * 1.0 + config.FACE_CONFIDENCE_WEIGHT * 0.9
    # Frame 2: Alice is the incumbent (temporal bonus); lip ratio unchanged so mouth motion is 0.
    score_2 = (
        config.VOICE_ACTIVITY_WEIGHT * 1.0
        + config.FACE_CONFIDENCE_WEIGHT * 0.9
        + config.TEMPORAL_WEIGHT * 1.0
    )
    expected_mean = (score_1 + score_2) / 2
    assert abs(event["confidence"] - expected_mean) < 1e-6
