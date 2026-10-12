import pytest

import config
import transcript_refiner
from transcript_refiner import TranscriptRefinementError


def _segments(*items):
    return [
        {
            "segment_id": index,
            "speaker_label": speaker,
            "start_ms": start,
            "end_ms": end,
            "text": text,
        }
        for index, (speaker, start, end, text) in enumerate(items, start=1)
    ]


def _reply(target_segments, group_refs=None, text_by_id=None, recommendations=None):
    ids = [item["segment_id"] for item in target_segments]
    groups = []
    if group_refs is None:
        group_refs = [[segment_id] for segment_id in ids]
    for refs in group_refs:
        text = " ".join(
            (text_by_id or {}).get(
                segment_id,
                next(item["text"] for item in target_segments if item["segment_id"] == segment_id),
            )
            for segment_id in refs
        )
        groups.append({"source_segment_ids": refs, "text": text})
    return {
        "groups": groups,
        "speaker_change_recommendations": recommendations or [],
    }


def test_refiner_merges_adjacent_fragments_for_same_speaker(monkeypatch):
    items = _segments(
        ("Alice", 0, 900, "Hello"),
        ("Alice", 950, 1800, "there."),
    )
    monkeypatch.setattr(
        transcript_refiner,
        "_post_ollama_chat",
        lambda _payload: _reply(
            items,
            [[1, 2]],
            text_by_id={1: "Hello", 2: "there."},
        ),
    )

    result = transcript_refiner.refine_transcript(items)

    assert result.groups[0].source_segment_ids == (1, 2)
    assert result.groups[0].text == "Hello there."


def test_refiner_never_merges_different_speakers_or_unknown_with_known(monkeypatch):
    items = _segments(
        ("Alice", 0, 500, "Good morning."),
        ("UNKNOWN", 550, 900, "Hello."),
        ("Bob", 950, 1400, "Hi."),
    )
    monkeypatch.setattr(
        transcript_refiner,
        "_post_ollama_chat",
        lambda _payload: _reply(items, [[1, 2, 3]]),
    )

    with pytest.raises(TranscriptRefinementError, match="speaker change"):
        transcript_refiner.refine_transcript(items)


def test_unknown_fragment_between_known_speech_stays_unresolved(monkeypatch):
    items = _segments(
        ("Alice", 0, 500, "Good morning."),
        ("UNKNOWN", 550, 900, "Hello."),
        ("Alice", 950, 1400, "Welcome."),
    )
    monkeypatch.setattr(
        transcript_refiner,
        "_post_ollama_chat",
        lambda _payload: _reply(items),
    )

    result = transcript_refiner.refine_transcript(items)

    assert [group.source_segment_ids for group in result.groups] == [(1,), (2,), (3,)]
    assert [items[segment_id - 1]["speaker_label"] for group in result.groups
            for segment_id in group.source_segment_ids] == ["Alice", "UNKNOWN", "Alice"]


def test_refiner_rejects_overlapping_speech_merge(monkeypatch):
    items = _segments(
        ("Alice", 0, 800, "We agree"),
        ("Alice", 700, 1200, "today."),
    )
    monkeypatch.setattr(
        transcript_refiner,
        "_post_ollama_chat",
        lambda _payload: _reply(items, [[1, 2]]),
    )

    with pytest.raises(TranscriptRefinementError, match="overlap"):
        transcript_refiner.refine_transcript(items)


def test_refiner_preserves_multilingual_words_without_translation(monkeypatch):
    text = "Selamat pagi, semua."
    items = _segments(("UNKNOWN", 0, 1200, text))
    monkeypatch.setattr(
        transcript_refiner,
        "_post_ollama_chat",
        lambda _payload: _reply(items),
    )

    result = transcript_refiner.refine_transcript(items)

    assert result.groups[0].text == text


@pytest.mark.parametrize(
    "item,match",
    [
        ({"segment_id": 1, "speaker_label": "A", "start_ms": 1, "end_ms": 1, "text": "x"}, "invalid source"),
        ({"segment_id": 1, "speaker_label": "A", "start_ms": None, "end_ms": 5, "text": "x"}, "invalid source"),
    ],
)
def test_refiner_rejects_zero_duration_and_missing_timestamps(item, match):
    with pytest.raises(TranscriptRefinementError, match=match):
        transcript_refiner.refine_transcript([item])


def test_refiner_chunks_long_transcripts_with_context_and_no_duplicate_ids(monkeypatch):
    monkeypatch.setattr(config, "TRANSCRIPT_REFINEMENT_CHUNK_SEGMENTS", 10)
    items = _segments(*[
        ("Alice", index * 1100, index * 1100 + 900, f"Word {index}.")
        for index in range(25)
    ])
    calls = []

    def fake_chat(payload):
        import json

        data = json.loads(payload)
        calls.append(data)
        return _reply(data["target_segments"])

    monkeypatch.setattr(transcript_refiner, "_post_ollama_chat", fake_chat)

    result = transcript_refiner.refine_transcript(items)

    assert len(calls) == 3
    assert all("read_only_context_segments" in call for call in calls)
    source_ids = [
        segment_id
        for group in result.groups
        for segment_id in group.source_segment_ids
    ]
    assert source_ids == list(range(1, 26))


def test_refiner_rejects_omitted_duplicate_and_invented_words(monkeypatch):
    items = _segments(
        ("Alice", 0, 500, "Hello"),
        ("Alice", 600, 1000, "there."),
    )
    responses = iter([
        {"groups": [{"source_segment_ids": [1], "text": "Hello"}],
         "speaker_change_recommendations": []},
        {"groups": [{"source_segment_ids": [1, 1, 2], "text": "Hello there."}],
         "speaker_change_recommendations": []},
        {"groups": [{"source_segment_ids": [1, 2], "text": "Hello everyone."}],
         "speaker_change_recommendations": []},
    ])
    monkeypatch.setattr(
        transcript_refiner,
        "_post_ollama_chat",
        lambda _payload: next(responses),
    )

    for message in ("omitted", "invalid references", "changed spoken words"):
        with pytest.raises(TranscriptRefinementError, match=message):
            transcript_refiner.refine_transcript(items)


def test_refiner_rejects_recommendation_with_unknown_segment_id(monkeypatch):
    items = _segments(("Alice", 0, 500, "Hello."))
    monkeypatch.setattr(
        transcript_refiner,
        "_post_ollama_chat",
        lambda _payload: _reply(
            items,
            recommendations=[{"after_segment_id": 999, "reason": "Review audio."}],
        ),
    )

    with pytest.raises(TranscriptRefinementError, match="recommendation is invalid"):
        transcript_refiner.refine_transcript(items)


def test_refiner_requires_no_audio_file_and_does_not_change_words(monkeypatch):
    items = _segments(("UNKNOWN", 0, 800, "Apa khabar?"))
    monkeypatch.setattr(transcript_refiner, "_post_ollama_chat", lambda _payload: _reply(items))

    result = transcript_refiner.refine_transcript(items)

    assert result.groups[0].text == "Apa khabar?"
