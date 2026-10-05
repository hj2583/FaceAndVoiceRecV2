from subtitle_renderer import (
    SubtitleCue,
    SubtitleWord,
    VideoMuxError,
    build_subtitle_cues,
    mux_tracked_video,
    write_ass_subtitles,
)
from pathlib import Path
from types import SimpleNamespace
import pytest
import config


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


def test_build_subtitle_cues_sorts_unsorted_input():
    """Verify words are sorted by start_ms before processing."""
    words = [
        SubtitleWord("Alice", 600, 800, "world"),
        SubtitleWord("Alice", 0, 250, "Hello"),
        SubtitleWord("Alice", 260, 500, "there"),
    ]
    cues = build_subtitle_cues(words, max_gap_ms=800, max_chars_per_line=42, max_lines=2)
    # After sorting by start_ms: [Hello, there, world]
    assert len(cues) == 1
    assert cues[0].lines == ("Alice: Hello there world",)
    assert cues[0].start_ms == 0
    assert cues[0].end_ms == 800


def test_build_subtitle_cues_with_none_defaults():
    """Verify None defaults use config values."""
    words = [
        SubtitleWord("Alice", 0, 250, "Hello"),
        SubtitleWord("Alice", 260, 500, "there"),
    ]
    # Call with None should use config defaults
    cues = build_subtitle_cues(words, max_gap_ms=None, max_chars_per_line=None, max_lines=None)
    # Should succeed without error and use config.SUBTITLE_MAX_GAP_MS, etc.
    assert len(cues) == 1
    assert cues[0].speaker_label == "Alice"


def test_build_subtitle_cues_discards_empty_text_words():
    """Verify empty-text words are discarded."""
    words = [
        SubtitleWord("Alice", 0, 250, "Hello"),
        SubtitleWord("Alice", 260, 500, ""),  # Empty text
        SubtitleWord("Alice", 510, 750, "there"),
    ]
    cues = build_subtitle_cues(words, max_gap_ms=800, max_chars_per_line=42, max_lines=2)
    # Empty word should be skipped
    assert len(cues) == 1
    assert "Alice: Hello there" in cues[0].lines[0]


def test_build_subtitle_cues_single_long_word_does_not_overflow_max_lines():
    """Verify a single long word doesn't create extra lines beyond max_lines."""
    words = [
        SubtitleWord("Alice", 0, 250, "Supercalifragilisticexpialidocious"),
    ]
    cues = build_subtitle_cues(words, max_gap_ms=800, max_chars_per_line=20, max_lines=2)
    assert len(cues) == 1
    cue = cues[0]
    # Should have exactly 2 lines (max_lines), not 3+
    assert len(cue.lines) <= 2
    # First line should have speaker prefix
    assert cue.lines[0].startswith("Alice: ")


def test_build_subtitle_cues_starts_new_cue_before_exceeding_max_lines():
    """Verify cue breaks before adding word that would exceed max lines."""
    words = [
        SubtitleWord("Alice", i * 250, i * 250 + 200, word)
        for i, word in enumerate(["short"] * 8)  # 8 short words
    ]
    cues = build_subtitle_cues(words, max_gap_ms=800, max_chars_per_line=20, max_lines=2)
    # With short words and max_lines=2, should create multiple cues
    assert len(cues) > 1
    for cue in cues:
        assert len(cue.lines) <= 2


def test_ass_writer_escapes_backslash_in_speaker_label(tmp_path):
    """Verify ASS control characters are escaped in speaker labels."""
    path = tmp_path / "test_escape_speaker.ass"
    cues = [
        SubtitleCue("Alice\\N", 100, 200, ("Alice\\N: Hello",)),
    ]
    write_ass_subtitles(cues, path)
    contents = path.read_text(encoding="utf-8")
    # Backslash should be escaped as \\
    assert "Alice\\\\N" in contents


def test_ass_writer_escapes_special_chars_in_text(tmp_path):
    """Verify ASS control characters are escaped in text content."""
    path = tmp_path / "test_escape_text.ass"
    cues = [
        SubtitleCue("Alice", 100, 200, ("Test \\c&H00FF00& colored text",)),
    ]
    write_ass_subtitles(cues, path)
    contents = path.read_text(encoding="utf-8")
    # Backslash should be escaped
    assert "\\\\c&H00FF00&" in contents


def test_ass_writer_escapes_braces_in_text(tmp_path):
    """Verify ASS control characters like braces are escaped."""
    path = tmp_path / "test_escape_braces.ass"
    cues = [
        SubtitleCue("Alice", 100, 200, ("Test {\\an8} alignment override",)),
    ]
    write_ass_subtitles(cues, path)
    contents = path.read_text(encoding="utf-8")
    # Braces should be escaped as \{ and \}
    # Verify the exact escaped sequence is present
    assert "\\{\\\\an8\\}" in contents


def test_oversized_first_word_stays_on_line_one_with_prefix():
    """Verify a single oversized first word stays on line 1 with speaker prefix.
    
    The spec requires: "Keep speaker prefix and oversized first word together on 
    line 1, even when it exceeds the character target."
    """
    words = [
        SubtitleWord("Alice", 0, 250, "Supercalifragilisticexpialidocious"),
        SubtitleWord("Alice", 260, 500, "there"),
    ]
    cues = build_subtitle_cues(words, max_gap_ms=800, max_chars_per_line=20, max_lines=2)
    assert len(cues) == 1
    cue = cues[0]
    # First line should have speaker prefix + oversized word, even if exceeding target
    assert cue.lines[0].startswith("Alice: Supercalifragilisticexpialidocious")
    # "there" should be on line 2
    assert len(cue.lines) >= 2
    assert "there" in cue.lines[1]


def test_max_lines_one_with_oversized_first_word():
    """Verify max_lines=1 with oversized first word retains the word.
    
    When max_lines=1, the oversized first word should still be consumed,
    not dropped by _get_cue_words.
    """
    words = [
        SubtitleWord("Alice", 0, 250, "Supercalifragilisticexpialidocious"),
        SubtitleWord("Alice", 260, 500, "there"),
    ]
    cues = build_subtitle_cues(words, max_gap_ms=800, max_chars_per_line=20, max_lines=1)
    # Should create a cue with the first word, even if oversized
    assert len(cues) >= 1
    assert "Supercalifragilisticexpialidocious" in cues[0].lines[0]


def test_mux_tracked_video_retries_copy_to_aac_and_replaces_output(tmp_path, monkeypatch):
    annotated_video = tmp_path / "annotated.mp4"
    source_video = tmp_path / "source.mp4"
    subtitle_path = tmp_path / "subs" / "subtitles.ass"
    output_path = tmp_path / "final.mp4"
    subtitle_path.parent.mkdir(parents=True, exist_ok=True)
    annotated_video.write_bytes(b"annotated")
    source_video.write_bytes(b"source")
    subtitle_path.write_text("dummy", encoding="utf-8")

    calls = []

    def fake_run(args, **kwargs):
        calls.append({
            "args": list(args),
            "cwd": kwargs.get("cwd"),
            "shell": kwargs.get("shell"),
            "output_exists_before": output_path.exists(),
        })
        if len(calls) == 1:
            return SimpleNamespace(returncode=1, stderr="copy codec unsupported")
        Path(args[-1]).write_bytes(b"muxed")
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr("subtitle_renderer.subprocess.run", fake_run)

    result = mux_tracked_video(
        annotated_video=annotated_video,
        source_video=source_video,
        output_path=output_path,
        subtitle_path=subtitle_path,
        ffmpeg="ffmpeg",
    )

    assert result == output_path
    assert output_path.read_bytes() == b"muxed"
    assert len(calls) == 2
    assert all(not call["output_exists_before"] for call in calls)

    first_args = calls[0]["args"]
    second_args = calls[1]["args"]

    assert first_args[:2] == ["ffmpeg", "-y"]
    assert first_args[2:6] == ["-i", str(annotated_video), "-i", str(source_video)]
    assert second_args[2:6] == ["-i", str(annotated_video), "-i", str(source_video)]

    assert "-map" in first_args
    first_map_positions = [i for i, token in enumerate(first_args) if token == "-map"]
    assert first_args[first_map_positions[0] + 1] == "0:v:0"
    assert first_args[first_map_positions[1] + 1] == "1:a:0?"

    assert first_args[first_args.index("-vf") + 1] == "subtitles=filename=subtitles.ass"
    assert Path(calls[0]["cwd"]) == subtitle_path.parent
    assert Path(calls[1]["cwd"]) == subtitle_path.parent
    assert calls[0]["shell"] in (None, False)
    assert calls[1]["shell"] in (None, False)

    assert first_args[first_args.index("-c:a") + 1] == "copy"
    assert second_args[second_args.index("-c:a") + 1] == "aac"
    assert "-b:a" in second_args
    assert second_args[second_args.index("-b:a") + 1] == "192k"

    assert Path(first_args[-1]) != output_path
    assert Path(second_args[-1]) != output_path


def test_mux_tracked_video_raises_and_preserves_existing_output_on_double_failure(tmp_path, monkeypatch):
    annotated_video = tmp_path / "annotated.mp4"
    source_video = tmp_path / "source.mp4"
    output_path = tmp_path / "final.mp4"
    annotated_video.write_bytes(b"annotated")
    source_video.write_bytes(b"source")
    output_path.write_bytes(b"original")

    calls = []

    def fake_run(args, **kwargs):
        calls.append(list(args))
        temp_path = Path(args[-1])
        temp_path.write_bytes(b"partial")
        return SimpleNamespace(returncode=1, stderr="mux failed")

    monkeypatch.setattr("subtitle_renderer.subprocess.run", fake_run)

    with pytest.raises(VideoMuxError):
        mux_tracked_video(
            annotated_video=annotated_video,
            source_video=source_video,
            output_path=output_path,
            subtitle_path=None,
            ffmpeg="ffmpeg",
        )

    assert len(calls) == 2
    assert output_path.read_bytes() == b"original"
    temp_candidates = list(tmp_path.glob(f".{output_path.stem}.*.tmp{output_path.suffix}"))
    assert temp_candidates == []


def test_mux_tracked_video_without_subtitles_omits_vf_and_maps_optional_audio(tmp_path, monkeypatch):
    annotated_video = tmp_path / "annotated.mp4"
    source_video = tmp_path / "source.mp4"
    output_path = tmp_path / "final.mp4"
    annotated_video.write_bytes(b"annotated")
    source_video.write_bytes(b"source")

    captured_args = []

    def fake_run(args, **kwargs):
        captured_args.extend(args)
        Path(args[-1]).write_bytes(b"muxed")
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr("subtitle_renderer.subprocess.run", fake_run)

    mux_tracked_video(
        annotated_video=annotated_video,
        source_video=source_video,
        output_path=output_path,
        subtitle_path=None,
        ffmpeg="ffmpeg",
    )

    assert "-vf" not in captured_args
    map_positions = [i for i, token in enumerate(captured_args) if token == "-map"]
    assert captured_args[map_positions[0] + 1] == "0:v:0"
    assert captured_args[map_positions[1] + 1] == "1:a:0?"

