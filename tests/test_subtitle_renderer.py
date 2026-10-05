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
import subprocess
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
    assert "PlayResX: 1920" in contents
    assert "PlayResY: 1080" in contents
    assert "WrapStyle: 2" in contents
    assert "Style: Default,Arial,42," in contents


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
    """Verify literal backslashes cannot trigger ASS control sequences."""
    path = tmp_path / "test_escape_speaker.ass"
    cues = [
        SubtitleCue("Alice\\N", 100, 200, ("Alice\\N: Hello",)),
    ]
    write_ass_subtitles(cues, path)
    contents = path.read_text(encoding="utf-8")
    assert "Alice\\" + chr(0x2060) + "N" in contents


def test_ass_writer_escapes_special_chars_in_text(tmp_path):
    """Verify ASS control characters are neutralized in text content."""
    path = tmp_path / "test_escape_text.ass"
    cues = [
        SubtitleCue("Alice", 100, 200, ("Test \\c&H00FF00& colored text",)),
    ]
    write_ass_subtitles(cues, path)
    contents = path.read_text(encoding="utf-8")
    assert "\\" + chr(0x2060) + "c&H00FF00&" in contents


def test_ass_writer_escapes_braces_in_text(tmp_path):
    """Verify braces are rendered literally as full-width lookalikes."""
    path = tmp_path / "test_escape_braces.ass"
    cues = [
        SubtitleCue("Alice", 100, 200, ("Test {\\an8} alignment override",)),
    ]
    write_ass_subtitles(cues, path)
    contents = path.read_text(encoding="utf-8")
    left_brace = chr(0xFF5B)
    right_brace = chr(0xFF5D)
    assert f"Test {left_brace}\\{chr(0x2060)}an8{right_brace} alignment override" in contents


def test_ass_writer_normalizes_crlf_to_spaces(tmp_path):
    path = tmp_path / "test_newlines.ass"
    cues = [
        SubtitleCue("Alice", 100, 300, ("one\r\ntwo\nthree\rfour",)),
    ]

    write_ass_subtitles(cues, path)
    contents = path.read_text(encoding="utf-8")

    assert "Dialogue: 0,0:00:00.10,0:00:00.30,Default,,0,0,0,,one two three four" in contents


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


def test_mux_tracked_video_resolves_relative_paths_and_retries_only_for_codec_container_failure(tmp_path, monkeypatch):
    media_dir = tmp_path / "media"
    output_dir = tmp_path / "tracked"
    subtitle_dir = tmp_path / "subs"
    media_dir.mkdir()
    output_dir.mkdir()
    subtitle_dir.mkdir()

    annotated_video = media_dir / "annotated.mp4"
    source_video = media_dir / "source.mp4"
    subtitle_path = subtitle_dir / "subtitles.ass"
    output_path = output_dir / "final.mp4"
    annotated_video.write_bytes(b"annotated")
    source_video.write_bytes(b"source")
    subtitle_path.write_text("dummy", encoding="utf-8")

    monkeypatch.chdir(tmp_path)
    rel_annotated = Path("media") / "annotated.mp4"
    rel_source = Path("media") / "source.mp4"
    rel_subtitle = Path("subs") / "subtitles.ass"
    rel_output = Path("tracked") / "final.mp4"

    calls = []

    def fake_run(args, **kwargs):
        calls.append({"args": list(args), "kwargs": dict(kwargs)})
        if len(calls) == 1:
            return SimpleNamespace(
                returncode=1,
                stdout=b"",
                stderr=b"Could not find tag for codec pcm_s16le in stream #1, codec not currently supported in container",
            )
        Path(args[-1]).write_bytes(b"muxed")
        return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")

    monkeypatch.setattr("subtitle_renderer.subprocess.run", fake_run)

    result = mux_tracked_video(
        annotated_video=rel_annotated,
        source_video=rel_source,
        output_path=rel_output,
        subtitle_path=rel_subtitle,
        ffmpeg="ffmpeg",
    )

    assert result == rel_output.resolve()
    assert rel_output.read_bytes() == b"muxed"
    assert len(calls) == 2

    first_args = calls[0]["args"]
    second_args = calls[1]["args"]
    first_kwargs = calls[0]["kwargs"]
    second_kwargs = calls[1]["kwargs"]

    assert first_args[:2] == ["ffmpeg", "-y"]
    assert first_args[2:6] == ["-i", str(rel_annotated.resolve()), "-i", str(rel_source.resolve())]
    assert second_args[2:6] == ["-i", str(rel_annotated.resolve()), "-i", str(rel_source.resolve())]

    assert first_args[first_args.index("-vf") + 1] == "subtitles=filename=subtitles.ass"
    assert Path(first_kwargs["cwd"]) == rel_subtitle.resolve().parent
    assert Path(second_kwargs["cwd"]) == rel_subtitle.resolve().parent

    map_positions = [i for i, token in enumerate(first_args) if token == "-map"]
    assert first_args[map_positions[0] + 1] == "0:v:0"
    assert first_args[map_positions[1] + 1] == "1:a:0?"

    assert first_args[first_args.index("-c:a") + 1] == "copy"
    assert second_args[second_args.index("-c:a") + 1] == "aac"
    assert second_args[second_args.index("-b:a") + 1] == "192k"

    first_temp = Path(first_args[-1])
    second_temp = Path(second_args[-1])
    assert first_temp.parent != rel_output.resolve().parent
    assert second_temp.parent != rel_output.resolve().parent
    assert first_temp.parent.parent == rel_output.resolve().parent.parent
    assert second_temp.parent.parent == rel_output.resolve().parent.parent
    assert not first_temp.parent.exists()
    assert not second_temp.parent.exists()


def test_mux_tracked_video_does_not_retry_aac_for_non_codec_failures(tmp_path, monkeypatch):
    annotated_video = tmp_path / "annotated.mp4"
    source_video = tmp_path / "source.mp4"
    output_path = tmp_path / "final.mp4"
    annotated_video.write_bytes(b"annotated")
    source_video.write_bytes(b"source")

    calls = []

    def fake_run(args, **kwargs):
        calls.append(list(args))
        return SimpleNamespace(returncode=1, stdout=b"", stderr=b"No such file or directory")

    monkeypatch.setattr("subtitle_renderer.subprocess.run", fake_run)

    with pytest.raises(VideoMuxError, match="No such file or directory"):
        mux_tracked_video(
            annotated_video=annotated_video,
            source_video=source_video,
            output_path=output_path,
            subtitle_path=None,
            ffmpeg="ffmpeg",
        )

    assert len(calls) == 1


def test_mux_tracked_video_wraps_subprocess_oserror_and_cleans_temp_dir(tmp_path, monkeypatch):
    output_path = tmp_path / "tracked" / "final.mp4"
    annotated_video = tmp_path / "annotated.mp4"
    source_video = tmp_path / "source.mp4"
    annotated_video.write_bytes(b"annotated")
    source_video.write_bytes(b"source")

    temp_dirs = []

    def fake_run(args, **kwargs):
        temp_path = Path(args[-1])
        temp_dirs.append(temp_path.parent)
        temp_path.write_bytes(b"partial")
        raise FileNotFoundError("ffmpeg executable missing")

    monkeypatch.setattr("subtitle_renderer.subprocess.run", fake_run)

    with pytest.raises(VideoMuxError, match="ffmpeg executable missing"):
        mux_tracked_video(annotated_video, source_video, output_path, ffmpeg="ffmpeg")

    assert temp_dirs
    assert all(not temp_dir.exists() for temp_dir in temp_dirs)


def test_mux_tracked_video_wraps_replace_oserror_and_preserves_existing_output(tmp_path, monkeypatch):
    output_path = tmp_path / "tracked" / "final.mp4"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(b"original")
    annotated_video = tmp_path / "annotated.mp4"
    source_video = tmp_path / "source.mp4"
    annotated_video.write_bytes(b"annotated")
    source_video.write_bytes(b"source")

    temp_dirs = []

    def fake_run(args, **kwargs):
        temp_path = Path(args[-1])
        temp_dirs.append(temp_path.parent)
        temp_path.write_bytes(b"muxed")
        return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")

    def fake_replace(_src, _dst):
        raise PermissionError("destination file is in use")

    monkeypatch.setattr("subtitle_renderer.subprocess.run", fake_run)
    monkeypatch.setattr("subtitle_renderer.os.replace", fake_replace)

    with pytest.raises(VideoMuxError, match="destination file is in use"):
        mux_tracked_video(annotated_video, source_video, output_path, ffmpeg="ffmpeg")

    assert output_path.read_bytes() == b"original"
    assert temp_dirs
    assert all(not temp_dir.exists() for temp_dir in temp_dirs)


def test_mux_tracked_video_configures_binary_stderr_capture_and_decodes_utf8_replacement(tmp_path, monkeypatch):
    annotated_video = tmp_path / "annotated.mp4"
    source_video = tmp_path / "source.mp4"
    output_path = tmp_path / "final.mp4"
    annotated_video.write_bytes(b"annotated")
    source_video.write_bytes(b"source")

    kwargs_seen = {}

    def fake_run(args, **kwargs):
        kwargs_seen.update(kwargs)
        return SimpleNamespace(returncode=1, stdout=b"", stderr=b"invalid:\xff")

    monkeypatch.setattr("subtitle_renderer.subprocess.run", fake_run)

    with pytest.raises(VideoMuxError, match="invalid") as excinfo:
        mux_tracked_video(annotated_video, source_video, output_path, ffmpeg="ffmpeg")

    assert "\ufffd" in str(excinfo.value)
    assert kwargs_seen["check"] is False
    assert kwargs_seen["stdout"] == subprocess.PIPE
    assert kwargs_seen["stderr"] == subprocess.PIPE
    assert "text" not in kwargs_seen
    assert "encoding" not in kwargs_seen


def test_mux_tracked_video_missing_ffmpeg_raises(tmp_path, monkeypatch):
    annotated_video = tmp_path / "annotated.mp4"
    source_video = tmp_path / "source.mp4"
    output_path = tmp_path / "final.mp4"
    annotated_video.write_bytes(b"annotated")
    source_video.write_bytes(b"source")

    monkeypatch.setattr("subtitle_renderer.shutil.which", lambda _bin: None)

    with pytest.raises(VideoMuxError, match="FFmpeg was not found in PATH"):
        mux_tracked_video(annotated_video, source_video, output_path)


def test_mux_tracked_video_success_without_output_file_raises(tmp_path, monkeypatch):
    annotated_video = tmp_path / "annotated.mp4"
    source_video = tmp_path / "source.mp4"
    output_path = tmp_path / "final.mp4"
    annotated_video.write_bytes(b"annotated")
    source_video.write_bytes(b"source")

    def fake_run(args, **kwargs):
        return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")

    monkeypatch.setattr("subtitle_renderer.subprocess.run", fake_run)

    with pytest.raises(VideoMuxError, match="did not create a valid output file"):
        mux_tracked_video(annotated_video, source_video, output_path, ffmpeg="ffmpeg")


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
        return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")

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
    assert captured_args[captured_args.index("-c:v") + 1] == "copy"


def test_mux_tracked_video_double_failure_preserves_existing_output_and_cleans_temp_artifacts(tmp_path, monkeypatch):
    output_path = tmp_path / "tracked" / "final.mp4"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(b"original")
    annotated_video = tmp_path / "annotated.mp4"
    source_video = tmp_path / "source.mp4"
    annotated_video.write_bytes(b"annotated")
    source_video.write_bytes(b"source")

    calls = []
    temp_dirs = []

    def fake_run(args, **kwargs):
        calls.append(list(args))
        temp_path = Path(args[-1])
        temp_dirs.append(temp_path.parent)
        temp_path.write_bytes(b"attempt")
        if len(calls) == 1:
            return SimpleNamespace(
                returncode=1,
                stdout=b"",
                stderr=b"codec not currently supported in container",
            )
        return SimpleNamespace(returncode=1, stdout=b"", stderr=b"aac encode failed")

    monkeypatch.setattr("subtitle_renderer.subprocess.run", fake_run)

    with pytest.raises(VideoMuxError, match="aac encode failed"):
        mux_tracked_video(annotated_video, source_video, output_path, ffmpeg="ffmpeg")

    assert len(calls) == 2
    assert calls[0][calls[0].index("-c:a") + 1] == "copy"
    assert calls[1][calls[1].index("-c:a") + 1] == "aac"
    assert output_path.read_bytes() == b"original"
    assert temp_dirs
    assert all(not temp_dir.exists() for temp_dir in temp_dirs)


def test_mux_tracked_video_replaces_existing_output_only_after_successful_temp_render(tmp_path, monkeypatch):
    output_path = tmp_path / "tracked" / "final.mp4"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(b"original")
    annotated_video = tmp_path / "annotated.mp4"
    source_video = tmp_path / "source.mp4"
    annotated_video.write_bytes(b"annotated")
    source_video.write_bytes(b"source")

    bytes_seen_during_ffmpeg = []

    def fake_run(args, **kwargs):
        bytes_seen_during_ffmpeg.append(output_path.read_bytes())
        temp_path = Path(args[-1])
        temp_path.write_bytes(b"complete-new-output")
        return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")

    monkeypatch.setattr("subtitle_renderer.subprocess.run", fake_run)

    result = mux_tracked_video(annotated_video, source_video, output_path, ffmpeg="ffmpeg")

    assert result == output_path.resolve()
    assert bytes_seen_during_ffmpeg == [b"original"]
    assert output_path.read_bytes() == b"complete-new-output"


def test_mux_tracked_video_wraps_output_directory_creation_oserror(tmp_path, monkeypatch):
    output_path = tmp_path / "tracked" / "final.mp4"
    annotated_video = tmp_path / "annotated.mp4"
    source_video = tmp_path / "source.mp4"
    annotated_video.write_bytes(b"annotated")
    source_video.write_bytes(b"source")

    real_mkdir = Path.mkdir

    def fake_mkdir(self, *args, **kwargs):
        if self == output_path.parent:
            raise PermissionError("cannot create output directory")
        return real_mkdir(self, *args, **kwargs)

    monkeypatch.setattr("subtitle_renderer.Path.mkdir", fake_mkdir)

    with pytest.raises(VideoMuxError, match="cannot create output directory"):
        mux_tracked_video(annotated_video, source_video, output_path, ffmpeg="ffmpeg")

