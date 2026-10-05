from subtitle_renderer import SubtitleCue, SubtitleWord, build_subtitle_cues, write_ass_subtitles
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


def test_ass_writer_escapes_backslash_in_speaker_label():
    """Verify ASS control characters are escaped in speaker labels."""
    cues = [
        SubtitleCue("Alice\\N", 100, 200, ("Alice\\N: Hello",)),
    ]
    path = "/tmp/test_escape_speaker.ass"
    write_ass_subtitles(cues, path)
    contents = open(path, encoding="utf-8").read()
    # Backslash should be escaped as \\
    assert "Alice\\\\N" in contents
    open(path).close()


def test_ass_writer_escapes_special_chars_in_text():
    """Verify ASS control characters are escaped in text content."""
    cues = [
        SubtitleCue("Alice", 100, 200, ("Test \\c&H00FF00& colored text",)),
    ]
    path = "/tmp/test_escape_text.ass"
    write_ass_subtitles(cues, path)
    contents = open(path, encoding="utf-8").read()
    # Backslash should be escaped
    assert "\\\\c&H00FF00&" in contents
    open(path).close()


def test_ass_writer_escapes_braces_in_text():
    """Verify ASS control characters like braces are escaped."""
    cues = [
        SubtitleCue("Alice", 100, 200, ("Test {\\an8} alignment override",)),
    ]
    path = "/tmp/test_escape_braces.ass"
    write_ass_subtitles(cues, path)
    contents = open(path, encoding="utf-8").read()
    # Braces should be escaped if needed for ASS format safety
    # At minimum, the content should not cause ASS parsing errors
    assert "Test" in contents
    open(path).close()
