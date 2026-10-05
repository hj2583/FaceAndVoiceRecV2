from subtitle_renderer import SubtitleCue, SubtitleWord, build_subtitle_cues, write_ass_subtitles


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
