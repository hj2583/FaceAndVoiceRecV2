from dataclasses import dataclass
from typing import List, Tuple
from pathlib import Path


@dataclass
class SubtitleWord:
    """Represents a single word in a subtitle sequence."""
    speaker_label: str
    start_ms: int
    end_ms: int
    text: str


@dataclass
class SubtitleCue:
    """Represents a complete subtitle cue with timing and content."""
    speaker_label: str
    start_ms: int
    end_ms: int
    lines: Tuple[str, ...]


def build_subtitle_cues(words: List[SubtitleWord], max_gap_ms: int, max_chars_per_line: int, max_lines: int) -> List[SubtitleCue]:
    """Build subtitle cues from words, splitting on speaker changes and long gaps."""
    if not words:
        return []
    
    # Filter out invalid words (start_ms >= end_ms)
    valid_words = [w for w in words if w.start_ms < w.end_ms]
    if not valid_words:
        return []
    
    cues = []
    current_cue_words = []
    current_speaker = valid_words[0].speaker_label
    current_start = valid_words[0].start_ms
    
    for word in valid_words:
        # Check for speaker change or gap
        if word.speaker_label != current_speaker:
            # Finalize current cue and start new one
            if current_cue_words:
                cue = _create_cue(current_cue_words, current_speaker, current_start, max_chars_per_line, max_lines)
                if cue:
                    cues.append(cue)
            current_cue_words = [word]
            current_speaker = word.speaker_label
            current_start = word.start_ms
        elif current_cue_words and word.start_ms - current_cue_words[-1].end_ms > max_gap_ms:
            # Gap too large, finalize cue
            if current_cue_words:
                cue = _create_cue(current_cue_words, current_speaker, current_start, max_chars_per_line, max_lines)
                if cue:
                    cues.append(cue)
            current_cue_words = [word]
            current_start = word.start_ms
        else:
            current_cue_words.append(word)
    
    # Finalize last cue
    if current_cue_words:
        cue = _create_cue(current_cue_words, current_speaker, current_start, max_chars_per_line, max_lines)
        if cue:
            cues.append(cue)
    
    return cues


def _create_cue(words: List[SubtitleWord], speaker: str, start_ms: int, max_chars_per_line: int, max_lines: int) -> SubtitleCue:
    """Create a single subtitle cue from a list of words."""
    end_ms = words[-1].end_ms
    
    # Build lines with speaker prefix on first line
    lines = []
    current_line = f"{speaker}: " if speaker != "UNKNOWN" else f"{speaker}: "
    
    for word in words:
        test_line = current_line + word.text if current_line.endswith(": ") else current_line + " " + word.text
        
        if len(test_line) <= max_chars_per_line:
            current_line = test_line
        else:
            # Start new line
            if len(lines) < max_lines - 1 or (len(lines) == max_lines - 1 and len(current_line) > len(f"{speaker}: ")):
                lines.append(current_line)
                current_line = word.text
            else:
                # Max lines reached, append to last line
                current_line = current_line + " " + word.text
    
    if current_line:
        lines.append(current_line)
    
    return SubtitleCue(speaker, start_ms, end_ms, tuple(lines))


def write_ass_subtitles(cues: List[SubtitleCue], path) -> Path:
    """Write subtitles to an ASS (Advanced SubStation Alpha) format file."""
    path = Path(path)
    
    # ASS format header
    header = """[Script Info]
Title: Subtitles
ScriptType: v4.00+
Collisions: Normal
PlayDepth: 0
Timer: 100.0000
WrapStyle: 0

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,20,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,2,2,2,10,10,10,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    
    lines = [header]
    
    for cue in cues:
        start_time = _ms_to_ass_time(cue.start_ms)
        end_time = _ms_to_ass_time(cue.end_ms)
        text = "\\N".join(cue.lines)
        
        event = f"Dialogue: 0,{start_time},{end_time},Default,,0,0,0,,{text}"
        lines.append(event)
    
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _ms_to_ass_time(ms: int) -> str:
    """Convert milliseconds to ASS time format (H:MM:SS.CS)."""
    total_seconds = ms // 1000
    centiseconds = (ms % 1000) // 10
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    
    return f"{hours}:{minutes:02d}:{seconds:02d}.{centiseconds:02d}"
