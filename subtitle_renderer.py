from dataclasses import dataclass
from typing import List, Tuple, Optional
from pathlib import Path
import os
import shutil
import subprocess
import tempfile
import uuid
import config


_WORD_JOINER = chr(0x2060)
_FULLWIDTH_LEFT_BRACE = chr(0xFF5B)
_FULLWIDTH_RIGHT_BRACE = chr(0xFF5D)


class VideoMuxError(RuntimeError):
    """Raised when FFmpeg subtitle burn-in and muxing cannot be safely finalized."""


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


def build_subtitle_cues(
    words: List[SubtitleWord],
    max_gap_ms: Optional[int] = None,
    max_chars_per_line: Optional[int] = None,
    max_lines: Optional[int] = None,
) -> List[SubtitleCue]:
    """Build subtitle cues from words, splitting on speaker changes and long gaps."""
    # Use config defaults if None provided
    if max_gap_ms is None:
        max_gap_ms = config.SUBTITLE_MAX_GAP_MS
    if max_chars_per_line is None:
        max_chars_per_line = config.SUBTITLE_MAX_CHARS_PER_LINE
    if max_lines is None:
        max_lines = config.SUBTITLE_MAX_LINES
    
    if not words:
        return []
    
    # Sort by start_ms
    sorted_words = sorted(words, key=lambda w: w.start_ms)
    
    # Filter out invalid words: start_ms >= end_ms or empty text
    valid_words = [w for w in sorted_words if w.start_ms < w.end_ms and w.text.strip()]
    if not valid_words:
        return []
    
    cues = []
    i = 0
    
    while i < len(valid_words):
        # Start a new cue with the current word
        current_speaker = valid_words[i].speaker_label
        current_start = valid_words[i].start_ms
        current_group = []
        
        # Gather words for this cue: same speaker, no long gaps
        while i < len(valid_words):
            word = valid_words[i]
            
            # Check speaker change
            if word.speaker_label != current_speaker:
                break
            
            # Check gap
            if current_group and word.start_ms - current_group[-1].end_ms > max_gap_ms:
                break
            
            current_group.append(word)
            i += 1
        
        # Now split current_group if it exceeds max_lines
        j = 0
        while j < len(current_group):
            # Get as many words as fit in max_lines
            group_for_cue, words_consumed = _get_cue_words(
                current_group[j:], current_speaker, max_chars_per_line, max_lines
            )
            
            if group_for_cue:
                cue_start = group_for_cue[0].start_ms
                cue_end = group_for_cue[-1].end_ms
                cue = _create_cue(group_for_cue, current_speaker, cue_start, max_chars_per_line, max_lines)
                if cue:
                    cues.append(cue)
            
            j += words_consumed
            if words_consumed == 0:
                # Safety: if we can't consume any words, break to avoid infinite loop
                break
    
    return cues


def _get_cue_words(
    words: List[SubtitleWord], speaker: str, max_chars_per_line: int, max_lines: int
) -> tuple:
    """Determine how many words fit within max_lines and return them.
    
    Special handling: the first word on the first line (with speaker prefix) is allowed
    to exceed max_chars_per_line, as per spec: "A single unusually long word may exceed
    the character target but must not create a third line."
    
    Returns (words_list, count_consumed).
    """
    if not words:
        return [], 0
    
    lines = []
    current_line = f"{speaker}: "
    words_used = 0
    is_first_word = True
    
    for i, word in enumerate(words):
        test_line = current_line + word.text if current_line.endswith(": ") else current_line + " " + word.text
        
        # Special case: first word is allowed to exceed max_chars_per_line
        if is_first_word and current_line.endswith(": "):
            # First word on the first line - always add it, even if it exceeds limit
            current_line = test_line
            words_used = i + 1
            is_first_word = False
        elif len(test_line) <= max_chars_per_line:
            # Word fits on current line
            current_line = test_line
            words_used = i + 1
        else:
            # Word doesn't fit on current line
            if len(lines) < max_lines - 1:
                # Can start a new line
                lines.append(current_line)
                current_line = word.text
                words_used = i + 1
                is_first_word = False
            else:
                # Can't add more lines - return words up to this point
                break
    
    # Return the words that fit
    return words[:words_used], words_used


def _create_cue(words: List[SubtitleWord], speaker: str, start_ms: int, max_chars_per_line: int, max_lines: int) -> SubtitleCue:
    """Create a single subtitle cue from a list of words.
    
    Special handling: the first word on the first line (with speaker prefix) is allowed
    to exceed max_chars_per_line, as per spec: "A single unusually long word may exceed
    the character target but must not create a third line."
    """
    if not words:
        return None
    
    end_ms = words[-1].end_ms
    
    # Build lines with speaker prefix on first line
    lines = []
    current_line = f"{speaker}: "
    is_first_word = True
    
    for word in words:
        test_line = current_line + word.text if current_line.endswith(": ") else current_line + " " + word.text
        
        # Special case: first word is allowed to exceed max_chars_per_line
        if is_first_word and current_line.endswith(": "):
            # First word on the first line - always add it, even if it exceeds limit
            current_line = test_line
            is_first_word = False
        elif len(test_line) <= max_chars_per_line:
            # Word fits on current line
            current_line = test_line
        else:
            # Word doesn't fit on current line
            if len(lines) < max_lines - 1:
                # Start new line
                lines.append(current_line)
                current_line = word.text
            else:
                # Can't fit more lines (shouldn't reach here due to _get_cue_words)
                break
    
    # Add the last line if it has content
    if current_line and current_line != f"{speaker}: ":
        lines.append(current_line)
    
    if not lines:
        return None
    
    return SubtitleCue(speaker, start_ms, end_ms, tuple(lines))


def _escape_ass_text(text: str) -> str:
    """Render transcript text literally and block ASS control-sequence injection."""
    normalized = str(text).replace("\r\n", "\n").replace("\r", "\n").replace("\n", " ")
    normalized = normalized.replace("\\", "\\" + _WORD_JOINER)
    normalized = normalized.replace("{", _FULLWIDTH_LEFT_BRACE)
    normalized = normalized.replace("}", _FULLWIDTH_RIGHT_BRACE)
    return normalized


def write_ass_subtitles(cues: List[SubtitleCue], path) -> Path:
    """Write subtitles to an ASS (Advanced SubStation Alpha) format file."""
    path = Path(path)
    
    # ASS format header
    header = """[Script Info]
Title: Subtitles
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,42,&H00FFFFFF,&H000000FF,&H00000000,&H64000000,0,0,0,0,100,100,0,0,1,2,0,2,40,40,40,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    
    lines = [header]
    
    for cue in cues:
        start_time = _ms_to_ass_time(cue.start_ms)
        end_time = _ms_to_ass_time(cue.end_ms)
        # Escape each line and join with ASS line break, then escape the final text
        escaped_lines = [_escape_ass_text(line) for line in cue.lines]
        text = "\\N".join(escaped_lines)
        
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


def unique_temp_sibling(output_path) -> tuple[Path, Path]:
    """Create a unique temp directory beside output folder and return (dir, temp_file)."""
    output_path = Path(output_path).resolve()
    tracked_output_dir = output_path.parent
    sibling_root = tracked_output_dir.parent if tracked_output_dir.parent != tracked_output_dir else tracked_output_dir
    temp_dir = Path(
        tempfile.mkdtemp(
            prefix=f".{tracked_output_dir.name}-subtitle-",
            dir=str(sibling_root),
        )
    )
    temp_output = temp_dir / f"{output_path.stem}.{uuid.uuid4().hex}.tmp{output_path.suffix}"
    return temp_dir, temp_output


def _decode_process_output(value) -> str:
    """Decode subprocess outputs consistently as UTF-8 with replacement."""
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _is_codec_container_failure(stderr_text: str) -> bool:
    """Return True only for codec/container copy incompatibility diagnostics."""
    message = stderr_text.lower()
    markers = (
        "could not find tag for codec",
        "codec not currently supported in container",
        "not currently supported in container",
    )
    return any(marker in message for marker in markers)


def _mux_args(ffmpeg, annotated_video, source_video, temp_output, subtitle_path, audio_codec) -> List[str]:
    """Build FFmpeg args for subtitle burn-in + source audio mux."""
    args = [ffmpeg, "-y", "-i", str(annotated_video), "-i", str(source_video)]
    if subtitle_path is not None:
        args.extend(["-vf", f"subtitles=filename={Path(subtitle_path).name}"])
    args.extend([
        "-map", "0:v:0",
        "-map", "1:a:0?",
        "-c:v", "libx264" if subtitle_path is not None else "copy",
        "-c:a", audio_codec,
    ])
    if audio_codec == "aac":
        args.extend(["-b:a", "192k"])
    args.extend(["-movflags", "+faststart", str(temp_output)])
    return args


def mux_tracked_video(annotated_video, source_video, output_path, subtitle_path=None, ffmpeg=None) -> Path:
    """Burn ASS subtitles into an annotated video and mux audio from source video."""
    output_path = Path(output_path).resolve()
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise VideoMuxError(str(error)) from error
    annotated_video = Path(annotated_video).resolve()
    source_video = Path(source_video).resolve()
    subtitle_file = Path(subtitle_path).resolve() if subtitle_path is not None else None

    ffmpeg_bin = ffmpeg or shutil.which("ffmpeg")
    if ffmpeg_bin is None:
        raise VideoMuxError("FFmpeg was not found in PATH")

    subtitle_cwd = str(subtitle_file.parent) if subtitle_file is not None else None

    last_error = ""
    attempts = ["copy"]
    while attempts:
        audio_codec = attempts.pop(0)
        temp_dir = None
        temp_output = None

        try:
            temp_dir, temp_output = unique_temp_sibling(output_path)
            args = _mux_args(
                ffmpeg_bin,
                annotated_video,
                source_video,
                temp_output,
                subtitle_file,
                audio_codec,
            )
            result = subprocess.run(
                args,
                cwd=subtitle_cwd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

            stderr_text = _decode_process_output(result.stderr)
            stdout_text = _decode_process_output(result.stdout)
            if result.returncode == 0 and temp_output.is_file():
                try:
                    os.replace(temp_output, output_path)
                except OSError as error:
                    raise VideoMuxError(str(error)) from error
                return output_path

            if result.returncode == 0 and not temp_output.is_file():
                last_error = "FFmpeg did not create a valid output file"
            else:
                last_error = (stderr_text or stdout_text or "FFmpeg muxing failed").strip()

            if audio_codec == "copy" and _is_codec_container_failure(stderr_text):
                attempts.append("aac")

        except OSError as error:
            raise VideoMuxError(str(error)) from error

        finally:
            if temp_output is not None:
                try:
                    temp_output.unlink(missing_ok=True)
                except OSError:
                    pass
            if temp_dir is not None:
                shutil.rmtree(temp_dir, ignore_errors=True)

    if not last_error:
        last_error = "FFmpeg muxing failed"
    raise VideoMuxError(last_error)
