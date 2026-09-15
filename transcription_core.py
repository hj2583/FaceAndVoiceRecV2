"""Offline meeting transcription primitives.

Whisper provides speech recognition and timestamps, but not diarization.
Callers can provide a diarization function that returns speaker intervals;
without one, segments are deliberately labeled ``Unknown Speaker``.
"""

from dataclasses import dataclass
import logging
from pathlib import Path
import shutil
import sqlite3
import subprocess
from typing import Callable, Iterable, Optional

import config


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TranscriptionSegment:
    speaker_label: str
    start_ms: int
    end_ms: int
    text: str
    confidence: Optional[float] = None


def extract_audio_from_video(video_path: str | Path, output_path: str | Path) -> Path:
    """Extract mono 16 kHz PCM audio with FFmpeg."""
    video_path = Path(video_path)
    output_path = Path(output_path)
    if not video_path.exists():
        raise FileNotFoundError(video_path)

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("FFmpeg was not found in PATH")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [
            ffmpeg, "-y", "-i", str(video_path), "-vn", "-ac", "1",
            "-ar", "16000", "-c:a", "pcm_s16le", str(output_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg audio extraction failed: {result.stderr[-2000:]}")
    if not output_path.exists():
        raise RuntimeError("FFmpeg completed without creating the audio file")
    return output_path


def _segment_confidence(segment: dict) -> Optional[float]:
    value = segment.get("avg_logprob")
    if value is None:
        return None
    # Map Whisper's negative log probability to a bounded, readable score.
    return max(0.0, min(1.0, 1.0 + float(value) / 2.0))


def transcribe_with_diarization(
    audio_path: str | Path,
    model_size: Optional[str] = None,
    diarize: Optional[Callable[[str], Iterable[tuple[str, float, float]]]] = None,
) -> list[TranscriptionSegment]:
    """Transcribe audio and assign speakers from an optional diarizer.

    ``diarize`` must yield ``(speaker_label, start_seconds, end_seconds)``.
    Plain Whisper has no speaker separation; when no diarizer is supplied the
    returned label is ``Unknown Speaker`` rather than a misleading speaker ID.
    """
    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(audio_path)

    try:
        import whisper
    except ImportError as error:
        raise RuntimeError("Install openai-whisper to transcribe audio") from error

    model = whisper.load_model(model_size or config.WHISPER_MODEL)
    result = model.transcribe(str(audio_path), verbose=False)
    diarization = list(diarize(str(audio_path))) if diarize else []
    segments = []

    for raw in result.get("segments", []):
        text = raw.get("text", "").strip()
        start_ms = int(float(raw.get("start", 0.0)) * 1000)
        end_ms = int(float(raw.get("end", 0.0)) * 1000)
        if not text or end_ms <= start_ms:
            continue

        label = "Unknown Speaker"
        if diarization:
            best_overlap = 0.0
            for speaker, speaker_start, speaker_end in diarization:
                overlap = max(
                    0.0,
                    min(raw["end"], speaker_end) - max(raw["start"], speaker_start),
                )
                if overlap > best_overlap:
                    best_overlap = overlap
                    label = str(speaker)

        segments.append(
            TranscriptionSegment(
                label, start_ms, end_ms, text, _segment_confidence(raw)
            )
        )
    return segments


def apply_text_cleanup(text: str, level: Optional[str] = None) -> str:
    """Apply dependency-free cleanup suitable for raw Whisper text."""
    if level or config.TRANSCRIPTION_CLEANUP_LEVEL == "none":
        if (level or config.TRANSCRIPTION_CLEANUP_LEVEL) == "none":
            return text
    return " ".join(text.split()).strip()


def save_transcripts(
    segments: Iterable[TranscriptionSegment],
    output_dir: str | Path = config.TRANSCRIPTS_DIR,
    meeting_id: Optional[int] = None,
) -> dict[str, Path]:
    """Write one timestamped UTF-8 text file per speaker label.

    When ``meeting_id`` is supplied, also replace that meeting's stored
    transcript segments in one database transaction.
    """
    segments = list(segments)
    output_dir = Path(output_dir)
    grouped: dict[str, list[TranscriptionSegment]] = {}
    for segment in segments:
        grouped.setdefault(segment.speaker_label, []).append(segment)

    paths = {}
    for label, speaker_segments in grouped.items():
        safe_label = "_".join(
            part for part in "".join(
                character if character.isalnum() else " "
                for character in label.lower()
            ).split()
            if part
        ) or "unknown_speaker"
        path = output_dir / f"{safe_label}.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as transcript_file:
            for segment in sorted(speaker_segments, key=lambda item: item.start_ms):
                total_seconds = segment.start_ms // 1000
                timestamp = f"{total_seconds // 60:02d}:{total_seconds % 60:02d}"
                transcript_file.write(f"[{timestamp}] {apply_text_cleanup(segment.text)}\n")
        paths[label] = path

    if meeting_id is not None:
        with sqlite3.connect(config.DB_PATH) as connection:
            connection.execute("PRAGMA foreign_keys=ON")
            connection.execute(
                "DELETE FROM transcription_segments WHERE meeting_id=?",
                (meeting_id,),
            )
            connection.executemany(
                """
                INSERT INTO transcription_segments(
                    meeting_id, speaker_label, start_ms, end_ms,
                    text, confidence, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                """,
                [
                    (
                        meeting_id,
                        segment.speaker_label,
                        segment.start_ms,
                        segment.end_ms,
                        apply_text_cleanup(segment.text),
                        segment.confidence,
                    )
                    for segment in segments
                ],
            )

    return paths


def process_meeting_transcription(
    video_path: str | Path,
    diarize: Optional[Callable[[str], Iterable[tuple[str, float, float]]]] = None,
) -> int:
    """Run the offline transcription pipeline for one meeting video."""
    if not config.ENABLE_TRANSCRIPTION:
        raise RuntimeError("Transcription is disabled in config")

    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(video_path)

    with sqlite3.connect(config.DB_PATH) as connection:
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute(
            """
            INSERT INTO meetings(video_path, created_at, transcription_status)
            VALUES (?, CURRENT_TIMESTAMP, 'in_progress')
            ON CONFLICT(video_path) DO UPDATE SET
                transcription_status='in_progress', error_log=NULL
            """,
            (str(video_path),),
        )
        meeting_id = connection.execute(
            "SELECT meeting_id FROM meetings WHERE video_path=?",
            (str(video_path),),
        ).fetchone()[0]

    meeting_dir = Path(config.TRANSCRIPTS_DIR) / str(meeting_id)
    audio_path = meeting_dir / "audio.wav"

    try:
        extract_audio_from_video(video_path, audio_path)
        segments = transcribe_with_diarization(
            audio_path,
            model_size=config.WHISPER_MODEL,
            diarize=diarize,
        )
        cleaned_segments = [
            TranscriptionSegment(
                segment.speaker_label,
                segment.start_ms,
                segment.end_ms,
                apply_text_cleanup(segment.text),
                segment.confidence,
            )
            for segment in segments
        ]
        save_transcripts(cleaned_segments, meeting_dir, meeting_id)

        with sqlite3.connect(config.DB_PATH) as connection:
            connection.execute(
                """
                UPDATE meetings
                SET processed_at=CURRENT_TIMESTAMP,
                    transcription_status='completed', error_log=NULL
                WHERE meeting_id=?
                """,
                (meeting_id,),
            )
        return int(meeting_id)
    except Exception as error:
        with sqlite3.connect(config.DB_PATH) as connection:
            connection.execute(
                """
                UPDATE meetings
                SET transcription_status='failed', error_log=?
                WHERE meeting_id=?
                """,
                (str(error), meeting_id),
            )
        raise