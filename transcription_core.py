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
import re

import config
import voice_core
from audio_core import detect_speech_segments


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


def _detect_meeting_language(model, audio_path, speech_regions):
    """Aggregate Whisper language probabilities over speech, not just the opening."""
    try:
        import numpy as np
        import whisper

        audio = whisper.load_audio(str(audio_path))
        samples_per_window = int(config.WHISPER_LANGUAGE_SAMPLE_SECONDS * whisper.audio.SAMPLE_RATE)
        chunks = []
        regions = list(speech_regions or [])
        if len(regions) > config.WHISPER_LANGUAGE_MAX_SAMPLES:
            sample_indexes = np.linspace(
                0,
                len(regions) - 1,
                config.WHISPER_LANGUAGE_MAX_SAMPLES,
                dtype=int,
            )
            regions = [regions[index] for index in sample_indexes]

        for start, end in regions:
            start_sample = max(0, int(start * whisper.audio.SAMPLE_RATE))
            end_sample = min(len(audio), int(end * whisper.audio.SAMPLE_RATE))
            if end_sample <= start_sample:
                continue
            region = audio[start_sample:end_sample]
            if len(region) >= samples_per_window:
                chunks.append(region[:samples_per_window])
            elif len(region) >= whisper.audio.SAMPLE_RATE:
                chunks.append(region)
        if not chunks:
            return None

        probabilities = []
        for chunk in chunks:
            mel = whisper.log_mel_spectrogram(
                whisper.pad_or_trim(chunk),
                n_mels=model.dims.n_mels,
            ).to(model.device)
            _languages, language_probs = model.detect_language(mel)
            probabilities.append(language_probs)

        languages = sorted({language for probs in probabilities for language in probs})
        aggregate = {
            language: float(np.mean([
                probs.get(language, 0.0) for probs in probabilities
            ]))
            for language in languages
        }
        language, confidence = max(aggregate.items(), key=lambda item: item[1])
        logger.info(
            "Aggregated Whisper language detection: %s (confidence %.3f from %d samples)",
            language,
            confidence,
            len(probabilities),
        )
        if confidence < config.WHISPER_LANGUAGE_CONFIDENCE_THRESHOLD:
            return None
        return language
    except Exception:
        logger.exception("Whisper language detection failed; leaving language unconstrained")
        return None


def _speaker_for_interval(
    start_seconds: float,
    end_seconds: float,
    diarization: list[tuple[str, float, float]],
) -> str:
    overlap_by_speaker: dict[str, float] = {}
    for speaker, speaker_start, speaker_end in diarization:
        overlap = min(end_seconds, speaker_end) - max(start_seconds, speaker_start)
        if overlap > 0:
            overlap_by_speaker[str(speaker)] = (
                overlap_by_speaker.get(str(speaker), 0.0) + overlap
            )
    return max(overlap_by_speaker, key=overlap_by_speaker.get) if overlap_by_speaker else "Unknown Speaker"


def _split_segment_by_words(raw_segment, diarization):
    """Split a Whisper segment when timed words cross speaker intervals."""
    raw_text = str(raw_segment.get("text", "")).strip()
    words = raw_segment.get("words") or []
    timed_words = [
        word for word in words
        if word.get("word")
        and word.get("start") is not None
        and word.get("end") is not None
        and float(word["end"]) > float(word["start"])
    ]
    if not timed_words:
        return None

    # Some Whisper backends return partial or unreliable word timing data.
    # Never replace a complete ASR segment with a truncated word subset.
    raw_token_count = len(re.findall(r"\S+", raw_text))
    timed_token_count = len(re.findall(r"\S+", " ".join(
        str(word["word"]) for word in timed_words
    )))
    if raw_token_count and timed_token_count < max(1, int(raw_token_count * 0.8)):
        logger.warning(
            "Ignoring incomplete Whisper word timings: %s/%s tokens",
            timed_token_count,
            raw_token_count,
        )
        return None

    pieces = []
    current_speaker = None
    current_words = []
    for word in timed_words:
        word_start = float(word["start"])
        word_end = float(word["end"])
        speaker = _speaker_for_interval(word_start, word_end, diarization)
        if current_words and speaker != current_speaker:
            pieces.append((current_speaker, current_words[0][0], current_words[-1][1], "".join(current_words[i][2] for i in range(len(current_words)))))
            current_words = []
        current_speaker = speaker
        current_words.append((word_start, word_end, str(word["word"])))

    if current_words:
        pieces.append((current_speaker, current_words[0][0], current_words[-1][1], "".join(item[2] for item in current_words)))
    if len(pieces) < 2:
        return None
    return pieces


def transcribe_with_diarization(
    audio_path: str | Path,
    model_size: Optional[str] = None,
    diarize: Optional[Callable[[Path, list[tuple[float, float]]], Iterable[tuple[str, float, float]]]] = None,
    task: Optional[str] = None,
) -> list[TranscriptionSegment]:
    """Transcribe audio and assign speakers from an optional diarizer.

    ``diarize`` receives ``(audio_path, speech_regions)`` and must yield
    ``(speaker_label, start_seconds, end_seconds)``. When no diarizer is
    supplied, ``voice_core.diarize_meeting_audio`` is used by default; if
    diarization is unavailable or fails, the returned label is
    ``Unknown Speaker`` rather than a misleading speaker ID.
    """
    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(audio_path)

    try:
        import whisper
    except ImportError as error:
        raise RuntimeError("Install openai-whisper to transcribe audio") from error

    from audio_core import TORCH_DEVICE

    model = whisper.load_model(model_size or config.WHISPER_MODEL, device=TORCH_DEVICE)

    try:
        speech_regions = detect_speech_segments(audio_path)
    except Exception:
        # VAD unavailable/broken: degrade gracefully by skipping VAD
        # filtering entirely rather than failing the whole transcription.
        logger.exception("VAD speech detection failed; transcribing unfiltered")
        speech_regions = None

    active_task = task or config.WHISPER_TASK
    transcribe_kwargs = {
        "condition_on_previous_text": False,
        "word_timestamps": True,
        "task": active_task,
    }
    if config.WHISPER_LANGUAGE_MODE == "fixed" and config.WHISPER_LANGUAGE:
        transcribe_kwargs["language"] = config.WHISPER_LANGUAGE
    elif config.WHISPER_LANGUAGE_MODE == "auto":
        detected_language = _detect_meeting_language(
            model,
            audio_path,
            speech_regions,
        )
        if detected_language:
            transcribe_kwargs["language"] = detected_language
    if config.WHISPER_INITIAL_PROMPT:
        transcribe_kwargs["initial_prompt"] = config.WHISPER_INITIAL_PROMPT

    if speech_regions is not None and not speech_regions:
        return []

    result = model.transcribe(str(audio_path), verbose=False, **transcribe_kwargs)
    active_diarize = diarize if diarize is not None else voice_core.diarize_meeting_audio
    try:
        diarization = list(active_diarize(audio_path, speech_regions))
    except Exception:
        logger.exception("Diarization failed; falling back to Unknown Speaker labels")
        diarization = []
    segments = []

    for raw in result.get("segments", []):
        text = raw.get("text", "").strip()
        start_seconds = float(raw.get("start", 0.0))
        end_seconds = float(raw.get("end", 0.0))
        if not text or end_seconds <= start_seconds:
            continue

        # Whisper transcribes the whole file; drop segments that don't
        # overlap any VAD-detected speech region (silence hallucinations).
        # speech_regions is None when VAD was unavailable/failed, meaning
        # filtering is skipped and all Whisper segments pass through.
        if speech_regions is not None and not any(
            min(end_seconds, region_end) - max(start_seconds, region_start) > 0
            for region_start, region_end in speech_regions
        ):
            continue

        start_ms = int(start_seconds * 1000)
        end_ms = int(end_seconds * 1000)

        confidence = _segment_confidence(raw)
        word_pieces = _split_segment_by_words(raw, diarization) if diarization else None
        if word_pieces:
            segments.extend(
                TranscriptionSegment(
                    label,
                    int(piece_start * 1000),
                    int(piece_end * 1000),
                    piece_text.strip(),
                    confidence,
                )
                for label, piece_start, piece_end, piece_text in word_pieces
                if piece_text.strip()
            )
            continue

        label = _speaker_for_interval(start_seconds, end_seconds, diarization) if diarization else "Unknown Speaker"
        segments.append(TranscriptionSegment(label, start_ms, end_ms, text, confidence))
    return segments


def resolve_speaker_name(label):
    """Look up person_id for a speaker label that matches an enrolled person's name."""
    with sqlite3.connect(config.DB_PATH) as connection:
        row = connection.execute(
            "SELECT person_id FROM persons WHERE name=?",
            (label,),
        ).fetchone()
    return (int(row[0]) if row else None), label


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
            distinct_labels = {segment.speaker_label for segment in segments}
            # Resolve all distinct labels in one query instead of one connection per segment.
            person_ids_by_label: dict[str, int] = {}
            if distinct_labels:
                placeholders = ",".join("?" * len(distinct_labels))
                rows = connection.execute(
                    f"SELECT name, person_id FROM persons WHERE name IN ({placeholders})",
                    tuple(distinct_labels),
                ).fetchall()
                person_ids_by_label = {name: int(person_id) for name, person_id in rows}
            connection.executemany(
                """
                INSERT INTO transcription_segments(
                    meeting_id, speaker_label, person_id, start_ms, end_ms,
                    text, confidence, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                """,
                [
                    (
                        meeting_id,
                        segment.speaker_label,
                        person_ids_by_label.get(segment.speaker_label),
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
    diarize: Optional[Callable[[Path, list[tuple[float, float]]], Iterable[tuple[str, float, float]]]] = None,
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

        if config.WHISPER_TRANSLATION_ENABLED:
            translated_segments = transcribe_with_diarization(
                audio_path,
                model_size=config.WHISPER_MODEL,
                diarize=diarize,
                task="translate",
            )
            save_transcripts(
                translated_segments,
                meeting_dir / "english_translation",
            )
            logger.info(
                "Saved %d English translation segments for meeting %s",
                len(translated_segments),
                meeting_id,
            )

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