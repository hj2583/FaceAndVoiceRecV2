import json
import shutil
import tempfile
from pathlib import Path
from typing import Callable

import config
from subtitle_renderer import (
    SubtitleWord,
    VideoMuxError,
    build_subtitle_cues,
    mux_tracked_video,
    write_ass_subtitles,
)
from transcription_core import build_face_event_diarizer, process_meeting_transcription
from video_processor import process_video_pipeline


class TranscriptStageError(RuntimeError):
    """Raised when transcription fails after video processing succeeded."""

    def __init__(self, video_output_path, cause, stage: str):
        self.video_output_path = Path(video_output_path)
        self.cause = cause
        self.stage = stage
        super().__init__(str(cause))


def _load_speech_events(log_path: Path) -> list[dict]:
    payload = json.loads(log_path.read_text(encoding="utf-8"))
    speech_events = payload.get("speech", [])
    if isinstance(speech_events, list):
        return speech_events
    return []


def _workflow_temp_dir(output_path: Path) -> Path:
    tracked_output_dir = output_path.parent
    sibling_root = tracked_output_dir.parent if tracked_output_dir.parent != tracked_output_dir else tracked_output_dir
    return Path(tempfile.mkdtemp(prefix=f".{tracked_output_dir.name}-workflow-", dir=str(sibling_root)))


def _try_audio_only_mux(annotated_video: Path, source_video: Path, output_path: Path):
    try:
        mux_tracked_video(annotated_video, source_video, output_path, subtitle_path=None)
        return None
    except Exception as mux_error:
        return mux_error


def _copy_annotated_to_output(annotated_video: Path, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(annotated_video, output_path)


def _report_stage(stage_callback: Callable[[str], None] | None, message: str) -> None:
    if stage_callback is not None:
        stage_callback(message)


def _raise_audio_mux_error(output_path: Path, primary_error: Exception, audio_mux_error: Exception) -> None:
    combined_error = RuntimeError(
        f"{primary_error} | audio-only mux failed: {audio_mux_error}"
    )
    raise TranscriptStageError(output_path, combined_error, stage="audio_mux") from audio_mux_error


def process_video_and_transcribe(
    video_path,
    output_path,
    log_path,
    progress_callback=None,
    stage_callback: Callable[[str], None] | None = None,
) -> int:
    """Run video face processing, then transcribe using face-attributed speech events."""
    video_path = Path(video_path)
    output_path = Path(output_path)
    log_path = Path(log_path)
    temp_dir = _workflow_temp_dir(output_path)
    annotated_video_path = temp_dir / "annotated.mp4"
    subtitle_path = temp_dir / "subtitles.ass"

    try:
        _report_stage(stage_callback, "Processing faces and speaker events")
        process_video_pipeline(
            video_path,
            annotated_video_path,
            log_path,
            progress_callback=progress_callback,
        )

        _report_stage(stage_callback, "Transcribing and timing subtitles")
        subtitle_words: list[SubtitleWord] = []
        try:
            speech_events = _load_speech_events(log_path)
            diarize = build_face_event_diarizer(speech_events)
            meeting_id = process_meeting_transcription(
                video_path,
                diarize=diarize,
                minimum_speaker_overlap=config.SPEAKER_FACE_OVERLAP_THRESHOLD,
                subtitle_word_callback=subtitle_words.append,
            )
            cues = build_subtitle_cues(subtitle_words)
            write_ass_subtitles(cues, subtitle_path)
        except Exception as transcription_error:
            audio_mux_error = _try_audio_only_mux(annotated_video_path, video_path, output_path)
            if audio_mux_error is None:
                raise TranscriptStageError(
                    output_path,
                    transcription_error,
                    stage="transcription",
                ) from transcription_error
            _copy_annotated_to_output(annotated_video_path, output_path)
            _raise_audio_mux_error(output_path, transcription_error, audio_mux_error)

        _report_stage(stage_callback, "Burning subtitles and restoring audio")
        try:
            mux_tracked_video(
                annotated_video_path,
                video_path,
                output_path,
                subtitle_path=subtitle_path,
            )
        except VideoMuxError as subtitle_error:
            audio_mux_error = _try_audio_only_mux(annotated_video_path, video_path, output_path)
            if audio_mux_error is None:
                raise TranscriptStageError(
                    output_path,
                    subtitle_error,
                    stage="subtitle",
                ) from subtitle_error
            _copy_annotated_to_output(annotated_video_path, output_path)
            _raise_audio_mux_error(output_path, subtitle_error, audio_mux_error)

        return meeting_id
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
