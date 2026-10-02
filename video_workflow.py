import json
from pathlib import Path

import config
from transcription_core import build_face_event_diarizer, process_meeting_transcription
from video_processor import process_video_pipeline


class TranscriptStageError(RuntimeError):
    """Raised when transcription fails after video processing succeeded."""

    def __init__(self, video_output_path, cause):
        self.video_output_path = Path(video_output_path)
        self.cause = cause
        super().__init__(str(cause))


def _load_speech_events(log_path: Path) -> list[dict]:
    payload = json.loads(log_path.read_text(encoding="utf-8"))
    speech_events = payload.get("speech", [])
    if isinstance(speech_events, list):
        return speech_events
    return []


def process_video_and_transcribe(video_path, output_path, log_path, progress_callback=None) -> int:
    """Run video face processing, then transcribe using face-attributed speech events."""
    video_path = Path(video_path)
    output_path = Path(output_path)
    log_path = Path(log_path)

    process_video_pipeline(
        video_path,
        output_path,
        log_path,
        progress_callback=progress_callback,
    )

    try:
        speech_events = _load_speech_events(log_path)
        diarize = build_face_event_diarizer(speech_events)
        return process_meeting_transcription(
            video_path,
            diarize=diarize,
            minimum_speaker_overlap=config.SPEAKER_FACE_OVERLAP_THRESHOLD,
        )
    except Exception as error:
        raise TranscriptStageError(output_path, error) from error
