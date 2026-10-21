"""Smoke tests for package-qualified storage and face-domain imports."""

import inspect

from face_voice_rec.domain.face.detection_core import detect_faces_tiled
from face_voice_rec.domain.face.face_backend import get_face_backend
from face_voice_rec.domain.face.face_core import FaceIndex
from face_voice_rec.domain.face.tracking import CentroidTracker
from face_voice_rec.domain.transcription.speaker_introductions import extract_self_introductions
from face_voice_rec.domain.transcription.subtitle_renderer import SubtitleWord
from face_voice_rec.domain.transcription.transcript_refiner import TranscriptRefinementError
from face_voice_rec.domain.transcription.transcription_core import process_meeting_transcription
from face_voice_rec.domain.voice.speaker_attribution import SpeakerAttributor
from face_voice_rec.domain.voice.voice_core import diarize_meeting_audio
from face_voice_rec.domain.voice.audio_core import RealtimeVAD
from face_voice_rec.storage.database import init_db


def test_storage_and_face_package_imports() -> None:
    """Verify representative storage and face symbols are importable via package paths."""
    assert inspect.isfunction(init_db)
    assert inspect.isclass(FaceIndex)
    assert inspect.isfunction(get_face_backend)
    assert inspect.isfunction(detect_faces_tiled)
    assert inspect.isclass(CentroidTracker)


def test_voice_and_transcription_package_imports() -> None:
    """Verify voice/transcription symbols are importable via package paths."""
    assert inspect.isclass(RealtimeVAD)
    assert inspect.isfunction(diarize_meeting_audio)
    assert inspect.isclass(SpeakerAttributor)
    assert inspect.isfunction(process_meeting_transcription)
    assert inspect.isclass(TranscriptRefinementError)
    assert inspect.isclass(SubtitleWord)
    assert inspect.isfunction(extract_self_introductions)
