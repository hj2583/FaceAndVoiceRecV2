"""Diagnostic report for offline speaker diarization.

Usage:
    python evaluate_speaker_recognition.py meeting.mp4

This is a diagnostic report, not an accuracy benchmark. Ground-truth labels
are required before reporting identification accuracy or diarization error.
"""

import argparse
import json
import tempfile
import wave
from pathlib import Path

import config
from audio_core import detect_speech_segments
from transcription_core import extract_audio_from_video
from voice_core import diarize_meeting_audio


def _duration(regions):
    return sum(max(0.0, end - start) for start, end in regions)


def main():
    parser = argparse.ArgumentParser(description="Report speaker-recognition diagnostics")
    parser.add_argument("video", type=Path, help="meeting video to analyze")
    args = parser.parse_args()

    if not args.video.exists():
        parser.error(f"Video not found: {args.video}")

    with tempfile.TemporaryDirectory(prefix="speaker_eval_") as temp_dir:
        audio_path = Path(temp_dir) / "audio.wav"
        diagnostics_path = Path(temp_dir) / "speaker_diagnostics.json"
        original_enabled = config.VOICE_DIAGNOSTICS_ENABLED
        original_path = config.VOICE_DIAGNOSTICS_PATH
        config.VOICE_DIAGNOSTICS_ENABLED = True
        config.VOICE_DIAGNOSTICS_PATH = diagnostics_path
        try:
            extract_audio_from_video(args.video, audio_path)
            with wave.open(str(audio_path), "rb") as audio_file:
                audio_duration = audio_file.getnframes() / audio_file.getframerate()
            speech_regions = detect_speech_segments(audio_path)
            diarization = diarize_meeting_audio(audio_path, speech_regions)
            report = json.loads(diagnostics_path.read_text(encoding="utf-8")) if diagnostics_path.exists() else {"summary": {}, "windows": []}
        finally:
            config.VOICE_DIAGNOSTICS_ENABLED = original_enabled
            config.VOICE_DIAGNOSTICS_PATH = original_path

    windows = report.get("windows", [])
    summary = report.get("summary", {})
    labels = [label for label, _start, _end in diarization]
    switches = sum(previous != current for previous, current in zip(labels, labels[1:]))
    uncertain = sum(
        window.get("final_speaker", "").startswith("Unknown")
        for window in windows
        if window.get("accepted")
    )
    print("SPEAKER RECOGNITION REPORT")
    print(f"Audio duration: {audio_duration:.2f}s")
    print(f"Speech duration: {_duration(speech_regions):.2f}s")
    print(f"Number of embeddings: {summary.get('accepted_embeddings', 0)}")
    print(f"Rejected embeddings: {summary.get('rejected_embeddings', 0)}")
    print(f"Average embedding quality: {summary.get('average_embedding_quality', 0.0):.3f}")
    print(f"Number of clusters: {summary.get('cluster_count', 0)}")
    print(f"Cluster sizes: {summary.get('cluster_sizes', {})}")
    print(f"Speaker switches: {switches}")
    print(f"Uncertain windows: {uncertain}")
    print(f"Speaker intervals: {len(diarization)}")
    print("Speakers:")
    for label in sorted(set(labels)):
        print(f"  {label}")
    print("\nThis report is diagnostic only; no accuracy percentage is reported without ground truth.")


if __name__ == "__main__":
    main()
