# Face and Voice Recognition

A Streamlit application for face recognition, active-speaker estimation, and optional video transcription.

This project is built to run comfortably on CPU by default. If CUDA is available, the app can use it automatically. If not, it falls back to CPU instead of failing at startup.

## Overview

The system combines:

- UniFace SCRFD + ArcFace for face detection and embedding extraction
- UniFace FaceMesh for facial landmarks and mouth-motion measurements
- OpenCV for fallback detection and video handling
- SQLite for person and identity storage
- NumPy for face-index matching
- Silero VAD for speech activity detection
- OpenAI Whisper for optional transcription
- SpeechBrain ECAPA-TDNN for voice embeddings and speaker clustering
- Streamlit for the web UI

## Models and processing flow

No single model handles face recognition, speaker attribution, transcription,
and attendance. The application combines these models and processing stages:

| Component | Model or tool | Purpose |
|---|---|---|
| Face detection | UniFace SCRFD (larger model for uploaded videos, lighter model for realtime) | Detects face locations in video frames. |
| Face recognition | UniFace ArcFace | Converts a face into a 512-dimensional embedding and compares it with enrolled face embeddings using cosine similarity and a confidence threshold. |
| Facial landmarks | UniFace FaceMesh | Locates facial features, including the mouth, to measure mouth opening and motion. |
| Speech detection | Silero VAD | Finds speech regions in audio; it does not generate transcript text. |
| Transcription | OpenAI Whisper (`small`) | Converts extracted audio into text and timestamps. FFmpeg extracts audio from uploaded video. Whisper does not identify speakers. |
| Voice embeddings | SpeechBrain ECAPA-TDNN | Represents voice samples numerically; the project uses embeddings to cluster speech and attempt matching against enrolled voice samples. |
| Active-speaker estimate | Application logic combining VAD timing and FaceMesh lip motion | Estimates which visible face is speaking only when lip movement is measurable and competitive; face-recognition similarity and temporal continuity are not treated as evidence of speaker identity. |
| Identity storage and matching | SQLite and NumPy | Stores people and their embeddings, then compares face embeddings against enrolled samples. |
| Attendance | Application logic and SQLite | Records a known person when the video or realtime pipeline recognizes their face with sufficient confidence. |
| Optional transcript formatting | Local Ollama model (disabled by default) | Formats and groups existing transcript fragments while preserving the spoken words and speaker labels. It does not replace Whisper or determine speaker identities. |

Face and voice enrollment are independent: enroll face samples for visual
identity recognition, and enroll clean voice samples to support voice matching.
Voice activity or an unknown face alone is not enough to create a known-person
attendance record. Transcript and attendance data are stored in SQLite.

Voice identity matching compares ECAPA cosine similarity with quality-filtered
enrollment samples, deduplicates identical vectors, and checks both the best
profile score and the margin to the next person. Defaults require cosine
similarity of at least `VOICE_MATCH_THRESHOLD` (0.50), a lead of at least
`VOICE_AMBIGUITY_MARGIN` (0.05), and two supporting enrollment samples when
two or more independent samples are available. Automatic voice identity is
withheld for clusters shorter than `VOICE_MIN_IDENTITY_DURATION_SECONDS`
(3 seconds). If a visual active-speaker candidate conflicts with a supported
voice match, the result is left unresolved rather than forcing either label.
Cosine similarity and the lip-motion activity score are not calibrated
probabilities; the UI labels them as scores, not confidence percentages.
Optional candidate diagnostics can be enabled with
`VOICE_DIAGNOSTICS_ENABLED=1`; reports contain sample timing and identity-score
summaries but not audio content or full media paths.

### Multilingual transcription

Whisper uses `transcribe` mode so it retains the spoken language rather than
translating to English. The default language policy remains fixed English for
the project's English-dominant collection. For another dominant meeting
language, set `WHISPER_LANGUAGE_MODE=auto` before launching the app; the
pipeline samples multiple VAD speech regions and uses the detected meeting
language. `WHISPER_LANGUAGE` can set a fixed Whisper language code when the
mode is `fixed`. Automatic detection chooses one language for the meeting, so
heavily code-switched recordings may still need transcript review.

### Optional readable transcript refinement

Transcription always retains the detailed Whisper segments. A separate readable
version is built from those segments and can be formatted by a local Ollama
service. Refinement is disabled by default; the deterministic readable view
still works without Ollama.

In the UI, the readable view also presents continuous nearby speech as a
paragraph, even if its detailed source rows have different speaker labels. Such
a paragraph is marked **Mixed/uncertain attribution** and lists every source
speaker label; it does not reassign any segment. Use the Detailed view to
inspect the original labels and timestamps.

To enable local refinement, install and start Ollama, download a model, then
start Streamlit with these PowerShell environment settings:

```powershell
ollama pull qwen2.5:7b
$env:TRANSCRIPT_REFINEMENT_ENABLED = "1"
$env:TRANSCRIPT_REFINEMENT_URL = "http://localhost:11434"
$env:TRANSCRIPT_REFINEMENT_MODEL = "qwen2.5:7b"
streamlit run app.py
```

The URL and model can be changed for another Ollama endpoint or model. Only
enable a non-local endpoint if you intend to send transcript text to that
service. The model receives transcript text, not audio, and cannot verify or
correct spoken words; validation rejects changed, omitted, reordered, or
invented words and rejects grouping across speaker changes, overlaps, and long
pauses. Speaker-change suggestions are advisory and never change attribution.
Failures leave the detailed Whisper transcript intact and show a deterministic
readable transcript with an error status. The app stores the original Whisper
text, manually corrected detailed text, and readable output separately so they
can be compared.

## Features

- Realtime webcam recognition
- Uploaded-video processing
- Active-speaker estimation from voice + lip movement
- Unknown-face tracking and review
- Face database management
- Transcription support for video/audio content

## Requirements

- Python 3.10+ 64-bit
- FFmpeg installed and available on PATH
- A webcam for realtime mode
- A microphone for live audio/VAD analysis

## CPU setup

1. Create and activate a virtual environment:

```powershell
python -m venv venv
.\venv\Scripts\activate
python -m pip install --upgrade pip
```

2. Install the CPU requirements:

```powershell
pip install -r requirements.txt
```

3. Verify FFmpeg is installed:

```powershell
ffmpeg -version
```

4. Launch the app:

```powershell
streamlit run app.py
```

## Optional GPU setup

If your machine has NVIDIA CUDA support and you want to use the GPU-accelerated path, install the GPU variant instead:

```powershell
pip install -r requirements-gpu.txt
```

The project prefers CUDA automatically when it is available. If CUDA is missing or unavailable, it falls back to CPU mode. GPU acceleration covers both face recognition (ONNX Runtime) and the torch-based audio models (Whisper, Silero VAD, SpeechBrain voice embeddings) — `requirements-gpu.txt` pulls a CUDA-enabled build of `torch` from PyTorch's own package index for this reason. If you already have a CPU-only `torch` installed from `requirements.txt`, re-run `pip install -r requirements-gpu.txt` to replace it with the CUDA build.

## Project structure

```text
FaceAndVoiceRecV2/
├── app.py
├── audio_core.py
├── config.py
├── database.py
├── detection_core.py
├── face_backend.py
├── face_core.py
├── face_index.npz
├── faces.db
├── README.md
├── realtime.py
├── realtime_launcher.py
├── requirements.txt
├── requirements-gpu.txt
├── tracking.py
├── transcription_core.py
├── transcript_refiner.py
├── video_processor.py
├── known_faces/
├── unknown_faces/
├── transcripts/
├── trackedVideo/
├── initialVideo/
├── logs/
├── tests/
└── .gitignore
```

## Typical workflow

1. Start the Streamlit app.
2. Manage persons in the database.
3. Run realtime recognition or process a video.
4. Review unknown faces and assign them to a known identity.
5. Open Voice Enrollment to listen to enrolled samples and unresolved speakers,
   edit the person assigned to an existing saved voice sample, or confirm an
   assignment/create a person for an unidentified speaker.
6. Rebuild or reload the face index if needed.
7. Continue recognition using stored embeddings.

## Important data folders

- `known_faces/`: label/reference face images
- `known_voices/`: enrolled voice embeddings and their saved audio samples
- `unknown_faces/`: unresolved face samples
- `unknown_voices/`: unresolved voice embeddings and representative audio clips
- `transcripts/`: generated transcription outputs
- `trackedVideo/`: processed output videos
- `logs/`: runtime logs
- `faces.db`: SQLite database with identities and logs
- `face_index.npz`: cached embedding index

## Development and testing

Run the Python test suite:

```powershell
pytest -q
```

Check dependency health:

```powershell
python -m pip check
```

## Notes

- Active-speaker detection is a visual speech estimate, not a biometric voice ID system.
- Whisper transcription is optional and depends on FFmpeg for audio extraction.
- If microphone capture fails, realtime voice analysis may be disabled while face recognition continues.
- If ONNX Runtime does not detect CUDA, the app will continue in CPU mode.

## Troubleshooting

- FFmpeg is taken from PATH when available; otherwise the `imageio-ffmpeg` dependency supplies a bundled executable.
- If realtime startup fails, inspect the latest log file under `logs/`.
- If CUDA is unavailable, do not worry; CPU mode is supported.
