# Face and Voice Recognition

A Streamlit application for face recognition, active-speaker estimation, and optional video transcription.

This project is built to run comfortably on CPU by default. If CUDA is available, the app can use it automatically. If not, it falls back to CPU instead of failing at startup.

## Overview

The system combines:

- UniFace SCRFD + ArcFace for face detection and embedding extraction
- OpenCV for fallback detection and video handling
- SQLite for person and identity storage
- NumPy for face-index matching
- Silero VAD for speech activity detection
- Whisper for optional transcription
- Streamlit for the web UI

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
5. Rebuild or reload the face index if needed.
6. Continue recognition using stored embeddings.

## Important data folders

- `known_faces/`: label/reference face images
- `unknown_faces/`: unresolved face samples
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

- If `ffmpeg` is not found, install FFmpeg and add its `bin` folder to PATH.
- If realtime startup fails, inspect the latest log file under `logs/`.
- If CUDA is unavailable, do not worry; CPU mode is supported.
