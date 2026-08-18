# AI Face + Active Speaker Recognition

This version intentionally removes FAISS.

## Architecture

- Streamlit UI
- ArcFace embeddings through DeepFace
- MediaPipe Face Mesh
- NumPy cosine-similarity index
- SQLite shared database
- WebRTC VAD for microphone/video audio
- Lip movement + VAD for active speaker estimation
- FFmpeg for video audio extraction and H.264 output

Realtime and uploaded-video modes use the same:

- `faces.db`
- `face_index.npz`
- `known_faces/`
- `unknown_faces/`

## 1. Install

Create a fresh virtual environment.

Windows:

```powershell
python -m venv venv
.\venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

PyAudio can be the difficult dependency on Windows. If normal installation fails, install a matching PyAudio wheel for your Python version, then run:

```powershell
pip install webrtcvad-wheels
```

## 2. FFmpeg

FFmpeg must be installed and available in PATH.

Verify:

```powershell
ffmpeg -version
```

If Windows says `ffmpeg is not recognized`, install FFmpeg and add its `bin` directory to PATH.

## 3. Start

```powershell
streamlit run app.py
```

## 4. First-time face registration

The system does not automatically turn every unknown face into a permanent person.

Recommended workflow:

1. Run realtime or video mode.
2. Unknown faces are temporarily tracked.
3. Stable unknown tracks are saved in `unknown_faces/`.
4. Open `Face Database`.
5. Assign an unknown face to a person.
6. The embedding becomes a training/reference sample for that person.
7. Rebuild the NumPy index.
8. Future realtime/video recognition can recognize that person.

## 5. Important

This is an active-speaker estimator, not a biometric voice-identification system.

The system decides:

- Is there speech?
- Which visible face has the strongest lip movement while speech is present?

For transcription, add Whisper/faster-whisper later.

## 6. Tuning

Start with:

- `RECOGNITION_THRESHOLD = 0.70`
- `AMBIGUITY_MARGIN = 0.05`
- `LIP_OPEN_THRESHOLD = 0.035`

These are not universal biometric thresholds. Calibrate them using your own camera, lighting, distance and subjects.

FaceAndVoiceRec/
│
├── app.py
├── audio_core.py
├── config.py
├── database.py
├── face_core.py
├── face_index.npz
├── face_landmarker.task
├── faces.db
├── realtime.py
├── video_processor.py
└── tracking.py

app.py
├── imports
│
├── Page Config & Initialization
│   ├── st.set_page_config()
│   └── init_db()
│
├── get_face_index()
│
├── fmt_time()
│
├── render_realtime()
│
├── render_video()
│
├── render_database()
│
├── render_audio_logs()
│
└── main()

audio_core.py
├── imports
│
├── Audio configuration
│   ├── SAMPLE_RATE
│   ├── FRAME_MS
│   └── FRAME_BYTES
│
├── VAD configuration
│   ├── VAD_AGGRESSIVENESS
│   ├── REALTIME_SPEECH_CONFIRM_FRAMES
│   ├── REALTIME_SILENCE_CONFIRM_FRAMES
│   ├── MIN_SPEECH_SEGMENT_MS
│   ├── SPEECH_START_PADDING_MS
│   ├── SPEECH_END_PADDING_MS
│   └── SPEECH_MERGE_GAP_MS
│
├── check_audio_dependencies()
│
├── RealtimeVAD
│   ├── start()
│   ├── stop()
│   ├── _emit()
│   ├── _update_speech_state()
│   └── _run()
│
├── read_wav_pcm()
│
├── detect_speech_segments()
│
├── _merge_segments()
│
└── audio_rms_from_pcm()

database.py
├── imports
│
├── Global State / Locks
│   └── _DB_LOCK
│
├── utc_now()
│
├── init_db()
│
├── get_conn()
│
├── create_person()
│
├── rename_person()
│
├── add_embedding()
│
├── create_unknown()
│
├── add_unknown_sample()
│
├── update_unknown_image()
│
├── list_unknown_samples()
│
├── resolve_unknown()
│
├── list_persons()
│
├── list_embeddings()
│
├── list_unknowns()
│
├── list_unknown_samples_with_embeddings()
│
├── log_recognition()
│
├── log_audio()
│
└── fetch_audio_logs()

face_core.py
├── imports
│
├── Global State / Locks
│   ├── _MODEL_LOCK
│   └── _MODEL_READY
│
├── _normalize()
│
├── face_quality()
│
├── build_face_model()
│
├── extract_embedding()
│
├── FaceIndex
│   ├── init()
│   ├── reload()
│   ├── rebuild()
│   ├── _save_locked()
│   ├── add()
│   └── search()
│
├── save_embedding_for_person()
│
├── find_matching_unknown()
│
└── register_unknown()

realtime.py
│
├── imports
│
├── SpeakingState
│
├── create_face_landmarker()
│
└── run()
     │
     ├── Open camera
     │    ├── MSMF
     │    ├── DirectShow
     │    └── OpenCV default
     │
     ├── Load FaceIndex
     │
     ├── Start VAD
     │
     ├── Load MediaPipe
     │
     ├── while camera running
     │    │
     │    ├── Read frame
     │    ├── MediaPipe detection
     │    ├── Create bounding boxes
     │    ├── Centroid tracking
     │    ├── Face recognition
     │    ├── Lip detection
     │    ├── Active speaker detection
     │    ├── Database logging (max 1/sec)
     │    └── Display
     │
     └── finally
          ├── Save final audio event
          ├── Stop VAD
          ├── Release camera
          ├── Close MediaPipe
          └── Destroy windows

video_processor.py
├── imports
│
├── Recognition stability / Constants
│   ├── RECOGNITION_MIN_SIMILARITY
│   └── MAX_RECOGNITION_FAILURES
│
├── get_ffmpeg()
│
├── extract_audio_to_wav()
│
├── convert_h264()
│
├── create_face_landmarker()
│
├── bbox_from_landmarks()
│
├── lip_open_ratio()
│
└── process_video_pipeline()

