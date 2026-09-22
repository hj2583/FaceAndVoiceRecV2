# Face and Voice Recognition

Streamlit application for face recognition, active-speaker estimation, and optional video transcription.
## Supported Windows setup

The default setup uses UniFace (ONNX Runtime) for face detection and recognition, with an optional NVIDIA CUDA GPU tier.
- Python 3.10+ 64-bit (UniFace supports up to Python 3.14)
- UniFace SCRFD detector + ArcFace recognizer (ONNX Runtime, NVIDIA CUDA)
- UniFace FaceMesh landmarks
- OpenCV face detection and video handling
- SQLite and NumPy face index storage
- Silero VAD (via sounddevice mic capture) and Whisper for audio features

Use `requirements-gpu.txt` for the NVIDIA CUDA tier. The root `requirements.txt` is kept identical for compatibility with existing setup commands.
## Installation

Open PowerShell in the repository directory.
### 1. Install prerequisites

Install Python 3.10 64-bit and FFmpeg. For Python:

```powershell
winget install --id Python.Python.3.10 --exact
```

FFmpeg must be available on `PATH`:
```powershell
ffmpeg -version
```

### 2. Create the environment

The repository's `directmlvenv` folder is ignored by Git and can be recreated safely. Remove it first only when rebuilding the environment.
```powershell
py -3.12 -m venv directmlvenv
.\directmlvenv\Scripts\python.exe -m pip install --upgrade pip
.\directmlvenv\Scripts\python.exe -m pip install --no-cache-dir -r requirements-gpu.txt
```

### 3. Verify the GPU provider

```powershell
.\directmlvenv\Scripts\python.exe -c "import onnxruntime as ort; print(ort.get_available_providers())"
```

`CUDAExecutionProvider` appearing in the list only means the plugin loaded; onnxruntime still silently falls back to CPU per-session if the actual CUDA Toolkit/cuDNN runtime DLLs are missing. Run a real detection once (e.g. start the app) and check the console for `CUDAExecutionProvider` load errors like `cublasLt64_13.dll ... missing` — if you see one, install the matching CUDA Toolkit + cuDNN redistributables (not just the GPU driver) from NVIDIA.

### 4. Start the application
```powershell
.\directmlvenv\Scripts\python.exe -m streamlit run .\app.py
```

Open the URL printed by Streamlit. Realtime mode opens a local camera window; press `Q` or `Esc` to stop it.

GPU execution is required. If CUDA/cuDNN cannot be loaded, the application now fails during UniFace initialization instead of silently consuming the CPU.
## Recognition workflow

1. Start the application and open the face database view.
2. Create a person or process a video containing an unknown person.
3. Unknown tracks are saved under `unknown_faces/`.
4. Assign an unknown track to a person in the UI.
5. Rebuild or reload the face index when prompted.
6. Run realtime or uploaded-video recognition again.

Recognition uses cosine similarity against normalized 512-dimensional ArcFace (UniFace) embeddings. The default acceptance threshold is `0.70`; tune it in `config.py` for the camera, lighting, and distance used by your application.
Active-speaker detection is visual speaker estimation, not voice identity recognition. It combines Silero VAD speech activity with visible lip movement. Whisper transcription is optional and requires FFmpeg for audio extraction.

## Important data

Do not delete these while you want to preserve recognition data:

- `known_faces/`: labeled face images
- `unknown_faces/`: unresolved face samples and embeddings
- `faces.db`: SQLite people, embeddings, and logs
- `face_index.npz`: cached NumPy embedding index

Generated outputs can be removed and regenerated:
- `trackedVideo/`
- `transcripts/`
- `logs/`
- `__pycache__/`
- `.pytest_cache/`

The `initialVideo/` folder contains user-provided input videos. Delete its contents only when those source videos are no longer needed.

## Tests and diagnostics

Run the tests with the CPU environment or any environment containing the test dependencies:

```powershell
.\directmlvenv\Scripts\python.exe -m pytest -q
```

Check installed package consistency:

```powershell
.\directmlvenv\Scripts\python.exe -m pip check
```

Useful diagnostics:
```powershell
.\directmlvenv\Scripts\python.exe -c "import onnxruntime as ort; print(ort.__version__, ort.get_available_providers())"
.\directmlvenv\Scripts\python.exe -c "import cv2; print(cv2.__version__, cv2.data.haarcascades)"
```

If realtime startup fails, inspect the newest `logs/realtime-*.log` file. If the microphone is unavailable, face recognition can continue while active-speaker estimation is disabled.

## Project layout

| Path | Role |
| --- | --- |
| `app.py` | Streamlit entry point |
| `realtime.py` | Camera, tracking, recognition, and active-speaker loop |
| `realtime_launcher.py` | Starts realtime mode in a child process |
| `video_processor.py` | Uploaded-video pipeline |
| `face_core.py` | Embedding index and recognition |
| `face_backend.py` | UniFace (SCRFD + ArcFace) backend |
| `detection_core.py` | Detection and OpenCV fallback |
| `audio_core.py` | Microphone and VAD processing |
| `transcription_core.py` | Whisper and transcript handling |
| `database.py` | SQLite persistence |
| `config.py` | Application settings |
| `tests/` | Automated tests |

## Cleanup

The following PowerShell command removes only caches and generated outputs. It does not remove face data, the database, source videos, or virtual environments:

```powershell
Remove-Item .\__pycache__, .\.pytest_cache, .\logs\*.log, .\transcripts\* -Recurse -Force -ErrorAction SilentlyContinue
```

To remove processed videos, review them first and then delete only the selected files from `trackedVideo/`.
Here is your Markdown cleaned up, formatted, and visually structured for clarity and scannability.

---

# AI Face + Active Speaker Recognition

This version intentionally removes FAISS.

## System Architecture

* **Streamlit UI:** Web interface
* **UniFace (SCRFD + ArcFace):** Face detection + embedding extraction
* **UniFace FaceMesh:** Landmark detection
* **NumPy:** Cosine-similarity indexing
* **SQLite:** Shared database
* **Silero VAD:** Audio Voice Activity Detection (mic & video)
* **Lip Movement + VAD:** Active speaker estimation
* **FFmpeg:** Video audio extraction & H.264 rendering

Both **Realtime** and **Uploaded Video** modes share the following assets:

* `faces.db`
* `face_index.npz`
* `known_faces/`
* `unknown_faces/`

---

## 1. Installation

Create and activate a fresh virtual environment.

**Windows (PowerShell):**

```powershell
python -m venv venv
.\venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt

```

> **Note:** sounddevice depends on the PortAudio library; on Windows it ships bundled with the wheel, so no extra system install is normally required.

---

## 2. FFmpeg Setup

FFmpeg must be installed and added to your system's `PATH`.

**Verification:**

```powershell
ffmpeg -version

```

If Windows displays `ffmpeg is not recognized`, install FFmpeg manually and append its `bin` folder path to your environment variables.

---

## 3. Launch Application

```powershell
streamlit run app.py

```

## Realtime Mode Behavior

Realtime mode uses lightweight OpenCV face detection to keep the native camera window responsive. Uploaded-video mode retains the heavier long-range detector. The window displays rolling FPS; Q or Escape stops it. If microphone initialization fails, face detection continues with `MIC: UNAVAILABLE` and active-speaker estimation is disabled.

If realtime startup fails immediately, Streamlit shows an actionable error message. Detailed child-process output is written to `logs/realtime-*.log` for troubleshooting.

---

## 4. First-Time Face Registration Workflow

The system does not automatically convert every unknown face into a permanent profile.

**Recommended Workflow:**

1. Run **Realtime** or **Video** mode.
2. Unknown faces are temporarily tracked.
3. Stable unknown tracks are persisted in `unknown_faces/`.
4. Open the **Face Database** view in the UI.
5. Assign an unknown face to a specific person.
6. The face embedding becomes a training/reference sample for that person.
7. Rebuild the **NumPy index**.
8. Future realtime and video pipelines will recognize that person.

---

## 5. Important Notes

> This is an **active-speaker estimator**, not a biometric voice-identification system.

The pipeline evaluates two criteria:

* **Speech Presence:** Is voice activity present via VAD?
* **Visual Anchor:** Which visible face exhibits the strongest lip displacement while speech is active?

*For automatic speech transcription, integrate Whisper or faster-whisper.*

---

## 6. Parameter Tuning

Default reference values:

```python
RECOGNITION_THRESHOLD = 0.70
AMBIGUITY_MARGIN      = 0.05
LIP_OPEN_THRESHOLD    = 0.035

```

*These are non-universal threshold values. Calibrate them according to your camera setup, lighting conditions, subject distance, and environment.*

---

## Project Structure

```text
FaceAndVoiceRec/
│
├── app.py
├── audio_core.py
├── config.py
├── database.py
├── face_core.py
├── face_index.npz
├── faces.db
├── realtime.py
├── video_processor.py
└── tracking.py

```

---

## Code Base & Execution Tree

### `app.py`

* **Imports**
* **Page Config & Initialization**
* `st.set_page_config()`
* `init_db()`


* `get_face_index()`
* `fmt_time()`
* `render_realtime()`
* `render_video()`
* `render_database()`
* `render_audio_logs()`
* `main()`

---

### `audio_core.py`

* **Imports**
* **Audio Configuration**
* `SAMPLE_RATE`
* `FRAME_MS`
* `FRAME_BYTES`


* **VAD Configuration**
* `VAD_SPEECH_THRESHOLD`
* `REALTIME_SPEECH_CONFIRM_FRAMES`
* `REALTIME_SILENCE_CONFIRM_FRAMES`
* `MIN_SPEECH_SEGMENT_MS`
* `SPEECH_START_PADDING_MS`
* `SPEECH_END_PADDING_MS`
* `SPEECH_MERGE_GAP_MS`


* `check_audio_dependencies()`
* **RealtimeVAD Class**
* `start()`
* `stop()`
* `_emit()`
* `_update_speech_state()`
* `_run()`


* `read_wav_pcm()`
* `detect_speech_segments()`
* `_merge_segments()`
* `audio_rms_from_pcm()`

---

### `database.py`

* **Imports**
* **Global State / Locks**
* `_DB_LOCK`


* `utc_now()`
* `init_db()`
* `get_conn()`
* `create_person()`
* `rename_person()`
* `add_embedding()`
* `create_unknown()`
* `add_unknown_sample()`
* `update_unknown_image()`
* `list_unknown_samples()`
* `resolve_unknown()`
* `list_persons()`
* `list_embeddings()`
* `list_unknowns()`
* `list_unknown_samples_with_embeddings()`
* `log_recognition()`
* `log_audio()`
* `fetch_audio_logs()`

---

### `face_core.py`

* **Imports**
* **Global State / Locks**
* `_MODEL_LOCK`
* `_MODEL_READY`


* `_normalize()`
* `face_quality()`
* `build_face_model()`
* `extract_embedding()`
* **FaceIndex Class**
* `init()`
* `reload()`
* `rebuild()`
* `_save_locked()`
* `add()`
* `search()`


* `save_embedding_for_person()`
* `find_matching_unknown()`
* `register_unknown()`

---

### `realtime.py`

* **Imports**
* **SpeakingState**
* `create_face_landmarker()`
* **`run()` Execution Loop**
1. Open camera (*MSMF*, *DirectShow*, or *OpenCV default*)
2. Load `FaceIndex`
3. Start `VAD`
4. Load face landmarker
5. **Frame Processing Loop:**
* Read frame
* Face landmark detection
* Create bounding boxes
* Centroid tracking
* Face recognition
* Lip detection
* Active speaker detection
* Database logging (max 1/sec)
* Display output


6. **Cleanup (finally block):**
* Save final audio event
* Stop VAD
* Release camera
* Close face landmarker
* Destroy windows





---

### `video_processor.py`

* **Imports**
* **Recognition Stability / Constants**
* `RECOGNITION_MIN_SIMILARITY`
* `MAX_RECOGNITION_FAILURES`


* `get_ffmpeg()`
* `extract_audio_to_wav()`
* `convert_h264()`
* `create_face_landmarker()`
* `bbox_from_landmarks()`
* `lip_open_ratio()`
* `process_video_pipeline()`