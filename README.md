Here is your Markdown cleaned up, formatted, and visually structured for clarity and scannability.

---

# AI Face + Active Speaker Recognition

This version intentionally removes FAISS.

## System Architecture

* **Streamlit UI:** Web interface
* **DeepFace (ArcFace):** Face embedding extraction
* **MediaPipe Face Mesh:** Landmark detection
* **NumPy:** Cosine-similarity indexing
* **SQLite:** Shared database
* **WebRTC VAD:** Audio Voice Activity Detection (mic & video)
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

> **Note:** PyAudio can be a difficult dependency on Windows. If normal installation fails, install a matching PyAudio wheel for your Python version, then run:
> ```powershell
> pip install webrtcvad-wheels
> 
> ```
> 
> 

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
├── face_landmarker.task
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
* `VAD_AGGRESSIVENESS`
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
4. Load MediaPipe
5. **Frame Processing Loop:**
* Read frame
* MediaPipe detection
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
* Close MediaPipe
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