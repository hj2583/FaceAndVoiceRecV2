# Transcript Accuracy, Voice Diarization, and Voice Recognition Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make offline meeting transcripts more accurate, split them into distinct speaker turns, and label those speakers with enrolled identities (falling back to `Unknown Speaker N`), with realtime video speaker attribution using voice as a fallback confirmation signal.

**Architecture:** A new `voice_core.py` module (mirroring `face_core.py`) extracts SpeechBrain ECAPA-TDNN speaker embeddings, clusters per-meeting speech segments into speakers, and matches clusters/live audio against enrolled voiceprints stored in new `voice_embeddings` / `unknown_voices` / `unknown_voice_samples` tables. `transcription_core.py` gates Whisper on Silero VAD speech spans and wires the new diarizer into its existing `diarize` callback parameter. `app.py` gets enrollment/resolution UI mirroring the existing unknown-face flow. `video_processor.py` / `realtime.py` use a voice match only when face recognition is absent/low-confidence.

**Tech Stack:** Python, SpeechBrain (`speechbrain/spkrec-ecapa-voxceleb`), PyTorch (existing dependency), `scipy.cluster.hierarchy` (existing dependency, no new package), Whisper (existing), Silero VAD (existing), SQLite, Streamlit.

## Global Constraints

- No gated/authenticated model downloads — `speechbrain/spkrec-ecapa-voxceleb` is public on Hugging Face.
- No new clustering dependency — use `scipy.cluster.hierarchy` (already in `requirements.txt`), not `scikit-learn`.
- A confident face match must never be overridden by a voice match in the realtime path.
- Missing/failed voice model must degrade gracefully: offline transcription still produces `"Unknown Speaker"` output (existing behavior) instead of raising; realtime voice fallback must not crash or block the video pipeline.
- Follow existing code conventions: `database.py` functions take/return plain values and use `get_conn()`; `face_core.py`-style modules keep embeddings L2-normalized float32 arrays of a fixed dimension; tests live under `tests/` and use `monkeypatch`/`tmp_path`, matching `tests/test_face_core_preprocessing.py` and `tests/test_database_unknown_cleanup.py` style.

---

## Task 1: Config additions

**Files:**
- Modify: `config.py` (append near the existing "Active speaker settings" section, before "Video output")

**Interfaces:**
- Produces: `config.VOICE_EMBEDDING_MODEL` (str), `config.VOICE_EMBEDDING_DIM` (int), `config.VOICE_MATCH_THRESHOLD` (float), `config.VOICE_AMBIGUITY_MARGIN` (float), `config.VOICE_CLUSTER_DISTANCE_THRESHOLD` (float), `config.UNKNOWN_VOICES_DIR` (Path, created), `config.WHISPER_LANGUAGE` (Optional[str]), `config.WHISPER_INITIAL_PROMPT` (Optional[str]).

- [ ] **Step 1: Add the config values**

Add to `config.py`, directly after the `LIP_OPEN_THRESHOLD` / `SPEECH_CONFIRM_FRAMES` block (the "Active speaker settings" section):

```python
# ============================================================
# Voice embeddings / diarization
# ============================================================

# SpeechBrain speaker-embedding model (public, no HF auth required).
VOICE_EMBEDDING_MODEL = "speechbrain/spkrec-ecapa-voxceleb"

# ECAPA-TDNN embedding size for the model above.
VOICE_EMBEDDING_DIM = 192

# Minimum cosine similarity required to accept a voice match
# against an enrolled voiceprint.
VOICE_MATCH_THRESHOLD = 0.72

# If the best and second-best voice matches are too close,
# the match is considered ambiguous and rejected.
VOICE_AMBIGUITY_MARGIN = 0.05

# Cosine-distance threshold used by agglomerative clustering
# when grouping a meeting's speech segments into speakers.
# Lower = more/smaller clusters (more distinct speakers found).
VOICE_CLUSTER_DISTANCE_THRESHOLD = 0.35

# Whisper decoding language ("en", "ms", ...); None = auto-detect.
WHISPER_LANGUAGE = None

# Optional Whisper decoding hint (domain vocabulary, names, etc).
WHISPER_INITIAL_PROMPT = None
```

Add `UNKNOWN_VOICES_DIR = BASE_DIR / "unknown_voices"` to the existing `BASE_DIR`-relative path block near the top of the file, and add it to the directory-creation loop (the `for directory in (...)` block) alongside `UNKNOWN_FACES_DIR`.

- [ ] **Step 2: Verify import**

Run: `python -c "import config; print(config.VOICE_EMBEDDING_DIM, config.UNKNOWN_VOICES_DIR.exists())"`
Expected: `192 True`

- [ ] **Step 3: Commit**

```bash
git add config.py
git commit -m "config: add voice embedding and Whisper accuracy settings"
```

---

## Task 2: Database schema and CRUD for voice tables

**Files:**
- Modify: `database.py`
- Test: `tests/test_database_voice.py` (new)

**Interfaces:**
- Consumes: `database.get_conn()`, `database.utc_now()`, `database.create_person(name)` (all existing).
- Produces:
  - `database.add_voice_embedding(person_id, embedding_path, quality=0.0) -> int`
  - `database.list_voice_embeddings() -> list[tuple[embedding_id, person_id, name, embedding_path, quality]]`
  - `database.create_unknown_voice(label, embedding_path) -> int`
  - `database.add_unknown_voice_sample(unknown_voice_id, embedding_path, quality=0.0) -> int`
  - `database.list_unknown_voice_samples(unknown_voice_id) -> list[tuple[sample_id, embedding_path, quality, created_at]]`
  - `database.list_unknown_voice_samples_with_embeddings() -> list[tuple[sample_id, unknown_voice_id, embedding_path, quality]]` (only unresolved)
  - `database.list_unknown_voices(include_resolved=False) -> list[tuple[unknown_voice_id, label, embedding_path, created_at, resolved_person_id]]`
  - `database.resolve_unknown_voice(unknown_voice_id, person_id) -> None`
  - `database.delete_unknown_voice(unknown_voice_id) -> bool` (mirrors `delete_unknown`; only deletes unresolved rows, unlinks orphaned files)

- [ ] **Step 1: Write the failing tests**

Create `tests/test_database_voice.py`:

```python
from pathlib import Path

import database


def _setup_database(tmp_path, monkeypatch):
    monkeypatch.setattr(database, "DB_PATH", tmp_path / "faces.db")
    database.init_db()


def test_add_and_list_voice_embeddings(tmp_path, monkeypatch):
    _setup_database(tmp_path, monkeypatch)
    person_id = database.create_person("Alice")
    embedding_path = tmp_path / "alice.npy"
    embedding_path.write_bytes(b"embedding")

    embedding_id = database.add_voice_embedding(person_id, embedding_path, quality=0.9)

    rows = database.list_voice_embeddings()
    assert rows == [(embedding_id, person_id, "Alice", str(embedding_path), 0.9)]


def test_create_unknown_voice_and_add_samples(tmp_path, monkeypatch):
    _setup_database(tmp_path, monkeypatch)
    embedding_path = tmp_path / "cluster.npy"
    embedding_path.write_bytes(b"embedding")

    unknown_voice_id = database.create_unknown_voice("Speaker 1", embedding_path)
    sample_path = tmp_path / "sample.npy"
    sample_path.write_bytes(b"sample")
    database.add_unknown_voice_sample(unknown_voice_id, sample_path, quality=0.5)

    samples = database.list_unknown_voice_samples(unknown_voice_id)
    assert len(samples) == 1
    assert samples[0][1] == str(sample_path)

    unresolved = database.list_unknown_voices()
    assert unresolved[0][0] == unknown_voice_id
    assert unresolved[0][4] is None


def test_resolve_unknown_voice_sets_resolved_person_id(tmp_path, monkeypatch):
    _setup_database(tmp_path, monkeypatch)
    person_id = database.create_person("Bob")
    embedding_path = tmp_path / "cluster.npy"
    embedding_path.write_bytes(b"embedding")
    unknown_voice_id = database.create_unknown_voice("Speaker 1", embedding_path)

    database.resolve_unknown_voice(unknown_voice_id, person_id)

    rows = database.list_unknown_voices(include_resolved=True)
    assert rows[0][4] == person_id
    assert database.list_unknown_voices() == []


def test_delete_unknown_voice_removes_record_samples_and_files(tmp_path, monkeypatch):
    _setup_database(tmp_path, monkeypatch)
    embedding_path = tmp_path / "cluster.npy"
    embedding_path.write_bytes(b"embedding")
    unknown_voice_id = database.create_unknown_voice("Speaker 1", embedding_path)
    sample_path = tmp_path / "sample.npy"
    sample_path.write_bytes(b"sample")
    database.add_unknown_voice_sample(unknown_voice_id, sample_path)

    assert database.delete_unknown_voice(unknown_voice_id) is True

    assert not embedding_path.exists()
    assert not sample_path.exists()
    assert database.list_unknown_voice_samples(unknown_voice_id) == []
    assert database.list_unknown_voices(include_resolved=True) == []


def test_delete_unknown_voice_refuses_resolved_records(tmp_path, monkeypatch):
    _setup_database(tmp_path, monkeypatch)
    person_id = database.create_person("Carol")
    embedding_path = tmp_path / "cluster.npy"
    embedding_path.write_bytes(b"embedding")
    unknown_voice_id = database.create_unknown_voice("Speaker 1", embedding_path)
    database.resolve_unknown_voice(unknown_voice_id, person_id)

    assert database.delete_unknown_voice(unknown_voice_id) is False
    assert embedding_path.exists()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_database_voice.py -v`
Expected: FAIL — `AttributeError: module 'database' has no attribute 'add_voice_embedding'` (and similar for the other new functions).

- [ ] **Step 3: Add the schema in `init_db()`**

In `database.py`, inside `init_db()`, add after the existing `transcription_segments` index creation block:

```python
        conn.execute("""
            CREATE TABLE IF NOT EXISTS voice_embeddings (
                embedding_id INTEGER PRIMARY KEY AUTOINCREMENT,
                person_id INTEGER NOT NULL,
                embedding_path TEXT NOT NULL,
                quality REAL DEFAULT 0,
                created_at TEXT NOT NULL,
                FOREIGN KEY(person_id) REFERENCES persons(person_id)
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS unknown_voices (
                unknown_voice_id INTEGER PRIMARY KEY AUTOINCREMENT,
                label TEXT NOT NULL,
                embedding_path TEXT,
                created_at TEXT NOT NULL,
                resolved_person_id INTEGER,
                FOREIGN KEY(resolved_person_id) REFERENCES persons(person_id)
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS unknown_voice_samples (
                sample_id INTEGER PRIMARY KEY AUTOINCREMENT,
                unknown_voice_id INTEGER NOT NULL,
                embedding_path TEXT NOT NULL,
                quality REAL DEFAULT 0,
                created_at TEXT NOT NULL,
                FOREIGN KEY(unknown_voice_id)
                    REFERENCES unknown_voices(unknown_voice_id)
                    ON DELETE CASCADE
            )
        """)
```

- [ ] **Step 4: Add the CRUD functions**

Append to `database.py` (after the existing `delete_low_quality_unknowns` function):

```python
def add_voice_embedding(person_id, embedding_path, quality=0.0):
    with get_conn() as conn:
        cur = conn.execute(
            """
            INSERT INTO voice_embeddings(person_id, embedding_path, quality, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (int(person_id), str(embedding_path), float(quality), utc_now()),
        )
        return int(cur.lastrowid)


def list_voice_embeddings():
    with get_conn() as conn:
        return conn.execute(
            """
            SELECT ve.embedding_id, ve.person_id, p.name, ve.embedding_path, ve.quality
            FROM voice_embeddings ve
            JOIN persons p ON p.person_id = ve.person_id
            ORDER BY ve.embedding_id
            """
        ).fetchall()


def create_unknown_voice(label, embedding_path):
    with get_conn() as conn:
        cur = conn.execute(
            """
            INSERT INTO unknown_voices(label, embedding_path, created_at)
            VALUES (?, ?, ?)
            """,
            (label, str(embedding_path), utc_now()),
        )
        return int(cur.lastrowid)


def add_unknown_voice_sample(unknown_voice_id, embedding_path, quality=0.0):
    with get_conn() as conn:
        cur = conn.execute(
            """
            INSERT INTO unknown_voice_samples(unknown_voice_id, embedding_path, quality, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (int(unknown_voice_id), str(embedding_path), float(quality), utc_now()),
        )
        return int(cur.lastrowid)


def list_unknown_voice_samples(unknown_voice_id):
    with get_conn() as conn:
        return conn.execute(
            """
            SELECT sample_id, embedding_path, quality, created_at
            FROM unknown_voice_samples
            WHERE unknown_voice_id=?
            ORDER BY quality DESC
            """,
            (int(unknown_voice_id),),
        ).fetchall()


def list_unknown_voice_samples_with_embeddings():
    with get_conn() as conn:
        return conn.execute(
            """
            SELECT uvs.sample_id, uvs.unknown_voice_id, uvs.embedding_path, uvs.quality
            FROM unknown_voice_samples uvs
            JOIN unknown_voices uv ON uv.unknown_voice_id = uvs.unknown_voice_id
            WHERE uv.resolved_person_id IS NULL
            ORDER BY uvs.quality DESC
            """
        ).fetchall()


def list_unknown_voices(include_resolved=False):
    query = """
        SELECT unknown_voice_id, label, embedding_path, created_at, resolved_person_id
        FROM unknown_voices
    """
    if not include_resolved:
        query += " WHERE resolved_person_id IS NULL"
    query += " ORDER BY unknown_voice_id DESC"
    with get_conn() as conn:
        return conn.execute(query).fetchall()


def resolve_unknown_voice(unknown_voice_id, person_id):
    with get_conn() as conn:
        conn.execute(
            "UPDATE unknown_voices SET resolved_person_id=? WHERE unknown_voice_id=?",
            (person_id, unknown_voice_id),
        )


def _delete_unknown_voice_locked(conn, unknown_voice_id):
    parent = conn.execute(
        "SELECT embedding_path, resolved_person_id FROM unknown_voices WHERE unknown_voice_id=?",
        (int(unknown_voice_id),),
    ).fetchone()
    if parent is None or parent[1] is not None:
        return False

    sample_rows = conn.execute(
        "SELECT embedding_path FROM unknown_voice_samples WHERE unknown_voice_id=?",
        (int(unknown_voice_id),),
    ).fetchall()
    paths = [parent[0]] + [row[0] for row in sample_rows]

    conn.execute(
        "DELETE FROM unknown_voice_samples WHERE unknown_voice_id=?",
        (int(unknown_voice_id),),
    )
    conn.execute(
        "DELETE FROM unknown_voices WHERE unknown_voice_id=?",
        (int(unknown_voice_id),),
    )
    for path in paths:
        if path:
            try:
                Path(path).unlink(missing_ok=True)
            except OSError:
                pass
    return True


def delete_unknown_voice(unknown_voice_id):
    with get_conn() as conn:
        return _delete_unknown_voice_locked(conn, unknown_voice_id)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_database_voice.py -v`
Expected: PASS (5 tests)

- [ ] **Step 6: Commit**

```bash
git add database.py tests/test_database_voice.py
git commit -m "feat: add voice_embeddings/unknown_voices tables and CRUD"
```

---

## Task 3: Whisper transcription accuracy improvements

**Files:**
- Modify: `transcription_core.py`
- Modify: `tests/test_transcription_core.py`

**Interfaces:**
- Consumes: `audio_core.detect_speech_segments(wav_path) -> list[tuple[float, float]]` (existing), `config.WHISPER_LANGUAGE`, `config.WHISPER_INITIAL_PROMPT` (Task 1).
- Produces: `transcription_core.transcribe_with_diarization(...)` unchanged signature, but now VAD-gated internally; segments carry word-level timing info already flattened into existing `start_ms`/`end_ms` (no interface change to `TranscriptionSegment`).

- [ ] **Step 1: Write the failing test**

Add to `tests/test_transcription_core.py`:

```python
def test_transcribe_with_diarization_only_transcribes_vad_speech_regions(tmp_path, monkeypatch):
    wav_path = tmp_path / "audio.wav"
    wav_path.write_bytes(b"fake wav")

    monkeypatch.setattr(
        "transcription_core.detect_speech_segments",
        lambda *_a, **_k: [(0.0, 2.0), (5.0, 7.0)],
    )

    calls = []

    class FakeModel:
        def transcribe(self, path, verbose=False, **kwargs):
            calls.append(kwargs)
            return {
                "segments": [
                    {"start": 0.1, "end": 1.0, "text": "hello", "avg_logprob": -0.1},
                ]
            }

    fake_whisper = type("FakeWhisperModule", (), {"load_model": staticmethod(lambda *_a, **_k: FakeModel())})
    monkeypatch.setitem(__import__("sys").modules, "whisper", fake_whisper)

    from transcription_core import transcribe_with_diarization

    segments = transcribe_with_diarization(wav_path)

    assert segments[0].text == "hello"
    assert calls[0]["condition_on_previous_text"] is False
    assert calls[0]["word_timestamps"] is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_transcription_core.py::test_transcribe_with_diarization_only_transcribes_vad_speech_regions -v`
Expected: FAIL — `KeyError: 'condition_on_previous_text'` (the fake `transcribe` receives no such kwarg yet).

- [ ] **Step 3: Implement VAD gating and decode parameters**

In `transcription_core.py`, add the import and rewrite `transcribe_with_diarization`:

```python
from audio_core import detect_speech_segments
```

Replace the body of `transcribe_with_diarization` (the section between `model = whisper.load_model(...)` and the `for raw in result.get("segments", [])` loop) with:

```python
    model = whisper.load_model(model_size or config.WHISPER_MODEL)

    speech_regions = detect_speech_segments(audio_path)
    transcribe_kwargs = {
        "condition_on_previous_text": False,
        "word_timestamps": True,
    }
    if config.WHISPER_LANGUAGE:
        transcribe_kwargs["language"] = config.WHISPER_LANGUAGE
    if config.WHISPER_INITIAL_PROMPT:
        transcribe_kwargs["initial_prompt"] = config.WHISPER_INITIAL_PROMPT

    if not speech_regions:
        return []

    result = model.transcribe(str(audio_path), verbose=False, **transcribe_kwargs)
    diarization = list(diarize(str(audio_path))) if diarize else []
    segments = []

    for raw in result.get("segments", []):
        text = raw.get("text", "").strip()
        start_seconds = float(raw.get("start", 0.0))
        end_seconds = float(raw.get("end", 0.0))
        if not text or end_seconds <= start_seconds:
            continue

        # Whisper transcribes the whole file; drop segments that don't
        # overlap any VAD-detected speech region (silence hallucinations).
        if not any(
            min(end_seconds, region_end) - max(start_seconds, region_start) > 0
            for region_start, region_end in speech_regions
        ):
            continue

        start_ms = int(start_seconds * 1000)
        end_ms = int(end_seconds * 1000)
```

Keep the rest of the loop body (label assignment via `diarization`, and appending `TranscriptionSegment(...)`) unchanged, but change the two `raw["end"]`/`raw["start"]` references inside the diarization-overlap loop to use the already-extracted `start_seconds`/`end_seconds` local variables instead of re-indexing `raw`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_transcription_core.py -v`
Expected: PASS (all tests, including the new one and the pre-existing ones)

- [ ] **Step 5: Commit**

```bash
git add transcription_core.py tests/test_transcription_core.py
git commit -m "feat: gate Whisper transcription on VAD speech regions and tune decoding"
```

---

## Task 4: `voice_core.py` — embedding extraction and voiceprint matching

**Files:**
- Create: `voice_core.py`
- Test: `tests/test_voice_core.py` (new)

**Interfaces:**
- Consumes: `config.VOICE_EMBEDDING_MODEL`, `config.VOICE_EMBEDDING_DIM`, `config.VOICE_MATCH_THRESHOLD`, `config.VOICE_AMBIGUITY_MARGIN` (Task 1); `database.list_voice_embeddings()` (Task 2).
- Produces:
  - `voice_core.extract_voice_embedding(pcm_int16_bytes, sample_rate=16000) -> Optional[np.ndarray]` (shape `(VOICE_EMBEDDING_DIM,)`, L2-normalized float32, or `None` on failure/too-short audio)
  - `voice_core.match_voice_embedding(query_embedding) -> Optional[dict]` with keys `person_id`, `person_name`, `similarity` — mirrors `face_core.FaceIndex.search`'s ambiguity-margin behavior but implemented as a direct linear scan (voiceprint counts are small; no persisted index needed).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_voice_core.py`:

```python
import numpy as np

import database
import voice_core


def _setup_database(tmp_path, monkeypatch):
    monkeypatch.setattr(database, "DB_PATH", tmp_path / "faces.db")
    database.init_db()


def _fake_model(vector):
    class FakeModel:
        def encode_batch(self, tensor):
            import torch
            return torch.from_numpy(np.asarray([vector], dtype=np.float32)).unsqueeze(0)
    return FakeModel()


def test_extract_voice_embedding_normalizes_and_matches_dimension(monkeypatch):
    raw_vector = np.zeros(voice_core.EMBEDDING_DIM, dtype=np.float32)
    raw_vector[3] = 5.0  # not unit-norm on purpose; extract_voice_embedding must normalize it
    monkeypatch.setattr(voice_core, "_load_model", lambda: _fake_model(raw_vector))

    pcm = (b"\x00\x01" * 8000)  # 16000 samples of fake PCM16 audio, above _MIN_SAMPLES

    embedding = voice_core.extract_voice_embedding(pcm)

    assert embedding is not None
    assert embedding.shape == (voice_core.EMBEDDING_DIM,)
    assert abs(float(np.linalg.norm(embedding)) - 1.0) < 1e-5


def test_extract_voice_embedding_returns_none_for_empty_audio(monkeypatch):
    assert voice_core.extract_voice_embedding(b"") is None


def test_match_voice_embedding_returns_best_person_above_threshold(tmp_path, monkeypatch):
    _setup_database(tmp_path, monkeypatch)
    person_id = database.create_person("Dana")
    embedding = np.zeros(voice_core.EMBEDDING_DIM, dtype=np.float32)
    embedding[0] = 1.0
    embedding_path = tmp_path / "dana.npy"
    np.save(embedding_path, embedding)
    database.add_voice_embedding(person_id, embedding_path, quality=1.0)

    query = np.zeros(voice_core.EMBEDDING_DIM, dtype=np.float32)
    query[0] = 1.0

    match = voice_core.match_voice_embedding(query)

    assert match is not None
    assert match["person_id"] == person_id
    assert match["person_name"] == "Dana"
    assert match["similarity"] > 0.99


def test_match_voice_embedding_returns_none_below_threshold(tmp_path, monkeypatch):
    _setup_database(tmp_path, monkeypatch)
    person_id = database.create_person("Eve")
    embedding = np.zeros(voice_core.EMBEDDING_DIM, dtype=np.float32)
    embedding[0] = 1.0
    embedding_path = tmp_path / "eve.npy"
    np.save(embedding_path, embedding)
    database.add_voice_embedding(person_id, embedding_path, quality=1.0)

    query = np.zeros(voice_core.EMBEDDING_DIM, dtype=np.float32)
    query[-1] = 1.0  # orthogonal -> similarity 0.0

    assert voice_core.match_voice_embedding(query) is None


def test_match_voice_embedding_returns_none_when_no_voiceprints(tmp_path, monkeypatch):
    _setup_database(tmp_path, monkeypatch)
    query = np.zeros(voice_core.EMBEDDING_DIM, dtype=np.float32)
    query[0] = 1.0
    assert voice_core.match_voice_embedding(query) is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_voice_core.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'voice_core'`

- [ ] **Step 3: Implement `voice_core.py`**

```python
"""Speaker-embedding extraction and voiceprint matching.

Mirrors face_core.py's shape: extract a fixed-size, L2-normalized
embedding, then match it against enrolled voiceprints by cosine similarity.
"""

import logging
import threading
from pathlib import Path

import numpy as np

import config
from database import get_conn, list_voice_embeddings

logger = logging.getLogger(__name__)

EMBEDDING_DIM = config.VOICE_EMBEDDING_DIM

_MODEL_LOCK = threading.RLock()
_MODEL = None

# Minimum audio length SpeechBrain's ECAPA-TDNN model can embed reliably.
_MIN_SAMPLES = 400


def _normalize(vector):
    vector = np.asarray(vector, dtype=np.float32).reshape(-1)
    if vector.size == 0 or not np.all(np.isfinite(vector)):
        return None
    norm = np.linalg.norm(vector)
    if not np.isfinite(norm) or norm <= 1e-8:
        return None
    return (vector / norm).astype(np.float32)


def _load_model():
    global _MODEL
    with _MODEL_LOCK:
        if _MODEL is None:
            from speechbrain.inference.speaker import EncoderClassifier

            logger.info("Loading SpeechBrain speaker embedding model...")
            _MODEL = EncoderClassifier.from_hparams(source=config.VOICE_EMBEDDING_MODEL)
            logger.info("SpeechBrain speaker embedding model loaded.")
        return _MODEL


def extract_voice_embedding(pcm_int16_bytes, sample_rate=16000):
    """Extract a speaker embedding from raw PCM16 mono audio, or None on failure."""
    if not pcm_int16_bytes:
        return None

    audio = np.frombuffer(pcm_int16_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    if audio.size < _MIN_SAMPLES:
        return None

    try:
        import torch

        model = _load_model()
        with torch.no_grad():
            tensor = torch.from_numpy(audio).unsqueeze(0)
            embedding = model.encode_batch(tensor)
        embedding = embedding.squeeze().cpu().numpy()
        embedding = _normalize(embedding)
        if embedding is None or embedding.shape[0] != EMBEDDING_DIM:
            return None
        return embedding
    except Exception:
        logger.exception("Voice embedding extraction failed")
        return None


def match_voice_embedding(query_embedding, threshold=None, ambiguity_margin=None):
    """Compare a query embedding against every enrolled voiceprint.

    Returns {"person_id", "person_name", "similarity"} or None.
    """
    threshold = config.VOICE_MATCH_THRESHOLD if threshold is None else threshold
    ambiguity_margin = (
        config.VOICE_AMBIGUITY_MARGIN if ambiguity_margin is None else ambiguity_margin
    )

    query = _normalize(query_embedding)
    if query is None:
        return None

    rows = list_voice_embeddings()
    if not rows:
        return None

    scores = []
    for embedding_id, person_id, person_name, embedding_path, _quality in rows:
        path = Path(embedding_path)
        if not path.exists():
            continue
        try:
            stored = _normalize(np.load(path))
            if stored is None or stored.shape[0] != EMBEDDING_DIM:
                continue
            similarity = float(np.dot(query, stored))
            scores.append((similarity, person_id, person_name))
        except Exception:
            logger.exception("Failed loading voice embedding: %s", path)

    if not scores:
        return None

    scores.sort(key=lambda item: item[0], reverse=True)
    best_similarity, best_person_id, best_person_name = scores[0]

    second_similarity = next(
        (score for score, person_id, _name in scores[1:] if person_id != best_person_id),
        -1.0,
    )

    if best_similarity < threshold:
        return None
    if second_similarity >= 0.0 and (best_similarity - second_similarity) < ambiguity_margin:
        return None

    return {
        "person_id": best_person_id,
        "person_name": best_person_name,
        "similarity": best_similarity,
    }
```

Remove the broken placeholder test (`test_extract_voice_embedding_normalizes_and_matches_dimension`) from `tests/test_voice_core.py` now — it was a draft stub, not a real test.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_voice_core.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add voice_core.py tests/test_voice_core.py
git commit -m "feat: add voice_core embedding extraction and voiceprint matching"
```

---

## Task 5: `voice_core.py` — diarization clustering and unknown-voice registration

**Files:**
- Modify: `voice_core.py`
- Modify: `tests/test_voice_core.py`

**Interfaces:**
- Consumes: `voice_core.extract_voice_embedding`, `voice_core.match_voice_embedding` (Task 4); `audio_core.read_wav_pcm`, `audio_core.SAMPLE_RATE`, `audio_core.FRAME_BYTES` (existing); `config.VOICE_CLUSTER_DISTANCE_THRESHOLD` (Task 1); `database.create_unknown_voice`, `database.add_unknown_voice_sample`, `database.list_unknown_voice_samples_with_embeddings` (Task 2).
- Produces:
  - `voice_core.diarize_meeting_audio(wav_path, speech_regions) -> list[tuple[str, float, float]]` — the `(speaker_label, start_seconds, end_seconds)` list matching `transcription_core.transcribe_with_diarization`'s `diarize` callable contract. Labels are either an enrolled person's name or `"Unknown Speaker N"`.
  - `voice_core.find_matching_unknown_voice(query_embedding) -> Optional[dict]` with keys `unknown_voice_id`, `similarity` — mirrors `face_core.find_matching_unknown`.
  - `voice_core.register_unknown_voice(label, embedding) -> int` — mirrors `face_core.register_unknown`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_voice_core.py`:

```python
def test_diarize_meeting_audio_clusters_segments_by_similarity(tmp_path, monkeypatch):
    speaker_a = np.zeros(voice_core.EMBEDDING_DIM, dtype=np.float32)
    speaker_a[0] = 1.0
    speaker_b = np.zeros(voice_core.EMBEDDING_DIM, dtype=np.float32)
    speaker_b[1] = 1.0

    embeddings_by_region = {
        (0.0, 1.0): speaker_a,
        (1.0, 2.0): speaker_b,
        (2.0, 3.0): speaker_a,
    }

    def fake_extract(_pcm, sample_rate=16000):
        return None  # overridden per-call below

    calls = iter(embeddings_by_region.values())
    monkeypatch.setattr(voice_core, "extract_voice_embedding", lambda *_a, **_k: next(calls))
    monkeypatch.setattr(voice_core, "_read_region_pcm", lambda *_a, **_k: b"\x00\x00" * 8000)
    monkeypatch.setattr(voice_core, "match_voice_embedding", lambda *_a, **_k: None)

    result = voice_core.diarize_meeting_audio(
        tmp_path / "audio.wav",
        list(embeddings_by_region.keys()),
    )

    labels = [label for label, _start, _end in result]
    assert labels[0] == labels[2]
    assert labels[0] != labels[1]
    assert all(label.startswith("Unknown Speaker") for label in labels)


def test_diarize_meeting_audio_uses_enrolled_person_name(tmp_path, monkeypatch):
    embedding = np.zeros(voice_core.EMBEDDING_DIM, dtype=np.float32)
    embedding[0] = 1.0

    monkeypatch.setattr(voice_core, "extract_voice_embedding", lambda *_a, **_k: embedding)
    monkeypatch.setattr(voice_core, "_read_region_pcm", lambda *_a, **_k: b"\x00\x00" * 8000)
    monkeypatch.setattr(
        voice_core,
        "match_voice_embedding",
        lambda *_a, **_k: {"person_id": 7, "person_name": "Frank", "similarity": 0.9},
    )

    result = voice_core.diarize_meeting_audio(tmp_path / "audio.wav", [(0.0, 1.0)])

    assert result == [("Frank", 0.0, 1.0)]


def test_find_matching_unknown_voice_and_register(tmp_path, monkeypatch):
    monkeypatch.setattr(database, "DB_PATH", tmp_path / "faces.db")
    monkeypatch.setattr(config, "UNKNOWN_VOICES_DIR", tmp_path)
    database.init_db()

    embedding = np.zeros(voice_core.EMBEDDING_DIM, dtype=np.float32)
    embedding[0] = 1.0

    unknown_voice_id = voice_core.register_unknown_voice("Speaker 1", embedding)

    match = voice_core.find_matching_unknown_voice(embedding)
    assert match is not None
    assert match["unknown_voice_id"] == unknown_voice_id
    assert match["similarity"] > 0.99
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_voice_core.py -v`
Expected: FAIL — `AttributeError: module 'voice_core' has no attribute 'diarize_meeting_audio'`

- [ ] **Step 3: Implement clustering and unknown-voice helpers**

Append to `voice_core.py`:

```python
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist

from audio_core import read_wav_pcm, SAMPLE_RATE, SAMPLE_WIDTH
from database import (
    add_unknown_voice_sample,
    create_unknown_voice,
    list_unknown_voice_samples_with_embeddings,
)


def _read_region_pcm(wav_path, start_seconds, end_seconds):
    pcm = read_wav_pcm(wav_path)
    start_byte = int(start_seconds * SAMPLE_RATE) * SAMPLE_WIDTH
    end_byte = int(end_seconds * SAMPLE_RATE) * SAMPLE_WIDTH
    return pcm[start_byte:end_byte]


def find_matching_unknown_voice(query_embedding, threshold=None):
    threshold = config.VOICE_MATCH_THRESHOLD if threshold is None else threshold
    query = _normalize(query_embedding)
    if query is None:
        return None

    rows = list_unknown_voice_samples_with_embeddings()
    if not rows:
        return None

    scores = []
    for sample_id, unknown_voice_id, embedding_path, _quality in rows:
        path = Path(embedding_path)
        if not path.exists():
            continue
        try:
            stored = _normalize(np.load(path))
            if stored is None:
                continue
            scores.append((float(np.dot(query, stored)), unknown_voice_id, sample_id))
        except Exception:
            logger.exception("Failed loading unknown voice embedding: %s", path)

    if not scores:
        return None

    scores.sort(key=lambda item: item[0], reverse=True)
    best_similarity, best_unknown_voice_id, best_sample_id = scores[0]
    if best_similarity < threshold:
        return None

    return {
        "unknown_voice_id": best_unknown_voice_id,
        "similarity": best_similarity,
        "sample_id": best_sample_id,
    }


def register_unknown_voice(label, embedding):
    config.UNKNOWN_VOICES_DIR.mkdir(parents=True, exist_ok=True)
    safe_label = label.replace("/", "_").replace("\\", "_")
    embedding_path = config.UNKNOWN_VOICES_DIR / f"{safe_label}.npy"
    np.save(embedding_path, _normalize(embedding))
    return create_unknown_voice(label, embedding_path)


def diarize_meeting_audio(wav_path, speech_regions):
    """Cluster VAD speech regions into speakers and label each region.

    Returns [(speaker_label, start_seconds, end_seconds), ...] suitable for
    transcription_core.transcribe_with_diarization's `diarize` parameter.
    """
    embeddings = []
    valid_regions = []
    for start, end in speech_regions:
        pcm = _read_region_pcm(wav_path, start, end)
        embedding = extract_voice_embedding(pcm)
        if embedding is None:
            continue
        embeddings.append(embedding)
        valid_regions.append((start, end))

    if not embeddings:
        return []

    if len(embeddings) == 1:
        cluster_ids = [1]
    else:
        distances = pdist(np.vstack(embeddings), metric="cosine")
        linkage_matrix = linkage(distances, method="average")
        cluster_ids = fcluster(
            linkage_matrix,
            t=config.VOICE_CLUSTER_DISTANCE_THRESHOLD,
            criterion="distance",
        )

    cluster_centroids = {}
    for cluster_id, embedding in zip(cluster_ids, embeddings):
        cluster_centroids.setdefault(cluster_id, []).append(embedding)
    cluster_centroids = {
        cluster_id: _normalize(np.mean(vectors, axis=0))
        for cluster_id, vectors in cluster_centroids.items()
    }

    cluster_labels = {}
    unknown_counter = 0
    for cluster_id, centroid in cluster_centroids.items():
        match = match_voice_embedding(centroid) if centroid is not None else None
        if match is not None:
            cluster_labels[cluster_id] = match["person_name"]
            continue
        unknown_counter += 1
        cluster_labels[cluster_id] = f"Unknown Speaker {unknown_counter}"

    return [
        (cluster_labels[cluster_id], start, end)
        for cluster_id, (start, end) in zip(cluster_ids, valid_regions)
    ]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_voice_core.py -v`
Expected: PASS (8 tests)

- [ ] **Step 5: Commit**

```bash
git add voice_core.py tests/test_voice_core.py
git commit -m "feat: add diarization clustering and unknown-voice matching to voice_core"
```

---

## Task 6: Wire diarization into the meeting transcription pipeline

**Files:**
- Modify: `transcription_core.py`
- Modify: `tests/test_transcription_core.py`

**Interfaces:**
- Consumes: `voice_core.diarize_meeting_audio(wav_path, speech_regions)` (Task 5).
- Produces: `transcription_core.process_meeting_transcription(video_path, diarize=None)` now defaults `diarize` to a wrapper around `voice_core.diarize_meeting_audio` instead of `None`, and populates `TranscriptionSegment` with a `person_id` field (new field, default `None`) so `save_transcripts` can persist it.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_transcription_core.py`:

```python
def test_process_meeting_transcription_uses_voice_core_diarizer_by_default(tmp_path, monkeypatch):
    db_path = tmp_path / "meeting.db"
    monkeypatch.setattr(config, "DB_PATH", db_path)
    monkeypatch.setattr(database, "DB_PATH", db_path)
    monkeypatch.setattr(config, "TRANSCRIPTS_DIR", tmp_path / "transcripts")
    database.init_db()

    video_path = tmp_path / "meeting.mp4"
    video_path.write_bytes(b"video")

    def fake_extract(_video_path, output_path):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"audio")
        return output_path

    monkeypatch.setattr("transcription_core.extract_audio_from_video", fake_extract)
    monkeypatch.setattr("transcription_core.detect_speech_segments", lambda *_a, **_k: [(0.0, 1.5)])

    diarize_calls = []

    def fake_diarize(wav_path, speech_regions):
        diarize_calls.append((wav_path, speech_regions))
        return [("Gina", 0.0, 1.5)]

    monkeypatch.setattr("voice_core.diarize_meeting_audio", fake_diarize)

    class FakeModel:
        def transcribe(self, path, verbose=False, **kwargs):
            return {"segments": [{"start": 0.1, "end": 1.0, "text": "hi", "avg_logprob": -0.1}]}

    fake_whisper = type("FakeWhisperModule", (), {"load_model": staticmethod(lambda *_a, **_k: FakeModel())})
    monkeypatch.setitem(__import__("sys").modules, "whisper", fake_whisper)

    meeting_id = process_meeting_transcription(video_path)

    assert diarize_calls  # the default diarizer was invoked
    with database.get_conn() as connection:
        row = connection.execute(
            "SELECT speaker_label FROM transcription_segments WHERE meeting_id=?",
            (meeting_id,),
        ).fetchone()
    assert row[0] == "Gina"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_transcription_core.py::test_process_meeting_transcription_uses_voice_core_diarizer_by_default -v`
Expected: FAIL — diarize_calls is empty, because no default diarizer is wired yet.

- [ ] **Step 3: Wire the default diarizer**

In `transcription_core.py`, add the import:

```python
import voice_core
```

In `transcribe_with_diarization`, change the `diarize` parameter default handling — replace:

```python
    diarization = list(diarize(str(audio_path))) if diarize else []
```

with:

```python
    active_diarize = diarize if diarize is not None else voice_core.diarize_meeting_audio
    try:
        diarization = list(active_diarize(audio_path, speech_regions))
    except Exception:
        logger.exception("Diarization failed; falling back to Unknown Speaker labels")
        diarization = []
```

(`speech_regions` is the variable already introduced in Task 3's implementation.)

Also update the `diarize` parameter's type hint on `transcribe_with_diarization`'s signature from
`Optional[Callable[[str], Iterable[tuple[str, float, float]]]]` to
`Optional[Callable[[Path, list[tuple[float, float]]], Iterable[tuple[str, float, float]]]]`
to reflect the new two-argument contract, and update the function's docstring line
`` ``diarize`` must yield ``(speaker_label, start_seconds, end_seconds)``. `` to mention it
now receives `(audio_path, speech_regions)`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_transcription_core.py -v`
Expected: PASS (all tests)

- [ ] **Step 5: Commit**

```bash
git add transcription_core.py tests/test_transcription_core.py
git commit -m "feat: wire voice_core diarizer as the default meeting diarizer"
```

---

## Task 7: Populate `person_id` and derive speaker labels from `persons.name`

**Files:**
- Modify: `transcription_core.py`
- Modify: `tests/test_transcription_core.py`

**Interfaces:**
- Consumes: `database.get_conn()` (existing).
- Produces: `transcription_core.TranscriptionSegment` gains a `person_id: Optional[int] = None` field; `save_transcripts` writes `person_id` into `transcription_segments.person_id`; a new `transcription_core.resolve_speaker_name(label) -> tuple[Optional[int], str]` helper looks up whether `label` matches an existing `persons.name` (used so `voice_core`-labeled segments carrying an enrolled person's name get their `person_id` populated even when the diarizer only returns a name string).

- [ ] **Step 1: Write the failing test**

Add to `tests/test_transcription_core.py`:

```python
def test_process_meeting_transcription_populates_person_id_for_known_speaker(tmp_path, monkeypatch):
    db_path = tmp_path / "meeting.db"
    monkeypatch.setattr(config, "DB_PATH", db_path)
    monkeypatch.setattr(database, "DB_PATH", db_path)
    monkeypatch.setattr(config, "TRANSCRIPTS_DIR", tmp_path / "transcripts")
    database.init_db()
    person_id = database.create_person("Henry")

    video_path = tmp_path / "meeting.mp4"
    video_path.write_bytes(b"video")

    def fake_extract(_video_path, output_path):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"audio")
        return output_path

    monkeypatch.setattr("transcription_core.extract_audio_from_video", fake_extract)
    monkeypatch.setattr(
        "transcription_core.transcribe_with_diarization",
        lambda *_a, **_k: [TranscriptionSegment("Henry", 0, 1000, "hello", 0.9)],
    )

    meeting_id = process_meeting_transcription(video_path)

    with database.get_conn() as connection:
        row = connection.execute(
            "SELECT speaker_label, person_id FROM transcription_segments WHERE meeting_id=?",
            (meeting_id,),
        ).fetchone()

    assert row == ("Henry", person_id)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_transcription_core.py::test_process_meeting_transcription_populates_person_id_for_known_speaker -v`
Expected: FAIL — `person_id` column is `None` because nothing populates it yet.

- [ ] **Step 3: Implement `person_id` population**

In `transcription_core.py`, add:

```python
def resolve_speaker_name(label):
    """Look up person_id for a speaker label that matches an enrolled person's name."""
    with sqlite3.connect(config.DB_PATH) as connection:
        row = connection.execute(
            "SELECT person_id FROM persons WHERE name=?",
            (label,),
        ).fetchone()
    return (int(row[0]) if row else None), label
```

In `save_transcripts`, change the `INSERT INTO transcription_segments` block's `executemany` values to resolve and include `person_id`:

```python
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
                        resolve_speaker_name(segment.speaker_label)[0],
                        segment.start_ms,
                        segment.end_ms,
                        apply_text_cleanup(segment.text),
                        segment.confidence,
                    )
                    for segment in segments
                ],
            )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_transcription_core.py -v`
Expected: PASS (all tests)

- [ ] **Step 5: Commit**

```bash
git add transcription_core.py tests/test_transcription_core.py
git commit -m "feat: populate transcription_segments.person_id from resolved speaker names"
```

---

## Task 8: Retroactive relabeling when an unknown voice is resolved or a person is renamed

**Files:**
- Modify: `database.py`
- Test: `tests/test_database_voice.py`

**Interfaces:**
- Consumes: `voice_core` unknown-voice cluster label conventions (`"Unknown Speaker N"`), existing `resolve_unknown_voice`, `rename_person`.
- Produces: `database.resolve_unknown_voice(unknown_voice_id, person_id)` now also updates every `transcription_segments` / `audio_logs` row whose `speaker_label`/`person_name` equals that unknown voice's `label`, setting `person_id` and rewriting the label to the resolved person's name. `database.rename_person(person_id, name)` now also updates `transcription_segments.speaker_label` / `audio_logs.person_name` for rows already carrying that `person_id`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_database_voice.py`:

```python
def test_resolve_unknown_voice_relabels_existing_transcript_segments(tmp_path, monkeypatch):
    _setup_database(tmp_path, monkeypatch)
    person_id = database.create_person("Ivy")
    embedding_path = tmp_path / "cluster.npy"
    embedding_path.write_bytes(b"embedding")
    unknown_voice_id = database.create_unknown_voice("Unknown Speaker 1", embedding_path)

    with database.get_conn() as conn:
        conn.execute(
            "INSERT INTO meetings(video_path, created_at, transcription_status) VALUES (?, ?, 'completed')",
            ("video.mp4", database.utc_now()),
        )
        meeting_id = conn.execute("SELECT meeting_id FROM meetings").fetchone()[0]
        conn.execute(
            """
            INSERT INTO transcription_segments(
                meeting_id, speaker_label, start_ms, end_ms, text, created_at
            ) VALUES (?, 'Unknown Speaker 1', 0, 1000, 'hi', ?)
            """,
            (meeting_id, database.utc_now()),
        )

    database.resolve_unknown_voice(unknown_voice_id, person_id)

    with database.get_conn() as conn:
        row = conn.execute(
            "SELECT speaker_label, person_id FROM transcription_segments WHERE meeting_id=?",
            (meeting_id,),
        ).fetchone()

    assert row == ("Ivy", person_id)


def test_rename_person_relabels_existing_transcript_segments(tmp_path, monkeypatch):
    _setup_database(tmp_path, monkeypatch)
    person_id = database.create_person("Old Name")

    with database.get_conn() as conn:
        conn.execute(
            "INSERT INTO meetings(video_path, created_at, transcription_status) VALUES (?, ?, 'completed')",
            ("video.mp4", database.utc_now()),
        )
        meeting_id = conn.execute("SELECT meeting_id FROM meetings").fetchone()[0]
        conn.execute(
            """
            INSERT INTO transcription_segments(
                meeting_id, speaker_label, person_id, start_ms, end_ms, text, created_at
            ) VALUES (?, 'Old Name', ?, 0, 1000, 'hi', ?)
            """,
            (meeting_id, person_id, database.utc_now()),
        )

    database.rename_person(person_id, "New Name")

    with database.get_conn() as conn:
        row = conn.execute(
            "SELECT speaker_label FROM transcription_segments WHERE meeting_id=?",
            (meeting_id,),
        ).fetchone()

    assert row[0] == "New Name"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_database_voice.py -v`
Expected: FAIL — both new tests fail because labels stay `"Unknown Speaker 1"` / `"Old Name"`.

- [ ] **Step 3: Implement relabeling**

In `database.py`, modify `resolve_unknown_voice`:

```python
def resolve_unknown_voice(unknown_voice_id, person_id):
    with get_conn() as conn:
        label_row = conn.execute(
            "SELECT label FROM unknown_voices WHERE unknown_voice_id=?",
            (int(unknown_voice_id),),
        ).fetchone()
        conn.execute(
            "UPDATE unknown_voices SET resolved_person_id=? WHERE unknown_voice_id=?",
            (person_id, unknown_voice_id),
        )
        if label_row is None:
            return
        label = label_row[0]
        person_name = conn.execute(
            "SELECT name FROM persons WHERE person_id=?",
            (person_id,),
        ).fetchone()[0]
        conn.execute(
            "UPDATE transcription_segments SET speaker_label=?, person_id=? WHERE speaker_label=?",
            (person_name, person_id, label),
        )
        conn.execute(
            "UPDATE audio_logs SET person_name=?, person_id=? WHERE person_name=?",
            (person_name, person_id, label),
        )
```

And modify `rename_person`:

```python
def rename_person(person_id, name):
    name = name.strip()
    if not name:
        raise ValueError("Name cannot be empty")
    with get_conn() as conn:
        conn.execute(
            "UPDATE persons SET name=?, updated_at=? WHERE person_id=?",
            (name, utc_now(), person_id),
        )
        conn.execute(
            "UPDATE transcription_segments SET speaker_label=? WHERE person_id=?",
            (name, person_id),
        )
        conn.execute(
            "UPDATE audio_logs SET person_name=? WHERE person_id=?",
            (name, person_id),
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_database_voice.py -v`
Expected: PASS (7 tests)

- [ ] **Step 5: Commit**

```bash
git add database.py tests/test_database_voice.py
git commit -m "feat: relabel existing transcripts/audio logs on voice resolve and person rename"
```

---

## Task 9: Enrollment and unknown-voice review UI in `app.py`

**Files:**
- Modify: `app.py`

**Interfaces:**
- Consumes: `database.list_unknown_voices`, `database.list_unknown_voice_samples`, `database.resolve_unknown_voice`, `database.delete_unknown_voice`, `database.add_voice_embedding`, `database.create_person` (existing/Task 2); `voice_core.extract_voice_embedding` (Task 4).
- Produces: a new `render_voice_enrollment()` function called from `main()`, added as its own sidebar/page section next to the existing unknown-face review page.

- [ ] **Step 1: Add the imports**

At the top of `app.py`, alongside the existing `from database import (...)` block, add:

```python
from database import (
    add_voice_embedding,
    delete_unknown_voice,
    list_unknown_voice_samples,
    list_unknown_voices,
    resolve_unknown_voice,
)
import voice_core
```

- [ ] **Step 2: Implement the review/enrollment section**

Add this function to `app.py` (placed near `render_transcripts`, following its structure):

```python
def render_voice_enrollment():
    st.header("🎙️ Voice Enrollment")

    persons = {name: person_id for person_id, name, _created, _updated in database.list_persons()}

    st.subheader("Enroll a clean voice sample")
    selected_name = st.selectbox(
        "Person",
        ["-- Select --"] + list(persons.keys()),
        key="voice_enroll_person",
    )
    uploaded = st.file_uploader("Upload a short WAV sample (16kHz mono PCM16)", type=["wav"])
    if st.button("➕ Add Voice Sample", key="add_voice_sample"):
        if selected_name == "-- Select --":
            st.warning("Please select a person first.")
        elif uploaded is None:
            st.warning("Please upload a WAV file first.")
        else:
            import wave

            with wave.open(uploaded, "rb") as wf:
                pcm = wf.readframes(wf.getnframes())
            embedding = voice_core.extract_voice_embedding(pcm)
            if embedding is None:
                st.error("Could not extract a voice embedding from this sample.")
            else:
                person_id = persons[selected_name]
                embedding_dir = config.KNOWN_FACES_DIR.parent / "known_voices" / str(person_id)
                embedding_dir.mkdir(parents=True, exist_ok=True)
                embedding_path = embedding_dir / f"{selected_name}_{len(list(embedding_dir.glob('*.npy')))}.npy"
                import numpy as np

                np.save(embedding_path, embedding)
                add_voice_embedding(person_id, embedding_path, quality=1.0)
                st.success(f"Voice sample added for {selected_name}.")
                st.rerun()

    st.subheader("Unresolved unknown voices")
    unknown_voices = list_unknown_voices()
    if not unknown_voices:
        st.info("No unresolved unknown voices.")
        return

    for unknown_voice_id, label, embedding_path, created_at, _resolved in unknown_voices:
        with st.container(border=True):
            st.markdown(f"### ❓ {label} (#{unknown_voice_id})")
            st.caption(f"Created: {created_at}")

            if st.button("🗑️ Delete", key=f"delete_unknown_voice_{unknown_voice_id}"):
                if delete_unknown_voice(unknown_voice_id):
                    st.rerun()
                else:
                    st.warning("This unknown voice is no longer unresolved.")

            assign_name = st.selectbox(
                "Assign to person",
                ["-- Select --"] + list(persons.keys()),
                key=f"assign_voice_{unknown_voice_id}",
            )
            if st.button("✅ Assign", key=f"assign_voice_btn_{unknown_voice_id}"):
                if assign_name == "-- Select --":
                    st.warning("Please select a person first.")
                else:
                    person_id = persons[assign_name]
                    resolve_unknown_voice(unknown_voice_id, person_id)
                    import numpy as np

                    embedding = np.load(embedding_path)
                    add_voice_embedding(person_id, embedding_path, quality=0.7)
                    st.success(f"{label} assigned to {assign_name}.")
                    st.rerun()
```

- [ ] **Step 3: Call it from `main()`**

In `app.py`'s `main()`, alongside the existing call to whatever renders the unknown-face review page, add a call to `render_voice_enrollment()` (e.g., as an additional `st.tabs`/section entry — follow however the existing pages are already switched between, and add `"Voice Enrollment"` as a new option in that same mechanism).

- [ ] **Step 4: Manual smoke test**

Run: `streamlit run app.py` and confirm the new "Voice Enrollment" section renders without exceptions, an unresolved unknown voice (created via a quick `python -c` call to `database.create_unknown_voice`/`add_unknown_voice_sample` in a throwaway DB) can be listed, and clicking "Delete" removes it.

- [ ] **Step 5: Commit**

```bash
git add app.py
git commit -m "feat: add voice enrollment and unknown-voice review UI"
```

---

## Task 10: Realtime voice fallback for active-speaker identification

**Files:**
- Modify: `audio_core.py` (rolling PCM buffer on `RealtimeVAD`)
- Modify: `realtime.py` (active-speaker selection fallback)
- Test: `tests/test_audio_core.py`

**Interfaces:**
- Consumes: `voice_core.extract_voice_embedding`, `voice_core.match_voice_embedding` (Task 4).
- Produces: `RealtimeVAD.get_recent_pcm(seconds=1.5) -> bytes` — returns the last N seconds of raw PCM16 audio captured by the worker thread.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_audio_core.py`:

```python
def test_realtime_vad_get_recent_pcm_returns_trailing_buffer():
    from audio_core import RealtimeVAD, FRAME_BYTES

    vad = RealtimeVAD()
    vad._pcm_buffer = bytearray(FRAME_BYTES * 100)
    vad._append_pcm(b"\x01\x02" * (FRAME_BYTES // 2))

    recent = vad.get_recent_pcm(seconds=0.032)

    assert recent == b"\x01\x02" * (FRAME_BYTES // 2)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_audio_core.py::test_realtime_vad_get_recent_pcm_returns_trailing_buffer -v`
Expected: FAIL — `AttributeError: 'RealtimeVAD' object has no attribute '_pcm_buffer'`

- [ ] **Step 3: Implement the rolling buffer**

In `audio_core.py`'s `RealtimeVAD.__init__`, add:

```python
        self._pcm_buffer = bytearray()
        self._pcm_buffer_lock = threading.Lock()
        # Cap the buffer so memory use stays bounded regardless of session length.
        self._pcm_buffer_max_bytes = SAMPLE_RATE * SAMPLE_WIDTH * 5
```

Add these methods to `RealtimeVAD`:

```python
    def _append_pcm(self, frame_bytes):
        with self._pcm_buffer_lock:
            self._pcm_buffer.extend(frame_bytes)
            overflow = len(self._pcm_buffer) - self._pcm_buffer_max_bytes
            if overflow > 0:
                del self._pcm_buffer[:overflow]

    def get_recent_pcm(self, seconds=1.5):
        with self._pcm_buffer_lock:
            wanted_bytes = int(seconds * SAMPLE_RATE) * SAMPLE_WIDTH
            return bytes(self._pcm_buffer[-wanted_bytes:])
```

In `RealtimeVAD._run`, right after `data, _overflowed = stream.read(FRAME_SAMPLES)`, add:

```python
                self._append_pcm(bytes(data))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_audio_core.py -v`
Expected: PASS (all tests)

- [ ] **Step 5: Commit**

```bash
git add audio_core.py tests/test_audio_core.py
git commit -m "feat: buffer recent realtime microphone PCM for voice fallback matching"
```

- [ ] **Step 6: Wire the fallback into `realtime.py`'s active-speaker selection**

In `realtime.py`, in the active-speaker block (the section shown around the `if audio_available and audio_state.value and candidates:` block), after `speaker = max(candidates, key=lambda t: t.lip_open)`, insert the fallback before checking `speaker.person_id is not None`:

```python
                if speaker.person_id is None or speaker.confidence < config.RECOGNITION_THRESHOLD:
                    recent_pcm = vad.get_recent_pcm(seconds=1.5) if vad is not None else b""
                    voice_embedding = voice_core.extract_voice_embedding(recent_pcm)
                    voice_match = (
                        voice_core.match_voice_embedding(voice_embedding)
                        if voice_embedding is not None
                        else None
                    )
                    if voice_match is not None:
                        speaker.person_id = voice_match["person_id"]
                        speaker.person_name = voice_match["person_name"]
                        speaker.confidence = voice_match["similarity"]
```

Add `import voice_core` and `import config` (if not already imported) at the top of `realtime.py`. This only fills in identity when the face signal was missing/weak — it never runs when `speaker.person_id is not None and speaker.confidence >= config.RECOGNITION_THRESHOLD`, satisfying the "never override a confident face match" constraint.

- [ ] **Step 7: Manual verification**

Run: `pytest tests/test_realtime.py -v` to confirm no regressions in the existing active-speaker/VAD wiring tests (they mock `vad_factory`/`RealtimeVAD`, so this change must not break their mocks — if a test constructs a `Mock()` in place of `RealtimeVAD`, confirm `get_recent_pcm` is only called through the real object, guarded by `vad is not None`).
Expected: PASS

- [ ] **Step 8: Commit**

```bash
git add realtime.py
git commit -m "feat: fall back to voice matching for realtime active-speaker identification"
```

---

## Task 11: Dependency and final verification

**Files:**
- Modify: `requirements.txt`

**Interfaces:**
- None (final integration task).

- [ ] **Step 1: Add the new dependency**

Add to `requirements.txt`, after `openai-whisper>=20231117`:

```
speechbrain>=1.0
```

- [ ] **Step 2: Install and run the full test suite**

Run: `pip install -r requirements.txt`
Run: `pytest -v`
Expected: All tests PASS (existing suite + all new tests from Tasks 2-10).

- [ ] **Step 3: Commit**

```bash
git add requirements.txt
git commit -m "chore: add speechbrain dependency for voice diarization/recognition"
```
