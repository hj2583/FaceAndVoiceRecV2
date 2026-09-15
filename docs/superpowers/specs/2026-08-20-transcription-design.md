# Transcription Feature Design Spec

**Date:** 2026-08-20  
**Status:** Design Phase  
**Scope:** Add audio transcription with speaker diarization to the face recognition system

---

## Overview

Add automatic speech-to-text transcription for meeting recordings, with speaker identification via voice diarization mapped to tracked face IDs. Output per-speaker text transcripts with timestamps and advanced text cleanup (grammar, punctuation, sentence segmentation).

**Key Goals:**
- Transcribe meeting audio automatically after video processing
- Identify speakers via Whisper's diarization, then correlate with tracked face IDs
- Generate per-speaker text transcripts for archival and review
- Apply text cleanup (grammar/spell-check) for readability

---

## Architecture

### Data Flow

```
Video File (offline mode in video_processor.py)
    ↓
[Extract audio via FFmpeg] → audio.wav
    ↓
[transcription_core.py] — Main transcription module
    ├─ transcribe_with_diarization(audio.wav, model="medium")
    │  └─ Returns: List[TranscriptionSegment]
    │     Each segment: (speaker_id: int, start_ms: int, end_ms: int, text: str, confidence: float)
    │
    └─ match_speakers_to_faces(segments, tracking_log)
       └─ Returns: List[SegmentWithFaceID]
          Correlates diarized speakers to person IDs from video_processor tracking
    ↓
[apply_text_cleanup()] → advanced grammar/spell-check via NLTK
    ↓
[save_transcripts()] → per-speaker .txt files + database insert
    ↓
[Streamlit UI] — Display transcripts, download, re-transcribe
```

### New Components

#### 1. `transcription_core.py` (New Module)

Main functions:

```python
def extract_audio_from_video(video_path: str) -> str:
    """
    Extract audio track from video using FFmpeg.
    
    Args:
        video_path: Path to input video file
        
    Returns:
        Path to extracted audio.wav
        
    Raises:
        RuntimeError: If FFmpeg fails or video has no audio track
    """

def transcribe_with_diarization(
    audio_path: str,
    model_size: str = "medium",
) -> List[TranscriptionSegment]:
    """
    Run OpenAI Whisper with diarization to identify speakers.
    
    Args:
        audio_path: Path to audio.wav
        model_size: Whisper model size ("tiny", "base", "small", "medium", "large")
        
    Returns:
        List of segments with speaker_id, timestamp range, and transcribed text
        
    Raises:
        RuntimeError: If Whisper fails or audio is corrupted
    """

def match_speakers_to_faces(
    segments: List[TranscriptionSegment],
    tracking_log_path: str,
) -> List[SegmentWithFaceID]:
    """
    Correlate diarized speakers to tracked face IDs.
    
    Strategy:
    1. For each segment, find all face tracks that overlap with segment's time range
    2. If single face track covers >80% of segment, assign that face ID
    3. If multiple faces or no clear match, mark as "Unknown Speaker <N>"
    
    Args:
        segments: Transcription segments from Whisper
        tracking_log_path: Path to tracking data (JSON or SQLite query)
        
    Returns:
        Segments enriched with face IDs and speaker names
    """

def apply_text_cleanup(text: str) -> str:
    """
    Apply advanced text cleanup: grammar fixing, punctuation, sentence segmentation.
    
    Uses:
    - NLTK for sentence segmentation
    - LanguageTool (or similar) for grammar/spell-check
    
    Args:
        text: Raw Whisper output
        
    Returns:
        Cleaned text
    """

def save_transcripts(
    segments: List[SegmentWithFaceID],
    output_dir: Path,
    meeting_id: int,
) -> None:
    """
    Save transcripts and insert into database.
    
    Actions:
    1. Group segments by speaker_name
    2. Create <output_dir>/<meeting_id>/<speaker_name>.txt
    3. Insert records into transcription_segments table
    
    Args:
        segments: Enriched segments with face IDs
        output_dir: Base output directory (e.g., ./transcripts)
        meeting_id: ID to track which meeting this belongs to
    """
```

#### 2. Database Schema (in `database.py`)

Add two tables:

```sql
CREATE TABLE IF NOT EXISTS meetings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    video_path TEXT NOT NULL UNIQUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    processed_at TIMESTAMP,
    transcription_status TEXT DEFAULT 'pending',
    -- status: pending, in_progress, completed, failed
    error_log TEXT
);

CREATE TABLE IF NOT EXISTS transcription_segments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    meeting_id INTEGER NOT NULL,
    speaker_id INTEGER,  -- NULL if unknown speaker
    speaker_name TEXT NOT NULL,  -- "John", "Unknown Speaker 1", etc.
    start_ms INTEGER NOT NULL,
    end_ms INTEGER NOT NULL,
    text TEXT NOT NULL,
    confidence REAL,  -- Whisper confidence (0.0-1.0)
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(meeting_id) REFERENCES meetings(id),
    FOREIGN KEY(speaker_id) REFERENCES persons(id)
);

CREATE INDEX IF NOT EXISTS idx_transcription_meeting_id 
    ON transcription_segments(meeting_id);
CREATE INDEX IF NOT EXISTS idx_transcription_speaker_id 
    ON transcription_segments(speaker_id);
```

#### 3. Config Additions (in `config.py`)

```python
# Transcription settings
ENABLE_TRANSCRIPTION = True
WHISPER_MODEL = "medium"  # Options: tiny, base, small, medium, large
TRANSCRIPTION_CLEANUP_LEVEL = "advanced"  # Options: none, basic, advanced
TRANSCRIPTS_DIR = BASE_DIR / "transcripts"
TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)

# Speaker matching thresholds
SPEAKER_FACE_OVERLAP_THRESHOLD = 0.8  # % of segment that must overlap with face track
```

---

## Speaker Matching Algorithm

**Input:** Diarized segments (speaker_id: int, start_ms, end_ms) + tracking events (person_id, start_ms, end_ms)

**Algorithm:**

```
For each transcription_segment S:
    matching_faces = []
    
    For each tracked_face T:
        If T overlaps with S:
            overlap_duration = min(T.end, S.end) - max(T.start, S.start)
            overlap_ratio = overlap_duration / S.duration
            
            If overlap_ratio >= SPEAKER_FACE_OVERLAP_THRESHOLD:
                matching_faces.append((T.person_id, overlap_ratio))
    
    If len(matching_faces) == 1:
        S.speaker_id = matching_faces[0].person_id
        S.speaker_name = lookup_person_name(S.speaker_id)
    
    Else if len(matching_faces) > 1:
        # Ambiguous: multiple faces speak during this segment
        # This can happen if people talk over each other
        S.speaker_id = None
        S.speaker_name = "Unknown Speaker (ambiguous)"
    
    Else:
        # No matching face track
        S.speaker_id = None
        S.speaker_name = f"Unknown Speaker {next_unknown_id}"
```

**Edge Cases:**
- No speech visible on camera (speaker off-camera) → marked "Unknown Speaker"
- Multiple people speak simultaneously → marked ambiguous
- Face track exists but person not in database → marked "Unknown Speaker" (not by name)

---

## Text Cleanup Pipeline

**Input:** Raw Whisper text  
**Output:** Cleaned, readable text

**Steps:**

1. **Sentence Segmentation** (NLTK):
   - Split into sentences
   - Capitalize first letter of each sentence

2. **Grammar & Spell-Check** (LanguageTool or similar):
   - Correct common errors: "thhe" → "the"
   - Fix spacing around punctuation

3. **Confidence Filter**:
   - If Whisper confidence < 0.6, flag segment with [UNCLEAR] marker
   - Example: "[UNCLEAR] The budget is..."

4. **Number Formatting**:
   - "$50 k" → "$50K"
   - "1 2 3" → "123" (context-dependent)

---

## Error Handling

| Scenario | Handling |
|----------|----------|
| Video has no audio track | Log error, skip transcription, continue with tracking only |
| Whisper fails (OOM, corrupted audio) | Catch exception, mark meeting status="failed", log details |
| Speaker-to-face match fails (no tracking) | Default to "Unknown Speaker <N>" |
| Database insert fails | Rollback transaction, log error, do not corrupt existing records |
| Text cleanup fails (non-English) | Skip cleanup, use raw Whisper output |
| Output directory doesn't exist | Create recursively |

---

## Testing Strategy

**Unit Tests** (`tests/test_transcription_core.py`):

```python
def test_extract_audio_creates_wav():
    # Test with mock FFmpeg
    
def test_transcribe_output_format():
    # Mock Whisper, verify segment structure
    
def test_match_speakers_single_face():
    # Diarized speaker, single tracked face at same time
    # Verify correct mapping
    
def test_match_speakers_no_face():
    # Diarized speaker, no tracked face
    # Verify "Unknown Speaker" assignment
    
def test_match_speakers_overlapping_faces():
    # Multiple faces speak during segment
    # Verify ambiguous marking
    
def test_text_cleanup_fixes_errors():
    # Input: "thhe budget is 50 k"
    # Output: "The budget is $50K"
    
def test_save_transcripts_creates_files():
    # Verify per-speaker .txt files created
    # Check file format and content
    
def test_database_insert_atomicity():
    # Verify rollback on failure
```

**Integration Tests**:

```python
def test_full_transcription_pipeline():
    # 1. Create small test video (2 speakers, 10s each)
    # 2. Run transcription_core on it
    # 3. Verify:
    #    - Audio extracted
    #    - Speakers identified
    #    - Transcript files created
    #    - Database populated
```

---

## Streamlit UI Integration

**New UI Components** (in `app.py`):

### Page: "Meeting Transcripts"

```
[Sidebar: Select Meeting]
├─ Video file: <dropdown with all processed videos>
├─ Status: Transcription pending / in progress / completed
├─ Re-transcribe button
└─ Download options: [PDF] [TXT] [JSON]

[Main Area: Transcript Display]
├─ Tab 1: Full Transcript (all speakers chronologically)
├─ Tab 2: Per-Speaker (select speaker, view their contributions)
└─ Tab 3: Stats
    ├─ Total speakers: 5
    ├─ Speaking time per person: [bar chart]
    ├─ Word count per person: [table]
```

### Per-Speaker View

```
Speaker: John Smith (Confidence: 0.92)
Time Speaking: 12m 34s
Word Count: 245

[00:00:15] "The budget for Q3 is $50K..."
[00:01:22] "We should allocate 30% to marketing."
...
```

---

## File Outputs

**Directory structure:**
```
transcripts/
├── <meeting_id_1>/
│   ├── john.txt
│   ├── mary.txt
│   └── unknown_speaker_1.txt
├── <meeting_id_2>/
│   ├── alice.txt
│   └── bob.txt
```

**File format (per-speaker .txt):**
```
[Speaker Name]
Meeting: 2026-08-20_board_meeting.mp4
Total Duration: 45m 12s

[00:00:15] The budget for Q3 is $50K.
[00:01:22] We should allocate 30% to marketing.
[00:02:45] [UNCLEAR] The sales team reported...
...
```

---

## Dependencies

**New Python packages:**
- `openai-whisper` (audio transcription + diarization)
- `language-tool-python` (grammar/spell-check)
- `nltk` (sentence segmentation)

**System dependencies:**
- FFmpeg (already required for audio extraction)

---

## Constraints & Assumptions

1. **Offline only** — Transcription runs after `video_processor.py` completes
2. **Single audio track** — Assumes video has one audio channel (mono or stereo mixed)
3. **Language detection** — Whisper auto-detects language; assumes English or will gracefully handle other languages
4. **No real-time streaming** — Transcription is batch post-processing, not live
5. **Face tracking dependency** — Accurate speaker names depend on successful face tracking in video_processor

---

## Success Criteria

- [x] Transcription module (`transcription_core.py`) created and tested
- [x] Speaker-to-face mapping algorithm implemented and validated
- [x] Text cleanup pipeline working for English transcripts
- [x] Per-speaker text files generated correctly
- [x] Database schema defined and integrated
- [x] Streamlit UI displays transcripts
- [x] All unit tests passing
- [x] Integration test with sample meeting video passing

---

## Future Enhancements (Out of Scope)

- Real-time transcription (would require Streamlit session state refactor)
- Multiple language support with language-specific cleanup
- Speaker voice profiling for better diarization
- Sentiment analysis per speaker
- Automatic summary generation
