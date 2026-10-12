import os
from pathlib import Path


# ============================================================
# Base paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DB_PATH = BASE_DIR / "faces.db"
INDEX_PATH = BASE_DIR / "face_index.npz"

KNOWN_FACES_DIR = BASE_DIR / "known_faces"
UNKNOWN_FACES_DIR = BASE_DIR / "unknown_faces"
UNKNOWN_VOICES_DIR = BASE_DIR / "unknown_voices"
INITIAL_VIDEO_DIR = BASE_DIR / "initialVideo"
TRACKED_VIDEO_DIR = BASE_DIR / "trackedVideo"
WORKFLOW_TEMP_DIR = BASE_DIR / ".workflow-temp"
LIVE_VIDEO_DIR = BASE_DIR / "live"
LOG_DIR = BASE_DIR / "logs"


# ============================================================
# Create required directories
# ============================================================

for directory in (
    KNOWN_FACES_DIR,
    UNKNOWN_FACES_DIR,
    UNKNOWN_VOICES_DIR,
    INITIAL_VIDEO_DIR,
    TRACKED_VIDEO_DIR,
    LIVE_VIDEO_DIR,
    LOG_DIR,
):
    directory.mkdir(
        parents=True,
        exist_ok=True,
    )


# ============================================================
# Face embedding
# ============================================================

# ArcFace normally produces 512-dimensional embeddings.
EMBEDDING_DIM = 512


# ============================================================
# Face recognition
# ============================================================

# Minimum cosine similarity required to accept
# a known identity.
#
# Higher:
#   Safer
#   Fewer false matches
#
# Lower:
#   More tolerant
#   Higher risk of wrong identity
#
# Start at 0.70 and calibrate using your own camera.
RECOGNITION_THRESHOLD = 0.70


# If the best and second-best matches are too close,
# recognition can be considered ambiguous.
AMBIGUITY_MARGIN = 0.05


# Recognition is computationally expensive.
# Re-run ArcFace recognition every N frames.
RECOGNITION_INTERVAL = 30


# ============================================================
# Face detection / tracking
# ============================================================

# Minimum face size allowed into the tracking pipeline. This is intentionally
# lower than the recognition gate so distant faces can still be detected and
# tracked while remaining Unknown when they lack enough detail.
MIN_FACE_SIZE = 12

# Minimum face size before attempting ArcFace recognition.
# Lowered to allow best-effort recognition attempts on small/distant
# faces detected via the long-range tiled pipeline. Low-confidence
# matches still fall back to "Unknown" through the existing thresholds.
MIN_RECOGNITION_FACE_SIZE = 15

# Number of faces MediaPipe should detect.
MAX_FACES = 8

# Minimum face size required for reliable MediaPipe lip landmarks.
LANDMARK_MIN_FACE_SIZE = 60

# Tracker settings.
TRACK_MAX_MISSED_FRAMES = 20
TRACK_DISTANCE_PX = 180

# Recognition debugging
DEBUG_FACE_SIZE = True


# ============================================================
# Long-range tiled detection (video_processor.py only)
# ============================================================

# Tile grid used to split a frame: (columns, rows).
TILE_GRID = (3, 2)

# Fractional overlap between adjacent tiles.
TILE_OVERLAP = 0.2

# Upscale factor applied to each tile before detection.
TILE_UPSCALE = 2.0

# Run tiled detection every N frames and rely on tracker continuity between.
TILED_DETECTION_INTERVAL = 5

# IoU threshold for de-duplicating overlapping tile detections.
NMS_IOU_THRESHOLD = 0.4


# ============================================================
# Small-face recognition enhancement
# ============================================================

# If the face is below this size, enlarge the crop
# before passing it to ArcFace.
FACE_UPSCALE_THRESHOLD = 100


# Enlargement factor.
#
# Example:
# 60x60 face → 120x120 crop
FACE_UPSCALE_FACTOR = 2.0

# Target minimum crop dimension for ArcFace input.
FACE_UPSCALE_TARGET_SIZE = 112


# ============================================================
# Unknown face handling
# ============================================================

# Number of recognition observations before an unknown
# face can be registered.
UNKNOWN_CONFIRMATIONS = 5


# Minimum quality required before saving an unknown sample.
UNKNOWN_MIN_QUALITY = 0.35


# Maximum number of samples stored for one unknown identity.
UNKNOWN_MAX_SAMPLES = 6


# Re-capture an unknown face every N frames.
UNKNOWN_CAPTURE_INTERVAL = 30


# Minimum number of tracking observations before
# unknown-face sampling begins.
UNKNOWN_MIN_TRACK_FRAMES = 10


# Minimum physical face size for unknown-face image capture.
UNKNOWN_MIN_FACE_SIZE = 70


# Matching threshold when comparing an unknown face
# with previously stored unknown identities.
#
# Higher = safer against accidentally merging
# two different people.
UNKNOWN_MATCH_THRESHOLD = 0.78


# ============================================================
# Active speaker settings
# ============================================================

LIP_OPEN_THRESHOLD = 0.035

SPEECH_CONFIRM_FRAMES = 2
SPEECH_RELEASE_FRAMES = 4


# ============================================================
# Speaker attribution / fusion
# ============================================================

# Rolling window used both to smooth per-track scores (for switch decisions)
# and to compute mouth_motion_score from recent lip_open_ratio history.
SPEAKER_WINDOW_MS = 800

# A challenger must beat the incumbent's smoothed score by this margin
# before the active speaker changes. Does not apply to the very first
# assignment (no incumbent yet) or to the incumbent retaining its own track.
SPEAKER_SWITCH_THRESHOLD = 0.15

# Minimum normalized lip-motion activity score required before a face can be
# identified as the active speaker. This is not an identity probability.
SPEAKER_MIN_ACTIVITY_SCORE = 0.35
# Deprecated compatibility alias; use SPEAKER_MIN_ACTIVITY_SCORE.
SPEAKER_MIN_CONFIDENCE = SPEAKER_MIN_ACTIVITY_SCORE

# How long to keep reporting the current speaker after voice activity
# drops, before falling back to active_speaker=None.
SPEAKER_GRACE_PERIOD_MS = 600

# Maximum gap allowed when matching a face observation's timestamp to the
# nearest audio sample. Beyond this, no audio data is treated as available.
AUDIO_SYNC_TOLERANCE_MS = 250

# Retained for compatibility with older consumers. Identity selection no
# longer combines these values into a confidence-like score.
VOICE_ACTIVITY_WEIGHT = 0.40
LIP_MOTION_WEIGHT = 0.30
FACE_CONFIDENCE_WEIGHT = 0.20
TEMPORAL_WEIGHT = 0.10

# Standard deviation of a track's recent lip_open_ratio history is divided
# by this constant, then clamped to [0, 1], to produce mouth_motion_score.
LIP_MOTION_NORM = 0.02

# Minimum recent mouth-motion score required before a visible face is eligible
# for active-speaker attribution; a single open-mouth observation is not enough.
SPEAKER_MIN_MOUTH_MOTION_SCORE = 0.25

# Drop per-track score/lip history after this much inactivity to prevent
# unbounded growth of the _track_history map over long sessions.
SPEAKER_TRACK_HISTORY_TTL_MS = 4000

# Max samples kept per track (lip ratios, scores) and for audio; must cover
# SPEAKER_WINDOW_MS at the highest expected frame rate.
SPEAKER_HISTORY_MAXLEN = 64


# ============================================================
# Voice embeddings / diarization
# ============================================================

# SpeechBrain speaker-embedding model (public, no HF auth required).
VOICE_EMBEDDING_MODEL = "speechbrain/spkrec-ecapa-voxceleb"

# ECAPA-TDNN embedding size for the model above.
VOICE_EMBEDDING_DIM = 192

# Minimum cosine similarity required to accept a voice match
# against an enrolled voiceprint (also used for unknown-voice matching).
# Same-speaker ECAPA cosine similarity is commonly ~0.4-0.7 (lower for
# short clips). Starting point only: validate against real recordings.
VOICE_MATCH_THRESHOLD = 0.5

# If the best and second-best voice matches are too close,
# the match is considered ambiguous and rejected.
VOICE_AMBIGUITY_MARGIN = 0.05

# Minimum enrollment quality and independent references required for automatic
# known-speaker matching. Cosine similarity is not a calibrated probability.
VOICE_PROFILE_MIN_QUALITY = 0.55
VOICE_MATCH_MIN_SUPPORTING_SAMPLES = 2

# Cosine-distance threshold used by agglomerative clustering
# when grouping a meeting's speech segments into speakers.
# Lower = more/smaller clusters (more distinct speakers found).
# Distance = 1 - similarity, so 0.55 merges windows with similarity >= ~0.45.
# Starting point only: validate against real recordings.
VOICE_CLUSTER_DISTANCE_THRESHOLD = 0.55

# Sliding-window size/step used to sub-segment each VAD speech region
# before embedding, so speaker changes without a pause can be split.
VOICE_DIARIZATION_WINDOW_SECONDS = 1.5
VOICE_DIARIZATION_STEP_SECONDS = 0.75
VOICE_MIN_IDENTITY_DURATION_SECONDS = 3.0

# Reject speaker windows that contain too little usable signal. These are
# quality gates, not identity thresholds, and should be tuned from diagnostics.
VOICE_MIN_SPEECH_RATIO = 0.55
VOICE_MIN_RMS = 0.008
VOICE_MAX_CLIPPING_RATIO = 0.02

# Use the strongest enrolled samples when building a person-level voice profile.
VOICE_PROFILE_TOP_K = 5

# A new cluster may reuse an identity already assigned earlier in the same
# meeting only when its centroid is a strong, unambiguous match.
VOICE_IN_MEETING_IDENTITY_THRESHOLD = 0.68
VOICE_IN_MEETING_IDENTITY_MARGIN = 0.08

# Diagnostic output is disabled by default and does not change recognition.
VOICE_DIAGNOSTICS_ENABLED = os.environ.get(
    "VOICE_DIAGNOSTICS_ENABLED", "0"
).strip().lower() in {"1", "true", "yes", "on"}
VOICE_DIAGNOSTICS_PATH = BASE_DIR / "logs" / "speaker_diagnostics.json"

# Realtime voice fallback runs at most once per this many frames per track.
VOICE_FALLBACK_INTERVAL_FRAMES = 30

# Whisper language policy. "auto" samples several speech regions; "fixed"
# uses WHISPER_LANGUAGE for the complete meeting.
# This meeting collection is English-dominant. Fixed language prevents a short
# opening phrase from causing Whisper to transcribe the whole meeting as Malay.
# Use mode="auto" again for genuinely multilingual meetings.
WHISPER_LANGUAGE_MODE = os.environ.get(
    "WHISPER_LANGUAGE_MODE", "fixed"
).strip().lower()
WHISPER_LANGUAGE = os.environ.get("WHISPER_LANGUAGE", "en").strip()

# Minimum confidence required before auto-detection supplies a language to
# Whisper. Otherwise Whisper is allowed to perform its own unconstrained
# detection rather than receiving a weak forced guess.
WHISPER_LANGUAGE_CONFIDENCE_THRESHOLD = 0.65
WHISPER_LANGUAGE_SAMPLE_SECONDS = 8.0
WHISPER_LANGUAGE_MAX_SAMPLES = 6

# Keep the original spoken language by default.
WHISPER_TASK = "transcribe"

# Optional Whisper decoding hint (domain vocabulary, names, etc).
WHISPER_INITIAL_PROMPT = None

# Consecutive same-speaker fragments closer than this are joined into one turn.
# Keep this short so readable transcript ranges do not span long silences.
TRANSCRIPT_MERGE_MAX_GAP_MS = 1500
TRANSCRIPT_MERGE_MAX_CHARS = 600

# Legacy thresholds retained for compatibility. Transcript text and duration
# alone are no longer used to infer an unknown speaker's identity.
TRANSCRIPT_SPEAKER_BLIP_MS = 3000
TRANSCRIPT_MINOR_SPEAKER_TOTAL_MS = 8000

# Optional local Ollama transcript formatting. No transcript is sent anywhere
# unless explicitly enabled; the service must be available at the configured URL.
TRANSCRIPT_REFINEMENT_ENABLED = os.environ.get(
    "TRANSCRIPT_REFINEMENT_ENABLED", "0"
).strip().lower() in {"1", "true", "yes", "on"}
TRANSCRIPT_REFINEMENT_URL = os.environ.get(
    "TRANSCRIPT_REFINEMENT_URL", "http://localhost:11434"
).rstrip("/")
TRANSCRIPT_REFINEMENT_MODEL = os.environ.get(
    "TRANSCRIPT_REFINEMENT_MODEL", "qwen2.5:7b"
)
TRANSCRIPT_REFINEMENT_TIMEOUT_SECONDS = 90
TRANSCRIPT_REFINEMENT_CHUNK_SEGMENTS = 30
TRANSCRIPT_REFINEMENT_CHUNK_CHARS = 9000


# ============================================================
# Video output
# ============================================================

OUTPUT_FPS_FALLBACK = 30.0


# Realtime mode favors display responsiveness over long-range detection.
REALTIME_DETECTION_INTERVAL = 3
REALTIME_HAAR_SCALE_FACTOR = 1.1
REALTIME_HAAR_MIN_NEIGHBORS = 5
REALTIME_FPS_WINDOW_SECONDS = 1.0


# ============================================================
# Optional transcription
# ============================================================

ENABLE_TRANSCRIPTION = True
WHISPER_MODEL = "small"

TRANSCRIPTION_CLEANUP_LEVEL = "basic"
TRANSCRIPTS_DIR = BASE_DIR / "transcripts"
TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)

# Minimum fraction of a speech segment covered by one face track before
# assigning that face as the speaker.
SPEAKER_FACE_OVERLAP_THRESHOLD = 0.8


# ============================================================
# Subtitle rendering
# ============================================================

SUBTITLE_MAX_GAP_MS = 800
SUBTITLE_MAX_CHARS_PER_LINE = 42
SUBTITLE_MAX_LINES = 2