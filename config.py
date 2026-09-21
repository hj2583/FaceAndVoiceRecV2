from pathlib import Path


# ============================================================
# Base paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DB_PATH = BASE_DIR / "faces.db"
INDEX_PATH = BASE_DIR / "face_index.npz"

KNOWN_FACES_DIR = BASE_DIR / "known_faces"
UNKNOWN_FACES_DIR = BASE_DIR / "unknown_faces"
INITIAL_VIDEO_DIR = BASE_DIR / "initialVideo"
TRACKED_VIDEO_DIR = BASE_DIR / "trackedVideo"
LOG_DIR = BASE_DIR / "logs"


# ============================================================
# Create required directories
# ============================================================

for directory in (
    KNOWN_FACES_DIR,
    UNKNOWN_FACES_DIR,
    INITIAL_VIDEO_DIR,
    TRACKED_VIDEO_DIR,
    LOG_DIR,
):
    directory.mkdir(
        parents=True,
        exist_ok=True,
    )


# ============================================================
# Face backend selection
# ============================================================

# Choose the preferred backend for detection and embedding extraction.
# Supported values: "deepface", "uniface"
# For a UniFace test run, prefer UniFace and keep DeepFace as the fallback.
FACE_BACKEND = "uniface"
FACE_BACKEND_FALLBACK = "deepface"
FACE_BACKEND_STRICT_COMPATIBILITY = True


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