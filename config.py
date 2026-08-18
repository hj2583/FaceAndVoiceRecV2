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

# Minimum face size allowed into the tracking pipeline
# a face this small.
MIN_FACE_SIZE = 25

# Minimum face size before attempting ArcFace recognition.
MIN_RECOGNITION_FACE_SIZE = 30

# Number of faces MediaPipe should detect.
MAX_FACES = 8

# Tracker settings.
TRACK_MAX_MISSED_FRAMES = 20
TRACK_DISTANCE_PX = 180

# Recognition debugging
DEBUG_FACE_SIZE = True
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


# ============================================================
# Optional transcription
# ============================================================

ENABLE_TRANSCRIPTION = False
WHISPER_MODEL = "small"