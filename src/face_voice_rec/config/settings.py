import os
from pathlib import Path

from dotenv import load_dotenv


def _load_environment(project_root: Path | None = None) -> None:
    root = project_root if project_root is not None else Path(__file__).resolve().parents[3]
    load_dotenv(root / ".env", override=True)


_load_environment()

DATABASE_URL = os.environ.get("DATABASE_URL", "").strip()
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333").strip()
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY", "").strip() or None
QDRANT_TIMEOUT_SECONDS = float(os.environ.get("QDRANT_TIMEOUT_SECONDS", "10"))
QDRANT_FACE_COLLECTION = os.environ.get(
    "QDRANT_FACE_COLLECTION", "face_embeddings"
).strip()
QDRANT_VOICE_COLLECTION = os.environ.get(
    "QDRANT_VOICE_COLLECTION", "voice_embeddings"
).strip()

VOICE_DIAGNOSTICS_ENABLED = os.environ.get(
    "VOICE_DIAGNOSTICS_ENABLED", "0"
).strip().lower() in {"1", "true", "yes", "on"}

WHISPER_LANGUAGE_MODE = os.environ.get(
    "WHISPER_LANGUAGE_MODE", "fixed"
).strip().lower()
WHISPER_LANGUAGE = os.environ.get("WHISPER_LANGUAGE", "en").strip()

TRANSCRIPT_REFINEMENT_ENABLED = os.environ.get(
    "TRANSCRIPT_REFINEMENT_ENABLED", "0"
).strip().lower() in {"1", "true", "yes", "on"}
TRANSCRIPT_REFINEMENT_URL = os.environ.get(
    "TRANSCRIPT_REFINEMENT_URL", "http://localhost:11434"
).rstrip("/")
TRANSCRIPT_REFINEMENT_MODEL = os.environ.get(
    "TRANSCRIPT_REFINEMENT_MODEL", "qwen2.5:7b"
)
_REFINEMENT_APIS = ("ollama", "openai")


def _refinement_api(value: str | None) -> str:
    api = (value or "").strip().lower() or "ollama"
    if api not in _REFINEMENT_APIS:
        raise ValueError(
            "TRANSCRIPT_REFINEMENT_API must be one of: " + ", ".join(_REFINEMENT_APIS)
        )
    return api


TRANSCRIPT_REFINEMENT_API = _refinement_api(os.environ.get("TRANSCRIPT_REFINEMENT_API"))
TRANSCRIPT_REFINEMENT_USE_CLOUD = os.environ.get(
    "TRANSCRIPT_REFINEMENT_USE_CLOUD", "0"
).strip().lower() in {"1", "true", "yes", "on"}
ANTHROPIC_API_URL = os.environ.get("ANTHROPIC_API_URL", "").strip()
API_KEY = os.environ.get("API_KEY", "").strip()
ANTHROPIC_VERSION = os.environ.get("ANTHROPIC_VERSION", "").strip()
MODEL = os.environ.get("MODEL", "").strip()
TRANSCRIPT_REFINEMENT_TIMEOUT_SECONDS = int(
    os.environ.get("TRANSCRIPT_REFINEMENT_TIMEOUT_SECONDS", "90")
)
TRANSCRIPT_REFINEMENT_CHUNK_SEGMENTS = 30
TRANSCRIPT_REFINEMENT_CHUNK_CHARS = 9000


__all__ = [
    "DATABASE_URL",
    "QDRANT_URL",
    "QDRANT_API_KEY",
    "QDRANT_TIMEOUT_SECONDS",
    "QDRANT_FACE_COLLECTION",
    "QDRANT_VOICE_COLLECTION",
    "VOICE_DIAGNOSTICS_ENABLED",
    "WHISPER_LANGUAGE_MODE",
    "WHISPER_LANGUAGE",
    "TRANSCRIPT_REFINEMENT_ENABLED",
    "TRANSCRIPT_REFINEMENT_URL",
    "TRANSCRIPT_REFINEMENT_MODEL",
    "TRANSCRIPT_REFINEMENT_API",
    "TRANSCRIPT_REFINEMENT_USE_CLOUD",
    "ANTHROPIC_API_URL",
    "API_KEY",
    "ANTHROPIC_VERSION",
    "MODEL",
    "TRANSCRIPT_REFINEMENT_TIMEOUT_SECONDS",
    "TRANSCRIPT_REFINEMENT_CHUNK_SEGMENTS",
    "TRANSCRIPT_REFINEMENT_CHUNK_CHARS",
]