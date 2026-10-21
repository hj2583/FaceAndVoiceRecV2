"""Report or backfill known and unresolved face/voice vectors into Qdrant."""

import argparse

from face_voice_rec import config
from face_voice_rec.storage.qdrant_backfill import (
    apply_known_embedding_backfill,
    prepare_known_embedding_backfill,
)
from face_voice_rec.storage.qdrant_store import create_qdrant_vector_store


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="write validated enrolled vectors to Qdrant (default: report only)",
    )
    args = parser.parse_args()

    prepared = prepare_known_embedding_backfill()
    print(
        f"Known face: {len(prepared.face_points)} valid, {prepared.face_skipped} skipped; "
        f"known voice: {len(prepared.voice_points)} valid, {prepared.voice_skipped} skipped."
    )
    print(
        f"Unknown face samples: {len(prepared.unknown_face_points)} valid, "
        f"{len(prepared.unknown_face_skips)} skipped; unknown voice samples: "
        f"{len(prepared.unknown_voice_points)} valid, "
        f"{len(prepared.unknown_voice_skips)} skipped."
    )
    for modality, skips in (
        ("face", prepared.face_skips),
        ("voice", prepared.voice_skips),
        ("unknown face sample", prepared.unknown_face_skips),
        ("unknown voice sample", prepared.unknown_voice_skips),
    ):
        for embedding_id, reason in skips:
            print(f"Skipped {modality} embedding {embedding_id}: {reason}.")
    if not args.apply:
        print("Report only. Pass --apply to write these vectors to Qdrant.")
        return 0
    if prepared.skipped:
        print("No vectors were written. Resolve skipped source files and retry.")
        return 2

    store = create_qdrant_vector_store()
    apply_known_embedding_backfill(store, prepared)
    print(
        "Backfilled "
        f"{len(prepared.face_points) + len(prepared.unknown_face_points)} face and "
        f"{len(prepared.voice_points) + len(prepared.unknown_voice_points)} voice "
        "vectors into Qdrant."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())