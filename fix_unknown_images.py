from pathlib import Path
import sqlite3

from config import DB_PATH, UNKNOWN_FACES_DIR


def find_best_image(label):
    """
    Find the best available image for an unknown track.

    Priority:
    1. track_X.jpg
    2. track_X_sample_1.jpg
    3. track_X_sample_2.jpg
    4. ...
    """

    candidates = []

    # Extract track number from labels such as track_1
    if not label.startswith("track_"):
        return None

    track_name = label.strip()

    # Original image
    original = UNKNOWN_FACES_DIR / f"{track_name}.jpg"

    if original.exists() and original.stat().st_size > 0:
        candidates.append(original)

    # Sample images
    sample_files = sorted(
        UNKNOWN_FACES_DIR.glob(
            f"{track_name}_sample_*.jpg"
        ),
        key=lambda p: p.name,
    )

    for path in sample_files:
        if path.exists() and path.stat().st_size > 0:
            candidates.append(path)

    if not candidates:
        return None

    # Prefer original image
    return candidates[0]


def main():
    print("Scanning unknown face records...")
    print(f"Unknown faces directory: {UNKNOWN_FACES_DIR}")
    print()

    if not UNKNOWN_FACES_DIR.exists():
        print("Unknown faces directory does not exist.")
        return

    conn = sqlite3.connect(DB_PATH)

    try:
        rows = conn.execute(
            """
            SELECT
                unknown_id,
                label,
                image_path,
                embedding_path
            FROM unknown_tracks
            ORDER BY unknown_id
            """
        ).fetchall()

        repaired = 0
        skipped = 0

        for unknown_id, label, image_path, embedding_path in rows:

            # Check whether current image path is already valid
            current_valid = False

            if image_path:
                current_path = Path(str(image_path))

                if current_path.exists():
                    current_valid = True

                # Also try relative path from project directory
                if not current_valid:
                    relative_path = Path.cwd() / current_path

                    if relative_path.exists():
                        current_valid = True

            if current_valid:
                skipped += 1
                continue

            # Find replacement image
            replacement = find_best_image(label)

            if replacement is None:
                print(
                    f"[NO IMAGE] ID={unknown_id} "
                    f"label={label}"
                )
                continue

            replacement = replacement.resolve()

            conn.execute(
                """
                UPDATE unknown_tracks
                SET image_path=?
                WHERE unknown_id=?
                """,
                (
                    str(replacement),
                    unknown_id,
                ),
            )

            repaired += 1

            print(
                f"[REPAIRED] ID={unknown_id} "
                f"{label}"
            )
            print(
                f"           -> {replacement}"
            )

        conn.commit()

        print()
        print("=" * 60)
        print("Repair complete.")
        print(f"Records repaired: {repaired}")
        print(f"Records already valid: {skipped}")
        print("=" * 60)

    finally:
        conn.close()


if __name__ == "__main__":
    main()