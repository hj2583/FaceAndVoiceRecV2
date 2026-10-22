"""Report or import the current SQLite records into an empty PostgreSQL database."""

import argparse
from pathlib import Path

from face_voice_rec import config
from face_voice_rec.storage.database import init_db
from face_voice_rec.storage.sqlite_postgres_migration import (
    copy_sqlite_database,
    inspect_sqlite_database,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=config.DB_PATH,
        help="SQLite database to import (default: data/db/faces.db)",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="create the PostgreSQL schema and copy rows (default: report only)",
    )
    args = parser.parse_args()

    counts = inspect_sqlite_database(args.source)
    print(f"SQLite source: {args.source}")
    for table_name, count in counts.items():
        print(f"{table_name}: {count}")
    if not args.apply:
        print("Report only. Set DATABASE_URL and pass --apply to import.")
        return 0
    if not config.DATABASE_URL:
        raise RuntimeError("Set DATABASE_URL to PostgreSQL before importing")

    init_db()
    copied = copy_sqlite_database(args.source, config.DATABASE_URL)
    print(f"Imported {sum(copied.values())} rows into PostgreSQL.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())