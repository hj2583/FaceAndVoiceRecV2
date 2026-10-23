import os
import sys
import time
from urllib.request import urlopen

from sqlalchemy.engine import make_url


def rewrite_database_url(database_url: str, database_host: str) -> str:
    return make_url(database_url).set(host=database_host).render_as_string(
        hide_password=False
    )


def main() -> None:
    os.environ["DATABASE_URL"] = rewrite_database_url(
        os.environ["DATABASE_URL"],
        os.environ["DATABASE_HOST"],
    )
    qdrant_url = os.environ.get("QDRANT_URL", "http://localhost:6333").rstrip("/")
    deadline = time.monotonic() + 180
    while time.monotonic() < deadline:
        try:
            with urlopen(f"{qdrant_url}/readyz", timeout=2) as response:
                if response.status == 200:
                    break
        except OSError:
            time.sleep(min(2, max(0, deadline - time.monotonic())))
    else:
        raise SystemExit("Qdrant did not become ready within 180 seconds")

    os.execv(
        sys.executable,
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            "app/main.py",
            "--server.address=0.0.0.0",
            "--server.port=8501",
            "--server.headless=true",
        ],
    )


if __name__ == "__main__":
    main()
