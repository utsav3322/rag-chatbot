"""Convenience launcher: `python run.py` starts the web UI.

Adds ``src/`` to the path so the project runs without an editable install,
generates the sample PDFs on first run if they are missing, then serves the app.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from docqa.config import PDF_DIR  # noqa: E402


def _ensure_pdfs() -> None:
    if not any(PDF_DIR.glob("*.pdf")):
        print("No sample PDFs found — generating them now...")
        from scripts import generate_sample_pdfs  # type: ignore

        generate_sample_pdfs.generate()


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    _ensure_pdfs()
    from docqa.web.app import main

    main()
