"""Build (or rebuild) the full data layer for the demo.

Steps:
  1. Generate the sample earnings-call PDFs if they are missing.
  2. Create and seed the SQLite catalogue (companies, users, access).
  3. Embed every PDF chunk into the persistent Chroma vector store.
  4. Record each document in the SQLite ``documents`` table.

Run:  python scripts/build_index.py  [--reset]

``--reset`` drops the existing Chroma collection and re-embeds from scratch.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from docqa.config import PDF_DIR  # noqa: E402
from docqa.db import Database  # noqa: E402
from docqa.engine import _record_documents  # noqa: E402
from docqa.ingest import load_corpus  # noqa: E402


def _ensure_pdfs() -> None:
    if not any(PDF_DIR.glob("*.pdf")):
        print("· No sample PDFs found — generating them…")
        from scripts import generate_sample_pdfs  # type: ignore

        generate_sample_pdfs.generate()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the DocQA data layer.")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Drop and re-embed the Chroma collection from scratch.",
    )
    args = parser.parse_args()

    _ensure_pdfs()

    print("· Seeding SQLite catalogue (users, companies, access)…")
    db = Database()
    db.ensure_seeded()

    print("· Loading and chunking PDFs…")
    chunks = load_corpus()
    _record_documents(db, chunks)

    print("· Embedding chunks into Chroma vector store…")
    from docqa.vectorstore import VectorStore

    store = VectorStore()
    if args.reset:
        store.reset()
    if store.is_empty():
        store.add_chunks(chunks)
    else:
        print(f"  (collection already has {store.count} vectors — skipping; "
              "use --reset to rebuild)")

    print(
        "\nDone. "
        f"users={db.user_count()} documents={db.document_count()} "
        f"vectors={store.count}"
    )


if __name__ == "__main__":
    main()
