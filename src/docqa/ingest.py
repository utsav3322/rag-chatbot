"""Load earnings-call PDFs and split them into overlapping, labelled chunks.

Each PDF filename stem must match a company key in the registry (for example
``novatech.pdf``). Pages are extracted with :mod:`pypdf`, concatenated per
document, then split into word-window chunks that remember which page range
they came from so citations can point back to the source.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from pypdf import PdfReader

from docqa.config import COMPANY_BY_KEY, PDF_DIR, SETTINGS
from docqa.models import Chunk

logger = logging.getLogger(__name__)

_WHITESPACE = re.compile(r"\s+")


def _clean(text: str) -> str:
    return _WHITESPACE.sub(" ", text).strip()


def _read_pages(path: Path) -> list[str]:
    """Return cleaned text for each page of ``path`` (never raises on a bad page)."""

    reader = PdfReader(str(path))
    pages: list[str] = []
    for index, page in enumerate(reader.pages):
        try:
            pages.append(_clean(page.extract_text() or ""))
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Failed to extract page %d of %s: %s", index, path.name, exc)
            pages.append("")
    return pages


def _chunk_pages(
    pages: list[str],
    *,
    chunk_size: int,
    overlap: int,
) -> list[tuple[int, int, str]]:
    """Split page texts into overlapping word windows.

    Returns tuples of ``(page_start, page_end, text)`` using 1-based page
    numbers. A running list of ``(word, page_number)`` lets each chunk report
    the page range it spans.
    """

    tokens: list[tuple[str, int]] = []
    for page_number, text in enumerate(pages, start=1):
        for word in text.split():
            tokens.append((word, page_number))

    if not tokens:
        return []

    step = max(1, chunk_size - overlap)
    chunks: list[tuple[int, int, str]] = []
    for start in range(0, len(tokens), step):
        window = tokens[start : start + chunk_size]
        if not window:
            break
        words = [w for w, _ in window]
        page_start = window[0][1]
        page_end = window[-1][1]
        chunks.append((page_start, page_end, " ".join(words)))
        if start + chunk_size >= len(tokens):
            break
    return chunks


def load_document(path: Path) -> list[Chunk]:
    """Load a single PDF into chunks. The filename stem selects the company."""

    key = path.stem.lower()
    company = COMPANY_BY_KEY.get(key)
    if company is None:
        raise ValueError(
            f"PDF {path.name!r} does not map to a known company. "
            f"Expected stem one of: {sorted(COMPANY_BY_KEY)}"
        )

    pages = _read_pages(path)
    raw_chunks = _chunk_pages(
        pages,
        chunk_size=SETTINGS.chunk_size_words,
        overlap=SETTINGS.chunk_overlap_words,
    )
    chunks = [
        Chunk(
            chunk_id=f"{key}-{i:03d}",
            company_key=key,
            company_name=company.name,
            source_file=path.name,
            page_start=page_start,
            page_end=page_end,
            text=text,
        )
        for i, (page_start, page_end, text) in enumerate(raw_chunks)
    ]
    logger.info("Loaded %d chunks from %s", len(chunks), path.name)
    return chunks


def load_corpus(pdf_dir: Path = PDF_DIR) -> list[Chunk]:
    """Load every ``*.pdf`` in ``pdf_dir`` whose stem matches a known company."""

    if not pdf_dir.exists():
        raise FileNotFoundError(
            f"PDF directory {pdf_dir} does not exist. "
            "Run `python scripts/generate_sample_pdfs.py` first."
        )

    chunks: list[Chunk] = []
    for path in sorted(pdf_dir.glob("*.pdf")):
        if path.stem.lower() not in COMPANY_BY_KEY:
            logger.warning("Skipping unrecognised PDF: %s", path.name)
            continue
        chunks.extend(load_document(path))

    if not chunks:
        raise ValueError(f"No usable PDFs found in {pdf_dir}.")
    logger.info("Loaded %d total chunks from %s", len(chunks), pdf_dir)
    return chunks
