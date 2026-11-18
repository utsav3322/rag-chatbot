"""Tests for PDF ingestion and chunking against the generated sample corpus."""

from __future__ import annotations

import pytest

from docqa.config import PDF_DIR
from docqa.ingest import load_corpus, load_document

pytestmark = pytest.mark.integration

_HAS_PDFS = any(PDF_DIR.glob("*.pdf"))
_skip = pytest.mark.skipif(
    not _HAS_PDFS,
    reason="sample PDFs not generated; run scripts/generate_sample_pdfs.py",
)


@_skip
def test_load_corpus_produces_chunks() -> None:
    chunks = load_corpus()
    assert len(chunks) >= 5
    companies = {c.company_key for c in chunks}
    assert {"novatech", "helios", "verdant", "orion", "lumina"} <= companies


@_skip
def test_chunks_carry_company_and_pages() -> None:
    chunk = load_document(PDF_DIR / "novatech.pdf")[0]
    assert chunk.company_key == "novatech"
    assert chunk.company_name == "NovaTech Industries"
    assert chunk.page_start >= 1
    assert chunk.text


@_skip
def test_revenue_figure_survives_extraction() -> None:
    text = " ".join(c.text for c in load_document(PDF_DIR / "novatech.pdf"))
    assert "2.34" in text
