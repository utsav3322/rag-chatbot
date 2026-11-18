"""Shared pytest fixtures: a small in-memory corpus, access policy, and engine.

Tests never touch the LLM, the real PDFs, or the Chroma vector store — they build
chunks directly and use the dependency-free TF-IDF retriever so they run fast,
offline, and deterministically. The SQLite catalogue, however, is exercised for
real against a temporary database file (users, access, documents, passwords).
"""

from __future__ import annotations

import pytest

from docqa.access import AccessControl
from docqa.db import Database
from docqa.engine import QAEngine, _record_documents
from docqa.llm import ExtractiveAnswerer
from docqa.models import Chunk
from docqa.retrieval import Retriever


def _chunk(cid: str, company_key: str, name: str, text: str) -> Chunk:
    return Chunk(
        chunk_id=cid,
        company_key=company_key,
        company_name=name,
        source_file=f"{company_key}.pdf",
        page_start=1,
        page_end=1,
        text=text,
    )


@pytest.fixture
def chunks() -> list[Chunk]:
    return [
        _chunk(
            "novatech-000",
            "novatech",
            "NovaTech Industries",
            "NovaTech fourth quarter revenue was 2.34 billion dollars up 28 percent "
            "driven by cloud platform growth and strong net revenue retention.",
        ),
        _chunk(
            "novatech-001",
            "novatech",
            "NovaTech Industries",
            "NovaTech guidance for fiscal 2026 expects revenue of 10.4 to 10.6 billion "
            "and operating margin of about 32 percent.",
        ),
        _chunk(
            "helios-000",
            "helios",
            "Helios Energy Corp",
            "Helios revenue was 3.78 billion dollars with solar generation leading "
            "growth and a contracted backlog of 9.2 gigawatts.",
        ),
        _chunk(
            "lumina-000",
            "lumina",
            "Lumina Pharmaceuticals",
            "Lumina revenue was 4.11 billion dollars led by the immunology franchise "
            "and the Velora therapy with strong oncology contribution.",
        ),
    ]


@pytest.fixture
def access() -> AccessControl:
    """A JSON/mapping-backed policy for pure access-control unit tests."""

    return AccessControl.from_mapping(
        {
            "users": [
                {"email": "alice@email.com", "display_name": "Alice", "companies": ["novatech"]},
                {
                    "email": "bob@email.com",
                    "display_name": "Bob",
                    "companies": ["helios", "lumina"],
                },
            ]
        }
    )


@pytest.fixture
def db(tmp_path, chunks) -> Database:
    """A real SQLite catalogue in a temp file, seeded with users + documents."""

    database = Database(tmp_path / "test.db")
    database.init_schema()
    database.seed_companies()
    database.upsert_user("alice@email.com", "Alice", "123456", ["novatech"])
    database.upsert_user("bob@email.com", "Bob", "123456", ["helios", "lumina"])
    _record_documents(database, chunks)
    return database


@pytest.fixture
def retriever(chunks) -> Retriever:
    return Retriever(chunks)


@pytest.fixture
def engine(retriever, db) -> QAEngine:
    access = AccessControl.from_db(db)
    return QAEngine(
        retriever=retriever,
        access=access,
        answerer=ExtractiveAnswerer(),
        db=db,
    )
