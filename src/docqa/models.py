"""Immutable domain models shared across the application."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Company:
    """A company whose earnings-call document is in the corpus."""

    key: str
    name: str
    ticker: str


@dataclass(frozen=True)
class Chunk:
    """A retrievable passage extracted from a source document.

    Chunks carry the owning company so that access control can filter the
    candidate set *before* any similarity scoring happens.
    """

    chunk_id: str
    company_key: str
    company_name: str
    source_file: str
    page_start: int
    page_end: int
    text: str


@dataclass(frozen=True)
class RetrievedChunk:
    """A chunk paired with the relevance score from a single query."""

    chunk: Chunk
    score: float


@dataclass(frozen=True)
class Document:
    """Metadata for one source PDF, as recorded in the SQLite catalogue."""

    company_key: str
    company_name: str
    filename: str
    title: str
    num_pages: int
    num_chunks: int


@dataclass(frozen=True)
class Citation:
    """A human-readable reference to the passage that supported an answer."""

    company_name: str
    source_file: str
    page_start: int
    page_end: int
    excerpt: str
    score: float

    @property
    def label(self) -> str:
        if self.page_start == self.page_end:
            pages = f"p.{self.page_start}"
        else:
            pages = f"pp.{self.page_start}-{self.page_end}"
        return f"{self.company_name} — {self.source_file} ({pages})"


@dataclass(frozen=True)
class Turn:
    """One question/answer exchange within a conversation."""

    question: str
    answer: str


@dataclass(frozen=True)
class Scope:
    """What the current user is allowed to ask about, passed to the answerer.

    Lets answers decline gracefully and point the user at what they *can* do.
    """

    companies: tuple[str, ...]
    examples: tuple[str, ...]


@dataclass(frozen=True)
class Generation:
    """What an answerer returns: the display text plus which contexts it used.

    ``source_indices`` are 1-based positions into the retrieved-context list that
    the answer actually relied on. ``None`` means "unknown" (the engine then
    falls back to score-based gating); an empty tuple means "used no sources"
    (e.g. a graceful out-of-scope reply), so no citations are shown.
    """

    text: str
    source_indices: tuple[int, ...] | None = None


@dataclass(frozen=True)
class Answer:
    """The full result returned for a single user query."""

    text: str
    citations: tuple[Citation, ...] = field(default_factory=tuple)
    used_llm: bool = False
    accessible_companies: tuple[str, ...] = field(default_factory=tuple)
