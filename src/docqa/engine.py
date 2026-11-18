"""The query engine: the one place access control, intent routing, retrieval,
conversational context, and answer generation come together.

A single shared :class:`QAEngine` is safe for concurrent use by many users: the
corpus/index is read-only, and all per-user state (conversation history) lives
in separate :class:`Conversation` objects keyed by session. This is what makes
simultaneous multi-user access non-interfering.

Request flow for ``ask``:

    message ──▶ classify intent
                 ├─ greeting / thanks / goodbye / help ─▶ conversational reply
                 └─ query ─▶ retrieve (access-filtered) ─▶ answerer ─▶ answer
                                                          └─▶ gated citations
"""

from __future__ import annotations

import logging
import threading
from collections import deque

from collections import defaultdict

from docqa import intent as intent_mod
from docqa.access import AccessControl, AccessError, User
from docqa.config import DEMO_PASSWORD, SETTINGS, company_name
from docqa.db import Database
from docqa.ingest import load_corpus
from docqa.intent import Intent
from docqa.llm import Answerer, build_answerer
from docqa.llm import _chunk_text as llm_chunk_text
from docqa.models import Answer, Citation, Chunk, Document, Scope, Turn
from docqa.retrieval import Retriever

logger = logging.getLogger(__name__)


def _record_documents(db: Database, chunks: list[Chunk]) -> None:
    """Populate the SQLite ``documents`` catalogue from the loaded chunks."""

    groups: dict[str, list[Chunk]] = defaultdict(list)
    for chunk in chunks:
        groups[chunk.source_file].append(chunk)
    for filename, group in groups.items():
        first = group[0]
        db.record_document(
            company_key=first.company_key,
            filename=filename,
            title=f"{first.company_name} — Q4 FY2025 Earnings Call",
            num_pages=max(c.page_end for c in group),
            num_chunks=len(group),
        )


def _build_retriever(db: Database) -> tuple[object, bool]:
    """Build the retriever, preferring Chroma embeddings over TF-IDF.

    Returns ``(retriever, uses_vectors)``. Chunks are loaded from the PDFs once;
    they populate both the SQLite document catalogue and (when enabled and
    importable) the persistent Chroma collection. Any failure building the vector
    store degrades gracefully to the dependency-free TF-IDF retriever.
    """

    chunks = load_corpus()
    _record_documents(db, chunks)

    if SETTINGS.vector_store_enabled:
        try:
            from docqa.vectorstore import VectorStore

            store = VectorStore()
            if store.is_empty():
                store.add_chunks(chunks)
            logger.info("Using Chroma vector store (%d vectors).", store.count)
            return store.as_retriever(), True
        except Exception as exc:  # noqa: BLE001 - fall back if Chroma unavailable
            logger.warning(
                "Vector store unavailable (%s); falling back to TF-IDF.", exc
            )

    return Retriever(chunks), False


class Conversation:
    """Bounded, per-session history of question/answer turns.

    A deque capped at ``history_window`` keeps context relevant and memory
    bounded. Instances are never shared between users.
    """

    def __init__(self, maxlen: int = SETTINGS.history_window) -> None:
        self._turns: deque[Turn] = deque(maxlen=maxlen)

    def add(self, question: str, answer: str) -> None:
        self._turns.append(Turn(question=question, answer=answer))

    @property
    def turns(self) -> list[Turn]:
        return list(self._turns)

    def clear(self) -> None:
        self._turns.clear()


def _rewrite_follow_up(question: str, history: list[Turn]) -> str:
    """Give the keyword retriever extra context for short follow-up questions.

    A terse follow-up such as "what about margins?" has little to match on.
    Prepending recent questions biases retrieval toward the ongoing topic. The
    LLM path does its own resolution, but this harmlessly improves recall there
    too.
    """

    if not history:
        return question
    if len(question.split()) > 6:
        return question
    prior = " ".join(t.question for t in history[-2:])
    return f"{prior} {question}".strip()


class QAEngine:
    """Thread-safe facade over the whole pipeline."""

    _CONVERSATIONAL = {
        Intent.GREETING,
        Intent.THANKS,
        Intent.GOODBYE,
        Intent.HELP,
    }

    def __init__(
        self,
        retriever: object,
        access: AccessControl,
        answerer: Answerer,
        db: Database | None = None,
        uses_vectors: bool = False,
    ) -> None:
        self._retriever = retriever
        self._access = access
        self._answerer = answerer
        self._db = db
        self._uses_vectors = uses_vectors
        self._lock = threading.Lock()
        self._conversations: dict[str, Conversation] = {}

    # -- Construction -------------------------------------------------------

    @classmethod
    def bootstrap(cls) -> "QAEngine":
        """Build an engine from the SQLite catalogue + PDFs (used by the web app).

        On first run this creates and seeds the database, embeds the PDF chunks
        into the Chroma vector store, and records each document in the catalogue;
        subsequent runs reuse the persisted store.
        """

        db = Database()
        db.ensure_seeded()
        access = AccessControl.from_db(db)
        retriever, uses_vectors = _build_retriever(db)
        answerer = build_answerer()
        logger.info(
            "Engine ready: %d chunks, %d users, %d documents, vectors=%s, llm=%s",
            retriever.size,
            len(access.users),
            db.document_count(),
            uses_vectors,
            answerer.uses_llm,
        )
        return cls(
            retriever=retriever,
            access=access,
            answerer=answerer,
            db=db,
            uses_vectors=uses_vectors,
        )

    @property
    def access(self) -> AccessControl:
        return self._access

    @property
    def db(self) -> Database | None:
        return self._db

    @property
    def uses_llm(self) -> bool:
        return self._answerer.uses_llm

    @property
    def uses_vectors(self) -> bool:
        return self._uses_vectors

    # -- Authentication -----------------------------------------------------

    def verify_login(self, email: str, password: str) -> User | None:
        """Validate ``email`` + ``password`` and return the user, or ``None``.

        Credentials are checked against the SQLite ``users`` table when a
        database is attached; a database-less engine (used in some unit tests)
        accepts a known email with the shared demo password.
        """

        if self._db is not None:
            if not self._db.verify_credentials(email, password):
                return None
            return self._access.authenticate(email)
        try:
            user = self._access.authenticate(email)
        except AccessError:
            return None
        return user if password == DEMO_PASSWORD else None

    def documents_for(self, email: str) -> list[Document]:
        """List the documents the user is authorised to see, from the catalogue."""

        user = self._access.authenticate(email)
        if self._db is None:
            return []
        return self._db.list_documents(sorted(user.companies))

    # -- Per-session conversation state ------------------------------------

    def _conversation(self, session_id: str) -> Conversation:
        with self._lock:
            convo = self._conversations.get(session_id)
            if convo is None:
                convo = Conversation()
                self._conversations[session_id] = convo
            return convo

    def reset(self, session_id: str) -> None:
        with self._lock:
            self._conversations.pop(session_id, None)

    # -- The main entry point ----------------------------------------------

    def ask(self, *, email: str, session_id: str, question: str) -> Answer:
        """Answer ``question`` for ``email`` within their isolated session.

        Raises :class:`docqa.access.AccessError` if the user is unknown.
        """

        user = self._access.authenticate(email)
        company_names = tuple(sorted(company_name(k) for k in user.companies))
        scope = Scope(
            companies=company_names,
            examples=intent_mod.example_questions(company_names),
        )
        convo = self._conversation(session_id)

        intent = intent_mod.classify(question)

        # Offline only (no LLM): greetings/small talk get a canned reply; with an
        # LLM, small talk is answered by the model for a natural, varied response.
        if intent in self._CONVERSATIONAL and not self._answerer.uses_llm:
            text = self._conversational_reply(intent, company_names)
            convo.add(question, text)
            return Answer(
                text=text,
                citations=(),
                used_llm=False,
                accessible_companies=company_names,
            )

        # --- Answer path (RAG) --------------------------------------------
        if intent in self._CONVERSATIONAL:
            retrieved: list = []
        else:
            search_query = _rewrite_follow_up(question, convo.turns)
            retrieved = self._retriever.search(
                search_query,
                allowed_companies=user.companies,
                top_k=SETTINGS.top_k,
                min_score=SETTINGS.min_score,
            )

        generation = self._answerer.answer(question, retrieved, convo.turns, scope)
        convo.add(question, generation.text)

        citations = self._build_citations(retrieved, generation.source_indices)
        return Answer(
            text=generation.text,
            citations=citations,
            used_llm=self._answerer.uses_llm,
            accessible_companies=company_names,
        )

    def _build_citations(
        self,
        retrieved: list,
        source_indices: tuple[int, ...] | None,
    ) -> tuple[Citation, ...]:
        """Cite only the passages the answer actually used.

        When the answerer reports which contexts it used (``source_indices``),
        we cite exactly those — so an out-of-scope decline or a follow-up shows
        no noisy, unused sources. When it doesn't report (``None``), we fall back
        to citing passages whose relevance clears ``citation_min_score``.
        """

        if source_indices is None:
            selected = [
                rc for rc in retrieved if rc.score >= SETTINGS.citation_min_score
            ]
        else:
            selected = [
                retrieved[i - 1]
                for i in source_indices
                if 1 <= i <= len(retrieved)
            ]
        return tuple(
            Citation(
                company_name=rc.chunk.company_name,
                source_file=rc.chunk.source_file,
                page_start=rc.chunk.page_start,
                page_end=rc.chunk.page_end,
                excerpt=_shorten(rc.chunk.text, SETTINGS.excerpt_chars),
                score=round(rc.score, 4),
            )
            for rc in selected
        )

    def ask_stream(self, *, email: str, session_id: str, question: str):
        """Stream the answer for ``question`` as it is produced.

        Yields ``("token", text)`` events as the answer is generated, then a
        single terminal ``("final", Answer)`` event whose ``text`` is empty (it
        has already been streamed) but which carries the citations and metadata.

        Raises :class:`docqa.access.AccessError` if the user is unknown.
        """

        user = self._access.authenticate(email)
        company_names = tuple(sorted(company_name(k) for k in user.companies))
        scope = Scope(
            companies=company_names,
            examples=intent_mod.example_questions(company_names),
        )
        convo = self._conversation(session_id)

        intent = intent_mod.classify(question)

        # Offline only (no LLM): greetings/small talk get a canned, scope-aware
        # reply. When an LLM is available we let *it* handle small talk so the
        # response is natural and varied rather than a fixed template.
        if intent in self._CONVERSATIONAL and not self._answerer.uses_llm:
            text = self._conversational_reply(intent, company_names)
            for piece in llm_chunk_text(text):
                yield ("token", piece)
            convo.add(question, text)
            yield (
                "final",
                Answer(
                    text="",
                    citations=(),
                    used_llm=False,
                    accessible_companies=company_names,
                ),
            )
            return

        # --- Answer path (RAG, streamed) ----------------------------------
        # Small talk needs no retrieval; real questions retrieve first.
        if intent in self._CONVERSATIONAL:
            retrieved: list = []
        else:
            search_query = _rewrite_follow_up(question, convo.turns)
            retrieved = self._retriever.search(
                search_query,
                allowed_companies=user.companies,
                top_k=SETTINGS.top_k,
                min_score=SETTINGS.min_score,
            )

        collected: list[str] = []
        source_indices: tuple[int, ...] | None = None
        for kind, payload in self._answerer.stream(
            question, retrieved, convo.turns, scope
        ):
            if kind == "delta":
                assert isinstance(payload, str)
                collected.append(payload)
                yield ("token", payload)
            elif kind == "sources":
                source_indices = payload  # type: ignore[assignment]

        answer_text = "".join(collected).strip()
        convo.add(question, answer_text)
        citations = self._build_citations(retrieved, source_indices)
        yield (
            "final",
            Answer(
                text="",
                citations=citations,
                used_llm=self._answerer.uses_llm,
                accessible_companies=company_names,
            ),
        )

    # -- Helpers ------------------------------------------------------------

    def _conversational_reply(self, intent: Intent, companies: tuple[str, ...]) -> str:
        if intent is Intent.GREETING:
            return intent_mod.greeting_text(companies)
        if intent is Intent.HELP:
            return intent_mod.help_text(companies)
        if intent is Intent.THANKS:
            return intent_mod.thanks_text(companies)
        return intent_mod.goodbye_text()


def _shorten(text: str, width: int) -> str:
    text = text.strip()
    if len(text) <= width:
        return text
    return text[: width - 1].rsplit(" ", 1)[0] + " …"
