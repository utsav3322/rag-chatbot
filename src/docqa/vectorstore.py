"""Chroma-backed semantic retrieval: the "RAG" half of the system.

Chunks are embedded with a sentence-transformer model (all-MiniLM-L6-v2, run via
ONNX — no PyTorch needed) and stored in a persistent Chroma collection on disk.
Queries are embedded the same way and matched by cosine similarity.

Access control is enforced **at the database level**: every query passes a
``where={"company_key": {"$in": allowed}}`` filter, so unauthorised companies'
vectors are never even considered — the strongest possible isolation guarantee,
pushed right down into the store.

:class:`ChromaRetriever` exposes the same :meth:`search` signature as the
dependency-free :class:`docqa.retrieval.Retriever`, so the engine treats the two
interchangeably and transparently falls back to TF-IDF when Chroma or its
embedding model is unavailable (e.g. fully offline).
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence

from docqa.config import CHROMA_DIR, SETTINGS
from docqa.models import Chunk, RetrievedChunk

logger = logging.getLogger(__name__)

_ADD_BATCH = 256


def _build_collection(path: str, name: str):
    """Open (or create) the persistent collection with cosine distance."""

    import chromadb
    from chromadb.utils import embedding_functions

    client = chromadb.PersistentClient(path=path)
    embed_fn = embedding_functions.DefaultEmbeddingFunction()
    return client.get_or_create_collection(
        name=name,
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"},
    )


class VectorStore:
    """Owns the persistent Chroma collection and (re)indexes chunks into it."""

    def __init__(
        self,
        path: str = str(CHROMA_DIR),
        collection_name: str = SETTINGS.collection_name,
    ) -> None:
        self._path = path
        self._name = collection_name
        self._collection = _build_collection(path, collection_name)

    # -- Indexing -----------------------------------------------------------

    @property
    def count(self) -> int:
        return int(self._collection.count())

    def is_empty(self) -> bool:
        return self.count == 0

    def reset(self) -> None:
        """Drop and recreate the collection (used for a clean re-index)."""

        import chromadb

        client = chromadb.PersistentClient(path=self._path)
        try:
            client.delete_collection(self._name)
        except Exception:  # noqa: BLE001 - collection may not exist yet
            pass
        self._collection = _build_collection(self._path, self._name)

    def add_chunks(self, chunks: Iterable[Chunk]) -> int:
        """Embed and store ``chunks``. Returns the number added."""

        batch = list(chunks)
        for start in range(0, len(batch), _ADD_BATCH):
            window = batch[start : start + _ADD_BATCH]
            self._collection.add(
                ids=[c.chunk_id for c in window],
                documents=[c.text for c in window],
                metadatas=[
                    {
                        "company_key": c.company_key,
                        "company_name": c.company_name,
                        "source_file": c.source_file,
                        "page_start": c.page_start,
                        "page_end": c.page_end,
                    }
                    for c in window
                ],
            )
        logger.info("Indexed %d chunks into Chroma collection %r", len(batch), self._name)
        return len(batch)

    def as_retriever(self) -> "ChromaRetriever":
        return ChromaRetriever(self._collection)


class ChromaRetriever:
    """Semantic retriever over a Chroma collection, access-filtered per query."""

    def __init__(self, collection) -> None:
        self._collection = collection

    @property
    def size(self) -> int:
        return int(self._collection.count())

    def search(
        self,
        query: str,
        *,
        allowed_companies: Iterable[str],
        top_k: int,
        min_score: float = 0.0,
    ) -> list[RetrievedChunk]:
        """Return the top ``top_k`` chunks for ``query`` within allowed companies.

        The company filter is applied inside Chroma's ``where`` clause, so a user
        can never be matched against a document they are not authorised to read.
        """

        allowed = sorted(set(allowed_companies))
        if not allowed or not query.strip():
            return []

        result = self._collection.query(
            query_texts=[query],
            n_results=max(1, top_k),
            where={"company_key": {"$in": allowed}},
            include=["documents", "metadatas", "distances"],
        )
        return self._to_retrieved(result, min_score)

    @staticmethod
    def _to_retrieved(result, min_score: float) -> list[RetrievedChunk]:
        ids = _first(result.get("ids"))
        docs = _first(result.get("documents"))
        metas = _first(result.get("metadatas"))
        dists = _first(result.get("distances"))

        out: list[RetrievedChunk] = []
        for chunk_id, text, meta, dist in zip(ids, docs, metas, dists):
            # Cosine distance in Chroma is 1 - cosine_similarity, so similarity
            # (our "score", higher = better) is 1 - distance.
            score = 1.0 - float(dist)
            if score < min_score:
                continue
            out.append(
                RetrievedChunk(
                    chunk=Chunk(
                        chunk_id=str(chunk_id),
                        company_key=str(meta.get("company_key", "")),
                        company_name=str(meta.get("company_name", "")),
                        source_file=str(meta.get("source_file", "")),
                        page_start=int(meta.get("page_start", 0)),
                        page_end=int(meta.get("page_end", 0)),
                        text=str(text),
                    ),
                    score=score,
                )
            )
        return out


def _first(value: Sequence | None) -> list:
    """Chroma returns per-query lists; we only ever issue one query."""

    if not value:
        return []
    return list(value[0]) if value and value[0] is not None else []
