"""A dependency-free TF-IDF retriever with cosine similarity.

Implementing TF-IDF directly (instead of pulling in scikit-learn / FAISS) keeps
the project installable on any Python with no native wheels, and makes the
ranking fully transparent. The design swaps cleanly for an embedding-based
retriever later: only :meth:`Retriever.search` would change.

Access control is enforced *inside* the retriever: callers pass the set of
company keys the user may read, and chunks outside that set are removed from the
candidate pool before any scoring. A user therefore cannot receive — or even be
ranked against — a passage they are not authorised to see.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Iterable

from docqa.models import Chunk, RetrievedChunk

_TOKEN = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?")

# A small, domain-aware stopword list. Financial numbers and tickers are kept.
_STOPWORDS: frozenset[str] = frozenset(
    """
    a an and are as at be been being by for from had has have he her his in into is it
    its of on or our that the their them they this to was were will with you your we us
    i me my
    """.split()
)


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOPWORDS]


class Retriever:
    """TF-IDF index over a fixed corpus of :class:`Chunk` objects."""

    def __init__(self, chunks: Iterable[Chunk]) -> None:
        self._chunks: list[Chunk] = list(chunks)
        self._doc_freq: Counter[str] = Counter()
        self._idf: dict[str, float] = {}
        self._vectors: list[dict[str, float]] = []
        self._norms: list[float] = []
        self._build()

    @property
    def size(self) -> int:
        return len(self._chunks)

    def _build(self) -> None:
        tokenized: list[list[str]] = [tokenize(c.text) for c in self._chunks]
        for tokens in tokenized:
            for term in set(tokens):
                self._doc_freq[term] += 1

        n_docs = max(1, len(self._chunks))
        # Smoothed idf keeps common terms from dominating and avoids div-by-zero.
        self._idf = {
            term: math.log((1 + n_docs) / (1 + df)) + 1.0
            for term, df in self._doc_freq.items()
        }

        for tokens in tokenized:
            vector = self._vectorize(tokens)
            norm = math.sqrt(sum(w * w for w in vector.values()))
            self._vectors.append(vector)
            self._norms.append(norm)

    def _vectorize(self, tokens: list[str]) -> dict[str, float]:
        counts = Counter(tokens)
        total = max(1, len(tokens))
        return {
            term: (count / total) * self._idf.get(term, 0.0)
            for term, count in counts.items()
            if self._idf.get(term, 0.0) > 0.0
        }

    def search(
        self,
        query: str,
        *,
        allowed_companies: Iterable[str],
        top_k: int,
        min_score: float = 0.0,
    ) -> list[RetrievedChunk]:
        """Return the top ``top_k`` chunks for ``query`` within allowed companies.

        Chunks whose company is not in ``allowed_companies`` are never scored.
        """

        allowed = set(allowed_companies)
        query_vec = self._vectorize(tokenize(query))
        query_norm = math.sqrt(sum(w * w for w in query_vec.values()))
        if query_norm == 0.0 or not allowed:
            return []

        scored: list[RetrievedChunk] = []
        for index, chunk in enumerate(self._chunks):
            if chunk.company_key not in allowed:
                continue
            doc_norm = self._norms[index]
            if doc_norm == 0.0:
                continue
            doc_vec = self._vectors[index]
            # Iterate the smaller vector for the dot product.
            if len(query_vec) <= len(doc_vec):
                dot = sum(w * doc_vec.get(t, 0.0) for t, w in query_vec.items())
            else:
                dot = sum(w * query_vec.get(t, 0.0) for t, w in doc_vec.items())
            if dot <= 0.0:
                continue
            score = dot / (query_norm * doc_norm)
            if score >= min_score:
                scored.append(RetrievedChunk(chunk=chunk, score=score))

        scored.sort(key=lambda rc: rc.score, reverse=True)
        return scored[:top_k]
