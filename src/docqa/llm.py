"""Answer generation from retrieved context.

Two interchangeable implementations satisfy the :class:`Answerer` protocol:

* :class:`ClaudeAnswerer` — grounded, conversational answers via the Anthropic
  API (used automatically when ``ANTHROPIC_API_KEY`` is set). Claude decides
  whether the context actually answers the question, and declines gracefully —
  pointing the user at what they *can* ask — when it does not.
* :class:`ExtractiveAnswerer` — a zero-dependency fallback that uses a relevance
  threshold to decide between an excerpt answer and a graceful out-of-scope
  message, so the system is fully demonstrable offline.

Both receive only the chunks the user is authorised to see, so neither can leak
unauthorised content.
"""

from __future__ import annotations

import logging
import re
import textwrap
from collections.abc import Iterator
from typing import Protocol, runtime_checkable

from docqa import intent
from docqa.config import SETTINGS, anthropic_api_key
from docqa.models import Generation, RetrievedChunk, Scope, Turn

logger = logging.getLogger(__name__)

_SOURCES_LINE = re.compile(r"(?im)^\s*SOURCES?\s*:\s*(.*)\s*$")

# A streaming event is either an incremental piece of display text or, once, the
# final set of source indices the answer used:
#   ("delta", "some text")        -> append to the answer
#   ("sources", (1, 3) | () | None) -> which contexts were used (last event)
StreamEvent = tuple[str, object]

# Hold back enough trailing characters while streaming to recognise a
# "\nSOURCES:" marker before any of it reaches the user.
_SOURCES_HOLDBACK = 16


def _chunk_text(text: str, group: int = 3) -> Iterator[str]:
    """Yield ``text`` in small word-groups (keeps whitespace/newlines attached).

    Used to give the extractive fallback and canned replies the same smooth
    streaming feel as the LLM path.
    """

    tokens = re.findall(r"\S+\s*|\s+", text)
    for i in range(0, len(tokens), group):
        piece = "".join(tokens[i : i + group])
        if piece:
            yield piece


def _parse_sources(text: str, n_contexts: int) -> tuple[str, tuple[int, ...] | None]:
    """Split a trailing ``SOURCES: 1, 3`` marker off the model's answer.

    Returns the cleaned display text and the 1-based indices the model said it
    used (``()`` for "none", ``None`` if no marker was found).
    """

    match = _SOURCES_LINE.search(text)
    if not match:
        return text.strip(), None

    raw = match.group(1).strip().lower()
    cleaned = _SOURCES_LINE.sub("", text).strip()
    if raw in {"none", "[]", "-", "n/a", ""}:
        return cleaned, ()
    indices = sorted(
        {
            int(tok)
            for tok in re.findall(r"\d+", raw)
            if 1 <= int(tok) <= n_contexts
        }
    )
    return cleaned, tuple(indices)


@runtime_checkable
class Answerer(Protocol):
    """Strategy for turning a query + context + history into a :class:`Generation`."""

    @property
    def uses_llm(self) -> bool: ...

    def answer(
        self,
        question: str,
        contexts: list[RetrievedChunk],
        history: list[Turn],
        scope: Scope,
    ) -> Generation: ...

    def stream(
        self,
        question: str,
        contexts: list[RetrievedChunk],
        history: list[Turn],
        scope: Scope,
    ) -> Iterator[StreamEvent]: ...


def _format_contexts(contexts: list[RetrievedChunk]) -> str:
    if not contexts:
        return "(no relevant passages were found in the user's accessible documents)"
    blocks = []
    for i, rc in enumerate(contexts, start=1):
        c = rc.chunk
        blocks.append(
            f"[{i}] {c.company_name} — {c.source_file} "
            f"(pages {c.page_start}-{c.page_end}, relevance {rc.score:.3f})\n{c.text}"
        )
    return "\n\n".join(blocks)


class ExtractiveAnswerer:
    """Build an answer from excerpts without any external model."""

    uses_llm = False

    def __init__(
        self,
        excerpt_chars: int = SETTINGS.excerpt_chars,
        answer_min_score: float = SETTINGS.answer_min_score,
    ) -> None:
        self._excerpt_chars = excerpt_chars
        self._answer_min_score = answer_min_score

    def answer(
        self,
        question: str,
        contexts: list[RetrievedChunk],
        history: list[Turn],
        scope: Scope,
    ) -> Generation:
        used = [
            i for i, rc in enumerate(contexts, start=1)
            if rc.score >= self._answer_min_score
        ]
        if not used:
            return Generation(
                text=intent.out_of_scope_text(scope.companies), source_indices=()
            )

        lines = [
            "Here are the most relevant excerpts from your authorised documents:",
            "",
        ]
        for n, i in enumerate(used, start=1):
            rc = contexts[i - 1]
            excerpt = textwrap.shorten(
                rc.chunk.text, width=self._excerpt_chars, placeholder=" …"
            )
            lines.append(f"{n}. ({rc.chunk.company_name}) {excerpt}")
        lines.append("")
        lines.append(
            "Note: running without an ANTHROPIC_API_KEY, so this is a direct "
            "excerpt view rather than a synthesised answer. See the citations below."
        )
        return Generation(text="\n".join(lines), source_indices=tuple(used))

    def stream(
        self,
        question: str,
        contexts: list[RetrievedChunk],
        history: list[Turn],
        scope: Scope,
    ) -> Iterator[StreamEvent]:
        generation = self.answer(question, contexts, history, scope)
        for piece in _chunk_text(generation.text):
            yield ("delta", piece)
        yield ("sources", generation.source_indices)


class ClaudeAnswerer:
    """Grounded, conversational answers via the Anthropic Messages API."""

    uses_llm = True

    def __init__(self, api_key: str, model: str = SETTINGS.anthropic_model) -> None:
        import anthropic  # imported lazily so the fallback path needs no dep

        self._client = anthropic.Anthropic(api_key=api_key)
        self._model = model

    def _system_prompt(self, scope: Scope) -> str:
        companies = intent._companies_phrase(scope.companies)
        examples = "; ".join(scope.examples[:3]) if scope.examples else ""
        return (
            "You are DocQA, a sharp, friendly assistant that answers questions "
            "about company earnings-call documents.\n\n"
            f"The user is authorised to access documents for: {companies}.\n\n"
            "Follow these rules:\n"
            "1. Answer questions using ONLY the provided context excerpts. Be "
            "concise, specific, and quote exact figures where relevant.\n"
            "2. If the user is only greeting you, thanking you, saying goodbye, or "
            "making light small talk (e.g. 'hi', 'hello', 'how are you', "
            "'thanks', 'bye'), respond warmly and naturally in a sentence or two — "
            "greet them back, and briefly invite them to ask about the earnings "
            f"documents for {companies}. Do NOT treat small talk as out of scope, "
            "and do NOT dump financial figures they did not ask for. Vary your "
            "wording naturally; don't sound scripted.\n"
            "3. Use the conversation history to resolve follow-up questions "
            '(for example "that quarter", "its margin", "what about guidance?").\n'
            "4. If a genuine question's context does NOT contain the answer, do "
            "not guess. Say you don't have that information in the user's "
            "accessible documents, and suggest what they can ask about instead.\n"
            "5. If the question is unrelated to these earnings documents (general "
            "knowledge, other companies, coding, weather, personal advice, etc.), "
            "politely explain that it's outside your scope — which is limited to "
            f"the earnings documents for {companies} — and offer a couple of "
            "example questions.\n"
            "6. Never reveal or infer information about companies the user cannot "
            "access, even if asked directly.\n"
            "7. Be warm and natural, like a knowledgeable equity analyst. When you "
            "use a figure, mention which company it came from.\n"
            "8. Finish EVERY reply with a final line of the form "
            "'SOURCES: <comma-separated context numbers you actually used>' — for "
            "example 'SOURCES: 1, 3'. If you did not use any context (small talk, "
            "a decline, or an out-of-scope reply), write 'SOURCES: none'. This line "
            "is parsed by the app and shown to no one, so it must always be present "
            "and last.\n"
            + (f"\nGood example questions for this user: {examples}." if examples else "")
        )

    @staticmethod
    def _build_messages(
        question: str,
        contexts: list[RetrievedChunk],
        history: list[Turn],
    ) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = []
        for turn in history:
            messages.append({"role": "user", "content": turn.question})
            messages.append({"role": "assistant", "content": turn.answer})
        prompt = (
            f"Context excerpts from the user's authorised documents:\n\n"
            f"{_format_contexts(contexts)}\n\n"
            f"User's question: {question}\n\n"
            "Answer per your rules, using only the context above."
        )
        messages.append({"role": "user", "content": prompt})
        return messages

    def answer(
        self,
        question: str,
        contexts: list[RetrievedChunk],
        history: list[Turn],
        scope: Scope,
    ) -> Generation:
        response = self._client.messages.create(
            model=self._model,
            max_tokens=SETTINGS.anthropic_max_tokens,
            system=self._system_prompt(scope),
            messages=self._build_messages(question, contexts, history),
        )
        parts = [block.text for block in response.content if block.type == "text"]
        text, indices = _parse_sources("\n".join(parts).strip(), len(contexts))
        return Generation(text=text, source_indices=indices)

    def stream(
        self,
        question: str,
        contexts: list[RetrievedChunk],
        history: list[Turn],
        scope: Scope,
    ) -> Iterator[StreamEvent]:
        messages = self._build_messages(question, contexts, history)
        with self._client.messages.stream(
            model=self._model,
            max_tokens=SETTINGS.anthropic_max_tokens,
            system=self._system_prompt(scope),
            messages=messages,
        ) as stream:
            yield from self._emit_hiding_sources(stream.text_stream, len(contexts))

    @staticmethod
    def _emit_hiding_sources(
        text_iter: Iterator[str], n_contexts: int
    ) -> Iterator[StreamEvent]:
        """Stream display deltas while withholding the trailing SOURCES marker.

        The model ends every reply with a ``SOURCES:`` line that must never reach
        the user. We buffer the tail so the marker is detected and stripped before
        it is shown, then emit the parsed source indices as the final event.
        """

        full = ""
        emitted = 0
        marker_start: int | None = None
        for delta in text_iter:
            full += delta
            match = _SOURCES_LINE.search(full)
            if match:
                marker_start = match.start()
            safe = (
                marker_start
                if marker_start is not None
                else max(0, len(full) - _SOURCES_HOLDBACK)
            )
            if safe > emitted:
                yield ("delta", full[emitted:safe])
                emitted = safe

        match = _SOURCES_LINE.search(full)
        display_end = match.start() if match else len(full)
        if display_end > emitted:
            yield ("delta", full[emitted:display_end])

        _, indices = _parse_sources(full, n_contexts)
        yield ("sources", indices)


def build_answerer() -> Answerer:
    """Pick the best available answerer based on environment configuration."""

    key = anthropic_api_key()
    if key:
        try:
            logger.info(
                "Using Claude (%s) for answer generation.", SETTINGS.anthropic_model
            )
            return ClaudeAnswerer(api_key=key)
        except Exception as exc:  # pragma: no cover - defensive import/runtime guard
            logger.warning("Falling back to extractive answerer: %s", exc)
    else:
        logger.info("ANTHROPIC_API_KEY not set; using extractive answerer.")
    return ExtractiveAnswerer()
