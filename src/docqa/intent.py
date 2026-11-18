"""Lightweight intent routing so the system behaves like a real assistant.

Before touching the RAG pipeline we classify the incoming message. Greetings,
thanks, goodbyes, and "what can you do" questions are answered conversationally
without a document lookup; everything else is treated as a potential document
query and sent through retrieval. This mirrors how production assistants route
between small talk and grounded answers.

The canned responses are intentionally *scope-aware*: they name the companies
the current user may access and offer concrete example questions, so a user is
never left guessing what they can ask.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from enum import Enum


class Intent(str, Enum):
    GREETING = "greeting"
    THANKS = "thanks"
    GOODBYE = "goodbye"
    HELP = "help"
    QUERY = "query"


# Anchored so that only a *pure* greeting matches; "hi, what was revenue?" is a
# query, not a greeting.
_GREETING = re.compile(
    r"^(hi|hello+|hey+|yo|hiya|howdy|sup|"
    r"good\s*(morning|afternoon|evening|day)|greetings?|namaste)"
    r"[\s!.,]*$",
    re.IGNORECASE,
)
_GOODBYE = re.compile(
    r"^(bye+|goodbye|see\s*(you|ya)|cya|later|take\s*care)[\s!.,]*$",
    re.IGNORECASE,
)
_THANKS = re.compile(
    r"\b(thanks?|thank\s*you|thx|ty|appreciate\s*it|cheers)\b",
    re.IGNORECASE,
)
_HELP = re.compile(
    r"("
    r"what\s+can\s+you\s+(do|help|answer)"
    r"|what\s+do\s+you\s+do"
    r"|who\s+are\s+you"
    r"|what\s+are\s+you"
    r"|how\s+do\s+you\s+work"
    r"|what\s+can\s+i\s+ask"
    r"|help\s*me"
    r"|your\s+capabilities"
    r"|^help[\s!.?]*$"
    r")",
    re.IGNORECASE,
)


def classify(message: str) -> Intent:
    """Classify a user message into a routing :class:`Intent`."""

    text = message.strip()
    if not text:
        return Intent.QUERY
    if _GREETING.match(text):
        return Intent.GREETING
    if _GOODBYE.match(text):
        return Intent.GOODBYE
    if _HELP.search(text):
        return Intent.HELP
    # Only treat as a bare "thanks" when the message is short, so
    # "thanks, now what was the revenue?" still routes to a query.
    if _THANKS.search(text) and len(text.split()) <= 4:
        return Intent.THANKS
    return Intent.QUERY


def example_questions(companies: Sequence[str]) -> tuple[str, ...]:
    """Concrete example questions tailored to the user's accessible companies."""

    if not companies:
        return ()
    first = companies[0]
    examples = [
        f"What was {first}'s revenue in Q4?",
        f"What guidance did {first} give for next year?",
        f"Summarise the key points from {first}'s earnings call.",
    ]
    if len(companies) > 1:
        examples.append(f"Compare revenue growth for {companies[0]} and {companies[1]}.")
    return tuple(examples)


def _companies_phrase(companies: Sequence[str]) -> str:
    if not companies:
        return "no companies (your account has no document access configured)"
    if len(companies) == 1:
        return companies[0]
    return ", ".join(companies[:-1]) + f" and {companies[-1]}"


def _examples_block(companies: Sequence[str]) -> str:
    examples = example_questions(companies)
    if not examples:
        return ""
    bullets = "\n".join(f"  • {q}" for q in examples[:3])
    return f"\n\nFor example, you could ask:\n{bullets}"


def greeting_text(companies: Sequence[str]) -> str:
    return (
        f"Hi! 👋 I'm your earnings-call assistant. I can answer questions about "
        f"the documents you have access to — {_companies_phrase(companies)}. "
        f"Ask me about revenue, guidance, margins, segments, or anything else "
        f"from their latest earnings calls, and I'll answer with the exact source "
        f"passages." + _examples_block(companies)
    )


def help_text(companies: Sequence[str]) -> str:
    return (
        "Here's what I can do:\n"
        "  • Search your authorised earnings-call documents and answer questions "
        "with cited excerpts.\n"
        "  • Hold a conversation — ask follow-ups like \"and the margin?\" and I'll "
        "keep the context.\n"
        "  • Stay within your access — I only ever use documents you're allowed to "
        "see.\n\n"
        f"Right now you have access to: {_companies_phrase(companies)}."
        + _examples_block(companies)
    )


def thanks_text(companies: Sequence[str]) -> str:
    return (
        "You're welcome! Happy to keep going — ask me anything else about "
        f"{_companies_phrase(companies)}."
    )


def goodbye_text() -> str:
    return "Goodbye! 👋 Come back anytime you need to dig into those earnings calls."


def out_of_scope_text(companies: Sequence[str]) -> str:
    return (
        "That's a bit outside what I can help with. I'm limited to the earnings-call "
        f"documents you have access to — {_companies_phrase(companies)} — and I "
        "couldn't find anything relevant to that question in them. "
        "Try asking about their revenue, guidance, margins, or segments."
        + _examples_block(companies)
    )
