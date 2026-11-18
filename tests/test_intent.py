"""Tests for intent classification and conversational routing in the engine."""

from __future__ import annotations

import pytest

from docqa.engine import QAEngine
from docqa.intent import Intent, classify, example_questions


@pytest.mark.unit
@pytest.mark.parametrize(
    "message, expected",
    [
        ("hi", Intent.GREETING),
        ("Hello!", Intent.GREETING),
        ("good morning", Intent.GREETING),
        ("hey there", Intent.QUERY),  # not a bare greeting
        ("bye", Intent.GOODBYE),
        ("thanks", Intent.THANKS),
        ("thank you so much", Intent.THANKS),
        ("what can you do?", Intent.HELP),
        ("who are you", Intent.HELP),
        ("What was the revenue in Q4?", Intent.QUERY),
        ("hi, what was the revenue?", Intent.QUERY),  # greeting + question => query
        ("thanks, now what about guidance?", Intent.QUERY),  # long => query
    ],
)
def test_classify(message: str, expected: Intent) -> None:
    assert classify(message) == expected


@pytest.mark.unit
def test_example_questions_reference_accessible_company() -> None:
    examples = example_questions(["Helios Energy Corp", "Verdant Foods"])
    assert examples
    assert any("Helios Energy Corp" in q for q in examples)


@pytest.mark.integration
def test_greeting_is_answered_without_citations(engine: QAEngine) -> None:
    ans = engine.ask(email="bob@email.com", session_id="s", question="hi")
    assert ans.citations == ()
    # Mentions the user's accessible companies so they know what they can ask.
    assert "Helios Energy Corp" in ans.text or "Verdant" in ans.text


@pytest.mark.integration
def test_help_lists_scope(engine: QAEngine) -> None:
    ans = engine.ask(email="alice@email.com", session_id="s", question="what can you do?")
    assert ans.citations == ()
    assert "NovaTech Industries" in ans.text


@pytest.mark.integration
def test_out_of_scope_question_declines_gracefully(engine: QAEngine) -> None:
    # Extractive answerer (test engine) returns the scope-aware decline text.
    ans = engine.ask(
        email="alice@email.com",
        session_id="s",
        question="What's the weather in Paris tomorrow?",
    )
    assert ans.citations == ()
    assert "outside" in ans.text.lower() or "scope" in ans.text.lower()
    assert "NovaTech Industries" in ans.text


@pytest.mark.integration
def test_real_question_still_answered_with_citations(engine: QAEngine) -> None:
    ans = engine.ask(
        email="alice@email.com",
        session_id="s",
        question="What was the revenue reported in Q4?",
    )
    assert ans.citations
    assert all(c.company_name == "NovaTech Industries" for c in ans.citations)


@pytest.mark.unit
def test_parse_sources_marker() -> None:
    from docqa.llm import _parse_sources

    text, idx = _parse_sources("Revenue was $3.78B.\nSOURCES: 1, 3", 4)
    assert text == "Revenue was $3.78B."
    assert idx == (1, 3)

    text2, idx2 = _parse_sources("That's out of scope.\nSOURCES: none", 4)
    assert idx2 == ()

    text3, idx3 = _parse_sources("No marker here.", 4)
    assert idx3 is None

    # Out-of-range indices are dropped.
    _, idx4 = _parse_sources("x\nSOURCES: 2, 9", 4)
    assert idx4 == (2,)
