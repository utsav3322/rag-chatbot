"""Integration tests for the QA engine: isolation, citations, conversation."""

from __future__ import annotations

import pytest

from docqa.access import AccessError
from docqa.engine import QAEngine, _rewrite_follow_up
from docqa.models import Turn


@pytest.mark.integration
def test_user_gets_answer_from_their_company(engine: QAEngine) -> None:
    ans = engine.ask(
        email="alice@email.com",
        session_id="s1",
        question="What was the revenue reported in Q4?",
    )
    assert ans.citations
    assert all(c.company_name == "NovaTech Industries" for c in ans.citations)
    assert ans.accessible_companies == ("NovaTech Industries",)


@pytest.mark.integration
def test_query_isolation_no_cross_company_leak(engine: QAEngine) -> None:
    # Alice (novatech only) asks about Helios -> no Helios citations.
    ans = engine.ask(
        email="alice@email.com",
        session_id="s1",
        question="What was Helios solar revenue and backlog?",
    )
    assert all(c.company_name == "NovaTech Industries" for c in ans.citations)


@pytest.mark.integration
def test_sessions_are_independent(engine: QAEngine) -> None:
    engine.ask(email="alice@email.com", session_id="sA", question="revenue?")
    engine.ask(email="bob@email.com", session_id="sB", question="revenue?")
    # Resetting one session must not affect the other.
    engine.reset("sA")
    convo_b = engine._conversation("sB")  # noqa: SLF001 - internal check
    assert len(convo_b.turns) == 1


@pytest.mark.integration
def test_conversation_history_accumulates(engine: QAEngine) -> None:
    engine.ask(email="bob@email.com", session_id="s", question="Helios revenue?")
    engine.ask(email="bob@email.com", session_id="s", question="And the backlog?")
    convo = engine._conversation("s")  # noqa: SLF001
    assert len(convo.turns) == 2


@pytest.mark.integration
def test_unknown_user_cannot_query(engine: QAEngine) -> None:
    with pytest.raises(AccessError):
        engine.ask(email="nobody@email.com", session_id="s", question="hi")


@pytest.mark.unit
def test_follow_up_rewrite_prepends_recent_context() -> None:
    history = [Turn(question="What was Helios revenue?", answer="3.78 billion")]
    rewritten = _rewrite_follow_up("and margins?", history)
    assert "Helios" in rewritten and "margins" in rewritten


@pytest.mark.unit
def test_follow_up_rewrite_leaves_long_questions_untouched() -> None:
    history = [Turn(question="x", answer="y")]
    long_q = "What was the full year revenue guidance for fiscal 2026 exactly?"
    assert _rewrite_follow_up(long_q, history) == long_q
