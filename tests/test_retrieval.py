"""Tests for the TF-IDF retriever, including access-scoped filtering."""

from __future__ import annotations

import pytest

from docqa.retrieval import Retriever, tokenize


@pytest.mark.unit
def test_tokenize_drops_stopwords_keeps_numbers() -> None:
    tokens = tokenize("The revenue was 2.34 billion")
    assert "the" not in tokens and "was" not in tokens
    assert "revenue" in tokens
    assert "2.34" in tokens


@pytest.mark.unit
def test_search_returns_relevant_chunk(retriever: Retriever) -> None:
    results = retriever.search(
        "What was NovaTech revenue?",
        allowed_companies={"novatech"},
        top_k=2,
    )
    assert results
    assert results[0].chunk.company_key == "novatech"
    assert results[0].score > 0


@pytest.mark.unit
def test_search_never_returns_disallowed_company(retriever: Retriever) -> None:
    # Ask about NovaTech but only allow lumina/helios — must not leak NovaTech.
    results = retriever.search(
        "NovaTech cloud platform revenue growth",
        allowed_companies={"helios", "lumina"},
        top_k=5,
    )
    assert all(rc.chunk.company_key in {"helios", "lumina"} for rc in results)


@pytest.mark.unit
def test_empty_allowed_set_returns_nothing(retriever: Retriever) -> None:
    assert retriever.search("revenue", allowed_companies=set(), top_k=5) == []


@pytest.mark.unit
def test_top_k_is_respected(retriever: Retriever) -> None:
    results = retriever.search(
        "revenue billion",
        allowed_companies={"novatech", "helios", "lumina"},
        top_k=1,
    )
    assert len(results) <= 1
