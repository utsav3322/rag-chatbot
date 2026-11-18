"""Tests for simulated authentication and access-control policy."""

from __future__ import annotations

import pytest

from docqa.access import AccessControl, AccessError


@pytest.mark.unit
def test_authenticate_known_user_is_case_insensitive(access: AccessControl) -> None:
    user = access.authenticate("ALICE@email.com")
    assert user.email == "alice@email.com"
    assert user.companies == frozenset({"novatech"})


@pytest.mark.unit
def test_authenticate_unknown_user_raises(access: AccessControl) -> None:
    with pytest.raises(AccessError):
        access.authenticate("mallory@email.com")


@pytest.mark.unit
def test_can_access_enforces_scope(access: AccessControl) -> None:
    assert access.can_access("bob@email.com", "helios") is True
    assert access.can_access("bob@email.com", "novatech") is False


@pytest.mark.unit
def test_unknown_company_in_policy_is_dropped() -> None:
    ac = AccessControl.from_mapping(
        {"users": [{"email": "x@e.com", "companies": ["novatech", "ghostco"]}]}
    )
    assert ac.authenticate("x@e.com").companies == frozenset({"novatech"})


@pytest.mark.unit
def test_empty_policy_rejected() -> None:
    with pytest.raises(ValueError):
        AccessControl.from_mapping({"users": []})
