"""Tests for the SQLite catalogue: users, passwords, access, and documents."""

from __future__ import annotations

import pytest

from docqa.access import AccessControl
from docqa.db import Database


@pytest.mark.unit
def test_verify_credentials_accepts_demo_password(db: Database) -> None:
    assert db.verify_credentials("alice@email.com", "123456") is True


@pytest.mark.unit
def test_verify_credentials_rejects_wrong_password(db: Database) -> None:
    assert db.verify_credentials("alice@email.com", "nope") is False


@pytest.mark.unit
def test_verify_credentials_rejects_unknown_user(db: Database) -> None:
    assert db.verify_credentials("ghost@email.com", "123456") is False


@pytest.mark.unit
def test_password_is_hashed_not_plaintext(db: Database, tmp_path) -> None:
    import sqlite3

    conn = sqlite3.connect(str(db.path))
    row = conn.execute(
        "SELECT password_hash FROM users WHERE email = ?", ("alice@email.com",)
    ).fetchone()
    conn.close()
    assert row is not None
    assert "123456" not in row[0]  # stored as a salted hash, never clear text


@pytest.mark.unit
def test_access_control_from_db_matches_seed(db: Database) -> None:
    access = AccessControl.from_db(db)
    assert access.authenticate("alice@email.com").companies == frozenset({"novatech"})
    assert access.authenticate("bob@email.com").companies == frozenset(
        {"helios", "lumina"}
    )


@pytest.mark.unit
def test_list_documents_is_company_scoped(db: Database) -> None:
    nova = db.list_documents(["novatech"])
    assert [d.company_key for d in nova] == ["novatech"]
    # No company filter -> all catalogued documents.
    assert len(db.list_documents()) >= len(nova)


@pytest.mark.unit
def test_upsert_user_is_idempotent_and_updates_access(tmp_path) -> None:
    database = Database(tmp_path / "x.db")
    database.init_schema()
    database.seed_companies()
    database.upsert_user("x@e.com", "X", "123456", ["novatech"])
    database.upsert_user("x@e.com", "X", "123456", ["helios"])  # re-scope
    assert database.user_count() == 1
    assert database.get_user("x@e.com")["companies"] == ["helios"]


@pytest.mark.unit
def test_unknown_company_in_access_is_ignored(tmp_path) -> None:
    database = Database(tmp_path / "y.db")
    database.init_schema()
    database.seed_companies()
    database.upsert_user("y@e.com", "Y", "123456", ["novatech", "ghostco"])
    assert database.get_user("y@e.com")["companies"] == ["novatech"]
