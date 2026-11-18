"""SQLite catalogue for users, companies, documents, and access control.

This is the system's *system-of-record* for everything except the chunk
embeddings (those live in the Chroma vector store). The schema is deliberately
small and relational:

    companies(key, name, ticker)
    users(id, email, password_hash, display_name, created_at)
    documents(id, company_key → companies, filename, title, num_pages, …)
    user_access(user_id → users, company_key → companies)   [many-to-many]

Passwords are never stored in clear text — only a salted hash (werkzeug's
PBKDF2). Login validates ``email`` + ``password`` against the ``users`` table;
the shared demo password is :data:`docqa.config.DEMO_PASSWORD` ("123456").

A fresh short-lived connection is opened per operation so the object is safe to
share across the Flask dev server's worker threads without juggling
``check_same_thread``.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from collections.abc import Iterable, Sequence
from contextlib import contextmanager
from pathlib import Path

from werkzeug.security import check_password_hash, generate_password_hash

from docqa.config import (
    ACCESS_CONTROL_FILE,
    COMPANIES,
    DB_FILE,
    DEMO_PASSWORD,
    company_name,
)
from docqa.models import Document

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS companies (
    key    TEXT PRIMARY KEY,
    name   TEXT NOT NULL,
    ticker TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS users (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    email         TEXT NOT NULL UNIQUE,
    display_name  TEXT NOT NULL,
    password_hash TEXT NOT NULL,
    created_at    TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS documents (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    company_key TEXT NOT NULL REFERENCES companies(key),
    filename    TEXT NOT NULL UNIQUE,
    title       TEXT NOT NULL,
    num_pages   INTEGER NOT NULL,
    num_chunks  INTEGER NOT NULL,
    indexed_at  TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS user_access (
    user_id     INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    company_key TEXT NOT NULL REFERENCES companies(key),
    PRIMARY KEY (user_id, company_key)
);
"""


class Database:
    """Thin, thread-safe wrapper over the SQLite catalogue."""

    def __init__(self, path: Path = DB_FILE) -> None:
        self._path = Path(path)
        if self._path.parent and str(self._path) != ":memory:":
            self._path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def path(self) -> Path:
        return self._path

    # -- Connection ---------------------------------------------------------

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(str(self._path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    # -- Seeding ------------------------------------------------------------

    def seed_companies(self) -> None:
        """Populate the ``companies`` table from the in-code registry."""

        with self._connect() as conn:
            conn.executemany(
                "INSERT OR REPLACE INTO companies(key, name, ticker) VALUES (?, ?, ?)",
                [(c.key, c.name, c.ticker) for c in COMPANIES],
            )

    def seed_users_from_json(
        self,
        path: Path = ACCESS_CONTROL_FILE,
        password: str = DEMO_PASSWORD,
    ) -> int:
        """Seed users + access from the access-control JSON (idempotent).

        Every user gets the same hashed demo ``password``. Returns the number of
        users written.
        """

        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        users = raw.get("users", [])
        count = 0
        for entry in users:
            email = str(entry["email"]).strip().lower()
            display_name = str(entry.get("display_name", email))
            companies = [str(c) for c in entry.get("companies", [])]
            self.upsert_user(email, display_name, password, companies)
            count += 1
        logger.info("Seeded %d users into %s", count, self._path.name)
        return count

    def upsert_user(
        self,
        email: str,
        display_name: str,
        password: str,
        company_keys: Iterable[str],
    ) -> int:
        """Create or update a user and (re)write their company access list."""

        email = email.strip().lower()
        pwd_hash = generate_password_hash(password)
        valid = {c.key for c in COMPANIES}
        keys = [k for k in company_keys if k in valid]
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO users(email, display_name, password_hash)
                VALUES (?, ?, ?)
                ON CONFLICT(email) DO UPDATE SET
                    display_name = excluded.display_name,
                    password_hash = excluded.password_hash
                """,
                (email, display_name, pwd_hash),
            )
            row = conn.execute(
                "SELECT id FROM users WHERE email = ?", (email,)
            ).fetchone()
            user_id = int(row["id"])
            conn.execute("DELETE FROM user_access WHERE user_id = ?", (user_id,))
            conn.executemany(
                "INSERT OR IGNORE INTO user_access(user_id, company_key) VALUES (?, ?)",
                [(user_id, k) for k in keys],
            )
        return user_id

    # -- Queries ------------------------------------------------------------

    def user_count(self) -> int:
        with self._connect() as conn:
            return int(conn.execute("SELECT COUNT(*) AS n FROM users").fetchone()["n"])

    def document_count(self) -> int:
        with self._connect() as conn:
            return int(
                conn.execute("SELECT COUNT(*) AS n FROM documents").fetchone()["n"]
            )

    def get_user(self, email: str) -> dict | None:
        """Return ``{email, display_name, companies}`` for ``email`` or ``None``."""

        email = (email or "").strip().lower()
        with self._connect() as conn:
            row = conn.execute(
                "SELECT id, email, display_name FROM users WHERE email = ?",
                (email,),
            ).fetchone()
            if row is None:
                return None
            companies = [
                r["company_key"]
                for r in conn.execute(
                    "SELECT company_key FROM user_access WHERE user_id = ? "
                    "ORDER BY company_key",
                    (row["id"],),
                ).fetchall()
            ]
        return {
            "email": row["email"],
            "display_name": row["display_name"],
            "companies": companies,
        }

    def list_users(self) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, email, display_name FROM users ORDER BY id"
            ).fetchall()
            out = []
            for row in rows:
                companies = [
                    r["company_key"]
                    for r in conn.execute(
                        "SELECT company_key FROM user_access WHERE user_id = ? "
                        "ORDER BY company_key",
                        (row["id"],),
                    ).fetchall()
                ]
                out.append(
                    {
                        "email": row["email"],
                        "display_name": row["display_name"],
                        "companies": companies,
                    }
                )
        return out

    def verify_credentials(self, email: str, password: str) -> bool:
        """Return ``True`` only if ``email`` exists and ``password`` matches."""

        email = (email or "").strip().lower()
        with self._connect() as conn:
            row = conn.execute(
                "SELECT password_hash FROM users WHERE email = ?", (email,)
            ).fetchone()
        if row is None:
            return False
        return check_password_hash(row["password_hash"], password or "")

    # -- Documents ----------------------------------------------------------

    def record_document(
        self,
        *,
        company_key: str,
        filename: str,
        title: str,
        num_pages: int,
        num_chunks: int,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO documents(company_key, filename, title, num_pages, num_chunks)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(filename) DO UPDATE SET
                    company_key = excluded.company_key,
                    title = excluded.title,
                    num_pages = excluded.num_pages,
                    num_chunks = excluded.num_chunks,
                    indexed_at = datetime('now')
                """,
                (company_key, filename, title, num_pages, num_chunks),
            )

    def list_documents(
        self, company_keys: Sequence[str] | None = None
    ) -> list[Document]:
        """List catalogued documents, optionally restricted to ``company_keys``."""

        sql = (
            "SELECT company_key, filename, title, num_pages, num_chunks "
            "FROM documents"
        )
        params: tuple = ()
        if company_keys is not None:
            keys = list(company_keys)
            if not keys:
                return []
            placeholders = ",".join("?" for _ in keys)
            sql += f" WHERE company_key IN ({placeholders})"
            params = tuple(keys)
        sql += " ORDER BY company_key, filename"
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [
            Document(
                company_key=r["company_key"],
                company_name=company_name(r["company_key"]),
                filename=r["filename"],
                title=r["title"],
                num_pages=int(r["num_pages"]),
                num_chunks=int(r["num_chunks"]),
            )
            for r in rows
        ]

    # -- Convenience --------------------------------------------------------

    def ensure_seeded(self) -> None:
        """Create the schema and seed companies + users if the DB is empty."""

        self.init_schema()
        self.seed_companies()
        if self.user_count() == 0:
            self.seed_users_from_json()
