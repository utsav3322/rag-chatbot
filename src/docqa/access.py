"""User authentication (simulated) and document access control.

Access control is intentionally simple: a JSON file maps each user's email to
the set of company keys they may read. The :class:`AccessControl` object is the
single source of truth consulted on every query, which keeps the isolation
guarantee in one auditable place.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from docqa.config import ACCESS_CONTROL_FILE, COMPANY_BY_KEY

if TYPE_CHECKING:
    from docqa.db import Database

logger = logging.getLogger(__name__)


class AccessError(Exception):
    """Raised when a user cannot be authenticated."""


@dataclass(frozen=True)
class User:
    email: str
    display_name: str
    companies: frozenset[str]


class AccessControl:
    """In-memory view of the access-control policy loaded from JSON."""

    def __init__(self, users: dict[str, User]) -> None:
        self._users = users

    @classmethod
    def from_file(cls, path: Path = ACCESS_CONTROL_FILE) -> "AccessControl":
        if not path.exists():
            raise FileNotFoundError(
                f"Access-control file not found at {path}. "
                "Did you copy config/access_control.json?"
            )
        raw = json.loads(path.read_text(encoding="utf-8"))
        return cls.from_mapping(raw)

    @classmethod
    def from_db(cls, db: "Database") -> "AccessControl":
        """Load the access policy from the SQLite catalogue.

        The email → company map is cached in memory for fast per-request lookups;
        passwords are verified separately against the database on login.
        """

        users: dict[str, User] = {}
        for record in db.list_users():
            email = str(record["email"]).strip().lower()
            companies = frozenset(str(c) for c in record["companies"])
            users[email] = User(
                email=email,
                display_name=str(record.get("display_name", email)),
                companies=companies & set(COMPANY_BY_KEY),
            )
        if not users:
            raise ValueError("No users found in the database. Seed it first.")
        return cls(users)

    @classmethod
    def from_mapping(cls, raw: dict[str, object]) -> "AccessControl":
        users: dict[str, User] = {}
        for entry in raw.get("users", []):  # type: ignore[union-attr]
            assert isinstance(entry, dict)
            email = str(entry["email"]).strip().lower()
            companies = frozenset(str(c) for c in entry.get("companies", []))
            unknown = companies - set(COMPANY_BY_KEY)
            if unknown:
                logger.warning(
                    "User %s references unknown companies: %s", email, sorted(unknown)
                )
            users[email] = User(
                email=email,
                display_name=str(entry.get("display_name", email)),
                companies=companies & set(COMPANY_BY_KEY),
            )
        if not users:
            raise ValueError("Access-control file contains no users.")
        return cls(users)

    # -- Queries ------------------------------------------------------------

    def authenticate(self, email: str) -> User:
        """Return the :class:`User` for ``email`` or raise :class:`AccessError`.

        Authentication here is simulated: a known email is sufficient. This
        mirrors the assignment's "dummy email / access token" requirement.
        """

        normalized = (email or "").strip().lower()
        user = self._users.get(normalized)
        if user is None:
            raise AccessError(f"Unknown user: {email!r}")
        return user

    def allowed_companies(self, email: str) -> frozenset[str]:
        return self.authenticate(email).companies

    def can_access(self, email: str, company_key: str) -> bool:
        return company_key in self.authenticate(email).companies

    @property
    def users(self) -> tuple[User, ...]:
        return tuple(self._users.values())
