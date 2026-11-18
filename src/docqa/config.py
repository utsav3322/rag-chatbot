"""Central configuration: paths, tunables, and the company registry.

All settings can be overridden with environment variables so the system can be
reconfigured for a demo without editing code.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from docqa.models import Company

# --- Paths -----------------------------------------------------------------

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent.parent


def _load_dotenv(path: Path) -> None:
    """Minimal .env loader (avoids a python-dotenv dependency).

    Only sets variables that are not already present in the environment, so
    real environment variables always win.
    """

    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


_load_dotenv(PROJECT_ROOT / ".env")

DATA_DIR = Path(os.environ.get("DOCQA_DATA_DIR", PROJECT_ROOT / "data"))
PDF_DIR = DATA_DIR / "pdfs"
CONFIG_DIR = Path(os.environ.get("DOCQA_CONFIG_DIR", PROJECT_ROOT / "config"))
ACCESS_CONTROL_FILE = CONFIG_DIR / "access_control.json"

# SQLite catalogue (users, companies, documents, access) and the Chroma vector
# store (chunk embeddings) both live under the data directory by default.
DB_FILE = Path(os.environ.get("DOCQA_DB_FILE", DATA_DIR / "docqa.db"))
CHROMA_DIR = Path(os.environ.get("DOCQA_CHROMA_DIR", DATA_DIR / "chroma"))

# The shared demo password for every seeded user (stored only as a salted hash).
DEMO_PASSWORD = os.environ.get("DOCQA_DEMO_PASSWORD", "123456")


# --- Retrieval / chunking tunables ----------------------------------------


@dataclass(frozen=True)
class Settings:
    """Runtime tunables for ingestion, retrieval, and answer generation."""

    chunk_size_words: int = 180
    chunk_overlap_words: int = 40
    top_k: int = 4
    min_score: float = 0.0  # retrieval floor; the LLM/heuristic judges relevance
    citation_min_score: float = 0.05  # only surface citations at least this strong
    answer_min_score: float = 0.05  # extractive fallback: below this => out of scope
    excerpt_chars: int = 320
    anthropic_model: str = "claude-opus-4-8"
    anthropic_max_tokens: int = 700
    history_window: int = 6  # number of prior turns kept for context
    # Vector store (Chroma) tunables.
    collection_name: str = "earnings_chunks"
    vector_store_enabled: bool = True  # fall back to TF-IDF if False or unavailable

    @classmethod
    def from_env(cls) -> "Settings":
        def _int(name: str, default: int) -> int:
            raw = os.environ.get(name)
            return int(raw) if raw else default

        def _bool(name: str, default: bool) -> bool:
            raw = os.environ.get(name)
            if raw is None:
                return default
            return raw.strip().lower() not in {"0", "false", "no", "off", ""}

        return cls(
            chunk_size_words=_int("DOCQA_CHUNK_SIZE", cls.chunk_size_words),
            chunk_overlap_words=_int("DOCQA_CHUNK_OVERLAP", cls.chunk_overlap_words),
            top_k=_int("DOCQA_TOP_K", cls.top_k),
            anthropic_model=os.environ.get("DOCQA_ANTHROPIC_MODEL", cls.anthropic_model),
            collection_name=os.environ.get(
                "DOCQA_COLLECTION", cls.collection_name
            ),
            vector_store_enabled=_bool(
                "DOCQA_VECTOR_STORE", cls.vector_store_enabled
            ),
        )


SETTINGS = Settings.from_env()


# --- Company registry ------------------------------------------------------
# The ``key`` is the stable identifier used in access-control config and as the
# PDF filename stem. ``name`` is shown in the UI.

COMPANIES: tuple[Company, ...] = (
    Company(key="novatech", name="NovaTech Industries", ticker="NOVA"),
    Company(key="helios", name="Helios Energy Corp", ticker="HELI"),
    Company(key="verdant", name="Verdant Foods", ticker="VERD"),
    Company(key="orion", name="Orion Logistics", ticker="ORIO"),
    Company(key="lumina", name="Lumina Pharmaceuticals", ticker="LUMA"),
)

COMPANY_BY_KEY: dict[str, Company] = {c.key: c for c in COMPANIES}


def company_name(key: str) -> str:
    company = COMPANY_BY_KEY.get(key)
    return company.name if company else key


def anthropic_api_key() -> str | None:
    """Return the Anthropic API key if configured, else ``None``."""

    key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    return key or None
