# Multi-User Document Search & Conversational Q&A

A Python web application that lets multiple users query a collection of
earnings-call PDFs and get relevant, **source-cited** answers — while enforcing
**per-user document access control** and maintaining **conversational context**
for follow-up questions.

Built for the assignment *"Multi-User Document Search and Conversational Q&A
System."*

---

## What it does

- 🔐 **Per-user access control** — users sign in with an **email + password**
  (validated against a **SQLite** `users` table; passwords are stored only as
  salted hashes). Each account may read only the companies it's authorised for,
  and the filter is pushed **into the vector database** as a `where` clause, so a
  user can never receive — or even be ranked against — a document they can't see.
- 🔎 **Semantic document search (RAG)** — PDFs are chunked, **embedded**
  (all-MiniLM-L6-v2, run via ONNX — no PyTorch), and stored in a persistent
  **Chroma** vector store. Queries are matched by **cosine similarity over
  embeddings**, so relevant passages are found even without keyword overlap.
  Every answer shows the exact source passages (company, file, page range,
  score). Falls back to a dependency-free TF-IDF retriever if embeddings are
  unavailable.
- 🗂️ **SQLite catalogue** — users, companies, the user↔company access map, and a
  `documents` table (title, page count, chunk count) are all stored in and
  fetched from SQLite; the embeddings live in Chroma alongside it.
- 💬 **Conversational Q&A** — per-session history lets follow-up questions keep
  context (e.g. *"What was the revenue?"* → *"And the operating margin?"*).
- 🧭 **Assistant-style intent routing** — greetings and "what can you do?" get a
  friendly, scope-aware reply; **out-of-scope questions** (general knowledge,
  coding, companies you can't access) are declined gracefully with a reminder of
  what you *can* ask — rather than a blunt "no results". Answers cite **only the
  passages actually used** (the model reports its sources), so declines and
  follow-ups show no noisy citations.
- 👥 **Simultaneous multi-user** — a shared read-only index plus isolated
  per-session conversation state means users never interfere with each other.
- 🖥️ **Professional streaming web UI** — a single-page chat app (Tailwind CSS)
  with **token-by-token streaming** over Server-Sent Events, live **Markdown**
  rendering, **syntax-highlighted** code blocks, a **Stop-generating** button
  (abortable mid-stream), a typing indicator, expandable source cards, and a
  login screen.
- 🤖 **Pluggable answers** — uses **Claude** (via the Anthropic API) when
  `ANTHROPIC_API_KEY` is set; otherwise falls back to an **extractive** mode so
  the system is fully demonstrable with **zero API keys**.

---

## Architecture

```
     Single-page UI (Tailwind + marked)  — streams tokens via SSE
                      │  fetch + ReadableStream   (one session cookie per user)
             Flask JSON + SSE API  (src/docqa/web/)
          GET /api/config · POST /api/login (email+password) · GET /api/documents
                      │                      · POST /api/ask (stream)
                QAEngine  (src/docqa/engine.py)   ← thread-safe, shared
                      │   ask_stream() yields ("token", …) then ("final", Answer)
            ┌─────────┼───────────────┬───────────────┐
            │         │               │               │
      AccessControl  ChromaRetriever  Conversation   Answerer
      (access.py)    (vectorstore.py) (per session)  (llm.py)
            │             │                             │
      ┌─────┴─────┐   embeddings query            Claude  OR  Extractive
      │  SQLite   │   where company_key ∈ allowed (if API key)  (fallback)
      │ docqa.db  │       │
      │ users ·   │   Chroma vector store  (data/chroma/)
      │ companies │       │   all-MiniLM-L6-v2 embeddings (ONNX)
      │ documents │       │
      │ access    │   ingest.py ── pypdf ── data/pdfs/*.pdf
      └───────────┘       ▲                      │
            └─ seeded + embedded by ─────────────┘
                scripts/build_index.py
```

**Why these choices**

| Concern | Choice | Reason |
|---|---|---|
| Retrieval | **Chroma** + MiniLM embeddings (ONNX) | Real semantic search (finds answers without keyword overlap); persisted on disk; no PyTorch. Degrades to pure-Python TF-IDF if the model can't be fetched (`DOCQA_VECTOR_STORE=0`). |
| Catalogue | **SQLite** (`docqa.db`) | Zero-setup relational store for users, companies, documents, and the access map; the system-of-record queried on every login. |
| Auth | Email + password, salted hash | Validated against the `users` table; passwords never stored in clear text. Demo password `123456` for all seeded users. |
| Isolation | Filter by company in the vector-store `where` clause | One auditable choke point, pushed into the DB; unauthorised vectors are never scored. |
| Multi-user | Shared read-only index + per-session `Conversation` | Safe concurrency without locks on the hot path. |
| LLM | Optional Claude, extractive fallback | Works offline for grading; upgrades to synthesised answers with a key. |
| Sample data | Generated fictional earnings calls | Self-contained, reproducible, no third-party copyright. |
| Document catalogue | SQLite `documents` table | `/api/documents` lists each user's authorised files (title, pages, chunk count), fetched straight from the DB. |
| Streaming | SSE + `fetch` ReadableStream | Real-time token streaming with no WebSocket/ASGI complexity. |
| Frontend | Tailwind + marked + DOMPurify + highlight.js via CDN | Professional, modern UI with **no Node/build step** — just open the page. |
| Stop control | `AbortController` cancels the `fetch` stream | Closes the SSE connection, which exits the Claude stream cleanly server-side. |

---

## Quick start

```bash
# 1. Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Build the data layer: generate PDFs, seed SQLite, embed chunks into Chroma
#    (also auto-runs on first launch; use --reset to re-embed from scratch)
python scripts/build_index.py

# 4. Run the app
python run.py
```

Then open **http://127.0.0.1:5000** and log in as one of the sample users with the
demo password **`123456`**. (The first launch downloads the ~80 MB embedding
model once; after that it runs fully offline.)

> **Optional — enable Claude answers:** copy `.env.example` to `.env` and set
> `ANTHROPIC_API_KEY=...`. Without it the app runs in extractive mode. The UI
> footer tells you which mode is active.

---

## Sample users & access

All sample users share the demo password **`123456`**.

| Email | Password | Can access |
|---|---|---|
| `alice@email.com` | `123456` | NovaTech Industries |
| `bob@email.com` | `123456` | Helios Energy Corp, Verdant Foods |
| `charlie@email.com` | `123456` | Orion Logistics, Lumina Pharmaceuticals |
| `dana@email.com` | `123456` | NovaTech Industries, Lumina Pharmaceuticals |
| `admin@email.com` | `123456` | All five companies |

Documents (fictional Q4 FY2025 earnings calls, one PDF each):
**NovaTech Industries** (NOVA), **Helios Energy Corp** (HELI),
**Verdant Foods** (VERD), **Orion Logistics** (ORIO),
**Lumina Pharmaceuticals** (LUMA).

`config/access_control.json` is the **seed** for the `users` / `user_access`
tables (loaded on first run, when the DB is empty). To change who sees what,
edit it and re-seed with `python scripts/build_index.py` (or delete `data/docqa.db`
and relaunch) — no code changes needed.

---

## Demo script (covers every required point)

1. **Log in as different users** — sign in as `alice@email.com`.
2. **Query & retrieve** — ask *"What was the revenue reported in Q4?"* →
   returns NovaTech's `$2.34 billion` with the source passage cited.
3. **Conversational follow-up** — ask *"And the full-year guidance?"* then
   *"What about the operating margin?"* → context carries across turns.
4. **Query isolation** — still as Alice, ask *"What was Helios's solar revenue?"*
   → the system returns nothing from Helios, because Alice isn't authorised.
   Log in as `bob@email.com` in a different browser/incognito window and ask the
   same question → Bob **does** get the Helios answer. The two sessions never mix.
5. **Simultaneous users** — keep both windows open and query at the same time;
   each conversation stays isolated.

---

## Running the tests

```bash
pip install -r requirements.txt
python -m pytest -q
```

59 tests cover access control, the SQLite catalogue (password hashing/validation,
access scoping, document listing), retrieval, cross-company isolation, intent
routing, source attribution, conversational history, PDF ingestion, and the Flask
JSON/SSE API (password login, the documents endpoint, SSE token streaming, and a
test that two browser clients don't share state).

> Tests run offline against the TF-IDF retriever and a temporary SQLite file — no
> Chroma or model download needed. CI-friendly: `DOCQA_VECTOR_STORE=0 pytest -q`.

```bash
# With coverage
pip install pytest-cov
python -m pytest --cov=src/docqa --cov-report=term-missing
```

---

## Project layout

```
multi-user-doc-qa/
├── README.md
├── requirements.txt
├── pyproject.toml              # black / isort / ruff / pytest config
├── run.py                      # launcher: python run.py
├── .env.example
├── config/
│   └── access_control.json     # seed: email → allowed companies
├── data/
│   ├── pdfs/                   # generated sample PDFs (committed)
│   ├── docqa.db                # SQLite catalogue (generated, git-ignored)
│   └── chroma/                 # Chroma vector store (generated, git-ignored)
├── scripts/
│   ├── generate_sample_pdfs.py
│   └── build_index.py          # seed SQLite + embed chunks into Chroma
├── src/docqa/
│   ├── config.py               # paths, tunables, company registry
│   ├── models.py               # frozen dataclasses (Chunk, Answer, Document…)
│   ├── access.py               # access control (from SQLite or JSON)
│   ├── db.py                   # SQLite catalogue: users, companies, docs, access
│   ├── ingest.py               # pypdf loading + chunking
│   ├── retrieval.py            # pure-Python TF-IDF retriever (fallback)
│   ├── vectorstore.py          # Chroma embeddings + semantic retriever
│   ├── intent.py               # intent routing + scope-aware replies
│   ├── llm.py                  # Claude + extractive answerers (streaming + attribution)
│   ├── engine.py               # orchestration, per-session conversations, ask_stream()
│   └── web/
│       ├── app.py              # Flask JSON + SSE API
│       └── static/index.html   # single-page streaming chat UI (Tailwind)
└── tests/                      # pytest suite
```

---

## Configuration (environment variables)

| Variable | Default | Purpose |
|---|---|---|
| `ANTHROPIC_API_KEY` | _(unset)_ | Enables Claude-synthesised answers |
| `DOCQA_ANTHROPIC_MODEL` | `claude-opus-4-8` | Model for answer generation |
| `DOCQA_VECTOR_STORE` | `1` | `0` forces the TF-IDF fallback (skips Chroma/embeddings) |
| `DOCQA_DB_FILE` | `data/docqa.db` | SQLite catalogue location |
| `DOCQA_CHROMA_DIR` | `data/chroma` | Chroma persistent store location |
| `DOCQA_COLLECTION` | `earnings_chunks` | Chroma collection name |
| `DOCQA_DEMO_PASSWORD` | `123456` | Shared password for seeded users |
| `DOCQA_HOST` / `DOCQA_PORT` | `127.0.0.1` / `5000` | Web server bind |
| `DOCQA_TOP_K` | `4` | Passages retrieved per query |
| `DOCQA_SECRET_KEY` | random | Flask session signing key |

---

## Notes & limitations

- Retrieval uses sentence-embeddings (all-MiniLM-L6-v2) in Chroma; a pure-Python
  TF-IDF retriever is kept as an automatic offline fallback. For larger corpora
  you'd use a hosted vector DB and a stronger embedding model — the
  `ChromaRetriever` / `Retriever` share one interface, so that's a drop-in swap.
- Authentication is **simple but real**: email + salted-hash password checked
  against SQLite. It intentionally omits registration, password reset, sessions-
  at-scale, and rate limiting — out of scope for this demo. All seeded users
  share the password `123456`.
- The bundled Flask server is for local/demo use; deploy behind a WSGI server
  (gunicorn/uWSGI) for production.
