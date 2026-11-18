"""Flask application factory: a JSON + Server-Sent-Events API behind a single-page UI.

Per-user isolation rests on two things:

* A signed Flask session cookie holds the logged-in email and a random
  ``session_id``; each browser session is independent.
* Every request re-derives the allowed companies from the server-side
  :class:`AccessControl`, never from anything the client sends, so a user cannot
  widen their own access by tampering with the cookie.

Answers are streamed token-by-token over SSE (``POST /api/ask``). The single
:class:`QAEngine` is shared and thread-safe, so the threaded dev server can
stream to several users at once without their conversations mixing.
"""

from __future__ import annotations

import json
import logging
import os
import secrets
from pathlib import Path

from flask import (
    Flask,
    Response,
    jsonify,
    request,
    send_from_directory,
    session,
    stream_with_context,
)

from docqa.access import AccessError
from docqa.config import company_name
from docqa.engine import QAEngine

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).resolve().parent / "static"


def _user_payload(engine: QAEngine, email: str) -> dict:
    user = engine.access.authenticate(email)
    return {
        "email": user.email,
        "display_name": user.display_name,
        "companies": [company_name(c) for c in sorted(user.companies)],
        "uses_llm": engine.uses_llm,
        "uses_vectors": engine.uses_vectors,
    }


def create_app(engine: QAEngine | None = None) -> Flask:
    app = Flask(__name__, static_folder=str(STATIC_DIR), static_url_path="/static")
    app.secret_key = os.environ.get("DOCQA_SECRET_KEY") or secrets.token_hex(16)

    qa_engine = engine or QAEngine.bootstrap()
    app.config["QA_ENGINE"] = qa_engine

    # ---- UI ---------------------------------------------------------------

    @app.route("/", methods=["GET"])
    def index():
        return send_from_directory(STATIC_DIR, "index.html")

    # ---- Public config (for the login screen) -----------------------------

    @app.route("/api/config", methods=["GET"])
    def api_config():
        return jsonify(
            {
                "uses_llm": qa_engine.uses_llm,
                "uses_vectors": qa_engine.uses_vectors,
                "users": [
                    {
                        "email": u.email,
                        "display_name": u.display_name,
                        "companies": [company_name(c) for c in sorted(u.companies)],
                    }
                    for u in qa_engine.access.users
                ],
            }
        )

    # ---- Auth -------------------------------------------------------------

    @app.route("/api/login", methods=["POST"])
    def api_login():
        data = request.get_json(silent=True) or {}
        email = str(data.get("email", "")).strip().lower()
        password = str(data.get("password", ""))
        if not email or not password:
            return jsonify({"error": "Email and password are required."}), 400

        user = qa_engine.verify_login(email, password)
        if user is None:
            # Same message whether the email is unknown or the password is wrong,
            # so the endpoint does not reveal which accounts exist.
            return jsonify({"error": "Invalid email or password."}), 401

        session.clear()
        session["email"] = user.email
        session["display_name"] = user.display_name
        session["session_id"] = secrets.token_hex(8)
        logger.info("Login: %s", user.email)
        return jsonify(_user_payload(qa_engine, user.email))

    @app.route("/api/me", methods=["GET"])
    def api_me():
        email = session.get("email")
        if not email:
            return jsonify({"error": "not authenticated"}), 401
        return jsonify(_user_payload(qa_engine, email))

    @app.route("/api/documents", methods=["GET"])
    def api_documents():
        email = session.get("email")
        if not email:
            return jsonify({"error": "not authenticated"}), 401
        docs = qa_engine.documents_for(email)
        return jsonify(
            {
                "documents": [
                    {
                        "company": d.company_name,
                        "filename": d.filename,
                        "title": d.title,
                        "pages": d.num_pages,
                        "chunks": d.num_chunks,
                    }
                    for d in docs
                ]
            }
        )

    @app.route("/api/logout", methods=["POST"])
    def api_logout():
        sid = session.get("session_id")
        if sid:
            qa_engine.reset(sid)
        session.clear()
        return jsonify({"ok": True})

    @app.route("/api/reset", methods=["POST"])
    def api_reset():
        sid = session.get("session_id")
        if sid:
            qa_engine.reset(sid)
        return jsonify({"ok": True})

    # ---- Streaming Q&A ----------------------------------------------------

    @app.route("/api/ask", methods=["POST"])
    def api_ask():
        email = session.get("email")
        session_id = session.get("session_id")
        if not email or not session_id:
            return jsonify({"error": "not authenticated"}), 401

        data = request.get_json(silent=True) or {}
        question = str(data.get("question", "")).strip()
        if not question:
            return jsonify({"error": "empty question"}), 400

        def event_stream():
            try:
                for kind, payload in qa_engine.ask_stream(
                    email=email, session_id=session_id, question=question
                ):
                    if kind == "token":
                        yield _sse("token", {"text": payload})
                    else:  # final
                        answer = payload
                        yield _sse(
                            "done",
                            {
                                "citations": [
                                    {
                                        "label": c.label,
                                        "excerpt": c.excerpt,
                                        "score": c.score,
                                    }
                                    for c in answer.citations
                                ],
                                "used_llm": answer.used_llm,
                            },
                        )
            except AccessError as exc:
                yield _sse("error", {"message": str(exc)})
            except Exception as exc:  # pragma: no cover - surface errors to client
                logger.exception("Streaming error")
                yield _sse("error", {"message": f"Server error: {exc}"})

        headers = {
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # disable proxy buffering
            "Connection": "keep-alive",
        }
        return Response(stream_with_context(event_stream()), headers=headers)

    @app.route("/healthz", methods=["GET"])
    def healthz():
        return jsonify(
            {
                "status": "ok",
                "uses_llm": qa_engine.uses_llm,
                "uses_vectors": qa_engine.uses_vectors,
            }
        )

    return app


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    app = create_app()
    host = os.environ.get("DOCQA_HOST", "127.0.0.1")
    port = int(os.environ.get("DOCQA_PORT", "5000"))
    app.run(host=host, port=port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
