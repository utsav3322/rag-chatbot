"""Tests for the Flask JSON + SSE API: auth, isolation, and streaming."""

from __future__ import annotations

import json

import pytest

from docqa.web.app import create_app


@pytest.fixture
def app(engine):
    app = create_app(engine=engine)
    app.config.update(TESTING=True)
    return app


@pytest.fixture
def client(app):
    return app.test_client()


def _collect_sse(response) -> list[dict]:
    """Parse an SSE streaming response into a list of {event, data} dicts."""

    text = response.get_data(as_text=True)
    events = []
    for block in text.split("\n\n"):
        event, data = None, None
        for line in block.splitlines():
            if line.startswith("event:"):
                event = line[6:].strip()
            elif line.startswith("data:"):
                data = line[5:].strip()
        if event and data:
            events.append({"event": event, "data": json.loads(data)})
    return events


def _stream_text(events) -> str:
    return "".join(e["data"]["text"] for e in events if e["event"] == "token")


def _done(events) -> dict:
    return next(e["data"] for e in events if e["event"] == "done")


def _login(client, email: str, password: str = "123456"):
    return client.post("/api/login", json={"email": email, "password": password})


@pytest.mark.integration
def test_index_served(client) -> None:
    resp = client.get("/")
    assert resp.status_code == 200
    assert b"DocQA" in resp.data


@pytest.mark.integration
def test_me_requires_auth(client) -> None:
    assert client.get("/api/me").status_code == 401


@pytest.mark.integration
def test_unknown_user_rejected(client) -> None:
    resp = _login(client, "mallory@email.com")
    assert resp.status_code == 401
    assert "Invalid email or password" in resp.get_json()["error"]


@pytest.mark.integration
def test_wrong_password_rejected(client) -> None:
    resp = _login(client, "alice@email.com", password="wrong")
    assert resp.status_code == 401
    assert "Invalid email or password" in resp.get_json()["error"]


@pytest.mark.integration
def test_missing_password_rejected(client) -> None:
    resp = client.post("/api/login", json={"email": "alice@email.com"})
    assert resp.status_code == 400


@pytest.mark.integration
def test_login_returns_scoped_companies(client) -> None:
    resp = _login(client, "alice@email.com")
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["companies"] == ["NovaTech Industries"]


@pytest.mark.integration
def test_documents_endpoint_is_access_scoped(client) -> None:
    _login(client, "alice@email.com")
    docs = client.get("/api/documents").get_json()["documents"]
    companies = {d["company"] for d in docs}
    assert companies == {"NovaTech Industries"}  # Alice sees only NovaTech


@pytest.mark.integration
def test_ask_streams_tokens_and_citations(client) -> None:
    _login(client, "alice@email.com")
    resp = client.post("/api/ask", json={"question": "What was the revenue in Q4?"})
    assert resp.status_code == 200
    assert resp.mimetype == "text/event-stream"
    events = _collect_sse(resp)
    assert _stream_text(events)  # some tokens streamed
    done = _done(events)
    assert done["citations"]
    assert all("NovaTech Industries" in c["label"] for c in done["citations"])


@pytest.mark.integration
def test_ask_isolation_no_leak(client) -> None:
    _login(client, "alice@email.com")
    resp = client.post("/api/ask", json={"question": "What was Helios solar revenue?"})
    done = _done(_collect_sse(resp))
    # Alice cannot access Helios -> no Helios citations, ever.
    assert all("Helios" not in c["label"] for c in done["citations"])


@pytest.mark.integration
def test_ask_requires_auth(client) -> None:
    assert client.post("/api/ask", json={"question": "hi"}).status_code == 401


@pytest.mark.integration
def test_two_clients_isolated(app) -> None:
    alice, bob = app.test_client(), app.test_client()
    _login(alice, "alice@email.com")
    _login(bob, "bob@email.com")

    a_me = alice.get("/api/me").get_json()
    b_me = bob.get("/api/me").get_json()
    assert a_me["email"] == "alice@email.com"
    assert b_me["email"] == "bob@email.com"
    assert a_me["companies"] != b_me["companies"]


@pytest.mark.integration
def test_greeting_streams_without_citations(client) -> None:
    _login(client, "bob@email.com")
    resp = client.post("/api/ask", json={"question": "hi"})
    events = _collect_sse(resp)
    assert _stream_text(events)
    assert _done(events)["citations"] == []


@pytest.mark.integration
def test_healthz(client) -> None:
    resp = client.get("/healthz")
    assert resp.status_code == 200
    assert resp.get_json()["status"] == "ok"
