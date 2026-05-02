"""Tests for registration webhook processing and API endpoint."""

import tempfile
from pathlib import Path

from competition.registration.app import create_app
from competition.registration.webhook_receiver import process_registration_payload
from competition.storage import SubmissionStore


def _valid_registration_payload():
    return {
        "Team Name": "AI Avengers",
        "Primary contact name": "Alice",
        "Primary contact email": "alice@example.com",
        "Second contact name": "Bob",
        "Second contact email": "bob@example.com",
        "Agreement to rules": "I agree",
    }


def test_process_registration_payload_creates_team_and_token():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    try:
        store = SubmissionStore(db_path=db_path)
        payload = _valid_registration_payload()

        result = process_registration_payload(payload=payload, store=store)

        assert result["status"] == "success"
        assert result["registration_mode"] == "new"
        assert result["team_name"] == "AI Avengers"
        assert len(result["submission_token"]) == 64
        assert result["canonical_team_id"].startswith("ai_avengers_")

        team = store.get_team(result["canonical_team_id"])
        assert team is not None
        assert team.primary_email == "alice@example.com"
        assert store.verify_token(result["canonical_team_id"], result["submission_token"])
    finally:
        Path(db_path).unlink()


def test_process_registration_payload_same_team_same_email_is_idempotent():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    try:
        store = SubmissionStore(db_path=db_path)
        payload = _valid_registration_payload()

        first = process_registration_payload(payload=payload, store=store)
        second = process_registration_payload(payload=payload, store=store)

        assert first["status"] == "success"
        assert second["status"] == "success"
        assert second["registration_mode"] == "existing"
        assert first["canonical_team_id"] == second["canonical_team_id"]
        assert first["submission_token"] != second["submission_token"]

        assert store.verify_token(second["canonical_team_id"], second["submission_token"])
    finally:
        Path(db_path).unlink()


def test_process_registration_payload_rejects_name_conflict_with_other_email():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    try:
        store = SubmissionStore(db_path=db_path)

        payload_1 = _valid_registration_payload()
        payload_2 = _valid_registration_payload()
        payload_2["Primary contact email"] = "different@example.com"

        first = process_registration_payload(payload=payload_1, store=store)
        second = process_registration_payload(payload=payload_2, store=store)

        assert first["status"] == "success"
        assert second["status"] == "error"
        assert second["error_code"] == "TEAM_NAME_ALREADY_REGISTERED"
    finally:
        Path(db_path).unlink()


def test_process_registration_payload_rejects_if_rules_not_accepted():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    try:
        store = SubmissionStore(db_path=db_path)
        payload = _valid_registration_payload()
        payload["Agreement to rules"] = "No"

        result = process_registration_payload(payload=payload, store=store)

        assert result["status"] == "error"
        assert result["error_code"] == "RULES_NOT_ACCEPTED"
    finally:
        Path(db_path).unlink()


def test_registration_api_requires_bearer_token(monkeypatch):
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    try:
        monkeypatch.setenv("REGISTRATION_DB_PATH", db_path)
        monkeypatch.setenv("REGISTRATION_WEBHOOK_AUTH_TOKEN", "top-secret")

        app = create_app()
        client = app.test_client()

        response = client.post("/register", json=_valid_registration_payload())
        assert response.status_code == 401

        response = client.post(
            "/register",
            json=_valid_registration_payload(),
            headers={"Authorization": "Bearer top-secret"},
        )
        assert response.status_code == 200
        body = response.get_json()
        assert body["status"] == "success"
    finally:
        Path(db_path).unlink()
