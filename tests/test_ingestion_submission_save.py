"""Test end-to-end submission save flow and database persistence."""

import io
import json
import sqlite3
import tempfile
import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from competition.ingestion import (
    extract_zip_bytes,
    process_submission_item,
    validate_zip_bytes,
)
from competition.storage import SubmissionStore


class TestSubmissionSaveFlow:
    """Test suite for end-to-end submission intake and persistence."""

    def _create_zip(self, files: dict) -> bytes:
        """Helper: create a zip from dict of {filename: content}."""
        bio = io.BytesIO()
        with zipfile.ZipFile(bio, "w", zipfile.ZIP_DEFLATED) as zf:
            for name, content in files.items():
                if isinstance(content, str):
                    content = content.encode("utf-8")
                zf.writestr(name, content)
        return bio.getvalue()

    def test_extract_zip_to_disk(self):
        """Extracted files should be saved to immutable storage path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            files = {
                "agent.py": "def act(obs): return 0",
                "weights.pt": b"fake_weights",
                "config/model.yaml": "learning_rate: 0.1",
            }
            zip_bytes = self._create_zip(files)
            
            # Validate to get manifest
            is_valid, _, manifest = validate_zip_bytes(zip_bytes)
            assert is_valid
            
            # Extract to target
            target_dir = Path(tmpdir) / "team-001" / "submission-uuid"
            extract_zip_bytes(zip_bytes, target_dir, manifest)
            
            # Verify files exist
            assert (target_dir / "agent.py").exists()
            assert (target_dir / "weights.pt").exists()
            assert (target_dir / "config" / "model.yaml").exists()
            
            # Verify content
            with open(target_dir / "agent.py") as f:
                assert "def act" in f.read()

    def test_extraction_creates_immutable_path_structure(self):
        """Extracted files should be in submissions/{team_id}/{submission_id}/ path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            files = {"agent.py": "def act(obs): return 0"}
            zip_bytes = self._create_zip(files)
            
            is_valid, _, manifest = validate_zip_bytes(zip_bytes)
            assert is_valid
            
            team_id = "team-alpha-123"
            submission_id = "a1b2c3d4-e5f6-47g8-h9i0-j1k2l3m4n5o6"
            target_dir = Path(tmpdir) / "submissions" / team_id / submission_id
            
            extract_zip_bytes(zip_bytes, target_dir, manifest)
            
            # Verify path structure
            assert target_dir.exists()
            assert (target_dir / "agent.py").exists()
            expected_path = f"submissions/{team_id}/{submission_id}"
            assert expected_path in str(target_dir)

    def test_save_submission_persists_to_database(self):
        """Saving a submission should create a database record."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        
        try:
            store = SubmissionStore(db_path)
            
            # Register team first
            store.register_team(
                canonical_team_id="team-001",
                team_name="Test Team",
                primary_email="test@example.com",
                token="token"
            )
            
            # Save a submission
            store.save_submission(
                submission_id="sub-001",
                canonical_team_id="team-001",
                response_id="response-001",
                drive_file_id="drive-file-001",
                original_filename="agent.zip",
                sha256="deadbeef",
                uploaded_at="2026-04-23T10:00:00Z",
                validation_status="valid",
                validation_reason=None,
                extracted_path="submissions/team-001/sub-001",
                extracted_manifest_json='{"agent.py": 1024}'
            )
            
            # Query database to verify
            with sqlite3.connect(db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT submission_id, canonical_team_id, validation_status FROM submissions WHERE submission_id = ?",
                    ("sub-001",)
                )
                row = cursor.fetchone()
            
            assert row is not None
            assert row[0] == "sub-001"
            assert row[1] == "team-001"
            assert row[2] == "valid"
        finally:
            Path(db_path).unlink()

    def test_save_invalid_submission_records_reason(self):
        """Invalid submissions should record validation reason."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        
        try:
            store = SubmissionStore(db_path)
            store.register_team("team-001", "Test Team", "test@example.com", "token")
            
            store.save_submission(
                submission_id="sub-002",
                canonical_team_id="team-001",
                response_id="response-002",
                drive_file_id="drive-file-002",
                original_filename="invalid.zip",
                sha256="cafebabe",
                uploaded_at="2026-04-23T11:00:00Z",
                validation_status="invalid",
                validation_reason="agent_py_missing_or_multiple",
                extracted_path=None,
                extracted_manifest_json=None
            )
            
            with sqlite3.connect(db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT validation_status, validation_reason FROM submissions WHERE submission_id = ?",
                    ("sub-002",)
                )
                row = cursor.fetchone()
            
            assert row[0] == "invalid"
            assert row[1] == "agent_py_missing_or_multiple"
        finally:
            Path(db_path).unlink()

    def test_duplicate_response_id_prevented(self):
        """Submitting same response_id twice should be detected as duplicate."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        
        try:
            store = SubmissionStore(db_path)
            store.register_team("team-001", "Test Team", "test@example.com", "token")
            
            # First submission
            store.save_submission(
                submission_id="sub-001",
                canonical_team_id="team-001",
                response_id="response-100",
                drive_file_id="drive-file-100",
                original_filename="agent.zip",
                sha256="abc123",
                uploaded_at="2026-04-23T10:00:00Z",
                validation_status="valid",
                validation_reason=None,
                extracted_path="submissions/team-001/sub-001",
                extracted_manifest_json='{}'
            )
            
            # Check first exists
            assert store.has_processed_response("response-100")
            
            # Second submission with same response_id should violate UNIQUE constraint
            with pytest.raises(sqlite3.IntegrityError):
                store.save_submission(
                    submission_id="sub-002",
                    canonical_team_id="team-001",
                    response_id="response-100",  # Same response_id
                    drive_file_id="drive-file-101",
                    original_filename="agent_v2.zip",
                    sha256="def456",
                    uploaded_at="2026-04-23T11:00:00Z",
                    validation_status="valid",
                    validation_reason=None,
                    extracted_path="submissions/team-001/sub-002",
                    extracted_manifest_json='{}'
                )
        finally:
            Path(db_path).unlink()

    def test_submission_manifest_stored_as_json(self):
        """Extracted manifest should be stored and retrievable as JSON."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        
        try:
            store = SubmissionStore(db_path)
            store.register_team("team-001", "Test Team", "test@example.com", "token")
            
            manifest_dict = {
                "agent.py": 1024,
                "weights.pt": 512000,
                "config/model.yaml": 256
            }
            manifest_json = json.dumps(manifest_dict, sort_keys=True)
            
            store.save_submission(
                submission_id="sub-003",
                canonical_team_id="team-001",
                response_id="response-300",
                drive_file_id="drive-file-300",
                original_filename="agent.zip",
                sha256="deadbeef",
                uploaded_at="2026-04-23T12:00:00Z",
                validation_status="valid",
                validation_reason=None,
                extracted_path="submissions/team-001/sub-003",
                extracted_manifest_json=manifest_json
            )
            
            # Retrieve and verify
            with sqlite3.connect(db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT extracted_manifest_json FROM submissions WHERE submission_id = ?",
                    ("sub-003",)
                )
                row = cursor.fetchone()
            
            stored_manifest = json.loads(row[0])
            assert stored_manifest == manifest_dict
        finally:
            Path(db_path).unlink()

    @patch("competition.ingestion.collector.download_drive_file_bytes")
    def test_process_submission_item_end_to_end_valid(self, mock_download):
        """End-to-end: process_submission_item should handle valid submission."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
                db_path = f.name
            
            try:
                # Setup
                store = SubmissionStore(db_path)
                store.register_team("team-001", "Test Team", "test@example.com", "secret-token")
                
                # Mock download
                zip_bytes = self._create_zip({"agent.py": "def act(obs): return 0"})
                mock_download.return_value = zip_bytes
                
                mock_service = MagicMock()
                
                # Process
                item = {
                    "drive_file_id": "drive-001",
                    "canonical_team_id": "team-001",
                    "submission_token": "secret-token",
                    "original_filename": "agent.zip"
                }
                
                ok, note = process_submission_item(
                    mock_service,
                    store,
                    str(Path(tmpdir) / "submissions"),
                    item
                )
                
                # Verify
                assert ok is True
                assert "stored:" in note
                
                # Check database
                with sqlite3.connect(db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute("SELECT validation_status FROM submissions")
                    status = cursor.fetchone()[0]
                
                assert status == "valid"
            finally:
                Path(db_path).unlink()

    def test_process_submission_item_auth_failed(self):
        """process_submission_item should reject bad token."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
                db_path = f.name
            
            try:
                store = SubmissionStore(db_path)
                store.register_team("team-001", "Test Team", "test@example.com", "correct-token")
                
                mock_service = MagicMock()
                
                item = {
                    "drive_file_id": "drive-002",
                    "canonical_team_id": "team-001",
                    "submission_token": "wrong-token",  # Wrong token
                    "original_filename": "agent.zip"
                }
                
                ok, note = process_submission_item(
                    mock_service,
                    store,
                    str(Path(tmpdir) / "submissions"),
                    item
                )
                
                assert ok is False
                assert "auth_failed" in note
            finally:
                Path(db_path).unlink()

    @patch("competition.ingestion.collector.download_drive_file_bytes")
    def test_process_submission_item_invalid_zip(self, mock_download):
        """process_submission_item should reject invalid zips."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
                db_path = f.name
            
            try:
                store = SubmissionStore(db_path)
                store.register_team("team-001", "Test Team", "test@example.com", "token")
                
                # Invalid zip: no agent.py
                invalid_zip = self._create_zip({"config.yaml": "test"})
                mock_download.return_value = invalid_zip
                
                mock_service = MagicMock()
                
                item = {
                    "drive_file_id": "drive-003",
                    "canonical_team_id": "team-001",
                    "submission_token": "token",
                    "original_filename": "bad.zip"
                }
                
                ok, note = process_submission_item(
                    mock_service,
                    store,
                    str(Path(tmpdir) / "submissions"),
                    item
                )
                
                assert ok is False
                assert "agent_py_missing" in note
                
                # Check database recorded invalid status
                with sqlite3.connect(db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute("SELECT validation_status FROM submissions")
                    status = cursor.fetchone()[0]
                
                assert status == "invalid"
            finally:
                Path(db_path).unlink()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
