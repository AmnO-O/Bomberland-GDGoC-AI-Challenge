"""Test token verification and team authentication in storage module."""

import sqlite3
import tempfile
from pathlib import Path

import pytest

from competition.storage import SubmissionStore, TeamRecord


class TestTokenHashingAndVerification:
    """Test suite for token hashing and verification logic."""

    def test_hash_token_returns_hex_string(self):
        """Token hash should be SHA256 hex string (64 chars)."""
        token = "my-secret-token"
        hashed = SubmissionStore.hash_token(token)
        
        assert isinstance(hashed, str)
        assert len(hashed) == 64  # SHA256 hex is 64 characters
        assert all(c in "0123456789abcdef" for c in hashed)

    def test_hash_token_deterministic(self):
        """Same token should always produce same hash."""
        token = "my-secret-token"
        hash1 = SubmissionStore.hash_token(token)
        hash2 = SubmissionStore.hash_token(token)
        
        assert hash1 == hash2

    def test_hash_token_different_tokens_different_hashes(self):
        """Different tokens should produce different hashes."""
        hash1 = SubmissionStore.hash_token("token-1")
        hash2 = SubmissionStore.hash_token("token-2")
        
        assert hash1 != hash2

    def test_token_never_stored_in_plaintext(self):
        """Plaintext token should never appear in database."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        
        try:
            store = SubmissionStore(db_path)
            plaintext_token = "super-secret-token-123"
            
            store.register_team(
                canonical_team_id="team-001",
                team_name="Test Team",
                primary_email="test@example.com",
                token=plaintext_token
            )
            
            # Check database doesn't contain plaintext token
            with sqlite3.connect(db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT submission_token_hash FROM teams WHERE canonical_team_id = ?", ("team-001",))
                stored_hash = cursor.fetchone()[0]
            
            assert plaintext_token not in stored_hash
            assert stored_hash == SubmissionStore.hash_token(plaintext_token)
        finally:
            Path(db_path).unlink()

    def test_verify_token_correct_token(self):
        """verify_token should return True for correct token."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        
        try:
            store = SubmissionStore(db_path)
            token = "correct-token"
            
            store.register_team(
                canonical_team_id="team-001",
                team_name="Test Team",
                primary_email="test@example.com",
                token=token
            )
            
            assert store.verify_token("team-001", token) is True
        finally:
            Path(db_path).unlink()

    def test_verify_token_incorrect_token(self):
        """verify_token should return False for incorrect token."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        
        try:
            store = SubmissionStore(db_path)
            store.register_team(
                canonical_team_id="team-001",
                team_name="Test Team",
                primary_email="test@example.com",
                token="correct-token"
            )
            
            assert store.verify_token("team-001", "wrong-token") is False
        finally:
            Path(db_path).unlink()

    def test_verify_token_nonexistent_team(self):
        """verify_token should return False if team doesn't exist."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        
        try:
            store = SubmissionStore(db_path)
            assert store.verify_token("nonexistent-team", "any-token") is False
        finally:
            Path(db_path).unlink()

    def test_register_and_retrieve_team(self):
        """Team registration should be retrievable via get_team()."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        
        try:
            store = SubmissionStore(db_path)
            store.register_team(
                canonical_team_id="team-alpha-123",
                team_name="Alpha Squadron",
                primary_email="alpha@example.com",
                token="secret-token"
            )
            
            team = store.get_team("team-alpha-123")
            assert team is not None
            assert isinstance(team, TeamRecord)
            assert team.canonical_team_id == "team-alpha-123"
            assert team.team_name == "Alpha Squadron"
            assert team.primary_email == "alpha@example.com"
            assert team.status == "active"
        finally:
            Path(db_path).unlink()

    def test_get_team_nonexistent(self):
        """get_team should return None for nonexistent team."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        
        try:
            store = SubmissionStore(db_path)
            team = store.get_team("nonexistent-team")
            assert team is None
        finally:
            Path(db_path).unlink()

    def test_upsert_team_updates_existing(self):
        """Re-registering a team should update its information."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        
        try:
            store = SubmissionStore(db_path)
            
            # Register team
            store.register_team(
                canonical_team_id="team-001",
                team_name="Old Name",
                primary_email="old@example.com",
                token="token-1"
            )
            
            # Update team (upsert)
            store.register_team(
                canonical_team_id="team-001",
                team_name="New Name",
                primary_email="new@example.com",
                token="token-2"
            )
            
            # Verify update
            team = store.get_team("team-001")
            assert team.team_name == "New Name"
            assert team.primary_email == "new@example.com"
            assert store.verify_token("team-001", "token-2") is True
            assert store.verify_token("team-001", "token-1") is False
        finally:
            Path(db_path).unlink()

    def test_team_uniqueness_by_id(self):
        """canonical_team_id should be unique (PRIMARY KEY)."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        
        try:
            store = SubmissionStore(db_path)
            
            store.register_team(
                canonical_team_id="team-001",
                team_name="Team A",
                primary_email="a@example.com",
                token="token-a"
            )
            
            # Upsert should work (replaces existing)
            store.register_team(
                canonical_team_id="team-001",
                team_name="Team A Updated",
                primary_email="a-updated@example.com",
                token="token-a-updated"
            )
            
            # Count should still be 1
            with sqlite3.connect(db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM teams WHERE canonical_team_id = ?", ("team-001",))
                count = cursor.fetchone()[0]
            
            assert count == 1
        finally:
            Path(db_path).unlink()

    def test_multiple_teams_isolated(self):
        """Multiple teams should have independent tokens."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        
        try:
            store = SubmissionStore(db_path)
            
            store.register_team("team-001", "Team 1", "t1@ex.com", "token-1")
            store.register_team("team-002", "Team 2", "t2@ex.com", "token-2")
            
            assert store.verify_token("team-001", "token-1") is True
            assert store.verify_token("team-001", "token-2") is False
            assert store.verify_token("team-002", "token-2") is True
            assert store.verify_token("team-002", "token-1") is False
        finally:
            Path(db_path).unlink()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
