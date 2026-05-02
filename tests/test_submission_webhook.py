"""Tests for submission webhook processing with quota enforcement and timezone handling."""

import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import Mock, patch, MagicMock

from competition.ingestion.submission_webhook import (
    process_submission_webhook,
    get_vietnam_day_identifier,
    MAX_SUBMISSIONS_PER_DAY,
    VIETNAM_TZ,
)
from competition.storage import SubmissionStore


class TestGetVietnamDayIdentifier:
    """Test Vietnam timezone day boundary logic (7 AM reset)."""

    def test_7am_boundary_morning_before_reset(self):
        """Before 7 AM = previous day."""
        # 2026-04-25 06:59:59 UTC+7 = 2026-04-24 23:59:59 UTC
        dt_utc = datetime(2026, 4, 24, 23, 59, 59, tzinfo=timezone.utc)
        day_id = get_vietnam_day_identifier(dt_utc)
        assert day_id == "2026-04-24", f"Before 7 AM should be previous day, got {day_id}"

    def test_7am_boundary_exact_reset(self):
        """Exactly 7 AM = day boundary."""
        # 2026-04-25 07:00:00 UTC+7 = 2026-04-25 00:00:00 UTC
        dt_utc = datetime(2026, 4, 25, 0, 0, 0, tzinfo=timezone.utc)
        day_id = get_vietnam_day_identifier(dt_utc)
        assert day_id == "2026-04-25", f"At 7 AM should be current day, got {day_id}"

    def test_7am_boundary_after_reset(self):
        """After 7 AM = current day."""
        # 2026-04-25 07:00:01 UTC+7 = 2026-04-25 00:00:01 UTC
        dt_utc = datetime(2026, 4, 25, 0, 0, 1, tzinfo=timezone.utc)
        day_id = get_vietnam_day_identifier(dt_utc)
        assert day_id == "2026-04-25", f"After 7 AM should be current day, got {day_id}"

    def test_noon_vietnam_time(self):
        """Noon Vietnam time = current day."""
        # 2026-04-25 12:00:00 UTC+7 = 2026-04-25 05:00:00 UTC
        dt_utc = datetime(2026, 4, 25, 5, 0, 0, tzinfo=timezone.utc)
        day_id = get_vietnam_day_identifier(dt_utc)
        assert day_id == "2026-04-25"

    def test_midnight_vietnam_time(self):
        """Midnight Vietnam time (00:00-06:59) = previous day (before 7 AM reset)."""
        # 2026-04-25 00:30:00 UTC+7 = 2026-04-24 17:30:00 UTC
        dt_utc = datetime(2026, 4, 24, 17, 30, 0, tzinfo=timezone.utc)
        day_id = get_vietnam_day_identifier(dt_utc)
        assert day_id == "2026-04-24"

    def test_evening_vietnam_time(self):
        """Evening Vietnam time = current day."""
        # 2026-04-25 22:00:00 UTC+7 = 2026-04-25 15:00:00 UTC
        dt_utc = datetime(2026, 4, 25, 15, 0, 0, tzinfo=timezone.utc)
        day_id = get_vietnam_day_identifier(dt_utc)
        assert day_id == "2026-04-25"

    def test_no_argument_uses_current_time(self):
        """Calling without argument uses current time."""
        day_id = get_vietnam_day_identifier()
        # Should return a valid YYYY-MM-DD format
        assert len(day_id) == 10
        assert day_id.count("-") == 2
        parts = day_id.split("-")
        assert len(parts[0]) == 4  # Year
        assert len(parts[1]) == 2  # Month
        assert len(parts[2]) == 2  # Day


class TestSubmissionWebhookValidation:
    """Test submission webhook field validation."""

    def test_missing_canonical_team_id(self):
        """Should reject missing canonical_team_id."""
        store = Mock(spec=SubmissionStore)
        service = Mock()
        
        payload = {
            "submission_token": "token123",
            "drive_file_id": "file_id_123",
            "changelog": "Test",
        }
        
        ok, result = process_submission_webhook(payload, store, service)
        assert ok is False
        assert result["error"] == "missing_field"
        assert result["reason"] == "canonical_team_id"

    def test_missing_submission_token(self):
        """Should reject missing submission_token."""
        store = Mock(spec=SubmissionStore)
        service = Mock()
        
        payload = {
            "canonical_team_id": "test_team_123",
            "drive_file_id": "file_id_123",
            "changelog": "Test",
        }
        
        ok, result = process_submission_webhook(payload, store, service)
        assert ok is False
        assert result["error"] == "missing_field"
        assert result["reason"] == "submission_token"

    def test_missing_drive_file_id(self):
        """Should reject missing drive_file_id."""
        store = Mock(spec=SubmissionStore)
        service = Mock()
        
        payload = {
            "canonical_team_id": "test_team_123",
            "submission_token": "token123",
            "changelog": "Test",
        }
        
        ok, result = process_submission_webhook(payload, store, service)
        assert ok is False
        assert result["error"] == "missing_field"
        assert result["reason"] == "drive_file_id"


class TestSubmissionWebhookAuthentication:
    """Test token verification and team validation."""

    def test_unknown_team(self):
        """Should reject unknown team."""
        store = Mock(spec=SubmissionStore)
        store.get_team.return_value = None
        service = Mock()
        
        payload = {
            "canonical_team_id": "unknown_team",
            "submission_token": "token123",
            "drive_file_id": "file_id_123",
            "changelog": "Test",
        }
        
        ok, result = process_submission_webhook(payload, store, service)
        assert ok is False
        assert result["error"] == "auth_failed"
        assert result["reason"] == "unknown_team"
        store.get_team.assert_called_once_with("unknown_team")

    def test_team_not_active(self):
        """Should reject inactive team."""
        from competition.storage import TeamRecord
        
        store = Mock(spec=SubmissionStore)
        store.get_team.return_value = TeamRecord(
            canonical_team_id="test_team_123",
            team_name="Test Team",
            primary_email="test@example.com",
            status="suspended"
        )
        service = Mock()
        
        payload = {
            "canonical_team_id": "test_team_123",
            "submission_token": "token123",
            "drive_file_id": "file_id_123",
            "changelog": "Test",
        }
        
        ok, result = process_submission_webhook(payload, store, service)
        assert ok is False
        assert result["error"] == "auth_failed"
        assert "team_not_active" in result["reason"]

    def test_token_mismatch(self):
        """Should reject mismatched token."""
        from competition.storage import TeamRecord
        
        store = Mock(spec=SubmissionStore)
        store.get_team.return_value = TeamRecord(
            canonical_team_id="test_team_123",
            team_name="Test Team",
            primary_email="test@example.com",
            status="active"
        )
        store.verify_token.return_value = False
        service = Mock()
        
        payload = {
            "canonical_team_id": "test_team_123",
            "submission_token": "wrong_token",
            "drive_file_id": "file_id_123",
            "changelog": "Test",
        }
        
        ok, result = process_submission_webhook(payload, store, service)
        assert ok is False
        assert result["error"] == "auth_failed"
        assert result["reason"] == "token_mismatch"


class TestSubmissionWebhookQuota:
    """Test daily submission quota enforcement."""

    def test_quota_not_exceeded(self):
        """Should accept submission within quota."""
        from competition.storage import TeamRecord
        
        store = Mock(spec=SubmissionStore)
        store.get_team.return_value = TeamRecord(
            canonical_team_id="test_team_123",
            team_name="Test Team",
            primary_email="test@example.com",
            status="active"
        )
        store.verify_token.return_value = True
        store.get_daily_quota_count.return_value = 1  # Already 1 submission today
        
        service = Mock()
        
        # Mock the collector call
        with patch("competition.ingestion.submission_webhook.process_submission_item") as mock_collector:
            mock_collector.return_value = (True, "stored:/path/to/submission")
            
            payload = {
                "canonical_team_id": "test_team_123",
                "submission_token": "token123",
                "drive_file_id": "file_id_123",
                "changelog": "Test submission",
            }
            
            ok, result = process_submission_webhook(payload, store, service)
            
            assert ok is True
            assert result["status"] == "success"
            assert result["remaining_today"] == 1  # 3 - (1 + 1) = 1

    def test_quota_exceeded_at_limit(self):
        """Should reject submission when quota is exhausted."""
        from competition.storage import TeamRecord
        
        store = Mock(spec=SubmissionStore)
        store.get_team.return_value = TeamRecord(
            canonical_team_id="test_team_123",
            team_name="Test Team",
            primary_email="test@example.com",
            status="active"
        )
        store.verify_token.return_value = True
        store.get_daily_quota_count.return_value = 3  # Already at max
        service = Mock()
        
        payload = {
            "canonical_team_id": "test_team_123",
            "submission_token": "token123",
            "drive_file_id": "file_id_123",
            "changelog": "Test submission",
        }
        
        ok, result = process_submission_webhook(payload, store, service)
        
        assert ok is False
        assert result["error"] == "quota_exceeded"
        assert "max 3 submissions per day" in result["reason"]
        store.get_daily_quota_count.assert_called_once()
        store.increment_daily_quota.assert_not_called()

    def test_quota_resets_by_day(self):
        """Should count quota per day based on Vietnam timezone."""
        from competition.storage import TeamRecord
        
        store = Mock(spec=SubmissionStore)
        store.get_team.return_value = TeamRecord(
            canonical_team_id="test_team_123",
            team_name="Test Team",
            primary_email="test@example.com",
            status="active"
        )
        store.verify_token.return_value = True
        store.get_daily_quota_count.return_value = 0
        
        service = Mock()
        
        with patch("competition.ingestion.submission_webhook.process_submission_item") as mock_collector:
            mock_collector.return_value = (True, "stored:/path/to/submission")
            
            # Mock time to be at 8 AM Vietnam time
            with patch("competition.ingestion.submission_webhook.get_vietnam_day_identifier") as mock_day:
                mock_day.return_value = "2026-04-25"
                
                payload = {
                    "canonical_team_id": "test_team_123",
                    "submission_token": "token123",
                    "drive_file_id": "file_id_123",
                    "changelog": "Test submission",
                }
                
                ok, result = process_submission_webhook(payload, store, service)
                
                assert ok is True
                store.increment_daily_quota.assert_called_once_with("test_team_123", "2026-04-25")


class TestSubmissionWebhookCollection:
    """Test submission intake and validation."""

    def test_successful_submission_processing(self):
        """Should successfully process valid submission."""
        from competition.storage import TeamRecord
        
        store = Mock(spec=SubmissionStore)
        store.get_team.return_value = TeamRecord(
            canonical_team_id="test_team_123",
            team_name="Test Team",
            primary_email="test@example.com",
            status="active"
        )
        store.verify_token.return_value = True
        store.get_daily_quota_count.return_value = 0
        
        service = Mock()
        
        with patch("competition.ingestion.submission_webhook.process_submission_item") as mock_collector:
            mock_collector.return_value = (True, "stored:/submissions/test_team_123/abc123/")
            
            payload = {
                "canonical_team_id": "test_team_123",
                "submission_token": "token123",
                "drive_file_id": "file_id_123",
                "changelog": "My agent beats random baseline",
                "original_filename": "submission.zip",
            }
            
            ok, result = process_submission_webhook(payload, store, service)
            
            assert ok is True
            assert result["status"] == "success"
            assert "submission_id" in result
            assert result["remaining_today"] == 2  # 3 - 1
            
            # Verify collector was called with correct parameters
            mock_collector.assert_called_once()
            call_args = mock_collector.call_args
            assert call_args[1]["item"]["canonical_team_id"] == "test_team_123"
            assert call_args[1]["item"]["drive_file_id"] == "file_id_123"

    def test_submission_validation_failure(self):
        """Should handle validation failure from collector."""
        from competition.storage import TeamRecord
        
        store = Mock(spec=SubmissionStore)
        store.get_team.return_value = TeamRecord(
            canonical_team_id="test_team_123",
            team_name="Test Team",
            primary_email="test@example.com",
            status="active"
        )
        store.verify_token.return_value = True
        store.get_daily_quota_count.return_value = 0
        
        service = Mock()
        
        with patch("competition.ingestion.submission_webhook.process_submission_item") as mock_collector:
            mock_collector.return_value = (False, "agent_py_missing_or_multiple")
            
            payload = {
                "canonical_team_id": "test_team_123",
                "submission_token": "token123",
                "drive_file_id": "file_id_123",
                "changelog": "Test",
            }
            
            ok, result = process_submission_webhook(payload, store, service)
            
            assert ok is False
            assert result["error"] == "validation_failed"
            assert "agent_py_missing_or_multiple" in result["reason"]
            # Should NOT increment quota if validation failed
            store.increment_daily_quota.assert_not_called()


class TestSubmissionWebhookIntegration:
    """Integration tests combining multiple components."""

    def test_end_to_end_successful_submission(self):
        """Full flow: valid team, valid token, within quota, successful validation."""
        from competition.storage import TeamRecord
        
        store = Mock(spec=SubmissionStore)
        store.get_team.return_value = TeamRecord(
            canonical_team_id="awesome_ai_a1b2c3d4",
            team_name="Awesome AI",
            primary_email="team@university.edu",
            status="active"
        )
        store.verify_token.return_value = True
        store.get_daily_quota_count.return_value = 0
        
        service = Mock()
        
        with patch("competition.ingestion.submission_webhook.process_submission_item") as mock_collector:
            mock_collector.return_value = (
                True,
                "stored:/submissions/awesome_ai_a1b2c3d4/sub_uuid/"
            )
            
            payload = {
                "canonical_team_id": "awesome_ai_a1b2c3d4",
                "submission_token": "token_xyz_123_longtoken",
                "drive_file_id": "1ABC_file_id_123_XYZ",
                "changelog": "Improved agent with better heuristics",
                "original_filename": "submission_v2.zip",
            }
            
            ok, result = process_submission_webhook(payload, store, service, storage_dir="submissions")
            
            assert ok is True
            assert result["status"] == "success"
            assert result["remaining_today"] == 2
            store.increment_daily_quota.assert_called_once()

    def test_rejected_submission_chain(self):
        """Multiple rejection scenarios in sequence."""
        store = Mock(spec=SubmissionStore)
        service = Mock()
        
        # Scenario 1: Missing field
        payload_missing = {"canonical_team_id": "test"}
        ok, result = process_submission_webhook(payload_missing, store, service)
        assert ok is False and result["error"] == "missing_field"
        
        # Scenario 2: Unknown team
        store.get_team.return_value = None
        payload_unknown = {
            "canonical_team_id": "unknown",
            "submission_token": "token",
            "drive_file_id": "file",
        }
        ok, result = process_submission_webhook(payload_unknown, store, service)
        assert ok is False and result["error"] == "auth_failed"
        
        # Scenario 3: Token mismatch
        from competition.storage import TeamRecord
        store.get_team.return_value = TeamRecord(
            canonical_team_id="test",
            team_name="Test",
            primary_email="test@example.com",
            status="active"
        )
        store.verify_token.return_value = False
        ok, result = process_submission_webhook(payload_unknown, store, service)
        assert ok is False and result["reason"] == "token_mismatch"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
