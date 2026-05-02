# Storage Module: Team and Submission Persistence

## Purpose

The **storage module** manages all database operations for the Phase 1 submission intake system. It provides a clean API for:
- **Team registration** with secure submission token handling (SHA256 hashing)
- **Token verification** for authentication during submission collection
- **Submission persistence** with validation status, extracted paths, and manifests
- **Daily quota tracking** to enforce the 3-submission-per-team-per-day limit

This module is the single source of truth for database schema and team/submission data, used by `competition.ingestion` to record intake outcomes.

## Key Classes and Functions

### `SubmissionStore` (Main Class)

**Purpose**: Encapsulates all database operations with automatic table initialization.

**Constructor**:
```python
store = SubmissionStore(db_path="/path/to/competition.db")
```
- If `db_path` is omitted, defaults to `{workspace_root}/competition.db`
- Constructor automatically calls `_init_db()` to create tables if missing

**Core Methods**:

#### `hash_token(token: str) -> str` (Static)
Hash a submission token using SHA256 for secure storage.
```python
token_hash = SubmissionStore.hash_token("my-secret-token")
# Returns: e.g., "a3c5f7d9e8b1c2e4f6a8c9d1e2f3a4b5c6d7e8f9"
```
- Called during team registration and token verification
- Ensures tokens are never stored in plaintext in the database

#### `register_team(canonical_team_id, team_name, primary_email, token)`
Register a new team or update an existing team's info.
```python
store.register_team(
    canonical_team_id="team-001",
    team_name="Alpha Squadron",
    primary_email="alpha@example.com",
    token="super-secret-submission-token"
)
```
- Token is automatically hashed before storage
- `canonical_team_id` must be unique (PRIMARY KEY)
- `team_name` must be unique (enforced by DB)
- Status defaults to "active"; teams start eligible for submissions

#### `get_team(canonical_team_id) -> Optional[TeamRecord]`
Retrieve team metadata by canonical ID.
```python
team = store.get_team("team-001")
if team:
    print(f"Team: {team.team_name}, Status: {team.status}, Email: {team.primary_email}")
else:
    print("Team not found")
```
- Returns `TeamRecord` dataclass with fields: `canonical_team_id`, `team_name`, `primary_email`, `status`
- Returns `None` if team doesn't exist

#### `verify_token(canonical_team_id, token) -> bool`
Verify a submission token matches the stored hash for a team.
```python
is_valid = store.verify_token("team-001", submitted_token)
if is_valid:
    print("Token matches!")
else:
    print("Invalid token")
```
- Called during authentication in `competition.ingestion.authenticate_submission()`
- Returns True only if token hash matches; False otherwise

#### `has_processed_response(response_id) -> bool`
Check if a submission (by response_id) has already been processed.
```python
if store.has_processed_response("drive-file-12345"):
    print("Already processed, skipping")
else:
    print("New submission, proceeding")
```
- Used to prevent duplicate processing of the same Drive file
- `response_id` = Google Drive file ID in Phase 1 intake

#### `save_submission(submission_id, canonical_team_id, response_id, drive_file_id, ...)`
Record a submission in the database with validation outcome.
```python
store.save_submission(
    submission_id="uuid-1234-5678",
    canonical_team_id="team-001",
    response_id="drive-file-12345",
    drive_file_id="drive-file-12345",
    original_filename="agent.zip",
    sha256="a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6",
    uploaded_at="2026-04-23T15:30:00Z",
    validation_status="valid",
    validation_reason=None,
    extracted_path="submissions/team-001/uuid-1234-5678",
    extracted_manifest_json='{"agent.py": 1024, "weights.pt": 512000}'
)
```
- `submission_id`: unique UUID for this submission record
- `response_id`: typically same as `drive_file_id` in Phase 1 (Google Form response row ID)
- `validation_status`: one of "valid", "invalid"
- `validation_reason`: error detail if invalid (e.g., "zip_too_large", "agent_py_missing_or_multiple")
- `extracted_manifest_json`: JSON string mapping extracted filenames to file sizes

#### `increment_daily_quota(canonical_team_id, day_utc)`
Increment the daily submission count for a team.
```python
store.increment_daily_quota("team-001", "2026-04-23")
```
- Called after a successful submission to track daily quota
- Uses upsert logic: creates entry if first submission of day, increments if exists

#### `get_daily_quota_count(canonical_team_id, day_utc) -> int`
Get the current submission count for a team on a given day.
```python
count = store.get_daily_quota_count("team-001", "2026-04-23")
if count >= 3:
    print("Daily quota exhausted")
```
- Returns 0 if no submissions recorded for that team/day
- Used before incrementing to enforce max 3 submissions per day

## Database Schema

### `teams` Table
```sql
CREATE TABLE teams (
    canonical_team_id TEXT PRIMARY KEY,
    team_name TEXT NOT NULL UNIQUE,
    primary_email TEXT NOT NULL,
    submission_token_hash TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'active',
    created_at TEXT NOT NULL
)
```
- `canonical_team_id`: Immutable team identifier (issued by organizer)
- `team_name`: Unique display name for leaderboard
- `primary_email`: Contact for organizer notifications
- `submission_token_hash`: SHA256(token) — **never store plaintext token**
- `status`: "active" or other future states (e.g., "suspended")

### `submissions` Table
```sql
CREATE TABLE submissions (
    submission_id TEXT PRIMARY KEY,
    canonical_team_id TEXT NOT NULL,
    response_id TEXT NOT NULL UNIQUE,
    drive_file_id TEXT NOT NULL,
    original_filename TEXT,
    sha256 TEXT,
    uploaded_at TEXT,
    created_at TEXT NOT NULL,
    validation_status TEXT NOT NULL,
    validation_reason TEXT,
    extracted_path TEXT,
    extracted_manifest_json TEXT,
    FOREIGN KEY (canonical_team_id) REFERENCES teams(canonical_team_id)
)
```
- `submission_id`: UUID for this submission record
- `response_id`: Unique identifier from source (e.g., Google Form response row ID) — **UNIQUE constraint prevents duplicates**
- `drive_file_id`: Google Drive file ID of the uploaded zip
- `validation_status`: "valid" or "invalid"
- `extracted_path`: Where files were extracted (e.g., "submissions/team-001/uuid/")
- `extracted_manifest_json`: JSON map of files and their sizes

### `daily_submission_quota` Table
```sql
CREATE TABLE daily_submission_quota (
    canonical_team_id TEXT NOT NULL,
    day_utc TEXT NOT NULL,
    submission_count INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (canonical_team_id, day_utc),
    FOREIGN KEY (canonical_team_id) REFERENCES teams(canonical_team_id)
)
```
- Composite key ensures one row per team per day
- `day_utc`: ISO date string (e.g., "2026-04-23")
- Used to enforce max 3 submissions per team per day

## Usage Example: Complete Workflow

```python
from competition.storage import SubmissionStore

# Initialize or open database
store = SubmissionStore("/workspace/competition.db")

# Register a team (one-time, or upsert to update)
store.register_team(
    canonical_team_id="team-alpha-123",
    team_name="Alpha Squadron",
    primary_email="alpha@example.com",
    token="very-secret-reusable-token"
)

# Verify a submission token
team_id = "team-alpha-123"
submitted_token = "very-secret-reusable-token"
if store.verify_token(team_id, submitted_token):
    print("✓ Token valid, proceeding")
    
    # Record the successful submission
    store.save_submission(
        submission_id="12345678-1234-1234-1234-123456789012",
        canonical_team_id=team_id,
        response_id="google-form-response-999",
        drive_file_id="drive-file-999",
        original_filename="agent_v2.zip",
        sha256="deadbeefdeadbeefdeadbeefdeadbeef",
        uploaded_at="2026-04-23T14:30:00Z",
        validation_status="valid",
        validation_reason=None,
        extracted_path="submissions/team-alpha-123/12345678-1234-1234-1234-123456789012",
        extracted_manifest_json='{"agent.py": 2048, "model.pt": 1000000}'
    )
    
    # Increment quota
    store.increment_daily_quota(team_id, "2026-04-23")
    
    # Check quota for next submission
    count = store.get_daily_quota_count(team_id, "2026-04-23")
    print(f"Submissions today: {count} / 3")
else:
    print("✗ Token mismatch")
```

## How competition.ingestion Uses This Module

The `competition.ingestion.collector` module calls storage functions:

1. **`authenticate_submission(store, canonical_team_id, token)`**
   - Calls `store.get_team()` to check team exists and is active
   - Calls `store.verify_token()` to validate token

2. **`process_submission_item(service, store, storage_dir, item)`**
   - Calls `store.has_processed_response()` to detect duplicates
   - Calls `store.save_submission()` to record outcome (valid or invalid)

3. **Phase 2+ evaluation modules will**
   - Query submissions by status ("valid") to find candidates for rating updates
   - Read extraction paths to load agent binaries
   - Check and increment daily quotas during batch evaluation

## Manual Setup: Team Registry

Before collections can run, **organizer must populate the teams table** with:

1. **Create initial team registry JSON** (e.g., `team_registry.json`):
   ```json
   [
       {
           "canonical_team_id": "team-alpha-123",
           "team_name": "Alpha Squadron",
           "primary_email": "alpha@example.com",
           "submission_token": "super-secret-token-1"
       },
       {
           "canonical_team_id": "team-bravo-456",
           "team_name": "Bravo Team",
           "primary_email": "bravo@example.com",
           "submission_token": "super-secret-token-2"
       }
   ]
   ```

2. **Bulk load via CLI**:
   ```bash
   python scripts/collect_submissions.py init-db --db-path competition.db
   ```

3. **Register each team** (or write a script to batch upsert):
   ```bash
   python scripts/collect_submissions.py upsert-team \
       --db-path competition.db \
       --team-id "team-alpha-123" \
       --team-name "Alpha Squadron" \
       --primary-email "alpha@example.com" \
       --token "super-secret-token-1"
   ```

### Immutability Guarantee

- `canonical_team_id` is the PRIMARY KEY and never changes
- Team names and emails can be updated via re-upsert, but team identity is fixed
- This ensures submission records stay linked to correct team even if display name changes

## Testing Notes

Tests for this module verify:
- SHA256 token hashing and verification
- Database persistence (team registration, submission records)
- Quota counting (increment and retrieval)
- Uniqueness constraints (team_name, response_id)
- Foreign key relationships (submissions → teams)

See `tests/test_storage_token_verification.py` for implementation.
