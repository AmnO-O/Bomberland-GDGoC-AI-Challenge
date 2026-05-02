# Ingestion Module: Submission Intake and Validation

## Purpose

The **ingestion module** orchestrates the complete submission collection pipeline:

1. **Download** submission zips from Google Drive (by file ID)
2. **Validate** zips against security and format rules
3. **Extract** files to immutable storage paths
4. **Record** outcome (valid or invalid) in the database

This module contains no database logic; it delegates storage to `competition.storage.SubmissionStore`. It is fully testable without Drive API mocking.

## Key Functions

### `get_drive_service(credentials_file: str)`

Create a Google Drive API service from a service account credentials JSON file.

```python
from competition.ingestion import get_drive_service

service = get_drive_service("service_account_credentials.json")
if service:
    print("✓ Drive API ready")
else:
    print("✗ Credentials file not found")
```

**Parameters:**
- `credentials_file`: Path to service account JSON (with Drive API read-only scope)

**Returns:**
- `googleapiclient.discovery.Resource` if credentials valid, `None` if file not found

**Error Handling:**
- Returns `None` if file doesn't exist (caller must check before proceeding)
- Raises `google.auth.exceptions.DefaultCredentialsError` if service account JSON is malformed

---

### `load_submission_metadata(metadata_file: str) -> list`

Load submission metadata from a JSON file.

```python
from competition.ingestion import load_submission_metadata

metadata = load_submission_metadata("submission_metadata.json")
# Returns: [
#     {
#         "drive_file_id": "abc123xyz",
#         "canonical_team_id": "team-001",
#         "submission_token": "secret-token-here",
#         "original_filename": "agent_v1.zip"  # optional
#     },
#     ...
# ]
```

**File Format:**

Must be a JSON array of objects with required and optional fields:

```json
[
    {
        "drive_file_id": "1a2b3c4d5e6f7g8h9i0jk1l2m",
        "canonical_team_id": "team-alpha-123",
        "submission_token": "reusable-secret-token",
        "original_filename": "agent_alpha_v2.zip"
    },
    {
        "drive_file_id": "2x3y4z5w6v7u8t9s0r1q2p3o",
        "canonical_team_id": "team-bravo-456",
        "submission_token": "reusable-secret-token-2"
    }
]
```

**Required Fields:**
- `drive_file_id`: Google Drive file ID of the zip to download
- `canonical_team_id`: Team's immutable identifier (must exist in DB)
- `submission_token`: Reusable team token for authentication

**Optional Fields:**
- `original_filename`: Name of the file in Google Drive (for logging/auditing)

**Returns:**
- List of dicts, each representing one submission to process

**Errors:**
- `FileNotFoundError` if metadata file doesn't exist
- `ValueError` if file is not valid JSON or not a JSON array

---

### `download_drive_file_bytes(service, file_id: str) -> bytes`

Download a file from Google Drive as raw bytes.

```python
from competition.ingestion import download_drive_file_bytes

service = get_drive_service("creds.json")
zip_bytes = download_drive_file_bytes(service, "drive-file-id-123")
print(f"Downloaded {len(zip_bytes)} bytes")
```

**Parameters:**
- `service`: Google Drive API service (from `get_drive_service()`)
- `file_id`: Google Drive file ID

**Returns:**
- `bytes`: Raw file content

**Errors:**
- `googleapiclient.errors.HttpError` if file not found or permission denied
- Network errors if Drive API unreachable

---

### `validate_zip_bytes(zip_data: bytes) -> (bool, Optional[str], Optional[dict])`

Validate a zip archive for safety and required structure.

```python
from competition.ingestion import validate_zip_bytes

is_valid, reason, manifest = validate_zip_bytes(zip_data)
if is_valid:
    print(f"✓ Valid zip with {len(manifest)} files:")
    for filename, size in manifest.items():
        print(f"  {filename}: {size} bytes")
else:
    print(f"✗ Invalid: {reason}")
```

**Returns:**
- `is_valid`: `True` if all checks pass
- `reason`: `None` if valid; error code/message if invalid (e.g., "zip_too_large", "agent_py_missing_or_multiple")
- `manifest`: `None` if invalid; dict of `{filename: file_size}` if valid

**Validation Rules:**

| Check | Limit | Error Code |
|-------|-------|-----------|
| Archive size | ≤ 100 MB | `zip_too_large` |
| Extracted total | ≤ 300 MB | `extracted_total_too_large` |
| Single file | ≤ 150 MB | `single_file_too_large` |
| File count | ≤ 200 files | `too_many_files` |
| **Paths** | No `/` prefix, no `..` | `unsafe_path` |
| **Extensions** | Only whitelisted | `disallowed_extension:{ext}` |
| **agent.py** | Exactly one, syntactically valid | `agent_py_missing_or_multiple` / `agent_py_syntax_error:{error}` |

**Allowed Extensions:**
`.py`, `.txt`, `.pt`, `.pth`, `.pkl`, `.onnx`, `.bin`, `.json`, `.yaml`, `.yml`, `.md`

**Attack Vectors Prevented:**
- **Zip bombs**: Checks extracted total size
- **Path traversal**: Rejects `..` and absolute paths
- **Symlink attacks**: Extracts only regular files (directories skipped)
- **Malicious extensions**: Whitelist enforcement
- **Missing agent**: Requires exactly one `agent.py`
- **Syntax errors**: Compiles agent.py to catch basic Python errors early

---

### `extract_zip_bytes(zip_data: bytes, target_dir: Path, manifest: dict) -> None`

Extract whitelisted files from a zip to a target directory.

```python
from pathlib import Path
from competition.ingestion import extract_zip_bytes

target_dir = Path("submissions/team-001/submission-uuid/")
extract_zip_bytes(zip_data, target_dir, manifest)
print(f"✓ Extracted to {target_dir}")
```

**Parameters:**
- `zip_data`: Raw zip bytes (must pass `validate_zip_bytes()` first)
- `target_dir`: Path object for extraction root
- `manifest`: Dict from `validate_zip_bytes()` (e.g., `{"agent.py": 1024, "weights.pt": 512000}`)

**Behavior:**
- Creates `target_dir` if it doesn't exist (including parents)
- Extracts only files listed in manifest (safe since manifest is pre-validated)
- Preserves directory structure within zip

**Errors:**
- `IOError` if disk full or permission denied

---

### `authenticate_submission(store: SubmissionStore, canonical_team_id: str, submission_token: str) -> (bool, Optional[str])`

Verify team identity and submission token.

```python
from competition.ingestion import authenticate_submission
from competition.storage import SubmissionStore

store = SubmissionStore("competition.db")
is_valid, reason = authenticate_submission(store, "team-001", submitted_token)
if is_valid:
    print("✓ Authentication passed")
else:
    print(f"✗ Auth failed: {reason}")
```

**Parameters:**
- `store`: SubmissionStore instance
- `canonical_team_id`: Team's immutable ID
- `submission_token`: Token to verify

**Returns:**
- `is_valid`: `True` if team exists, is active, and token matches
- `reason`: `None` if valid; error code if invalid:
  - `"unknown_team"`: Team not in DB
  - `"team_not_active:{status}"`: Team is suspended or has other non-active status
  - `"token_mismatch"`: Submitted token doesn't match stored hash

**Example - Check Reason:**
```python
if not is_valid:
    if reason == "unknown_team":
        print("Team not registered")
    elif reason.startswith("team_not_active"):
        print(f"Team is {reason.split(':')[1]}")
    elif reason == "token_mismatch":
        print("Invalid token (does not match registered token)")
```

---

### `process_submission_item(service, store: SubmissionStore, storage_dir: str, item: dict) -> (bool, str)`

Orchestrate the complete submission intake pipeline.

```python
from competition.ingestion import process_submission_item
from competition.storage import SubmissionStore
from googleapiclient.discovery import Resource

service: Resource = get_drive_service("creds.json")
store = SubmissionStore("competition.db")

item = {
    "drive_file_id": "abc123xyz",
    "canonical_team_id": "team-001",
    "submission_token": "secret-token",
    "original_filename": "agent_v1.zip"
}

ok, note = process_submission_item(service, store, "submissions", item)
if ok:
    print(f"✓ Success: {note}")  # e.g., "stored:submissions/team-001/uuid/"
else:
    print(f"✗ Failed: {note}")   # e.g., "auth_failed:token_mismatch"
```

**Parameters:**
- `service`: Google Drive API service
- `store`: SubmissionStore instance
- `storage_dir`: Root directory for extracted submissions (e.g., "submissions/")
- `item`: Metadata dict with drive_file_id, canonical_team_id, submission_token, optional original_filename

**Returns:**
- `ok`: `True` if stored successfully or already processed
- `note`: Status message:
  - On success: `"stored:{extracted_path}"` or `"already_processed"`
  - On failure: Error code/message (e.g., `"auth_failed:token_mismatch"`, `"zip_too_large"`)

**Execution Flow:**

```
1. Check required metadata fields
   ├─ Missing → return (False, "missing_metadata_fields:...")
   
2. Check if drive_file_id already processed (duplicate prevention)
   ├─ Yes → return (True, "already_processed")
   
3. Authenticate team and token
   ├─ Fail → return (False, "auth_failed:{reason}")
   
4. Download file from Drive
   ├─ Error → return (False, "download_failed:{error}")
   
5. Compute SHA256 hash
   
6. Validate zip archive
   ├─ Invalid → save invalid record to DB, return (False, reason)
   
7. Extract to target_dir
   ├─ Error → save extraction_failed record, return (False, reason)
   
8. Save valid submission record to DB
   
9. return (True, "stored:{target_dir}")
```

**Immutable Storage Path:**

Extracted files are always stored at:
```
{storage_dir}/{canonical_team_id}/{submission_id}/
```

Example:
```
submissions/team-alpha-123/a1b2c3d4-e5f6-47g8-h9i0-j1k2l3m4n5o6/
  agent.py
  model.pt
  config.yaml
```

---

## How Phase 2 Evaluator Will Use This Module

When pools and evaluation trigger (Phase 2), the evaluation runner will:

1. **Query valid submissions** from DB:
   ```python
   # Find latest valid submission per team
   cursor.execute(
       "SELECT * FROM submissions WHERE validation_status='valid' ORDER BY created_at DESC"
   )
   ```

2. **Load agent binaries** from extracted paths:
   ```python
   # From submission.extracted_path, load agent.py
   agent_path = Path(submission.extracted_path) / "agent.py"
   with open(agent_path) as f:
       agent_source = f.read()
   ```

3. **Run matches** and update ratings:
   ```python
   # Incrementally update submission.n_games, wins, losses, total_rank, total_steps
   ```

4. **Check daily quotas** if needed for background cycles:
   ```python
   count = store.get_daily_quota_count(team_id, day_utc)
   if count < MAX_SUBMISSIONS_PER_DAY:
       # eligible for this cycle
   ```

---

## Manual Setup: Drive Credentials and Metadata

### 1. Service Account Credentials

**Before running collections, organizer must**:

1. Create a Google Cloud Project
2. Enable Google Drive API
3. Create a Service Account with Drive read-only scope
4. Download credentials as `service_account_credentials.json`
5. Share the Google Drive folder with the service account email
6. Save credentials file in project root or specify `--credentials-file` when running collector

### 2. Submission Metadata JSON

**Organizer must generate `submission_metadata.json`** with entries for each submission to download:

**Option A: Manual JSON** (small competitions):
```json
[
    {
        "drive_file_id": "1a2b3c4d5e6f7g8h9i0jk1l2m",
        "canonical_team_id": "team-alpha-123",
        "submission_token": "token-1",
        "original_filename": "agent_v1.zip"
    }
]
```

**Option B: Google Form integration** (larger competitions):
- Use Google Forms for submission; responses are stored in a Google Sheet
- Write a script to export form responses to metadata JSON:
  ```python
  # Pseudo-code
  # 1. Query Google Sheets API for form responses
  # 2. Extract drive_file_id from File Upload column
  # 3. Extract team metadata from form fields
  # 4. Write to JSON file
  ```

**Option C: Custom ingestion** (external systems):
- If teams submit elsewhere (GitHub, email, S3), write a bridge script:
  ```python
  # 1. Fetch submissions from external source
  # 2. Upload to Google Drive
  # 3. Generate metadata JSON with drive_file_ids
  ```

---

## Testing Notes

Tests for this module verify:
- **Validation**: bomb detection, traversal prevention, extension whitelist, agent.py syntax
- **Extraction**: file preservation, directory structure, manifest accuracy
- **Authentication**: token verification, team status checks
- **End-to-end**: complete pipeline from metadata to stored records

See `tests/test_ingestion_zip_validation.py` and `tests/test_ingestion_submission_save.py`.

---

## Constants

Module-level limits (all configurable via imports if needed):

```python
MAX_ZIP_SIZE_BYTES = 100 * 1024 * 1024          # 100 MB
MAX_EXTRACTED_TOTAL_BYTES = 300 * 1024 * 1024  # 300 MB
MAX_SINGLE_FILE_BYTES = 150 * 1024 * 1024      # 150 MB
MAX_FILE_COUNT = 200                             # per zip

ALLOWED_EXTENSIONS = {
    ".py", ".txt", ".pt", ".pth", ".pkl", ".onnx", ".bin",
    ".json", ".yaml", ".yml", ".md"
}
```

---

## Submission Webhook Handler

The **submission webhook module** (`submission_webhook.py`) provides event-driven submission intake directly from Google Forms via Apps Script.

### `get_vietnam_day_identifier(now: Optional[datetime] = None) -> str`

Get the day identifier for Vietnam timezone (UTC+7) with 7 AM daily reset.

```python
from competition.ingestion.submission_webhook import get_vietnam_day_identifier

day_id = get_vietnam_day_identifier()
# Returns: "2026-04-25" (format: YYYY-MM-DD)
# Reset time: 7 AM Vietnam time (UTC+7) each day
```

**Behavior:**
- Converts current UTC time to Vietnam timezone (UTC+7)
- Applies 7 AM reset: times from 7 AM to 11:59 PM = current day
- Times from 00:00 AM to 6:59 AM = previous day (before reset)

**Example Time Boundaries:**
```
2026-04-24 23:59:59 UTC → 2026-04-25 06:59:59 UTC+7 → day_id = "2026-04-24"
2026-04-25 00:00:00 UTC → 2026-04-25 07:00:00 UTC+7 → day_id = "2026-04-25" (reset!)
```

**Parameters:**
- `now`: Optional UTC datetime. If None, uses `datetime.now(timezone.utc)`

**Returns:**
- `str`: Day identifier in "YYYY-MM-DD" format for quota tracking

---

### `process_submission_webhook(request_json: dict, store: SubmissionStore, service, storage_dir: str = "submissions") -> Tuple[bool, dict]`

Handle incoming submission webhook from Google Form.

Validates token, checks daily quota, invokes collector, saves to DB.

```python
from competition.ingestion.submission_webhook import process_submission_webhook
from competition.storage import SubmissionStore

store = SubmissionStore("competition.db")
service = get_drive_service("service_account_credentials.json")

payload = {
    "canonical_team_id": "awesome_ai_a1b2c3d4",
    "submission_token": "token_xyz_123_longtoken",
    "drive_file_id": "1ABC_file_id_123_XYZ",
    "changelog": "Improved agent with better heuristics",
    "original_filename": "submission_v2.zip",  # optional
}

ok, result = process_submission_webhook(payload, store, service)

if ok:
    print(f"✓ Success: {result['submission_id']}")
    print(f"  Remaining today: {result['remaining_today']}")
else:
    print(f"✗ {result['error']}: {result['reason']}")
```

**Parameters:**
- `request_json`: JSON payload from Apps Script (see "Input Format" below)
- `store`: SubmissionStore instance for DB operations
- `service`: Google Drive API service (from `get_drive_service()`)
- `storage_dir`: Root directory for extracted submissions (default: `"submissions"`)

**Input Format:**

```json
{
    "canonical_team_id": "awesome_ai_a1b2c3d4",
    "submission_token": "64_char_hex_token_here_...",
    "drive_file_id": "google_drive_file_id",
    "changelog": "Optional description of changes",
    "original_filename": "submission.zip"
}
```

**Required Fields:**
- `canonical_team_id`: Team's immutable ID (from registration)
- `submission_token`: Team's secret token (from registration email)
- `drive_file_id`: Google Drive file ID of the ZIP to download

**Optional Fields:**
- `changelog`: Text description of submission changes
- `original_filename`: Original filename (for logging)

**Returns:**
- `Tuple[bool, dict]`:
  - Success: `(True, {"status": "success", "submission_id": str, "remaining_today": int, ...})`
  - Failure: `(False, {"error": str, "reason": str})`

**Error Codes:**

| Error | HTTP | Reason |
|-------|------|--------|
| `missing_field` | 400 | Required field missing (see `reason` field) |
| `auth_failed` | 401 | `unknown_team` / `team_not_active` / `token_mismatch` |
| `quota_exceeded` | 429 | Max 3 submissions per day (Vietnam time, resets 7 AM) |
| `validation_failed` | 400 | ZIP validation error (invalid, agent.py missing, etc.) |

**Execution Flow:**

```
1. Validate all required fields present and non-empty
   ├─ Missing → return (False, {"error": "missing_field", "reason": "..."})

2. Verify team exists and is active
   ├─ Unknown → return (False, {"error": "auth_failed", "reason": "unknown_team"})
   ├─ Not active → return (False, {"error": "auth_failed", "reason": "team_not_active:status"})

3. Verify submission token matches stored hash
   ├─ Mismatch → return (False, {"error": "auth_failed", "reason": "token_mismatch"})

4. Check daily submission quota (Vietnam timezone, 7 AM reset)
   ├─ Exhausted → return (False, {"error": "quota_exceeded", "reason": "max 3 per day"})

5. Invoke collector pipeline:
   - Download ZIP from Google Drive
   - Validate ZIP (security, format, agent.py)
   - Extract to submissions/{canonical_team_id}/{submission_id}/
   - Save record to DB
   ├─ Failure → return (False, {"error": "validation_failed", "reason": "..."})

6. Increment daily quota counter for this team/day

7. Return success with submission ID and remaining quota
```

**Quota Enforcement:**

```python
MAX_SUBMISSIONS_PER_DAY = 3

# Quota resets at 7 AM Vietnam time (UTC+7) daily
# Day identifier: get_vietnam_day_identifier()

# Example:
# 2026-04-25 08:00 UTC+7 → day_id = "2026-04-25"
# 2026-04-25 06:30 UTC+7 → day_id = "2026-04-24" (before 7 AM reset)
```

**Integration with Flask:**

The Flask app (`competition/registration/app.py`) exposes a `POST /submit` endpoint that calls this function:

```python
@app.post("/submit")
def submit_solution():
    # Validates Bearer token
    # Initializes SubmissionStore and Google Drive service
    # Calls process_submission_webhook()
    # Returns JSON response with appropriate HTTP status code
```

---

## Apps Script Webhook Integration

The submission form uses an Apps Script trigger to call the `/submit` endpoint automatically.

**Script File:** `evaluation/google_apps_script_submission.gs`

**Form Trigger:** `onFormSubmit` event

**How it works:**
1. User submits Submission Form
2. Apps Script `onFormSubmit()` trigger fires
3. Extracts form fields:
   - Canonical Team ID
   - Submission Token
   - Changelog
   - Submission ZIP (Google Drive link)
4. Calls Flask `POST /submit` endpoint with Bearer token
5. Logs webhook response in Apps Script console

**Configuration in Apps Script:**

```javascript
const SUBMISSION_WEBHOOK_URL = "https://your-domain.com/submit";
const SUBMISSION_WEBHOOK_AUTH_TOKEN = "your_bearer_token_here";
```

**Testing:**

Before deploying, test webhook connectivity from Apps Script console:

```javascript
testWebhookConnectivity();  // Logs success/failure to console
```

---

## Testing

Tests for submission webhook verify:
- **Vietnam timezone**: 7 AM reset boundary, day identifier calculation
- **Field validation**: Missing required fields rejected
- **Authentication**: Team lookup, token verification
- **Quota enforcement**: Daily counter, per-team, per-day calculation
- **Successful processing**: Valid submission stored, quota incremented
- **Failure handling**: Invalid submissions don't increment quota

See `tests/test_submission_webhook.py` (20 comprehensive tests).

---

## Curl Examples for Testing

**Test successful submission:**

```bash
curl -X POST http://localhost:5000/submit \
  -H "Authorization: Bearer your_token_here" \
  -H "Content-Type: application/json" \
  -d '{
    "canonical_team_id": "test_team_a1b2c3d4",
    "submission_token": "your_team_token_here",
    "drive_file_id": "1ABC_google_drive_id_XYZ",
    "changelog": "Improved heuristics"
  }'

# Expected response (200):
{
  "status": "success",
  "submission_id": "uuid-here",
  "reason": "stored:/submissions/test_team_a1b2c3d4/uuid/",
  "remaining_today": 2
}
```

**Test quota exceeded:**

```bash
# After 3 submissions already made today
curl -X POST http://localhost:5000/submit \
  -H "Authorization: Bearer your_token_here" \
  -H "Content-Type: application/json" \
  -d '{"canonical_team_id": "test_team_a1b2c3d4", ...}'

# Expected response (429):
{
  "error": "quota_exceeded",
  "reason": "max 3 submissions per day (Vietnam time, resets 7 AM)"
}
```

**Test authentication failure:**

```bash
curl -X POST http://localhost:5000/submit \
  -H "Authorization: Bearer your_token_here" \
  -H "Content-Type: application/json" \
  -d '{
    "canonical_team_id": "unknown_team_xyz",
    "submission_token": "any_token",
    "drive_file_id": "file_id"
  }'

# Expected response (401):
{
  "error": "auth_failed",
  "reason": "unknown_team"
}
```

