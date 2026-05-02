from pathlib import Path

from competition.integrations.drive_upload import create_drive_token, get_drive_service


token_path = Path("evaluation/token.json")
if not token_path.exists():
    print("No OAuth token found, starting interactive login...")
    create_drive_token()

service = get_drive_service()
results = service.files().list(pageSize=10, fields="files(id, name)").execute()

files = results.get("files", [])

print("FILES:")
for f in files:
    print(f["name"], f["id"])