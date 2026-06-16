"""Stream-process-discard pipeline: download JSON match logs from Drive, extract metrics, delete.

Designed to run on the VM where the production Drive credentials exist.
Downloads one day at a time to respect disk constraints.

Usage:
    python analysis/drive_stream_pipeline.py --sample_rate 0.10 --output analysis/data/metrics.csv

Options:
    --sample_rate   Fraction of files to process per day (0.10 = 10% sampling, 1.0 = all)
    --start_date    Start from this date (YYYY-MM-DD), default: earliest
    --end_date      End at this date (YYYY-MM-DD), default: latest
    --output        Output CSV path
    --tmp_dir       Temporary directory for downloads (cleaned after each day)
"""

import argparse
import csv
import io
import json
import os
import random
import shutil
import sys
import time
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def get_drive_service():
    """Get authenticated Drive service using project credentials."""
    from competition.config import load_env
    load_env()
    from competition.integrations.drive_upload import get_drive_service as _get_svc
    return _get_svc()


def list_date_folders(service, json_folder_id: str) -> list[dict]:
    """List all date subfolders inside the json/ Drive folder."""
    folders = []
    page_token = None
    while True:
        resp = service.files().list(
            q=f"'{json_folder_id}' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false",
            fields="files(id, name), nextPageToken",
            pageSize=100,
            orderBy="name",
            supportsAllDrives=True,
            includeItemsFromAllDrives=True,
            pageToken=page_token,
        ).execute()
        folders.extend(resp.get("files", []))
        page_token = resp.get("nextPageToken")
        if not page_token:
            break
    return sorted(folders, key=lambda f: f["name"])


def list_files_in_folder(service, folder_id: str) -> list[dict]:
    """List all files in a Drive folder with pagination."""
    files = []
    page_token = None
    while True:
        resp = service.files().list(
            q=f"'{folder_id}' in parents and trashed=false",
            fields="files(id, name, size), nextPageToken",
            pageSize=1000,
            supportsAllDrives=True,
            includeItemsFromAllDrives=True,
            pageToken=page_token,
        ).execute()
        files.extend(resp.get("files", []))
        page_token = resp.get("nextPageToken")
        if not page_token:
            break
    return files


def download_file_content(service, file_id: str) -> bytes:
    """Download file content directly into memory."""
    from googleapiclient.http import MediaIoBaseDownload
    request = service.files().get_media(
        fileId=file_id,
        supportsAllDrives=True,
    )
    buffer = io.BytesIO()
    downloader = MediaIoBaseDownload(buffer, request)
    done = False
    while not done:
        _, done = downloader.next_chunk()
    return buffer.getvalue()


def download_file_to_disk(service, file_id: str, dest_path: str):
    """Download file to a local path."""
    from googleapiclient.http import MediaIoBaseDownload
    request = service.files().get_media(
        fileId=file_id,
        supportsAllDrives=True,
    )
    with open(dest_path, "wb") as f:
        downloader = MediaIoBaseDownload(f, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()


def run_pipeline(
    sample_rate: float = 0.10,
    start_date: str = "",
    end_date: str = "",
    output_csv: str = "analysis/data/metrics.csv",
    tmp_dir: str = "analysis/tmp_json",
    in_memory: bool = True,
):
    """Main pipeline: for each date folder on Drive, download JSONs → extract metrics → cleanup."""

    # Import metrics extraction
    from analysis.extract_match_metrics import process_json_file, HEADER, extract_metrics

    service = get_drive_service()
    drive_folder_id = os.getenv("DRIVE_FOLDER_ID", "")

    # Find json/ folder
    resp = service.files().list(
        q=f"'{drive_folder_id}' in parents and name='json' and mimeType='application/vnd.google-apps.folder' and trashed=false",
        fields="files(id)",
        supportsAllDrives=True,
        includeItemsFromAllDrives=True,
    ).execute()
    json_folder_id = resp["files"][0]["id"]

    # List date folders
    date_folders = list_date_folders(service, json_folder_id)
    print(f"Found {len(date_folders)} date folders on Drive")

    # Filter by date range
    if start_date:
        date_folders = [df for df in date_folders if df["name"] >= start_date]
    if end_date:
        date_folders = [df for df in date_folders if df["name"] <= end_date]
    print(f"Processing {len(date_folders)} date folders (sample_rate={sample_rate:.0%})")

    # Setup output CSV and load existing for safe resume
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not output_path.exists() or output_path.stat().st_size == 0
    
    processed_match_ids = set()
    if output_path.exists() and output_path.stat().st_size > 0:
        with open(output_csv, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if "match_id" in row:
                    processed_match_ids.add(row["match_id"])
        print(f"Found {len(processed_match_ids)} previously processed matches. Resuming safely.")

    tmp_path = Path(tmp_dir)

    total_matches = 0
    total_rows = 0
    t_start = time.time()

    with open(output_csv, "a", newline="") as csvf:
        writer = csv.DictWriter(csvf, fieldnames=HEADER)
        if write_header:
            writer.writeheader()

        for df_idx, date_folder in enumerate(date_folders):
            date_str = date_folder["name"]
            folder_id = date_folder["id"]

            # List files for this date
            files = list_files_in_folder(service, folder_id)
            json_files = [f for f in files if f["name"].endswith(".json")]
            json_files.sort(key=lambda x: x["name"])  # Ensure determinism
            total_in_day = len(json_files)

            # Deterministic Sample based on date
            if sample_rate < 1.0:
                n_sample = max(1, int(total_in_day * sample_rate))
                rng = random.Random(date_str)
                sampled_files = rng.sample(json_files, min(n_sample, total_in_day))
            else:
                sampled_files = json_files
                
            # Filter out already processed matches
            json_files = [f for f in sampled_files if Path(f["name"]).stem not in processed_match_ids]

            n_to_process = len(json_files)
            target_sample_size = len(sampled_files)
            
            if n_to_process == 0:
                print(f"\n[{df_idx+1}/{len(date_folders)}] {date_str}: "
                      f"Already fully processed ({target_sample_size}/{total_in_day} sampled files done).")
                continue

            skipped = target_sample_size - n_to_process
            print(f"\n[{df_idx+1}/{len(date_folders)}] {date_str}: "
                  f"{n_to_process} files to process (skipping {skipped} already done from {target_sample_size} sampled)")

            day_rows = 0

            if in_memory:
                import concurrent.futures
                import threading
                import time

                thread_local = threading.local()

                def get_thread_service():
                    if not hasattr(thread_local, "service"):
                        thread_local.service = get_drive_service()
                    return thread_local.service

                def process_single_file(file_info):
                    for attempt in range(3):
                        try:
                            svc = get_thread_service()
                            content = download_file_content(svc, file_info["id"])
                            data = json.loads(content)
                            rows = extract_metrics(data)

                            fname = Path(file_info["name"]).stem
                            parts = fname.split("_")
                            file_date = ""
                            if len(parts) >= 2 and len(parts[1]) == 8:
                                raw = parts[1]
                                file_date = f"{raw[:4]}-{raw[4:6]}-{raw[6:8]}"

                            for row in rows:
                                row["match_id"] = fname
                                row["date"] = file_date or date_str
                            return rows
                        except Exception as e:
                            if attempt == 2:
                                print(f"    WARN: {file_info['name']}: {e}", file=sys.stderr)
                                return []
                            time.sleep(1 + attempt * 2)

                completed = 0
                # Using 10 workers avoids hitting the Google Drive API rate limits
                with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                    futures = {executor.submit(process_single_file, fi): fi for fi in json_files}
                    for future in concurrent.futures.as_completed(futures):
                        rows = future.result()
                        for row in rows:
                            writer.writerow(row)
                            day_rows += 1
                        completed += 1
                        if completed % 100 == 0:
                            print(f"    {completed}/{n_to_process} downloaded+processed")

            else:
                # Disk-based processing (for very large files or debugging)
                tmp_path.mkdir(parents=True, exist_ok=True)
                for fi, file_info in enumerate(json_files):
                    dest = tmp_path / file_info["name"]
                    try:
                        download_file_to_disk(service, file_info["id"], str(dest))
                        rows = process_json_file(str(dest))
                        for row in rows:
                            row["date"] = row["date"] or date_str
                            writer.writerow(row)
                            day_rows += 1
                        dest.unlink()  # Delete immediately
                    except Exception as e:
                        print(f"    WARN: {file_info['name']}: {e}", file=sys.stderr)
                        if dest.exists():
                            dest.unlink()

                # Cleanup tmp dir
                if tmp_path.exists():
                    shutil.rmtree(str(tmp_path), ignore_errors=True)

            total_matches += n_to_process
            total_rows += day_rows
            elapsed = time.time() - t_start
            rate = total_matches / elapsed if elapsed > 0 else 0
            print(f"  → {day_rows} metric rows | cumulative: {total_matches} matches, "
                  f"{total_rows} rows | {rate:.1f} matches/sec")

            # Flush CSV periodically
            csvf.flush()

    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"Pipeline complete!")
    print(f"  Total matches processed: {total_matches}")
    print(f"  Total metric rows: {total_rows}")
    print(f"  Elapsed: {elapsed/60:.1f} minutes")
    print(f"  Output: {output_csv}")
    print(f"  Output size: {os.path.getsize(output_csv)/1024/1024:.1f} MB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stream JSON metrics from Drive")
    parser.add_argument("--sample_rate", type=float, default=0.10,
                        help="Fraction of files to process per day (default: 0.10)")
    parser.add_argument("--start_date", default="", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end_date", default="", help="End date (YYYY-MM-DD)")
    parser.add_argument("--output", default="analysis/data/metrics.csv",
                        help="Output CSV path")
    parser.add_argument("--tmp_dir", default="analysis/tmp_json",
                        help="Temp directory for disk-based download")
    parser.add_argument("--in_memory", action="store_true", default=True,
                        help="Download directly to memory (no disk writes)")
    parser.add_argument("--disk_mode", dest="in_memory", action="store_false",
                        help="Download to disk then process")
    args = parser.parse_args()

    run_pipeline(
        sample_rate=args.sample_rate,
        start_date=args.start_date,
        end_date=args.end_date,
        output_csv=args.output,
        tmp_dir=args.tmp_dir,
        in_memory=args.in_memory,
    )
