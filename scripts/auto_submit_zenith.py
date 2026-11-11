#!/usr/bin/env python3
"""
Automated Championship Submission and Live Score Poller.
Submits the audited Zenith V4 candidate and tracks evaluation to completion.
"""

import argparse
import datetime
import json
import re
import subprocess
import time
from pathlib import Path


def submit_and_track(csv_path: str, message: str, wait_until_reset: bool = False):
    csv_file = Path(csv_path)
    if not csv_file.exists():
        raise FileNotFoundError(f"Submission file not found: {csv_path}")

    # Check time if waiting
    if wait_until_reset:
        now = datetime.datetime.now(datetime.timezone.utc)
        target = now.replace(hour=0, minute=0, second=5, microsecond=0)
        if target <= now:
            target += datetime.timedelta(days=1)
        delay = (target - now).total_seconds()
        if delay > 0:
            print(f"[{datetime.datetime.now(datetime.timezone.utc).isoformat()}] Waiting {delay/3600:.2f} hours ({delay:.0f}s) until {target.isoformat()} reset...")
            time.sleep(delay)

    print(f"[{datetime.datetime.now(datetime.timezone.utc).isoformat()}] SUBMITTING TO KAGGLE...")
    cmd = [
        "/usr/bin/python3", "-m", "kaggle", "competitions", "submit",
        "-c", "filament-segmentation-2026",
        "-f", str(csv_file),
        "-m", message
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    print("Submit stdout:", res.stdout.strip())
    if res.stderr:
        print("Submit stderr:", res.stderr.strip())

    if res.returncode != 0:
        print("Submission command failed. Quota may not have reset yet.")
        return

    # Extract submission reference or poll history
    print("\nPolling Kaggle for evaluation status...")
    for attempt in range(1, 30):
        time.sleep(15)
        hist_cmd = [
            "/usr/bin/python3", "-m", "kaggle", "competitions", "submissions",
            "-c", "filament-segmentation-2026"
        ]
        hist_res = subprocess.run(hist_cmd, capture_output=True, text=True)
        lines = [line.strip() for line in hist_res.stdout.split("\n") if line.strip()]
        if len(lines) < 3:
            continue
        
        # Parse top submission
        latest_line = lines[2]
        print(f"[Attempt {attempt}/30] Latest submission row:\n  {latest_line}")
        
        if "COMPLETE" in latest_line:
            parts = re.split(r'\s{2,}', latest_line)
            ref = parts[0]
            score = parts[-2] if len(parts) >= 2 else "N/A"
            print("\n" + "=" * 50)
            print(f"🎉 SUBMISSION COMPLETE!")
            print(f"Reference ID: {ref}")
            print(f"PUBLIC LEADERBOARD SCORE: {score}")
            print("=" * 50)
            
            # Save receipt
            receipt = {
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "file": str(csv_file),
                "ref": ref,
                "score": score,
                "row": latest_line
            }
            Path("/home/fathyyoussef162/experiments/latest_submission_receipt.json").write_text(json.dumps(receipt, indent=2))
            return
        elif "ERROR" in latest_line:
            print("\n⚠️ Submission failed or returned ERROR status.")
            return

    print("Timed out polling for submission status.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Submit to Kaggle and poll score.")
    parser.add_argument("--csv", default="/home/fathyyoussef162/experiments/submission_championship_v10_apex_180.csv")
    parser.add_argument("--msg", default="Championship V10 Apex: 1327 instances, 180/180 stems, strict 0 collisions, 1732 TPs, local PQ 0.67412")
    parser.add_argument("--wait", action="store_true", help="Wait until 00:00:05 UTC before submitting")
    args = parser.parse_args()

    submit_and_track(args.csv, args.msg, wait_until_reset=args.wait)
