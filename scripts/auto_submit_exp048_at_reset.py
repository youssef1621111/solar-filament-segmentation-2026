#!/usr/bin/env python3
"""
scripts/auto_submit_exp048_at_reset.py
IEEE BigData 2026 Solar Filament Segmentation Challenge (filament-segmentation-2026)

Automated Submission Daemon for 00:00:00 UTC Quota Reset
1. Calculates exact time until 00:00:05 UTC (November 15, 2026).
2. Waits until quota reset.
3. Submits final_submission/submission_exp048_skeleton_adaptive_sq.csv to Kaggle.
4. Monitors Kaggle evaluation and logs the evaluated public LB score.
"""

import os
import sys
import time
import subprocess
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SUBMISSION_CSV = PROJECT_ROOT / "final_submission/submission_exp048_skeleton_adaptive_sq.csv"
KAGGLE_API_TOKEN = "KGAT_ec55127abae8c74b1427719d0582ce0b"
COMPETITION = "filament-segmentation-2026"
MESSAGE = "EXP-048: Skeleton-Width-Adaptive Boundary Refinement (+4.35% SQ envelope)"


def main():
    print("=" * 80)
    print("AUTOMATED SUBMISSION DAEMON: EXP-048 (SKELETON-WIDTH-ADAPTIVE SQ)")
    print("=" * 80)

    if not SUBMISSION_CSV.exists():
        print(f"[!] Error: Submission file not found: {SUBMISSION_CSV}")
        sys.exit(1)

    print(f"[*] Target file: {SUBMISSION_CSV.name} ({SUBMISSION_CSV.stat().st_size:,} bytes)")
    print(f"[*] Competition: {COMPETITION}")

    # Set Kaggle API token in environment
    env = os.environ.copy()
    env["KAGGLE_API_TOKEN"] = KAGGLE_API_TOKEN

    now_utc = datetime.now(timezone.utc)
    print(f"[*] Current UTC Time: {now_utc.strftime('%Y-%m-%d %H:%M:%S UTC')}")

    # Target reset time: 2026-11-15 00:00:05 UTC
    target_utc = datetime(2026, 11, 15, 0, 0, 5, tzinfo=timezone.utc)
    delta_sec = (target_utc - now_utc).total_seconds()

    if delta_sec > 0:
        hours = int(delta_sec // 3600)
        minutes = int((delta_sec % 3600) // 60)
        seconds = int(delta_sec % 60)
        print(f"[*] Quota resets at 00:00:00 UTC. Sleeping for {hours}h {minutes}m {seconds}s ({delta_sec:.1f}s)...")
        time.sleep(delta_sec)
    else:
        print("[*] Quota already reset for November 15! Proceeding immediately to submit...")

    print(f"\n[*] Launching Kaggle submission at {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}...")
    submit_cmd = [
        "kaggle", "competitions", "submit",
        "-c", COMPETITION,
        "-f", str(SUBMISSION_CSV),
        "-m", MESSAGE
    ]

    try:
        res = subprocess.run(submit_cmd, env=env, capture_output=True, text=True, check=True)
        print("[+] Kaggle Submit Output:\n" + res.stdout)
    except subprocess.CalledProcessError as e:
        print(f"[!] Submission Error (Code {e.returncode}):\n{e.stderr}\n{e.stdout}")
        sys.exit(1)

    print("[*] Monitoring submission status on Kaggle cluster...")
    for attempt in range(1, 21):
        time.sleep(25)
        check_cmd = ["kaggle", "competitions", "submissions", "-c", COMPETITION]
        try:
            check_res = subprocess.run(check_cmd, env=env, capture_output=True, text=True)
            lines = check_res.stdout.strip().split("\n")
            # Print top 4 lines
            print(f"\n--- Check Attempt {attempt} ({datetime.now(timezone.utc).strftime('%H:%M:%S UTC')}) ---")
            for line in lines[:4]:
                print(line)

            # Check if top submission is complete
            if len(lines) > 2 and "SubmissionStatus.COMPLETE" in lines[2]:
                print("\n[+] EVALUATION COMPLETE! Checked score above.")
                break
        except Exception as ex:
            print(f"[!] Warning: Status query error: {ex}")

    print("\n" + "=" * 80)
    print("DAEMON RUN FINISHED.")
    print("=" * 80)


if __name__ == "__main__":
    main()
