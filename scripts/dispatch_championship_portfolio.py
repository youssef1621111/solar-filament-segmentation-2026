#!/usr/bin/env python3
"""
Championship Portfolio Dispatcher for IEEE BigData 2026 Solar Filament Challenge.
Manages the deployment of the 5 Nov 4 submission slots.
"""

import argparse
import datetime
import json
import re
import subprocess
import sys
import time
from pathlib import Path

SUBMISSIONS = [
    {
        "slot": 1,
        "name": "Apex V10 (100% Coverage & Precision Flagship)",
        "path": "/home/fathyyoussef162/experiments/submission_championship_v10_apex_180.csv",
        "msg": "Championship V10 Apex: 1327 verified instances across all 180 stems (100% test coverage), 0 collisions, local PQ 0.67412",
        "expected_score": "0.66 - 0.70"
    },
    {
        "slot": 2,
        "name": "Ragnarok V8 (Multi-Annotator)",
        "path": "/home/fathyyoussef162/experiments/submission_championship_v8_ragnarok.csv",
        "msg": "Championship V8 Ragnarok: 1327 instances, 9 multi-annotator super solos, 0 collisions, local PQ 0.67445",
        "expected_score": "0.65 - 0.68"
    },
    {
        "slot": 3,
        "name": "Valhalla V7 (2px Dilation)",
        "path": "/home/fathyyoussef162/experiments/submission_championship_v7_valhalla.csv",
        "msg": "Championship V7 Valhalla: 1318 instances, 12 safe mutations, 0 collisions, local PQ 0.67228",
        "expected_score": "0.64 - 0.67"
    },
    {
        "slot": 4,
        "name": "Pantheon V6 (Conservative 1px)",
        "path": "/home/fathyyoussef162/experiments/submission_championship_v6_pantheon.csv",
        "msg": "Championship V6 Pantheon: 1318 instances, 8 safe mutations, 0 collisions, local PQ 0.67175",
        "expected_score": "0.62 - 0.65"
    },
    {
        "slot": 5,
        "name": "Olympus V5 (Raw Model Boundary)",
        "path": "/home/fathyyoussef162/experiments/submission_championship_v5_olympus.csv",
        "msg": "Championship V5 Olympus: 1318 instances, recovered stem, 0 collisions, local PQ 0.66974",
        "expected_score": "0.61 - 0.63"
    }
]


def submit_slot(slot_idx: int):
    entry = SUBMISSIONS[slot_idx - 1]
    csv_file = Path(entry["path"])
    if not csv_file.exists():
        raise FileNotFoundError(f"Missing file: {csv_file}")

    print(f"\n==========================================")
    print(f"Submitting Slot {entry['slot']}: {entry['name']}")
    print(f"File: {csv_file}")
    print(f"Expected Score: {entry['expected_score']}")
    print(f"==========================================")

    cmd = [
        "/usr/bin/python3", "-m", "kaggle", "competitions", "submit",
        "-c", "filament-segmentation-2026",
        "-f", str(csv_file),
        "-m", entry["msg"]
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    print("Stdout:", res.stdout.strip())
    if res.stderr:
        print("Stderr:", res.stderr.strip())

    if res.returncode != 0:
        print(f"Submission for Slot {entry['slot']} failed.")
        return None

    # Poll status
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
        
        latest_line = lines[2]
        print(f"[Attempt {attempt}/30] {latest_line}")
        if "COMPLETE" in latest_line:
            parts = re.split(r'\s{2,}', latest_line)
            ref = parts[0]
            score = parts[-2] if len(parts) >= 2 else "N/A"
            print(f"\n🎉 SLOT {entry['slot']} COMPLETE! Score: {score}")
            return score
        elif "ERROR" in latest_line:
            print(f"\n⚠️ Slot {entry['slot']} returned ERROR.")
            return None

    print(f"Timed out polling Slot {entry['slot']}.")
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--slot", type=int, default=1, choices=[1, 2, 3, 4, 5], help="Slot number (1-5)")
    args = parser.parse_args()
    submit_slot(args.slot)
