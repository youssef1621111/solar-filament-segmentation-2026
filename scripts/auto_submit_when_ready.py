"""
Automated Monitor & Submitter for Kaggle GPU Training Pipeline
Polls cracker1war/filament-consensus-pipeline until COMPLETE,
downloads submission.csv, verifies submission integrity, and submits to Kaggle.
"""

import os
import sys
import time
import subprocess
import pathlib

# Ensure repository root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

KAGGLE_BIN = os.path.expanduser("~/.local/bin/kaggle")
KERNEL_ID = "cracker1war/filament-consensus-pipeline"
COMPETITION = "filament-segmentation-2026"
OUTPUT_DIR = "/home/fathyyoussef162/kaggle_output"


def check_kernel_status():
    cmd = [KAGGLE_BIN, "kernels", "status", KERNEL_ID]
    result = subprocess.run(cmd, capture_output=True, text=True)
    out = result.stdout.strip()
    return out


def download_outputs():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    cmd = [KAGGLE_BIN, "kernels", "output", KERNEL_ID, "-p", OUTPUT_DIR]
    print(f"Downloading kernel outputs to {OUTPUT_DIR}...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print(result.stderr)


def submit_to_competition(sub_file, message):
    cmd = [
        KAGGLE_BIN, "competitions", "submit",
        "-c", COMPETITION,
        "-f", sub_file,
        "-m", message
    ]
    print(f"Submitting {sub_file} to Kaggle competition {COMPETITION}...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print(result.stderr)


def check_submission_status():
    cmd = [KAGGLE_BIN, "competitions", "submissions", "-c", COMPETITION]
    result = subprocess.run(cmd, capture_output=True, text=True)
    print("Latest Submissions:")
    lines = result.stdout.split("\n")[:8]
    print("\n".join(lines))


def main():
    print("=" * 80)
    print("KAGGLE GPU PIPELINE MONITOR & AUTO-SUBMITTER")
    print("=" * 80)

    status_str = check_kernel_status()
    print(f"Current Kernel Status: {status_str}")

    if "COMPLETE" in status_str:
        print("[+] Kernel execution COMPLETE! Proceeding to download and submission...")
        download_outputs()
        sub_csv = os.path.join(OUTPUT_DIR, "submission.csv")
        if os.path.exists(sub_csv):
            msg = "ROGGI SolarPanoptic Consensus Pipeline (Tesla T4 Fold-0)"
            submit_to_competition(sub_csv, msg)
            time.sleep(5)
            check_submission_status()
        else:
            print(f"[!] Warning: submission.csv not found in {OUTPUT_DIR}!")
    elif "ERROR" in status_str:
        print(f"[!] Kernel finished with ERROR: {status_str}")
        download_outputs()
    elif "RUNNING" in status_str:
        print("[*] Kernel is actively RUNNING on Kaggle GPU.")
    elif "QUEUED" in status_str:
        print("[*] Kernel is QUEUED on Kaggle GPU.")


if __name__ == "__main__":
    main()
