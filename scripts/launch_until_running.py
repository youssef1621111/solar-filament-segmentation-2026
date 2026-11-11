#!/usr/bin/env python3
"""
scripts/launch_until_running.py
Automated Kaggle GPU Launcher & Monitor
Pushes the 3-Architecture Native Ensemble kernel and monitors until RUNNING / COMPLETE.
"""

import os
import sys
import time
import subprocess
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi

KERNEL_DIR = "/home/fathyyoussef162/kaggle_kernel"
KERNEL_SLUG = "cracker1war/solar-filament-championship-gpu-inference"
OUTPUT_DIR = "/home/fathyyoussef162/kaggle_kernel/output"
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)


def get_api():
    api = KaggleApi()
    api.authenticate()
    return api


def push_kernel(api):
    print(f"[{time.strftime('%H:%M:%S')}] Pushing kernel from {KERNEL_DIR} to Kaggle...", flush=True)
    res = api.kernels_push(KERNEL_DIR)
    print(f"[{time.strftime('%H:%M:%S')}] Push response: {res}", flush=True)
    return res


def get_status(api):
    res = api.kernels_status(KERNEL_SLUG)
    return str(getattr(res, "status", ""))


def main():
    api = get_api()
    print("=" * 80)
    print("STARTING KAGGLE GPU RUNNER & MONITOR")
    print("Important: Kaggle GPU allocation requires 60-120s in QUEUED state.")
    print("Please DO NOT cancel in the web UI while queued!")
    print("=" * 80, flush=True)

    max_retries = 3
    for attempt in range(1, max_retries + 1):
        push_kernel(api)
        print(f"\n--- Monitoring Attempt {attempt}/{max_retries} ---", flush=True)

        is_running = False
        has_queued = False
        queue_count = 0

        while True:
            time.sleep(6)
            status = get_status(api)
            now_str = time.strftime('%H:%M:%S')
            print(f"[{now_str}] Status: {status}", flush=True)

            if "QUEUED" in status:
                has_queued = True
                queue_count += 1
                if queue_count % 5 == 0:
                    print(f"[{now_str}] (Still in Kaggle GPU allocation queue: {queue_count * 6}s elapsed. Please do not cancel!)", flush=True)
                continue

            if "RUNNING" in status:
                if not is_running:
                    print(f"\n{'='*80}\n[{now_str}] SUCCESS: Kernel is now actively RUNNING on remote GPU!\n{'='*80}", flush=True)
                    is_running = True
                continue

            if "COMPLETE" in status:
                if not is_running and not has_queued:
                    print(f"[{now_str}] Prior COMPLETE status detected before new job registered. Waiting...", flush=True)
                    continue
                print(f"\n{'='*80}\n[{now_str}] SUCCESS: Kernel execution COMPLETED!\n{'='*80}", flush=True)
                print(f"Downloading output to {OUTPUT_DIR}...", flush=True)
                api.kernels_output(KERNEL_SLUG, path=OUTPUT_DIR)
                print("Output download complete! Files:", os.listdir(OUTPUT_DIR), flush=True)
                return 0

            if "CANCEL" in status:
                print(f"[{now_str}] Warning: Kernel was cancelled (status: {status}).", flush=True)
                break

            if "ERROR" in status:
                print(f"[{now_str}] Kernel encountered error (status: {status}). Fetching logs...", flush=True)
                try:
                    logs = api.kernels_logs(KERNEL_SLUG)
                    print("Kernel Logs:\n", logs)
                except Exception as e:
                    print("Could not fetch logs:", e)
                return 1

        print(f"Attempt {attempt} ended without completion. Retrying in 10s...", flush=True)
        time.sleep(10)

    print("Max retries reached.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
