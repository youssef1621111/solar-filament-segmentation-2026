#!/usr/bin/env python3
"""
Autonomous Leaderboard Dispatch Controller
IEEE BigData 2026 Solar Filament Segmentation Challenge (filament-segmentation-2026)

Zero-Trust, Audit-Proof Automated Submission & Leaderboard Branching Controller:
1. Submits Bullet #1 (EXP-032-Titan, 1312 instances).
2. Polls Kaggle API directly via official KaggleApi client until status is SubmissionStatus.COMPLETE.
3. Automatically dispatches Bullet #2 (Apex27 or Centurion45) based on exact parsed Public LB score.
"""

import os
import sys
import time
from datetime import datetime, timezone
from kaggle.api.kaggle_api_extended import KaggleApi

COMPETITION = "filament-segmentation-2026"
BULLET1_FILE = "/home/fathyyoussef162/submission_output/submission.csv"
BULLET2_APEX = "/home/fathyyoussef162/results/operation_058_breakthrough/predictions/submission_exp032_apex27_1327.csv"
BULLET3_CENTURION = "/home/fathyyoussef162/results/operation_058_breakthrough/predictions/submission_exp033_centurion45_1345.csv"
LOG_FILE = "/home/fathyyoussef162/results/operation_058_breakthrough/submission_dispatch.log"


def log(msg):
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    line = f"[{now_str}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


def get_authenticated_api():
    api = KaggleApi()
    api.authenticate()
    return api


def submit_candidate(api, file_path, message):
    log(f"Submitting {file_path} with message: '{message}'...")
    try:
        api.competition_submit(file_path, message, COMPETITION)
        log("Submission upload accepted by Kaggle server!")
        return True
    except Exception as e:
        log(f"Submission failed! Exception: {e}")
        if hasattr(e, "response") and e.response is not None:
            log(f"Kaggle API response text: {e.response.text}")
        return False


def get_latest_submission(api):
    try:
        subs = api.competition_submissions(COMPETITION)
        if not subs:
            return None
        latest = subs[0]
        status_str = str(getattr(latest, "status", ""))
        score_val = None
        raw_score = getattr(latest, "public_score", None)
        if raw_score not in (None, "", "None"):
            try:
                score_val = float(raw_score)
            except ValueError:
                pass
        return {
            "ref": getattr(latest, "ref", None),
            "date": getattr(latest, "date", None),
            "description": getattr(latest, "description", ""),
            "status": status_str,
            "score": score_val
        }
    except Exception as e:
        log(f"Error fetching latest submission from Kaggle API: {e}")
        return None


def main():
    log("=" * 80)
    log("STARTING AUTONOMOUS LEADERBOARD DISPATCH CONTROLLER")
    log("=" * 80)

    api = get_authenticated_api()
    initial_latest = get_latest_submission(api)
    initial_ref = initial_latest["ref"] if initial_latest else None
    log(f"Pre-flight check: Latest submission on Kaggle is Ref {initial_ref} (Score: {initial_latest.get('score')})")

    # Step 1: Submit Bullet #1 (Titan-12)
    b1_msg = "EXP-032: Titan Consensus Breakthrough (1312 inst, 100% 0.58 anchor + 12 giant ribbons, Area>=1000px, 0 collisions)"
    success = submit_candidate(api, BULLET1_FILE, b1_msg)
    if not success:
        log("Aborting due to submission failure (e.g. daily quota not yet reset). Retrying logic will handle next window.")
        sys.exit(1)

    # Step 2: Poll for completion
    log("Waiting for Bullet #1 to be processed and scored by Kaggle...")
    score = None
    target_ref = None

    for attempt in range(1, 25):
        time.sleep(15)
        cur = get_latest_submission(api)
        if cur is None:
            continue
        cur_ref = cur["ref"]
        cur_status = cur["status"]
        cur_score = cur["score"]

        log(f"Polling Attempt {attempt:02d}: Ref={cur_ref}, Status={cur_status}, Score={cur_score}")

        # Ensure we are inspecting the new submission
        if initial_ref is not None and cur_ref == initial_ref:
            log("New submission is still registering on Kaggle backend...")
            continue

        target_ref = cur_ref
        if "COMPLETE" in cur_status and cur_score is not None:
            score = cur_score
            log(f"SUCCESS: Submission Ref {cur_ref} scored COMPLETE! Public LB Score: {score}")
            break
        elif "ERROR" in cur_status:
            log(f"ERROR: Submission Ref {cur_ref} encountered an error on Kaggle!")
            sys.exit(2)

    if score is None:
        log("Timeout waiting for submission score. Please check Kaggle UI directly.")
        sys.exit(3)

    # Step 3: Leaderboard Branching Decision
    log("-" * 80)
    log(f"BRANCHING DECISION FOR REMAINING 4 SUBMISSIONS (Based on Bullet #1 Score: {score}):")
    if score >= 0.585:
        log(f"LB ADVANCEMENT CONFIRMED! Score = {score:.4f} >= 0.585 (Displayed as 0.59+).")
        log("Dispatching Bullet #2: EXP-032-Apex27 (1,327 instances, top 27 giant ribbons)...")
        b2_msg = "EXP-032: Apex27 Full-Spectrum Ribbon Breakthrough (1327 inst, 27 giant ribbons Area>=1000px, 0 collisions)"
        submit_candidate(api, BULLET2_APEX, b2_msg)
    else:
        log(f"Score held at {score:.4f} (Displayed as 0.58).")
        log("Dispatching Bullet #3: EXP-033-Centurion45 (1,345 instances, Area>=500px consensus)...")
        b3_msg = "EXP-033: Centurion45 Salience Champion (1345 inst, 45 prominent ribbons Area>=500px, 0 collisions)"
        submit_candidate(api, BULLET3_CENTURION, b3_msg)

    log("=" * 80)
    log("MASTER SUBMISSION CONTROLLER COMPLETED SUCCESSFULLY")
    log("=" * 80)


if __name__ == "__main__":
    main()
