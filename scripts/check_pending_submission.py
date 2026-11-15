import os
import time
import subprocess
import sys

print("Starting submission status monitor for Ref 58358542...", flush=True)
env = os.environ.copy()
env["KAGGLE_API_TOKEN"] = "KGAT_a1b5b28617f853e558952c9bec22f8af"

for i in range(25):
    res = subprocess.run(
        ["kaggle", "competitions", "submissions", "-c", "filament-segmentation-2026"],
        env=env,
        capture_output=True,
        text=True
    )
    lines = res.stdout.splitlines()
    target = [l for l in lines if "58358542" in l]
    if target:
        line = target[0]
        print(f"[{time.strftime('%X')}] Check {i+1}: {line}", flush=True)
        if "COMPLETE" in line or "ERROR" in line:
            print("FINISHED EVALUATION RESULT:", line, flush=True)
            sys.exit(0)
    else:
        print(f"[{time.strftime('%X')}] Check {i+1}: Ref not found yet", flush=True)
    time.sleep(15)
