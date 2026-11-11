#!/bin/bash
# Autonomous submit script for EXP-032 at 00:00 UTC Reset
TARGET_FILE="/home/fathyyoussef162/submission_output/submission.csv"
MSG="EXP-032: Titan Consensus Breakthrough (1312 inst, 100% 0.58 anchor preserved + 12 giant dark ribbons, Area>=1000px, Deficit>=13.0, 0 collisions)"

echo "[$(date -u)] Starting submission monitor. Target reset: 00:00 UTC..."

while true; do
    UTC_TIME=$(date -u +%H%M)
    if [ "$UTC_TIME" -ge "0000" ] && [ "$UTC_TIME" -le "0030" ]; then
        echo "[$(date -u)] Within reset window ($UTC_TIME). Attempting submission..."
        python3 /home/fathyyoussef162/scripts/dispatch_leaderboard_controller.py
        if [ $? -eq 0 ]; then
            echo "[$(date -u)] Dispatch controller executed successfully!"
            break
        else
            echo "[$(date -u)] Submission attempt failed or quota not yet reset. Retrying in 60s..."
            sleep 60
        fi
    else
        echo "[$(date -u)] Current time $UTC_TIME UTC. Waiting for 00:00 UTC reset..."
        sleep 600
    fi
done
