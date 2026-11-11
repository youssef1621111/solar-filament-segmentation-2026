#!/bin/bash
# Status Dashboard for Operation 0.58 Breakthrough
echo "================================================================================"
echo "          IEEE BigData 2026: Solar Filament Challenge Status Dashboard"
echo "================================================================================"

echo -n "Current UTC Time:  " && date -u
python3 -c "
from datetime import datetime, timezone
now = datetime.now(timezone.utc)
target = datetime(now.year, now.month, now.day + 1, 0, 0, 0, tzinfo=timezone.utc)
diff = target - now
h = int(diff.total_seconds() // 3600)
m = int((diff.total_seconds() % 3600) // 60)
print(f'Quota Reset:       00:00 UTC (in {h}h {m}m)')
"

echo -e "\n--- DISK USAGE ---"
df -h /home | awk 'NR==1 || NR==2'

echo -e "\n--- STAGED SUBMISSION ---"
if [ -f "/home/fathyyoussef162/submission_output/submission.csv" ]; then
    ROWS=$(wc -l < /home/fathyyoussef162/submission_output/submission.csv)
    MD5=$(md5sum /home/fathyyoussef162/submission_output/submission.csv | awk '{print $1}')
    echo "File:              /home/fathyyoussef162/submission_output/submission.csv"
    echo "Rows:              $((ROWS - 1)) instances"
    echo "MD5:               $MD5"
else
    echo "ERROR: /home/fathyyoussef162/submission_output/submission.csv not found!"
fi

echo -e "\n--- RECENT KAGGLE SUBMISSIONS ---"
kaggle competitions submissions -c filament-segmentation-2026 | head -n 4

echo -e "\n--- STANDING LEADERBOARD POSITION ---"
kaggle competitions leaderboard -c filament-segmentation-2026 --show | grep -E "cracker_1_war|IONode Online|Sayandeep|Doge" || true

echo "================================================================================"
