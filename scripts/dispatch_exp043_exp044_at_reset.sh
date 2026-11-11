#!/usr/bin/env bash
# ==============================================================================
# EXP-043 & EXP-044 Automated Leaderboard Reset Dispatcher
# IEEE BigData 2026 Solar Filament Segmentation Challenge (filament-segmentation-2026)
# Target: 0.62 - 0.66+ Panoptic Quality Breakthrough
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

EXP043_CSV="$ROOT_DIR/final_submission/submission_exp043_consensus_purged_tight.csv"
EXP044_CSV="$ROOT_DIR/final_submission/submission_exp044_strict_majority_1040.csv"

echo "================================================================================"
echo "CHAMPIONSHIP SUBMISSION DISPATCH PROTOCOL: EXP-043 & EXP-044"
echo "================================================================================"

if [ ! -f "$EXP043_CSV" ]; then
    echo "[!] Missing EXP-043 CSV: $EXP043_CSV"
    exit 1
fi

if [ ! -f "$EXP044_CSV" ]; then
    echo "[!] Missing EXP-044 CSV: $EXP044_CSV"
    exit 1
fi

echo "[Step 1/4] Running Pre-Flight 6-Point Zero-Trust Submission Integrity Audit..."
echo "--- Verifying EXP-043 (1,158 instances) ---"
python3 "$SCRIPT_DIR/submission_integrity_check.py" --sub_csv "$EXP043_CSV"

echo ""
echo "--- Verifying EXP-044 (1,040 instances) ---"
python3 "$SCRIPT_DIR/submission_integrity_check.py" --sub_csv "$EXP044_CSV"

echo ""
echo "[Step 2/4] Checking Current Kaggle Daily Quota..."
kaggle competitions submissions -c filament-segmentation-2026 | head -n 10

echo ""
echo "[Step 3/4] Ready for Dispatch at 00:00:00 UTC Reset:"
echo "Command 1 (EXP-043 Primary Breakthrough):"
echo "  kaggle competitions submit -c filament-segmentation-2026 -f \"$EXP043_CSV\" -m \"EXP-043: Consensus Purged & Spine-Protected Penumbral Peeling (1158 inst, 0 Collisions, 0 QS)\""
echo ""
echo "Command 2 (EXP-044 Strict Majority Peak):"
echo "  kaggle competitions submit -c filament-segmentation-2026 -f \"$EXP044_CSV\" -m \"EXP-044: Pure 3/3 + 2/3 High-Precision Consensus Core (1040 inst, 0 Collisions, 0 QS)\""

echo ""
echo "[Step 4/4] Attempting Dispatch Now (if quota available):"
set +e
kaggle competitions submit -c filament-segmentation-2026 -f "$EXP043_CSV" -m "EXP-043: Consensus Purged & Spine-Protected Penumbral Peeling (1158 inst, 0 Collisions, 0 QS)"
STATUS_43=$?
if [ $STATUS_43 -eq 0 ]; then
    echo "[SUCCESS] EXP-043 submitted successfully!"
    kaggle competitions submit -c filament-segmentation-2026 -f "$EXP044_CSV" -m "EXP-044: Pure 3/3 + 2/3 High-Precision Consensus Core (1040 inst, 0 Collisions, 0 QS)"
    STATUS_44=$?
    if [ $STATUS_44 -eq 0 ]; then
        echo "[SUCCESS] EXP-044 submitted successfully!"
    fi
else
    echo "[INFO] Daily 5/5 submission limit active for Nov 11 UTC. Script is staged and ready to fire at 00:00:00 UTC reset."
fi

echo "================================================================================"
