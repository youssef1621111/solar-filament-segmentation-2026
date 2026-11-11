#!/usr/bin/env bash
# ==============================================================================
# EXP-039 Championship Automated Submission Dispatcher
# IEEE BigData 2026 Solar Filament Segmentation Challenge
# Target: 0.66+ Public Leaderboard Score
# ==============================================================================

set -euo pipefail

SUB_CSV="/home/fathyyoussef162/results/operation_058_breakthrough/predictions/submission_exp039_topological_consensus_1333.csv"
MSG="EXP-039: Topological Centerline Consensus Pipeline (SoftCLDice + Multi-Annotator Consensus Supervision + 100% Anchor Preserved)"

echo "================================================================================"
echo "DISPATCHING EXP-039 CHAMPIONSHIP SUBMISSION"
echo "Target file: $SUB_CSV"
echo "Message: $MSG"
echo "================================================================================"

# 1. Pre-flight 6-point Zero-Trust check
echo "[Step 1/3] Running 6-Point Zero-Trust Submission Integrity Audit..."
python3 /home/fathyyoussef162/scripts/submission_integrity_check.py \
    --sub_csv "$SUB_CSV"

# 2. Dispatch to Kaggle CLI
echo "[Step 2/3] Submitting to Kaggle Competition Leaderboard..."
kaggle competitions submit -c filament-segmentation-2026 -f "$SUB_CSV" -m "$MSG"

# 3. Poll status
echo "[Step 3/3] Checking submission queue status..."
kaggle competitions submissions -c filament-segmentation-2026 | head -n 5

echo "================================================================================"
echo "EXP-039 DISPATCH COMPLETED SUCCESSFULLY!"
echo "================================================================================"
