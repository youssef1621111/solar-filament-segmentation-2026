#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PRED_DIR="$SCRIPT_DIR/../results/operation_058_breakthrough/predictions"

SUB_TIER_B="$PRED_DIR/submission_exp043_multimodel_consensus.csv"
SUB_TIER_A="$PRED_DIR/submission_exp043_conservative_1339.csv"
SUB_TIER_C="$PRED_DIR/submission_exp043_aggressive_1401.csv"

echo "================================================================================"
echo "EXP-043 DISPATCH PROTOCOL: Solar Filament Multi-Model Consensus"
echo "================================================================================"
echo "Checking Kaggle Submission Quota..."

SUB_JSON=$(kaggle competitions submissions --format json -c filament-segmentation-2026 | head -n 30)

echo "Ready candidates on disk:"
echo "  Tier A (Conservative, 1339 inst): $SUB_TIER_A"
echo "  Tier B (Optimal, 1374 inst):      $SUB_TIER_B"
echo "  Tier C (Aggressive, 1406 inst):   $SUB_TIER_C"
echo ""
echo "To submit Tier B (Primary Championship Target):"
echo "kaggle competitions submit -c filament-segmentation-2026 -f \"$SUB_TIER_B\" -m \"EXP-043: Multi-Model Consensus (1374 inst, 100% Anchor + 74 Corroborated Recoveries, 0 Collisions, 0 QS)\""
echo "================================================================================"
