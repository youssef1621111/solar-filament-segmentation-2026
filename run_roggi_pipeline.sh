#!/usr/bin/env bash
set -e

echo "================================================================================"
echo "ROGGI Physics-Guided & Zero-Trust Pipeline (0.55 -> 0.60+ LB)"
echo "IEEE BigData 2026 Cup — Solar Filament Segmentation Challenge"
echo "================================================================================"

# Step 0: Zero-Trust Fold-0 Diagnostic Audit
echo -e "\n>>> Executing Step 0: Fold-0 Diagnostic Audit..."
python3 scripts/step0_fold0_solo_truth_audit.py

# Stage 1: Dual-Routed Multi-Channel Solo Acceptance Gate
echo -e "\n>>> Executing Stage 1: Multi-Channel Solo Acceptance & Routing..."
python3 scripts/stage1_roggi_multichannel_gate.py

# Stage 2: 2D Ribbon Alignment Matrix & FFBSi Particle Bridge
echo -e "\n>>> Executing Stage 2: FFBSi Particle Bridge (Healing 1 -> M Splits)..."
python3 scripts/stage2_roggi_ffbsi_bridge.py

# Stage 3: Physics Gap-Penalty Boundary Refinement
echo -e "\n>>> Executing Stage 3: Physics Gap-Penalty Boundary Refiner (SQ Boost)..."
python3 scripts/stage3_physics_boundary_refiner.py

# Stage 4: 5-Channel ConvNeXt / U-Net Refiner & Synthetic Fault Loop
echo -e "\n>>> Executing Stage 4: 5-Channel Tensor & Synthetic Fault Loop..."
python3 scripts/stage4_train_roggi_multichannel_unet.py

# Submission Integrity Checklist
echo -e "\n>>> Executing 6-Point Submission Integrity Verification..."
python3 scripts/submission_integrity_check.py --allow_missing_anchor

echo "================================================================================"
echo "[+] ROGGI Zero-Trust Pipeline successfully completed!"
echo "================================================================================"
