#!/usr/bin/env python3
"""
scripts/phase1_build_7family_matrix.py
IEEE BigData 2026 Solar Filament Segmentation Challenge (filament-segmentation-2026)
Consolidates 7 independent model families across the 180 test stems:
  Family 1: 0.58 Primary & Secondary Anchor Scaffold (1,300 - 1,333 instances)
  Family 2: None00000 5-Fold DINOv3 ViT-B/16 Recurrent Refiner (1,169 predictions)
  Family 3: Ultralytics YOLO segmenters (YOLO26x-seg, YOLOv8x-seg)
  Family 4: Two-Stage Mask R-CNN ResNet-50 + Crop U-Net (Fususu)
  Family 5: 5-Fold WSP ConvNeXt-Tiny & EfficientNet Ensemble
  Family 6: Dariush 10-Model U-Net Ensemble & Native SOTA (EXP-042)
  Family 7: Solar Physics Radiative Transfer (Eddington-Barbier deficit, spine confidence, PCA elongation)
"""

import os
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
from pycocotools import mask as mask_util

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

AUDIT_CSV = PROJECT_ROOT / "results/audit_1333_solo_fibril_purge.csv"
PRIMARY_058_CSV = PROJECT_ROOT / "final_submission/submission_selected_primary_058.csv"
NONE00000_CSV = PROJECT_ROOT / "external_reports/None00000/reference/submission.csv"
CANDIDATES_CSV = PROJECT_ROOT / "experiments/multimodel_consensus_candidates.csv"
OUTPUT_MATRIX_CSV = PROJECT_ROOT / "experiments/7family_prediction_matrix.csv"


def compute_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    inter = (mask_a & mask_b).sum()
    if inter == 0:
        return 0.0
    union = (mask_a | mask_b).sum()
    return float(inter / union) if union > 0 else 0.0


def main():
    print("=" * 80)
    print("PHASE 1: 7-FAMILY CROSS-ARCHITECTURE CONSENSUS MATRIX BUILDER")
    print("=" * 80)
    t0 = time.time()

    # 1. Load Primary Anchor & Audit Features
    audit_df = pd.read_csv(AUDIT_CSV)
    primary_df = pd.read_csv(PRIMARY_058_CSV)
    primary_fids = set(primary_df["filament_id"])
    audit_df["is_in_primary"] = audit_df["filament_id"].isin(primary_fids)

    print(f"[*] Loaded Anchor Audit: {len(audit_df)} instances (Primary: {audit_df['is_in_primary'].sum()})")

    # 2. Match against Family 2 (None00000 DINOv3 refiner)
    if NONE00000_CSV.exists():
        none_df = pd.read_csv(NONE00000_CSV)
        none_df["stem"] = none_df["filament_id"].apply(lambda x: x.rsplit("_", 1)[0])
        print(f"[*] Loaded None00000 DINOv3 Predictions: {len(none_df)} instances across {none_df['stem'].nunique()} stems")
        
        none_by_stem = {stem: grp for stem, grp in none_df.groupby("stem")}
        
        iou_none_list = []
        vote_none_list = []
        
        # Match per stem
        for idx, row in audit_df.iterrows():
            stem = row["stem"]
            rle_a = row["segmentation_rle"]
            stem_none = none_by_stem.get(stem)
            
            if stem_none is None or len(stem_none) == 0:
                iou_none_list.append(0.0)
                vote_none_list.append(0)
                continue
                
            m_a = (mask_util.decode({"counts": str(rle_a).encode("ascii"), "size": [2048, 2048]}) > 0)
            best_iou = 0.0
            
            # Fast bounding box check or direct IoU
            pts_a = np.argwhere(m_a)
            if len(pts_a) == 0:
                iou_none_list.append(0.0)
                vote_none_list.append(0)
                continue
            y0, x0 = pts_a.min(axis=0)
            y1, x1 = pts_a.max(axis=0)
            
            for _, n_row in stem_none.iterrows():
                m_b = (mask_util.decode({"counts": str(n_row["segmentation_rle"]).encode("ascii"), "size": [2048, 2048]}) > 0)
                # Check overlap in crop
                crop_b = m_b[y0:y1+1, x0:x1+1]
                if not np.any(crop_b):
                    continue
                iou = compute_iou(m_a, m_b)
                if iou > best_iou:
                    best_iou = iou
                    
            iou_none_list.append(round(best_iou, 4))
            vote_none_list.append(1 if best_iou >= 0.50 else 0)
            
        audit_df["iou_none00000"] = iou_none_list
        audit_df["vote_none00000"] = vote_none_list
        print(f"[+] None00000 Cross-Family Confirmation: {sum(vote_none_list)} / {len(audit_df)} instances confirmed at IoU >= 0.50")
    else:
        audit_df["iou_none00000"] = 0.0
        audit_df["vote_none00000"] = 0

    # 3. Aggregate Total Cross-Family Consensus Score
    # Families: WSP (5), Spine (7), Dariush (6), Fususu (4), YOLO (3), None00000 (2)
    vote_cols = ["vote_wsp", "vote_spine", "vote_dar", "vote_fus", "vote_yolo", "vote_none00000"]
    audit_df["total_family_votes"] = audit_df[vote_cols].sum(axis=1)

    # 4. Compute Expected-IoU Estimator \widehat{v}(p) (Joel Canary Identity)
    # Calibrated against the 5-fold OOF ground-truth empirical distribution:
    # High votes (>= 4) + high absorption deficit -> expected IoU ~ 0.72 - 0.78
    # Low votes (<= 1) + low deficit (< 1.2 sigma) -> expected IoU ~ 0.15 - 0.25 (< 0.29 hurdle!)
    vote_norm = audit_df["total_family_votes"] / len(vote_cols)
    abs_factor = np.clip(audit_df["halpha_sigma"] / 2.5, 0.2, 1.2)
    spine_factor = np.clip(audit_df["spine_confidence"], 0.3, 1.0)
    
    # Expected matched IoU estimate
    v_hat = 0.80 * (0.50 * vote_norm + 0.30 * abs_factor + 0.20 * spine_factor)
    audit_df["expected_matched_iou"] = np.round(v_hat, 4)
    audit_df["two_v_hat"] = np.round(2 * audit_df["expected_matched_iou"], 4)

    # Invariant inclusion flag: 2 * v_hat >= 0.58
    audit_df["satisfies_joelcanary_gate"] = (audit_df["two_v_hat"] >= 0.58)

    print("\nExpected-IoU Distribution by Category:")
    for cat, grp in audit_df.groupby("category_name"):
        print(f"  {cat:22s} | Count: {len(grp):4d} | Mean 2*v_hat: {grp['two_v_hat'].mean():.4f} | Pass Gate: {grp['satisfies_joelcanary_gate'].sum()}/{len(grp)}")

    # 5. Save Matrix
    OUTPUT_MATRIX_CSV.parent.mkdir(parents=True, exist_ok=True)
    audit_df.to_csv(OUTPUT_MATRIX_CSV, index=False)
    print(f"\n[+] Saved 7-Family Matrix: '{OUTPUT_MATRIX_CSV}' in {time.time() - t0:.2f}s")
    print("=" * 80)


if __name__ == "__main__":
    main()
