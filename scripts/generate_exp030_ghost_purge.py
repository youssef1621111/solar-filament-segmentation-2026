#!/usr/bin/env python3
"""
EXP-030: Surgical Ghost-FP Density Optimization
IEEE BigData 2026 Solar Filament Segmentation Challenge (filament-segmentation-2026)

Target: Break 0.58 Plateau by eliminating verified small (< 600 px) false-positive
blobs that have zero corroboration across both deep neural ensembles (WSP 5-fold and Fususu).

Key Invariants:
1. Base: EXP-024 (1,299 instances, official 0.58 anchor).
2. Zero Boundary Tampering: 100% exact bit-for-bit pixel RLEs preserved.
3. Zero Solo Additions: 0 unvetted additions to avoid multi-annotator FP explosion.
4. Stem Coverage Guard: Strictly maintains all 178 active stems (never prunes singleton stems).
5. Quiet-Sun Guard: Exactly 0 rows on solar minimum stems (20160513133714Th, 20180507140210Th).
6. Strict Disjoint Check: Strictly 0 pixel collision across all stems.
"""

import os
import sys
import numpy as np
import pandas as pd
from pycocotools import mask as mask_util

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from scripts.utils_metrics import decode_leb128_rle, encode_leb128_rle, decode_leb128_counts


def build_exp030(
    exp24_path: str = "results/operation_058_breakthrough/predictions/submission_exp024_wsp_consensus_ensemble.csv",
    corrob_path: str = "results/operation_058_breakthrough/zenith_1324_neural_corroboration.csv",
    output_path: str = "results/operation_058_breakthrough/predictions/submission_exp030_ghost_purge.csv"
):
    print("=" * 80, flush=True)
    print("BUILDING EXP-030: SURGICAL GHOST-FP DENSITY OPTIMIZATION", flush=True)
    print("=" * 80, flush=True)

    quiet_sun_stems = {"20160513133714Th", "20180507140210Th"}

    # 1. Load EXP-024 baseline and corroboration table
    df24 = pd.read_csv(exp24_path)
    corrob = pd.read_csv(corrob_path)
    merged = pd.merge(df24, corrob, on='filament_id', how='left')

    print(f"[1/5] Loaded EXP-024 Anchor: {len(df24)} instances across {df24['filament_id'].apply(lambda x: x.rsplit('_', 1)[0]).nunique()} active stems.", flush=True)

    # 2. Identify safe prune candidates:
    # - wsp_iou == 0.0 AND fus_iou == 0.0
    # - area < 600 pixels
    # - Stem has >= 2 instances (stem coverage guard)
    stems_counts = df24['filament_id'].apply(lambda x: x.rsplit('_', 1)[0]).value_counts()
    multi_instance_stems = set(stems_counts[stems_counts > 1].index)

    safe_prune_mask = (
        (merged['wsp_iou'] == 0.0) & 
        (merged['fus_iou'] == 0.0) & 
        (merged['area'] < 600) &
        (merged['stem'].isin(multi_instance_stems))
    )
    prune_cands = merged[safe_prune_mask]
    prune_ids = set(prune_cands['filament_id'].values)

    print(f"[2/5] Identified {len(prune_ids)} safe zero-support noise blobs to purge (area < 600 px):", flush=True)
    for _, row in prune_cands.iterrows():
        print(f"      - {row['filament_id']} (area={row['area']}, stem={row['stem']})", flush=True)

    # 3. Filter and re-index instances per stem
    surviving = df24[~df24['filament_id'].isin(prune_ids)].copy()
    surviving['stem'] = surviving['filament_id'].apply(lambda x: x.rsplit('_', 1)[0])

    final_records = []
    for stem, group in surviving.groupby('stem', sort=False):
        if stem in quiet_sun_stems:
            continue
        for idx, (_, row) in enumerate(group.iterrows(), start=1):
            final_records.append({
                "filament_id": f"{stem}_{idx}",
                "segmentation_rle": row["segmentation_rle"]
            })

    df30 = pd.DataFrame(final_records)

    print("\n" + "=" * 80, flush=True)
    print("EXP-030 GENERATION SUMMARY:")
    print(f"  Total Rows:               {len(df30)} (EXP-024 Baseline: {len(df24)})")
    print(f"  Purged Noise Blobs:       {len(prune_ids)}")
    print(f"  Active Stems:             {df30['filament_id'].apply(lambda x: x.rsplit('_', 1)[0]).nunique()} / 178")
    print("=" * 80, flush=True)

    # 4. Save CSV
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df30.to_csv(output_path, index=False)
    print(f"\n[4/5] Saved to: {output_path}", flush=True)

    # 5. Zero-Trust Integrity Checklist
    print("\n[5/5] Running zero-trust integrity checklist...", flush=True)
    assert len(df30) == len(df24) - len(prune_ids), "Row count mismatch!"
    stems_final = df30['filament_id'].apply(lambda x: x.rsplit('_', 1)[0])
    assert stems_final.nunique() == 178, f"Expected 178 stems, got {stems_final.nunique()}"
    assert not any(qs in stems_final.values for qs in quiet_sun_stems), "Quiet sun stem leaked!"

    # Pixel count & collision verification
    total_inst = len(df30)
    for i, r in enumerate(df30['segmentation_rle']):
        counts = decode_leb128_counts(r)
        assert sum(counts) == 4194304, f"Row {i} RLE does not sum to 4,194,304!"
    print("  [PASSED] 100% RLE counts sum to exactly 4,194,304 pixels.", flush=True)

    colliding = 0
    for stem, group in df30.groupby(stems_final):
        c = np.zeros((2048, 2048), dtype=bool)
        for r in group['segmentation_rle']:
            m = decode_leb128_rle(r)
            if np.any(c & m):
                colliding += 1
                break
            c |= m
    assert colliding == 0, f"Collisions in {colliding} stems!"
    print("  [PASSED] Exactly 0.000000 pixel collision across all stems.", flush=True)

    print("\n" + "=" * 80, flush=True)
    print("EXP-030 VERIFICATION COMPLETE! 100% READY FOR SUBMISSION!", flush=True)
    print("=" * 80, flush=True)


if __name__ == "__main__":
    build_exp030()
