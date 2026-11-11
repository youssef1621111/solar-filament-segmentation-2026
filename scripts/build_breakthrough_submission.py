#!/usr/bin/env python3
"""
scripts/build_breakthrough_submission.py
Operation 0.58 Breakthrough Championship Candidate Generator
IEEE BigData 2026 Cup — Solar Filament Segmentation

Generates:
    results/operation_058_breakthrough/predictions/submission_breakthrough_sota.csv

Fuses the verified 0.58 Clean Anchor (1,324 instances, 178 active stems, 0 quiet-sun rows)
with high-confidence disjoint candidate filaments extracted from the 5-fold ConvNeXt-Tiny +
WSP EfficientNet-B0 ensemble pool (submission_wsp_championship.csv).

Enforces:
    1. Zero Mask Shrinkage (Anchor masks are 100% immutable Tier 1).
    2. Strict Disjoint Stamping (Zero pixel collisions: forall i != j, M_i cap M_j = emptyset).
    3. Physics & Morphology Gating (Elongation >= 2.2, Circularity <= 0.35, Sunspot Veto).
    4. Bit-Exact COCO LEB128 Fortran-order RLE encoding.
"""

import os
import sys
import time
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils_metrics import decode_leb128_rle, encode_leb128_rle, compute_mask_bbox
from utils_solar_physics import compute_mask_shape_features

ANCHOR_CSV = "/home/fathyyoussef162/results/weight_space_programming/validation_results/submission_clean_zenith_1324.csv"
WSP_POOL_CSV = "/home/fathyyoussef162/results/weight_space_programming/validation_results/submission_wsp_championship.csv"
OUTPUT_DIR = Path("/home/fathyyoussef162/results/operation_058_breakthrough/predictions")
OUTPUT_CSV = OUTPUT_DIR / "submission_breakthrough_sota.csv"

QUIET_SUN_STEMS = {"20160513133714Th", "20180507140210Th"}

MIN_AREA = 400
MAX_AREA = 15000
MIN_ELONGATION = 2.20
MAX_CIRCULARITY = 0.35


def main():
    t0 = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70, flush=True)
    print("OPERATION 0.58 BREAKTHROUGH: SOTA CANDIDATE GENERATOR", flush=True)
    print("=" * 70, flush=True)

    print(f"Loading 0.58 Clean Anchor from: {ANCHOR_CSV}", flush=True)
    df_anchor = pd.read_csv(ANCHOR_CSV)
    df_anchor["stem"] = df_anchor["filament_id"].apply(lambda x: x.split("_")[0])

    print(f"Loading WSP Ensemble Pool from: {WSP_POOL_CSV}", flush=True)
    df_wsp = pd.read_csv(WSP_POOL_CSV)
    df_wsp["stem"] = df_wsp["filament_id"].apply(lambda x: x.split("_")[0])

    # All active stems present in the clean anchor
    all_anchor_stems = sorted(list(df_anchor["stem"].unique()))
    print(f"Total active observation stems in anchor: {len(all_anchor_stems)}", flush=True)

    # Pre-group WSP instances by stem
    wsp_by_stem = {stem: grp for stem, grp in df_wsp.groupby("stem")}

    final_records = []
    admitted_solos_meta = []
    total_anchor_instances = 0
    total_admitted_solos = 0

    for idx, stem in enumerate(all_anchor_stems):
        if stem in QUIET_SUN_STEMS:
            print(f"Skipping Quiet-Sun stem: {stem} (0 rows enforced)", flush=True)
            continue

        sub_a = df_anchor[df_anchor["stem"] == stem]
        sub_w = wsp_by_stem.get(stem, pd.DataFrame())

        # If no candidates exist in WSP for this stem, preserve anchor rows verbatim
        if len(sub_w) == 0:
            for _, row in sub_a.iterrows():
                final_records.append({
                    "filament_id": row["filament_id"],
                    "segmentation_rle": row["segmentation_rle"],
                })
                total_anchor_instances += 1
            continue

        # 1. Tier 1: Decode anchor masks and build aggregate canvas
        anchor_canvas = np.zeros((2048, 2048), dtype=bool)
        stem_anchor_masks = []
        for _, row in sub_a.iterrows():
            m = decode_leb128_rle(row["segmentation_rle"])
            stem_anchor_masks.append(m)
            anchor_canvas |= m
            total_anchor_instances += 1

        # 2. Tier 2: Extract candidate solos from WSP ensemble pool
        candidate_solos = []
        for _, row in sub_w.iterrows():
            mw = decode_leb128_rle(row["segmentation_rle"])
            # Fast check: must have zero intersection with anchor canvas
            if (mw & anchor_canvas).sum() == 0:
                area = int(mw.sum())
                if MIN_AREA <= area <= MAX_AREA:
                    y_min, y_max, x_min, x_max = compute_mask_bbox(mw)
                    crop = mw[y_min : y_max + 1, x_min : x_max + 1]
                    elong, circ, _, veto = compute_mask_shape_features(crop)
                    if not veto and elong >= MIN_ELONGATION and circ <= MAX_CIRCULARITY:
                        candidate_solos.append({
                            "orig_id": row["filament_id"],
                            "stem": stem,
                            "area": area,
                            "elongation": elong,
                            "circularity": circ,
                            "mask": mw,
                        })

        # Sort candidate solos by ribbon quality (Elongation * sqrt(Area))
        candidate_solos.sort(key=lambda c: c["elongation"] * np.sqrt(c["area"]), reverse=True)

        # 3. Stamp candidate solos onto canvas ensuring mutual disjointness
        current_canvas = anchor_canvas.copy()
        stem_admitted_masks = []
        for cand in candidate_solos:
            mw = cand["mask"]
            if (mw & current_canvas).sum() == 0:
                current_canvas |= mw
                stem_admitted_masks.append(mw)
                total_admitted_solos += 1
                admitted_solos_meta.append({
                    "stem": stem,
                    "area": cand["area"],
                    "elongation": cand["elongation"],
                    "circularity": cand["circularity"],
                })

        # 4. If no solos admitted, preserve original anchor RLEs (zero re-encode error)
        if len(stem_admitted_masks) == 0:
            for _, row in sub_a.iterrows():
                final_records.append({
                    "filament_id": row["filament_id"],
                    "segmentation_rle": row["segmentation_rle"],
                })
        else:
            # Anchor masks first
            next_idx = 1
            for m in stem_anchor_masks:
                final_records.append({
                    "filament_id": f"{stem}_{next_idx}",
                    "segmentation_rle": encode_leb128_rle(m),
                })
                next_idx += 1
            # Admitted solos next
            for m in stem_admitted_masks:
                final_records.append({
                    "filament_id": f"{stem}_{next_idx}",
                    "segmentation_rle": encode_leb128_rle(m),
                })
                next_idx += 1

        if (idx + 1) % 40 == 0 or (idx + 1) == len(all_anchor_stems):
            print(f"Processed {idx + 1}/{len(all_anchor_stems)} stems... (Solos admitted so far: {total_admitted_solos})", flush=True)

    df_out = pd.DataFrame(final_records)
    df_out.to_csv(OUTPUT_CSV, index=False)

    # Compute SHA256
    with open(OUTPUT_CSV, "rb") as f:
        file_bytes = f.read()
        sha256_hash = hashlib.sha256(file_bytes).hexdigest()

    print("\n" + "=" * 70, flush=True)
    print("SUBMISSION CANDIDATE GENERATION SUMMARY", flush=True)
    print("=" * 70, flush=True)
    print(f"Output File: {OUTPUT_CSV}", flush=True)
    print(f"File Size: {len(file_bytes):,} bytes", flush=True)
    print(f"Total Rows (Instances): {len(df_out):,}", flush=True)
    print(f"Anchor Instances Preserved: {total_anchor_instances:,}", flush=True)
    print(f"Physics-Gated Solos Admitted: {total_admitted_solos:,}", flush=True)
    n_stems = df_out["filament_id"].apply(lambda x: x.split("_")[0]).nunique()
    print(f"Active Stems: {n_stems}", flush=True)
    print(f"SHA-256 Checksum: {sha256_hash}", flush=True)

    if total_admitted_solos > 0:
        solos_df = pd.DataFrame(admitted_solos_meta)
        print(f"Admitted Solos Mean Area: {solos_df['area'].mean():.1f} px (median {solos_df['area'].median():.1f})", flush=True)
        print(f"Admitted Solos Mean Elongation: {solos_df['elongation'].mean():.2f}", flush=True)
        print(f"Admitted Solos Mean Circularity: {solos_df['circularity'].mean():.2f}", flush=True)

    print(f"\nExecution Time: {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
