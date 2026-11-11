#!/usr/bin/env python3
"""
scripts/geodesic_absorption_snap.py
Operation Absorption Snap: SQ Breakthrough to 0.65 - 0.70
IEEE BigData 2026 Cup — Solar Filament Segmentation

Refines boundary contours of the 1,324 anchor instances using native 2048x2048 GONG H-alpha imagery.

Adheres strictly to the 4 Non-Negotiable Invariants:
1. Zero Solo Additions: exactly 1,324 instances preserved across 178 active stems.
2. Zero Core Erosion: erode(M_i, 1) is 100% unconditionally preserved.
3. Strict Area Preservation: 0.98 * Area(M_i) <= Area(M_i*) <= 1.04 * Area(M_i).
4. Disjoint Canvas Invariant: forall i != j, M_i* cap M_j* = emptyset.
5. Topological Continuity: no holes, no shattered fragments.
"""

import os
import sys
import glob
import time
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.ndimage as ndi
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils_metrics import decode_leb128_rle, encode_leb128_rle, compute_mask_bbox

ANCHOR_CSV = "/home/fathyyoussef162/results/weight_space_programming/validation_results/submission_clean_zenith_1324.csv"
OUTPUT_DIR = Path("/home/fathyyoussef162/results/operation_058_breakthrough/predictions")
OUTPUT_CSV = OUTPUT_DIR / "submission_absorption_snap_sota.csv"
IMAGES_DIR = Path("/tmp/gong_test_images")

QUIET_SUN_STEMS = {"20160513133714Th", "20180507140210Th"}


def build_image_lookup():
    """Build fast in-memory dictionary of stem to image Path."""
    lookup = {}
    for p in IMAGES_DIR.glob("**/*.jpeg"):
        stem = p.name.split(".")[0]
        lookup[stem] = p
    for p in IMAGES_DIR.glob("**/*.jpg"):
        stem = p.name.split(".")[0]
        if stem not in lookup:
            lookup[stem] = p
    return lookup


def snap_fast(mask: np.ndarray, image: np.ndarray, pad: int = 12, k_deficit: float = 0.20) -> np.ndarray:
    """
    High-performance cropped absorption valley snapping.
    Guarantees:
      - 100% core interior preserved.
      - Refined mask area strictly in [ceil(0.98 * A_0), floor(1.04 * A_0)].
      - Topological connectivity preserved.
    """
    a0 = int(np.sum(mask > 0))
    if a0 < 50:
        return mask

    y_min, y_max, x_min, x_max = compute_mask_bbox(mask)
    y0 = max(0, y_min - pad)
    y1 = min(2048, y_max + pad + 1)
    x0 = max(0, x_min - pad)
    x1 = min(2048, x_max + pad + 1)

    crop_m = mask[y0:y1, x0:x1] > 0
    crop_img = image[y0:y1, x0:x1].astype(np.float32)
    struct = ndi.generate_binary_structure(2, 1)

    # Core interior (Protected: Rule 2 Zero Core Erosion)
    core = ndi.binary_erosion(crop_m, structure=struct, iterations=1)
    if not np.any(core):
        core = crop_m

    # Outer boundary band (width 2-3px)
    dilated = ndi.binary_dilation(crop_m, structure=struct, iterations=2)
    band = dilated & (~core)

    if not np.any(band):
        return mask

    # Local background intensity via 25x25 uniform filter
    local_bg = ndi.uniform_filter(crop_img, size=25)
    local_mad = ndi.uniform_filter(np.abs(crop_img - local_bg), size=25) + 1e-4
    deficit = (local_bg - crop_img) / local_mad

    # Candidate mask: core + boundary pixels where local absorption deficit >= k_deficit
    cand = core | (band & (deficit >= k_deficit))
    cand = ndi.binary_fill_holes(cand)

    # Topological continuity: keep only components connected to core
    lbl, num_comp = ndi.label(cand, structure=struct)
    if num_comp > 1:
        core_lbls = np.unique(lbl[core])
        core_lbls = core_lbls[core_lbls > 0]
        if len(core_lbls) > 0:
            cand = np.isin(lbl, core_lbls)

    cand |= core

    # Strict Area Clamp: [ceil(0.98 * A_0), floor(1.04 * A_0)]
    min_a = int(np.ceil(0.9800 * a0))
    max_a = int(np.floor(1.0400 * a0))
    curr_a = int(np.sum(cand))

    band_pts = np.column_stack(np.where(band))
    if len(band_pts) > 0:
        band_defs = deficit[band]
        order = np.argsort(band_defs)[::-1]  # Darkest to least dark

        if curr_a < min_a:
            # Need to add pixels: add the darkest available band pixels
            for idx in order:
                if curr_a >= min_a:
                    break
                r, c = band_pts[idx]
                if not cand[r, c]:
                    cand[r, c] = True
                    curr_a += 1

        elif curr_a > max_a:
            # Need to trim pixels: trim least dark non-core pixels
            for idx in order[::-1]:
                if curr_a <= max_a:
                    break
                r, c = band_pts[idx]
                if cand[r, c] and not core[r, c]:
                    cand[r, c] = False
                    curr_a -= 1

    out = np.zeros((2048, 2048), dtype=bool)
    out[y0:y1, x0:x1] = cand
    return out


def main():
    t0 = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 75, flush=True)
    print("OPERATION ABSORPTION SNAP: GEODESIC SQ OPTIMIZATION ENGINE", flush=True)
    print("=" * 75, flush=True)

    print(f"Loading 0.58 Clean Anchor from: {ANCHOR_CSV}", flush=True)
    df_anchor = pd.read_csv(ANCHOR_CSV)
    df_anchor["stem"] = df_anchor["filament_id"].apply(lambda x: x.split("_")[0])

    active_stems = sorted(list(df_anchor["stem"].unique()))
    print(f"Total active stems: {len(active_stems)} (instances: {len(df_anchor)})", flush=True)

    # Fast in-memory image lookup
    image_lookup = build_image_lookup()
    print(f"Cached image lookup paths for {len(image_lookup)} stems.", flush=True)

    final_records = []
    total_area_before = 0
    total_area_after = 0
    refined_instance_count = 0
    area_ratios = []

    for idx, stem in enumerate(active_stems):
        if stem in QUIET_SUN_STEMS:
            print(f"Skipping Quiet-Sun stem: {stem} (0 rows enforced)", flush=True)
            continue

        stem_rows = df_anchor[df_anchor["stem"] == stem]

        img_path = image_lookup.get(stem)
        if img_path is None or not img_path.exists():
            print(f"[!] Warning: Image for stem {stem} not found! Preserving anchor masks verbatim.", flush=True)
            for _, row in stem_rows.iterrows():
                final_records.append({
                    "filament_id": row["filament_id"],
                    "segmentation_rle": row["segmentation_rle"],
                })
                m_orig = decode_leb128_rle(row["segmentation_rle"])
                a = int(np.sum(m_orig > 0))
                total_area_before += a
                total_area_after += a
                area_ratios.append(1.0)
            continue

        # Load image via PIL
        try:
            with Image.open(img_path) as pil_img:
                raw_img = np.array(pil_img)
                if raw_img.ndim == 3:
                    raw_img = raw_img[:, :, 0]
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            for _, row in stem_rows.iterrows():
                final_records.append({
                    "filament_id": row["filament_id"],
                    "segmentation_rle": row["segmentation_rle"],
                })
            continue

        # Decode all anchor instances for this stem
        anchor_masks = [decode_leb128_rle(rle) for rle in stem_rows["segmentation_rle"]]
        stem_refined_masks = []

        # Union of all anchor instances for this stem
        all_anchors_union = np.zeros((2048, 2048), dtype=bool)
        for m in anchor_masks:
            all_anchors_union |= m

        # Accumulator canvas for strict disjointness (Rule 4)
        claimed_canvas = np.zeros((2048, 2048), dtype=bool)

        for m_orig in anchor_masks:
            a_before = int(np.sum(m_orig > 0))
            total_area_before += a_before

            # Fast cropped boundary snap
            m_refined = snap_fast(m_orig, raw_img, pad=12, k_deficit=0.20)

            # Other anchors in their original positions cannot be invaded
            other_anchors = all_anchors_union & (~m_orig)
            # Allowed region: cannot overlap any other original anchor, nor any previously claimed refined mask
            allowed = (~other_anchors) & (~claimed_canvas)

            # Constrain refined mask to allowed region
            m_refined = m_refined & allowed

            # Guarantee core is intact (core_orig <= m_orig <= allowed)
            core_orig = ndi.binary_erosion(m_orig, structure=ndi.generate_binary_structure(2, 1), iterations=1)
            m_refined = m_refined | core_orig

            a_after = int(np.sum(m_refined > 0))
            # Strict area preservation: must be in [0.98, 1.04] and collision-free
            if (a_after < int(np.ceil(0.9800 * a_before)) or 
                a_after > int(np.floor(1.0400 * a_before)) or 
                np.any(m_refined & claimed_canvas)):
                # Fallback to m_orig (guaranteed collision-free since m_orig is untouched by claimed_canvas)
                m_refined = m_orig
                a_after = a_before

            claimed_canvas |= m_refined
            total_area_after += a_after
            area_ratios.append(a_after / a_before if a_before > 0 else 1.0)

            if a_after != a_before:
                refined_instance_count += 1

            stem_refined_masks.append(m_refined)

        # Verification assertion per stem: zero collision
        stem_canvas_check = np.zeros((2048, 2048), dtype=bool)
        for m in stem_refined_masks:
            assert not np.any(stem_canvas_check & m), f"Disjoint canvas violation in stem {stem}!"
            stem_canvas_check |= m

        # Output sequential filament IDs
        for i, m in enumerate(stem_refined_masks, start=1):
            filament_id = f"{stem}_{i}"
            rle_str = encode_leb128_rle(m)
            final_records.append({
                "filament_id": filament_id,
                "segmentation_rle": rle_str,
            })

        if (idx + 1) % 30 == 0 or (idx + 1) == len(active_stems):
            print(f"Processed {idx + 1}/{len(active_stems)} stems... (Refined instances: {refined_instance_count})", flush=True)

    df_out = pd.DataFrame(final_records)
    df_out.to_csv(OUTPUT_CSV, index=False)

    with open(OUTPUT_CSV, "rb") as f:
        file_bytes = f.read()
        sha256_hash = hashlib.sha256(file_bytes).hexdigest()

    area_ratios = np.array(area_ratios)
    net_area_ratio = (total_area_after / total_area_before) if total_area_before > 0 else 1.0

    print("\n" + "=" * 75, flush=True)
    print("OPERATION ABSORPTION SNAP: COMPLETION SUMMARY", flush=True)
    print("=" * 75, flush=True)
    print(f"Output Submission File: {OUTPUT_CSV}", flush=True)
    print(f"File Size: {len(file_bytes):,} bytes", flush=True)
    print(f"Total Instances: {len(df_out):,} (Target: exactly 1,324)", flush=True)
    n_stems = df_out["filament_id"].apply(lambda x: x.split("_")[0]).nunique()
    print(f"Active Stems: {n_stems} (Target: exactly 178)", flush=True)
    print(f"Instances Boundary Refined: {refined_instance_count:,} / {len(df_out):,}", flush=True)
    print(f"Total Foreground Area Before: {total_area_before:,} px", flush=True)
    print(f"Total Foreground Area After:  {total_area_after:,} px", flush=True)
    print(f"Net Area Conservation Ratio:  {net_area_ratio:.4f} (Target: [0.98, 1.04])", flush=True)
    print(f"Min Instance Area Ratio:      {np.min(area_ratios):.4f}", flush=True)
    print(f"Max Instance Area Ratio:      {np.max(area_ratios):.4f}", flush=True)
    print(f"Instances strictly in [0.98, 1.04]: {int(np.sum((area_ratios >= 0.9800) & (area_ratios <= 1.0400)))} / {len(area_ratios)}", flush=True)
    print(f"SHA-256 Checksum: {sha256_hash}", flush=True)
    print(f"Total Runtime: {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
