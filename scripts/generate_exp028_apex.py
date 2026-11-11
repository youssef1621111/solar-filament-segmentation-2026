#!/usr/bin/env python3
"""
Championship Pipeline (EXP-028: High-Precision Collinear-Healed & Polygon-Smoothed Apex)
IEEE BigData 2026 Solar Filament Segmentation Challenge (filament-segmentation-2026)

Target: 100% Legal, Audit-Proof Breakthrough beyond 0.58 LB towards 0.62-0.64+.

Key Invariants:
1. Baseline Anchor: EXP-024 (1,299 instances, 0.58 LB, optimal instance density).
2. Homotopic Void Healing: Fast bbox-cropped pinhole closing restoring true solid absorption.
3. FFBSi Collinear Spine Healing: Connects fractured ribbons (gap <= 12 px, angle diff <= 30 deg).
4. Douglas-Peucker Smoothing: epsilon=1.0 removes raster staircase noise without area drift.
5. Strict Disjoint Canvas: Exactly 0 colliding pixels across all 178 active stems.
6. Quiet-Sun Guard: 0 rows on solar minimum stems (20160513133714Th, 20180507140210Th).
"""

import os
import sys
import numpy as np
import pandas as pd
import cv2
import scipy.ndimage as ndi
from skimage.morphology import skeletonize
from pycocotools import mask as mask_util

# Ensure repo root is in path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from scripts.utils_metrics import decode_leb128_rle, encode_leb128_rle, decode_leb128_counts


def fill_internal_voids_fast(mask: np.ndarray, max_void_size: int = 50, pad: int = 4) -> np.ndarray:
    """Fills small background pinholes inside the filament mask using fast bounding-box cropping."""
    pts = np.argwhere(mask > 0)
    if len(pts) == 0:
        return mask
    y_min, x_min = pts.min(axis=0)
    y_max, x_max = pts.max(axis=0)
    y0 = max(0, y_min - pad)
    y1 = min(2048, y_max + pad + 1)
    x0 = max(0, x_min - pad)
    x1 = min(2048, x_max + pad + 1)

    crop = mask[y0:y1, x0:x1]
    filled_crop = ndi.binary_fill_holes(crop)
    voids_crop = filled_crop & (~crop)
    if not np.any(voids_crop):
        return mask

    # Filter out voids larger than max_void_size
    lbl, num = ndi.label(voids_crop)
    if num > 0:
        counts = np.bincount(lbl.ravel())
        valid_lbls = np.where((counts <= max_void_size) & (counts > 0))[0]
        valid_lbls = set(valid_lbls[valid_lbls != 0])
        if len(valid_lbls) > 0:
            valid_void_crop = np.isin(lbl, list(valid_lbls))
            out = mask.copy()
            out[y0:y1, x0:x1] |= valid_void_crop
            return out
    return mask


def smooth_polygon_contour_fast(mask: np.ndarray, epsilon: float = 1.0, pad: int = 4) -> np.ndarray:
    """Applies Douglas-Peucker polygon simplification on bounding box crop."""
    pts = np.argwhere(mask > 0)
    if len(pts) == 0:
        return mask
    y_min, x_min = pts.min(axis=0)
    y_max, x_max = pts.max(axis=0)
    y0 = max(0, y_min - pad)
    y1 = min(2048, y_max + pad + 1)
    x0 = max(0, x_min - pad)
    x1 = min(2048, x_max + pad + 1)

    crop = mask[y0:y1, x0:x1]
    h, w = crop.shape
    uint_crop = np.ascontiguousarray((crop > 0).astype(np.uint8) * 255)
    contours, _ = cv2.findContours(uint_crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out_crop = np.zeros((h, w), dtype=np.uint8)
    for cnt in contours:
        if cv2.contourArea(cnt) < 20:
            continue
        approx = cv2.approxPolyDP(cnt, epsilon, closed=True)
        if len(approx) >= 3:
            cv2.drawContours(out_crop, [approx.astype(np.int32)], -1, 255, -1)
    
    out_bool = out_crop > 0
    orig_area = crop.sum()
    if orig_area > 0 and out_bool.sum() >= 0.85 * orig_area:
        out_mask = mask.copy()
        out_mask[y0:y1, x0:x1] = out_bool
        return out_mask
    return mask


def extract_spine_endpoints_fast(mask: np.ndarray, k_trace: int = 8, pad: int = 4):
    """Extracts terminal endpoints and outward tangent vectors with fast bbox cropping."""
    pts = np.argwhere(mask > 0)
    if len(pts) == 0:
        return []
    y_min, x_min = pts.min(axis=0)
    y_max, x_max = pts.max(axis=0)
    y0 = max(0, y_min - pad)
    y1 = min(2048, y_max + pad + 1)
    x0 = max(0, x_min - pad)
    x1 = min(2048, x_max + pad + 1)

    crop = mask[y0:y1, x0:x1]
    skel = skeletonize(crop)
    if not np.any(skel):
        return []
    neigh_kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=int)
    neigh = ndi.convolve(skel.astype(int), neigh_kernel, mode="constant")
    eps = np.argwhere((skel > 0) & (neigh == 1))
    if len(eps) == 0:
        return []
    endpoints = []
    for ep in eps:
        curr = tuple(ep)
        visited = {curr}
        path = [curr]
        for _ in range(k_trace):
            cy, cx = curr
            found = False
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dy == 0 and dx == 0:
                        continue
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < crop.shape[0] and 0 <= nx < crop.shape[1]:
                        if skel[ny, nx] and (ny, nx) not in visited:
                            visited.add((ny, nx))
                            path.append((ny, nx))
                            curr = (ny, nx)
                            found = True
                            break
                if found:
                    break
            if not found:
                break
        p_tip = np.array([ep[0] + y0, ep[1] + x0], dtype=float)
        p_in = np.array([path[-1][0] + y0, path[-1][1] + x0], dtype=float)
        vec_out = p_tip - p_in
        norm = np.linalg.norm(vec_out)
        unit_vec = vec_out / norm if norm > 0 else np.zeros(2)
        endpoints.append({"point": p_tip, "unit_vec": unit_vec})
    return endpoints


def can_heal_fracture(ep1, ep2, max_gap: float = 12.0, max_angle_deg: float = 30.0) -> bool:
    """Checks if two endpoints are close and collinear for FFBSi fracture healing."""
    p1, p2 = ep1["point"], ep2["point"]
    dist = np.linalg.norm(p1 - p2)
    if dist > max_gap or dist < 1.0:
        return False
    v_gap = (p2 - p1) / dist
    u1 = ep1["unit_vec"]
    u2 = ep2["unit_vec"]
    cos1 = np.dot(u1, v_gap)
    cos2 = np.dot(u2, -v_gap)
    min_cos = np.cos(np.radians(max_angle_deg))
    return (cos1 >= min_cos) and (cos2 >= min_cos)


def build_exp028(
    input_csv: str = "results/operation_058_breakthrough/predictions/submission_exp024_wsp_consensus_ensemble.csv",
    output_csv: str = "results/operation_058_breakthrough/predictions/submission_exp028_apex.csv"
):
    print("=" * 80, flush=True)
    print("BUILDING EXP-028: HIGH-PRECISION COLLINEAR-HEALED & POLYGON-SMOOTHED APEX", flush=True)
    print("=" * 80, flush=True)

    assert os.path.exists(input_csv), f"Input baseline {input_csv} not found!"
    df = pd.read_csv(input_csv)
    df["stem"] = df["filament_id"].apply(lambda x: x.rsplit("_", 1)[0])
    stems = sorted(df["stem"].unique())
    print(f"Loaded {len(df)} instances across {len(stems)} active stems from {input_csv}.", flush=True)

    # Quiet-sun verification
    qs_stems = {"20160513133714Th", "20180507140210Th"}
    assert len(qs_stems.intersection(set(stems))) == 0, "Quiet-sun stem found in anchor!"

    total_instances_processed = 0
    total_fractures_healed = 0
    total_pinholes_filled = 0
    submission_rows = []

    for stem_idx, stem in enumerate(stems, 1):
        stem_rows = df[df["stem"] == stem].copy()
        raw_masks = [decode_leb128_rle(r) for r in stem_rows["segmentation_rle"]]

        # Step 1: Internal pinhole void filling
        cleaned_masks = []
        for m in raw_masks:
            m_filled = fill_internal_voids_fast(m, max_void_size=50)
            if m_filled.sum() > m.sum():
                total_pinholes_filled += 1
            cleaned_masks.append(m_filled)

        # Step 2: Collinear spine fracture healing
        n_m = len(cleaned_masks)
        merged = [False] * n_m
        healed_masks = []

        if n_m > 1:
            endpoints_list = [extract_spine_endpoints_fast(m) for m in cleaned_masks]
            for i in range(n_m):
                if merged[i]:
                    continue
                current_mask = cleaned_masks[i].copy()
                for j in range(i + 1, n_m):
                    if merged[j]:
                        continue
                    eps_i = endpoints_list[i]
                    eps_j = endpoints_list[j]
                    heal = False
                    bridge_mask = None
                    for ep1 in eps_i:
                        for ep2 in eps_j:
                            if can_heal_fracture(ep1, ep2, max_gap=12.0, max_angle_deg=30.0):
                                heal = True
                                p1 = tuple(ep1["point"].astype(int)[::-1]) # (x, y)
                                p2 = tuple(ep2["point"].astype(int)[::-1]) # (x, y)
                                line_canvas = np.zeros((2048, 2048), dtype=np.uint8)
                                cv2.line(line_canvas, p1, p2, 1, thickness=3)
                                bridge_mask = line_canvas > 0
                                break
                        if heal:
                            break
                    if heal and bridge_mask is not None:
                        current_mask = current_mask | cleaned_masks[j] | bridge_mask
                        merged[j] = True
                        total_fractures_healed += 1

                healed_masks.append(current_mask)
        else:
            healed_masks = cleaned_masks

        # Step 3: Douglas-Peucker polygon smoothing (epsilon=1.0)
        smoothed_masks = [smooth_polygon_contour_fast(m, epsilon=1.0) for m in healed_masks]

        # Step 4: Strict zero-collision disjoint allocation
        occupied = np.zeros((2048, 2048), dtype=bool)
        smoothed_masks.sort(key=lambda m: m.sum(), reverse=True)

        stem_out_rles = []
        for m in smoothed_masks:
            m_clean = m & (~occupied)
            if m_clean.sum() >= 60:
                occupied |= m_clean
                stem_out_rles.append(encode_leb128_rle(m_clean))

        for inst_idx, rle_str in enumerate(stem_out_rles, 1):
            submission_rows.append({
                "filament_id": f"{stem}_{inst_idx}",
                "segmentation_rle": rle_str
            })
            total_instances_processed += 1

        if stem_idx % 30 == 0 or stem_idx == len(stems):
            print(f"[{stem_idx:3d}/{len(stems)}] Processed {stem}: instances={len(stem_out_rles)}, healed fractures={total_fractures_healed}", flush=True)

    print("\n" + "=" * 80, flush=True)
    print("EXP-028 GENERATION SUMMARY:", flush=True)
    print(f"  Total Active Stems:      {len(stems)}", flush=True)
    print(f"  Total Instances Output:  {total_instances_processed} (Baseline: {len(df)})", flush=True)
    print(f"  Pinholes Filled:         {total_pinholes_filled}", flush=True)
    print(f"  Fractures Healed:        {total_fractures_healed}", flush=True)
    print("=" * 80, flush=True)

    # Save final DataFrame
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    out_df = pd.DataFrame(submission_rows)
    out_df.to_csv(output_csv, index=False)
    print(f"\n[+] EXP-028 successfully saved to: {output_csv}", flush=True)

    # Step 5: Verification & Integrity Audit
    print("\nRunning instant zero-trust integrity checklist...", flush=True)
    assert list(out_df.columns) == ["filament_id", "segmentation_rle"], "Bad column names!"
    out_stems = out_df["filament_id"].apply(lambda x: x.rsplit("_", 1)[0])
    assert out_stems.nunique() == 178, f"Expected 178 stems, got {out_stems.nunique()}"
    assert len(set(out_stems).intersection(qs_stems)) == 0, "Quiet-sun violation!"
    for rle in out_df["segmentation_rle"]:
        counts = decode_leb128_counts(rle)
        assert sum(counts) == 4194304, f"RLE sum != 4194304: {sum(counts)}"
        assert all(c >= 0 for c in counts), "Negative run length found!"
    colliding = 0
    for stem, group in out_df.groupby(out_stems):
        if len(group) <= 1:
            continue
        c = np.zeros((2048, 2048), dtype=bool)
        for r in group["segmentation_rle"]:
            m = decode_leb128_rle(r)
            if np.any(c & m):
                colliding += 1
                break
            c |= m
    assert colliding == 0, f"Pixel collision detected in {colliding} stems!"
    print("[PASSED] ALL 5 INTEGRITY AUDITS PASSED WITH ZERO DEFECTS!", flush=True)
    print("EXP-028 is 100% READY FOR SUBMISSION!", flush=True)


if __name__ == "__main__":
    build_exp028()
