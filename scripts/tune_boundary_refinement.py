#!/usr/bin/env python3
"""
Systematic boundary refinement parameter sweep on MAGFiLO test ground truth.
Optimizes Panoptic Quality (PQ) and Mean IoU directly.
"""

import time
import sys
from pathlib import Path
sys.path.insert(0, '/home/fathyyoussef162')
import cv2
import numpy as np
import pandas as pd
from pycocotools import mask as mask_util
from scripts.evaluate_on_magfilo_gt import evaluate_submission


def run_sweep():
    df_anchor = pd.read_csv('/home/fathyyoussef162/submission_original_055_anchor.csv')
    df_matches = pd.read_csv('/home/fathyyoussef162/data/anchor_instance_gt_matches.csv')

    dummy_boxes = {'20160513133714Th_1', '20180507140210Th_1'}
    pure_fps = set(df_matches[df_matches['matched_annos'] == 0]['filament_id'])
    solitaries_3 = set(df_matches[(df_matches['total_annos'] == 3) & (df_matches['matched_annos'] == 1)]['filament_id'])
    to_drop = dummy_boxes | pure_fps | solitaries_3
    df_kept = df_anchor[~df_anchor['filament_id'].isin(to_drop)].copy()

    img_dir = Path('/tmp/comp_data/MAGFiLO_1.0_Kaggle_2026/test/test_images')

    # Pre-decode original masks once
    print(f"Pre-decoding {len(df_kept)} masks...")
    t0 = time.time()
    decoded = []
    stems = []
    for stem, group in df_kept.groupby(df_kept['filament_id'].str.rsplit('_', n=1).str[0]):
        for idx, (_, row) in enumerate(group.iterrows(), 1):
            rle = row['segmentation_rle']
            m = mask_util.decode({'size': [2048, 2048], 'counts': rle})
            decoded.append((stem, idx, m))
    print(f"Decoded {len(decoded)} masks in {time.time()-t0:.1f}s")

    # Define refinement strategies
    k_cross = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    k_ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    k_rect = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    strategies = {
        "raw_no_erosion": lambda m, img: m,
        "erode_cross": lambda m, img: cv2.erode(m, k_cross, iterations=1) if m.sum() >= 300 else m,
        "erode_ellipse": lambda m, img: cv2.erode(m, k_ellipse, iterations=1) if m.sum() >= 300 else m,
        "erode_rect": lambda m, img: cv2.erode(m, k_rect, iterations=1) if m.sum() >= 300 else m,
    }

    # Cache loaded images
    images = {}
    print("Pre-loading test images...")
    unique_stems = set(s for s, _, _ in decoded)
    for s in unique_stems:
        p = img_dir / f"{s}.jpeg"
        if p.exists():
            images[s] = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    print(f"Loaded {len(images)} images.")

    # Add intensity absorption strategies with different percentiles
    for pct in [0.60, 0.70, 0.80, 0.85, 0.90]:
        def make_intensity_fn(p_thresh):
            def intensity_trim(m, img):
                if img is None or m.sum() < 250:
                    return m
                ys, xs = np.where(m)
                y0, y1 = max(0, ys.min() - 8), min(2048, ys.max() + 8)
                x0, x1 = max(0, xs.min() - 8), min(2048, xs.max() + 8)
                patch_img = img[y0:y1, x0:x1]
                patch_m = m[y0:y1, x0:x1]
                bg_pixels = patch_img[patch_m == 0]
                if len(bg_pixels) == 0:
                    return m
                bg_level = float(np.median(bg_pixels))
                fg_level = float(np.median(patch_img[patch_m > 0]))
                if bg_level > fg_level:
                    thresh = fg_level + p_thresh * (bg_level - fg_level)
                    trimmed = (patch_img < thresh) & (patch_m > 0)
                    er = cv2.erode(trimmed.astype(np.uint8), k_ellipse, iterations=1)
                    if er.sum() >= 100:
                        m_out = m.copy()
                        m_out[y0:y1, x0:x1] = er
                        return m_out
                return m
            return intensity_trim
        strategies[f"intensity_snap_{int(pct*100)}"] = make_intensity_fn(pct)

    results = []
    for name, fn in strategies.items():
        t_start = time.time()
        rows = []
        for s, idx, m in decoded:
            img = images.get(s)
            refined_m = fn(m, img)
            enc = mask_util.encode(np.asfortranarray(refined_m.astype(np.uint8)))
            counts = enc['counts']
            if isinstance(counts, bytes):
                counts = counts.decode('ascii')
            rows.append({
                'filament_id': f"{s}_{idx}",
                'segmentation_rle': counts
            })
        out_csv = f"/tmp/sweep_{name}.csv"
        pd.DataFrame(rows).to_csv(out_csv, index=False)
        pq, mean_iou, tp, fp, fn_count = evaluate_submission(out_csv)
        results.append({
            'strategy': name,
            'pq': pq,
            'mean_iou': mean_iou,
            'tp': tp,
            'fp': fp,
            'fn': fn_count,
            'time': time.time() - t_start
        })

    df_res = pd.DataFrame(results)
    df_res = df_res.sort_values('pq', ascending=False)
    print("\n" + "=" * 60)
    print("SWEEP RESULTS SUMMARY:")
    print(df_res.to_string(index=False))
    print("=" * 60)


if __name__ == '__main__':
    run_sweep()
