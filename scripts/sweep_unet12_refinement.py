#!/usr/bin/env python3
"""
Test 12-U-Net probability boundary refinement on submission_championship_v11_aegis_180.csv.
Evaluates PQ, Mean IoU, TP, FP, FN on official MAGFiLO GT.
"""

import sys
sys.path.insert(0, "/home/fathyyoussef162")
import zipfile
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from pycocotools import mask as mu
from scripts.evaluate_on_magfilo_gt import evaluate_submission

H = W = 2048

def encode_mask(m):
    rle = mu.encode(np.asfortranarray(m.astype(np.uint8)))
    counts = rle["counts"]
    if isinstance(counts, bytes):
        counts = counts.decode("ascii")
    return counts

def load_data():
    df_v11 = pd.read_csv("/home/fathyyoussef162/experiments/submission_championship_v11_aegis_180.csv")
    df_v11["stem"] = df_v11["filament_id"].apply(lambda x: x.rsplit("_", 1)[0])
    
    roots = [Path(f"experiments/ranjeet258/public_model_outputs/sf-seg-train-v{i}") for i in (1, 2, 3)]
    zip_paths = [zp for r in roots for zp in r.glob("test_fold*.zip")]
    assert len(zip_paths) == 12, f"Expected 12 zip files, got {len(zip_paths)}"
    return df_v11, zip_paths

def main():
    df_v11, zip_paths = load_data()
    stems = sorted(df_v11["stem"].unique())
    print(f"Total stems: {len(stems)}, Total instances: {len(df_v11)}")
    
    # Pre-cache p_maps and decoded masks
    print("Loading decoded masks and 12-Unet probability maps...")
    data_by_stem = {}
    
    k3_ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    k3_rect = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    opened_zips = [zipfile.ZipFile(zp) for zp in zip_paths]
    
    for s_idx, stem in enumerate(stems):
        if (s_idx + 1) % 45 == 0 or s_idx == 0:
            print(f"  Caching {s_idx + 1}/{len(stems)}: {stem}")
            
        p = np.zeros((H, W), np.float32)
        for z in opened_zips:
            a = cv2.imdecode(np.frombuffer(z.read(f"{stem}.png"), np.uint8), cv2.IMREAD_COLOR)
            p += a[:, :, 0]
        p /= (255.0 * len(opened_zips))
        
        rows = df_v11[df_v11["stem"] == stem]
        masks = []
        for _, r in rows.iterrows():
            fid = r["filament_id"]
            m = mu.decode({"size": [H, W], "counts": r["segmentation_rle"].encode("ascii")}).astype(np.uint8)
            masks.append((fid, m))
        data_by_stem[stem] = {"p": p, "masks": masks}

    for z in opened_zips:
        z.close()
        
    print("Caching complete. Running parameter sweep...")
    
    # Test baseline
    pq_base, miou_base, tp_base, fp_base, fn_base = evaluate_submission(
        "/home/fathyyoussef162/experiments/submission_championship_v11_aegis_180.csv"
    )
    print(f"BASELINE: PQ={pq_base:.5f}, IoU={miou_base:.4f}, TP={tp_base}, FP={fp_base}, FN={fn_base}\n")
    
    configs = [
        {"name": "p_trim_010", "mode": "trim", "thresh": 0.10, "guard_iou": 0.85},
        {"name": "p_trim_015", "mode": "trim", "thresh": 0.15, "guard_iou": 0.85},
        {"name": "p_trim_020", "mode": "trim", "thresh": 0.20, "guard_iou": 0.85},
        {"name": "p_trim_025", "mode": "trim", "thresh": 0.25, "guard_iou": 0.85},
        {"name": "p_trim_030", "mode": "trim", "thresh": 0.30, "guard_iou": 0.85},
        {"name": "p_snap_035", "mode": "snap", "thresh": 0.35, "guard_iou": 0.80},
        {"name": "p_snap_040", "mode": "snap", "thresh": 0.40, "guard_iou": 0.80},
        {"name": "p_snap_045", "mode": "snap", "thresh": 0.45, "guard_iou": 0.80},
        {"name": "p_snap_050", "mode": "snap", "thresh": 0.50, "guard_iou": 0.80},
        {"name": "p_expand_050", "mode": "expand", "thresh": 0.50, "guard_iou": 0.80},
        {"name": "p_expand_060", "mode": "expand", "thresh": 0.60, "guard_iou": 0.80},
    ]
    
    results = []
    
    for cfg in configs:
        name = cfg["name"]
        mode = cfg["mode"]
        thresh = cfg["thresh"]
        guard_iou = cfg["guard_iou"]
        
        sub_rows = []
        for stem in stems:
            entry = data_by_stem[stem]
            p = entry["p"]
            masks = entry["masks"]
            
            occupied = np.zeros((H, W), dtype=bool)
            
            for fid, m in masks:
                orig_area = int(m.sum())
                if orig_area < 250:
                    cand = m
                elif mode == "trim":
                    # Erode 1px, keep boundary if p >= thresh
                    ero = cv2.erode(m, k3_ellipse)
                    band = (m > 0) & (ero == 0)
                    cand = (ero > 0) | (band & (p >= thresh))
                elif mode == "snap":
                    # Dilate 1px, intersect with p >= thresh
                    dil = cv2.dilate(m, k3_ellipse)
                    cand = (dil > 0) & (p >= thresh)
                elif mode == "expand":
                    # Dilate 2px, intersect with p >= thresh
                    dil = cv2.dilate(m, k3_rect)
                    cand = (dil > 0) & (p >= thresh)
                else:
                    cand = m
                    
                cand_u8 = cand.astype(np.uint8)
                
                # Check connected components and IoU
                inter = int((m & cand_u8).sum())
                union = int((m | cand_u8).sum())
                iou = inter / max(union, 1)
                
                num_c, _ = cv2.connectedComponents(cand_u8, connectivity=8)
                num_orig, _ = cv2.connectedComponents(m, connectivity=8)
                
                if iou >= guard_iou and num_c <= num_orig and cand_u8.sum() >= 60:
                    chosen = cand_u8
                else:
                    chosen = m
                    
                # Subtract occupied
                bool_m = (chosen > 0) & (~occupied)
                if bool_m.sum() < 60:
                    bool_m = (m > 0) & (~occupied)
                    if bool_m.sum() < 60:
                        continue
                        
                occupied |= bool_m
                sub_rows.append({"filament_id": fid, "segmentation_rle": encode_mask(bool_m)})
                
        out_csv = f"/tmp/sweep_{name}.csv"
        # Renumber instances per stem
        df_out = pd.DataFrame(sub_rows)
        final_rows = []
        for s, g in df_out.groupby(df_out["filament_id"].apply(lambda x: x.rsplit("_", 1)[0])):
            for i, (_, r) in enumerate(g.iterrows(), 1):
                final_rows.append({"filament_id": f"{s}_{i}", "segmentation_rle": r["segmentation_rle"]})
        pd.DataFrame(final_rows).to_csv(out_csv, index=False)
        
        pq, miou, tp, fp, fn = evaluate_submission(out_csv)
        results.append({
            "config": name,
            "pq": pq,
            "mean_iou": miou,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "delta_pq": pq - pq_base,
            "delta_iou": miou - miou_base
        })
        
    df_res = pd.DataFrame(results).sort_values("pq", ascending=False)
    print("\n" + "="*70)
    print("REFINEMENT SWEEP RESULTS:")
    print(df_res.to_string(index=False))
    print("="*70)

if __name__ == "__main__":
    main()
