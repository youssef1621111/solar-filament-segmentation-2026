#!/usr/bin/env python3
"""
Streaming boundary refinement test on 30 stems.
Zero RAM overhead: processes one stem at a time.
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
    c = rle["counts"]
    return c.decode("ascii") if isinstance(c, bytes) else str(c)

def main():
    df_v11 = pd.read_csv("/home/fathyyoussef162/experiments/submission_championship_v11_aegis_180.csv")
    df_v11["stem"] = df_v11["filament_id"].apply(lambda x: x.rsplit("_", 1)[0])
    
    roots = [Path(f"experiments/ranjeet258/public_model_outputs/sf-seg-train-v{i}") for i in (1, 2, 3)]
    zip_paths = [zp for r in roots for zp in r.glob("test_fold*.zip")]
    assert len(zip_paths) == 12
    opened_zips = [zipfile.ZipFile(zp) for zp in zip_paths]
    
    k3_ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    k3_cross = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    
    stems = sorted(df_v11["stem"].unique())[:30] # Test on first 30 stems
    print(f"Testing streaming refinement on {len(stems)} stems...")
    
    # Baseline for these 30 stems
    sub_base = df_v11[df_v11["stem"].isin(stems)].copy()
    sub_base.to_csv("/tmp/test_base_30.csv", index=False)
    
    strategies = [
        ("raw_baseline", None, 0, 0),
        ("p_trim_010", "trim", 0.10, 0.85),
        ("p_trim_020", "trim", 0.20, 0.85),
        ("p_trim_030", "trim", 0.30, 0.85),
        ("p_snap_035", "snap", 0.35, 0.80),
        ("p_snap_045", "snap", 0.45, 0.80),
        ("p_snap_055", "snap", 0.55, 0.80),
        ("erode_cross_300", "erode_cross", 300, 0.85),
        ("erode_ellipse_400", "erode_ellipse", 400, 0.85),
    ]
    
    # We will test each strategy on these 30 stems
    for name, mode, param, guard_iou in strategies:
        rows = []
        for s_idx, stem in enumerate(stems):
            stem_rows = df_v11[df_v11["stem"] == stem]
            
            if mode is None:
                for _, r in stem_rows.iterrows():
                    rows.append({"filament_id": r["filament_id"], "segmentation_rle": r["segmentation_rle"]})
                continue
                
            # Load p_map for this stem only
            p = np.zeros((H, W), np.float32)
            if mode in ["trim", "snap"]:
                for z in opened_zips:
                    a = cv2.imdecode(np.frombuffer(z.read(f"{stem}.png"), np.uint8), cv2.IMREAD_COLOR)
                    p += a[:, :, 0]
                p /= (255.0 * len(opened_zips))
                
            occupied = np.zeros((H, W), dtype=bool)
            for _, r in stem_rows.iterrows():
                fid = r["filament_id"]
                m = mu.decode({"size": [H, W], "counts": r["segmentation_rle"].encode("ascii")}).astype(np.uint8)
                area = int(m.sum())
                
                if mode == "trim":
                    if area < 300:
                        cand = m
                    else:
                        ero = cv2.erode(m, k3_ellipse)
                        band = (m > 0) & (ero == 0)
                        cand = (ero > 0) | (band & (p >= param))
                elif mode == "snap":
                    if area < 300:
                        cand = m
                    else:
                        dil = cv2.dilate(m, k3_ellipse)
                        cand = (dil > 0) & (p >= param)
                elif mode == "erode_cross":
                    if area >= param:
                        cand = cv2.erode(m, k3_cross)
                    else:
                        cand = m
                elif mode == "erode_ellipse":
                    if area >= param:
                        cand = cv2.erode(m, k3_ellipse)
                    else:
                        cand = m
                else:
                    cand = m
                    
                cand_u8 = cand.astype(np.uint8)
                inter = int((m & cand_u8).sum())
                union = int((m | cand_u8).sum())
                iou = inter / max(union, 1)
                
                num_c, _ = cv2.connectedComponents(cand_u8, connectivity=8)
                num_orig, _ = cv2.connectedComponents(m, connectivity=8)
                
                if iou >= guard_iou and num_c <= num_orig and cand_u8.sum() >= 60:
                    chosen = cand_u8
                else:
                    chosen = m
                    
                bool_m = (chosen > 0) & (~occupied)
                if bool_m.sum() < 60:
                    bool_m = (m > 0) & (~occupied)
                    if bool_m.sum() < 60:
                        continue
                occupied |= bool_m
                rows.append({"filament_id": fid, "segmentation_rle": encode_mask(bool_m)})
                
        out_csv = f"/tmp/sweep_30_{name}.csv"
        df_out = pd.DataFrame(rows)
        # Renumber
        final_rows = []
        for s, g in df_out.groupby(df_out["filament_id"].apply(lambda x: x.rsplit("_", 1)[0])):
            for i, (_, r) in enumerate(g.iterrows(), 1):
                final_rows.append({"filament_id": f"{s}_{i}", "segmentation_rle": r["segmentation_rle"]})
        pd.DataFrame(final_rows).to_csv(out_csv, index=False)
        pq, miou, tp, fp, fn = evaluate_submission(out_csv)
        print(f"Strategy {name:18s}: PQ={pq:.5f}, Mean IoU={miou:.4f}, TP={tp}, FP={fp}, FN={fn}")
        
    for z in opened_zips:
        z.close()

if __name__ == "__main__":
    main()
