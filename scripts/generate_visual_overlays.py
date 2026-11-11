#!/usr/bin/env python3
"""
Generate Visual Segmentation Overlay Diagnostics.
Renders high-resolution RGB overlays comparing Ground Truth (Green) vs.
V9 Yggdrasil Predictions (Red), where overlap (True Positives) renders as Yellow.
Crops tightly around filament bounding boxes for inspection.
"""

import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from pycocotools import mask as mask_util

OUTPUT_DIR = Path("/home/fathyyoussef162/reports/visualizations")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 1. Load GT
df_gt = pd.read_pickle("/home/fathyyoussef162/data/magfilo_test_gt.pkl")
gt_annotator_image_ids = df_gt["filament_id"].str.split("_", n=1).str[0]

# 2. Load V9
df_v9 = pd.read_csv("/home/fathyyoussef162/experiments/submission_championship_v9_yggdrasil.csv")
df_v9["stem"] = df_v9["filament_id"].apply(lambda x: x.rsplit("_", 1)[0])

# Key stems of interest
TARGET_STEMS = [
    ("20111208173814Mh", "Multi-Annotator Consensus Filaments Rescued"),
    ("20210701163730Mh", "Zero-Collision Spine Recovery"),
    ("20160513133714Th", "Restored Atmospheric Seeing Blur Stem"),
    ("20121229085414Th", "High-Precision Solar Disk Filament")
]

for stem, description in TARGET_STEMS:
    print(f"\nRendering diagnostic for {stem}: {description}...")

    # Build GT composite mask (Green channel)
    gt_canvas = np.zeros((2048, 2048), dtype=np.uint8)
    matching_views = [v for v in gt_annotator_image_ids.unique() if stem in v]
    for view in matching_views:
        rles = df_gt.loc[gt_annotator_image_ids == view, "segmentation_rle"].tolist()
        for r in rles:
            m = mask_util.decode({"size": [2048, 2048], "counts": r.encode("ascii")})
            gt_canvas = np.maximum(gt_canvas, m)

    # Build Pred composite mask (Red channel)
    pred_canvas = np.zeros((2048, 2048), dtype=np.uint8)
    pred_rles = df_v9.loc[df_v9["stem"] == stem, "segmentation_rle"].tolist()
    for r in pred_rles:
        m = mask_util.decode({"size": [2048, 2048], "counts": r.encode("ascii")})
        pred_canvas = np.maximum(pred_canvas, m)

    # RGB Composite:
    # Red Channel: Prediction
    # Green Channel: Ground Truth
    # Blue Channel: 0 (or slight background disk tint)
    # Overlap (Prediction + Ground Truth) = Yellow (R=255, G=255, B=0)
    # Pure Pred (False Positive) = Red (R=255, G=0, B=0)
    # Pure GT (False Negative) = Green (R=0, G=255, B=0)
    rgb = np.zeros((2048, 2048, 3), dtype=np.uint8)
    rgb[:, :, 2] = pred_canvas * 255  # Red (OpenCV BGR order: Blue=0, Green=1, Red=2)
    rgb[:, :, 1] = gt_canvas * 255    # Green
    rgb[:, :, 0] = 0                  # Blue

    # Find combined bounding box to crop tightly
    combined = np.maximum(gt_canvas, pred_canvas)
    if combined.sum() == 0:
        continue

    y_indices, x_indices = np.where(combined > 0)
    ymin, ymax = max(0, y_indices.min() - 40), min(2048, y_indices.max() + 40)
    xmin, xmax = max(0, x_indices.min() - 40), min(2048, x_indices.max() + 40)

    cropped = rgb[ymin:ymax, xmin:xmax]

    # Save full cropped overlay
    out_crop_path = OUTPUT_DIR / f"overlay_{stem}_crop.png"
    cv2.imwrite(str(out_crop_path), cropped)

    # Save downsampled 512x512 full disk thumbnail
    full_thumb = cv2.resize(rgb, (512, 512), interpolation=cv2.INTER_AREA)
    out_thumb_path = OUTPUT_DIR / f"overlay_{stem}_thumb512.png"
    cv2.imwrite(str(out_thumb_path), full_thumb)

    # Compute metrics for this stem
    intersection = np.logical_and(gt_canvas, pred_canvas).sum()
    union = np.logical_or(gt_canvas, pred_canvas).sum()
    pixel_iou = intersection / union if union > 0 else 0.0

    print(f"✓ Saved {out_crop_path.name} (Crop: {xmax-xmin}x{ymax-ymin}) | Pixel IoU: {pixel_iou:.4f}")

print("\nAll overlay visualizations successfully generated in /home/fathyyoussef162/reports/visualizations/")
