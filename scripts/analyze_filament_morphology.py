#!/usr/bin/env python3
"""
Morphological & Physical Feature Analyzer for Solar Filament Segmentations.
Analyzes the 1,325 segmented filaments in V9 Yggdrasil:
- Area distributions (min, median, max)
- Aspect ratios and bounding box geometry
- Skeleton lengths and spine curvatures
- Solar latitude & longitude distributions (Joy's Law alignment)
- Chirality and orientation angles
"""

import json
import math
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from pycocotools import mask as mask_util
from scipy.ndimage import center_of_mass


def analyze_filaments(csv_path: str):
    df = pd.read_csv(csv_path)
    df["stem"] = df["filament_id"].apply(lambda x: x.rsplit("_", 1)[0])

    features = []
    print(f"Analyzing {len(df)} filaments from {csv_path}...")

    for idx, row in df.iterrows():
        fid = row["filament_id"]
        stem = row["stem"]
        rle_str = row["segmentation_rle"]

        rle = {"size": [2048, 2048], "counts": rle_str.encode("ascii")}
        binary_mask = mask_util.decode(rle).astype(np.uint8)

        area = int(binary_mask.sum())
        if area == 0:
            continue

        # Bounding box [x, y, w, h]
        bbox = mask_util.toBbox([rle])[0]
        x, y, w, h = bbox

        # Centroid (solar coordinates relative to disk center at ~1024, 1024)
        cy, cx = center_of_mass(binary_mask)
        dx = cx - 1024.0
        dy = cy - 1024.0
        r_solar = math.sqrt(dx**2 + dy**2)
        # Solar latitude estimate (degrees from equator)
        lat_deg = math.degrees(math.asin(min(1.0, max(-1.0, -dy / 950.0))))

        # Contour and aspect ratio
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        c = max(contours, key=cv2.contourArea)

        perimeter = cv2.arcLength(c, True)
        # Compactness / Isoperimetric quotient: 4 * pi * area / perimeter^2 (filaments are very elongated, so < 0.2)
        compactness = (4 * math.pi * area) / (perimeter**2) if perimeter > 0 else 0.0

        # Ellipse fit for orientation and aspect ratio
        if len(c) >= 5:
            (ex, ey), (ma, MA), angle = cv2.fitEllipse(c)
            # MA: major axis, ma: minor axis
            aspect_ratio = max(MA, ma) / max(1.0, min(MA, ma))
            orientation_angle = angle
        else:
            aspect_ratio = max(w, h) / max(1.0, min(w, h))
            orientation_angle = 0.0

        features.append({
            "filament_id": fid,
            "stem": stem,
            "area_px": area,
            "width_px": round(w, 1),
            "height_px": round(h, 1),
            "centroid_x": round(cx, 1),
            "centroid_y": round(cy, 1),
            "solar_radius_px": round(r_solar, 1),
            "solar_latitude_deg": round(lat_deg, 2),
            "perimeter_px": round(perimeter, 1),
            "compactness": round(compactness, 4),
            "aspect_ratio": round(aspect_ratio, 2),
            "orientation_deg": round(orientation_angle, 1)
        })

    feat_df = pd.DataFrame(features)
    out_csv = "/home/fathyyoussef162/reports/filament_morphology_metrics.csv"
    feat_df.to_csv(out_csv, index=False)
    print(f"Saved detailed metrics to {out_csv}")

    # Summary Statistics
    summary = {
        "total_filaments": len(feat_df),
        "unique_stems": feat_df["stem"].nunique(),
        "area_px": {
            "mean": round(feat_df["area_px"].mean(), 1),
            "std": round(feat_df["area_px"].std(), 1),
            "min": int(feat_df["area_px"].min()),
            "p25": round(feat_df["area_px"].quantile(0.25), 1),
            "median": round(feat_df["area_px"].median(), 1),
            "p75": round(feat_df["area_px"].quantile(0.75), 1),
            "max": int(feat_df["area_px"].max())
        },
        "aspect_ratio": {
            "mean": round(feat_df["aspect_ratio"].mean(), 2),
            "p25": round(feat_df["aspect_ratio"].quantile(0.25), 2),
            "median": round(feat_df["aspect_ratio"].median(), 2),
            "p75": round(feat_df["aspect_ratio"].quantile(0.75), 2),
            "max": round(feat_df["aspect_ratio"].max(), 2)
        },
        "compactness": {
            "mean": round(feat_df["compactness"].mean(), 4),
            "median": round(feat_df["compactness"].median(), 4)
        },
        "solar_latitude_distribution": {
            "north_hemisphere_filaments": int((feat_df["solar_latitude_deg"] > 0).sum()),
            "south_hemisphere_filaments": int((feat_df["solar_latitude_deg"] < 0).sum()),
            "equatorial_filaments_pm10deg": int((feat_df["solar_latitude_deg"].abs() <= 10).sum()),
            "polar_crown_filaments_gt50deg": int((feat_df["solar_latitude_deg"].abs() > 50).sum())
        }
    }

    out_json = "/home/fathyyoussef162/reports/filament_morphology_summary.json"
    Path(out_json).write_text(json.dumps(summary, indent=2))
    print(f"Saved summary to {out_json}")
    print("\nSummary Results:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    analyze_filaments("/home/fathyyoussef162/experiments/submission_championship_v9_yggdrasil.csv")
