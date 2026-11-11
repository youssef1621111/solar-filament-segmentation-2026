#!/usr/bin/env python3
"""
src/data/coco_to_rle_gt.py
Extracts official competition ground truth into clean RLE DataFrames partitioned by validation fold.
Used for deterministic out-of-fold Panoptic Quality evaluation.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from pycocotools import mask as mask_util

HEIGHT = 2048
WIDTH = 2048


def build_gt_dataframe_for_fold(
    train_json_path: Path,
    manifest_csv_path: Path,
    fold: Optional[int] = None,
) -> pd.DataFrame:
    """
    Builds a ground-truth DataFrame with columns ['filament_id', 'segmentation_rle'].
    If fold is provided (0..4), filters to only the physical observations in that fold.
    If fold is None, returns the entire training ground truth.

    format of filament_id: '<image_id>_<anno_id>'
    e.g. '040301-20140609195854Bh_d7fd9cdb-3f34-4dfc-a1fa-c1fd25110a98'
    """
    manifest_df = pd.read_csv(manifest_csv_path)
    if fold is not None:
        target_image_ids = set(manifest_df[manifest_df["fold"] == fold]["image_id"])
    else:
        target_image_ids = set(manifest_df["image_id"])

    with open(train_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    rows = []
    for anno in data["annotations"]:
        iid = anno["image_id"]
        if iid not in target_image_ids:
            continue

        aid = anno["id"]
        seg = anno["segmentation"]
        rles = mask_util.frPyObjects(seg, HEIGHT, WIDTH)
        rle = mask_util.merge(rles)
        counts = rle["counts"]
        counts_str = counts.decode("ascii") if isinstance(counts, bytes) else str(counts)

        rows.append({
            "filament_id": f"{iid}_{aid}",
            "segmentation_rle": counts_str,
        })

    df_gt = pd.DataFrame(rows)
    return df_gt


if __name__ == "__main__":
    train_json = Path(
        "/home/fathyyoussef162/data/annotations/MAGFiLO_1.0_Kaggle_2026/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json"
    )
    manifest = Path("/home/fathyyoussef162/data/train_folds_manifest.csv")
    for f in range(5):
        df_fold_gt = build_gt_dataframe_for_fold(train_json, manifest, fold=f)
        print(f"Fold {f} GT DataFrame: {len(df_fold_gt)} instances extracted.")
