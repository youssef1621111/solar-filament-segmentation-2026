#!/usr/bin/env python3
"""
src/data/group_split.py
Deterministic, observation-grouped 5-fold cross-validation split for IEEE BigData 2026 Solar Filament Segmentation.
Ensures zero data leakage: all annotator passes for any physical observation stem reside in exactly one fold.
"""

import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


def create_grouped_folds(
    train_json_path: Path,
    out_manifest_csv: Path,
    out_manifest_json: Path,
    n_splits: int = 5,
    random_seed: int = 42,
) -> pd.DataFrame:
    """
    Groups training data by physical observation stem and splits into n_splits balanced folds.
    """
    print(f"Loading official training annotations from: {train_json_path}")
    with open(train_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    images = data["images"]
    annotations = data["annotations"]

    print(f"Loaded {len(images)} image records and {len(annotations)} annotations.")

    # 1. Map annotations per image record
    anno_counts_per_image = Counter()
    for anno in annotations:
        anno_counts_per_image[anno["image_id"]] += 1

    # 2. Extract physical observation stem for each image record
    # Format of image['file_name'] is '<stem>.jpeg' (e.g. '20140609195854Bh.jpeg')
    # Format of image['id'] is '<annotator_id>-<stem>' (e.g. '040301-20140609195854Bh')
    img_records = []
    for img in images:
        fn = img["file_name"]
        stem_from_fn = fn.split(".")[0]
        stem_from_id = img["id"].split("-", 1)[1] if "-" in img["id"] else img["id"]
        assert stem_from_fn == stem_from_id, f"Stem mismatch: {stem_from_fn} vs {stem_from_id}"

        img_records.append({
            "image_id": img["id"],
            "stem": stem_from_fn,
            "file_name": fn,
            "date_captured": img.get("date_captured", ""),
            "n_instances": anno_counts_per_image[img["id"]],
        })

    df_images = pd.DataFrame(img_records)

    # 3. Aggregate at physical observation stem level
    stem_summary = df_images.groupby("stem").agg(
        n_passes=("image_id", "count"),
        total_instances=("n_instances", "sum"),
        image_ids=("image_id", list),
        file_names=("file_name", "first"),
    ).reset_index()

    total_stems = len(stem_summary)
    print(f"Identified {total_stems} unique physical observation stems.")

    # 4. Stratified Group K-Fold split (stratifying by number of annotator passes: 1, 2, or 3)
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
    stem_summary["fold"] = -1

    for fold_idx, (train_idx, val_idx) in enumerate(
        sgkf.split(stem_summary, stem_summary["n_passes"], groups=stem_summary["stem"])
    ):
        stem_summary.loc[val_idx, "fold"] = fold_idx

    assert (stem_summary["fold"] >= 0).all(), "Unassigned folds detected!"

    # 5. Join fold assignment back to image level
    stem_to_fold = dict(zip(stem_summary["stem"], stem_summary["fold"]))
    df_images["fold"] = df_images["stem"].map(stem_to_fold)

    # 6. Verify strictly zero leakage across folds
    for f in range(n_splits):
        val_stems = set(df_images[df_images["fold"] == f]["stem"])
        train_stems = set(df_images[df_images["fold"] != f]["stem"])
        overlap = val_stems.intersection(train_stems)
        assert len(overlap) == 0, f"FATAL ERROR: Fold {f} has stem leakage: {overlap}"

    # 7. Write manifests
    out_manifest_csv.parent.mkdir(parents=True, exist_ok=True)
    df_images.sort_values(by=["fold", "stem", "image_id"]).to_csv(out_manifest_csv, index=False)
    print(f"Saved image-level fold manifest to: {out_manifest_csv}")

    manifest_json_data = {
        "metadata": {
            "source_json": str(train_json_path),
            "n_splits": n_splits,
            "random_seed": random_seed,
            "total_physical_stems": total_stems,
            "total_image_records": len(df_images),
            "total_annotations": len(annotations),
        },
        "folds": {},
    }

    print("\n" + "=" * 80)
    print("5-FOLD STRATIFIED OBSERVATION-GROUPED CROSS-VALIDATION STATISTICS")
    print("=" * 80)
    for f in range(n_splits):
        f_stems = stem_summary[stem_summary["fold"] == f]
        f_imgs = df_images[df_images["fold"] == f]
        n_obs = len(f_stems)
        n_imgs = len(f_imgs)
        n_inst = f_stems["total_instances"].sum()
        pass_dist = f_stems["n_passes"].value_counts().to_dict()

        manifest_json_data["folds"][f] = {
            "observations_count": n_obs,
            "images_count": n_imgs,
            "instances_count": int(n_inst),
            "annotator_passes_distribution": pass_dist,
            "observation_stems": sorted(f_stems["stem"].tolist()),
        }

        print(
            f"Fold {f}: {n_obs:3d} observations | {n_imgs:3d} images | {n_inst:4d} instances | "
            f"Passes: 1-anno={pass_dist.get(1, 0)}, 2-anno={pass_dist.get(2, 0)}, 3-anno={pass_dist.get(3, 0)}"
        )
    print("=" * 80)

    with open(out_manifest_json, "w", encoding="utf-8") as f:
        json.dump(manifest_json_data, f, indent=2)
    print(f"Saved observation-level JSON manifest to: {out_manifest_json}")

    return df_images


if __name__ == "__main__":
    train_json = Path(
        "/home/fathyyoussef162/data/annotations/MAGFiLO_1.0_Kaggle_2026/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json"
    )
    manifest_csv = Path("/home/fathyyoussef162/data/train_folds_manifest.csv")
    manifest_json = Path("/home/fathyyoussef162/data/train_folds_manifest.json")

    create_grouped_folds(train_json, manifest_csv, manifest_json)
