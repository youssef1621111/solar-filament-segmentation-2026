import json
import os
import math
from collections import defaultdict
import numpy as np
import pandas as pd
import cv2
import pycocotools.mask as mask_utils
from sklearn.model_selection import StratifiedGroupKFold
from solar_transforms import enhance_solar_contrast


def ann_to_mask(ann: dict, h: int = 2048, w: int = 2048) -> np.ndarray:
    rles = mask_utils.frPyObjects(ann["segmentation"], h, w)
    rle = mask_utils.merge(rles)
    return mask_utils.decode(rle).astype(bool)


def rle_from_mask(mask: np.ndarray) -> str:
    rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    return rle["counts"].decode("utf-8") if isinstance(rle["counts"], bytes) else rle["counts"]


def semantic_mask_for_image_id(image_id: str, annotations_by_image: dict, images_by_id: dict, h: int = 512, w: int = 512) -> np.ndarray:
    """Build a downsampled (h,w) binary mask via polygon scan-filling at low resolution."""
    info = images_by_id[image_id]
    orig_h, orig_w = int(info["height"]), int(info["width"])
    canvas = np.zeros((h, w), dtype=bool)
    scale_y = h / orig_h
    scale_x = w / orig_w
    for ann in annotations_by_image.get(image_id, []):
        if ann.get("iscrowd", 0):
            continue
        seg = ann["segmentation"]
        if isinstance(seg, dict):
            continue
        polygons = seg if isinstance(seg, list) else [seg]
        for poly in polygons:
            pts = np.array(poly, dtype=np.float32).reshape(-1, 2)
            if len(pts) < 3:
                continue
            pts[:, 0] = np.clip(pts[:, 0] * scale_x, 0, w - 1)
            pts[:, 1] = np.clip(pts[:, 1] * scale_y, 0, h - 1)
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(mask, [pts.astype(np.int32)], 255)
            canvas |= (mask > 0)
    return canvas


def select_medoid_annotators(
    image_ids_by_file: dict,
    annotations_by_image: dict,
    images_by_id: dict,
) -> dict:
    """Choose the annotator closest to the others (medoid), avoiding max-area bias.

    This replaces consensus OR-ed masks which blur borders and hurt Dice.
    Single-annotator images use their sole annotator.
    """
    selected = {}
    ds = 512
    for filename, image_ids in image_ids_by_file.items():
        if len(image_ids) == 1:
            selected[filename] = image_ids[0]
            continue
        masks_ds = [semantic_mask_for_image_id(iid, annotations_by_image, images_by_id, ds, ds) for iid in image_ids]
        n = len(masks_ds)
        pairwise_iou = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            union = masks_ds[i].sum()
            for j in range(i + 1, n):
                inter = np.logical_and(masks_ds[i], masks_ds[j]).sum()
                u = masks_ds[i].sum() + masks_ds[j].sum() - inter
                iou = inter / u if u > 0 else 0.0
                pairwise_iou[i, j] = iou
                pairwise_iou[j, i] = iou
        mean_agreement = (pairwise_iou.sum(axis=1) - 1.0) / max(len(image_ids) - 1, 1)
        selected[filename] = image_ids[int(np.argmax(mean_agreement))]
    return selected


def build_single_view_targets(
    data_dir: str = "/home/dead/filament-segmentation/data/MAGFiLO_1.0_Kaggle_2026",
    output_dir: str = "/home/dead/filament-segmentation/processed",
    n_splits: int = 5,
    seed: int = 42,
):
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(data_dir, "train/MAGFiLO_1.0_Annotations_kaggle2026_train.json")
    print(f"[1/5] Loading COCO annotations from {json_path}...")
    with open(json_path, "r") as f:
        coco_data = json.load(f)

    images_by_id = {str(img["id"]): img for img in coco_data["images"]}
    annotations_by_image = defaultdict(list)
    for ann in coco_data["annotations"]:
        annotations_by_image[str(ann["image_id"])].append(ann)

    image_ids_by_file = defaultdict(list)
    for img_id, img in images_by_id.items():
        image_ids_by_file[img["file_name"]].append(img_id)
    for fname in image_ids_by_file:
        image_ids_by_file[fname].sort()

    print(f"  Total physical images: {len(image_ids_by_file)}")
    print(f"  Total annotator records: {len(images_by_id)}")
    print(f"  Total annotations: {len(coco_data['annotations'])}")
    multi = sum(1 for v in image_ids_by_file.values() if len(v) > 1)
    print(f"  Multi-annotator images: {multi}")

    print("[2/5] Selecting medoid annotator per physical image (single-view GT)...")
    selected_image_ids = select_medoid_annotators(
        image_ids_by_file, annotations_by_image, images_by_id
    )
    vc = pd.Series(list(selected_image_ids.values())).value_counts()
    print(f"  Selected annotator-images distribution: {dict(vc)}")

    print("[3/5] Building single-view GT DataFrame (matches Kaggle scorer format)...")
    gt_rows = []
    for filename, image_id in selected_image_ids.items():
        info = images_by_id[image_id]
        h, w = int(info["height"]), int(info["width"])
        fname_base = os.path.splitext(filename)[0]
        for ann in annotations_by_image.get(image_id, []):
            if ann.get("iscrowd", 0):
                continue
            mask = ann_to_mask(ann, h, w)
            gt_rows.append({
                "filament_id": f"{fname_base}-{image_id}_{len([r for r in gt_rows if r['filament_id'].startswith(fname_base)]) + 1}",
                "segmentation_rle": rle_from_mask(mask),
                "annotator_image": image_id,
                "physical_file": filename,
                "stem": fname_base,
                "area": float(ann.get("area", 0.0)),
                "category_id": int(ann.get("category_id", 0)),
            })
    gt_df = pd.DataFrame(gt_rows)
    print(f"  Total single-view GT instances: {len(gt_df)}")

    print("[4/5] Creating stratified grouped folds...")
    observations = []
    for filename, image_ids in sorted(image_ids_by_file.items()):
        counts = [len(annotations_by_image.get(iid, [])) for iid in image_ids]
        observations.append({
            "file_name": filename,
            "stem": os.path.splitext(filename)[0],
            "num_annotators": len(image_ids),
            "mean_instances": float(np.mean(counts)),
            "max_instances": max(counts, default=0),
        })
    obs_df = pd.DataFrame(observations)
    obs_df["strata_bin"] = (
        pd.qcut(obs_df["mean_instances"], q=4, labels=False, duplicates="drop").astype(str)
        + "_" + obs_df["num_annotators"].astype(str)
    )

    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    obs_df["fold"] = -1
    for fold_idx, (_, val_idx) in enumerate(
        sgkf.split(obs_df, obs_df["strata_bin"], groups=obs_df["stem"])
    ):
        obs_df.loc[val_idx, "fold"] = fold_idx

    gt_df = gt_df.merge(
        obs_df[["stem", "fold"]], on="stem", how="left"
    )

    print("[5/5] Fold distribution:")
    for f in range(n_splits):
        f_sub = obs_df[obs_df["fold"] == f]
        gt_sub = gt_df[gt_df["fold"] == f]
        print(
            f"  Fold {f}: {len(f_sub)} physical images, {len(gt_sub)} single-view GT instances "
            f"(multi-annotator: {(f_sub['num_annotators'] > 1).sum()})"
        )

    folds_path = os.path.join(output_dir, "folds_single_view.csv")
    gt_path = os.path.join(output_dir, "gt_single_view.parquet")
    obs_df.to_csv(folds_path, index=False)
    gt_df.to_parquet(gt_path, index=False)
    print(f"\n[DONE] Saved folds to {folds_path}")
    print(f"[DONE] Saved single-view GT to {gt_path}")
    return obs_df, gt_df


if __name__ == "__main__":
    build_single_view_targets()
