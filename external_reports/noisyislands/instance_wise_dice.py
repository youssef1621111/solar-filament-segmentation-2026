"""Instance-wise Dice metric — the actual Kaggle competition metric.

The README claims Panoptic Quality (PQ), but the actual scorer uses
instance_wise_dice, which is a per-GT-instance best-overlap Dice matching.
For each GT instance, find the best-matching prediction (by IoU).
Dice = 2 * IoU / (1 + IoU) for matched pairs; unmatched GTs contribute 0.
Mean is taken across ALL GT instances (not matched only).

Unmatched predictions (FP) penalize by reducing the denominator.
Unmatched GTs (FN) contribute 0 to the numerator but count in the denominator.
"""

import numpy as np
import pandas as pd
import pycocotools.mask as mask_utils
from scipy.optimize import linear_sum_assignment


def masks_to_rle_list(masks: list[np.ndarray], H: int = 2048, W: int = 2048) -> list[dict]:
    rles = []
    for m in masks:
        m_uint8 = m.astype(np.uint8) if m.dtype != np.uint8 else m
        if not m_uint8.flags["F_CONTIGUOUS"]:
            m_uint8 = np.asfortranarray(m_uint8)
        rle = mask_utils.encode(m_uint8)
        rle["counts"] = rle["counts"].decode("ascii") if isinstance(rle["counts"], bytes) else rle["counts"]
        rles.append({"size": [H, W], "counts": rle["counts"]})
    return rles


def iou_matrix(gt_rles: list[dict], pred_rles: list[dict]) -> np.ndarray:
    if not gt_rles or not pred_rles:
        return np.zeros((len(gt_rles), len(pred_rles)), dtype=np.float64)
    gt_enc = [{"size": r["size"], "counts": r["counts"].encode("ascii") if isinstance(r["counts"], str) else r["counts"]} for r in gt_rles]
    pred_enc = [{"size": r["size"], "counts": r["counts"].encode("ascii") if isinstance(r["counts"], str) else r["counts"]} for r in pred_rles]
    return np.asarray(mask_utils.iou(gt_enc, pred_enc, [0] * len(pred_enc)), dtype=np.float64)


def instance_wise_dice(
    gt_masks: list[np.ndarray],
    pred_masks: list[np.ndarray],
    H: int = 2048,
    W: int = 2048,
    match_iou_threshold: float = 0.5,
) -> dict:
    n_gt = len(gt_masks)
    n_pred = len(pred_masks)

    if n_gt == 0 and n_pred == 0:
        return {"instance_wise_dice": 1.0, "sum_dice": 0.0, "n_gt": 0, "tp": 0, "fp": 0, "fn": 0, "matched": 0}

    if n_gt == 0:
        return {"instance_wise_dice": 0.0, "sum_dice": 0.0, "n_gt": 0, "tp": 0, "fp": n_pred, "fn": 0, "matched": 0}

    if n_pred == 0:
        return {"instance_wise_dice": 0.0, "sum_dice": 0.0, "n_gt": n_gt, "tp": 0, "fp": 0, "fn": n_gt, "matched": 0}

    gt_rles = masks_to_rle_list(gt_masks, H, W)
    pred_rles = masks_to_rle_list(pred_masks, H, W)
    iou = iou_matrix(gt_rles, pred_rles)

    gt_indices, pred_indices = linear_sum_assignment(-iou)
    matched_ious = iou[gt_indices, pred_indices]
    matched_mask = matched_ious > match_iou_threshold
    matched_ious = matched_ious[matched_mask]

    dice_scores = 2.0 * matched_ious / (1.0 + matched_ious)
    total_dice = float(dice_scores.sum())
    denominator = float(n_gt)
    score = total_dice / denominator if denominator > 0 else 0.0

    tp = int(len(matched_ious))
    fp = n_pred - tp
    fn = n_gt - tp

    return {
        "instance_wise_dice": score,
        "sum_dice": total_dice,
        "n_gt": n_gt,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "matched": tp,
    }


def evaluate_instance_wise_dice(
    gt_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    H: int = 2048,
    W: int = 2048,
    match_iou_threshold: float = 0.5,
) -> dict:
    """Evaluate predictions against GT using instance-wise Dice.

    Both DataFrames must have columns: filament_id, segmentation_rle.
    filament_id format: <stem>_<index> for predictions,
                         <stem-or-annotator>-<annotator_id>_<index> for GT.
    """
    gt_by_stem = {}
    for stem, group in gt_df.groupby("stem"):
        masks = []
        for rle_str in group["segmentation_rle"]:
            rle_dict = {"size": [H, W], "counts": rle_str.encode("ascii") if isinstance(rle_str, str) else rle_str}
            masks.append(mask_utils.decode(rle_dict).astype(bool))
        gt_by_stem[stem] = masks

    pred_by_stem = {}
    if not pred_df.empty:
        for stem, group in pred_df.groupby("stem"):
            masks = []
            for rle_str in group["segmentation_rle"]:
                rle_dict = {"size": [H, W], "counts": rle_str.encode("ascii") if isinstance(rle_str, str) else rle_str}
                masks.append(mask_utils.decode(rle_dict).astype(bool))
            pred_by_stem[stem] = masks

    sum_dice = 0.0
    total_gt_count = 0
    total_tp = 0
    total_fp = 0
    total_fn = 0
    n_images = 0
    per_image_scores = []

    for stem in sorted(gt_by_stem.keys()):
        gt_masks = gt_by_stem[stem]
        pred_masks = pred_by_stem.get(stem, [])
        result = instance_wise_dice(gt_masks, pred_masks, H, W, match_iou_threshold)
        n_images += 1
        sum_dice += result["sum_dice"]
        total_gt_count += result["n_gt"]
        total_tp += result["tp"]
        total_fp += result["fp"]
        total_fn += result["fn"]
        if len(gt_masks) > 0:
            per_image_scores.append(result["instance_wise_dice"])
        else:
            per_image_scores.append(1.0 if len(pred_masks) == 0 else 0.0)

    overall = sum_dice / total_gt_count if total_gt_count > 0 else 0.0
    mean_per_image = float(np.mean(per_image_scores)) if per_image_scores else 0.0

    return {
        "instance_wise_dice": overall,
        "mean_per_image_dice": mean_per_image,
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "n_images": n_images,
        "n_gt": total_gt_count,
    }


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Evaluate instance-wise Dice")
    parser.add_argument("--gt", type=str, required=True, help="Path to GT parquet or CSV")
    parser.add_argument("--pred", type=str, required=True, help="Path to prediction CSV")
    args = parser.parse_args()

    print(f"Loading GT from {args.gt}")
    if args.gt.endswith(".parquet"):
        gt_df = pd.read_parquet(args.gt)
    else:
        gt_df = pd.read_csv(args.gt)

    if "stem" not in gt_df.columns:
        gt_df["stem"] = gt_df["filament_id"].str.split("-", n=1).str[1].str.rsplit("_", n=1).str[0]

    print(f"Loading predictions from {args.pred}")
    pred_df = pd.read_csv(args.pred)
    if "stem" not in pred_df.columns:
        pred_df["stem"] = pred_df["filament_id"].str.rsplit("_", n=1).str[0]

    results = evaluate_instance_wise_dice(gt_df, pred_df)
    print(f"\n=== Instance-Wise Dice Results ===")
    print(f"  Instance-wise Dice: {results['instance_wise_dice']:.4f}")
    print(f"  Mean per-image Dice: {results['mean_per_image_dice']:.4f}")
    print(f"  TP: {results['tp']}, FP: {results['fp']}, FN: {results['fn']}")
    print(f"  Images: {results['n_images']}, Total GT: {results['n_gt']}")
