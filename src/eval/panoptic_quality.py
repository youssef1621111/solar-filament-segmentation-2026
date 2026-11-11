#!/usr/bin/env python3
"""
src/eval/panoptic_quality.py
Official competition Panoptic Quality (PQ) evaluator for IEEE BigData 2026 Solar Filament Segmentation Challenge.
Faithfully reproduces Dr. Azim Ahmadzadeh's official self-evaluation notebook implementation.

Metric Formula:
    A predicted instance matches a GT instance if IoU > 0.50 (strict inequality).
    TP: Set of matching pairs (p, g) with IoU(p, g) > 0.50.
    FP: Predictions with no qualifying match (column sum in hit matrix == 0).
    FN: GT instances with no qualifying match (row sum in hit matrix == 0).

    Segmentation Quality (SQ): sum(IoU over TP) / |TP|   (Mean IoU of true positives)
    Recognition Quality (RQ):  |TP| / (|TP| + 0.5 * |FP| + 0.5 * |FN|)
    Panoptic Quality (PQ):     SQ * RQ = sum(IoU over TP) / (|TP| + 0.5 * |FP| + 0.5 * |FN|)
"""

import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from pycocotools import mask as mask_util

HEIGHT = 2048
WIDTH = 2048
IOU_THRESHOLD = 0.50  # Host rule: match strictly requires IoU > 0.50


def rles_to_binary_layers(
    rles: List[Union[str, bytes, dict]], height: int = HEIGHT, width: int = WIDTH
) -> np.ndarray:
    """
    Decodes compressed COCO RLE strings into an (n_masks, H, W) binary mask stack.
    Returns float32 numpy array with values in {0, 1}.
    """
    if not rles:
        return np.zeros((0, height, width), dtype=np.float32)

    rle_dicts = []
    for r in rles:
        if isinstance(r, dict):
            rle_dicts.append(r)
        else:
            rle_str = r.encode("ascii") if isinstance(r, str) else r
            rle_dicts.append({"size": [height, width], "counts": rle_str})

    masks = mask_util.decode(rle_dicts)
    if masks.ndim == 2:
        masks = masks[:, :, np.newaxis]
    return masks.transpose(2, 0, 1).astype(np.float32)


def compute_pairwise_iou_dice(
    gt_layers: np.ndarray, pred_layers: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes pairwise IoU and Dice matrix between GT and prediction binary layers.
    Shape: (n_gt, n_pred).
    Matches Dr. Ahmadzadeh's get_overlap_matrices exactly.
    """
    n_gt = gt_layers.shape[0]
    n_pred = pred_layers.shape[0]

    if n_gt == 0 or n_pred == 0:
        return np.zeros((n_gt, n_pred), dtype=np.float32), np.zeros((n_gt, n_pred), dtype=np.float32)

    gt_flat = gt_layers.reshape(n_gt, -1)
    pred_flat = pred_layers.reshape(n_pred, -1)

    intersection = np.dot(gt_flat, pred_flat.T)
    gt_areas = gt_flat.sum(axis=1, keepdims=True)
    pred_areas = pred_flat.sum(axis=1, keepdims=True).T

    union = gt_areas + pred_areas - intersection

    iou_matrix = np.where(union == 0, 0.0, intersection / np.maximum(union, 1e-12))
    dice_matrix = np.where(union == 0, 0.0, 2 * intersection / np.maximum(gt_areas + pred_areas, 1e-12))

    return iou_matrix.astype(np.float32), dice_matrix.astype(np.float32)


def compute_panoptic_quality_exact(
    gt_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    iou_threshold: float = IOU_THRESHOLD,
    fast_coco: bool = True,
) -> Dict[str, Any]:
    """
    Computes the official Panoptic Quality metric over all annotator views.

    Args:
        gt_df: Ground-truth DataFrame with columns ['filament_id', 'segmentation_rle'].
               filament_id format in GT: '<annotator_id>-<image_stem>_<anno_id>'
        pred_df: Prediction DataFrame with columns ['filament_id', 'segmentation_rle'].
                 filament_id format in Pred: '<image_stem>_<pred_id>'
        iou_threshold: Strict matching threshold (default: 0.50).
        fast_coco: Whether to use pycocotools C-accelerated mask.iou (mathematically identical, 100x faster).

    Returns:
        Dict with keys:
            - 'pq': Panoptic Quality score
            - 'sq': Segmentation Quality (Mean IoU of TP pairs)
            - 'rq': Recognition Quality (F1-score of instances)
            - 'tp': Number of true positives
            - 'fp': Number of false positives
            - 'fn': Number of false negatives
            - 'tp_ious': List of matched IoUs
            - 'num_annotator_views': Number of annotator-image passes evaluated
    """
    t0 = time.time()

    # GT annotator view is the prefix before '_' (e.g. '040301-20140609195854Bh')
    gt_annotator_image_ids = gt_df["filament_id"].str.split("_", n=1).str[0]
    # Prediction image id is the stem before the last '_' (e.g. '20140609195854Bh')
    pred_image_ids = pred_df["filament_id"].str.rsplit("_", n=1).str[0]

    annotator_views = gt_annotator_image_ids.unique()

    tp_iou_scores: List[float] = []
    fp_count = 0
    fn_count = 0

    # Group predictions by stem for fast lookup
    pred_by_stem: Dict[str, List[Any]] = {}
    for stem, group in pred_df.groupby(pred_image_ids):
        pred_by_stem[stem] = group["segmentation_rle"].tolist()

    for annotator_view in annotator_views:
        # Extract physical image stem from '<annotator_id>-<stem>'
        stem = annotator_view.split("-", 1)[1] if "-" in annotator_view else annotator_view

        gt_rles = gt_df.loc[gt_annotator_image_ids == annotator_view, "segmentation_rle"].tolist()
        pred_rles = pred_by_stem.get(stem, [])

        n_gt = len(gt_rles)
        n_pred = len(pred_rles)

        if n_gt == 0:
            fp_count += n_pred
            continue
        if n_pred == 0:
            fn_count += n_gt
            continue

        if fast_coco:
            # Fast vectorized C evaluation via pycocotools
            gt_dicts = [
                {"size": [HEIGHT, WIDTH], "counts": r.encode("ascii") if isinstance(r, str) else r}
                for r in gt_rles
            ]
            dt_dicts = [
                {"size": [HEIGHT, WIDTH], "counts": r.encode("ascii") if isinstance(r, str) else r}
                for r in pred_rles
            ]
            iou_matrix = mask_util.iou(dt_dicts, gt_dicts, [0] * n_gt).T  # Transpose to (n_gt, n_pred)
        else:
            # Exact pixel-stack method matching host notebook
            gt_layers = rles_to_binary_layers(gt_rles)
            pred_layers = rles_to_binary_layers(pred_rles)
            iou_matrix, _ = compute_pairwise_iou_dice(gt_layers, pred_layers)

        # Strict inequality per host: IoU > 0.50
        hit_matrix = iou_matrix > iou_threshold

        tp_iou_scores.extend(iou_matrix[hit_matrix].tolist())

        # Unmatched predictions: column sum in hit_matrix == 0
        fp_count += int((hit_matrix.sum(axis=0) == 0).sum())
        # Unmatched GT filaments: row sum in hit_matrix == 0
        fn_count += int((hit_matrix.sum(axis=1) == 0).sum())

    tp_count = len(tp_iou_scores)
    denominator = tp_count + 0.5 * fp_count + 0.5 * fn_count

    sq = float(np.mean(tp_iou_scores)) if tp_count > 0 else 0.0
    rq = float(tp_count / denominator) if denominator > 0 else 0.0
    pq = float(sum(tp_iou_scores) / denominator) if denominator > 0 else 0.0

    return {
        "pq": pq,
        "sq": sq,
        "rq": rq,
        "tp": tp_count,
        "fp": fp_count,
        "fn": fn_count,
        "tp_ious": tp_iou_scores,
        "num_annotator_views": len(annotator_views),
        "eval_time_sec": time.time() - t0,
    }


def format_pq_summary(results: Dict[str, Any], title: str = "PQ EVALUATION REPORT") -> str:
    """Returns a cleanly formatted ASCII table of PQ evaluation results."""
    lines = [
        "=" * 60,
        f"{title:^60}",
        "=" * 60,
        f"Evaluated Views : {results['num_annotator_views']}",
        f"True Positives  : {results['tp']}",
        f"False Positives : {results['fp']}",
        f"False Negatives : {results['fn']}",
        f"Segmentation Q  : {results['sq']:.4f} (Mean IoU of TP pairs)",
        f"Recognition Q   : {results['rq']:.4f} (F1-score)",
        f"PANOPTIC QUALITY: {results['pq']:.5f}",
        f"Evaluation Time : {results['eval_time_sec']:.2f}s",
        "=" * 60,
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    print("Panoptic Quality Evaluator loaded successfully.")
