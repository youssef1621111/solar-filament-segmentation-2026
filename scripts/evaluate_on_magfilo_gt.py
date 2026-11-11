#!/usr/bin/env python3
"""
Evaluate any submission CSV against the official MAGFiLO Test Ground Truth in sub-second time.
Uses compiled C pycocotools.mask.iou for 0.75s full-dataset evaluation.
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from pycocotools import mask as mask_util


def evaluate_submission(sub_csv_path, gt_pkl_path="/home/fathyyoussef162/data/magfilo_test_gt.pkl"):
    if not Path(gt_pkl_path).exists():
        gt_pkl_path = "/tmp/magfilo_test_gt.pkl"

    t0 = time.time()
    df_gt = pd.read_pickle(gt_pkl_path)
    df_pred = pd.read_csv(sub_csv_path)

    gt_annotator_image_ids = df_gt['filament_id'].str.split('_', n=1).str[0]
    pred_image_ids = df_pred['filament_id'].str.rsplit('_', n=1).str[0]

    annotator_images = gt_annotator_image_ids.unique()

    tp_iou_scores = []
    fp_count = 0
    fn_count = 0

    for annotator_image in annotator_images:
        image_id = annotator_image.split('-', maxsplit=1)[1]
        gt_rles = df_gt.loc[gt_annotator_image_ids == annotator_image, 'segmentation_rle'].tolist()
        pred_rles = df_pred.loc[pred_image_ids == image_id, 'segmentation_rle'].tolist()

        n_gt = len(gt_rles)
        n_pred = len(pred_rles)

        if n_gt == 0:
            fp_count += n_pred
            continue
        if n_pred == 0:
            fn_count += n_gt
            continue

        dt_dicts = [{'size': [2048, 2048], 'counts': r.encode('ascii') if isinstance(r, str) else r} for r in pred_rles]
        gt_dicts = [{'size': [2048, 2048], 'counts': r.encode('ascii') if isinstance(r, str) else r} for r in gt_rles]

        iou_matrix = mask_util.iou(dt_dicts, gt_dicts, [0] * n_gt)
        hit_matrix = iou_matrix > 0.5

        tp_iou_scores.extend(iou_matrix[hit_matrix].tolist())
        fp_count += int((hit_matrix.sum(axis=1) == 0).sum())
        fn_count += int((hit_matrix.sum(axis=0) == 0).sum())

    tp_count = len(tp_iou_scores)
    denom = tp_count + 0.5 * fp_count + 0.5 * fn_count
    pq = sum(tp_iou_scores) / denom if denom > 0 else 0.0
    mean_iou = sum(tp_iou_scores) / max(tp_count, 1)

    print("=" * 50)
    print(f"RESULTS FOR {sub_csv_path}")
    print(f"Evaluated in {time.time()-t0:.2f}s")
    print(f"Total True Positives (TP): {tp_count}")
    print(f"Total False Positives (FP): {fp_count}")
    print(f"Total False Negatives (FN): {fn_count}")
    print(f"Mean IoU of Matched Filaments: {mean_iou:.4f}")
    print(f"PANOPTIC QUALITY (PQ): {pq:.5f}")
    print("=" * 50)
    return pq, mean_iou, tp_count, fp_count, fn_count


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "/home/fathyyoussef162/experiments/submission_championship_tier1_eroded.csv"
    evaluate_submission(path)
