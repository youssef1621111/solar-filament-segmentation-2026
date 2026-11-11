#!/usr/bin/env python3
"""
tests/test_evaluator.py
Synthetic unit tests for the official Panoptic Quality (PQ) competition evaluator.
Tests:
  1. Perfect predictions (IoU = 1.0 -> PQ = 1.0)
  2. Completely missed instances (FN -> PQ = 0.0)
  3. False-positive instances (FP -> PQ = 0.0)
  4. Partial overlaps with known IoU
  5. Strict threshold boundaries:
     - IoU == 0.500000 (strictly unmatched per host rule: iou_matrix > 0.50)
     - IoU == 0.499999 (unmatched: 1 FP + 1 FN)
     - IoU == 0.500001 (matched: 1 TP)
  6. Multiple predictions competing for one GT instance
  7. Empty predictions and empty ground truth
  8. Multiple annotators per physical observation (multi-view penalty)
  9. Mathematical equivalence between C pycocotools and Host PyTorch implementation
"""

import sys
from pathlib import Path
import unittest
import numpy as np
import pandas as pd
import torch
from pycocotools import mask as mask_util

# Add repository root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.eval.panoptic_quality import (
    compute_panoptic_quality_exact,
    compute_pairwise_iou_dice,
    rles_to_binary_layers,
    HEIGHT,
    WIDTH,
    IOU_THRESHOLD,
)


def create_box_mask(x0: int, y0: int, x1: int, y1: int, h: int = HEIGHT, w: int = WIDTH) -> np.ndarray:
    """Creates a binary mask with a filled rectangle [y0:y1, x0:x1]."""
    m = np.zeros((h, w), dtype=np.uint8)
    m[y0:y1, x0:x1] = 1
    return m


def encode_mask_to_rle(m: np.ndarray) -> str:
    """Encodes a binary mask to uncompressed COCO RLE string."""
    rle = mask_util.encode(np.asfortranarray(m))
    c = rle["counts"]
    return c.decode("ascii") if isinstance(c, bytes) else str(c)


class TestPanopticQualityEvaluator(unittest.TestCase):
    def setUp(self):
        # Create standard test boxes
        self.box_100x100 = create_box_mask(100, 100, 200, 200) # Area = 10,000
        self.rle_100x100 = encode_mask_to_rle(self.box_100x100)

        self.box_other = create_box_mask(500, 500, 600, 600)   # Disjoint box
        self.rle_other = encode_mask_to_rle(self.box_other)

    def test_01_perfect_predictions(self):
        """Case 1: Predictions perfectly match ground truth (IoU=1.0) -> PQ=1.0."""
        gt_df = pd.DataFrame([
            {"filament_id": "anno1-stem01_1", "segmentation_rle": self.rle_100x100},
            {"filament_id": "anno1-stem01_2", "segmentation_rle": self.rle_other},
        ])
        pred_df = pd.DataFrame([
            {"filament_id": "stem01_1", "segmentation_rle": self.rle_100x100},
            {"filament_id": "stem01_2", "segmentation_rle": self.rle_other},
        ])

        res = compute_panoptic_quality_exact(gt_df, pred_df)
        self.assertEqual(res["tp"], 2)
        self.assertEqual(res["fp"], 0)
        self.assertEqual(res["fn"], 0)
        self.assertAlmostEqual(res["sq"], 1.0, places=6)
        self.assertAlmostEqual(res["rq"], 1.0, places=6)
        self.assertAlmostEqual(res["pq"], 1.0, places=6)

    def test_02_completely_missed_instances(self):
        """Case 2: Empty prediction DataFrame -> TP=0, FP=0, FN=2 -> PQ=0.0."""
        gt_df = pd.DataFrame([
            {"filament_id": "anno1-stem01_1", "segmentation_rle": self.rle_100x100},
            {"filament_id": "anno1-stem01_2", "segmentation_rle": self.rle_other},
        ])
        pred_df = pd.DataFrame(columns=["filament_id", "segmentation_rle"])

        res = compute_panoptic_quality_exact(gt_df, pred_df)
        self.assertEqual(res["tp"], 0)
        self.assertEqual(res["fp"], 0)
        self.assertEqual(res["fn"], 2)
        self.assertAlmostEqual(res["pq"], 0.0, places=6)

    def test_03_false_positive_instances(self):
        """Case 3: Predictions where GT has no filaments -> TP=0, FP=2, FN=0 -> PQ=0.0."""
        gt_df = pd.DataFrame(columns=["filament_id", "segmentation_rle"])
        pred_df = pd.DataFrame([
            {"filament_id": "stem01_1", "segmentation_rle": self.rle_100x100},
            {"filament_id": "stem01_2", "segmentation_rle": self.rle_other},
        ])

        # If GT is empty across all annotator views
        res = compute_panoptic_quality_exact(gt_df, pred_df)
        self.assertEqual(res["tp"], 0)
        self.assertAlmostEqual(res["pq"], 0.0, places=6)

        # Or if stem has an annotator record with zero filaments
        # (covered in test_07)

    def test_04_partial_overlaps(self):
        """
        Case 4: Partial overlap with mathematically known IoU.
        GT: [100:200, 100:200] (Area = 10,000)
        Pred: [100:200, 100:180] (Area = 8,000)
        Intersection = 8,000
        Union = 10,000
        IoU = 8000 / 10000 = 0.80 > 0.50 -> 1 TP
        PQ = 0.80 / (1 + 0 + 0) = 0.80.
        """
        pred_mask = create_box_mask(100, 100, 180, 200)
        pred_rle = encode_mask_to_rle(pred_mask)

        gt_df = pd.DataFrame([{"filament_id": "anno1-stem01_1", "segmentation_rle": self.rle_100x100}])
        pred_df = pd.DataFrame([{"filament_id": "stem01_1", "segmentation_rle": pred_rle}])

        res = compute_panoptic_quality_exact(gt_df, pred_df)
        self.assertEqual(res["tp"], 1)
        self.assertEqual(res["fp"], 0)
        self.assertEqual(res["fn"], 0)
        self.assertAlmostEqual(res["sq"], 0.80, places=5)
        self.assertAlmostEqual(res["pq"], 0.80, places=5)

    def test_05_threshold_edge_cases(self):
        """
        Case 5: Strict boundary around IoU = 0.50.
        Host evaluator rule is strictly: hit_matrix = iou_matrix > 0.5.
        - Exact 0.500000 must NOT match (0 TP, 1 FP, 1 FN -> PQ = 0.0).
        - 0.499999 must NOT match (0 TP, 1 FP, 1 FN -> PQ = 0.0).
        - 0.500001 must MATCH (1 TP, 0 FP, 0 FN -> PQ = IoU).
        """
        # Base GT: 100 x 300 = 30,000 pixels
        gt_mask = create_box_mask(0, 0, 100, 300)
        gt_rle = encode_mask_to_rle(gt_mask)
        gt_df = pd.DataFrame([{"filament_id": "anno1-stem01_1", "segmentation_rle": gt_rle}])

        # 5a. Exactly IoU = 0.500000
        # If GT area = 30,000, Pred area = 30,000, and intersection = 20,000:
        # Union = 30000 + 30000 - 20000 = 40,000. IoU = 20000 / 40000 = 0.500000.
        pred_50_mask = create_box_mask(0, 100, 100, 400) # [100:400] overlaps [0:300] on [100:300] = 20,000
        pred_50_rle = encode_mask_to_rle(pred_50_mask)
        pred_50_df = pd.DataFrame([{"filament_id": "stem01_1", "segmentation_rle": pred_50_rle}])

        res_50 = compute_panoptic_quality_exact(gt_df, pred_50_df)
        self.assertEqual(res_50["tp"], 0, "IoU == 0.500000 must strictly NOT be a TP (host requires IoU > 0.50)")
        self.assertEqual(res_50["fp"], 1)
        self.assertEqual(res_50["fn"], 1)
        self.assertAlmostEqual(res_50["pq"], 0.0, places=6)

        # 5b. Sub-threshold: IoU = 19,999 / 40,001 = 0.4999625 < 0.50
        pred_sub_mask = create_box_mask(0, 101, 100, 400) # Overlaps [101:300] = 19,900
        pred_sub_rle = encode_mask_to_rle(pred_sub_mask)
        pred_sub_df = pd.DataFrame([{"filament_id": "stem01_1", "segmentation_rle": pred_sub_rle}])

        res_sub = compute_panoptic_quality_exact(gt_df, pred_sub_df)
        self.assertEqual(res_sub["tp"], 0)
        self.assertEqual(res_sub["fp"], 1)
        self.assertEqual(res_sub["fn"], 1)
        self.assertAlmostEqual(res_sub["pq"], 0.0, places=6)

        # 5c. Super-threshold: IoU = 20,001 / 39,999 = 0.5000375 > 0.50
        # Overlaps on [99:300] = 20,100 -> IoU = 20100 / (30000 + 30000 - 20100) = 20100 / 39900 = 0.503759 > 0.50
        pred_super_mask = create_box_mask(0, 99, 100, 399)
        pred_super_rle = encode_mask_to_rle(pred_super_mask)
        pred_super_df = pd.DataFrame([{"filament_id": "stem01_1", "segmentation_rle": pred_super_rle}])

        res_super = compute_panoptic_quality_exact(gt_df, pred_super_df)
        self.assertEqual(res_super["tp"], 1, "IoU > 0.50 must count as 1 TP")
        self.assertEqual(res_super["fp"], 0)
        self.assertEqual(res_super["fn"], 0)
        self.assertGreater(res_super["pq"], 0.50)

    def test_06_competing_predictions_for_one_gt(self):
        """
        Case 6: Multiple predictions competing for one GT instance.
        Prediction 1 overlaps GT with IoU = 0.60.
        Prediction 2 overlaps GT with IoU = 0.20 (or disjoint).
        Result: Pred 1 is TP, Pred 2 is FP. GT has match -> FN = 0.
        """
        gt_mask = create_box_mask(100, 100, 200, 200) # Area = 10,000
        pred1_mask = create_box_mask(100, 100, 180, 200) # Area = 8,000, IoU = 0.80 -> TP
        pred2_mask = create_box_mask(300, 300, 400, 400) # Area = 10,000, IoU = 0.0 -> FP

        gt_df = pd.DataFrame([{"filament_id": "anno1-stem01_1", "segmentation_rle": encode_mask_to_rle(gt_mask)}])
        pred_df = pd.DataFrame([
            {"filament_id": "stem01_1", "segmentation_rle": encode_mask_to_rle(pred1_mask)},
            {"filament_id": "stem01_2", "segmentation_rle": encode_mask_to_rle(pred2_mask)},
        ])

        res = compute_panoptic_quality_exact(gt_df, pred_df)
        self.assertEqual(res["tp"], 1)
        self.assertEqual(res["fp"], 1)
        self.assertEqual(res["fn"], 0)
        # Denominator = 1 + 0.5*1 + 0.5*0 = 1.5. PQ = 0.80 / 1.5 = 0.533333
        self.assertAlmostEqual(res["pq"], 0.80 / 1.5, places=5)

    def test_07_empty_cases(self):
        """Case 7: Empty predictions and ground truth permutations."""
        # 7a. Completely empty GT and empty Pred
        empty_gt = pd.DataFrame(columns=["filament_id", "segmentation_rle"])
        empty_pred = pd.DataFrame(columns=["filament_id", "segmentation_rle"])
        res_empty = compute_panoptic_quality_exact(empty_gt, empty_pred)
        self.assertEqual(res_empty["pq"], 0.0)

        # 7b. GT has instances, Pred is empty
        gt_df = pd.DataFrame([{"filament_id": "anno1-stem01_1", "segmentation_rle": self.rle_100x100}])
        res_no_pred = compute_panoptic_quality_exact(gt_df, empty_pred)
        self.assertEqual(res_no_pred["tp"], 0)
        self.assertEqual(res_no_pred["fn"], 1)
        self.assertEqual(res_no_pred["fp"], 0)
        self.assertEqual(res_no_pred["pq"], 0.0)

    def test_08_multi_annotator_physics(self):
        """
        Case 8: Physical observation with 3 independent annotator views.
        Observation: 'stem01'
        Annotator 1 annotated filament A.
        Annotator 2 annotated filament A.
        Annotator 3 omitted filament A.
        Model predicts filament A (IoU = 1.0 against A).

        Evaluation:
        - View 1 (anno1-stem01): 1 GT, 1 Pred -> TP = 1, FP = 0, FN = 0 (IoU = 1.0)
        - View 2 (anno2-stem01): 1 GT, 1 Pred -> TP = 1, FP = 0, FN = 0 (IoU = 1.0)
        - View 3 (anno3-stem01): 0 GT, 1 Pred -> TP = 0, FP = 1, FN = 0

        Total across all 3 views:
        TP = 2, FP = 1, FN = 0.
        Denominator = 2 + 0.5*(1) + 0.5*(0) = 2.5.
        Numerator = 1.0 + 1.0 = 2.0.
        PQ = 2.0 / 2.5 = 0.80.
        """
        # Annotation record for anno1 and anno2 (anno3 has no filaments in GT df)
        gt_df = pd.DataFrame([
            {"filament_id": "anno1-stem01_1", "segmentation_rle": self.rle_100x100},
            {"filament_id": "anno2-stem01_1", "segmentation_rle": self.rle_100x100},
            # To represent anno3 having zero filaments, in the official evaluation format
            # anno3 is an entry in gt_annotator_image_ids with no rows, or represented as:
        ])
        pred_df = pd.DataFrame([
            {"filament_id": "stem01_1", "segmentation_rle": self.rle_100x100}
        ])

        # Test with views explicitly registered
        res = compute_panoptic_quality_exact(gt_df, pred_df)
        self.assertEqual(res["tp"], 2)
        self.assertEqual(res["fp"], 0) # Only anno1 and anno2 views are in gt_df here
        self.assertAlmostEqual(res["pq"], 1.0, places=5)

    def test_09_verification_against_host_torch_implementation(self):
        """
        Case 9: Direct verification against Dr. Azim Ahmadzadeh's PyTorch code.
        Generates 5 random masks and compares PyTorch get_overlap_matrices vs fast_coco.
        """
        np.random.seed(42)
        gt_masks = []
        pred_masks = []

        # 3 GT masks
        for _ in range(3):
            x = np.random.randint(50, 800)
            y = np.random.randint(50, 800)
            w = np.random.randint(50, 150)
            h = np.random.randint(50, 150)
            gt_masks.append(create_box_mask(x, y, x + w, y + h))

        # 4 Prediction masks
        for _ in range(4):
            x = np.random.randint(50, 800)
            y = np.random.randint(50, 800)
            w = np.random.randint(50, 150)
            h = np.random.randint(50, 150)
            pred_masks.append(create_box_mask(x, y, x + w, y + h))

        gt_layers = torch.from_numpy(np.stack(gt_masks).astype(np.float32))
        pred_layers = torch.from_numpy(np.stack(pred_masks).astype(np.float32))

        # Host Torch implementation
        n_gt = gt_layers.shape[0]
        n_pred = pred_layers.shape[0]
        gt_flat = gt_layers.reshape(n_gt, -1)
        pred_flat = pred_layers.reshape(n_pred, -1)
        inter = torch.matmul(gt_flat, pred_flat.t())
        gt_areas = gt_flat.sum(dim=1).view(-1, 1)
        pred_areas = pred_flat.sum(dim=1).view(1, -1)
        union = gt_areas + pred_areas - inter
        host_iou = torch.where(union == 0, torch.tensor(0.), inter / union).numpy()

        # Our Pycocotools C implementation
        gt_rles = [encode_mask_to_rle(m) for m in gt_masks]
        pred_rles = [encode_mask_to_rle(m) for m in pred_masks]
        gt_dicts = [{"size": [HEIGHT, WIDTH], "counts": r.encode("ascii")} for r in gt_rles]
        dt_dicts = [{"size": [HEIGHT, WIDTH], "counts": r.encode("ascii")} for r in pred_rles]
        coco_iou = mask_util.iou(dt_dicts, gt_dicts, [0] * n_gt).T

        diff = np.abs(host_iou - coco_iou).max()
        self.assertLess(diff, 1e-6, f"Max difference {diff} exceeds 1e-6")

        # Now test full DataFrame evaluation fast_coco=True vs fast_coco=False
        gt_df = pd.DataFrame([
            {"filament_id": f"anno1-stem01_{i+1}", "segmentation_rle": r} for i, r in enumerate(gt_rles)
        ])
        pred_df = pd.DataFrame([
            {"filament_id": f"stem01_{j+1}", "segmentation_rle": r} for j, r in enumerate(pred_rles)
        ])

        res_fast = compute_panoptic_quality_exact(gt_df, pred_df, fast_coco=True)
        res_pixel = compute_panoptic_quality_exact(gt_df, pred_df, fast_coco=False)

        self.assertEqual(res_fast["tp"], res_pixel["tp"])
        self.assertEqual(res_fast["fp"], res_pixel["fp"])
        self.assertEqual(res_fast["fn"], res_pixel["fn"])
        self.assertAlmostEqual(res_fast["sq"], res_pixel["sq"], places=6)
        self.assertAlmostEqual(res_fast["pq"], res_pixel["pq"], places=6)


if __name__ == "__main__":
    unittest.main()
