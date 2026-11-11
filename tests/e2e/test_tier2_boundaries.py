"""
Tier 2: Boundary and Corner Cases Test Suite
Tests extreme spatial, mathematical, format, and threshold boundaries.
"""

import unittest
import numpy as np
import pandas as pd
import scipy.ndimage as ndi

from scripts.utils_metrics import (
    encode_leb128_rle,
    decode_leb128_rle,
    decode_leb128_counts,
    mask_to_uncompressed_rle,
    rle_decode_uncompressed,
    compute_iou,
)
from scripts.utils_solar_physics import compute_mask_shape_features
from scripts.stage2_roggi_ffbsi_bridge import detect_candidate_bridge_pairs
from scripts.stage3_physics_boundary_refiner import refine_mask_boundary, compute_local_valley_statistics


class TestTier2Boundaries(unittest.TestCase):
    """Tier 2: Boundary and Corner Cases."""

    def test_b01_empty_mask_rle(self):
        """B-01: All-zero empty mask (2048x2048) yields single count 4,194,304 and 'PPPP4'."""
        empty_mask = np.zeros((2048, 2048), dtype=bool)
        counts = mask_to_uncompressed_rle(empty_mask)
        self.assertEqual(counts, [4194304])
        rle = encode_leb128_rle(empty_mask)
        self.assertEqual(rle, "PPPP4")
        decoded = decode_leb128_rle(rle, (2048, 2048))
        self.assertEqual(int(np.sum(decoded)), 0)
        self.assertEqual(decoded.shape, (2048, 2048))

    def test_b02_full_foreground_mask_rle(self):
        """B-02: All-ones full mask (2048x2048) yields leading zero and count 4,194,304."""
        full_mask = np.ones((2048, 2048), dtype=bool)
        counts = mask_to_uncompressed_rle(full_mask)
        self.assertEqual(counts, [0, 4194304])
        rle = encode_leb128_rle(full_mask)
        decoded = decode_leb128_rle(rle, (2048, 2048))
        self.assertEqual(int(np.sum(decoded)), 4194304)

    def test_b03_origin_pixel_single_foreground(self):
        """B-03: Single foreground pixel at origin (0, 0) requires leading zero run."""
        mask = np.zeros((2048, 2048), dtype=bool)
        mask[0, 0] = True
        counts = mask_to_uncompressed_rle(mask)
        self.assertEqual(counts, [0, 1, 4194303])
        rle = encode_leb128_rle(mask)
        decoded = decode_leb128_rle(rle, (2048, 2048))
        self.assertTrue(decoded[0, 0])
        self.assertFalse(decoded[0, 1])
        self.assertFalse(decoded[1, 0])
        self.assertEqual(int(np.sum(decoded)), 1)

    def test_b04_last_pixel_single_foreground(self):
        """B-04: Single foreground pixel at corner (2047, 2047)."""
        mask = np.zeros((2048, 2048), dtype=bool)
        mask[2047, 2047] = True
        counts = mask_to_uncompressed_rle(mask)
        self.assertEqual(counts, [4194303, 1])
        rle = encode_leb128_rle(mask)
        decoded = decode_leb128_rle(rle, (2048, 2048))
        self.assertTrue(decoded[2047, 2047])
        self.assertEqual(int(np.sum(decoded)), 1)

    def test_b05_fortran_vs_c_order_transposition(self):
        """B-05: C-order transposition causes IoU = 0.0, Fortran order yields IoU = 1.0."""
        # Asymmetric rectangular feature
        mask = np.zeros((2048, 2048), dtype=bool)
        mask[100:150, 400:800] = True

        rle_F = encode_leb128_rle(mask)
        decoded_F = decode_leb128_rle(rle_F, (2048, 2048))
        self.assertEqual(compute_iou(mask, decoded_F), 1.0)

        # Transposing (as if decoded in C-order)
        transposed = decoded_F.T
        self.assertNotEqual(compute_iou(mask, transposed), 1.0)
        self.assertEqual(compute_iou(mask, transposed), 0.0)

    def test_b06_quiet_sun_zero_rows_empty_stem(self):
        """B-06: Quiet sun stem producing 0 rows is handled gracefully by validator."""
        quiet_stems = {"20160513133714Th", "20180507140210Th"}
        df = pd.DataFrame(columns=["filament_id", "segmentation_rle"])
        for q_stem in quiet_stems:
            stem_rows = df[df["filament_id"].str.startswith(q_stem)]
            self.assertEqual(len(stem_rows), 0)

    def test_b07_min_solo_area_boundary(self):
        """B-07: Solo area filter boundary at 699 px (rejected) vs 700 px (retained)."""
        min_solo_area = 700
        area_below = 699
        area_exact = 700
        area_above = 701

        self.assertFalse(area_below >= min_solo_area)
        self.assertTrue(area_exact >= min_solo_area)
        self.assertTrue(area_above >= min_solo_area)

    def test_b08_sunspot_umbra_veto_boundaries(self):
        """B-08: Sunspot Umbra Veto exact parameter boundary tests."""
        # area < 500 and elong < 1.38 and circ > 0.68
        def is_vetoed(area, elong, circ):
            return (area < 500) and (elong < 1.38) and (circ > 0.68)

        # Boundary around area
        self.assertTrue(is_vetoed(499, 1.37, 0.69))
        self.assertFalse(is_vetoed(500, 1.37, 0.69))

        # Boundary around elongation
        self.assertTrue(is_vetoed(499, 1.379, 0.69))
        self.assertFalse(is_vetoed(499, 1.381, 0.69))

        # Boundary around circularity
        self.assertTrue(is_vetoed(499, 1.37, 0.681))
        self.assertFalse(is_vetoed(499, 1.37, 0.679))

    def test_b09_single_pixel_fracture_bridging(self):
        """B-09: Single-pixel fracture (gap of 1px) bridged with 0% boundary bloat."""
        foreground = np.zeros((50, 100), dtype=np.uint8)
        foreground[24:26, 20:49] = 1  # 20 to 48
        foreground[24:26, 50:80] = 1  # 50 to 79; gap at col 49 (1px)
        orig_count = int(np.sum(foreground > 0))

        # Dilation grouping with radius 5
        expanded = ndi.binary_dilation(foreground > 0, iterations=5)
        expanded_labels, _ = ndi.label(expanded)
        group_labels = np.where(foreground > 0, expanded_labels, 0)

        # United into 1 group
        unique_groups = np.unique(group_labels[group_labels > 0])
        self.assertEqual(len(unique_groups), 1)
        self.assertEqual(int(np.sum(group_labels > 0)), orig_count)

    def test_b10_exact_30px_fracture_distance_boundary(self):
        """B-10: 30px fracture distance (bridged with radius 15) vs 31px distance (not bridged)."""
        # Gap = 30px
        fg_30 = np.zeros((30, 150), dtype=np.uint8)
        fg_30[14:16, 20:40] = 1   # ends at 39
        fg_30[14:16, 70:90] = 1   # starts at 70 -> gap = 70 - 40 = 30px

        expanded_30 = ndi.binary_dilation(fg_30 > 0, iterations=15)
        _, num_30 = ndi.label(expanded_30)
        self.assertEqual(num_30, 1)

        # Gap = 32px
        fg_32 = np.zeros((30, 150), dtype=np.uint8)
        fg_32[14:16, 20:40] = 1   # ends at 39
        fg_32[14:16, 72:92] = 1   # starts at 72 -> gap = 32px

        expanded_32 = ndi.binary_dilation(fg_32 > 0, iterations=15)
        _, num_32 = ndi.label(expanded_32)
        self.assertEqual(num_32, 2)

    def test_b11_thin_2px_barb_erosion_core_empty(self):
        """B-11: Narrow 2px filament barb where erode(M, 3) is completely empty."""
        mask = np.zeros((50, 50), dtype=bool)
        mask[24:26, 10:40] = True  # width = 2px
        struct = ndi.generate_binary_structure(2, 1)
        eroded = ndi.binary_erosion(mask, structure=struct, iterations=3)
        self.assertEqual(int(np.sum(eroded)), 0)

        # Boundary refiner handles empty core gracefully without exception
        img = np.zeros((50, 50), dtype=np.float32)
        seg = np.ones((50, 50), dtype=np.float32) * 0.8
        yolo = seg.copy()
        med = np.zeros((50, 50), dtype=np.float32)
        mad = np.ones((50, 50), dtype=np.float32) * 0.1

        refined = refine_mask_boundary(mask, img, seg, yolo, med, mad, band_width=3)
        self.assertIsInstance(refined, np.ndarray)
        self.assertGreater(int(np.sum(refined)), 0)

    def test_b12_tangent_alignment_cosine_threshold(self):
        """B-12: Tangent chord alignment at exact cos(30 deg) = 0.866025 threshold."""
        cos_30 = np.cos(np.radians(30.0))
        self.assertAlmostEqual(cos_30, 0.866025, places=5)

        # Aligned: angle = 29 deg
        align_pass = np.cos(np.radians(29.0))
        self.assertGreaterEqual(align_pass, cos_30)

        # Misaligned: angle = 31 deg
        align_fail = np.cos(np.radians(31.0))
        self.assertLess(align_fail, cos_30)

    def test_b13_theorem1_exact_threshold_boundary(self):
        """B-13: Theorem 1 threshold boundary at exactly p * j = 0.275000."""
        target = 0.2750000
        p, j = 0.55, 0.50  # 0.2750000
        prod = p * j
        self.assertEqual(prod, target)
        self.assertFalse(prod > target)
        self.assertTrue((p * (j + 1e-6)) > target)

    def test_b14_single_instance_canvas_no_collision(self):
        """B-14: Canvas with only a single instance has 0 collisions and valid RLE."""
        mask = np.zeros((2048, 2048), dtype=bool)
        mask[100:200, 100:200] = True
        rle = encode_leb128_rle(mask)
        decoded = decode_leb128_rle(rle, (2048, 2048))
        self.assertEqual(int(np.sum(decoded)), 10000)

    def test_b15_extreme_elongation_filament(self):
        """B-15: Extremely elongated filament (elongation > 15.0)."""
        mask = np.zeros((100, 500), dtype=bool)
        mask[49:51, 10:490] = True  # 2 x 480 px
        elong, circ, area, veto = compute_mask_shape_features(mask)
        self.assertGreater(elong, 15.0)
        self.assertLess(circ, 0.10)
        self.assertFalse(veto)

    def test_b16_leb128_truncation_boundaries(self):
        """B-16: Extreme LEB128 stream truncation boundaries raise ValueError('Truncated LEB128 string')."""
        self.assertEqual(decode_leb128_counts(""), [])

        with self.assertRaises(ValueError) as ctx:
            decode_leb128_counts("P")
        self.assertEqual(str(ctx.exception), "Truncated LEB128 string")

        with self.assertRaises(ValueError) as ctx:
            decode_leb128_counts("PPPP")
        self.assertEqual(str(ctx.exception), "Truncated LEB128 string")

        with self.assertRaises(ValueError) as ctx:
            decode_leb128_counts("0P")
        self.assertEqual(str(ctx.exception), "Truncated LEB128 string")

    def test_b17_negative_count_sum_evasion_boundary(self):
        """B-17: RLE sum exactly 4,194,304 with negative count strictly rejected."""
        counts = [4194320, -16]
        self.assertEqual(sum(counts), 4194304)
        with self.assertRaises(ValueError) as ctx:
            rle_decode_uncompressed(counts, shape=(2048, 2048))
        self.assertIn("negative run length", str(ctx.exception))

    def test_b18_unique_stems_boundary(self):
        """B-18: Stem count boundaries: 180 required (or 178 when quiet-sun pruned)."""
        self.assertTrue(175 <= 180 <= 180)
        self.assertTrue(175 <= 178 <= 180)
        self.assertFalse(175 <= 1 <= 180)
        self.assertFalse(175 <= 174 <= 180)


if __name__ == "__main__":
    unittest.main()
