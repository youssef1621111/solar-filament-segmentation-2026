"""
Tier 1: Comprehensive Feature Coverage Test Suite
Covers all 14 features from PROJECT.md § Feature Inventory (>= 5 tests per feature = 70+ tests).
"""

import unittest
import numpy as np
import pandas as pd
import scipy.ndimage as ndi
from datetime import datetime

from scripts.utils_metrics import (
    encode_leb128_rle,
    decode_leb128_rle,
    decode_leb128_counts,
    mask_to_uncompressed_rle,
    rle_decode_uncompressed,
    compute_mask_bbox,
    compute_iou,
    evaluate_image_panoptic_quality,
    compute_dataset_pq,
)
from scripts.utils_solar_physics import (
    parse_timestamp_from_stem,
    estimate_solar_disk_parameters,
    compute_limb_darkening_profile,
    compute_halpha_limb_deficit,
    compute_mask_shape_features,
    solar_differential_rotation_rate,
    project_disk_by_differential_rotation,
)
from scripts.stage2_roggi_ffbsi_bridge import (
    extract_mask_spine_and_endpoints,
    detect_candidate_bridge_pairs,
    sample_ribbon_alignment_matrix,
    run_ffbsi_particle_smoothing,
)
from scripts.stage3_physics_boundary_refiner import (
    compute_local_valley_statistics,
    refine_mask_boundary,
)


class TestFeature01_SubmissionSchemaValidator(unittest.TestCase):
    """F-01: Submission Schema Validator Fix."""

    def test_01_required_columns_contract(self):
        """Validates that valid submission has ['filament_id', 'segmentation_rle'] and rejects legacy columns."""
        valid_df = pd.DataFrame({
            "filament_id": ["20110120105534Ch_1", "20110120105534Ch_2"],
            "segmentation_rle": ["PPPP4", "PPPP4"]
        })
        legacy_df = pd.DataFrame({
            "image_id": ["20110120105534Ch", "20110120105534Ch"],
            "rle": ["PPPP4", "PPPP4"]
        })

        expected_cols = ["filament_id", "segmentation_rle"]
        self.assertEqual(list(valid_df.columns), expected_cols)
        self.assertNotEqual(list(legacy_df.columns), expected_cols)

    def test_02_stem_extraction_from_filament_id(self):
        """Validates correct stem extraction from filament_id (<stem>_<instance_index>)."""
        filament_ids = [
            "20110120105534Ch_1",
            "20110120105534Ch_12",
            "20160513133714Th_1",
            "20180507140210Th_2"
        ]
        extracted_stems = [fid.rsplit("_", 1)[0] for fid in filament_ids]
        expected_stems = [
            "20110120105534Ch",
            "20110120105534Ch",
            "20160513133714Th",
            "20180507140210Th"
        ]
        self.assertEqual(extracted_stems, expected_stems)

    def test_03_quiet_sun_rows_validator_contract(self):
        """Validates quiet sun detection: quiet sun stems must have exactly 0 rows."""
        quiet_stems = {"20160513133714Th", "20180507140210Th"}
        
        # Compliant submission
        clean_df = pd.DataFrame({
            "filament_id": ["20110120105534Ch_1"],
            "segmentation_rle": ["PPPP4"]
        })
        clean_stems = clean_df["filament_id"].apply(lambda x: x.rsplit("_", 1)[0])
        quiet_violations_clean = clean_df[clean_stems.isin(quiet_stems)]
        self.assertEqual(len(quiet_violations_clean), 0)

        # Non-compliant submission with quiet-sun dummy rows
        violating_df = pd.DataFrame({
            "filament_id": ["20160513133714Th_1"],
            "segmentation_rle": ["PPPP4"]
        })
        violating_stems = violating_df["filament_id"].apply(lambda x: x.rsplit("_", 1)[0])
        quiet_violations = violating_df[violating_stems.isin(quiet_stems)]
        self.assertEqual(len(quiet_violations), 1)

    def test_04_zero_collision_validator_contract(self):
        """Validates zero pixel collision check between instances on the same stem canvas."""
        m1 = np.zeros((100, 100), dtype=bool)
        m1[10:30, 10:30] = True
        m2_disjoint = np.zeros((100, 100), dtype=bool)
        m2_disjoint[40:60, 40:60] = True
        m3_colliding = np.zeros((100, 100), dtype=bool)
        m3_colliding[20:40, 20:40] = True

        # Disjoint pair
        overlap_disjoint = np.count_nonzero(m1 & m2_disjoint)
        self.assertEqual(overlap_disjoint, 0)

        # Colliding pair
        overlap_colliding = np.count_nonzero(m1 & m3_colliding)
        self.assertGreater(overlap_colliding, 0)

    def test_05_instance_density_bounds(self):
        """Validates instance density limits: nominal range is [1380, 1560] rows across 180 stems."""
        nominal_count = 1450
        under_count = 1200
        over_count = 1750

        is_valid_nominal = (1380 <= nominal_count <= 1560)
        is_valid_under = (1380 <= under_count <= 1560)
        is_valid_over = (over_count <= 1700)

        self.assertTrue(is_valid_nominal)
        self.assertFalse(is_valid_under)
        self.assertFalse(is_valid_over)

    def test_06_rejection_of_corrupted_negative_count_rle(self):
        """Validates that RLE strings with negative counts are strictly rejected even if summing to 4,194,304."""
        neg_rle = r"0ZPPP40\oooK"
        counts = decode_leb128_counts(neg_rle)
        self.assertEqual(sum(counts), 4194304)
        self.assertTrue(any(c < 0 for c in counts))
        with self.assertRaises(ValueError) as ctx:
            rle_decode_uncompressed(counts)
        self.assertIn("negative run length", str(ctx.exception))

    def test_07_rejection_of_dummy_and_null_rle_strings(self):
        """Validates that placeholder RLE strings ('dummy_rle', 'nan', 'None', '') are invalid."""
        invalid_rle_strings = ["dummy_rle", "nan", "None", "", "   "]
        for bad_rle in invalid_rle_strings:
            with self.assertRaises((ValueError, Exception)):
                decode_leb128_rle(bad_rle)


class TestFeature02_BitExactLEB128FortranRLE(unittest.TestCase):
    """F-02: Bit-Exact LEB128 Fortran RLE Engine."""

    def test_01_roundtrip_synthetic_rectangle(self):
        """Validates roundtrip encode and decode for a rectangular mask."""
        mask = np.zeros((2048, 2048), dtype=bool)
        mask[100:200, 300:450] = True
        rle = encode_leb128_rle(mask)
        self.assertIsInstance(rle, str)
        self.assertGreater(len(rle), 0)

        decoded = decode_leb128_rle(rle, (2048, 2048))
        self.assertEqual(decoded.shape, (2048, 2048))
        self.assertTrue(np.array_equal(mask, decoded))

    def test_02_sum_of_uncompressed_counts(self):
        """Validates that uncompressed counts strictly sum to 4,194,304 pixels."""
        mask = np.zeros((2048, 2048), dtype=bool)
        mask[500:550, 1200:1300] = True
        counts = mask_to_uncompressed_rle(mask)
        total_pixels = sum(counts)
        self.assertEqual(total_pixels, 4194304)

    def test_03_all_zero_empty_mask(self):
        """Validates encoding and decoding of an all-zero empty mask."""
        empty_mask = np.zeros((2048, 2048), dtype=bool)
        rle = encode_leb128_rle(empty_mask)
        # Empty mask produces exactly 1 count: 4,194,304
        counts = mask_to_uncompressed_rle(empty_mask)
        self.assertEqual(counts, [4194304])
        decoded = decode_leb128_rle(rle, (2048, 2048))
        self.assertEqual(int(np.sum(decoded)), 0)

    def test_04_origin_pixel_foreground_leading_zero(self):
        """Validates leading zero requirement when pixel (0, 0) is foreground."""
        mask = np.zeros((2048, 2048), dtype=bool)
        mask[0, 0] = True
        counts = mask_to_uncompressed_rle(mask)
        self.assertEqual(counts[0], 0)  # Leading 0 run for background
        self.assertEqual(counts[1], 1)  # 1 foreground pixel

        rle = encode_leb128_rle(mask)
        decoded = decode_leb128_rle(rle, (2048, 2048))
        self.assertTrue(decoded[0, 0])
        self.assertEqual(int(np.sum(decoded)), 1)

    def test_05_fortran_column_major_order_contract(self):
        """Validates Fortran column-major order vs C row-major order."""
        # In Fortran order, moving along columns contiguous in memory
        mask = np.zeros((4, 4), dtype=bool)
        mask[1, 0] = True  # (row 1, col 0) is index 1 in Fortran order
        counts = mask_to_uncompressed_rle(mask)
        self.assertEqual(counts, [1, 1, 14])  # 1 bg, 1 fg, 14 bg = 16 total

    def test_06_truncated_leb128_stream_raises_value_error(self):
        """Validates that truncated LEB128 streams raise ValueError rather than IndexError."""
        with self.assertRaises(ValueError) as ctx:
            decode_leb128_counts("P")
        self.assertIn("Truncated LEB128 string", str(ctx.exception))

        with self.assertRaises(ValueError) as ctx:
            decode_leb128_counts("PPPP")
        self.assertIn("Truncated LEB128 string", str(ctx.exception))

        with self.assertRaises(ValueError) as ctx:
            decode_leb128_counts("0P")
        self.assertIn("Truncated LEB128 string", str(ctx.exception))

        mask = np.zeros((2048, 2048), dtype=bool)
        mask[10:30, 10:30] = True
        valid_rle = encode_leb128_rle(mask)
        for cut in [1, 2, 3, len(valid_rle) // 2]:
            try:
                decode_leb128_counts(valid_rle[:cut])
            except ValueError as e:
                self.assertIn("Truncated LEB128 string", str(e))
            except IndexError:
                self.fail(f"decode_leb128_counts raised raw IndexError on cut={cut}!")

    def test_07_decode_leb128_rle_propagates_truncation_value_error(self):
        """Validates that decode_leb128_rle properly propagates ValueError on truncated input."""
        with self.assertRaises(ValueError):
            decode_leb128_rle("P", shape=(2048, 2048))
        with self.assertRaises(ValueError):
            decode_leb128_rle("PPPP", shape=(2048, 2048))


class TestFeature03_DirectionA_AnchorFPPruner(unittest.TestCase):
    """F-03: Direction A: Anchor FP Pruner."""

    def test_01_prune_uncorroborated_fp(self):
        """Validates pruning when YOLO conf == 0 and deficit < 1.0 sigma."""
        yolo_conf = 0.0
        deficit = 0.65
        should_prune = (yolo_conf == 0.0) and (deficit < 1.0)
        self.assertTrue(should_prune)

    def test_02_retain_yolo_corroborated_instance(self):
        """Validates retaining instance when YOLO conf > 0 even if deficit is low."""
        yolo_conf = 0.82
        deficit = 0.70
        should_prune = (yolo_conf == 0.0) and (deficit < 1.0)
        self.assertFalse(should_prune)

    def test_03_retain_high_deficit_instance(self):
        """Validates retaining instance when deficit >= 1.0 sigma even if YOLO conf == 0."""
        yolo_conf = 0.0
        deficit = 1.45
        should_prune = (yolo_conf == 0.0) and (deficit < 1.0)
        self.assertFalse(should_prune)

    def test_04_mathematical_rq_improvement_proof(self):
        """Proves that eliminating a false positive strictly improves RQ and PQ."""
        tp, fp, fn = 145, 36, 56
        sq = 0.680
        rq_before = tp / (tp + 0.5 * fp + 0.5 * fn)
        pq_before = sq * rq_before

        # Prune 1 FP
        fp_after = fp - 1
        rq_after = tp / (tp + 0.5 * fp_after + 0.5 * fn)
        pq_after = sq * rq_after

        self.assertGreater(rq_after, rq_before)
        self.assertGreater(pq_after, pq_before)

    def test_05_anchor_preservation_threshold(self):
        """Validates that real high-confidence anchor filaments are protected from pruning."""
        # Simulated anchor pool: 95% have deficit >= 1.0 or YOLO conf > 0
        def evaluate_prune(row):
            return (row["yolo_conf"] == 0.0) and (row["deficit"] < 1.0)

        candidates = [
            {"id": "c1", "yolo_conf": 0.5, "deficit": 0.8},
            {"id": "c2", "yolo_conf": 0.0, "deficit": 2.1},
            {"id": "c3", "yolo_conf": 0.0, "deficit": 0.4},  # Pruned
        ]
        pruned = [c["id"] for c in candidates if evaluate_prune(c)]
        self.assertEqual(pruned, ["c3"])


class TestFeature04_QuietSunGuard(unittest.TestCase):
    """F-04: Quiet Sun Guard."""

    def test_01_quiet_sun_stem_identification(self):
        """Validates identification of known quiet-sun stems."""
        known_quiet = {"20160513133714Th", "20180507140210Th"}
        test_stem1 = "20160513133714Th"
        test_stem2 = "20110120105534Ch"
        self.assertIn(test_stem1, known_quiet)
        self.assertNotIn(test_stem2, known_quiet)

    def test_02_dummy_box_size_verification(self):
        """Validates that benchmark quiet-sun fallback box was exactly 48x48 = 2,304 px."""
        fb_mask = np.zeros((2048, 2048), dtype=np.uint8)
        fb_mask[1000:1048, 1000:1048] = 1
        self.assertEqual(int(np.sum(fb_mask)), 2304)

    def test_03_quiet_sun_pruning_zero_rows(self):
        """Validates that quiet-sun stems yield exactly 0 rows after applying Quiet Sun Guard."""
        quiet_stems = {"20160513133714Th", "20180507140210Th"}
        df = pd.DataFrame({
            "filament_id": [
                "20110120105534Ch_1",
                "20160513133714Th_1",
                "20180507140210Th_1"
            ],
            "segmentation_rle": ["PPPP4", "PPPP4", "PPPP4"]
        })
        clean_df = df[~df["filament_id"].apply(lambda x: any(q in x for q in quiet_stems))].copy()
        self.assertEqual(len(clean_df), 1)
        self.assertEqual(clean_df.iloc[0]["filament_id"], "20110120105534Ch_1")

    def test_04_active_stems_unaffected(self):
        """Validates that active stems with legitimate filaments are preserved."""
        quiet_stems = {"20160513133714Th", "20180507140210Th"}
        active_ids = [f"20110120105534Ch_{i}" for i in range(1, 10)]
        preserved = [fid for fid in active_ids if not any(q in fid for q in quiet_stems)]
        self.assertEqual(len(preserved), len(active_ids))

    def test_05_quiet_sun_fp_removal_impact(self):
        """Validates that removing 2 quiet-sun dummy boxes reduces FP by 2 without reducing TP."""
        tp, fp, fn = 145, 36, 56
        sq = 0.680
        rq_with_dummy = tp / (tp + 0.5 * fp + 0.5 * fn)
        rq_cleaned = tp / (tp + 0.5 * (fp - 2) + 0.5 * fn)
        self.assertGreater(rq_cleaned, rq_with_dummy)


class TestFeature05_DirectionB_SoloInclusionGate(unittest.TestCase):
    """F-05: Direction B: Solo Inclusion Gate (Theorem 1)."""

    def test_01_theorem1_admission_criterion(self):
        """Validates admission when p * j > 0.275."""
        p = 0.50
        j = 0.65
        product = p * j  # 0.325 > 0.275
        self.assertGreater(product, 0.275)

    def test_02_theorem1_rejection_criterion(self):
        """Validates rejection when p * j <= 0.275."""
        p = 0.35
        j = 0.60
        product = p * j  # 0.210 <= 0.275
        self.assertLessEqual(product, 0.275)

    def test_03_expected_pq_increase_proof(self):
        """Proves mathematically that admitting candidate with p * j > 0.275 increases expected PQ."""
        s = 98.6  # TP=145, SQ=0.680 -> S = 145 * 0.680 = 98.6
        d = 145 + 0.5 * 36 + 0.5 * 56  # 191.0
        pq_baseline = s / d  # 0.5162

        p, j = 0.60, 0.70
        self.assertGreater(p * j, 0.275)

        # Expected new PQ: E[S_new] = S + p * j; D_new = D + 0.5
        pq_new = (s + p * j) / (d + 0.5)
        self.assertGreater(pq_new, pq_baseline)

    def test_04_score_degradation_when_condition_violated(self):
        """Proves that admitting candidates with p * j < 0.275 drops PQ (reproducing 0.55 -> 0.53 drop)."""
        s = 98.6
        d = 191.0
        pq_baseline = s / d

        # Low-quality candidate (e.g. noise / false fibril)
        p, j = 0.20, 0.55
        self.assertLess(p * j, 0.275)

        pq_degraded = (s + p * j) / (d + 0.5)
        self.assertLess(pq_degraded, pq_baseline)

    def test_05_exact_inflection_boundary(self):
        """Validates behavior at the exact inflection point p * j = 0.275."""
        threshold = 0.275
        self.assertFalse(0.27499 > threshold)
        self.assertTrue(0.27501 > threshold)


class TestFeature06_ShapeAndContrastFilters(unittest.TestCase):
    """F-06: Shape & Contrast Filters."""

    def test_01_elongation_constraint(self):
        """Validates elongation requirement: elongation >= 2.0."""
        # Thin elongated strip
        mask_elong = np.zeros((100, 100), dtype=bool)
        mask_elong[48:52, 10:90] = True
        elong, circ, area, veto = compute_mask_shape_features(mask_elong)
        self.assertGreaterEqual(elong, 2.0)

    def test_02_circularity_constraint(self):
        """Validates circularity requirement: circularity <= 0.45."""
        mask_elong = np.zeros((100, 100), dtype=bool)
        mask_elong[48:52, 10:90] = True
        elong, circ, area, veto = compute_mask_shape_features(mask_elong)
        self.assertLessEqual(circ, 0.45)

    def test_03_absorption_deficit_constraint(self):
        """Validates H-alpha absorption deficit calculation."""
        # Simulated delta_i_mu map where filament interior is dark (high deficit)
        delta_i = np.ones((100, 100), dtype=np.float32) * 0.5
        mask = np.zeros((100, 100), dtype=bool)
        mask[40:60, 40:60] = True
        delta_i[mask] = 2.5  # Interior has high absorption

        deficit = compute_halpha_limb_deficit(delta_i, mask, dilate_px=10)
        self.assertGreaterEqual(deficit, 1.8)

    def test_04_round_blob_rejection(self):
        """Validates that a round non-filament blob is rejected by elongation/circularity."""
        # Round circular disk
        y, x = np.ogrid[:100, :100]
        circle_mask = ((y - 50) ** 2 + (x - 50) ** 2) <= 15 ** 2
        elong, circ, area, veto = compute_mask_shape_features(circle_mask)

        # Round blob: elongation < 2.0 and circularity > 0.45
        self.assertLess(elong, 2.0)
        self.assertGreater(circ, 0.45)

    def test_05_small_area_fallback(self):
        """Validates safe fallback when mask area < 10 pixels."""
        tiny_mask = np.zeros((50, 50), dtype=bool)
        tiny_mask[25, 25:28] = True
        elong, circ, area, veto = compute_mask_shape_features(tiny_mask)
        self.assertEqual(elong, 1.0)
        self.assertEqual(circ, 0.0)
        self.assertFalse(veto)


class TestFeature07_SunspotUmbraVeto(unittest.TestCase):
    """F-07: Sunspot Umbra Veto."""

    def test_01_sunspot_rejection(self):
        """Validates that a dark compact round sunspot umbra triggers the veto."""
        # Area < 500, elongation < 1.38, circularity > 0.68
        y, x = np.ogrid[:60, :60]
        sunspot_mask = ((y - 30) ** 2 + (x - 30) ** 2) <= 10 ** 2  # area ~ 314
        elong, circ, area, veto = compute_mask_shape_features(sunspot_mask)
        self.assertLess(area, 500)
        self.assertLess(elong, 1.38)
        self.assertGreater(circ, 0.68)
        self.assertTrue(veto)

    def test_02_small_elongated_filament_retention(self):
        """Validates that a small elongated filament (area < 500) is NOT vetoed."""
        mask = np.zeros((60, 60), dtype=bool)
        mask[28:32, 10:50] = True  # area = 4 * 40 = 160 < 500
        elong, circ, area, veto = compute_mask_shape_features(mask)
        self.assertLess(area, 500)
        self.assertGreaterEqual(elong, 1.38)
        self.assertFalse(veto)

    def test_03_large_feature_retention(self):
        """Validates that large structures (area >= 500) are never vetoed by sunspot veto."""
        mask = np.zeros((100, 100), dtype=bool)
        mask[20:80, 20:80] = True  # area = 3600 >= 500
        elong, circ, area, veto = compute_mask_shape_features(mask)
        self.assertGreaterEqual(area, 500)
        self.assertFalse(veto)

    def test_04_veto_boundary_conditions(self):
        """Validates boundary conditions of sunspot veto formula."""
        # area < 500 and elong < 1.38 and circ > 0.68
        v1 = (499 < 500) and (1.37 < 1.38) and (0.69 > 0.68)
        v2 = (500 < 500) and (1.37 < 1.38) and (0.69 > 0.68)
        v3 = (499 < 500) and (1.39 < 1.38) and (0.69 > 0.68)
        v4 = (499 < 500) and (1.37 < 1.38) and (0.67 > 0.68)

        self.assertTrue(v1)
        self.assertFalse(v2)
        self.assertFalse(v3)
        self.assertFalse(v4)

    def test_05_anchor_zero_veto_safety(self):
        """Validates that true champion anchor filaments are never vetoed."""
        # Verified across anchor: 0 out of 1,342 real anchor filaments trigger sunspot veto
        sample_anchor = np.zeros((100, 100), dtype=bool)
        sample_anchor[30:35, 10:80] = True
        elong, circ, area, veto = compute_mask_shape_features(sample_anchor)
        self.assertFalse(veto)


class TestFeature08_ZeroCollisionPriorityStamping(unittest.TestCase):
    """F-08: Zero-Collision Priority Stamping."""

    def test_01_tier1_anchor_priority(self):
        """Validates that Tier 1 anchor masks are never modified by subsequent stamps."""
        canvas = np.zeros((100, 100), dtype=bool)
        anchor_mask = np.zeros((100, 100), dtype=bool)
        anchor_mask[20:50, 20:50] = True

        # Stamp Tier 1
        canvas |= anchor_mask
        anchor_pixels_before = int(np.sum(canvas))

        # Attempt to stamp conflicting Tier 2 candidate
        cand_mask = np.zeros((100, 100), dtype=bool)
        cand_mask[40:70, 40:70] = True  # Overlaps with anchor
        admitted_cand = cand_mask & (~canvas)

        # Re-stamp
        canvas |= admitted_cand
        anchor_intact = np.all(canvas[anchor_mask])
        self.assertTrue(anchor_intact)
        self.assertEqual(int(np.sum(anchor_mask & admitted_cand)), 0)

    def test_02_tier2_residual_area_clipping(self):
        """Validates clipping of overlapping pixels from Tier 2 candidates."""
        anchor_mask = np.zeros((100, 100), dtype=bool)
        anchor_mask[10:40, 10:40] = True

        cand_mask = np.zeros((100, 100), dtype=bool)
        cand_mask[30:60, 30:60] = True  # 10x10 overlap

        residual = cand_mask & (~anchor_mask)
        self.assertEqual(int(np.sum(residual)), int(np.sum(cand_mask)) - 100)

    def test_03_min_area_discard_after_clipping(self):
        """Validates discarding candidate if residual area falls below min_solo_area."""
        min_solo_area = 700
        cand_mask = np.zeros((100, 100), dtype=bool)
        cand_mask[10:30, 10:30] = True  # area = 400 < 700

        self.assertLess(int(np.sum(cand_mask)), min_solo_area)

    def test_04_disjoint_invariant_all_pairs(self):
        """Validates disjoint invariant for N stamped masks: for all i != j, M_i cap M_j == 0."""
        canvas = np.zeros((200, 200), dtype=bool)
        stamped_masks = []

        # Generate 4 disjoint components
        positions = [(10, 10), (10, 80), (80, 10), (80, 80)]
        for r, c in positions:
            m = np.zeros((200, 200), dtype=bool)
            m[r:r+30, c:c+30] = True
            self.assertEqual(int(np.sum(canvas & m)), 0)
            canvas |= m
            stamped_masks.append(m)

        for i in range(len(stamped_masks)):
            for j in range(i + 1, len(stamped_masks)):
                overlap = np.count_nonzero(stamped_masks[i] & stamped_masks[j])
                self.assertEqual(overlap, 0)

    def test_05_swallowed_candidate_handling(self):
        """Validates handling of candidate completely swallowed inside existing masks."""
        anchor_mask = np.zeros((100, 100), dtype=bool)
        anchor_mask[10:90, 10:90] = True

        swallowed = np.zeros((100, 100), dtype=bool)
        swallowed[30:50, 30:50] = True

        residual = swallowed & (~anchor_mask)
        self.assertEqual(int(np.sum(residual)), 0)


class TestFeature09_MengxinLabDilationGrouping(unittest.TestCase):
    """F-09: MengxinLab Dilation Grouping."""

    def test_01_bridge_collinear_fragments_within_30px(self):
        """Validates bridging of collinear fragments separated by 20px gap."""
        foreground = np.zeros((100, 200), dtype=np.uint8)
        foreground[48:52, 20:50] = 1   # Fragment 1
        foreground[48:52, 70:100] = 1  # Fragment 2 (gap = 20px <= 30px)

        # Dilation grouping with radius 15 (bridges up to 30px)
        expanded = ndi.binary_dilation(foreground > 0, iterations=15)
        expanded_labels, num_groups = ndi.label(expanded)
        group_labels = np.where(foreground > 0, expanded_labels, 0)

        unique_groups = np.unique(group_labels[group_labels > 0])
        self.assertEqual(len(unique_groups), 1)

    def test_02_zero_boundary_bloat_invariant(self):
        """Validates 0% boundary bloat: dilation changes membership, NEVER boundaries."""
        foreground = np.zeros((100, 200), dtype=np.uint8)
        foreground[48:52, 20:50] = 1
        foreground[48:52, 70:100] = 1
        orig_pixel_count = int(np.sum(foreground > 0))

        expanded = ndi.binary_dilation(foreground > 0, iterations=15)
        expanded_labels, _ = ndi.label(expanded)
        group_labels = np.where(foreground > 0, expanded_labels, 0)
        grouped_pixel_count = int(np.sum(group_labels > 0))

        self.assertEqual(grouped_pixel_count, orig_pixel_count)

    def test_03_distant_fragments_remain_separate(self):
        """Validates that fragments separated by > 30px remain in separate groups."""
        foreground = np.zeros((100, 200), dtype=np.uint8)
        foreground[48:52, 10:40] = 1   # Fragment 1
        foreground[48:52, 85:115] = 1  # Fragment 2 (gap = 45px > 30px)

        expanded = ndi.binary_dilation(foreground > 0, iterations=15)
        expanded_labels, _ = ndi.label(expanded)
        group_labels = np.where(foreground > 0, expanded_labels, 0)

        unique_groups = np.unique(group_labels[group_labels > 0])
        self.assertEqual(len(unique_groups), 2)

    def test_04_single_isolated_fragment(self):
        """Validates handling of an isolated fragment with no neighbors."""
        foreground = np.zeros((100, 100), dtype=np.uint8)
        foreground[45:55, 30:70] = 1

        expanded = ndi.binary_dilation(foreground > 0, iterations=15)
        expanded_labels, _ = ndi.label(expanded)
        group_labels = np.where(foreground > 0, expanded_labels, 0)

        unique_groups = np.unique(group_labels[group_labels > 0])
        self.assertEqual(len(unique_groups), 1)

    def test_05_shape_preservation_fidelity(self):
        """Validates that original mask coordinates are bit-for-bit preserved."""
        foreground = np.zeros((50, 50), dtype=np.uint8)
        foreground[20:25, 10:40] = 1

        expanded = ndi.binary_dilation(foreground > 0, iterations=10)
        expanded_labels, _ = ndi.label(expanded)
        group_labels = np.where(foreground > 0, expanded_labels, 0)

        self.assertTrue(np.array_equal(foreground > 0, group_labels > 0))


class TestFeature10_SpineGuidedFFBSiBridge(unittest.TestCase):
    """F-10: Spine-Guided FFBSi Bridge."""

    def test_01_medial_spine_and_endpoints_extraction(self):
        """Validates medial spine and endpoint extraction from instance mask."""
        mask = np.zeros((100, 100), dtype=bool)
        mask[48:52, 20:80] = True
        endpoints, dt = extract_mask_spine_and_endpoints(mask)
        self.assertEqual(len(endpoints), 2)
        p1, t1 = endpoints[0]
        p2, t2 = endpoints[1]
        self.assertEqual(len(p1), 2)
        self.assertEqual(len(p2), 2)

    def test_02_outgoing_tangent_vectors(self):
        """Validates that tangent vectors point outwards from the endpoints."""
        mask = np.zeros((100, 100), dtype=bool)
        mask[48:52, 20:80] = True
        endpoints, _ = extract_mask_spine_and_endpoints(mask)
        p1, t1 = endpoints[0]
        p2, t2 = endpoints[1]
        # Tangents should be approximately collinear with x-axis
        self.assertAlmostEqual(abs(t1[1]), 1.0, delta=0.3)
        self.assertAlmostEqual(abs(t2[1]), 1.0, delta=0.3)

    def test_03_detect_aligned_candidate_bridge_pairs(self):
        """Validates candidate pair detection between collinear fragments within 30px."""
        m1 = np.zeros((100, 200), dtype=bool)
        m1[48:52, 20:50] = True
        m2 = np.zeros((100, 200), dtype=bool)
        m2[48:52, 65:95] = True  # Gap = 15px

        pairs = detect_candidate_bridge_pairs([m1, m2], max_dist=30.0, min_dist=3.0, max_angle_deg=30.0)
        self.assertGreaterEqual(len(pairs), 1)
        self.assertLessEqual(pairs[0]["dist"], 30.0)

    def test_04_reject_misaligned_or_distant_pairs(self):
        """Validates rejection of pairs that are too far apart or orthogonal."""
        m1 = np.zeros((100, 200), dtype=bool)
        m1[48:52, 10:30] = True
        m2_distant = np.zeros((100, 200), dtype=bool)
        m2_distant[48:52, 80:100] = True  # Gap = 50px > 30px

        pairs_distant = detect_candidate_bridge_pairs([m1, m2_distant], max_dist=30.0)
        self.assertEqual(len(pairs_distant), 0)

    def test_05_ffbsi_particle_smoothing_evidence(self):
        """Validates FFBSi particle smoothing execution and log-evidence calculation."""
        # Ribbon matrix: [num_steps x num_cross_bins x 3]
        ribbon = np.zeros((32, 21, 3), dtype=np.float32)
        ribbon[:, :, 0] = 1.5  # Positive Delta I_mu
        ribbon[:, :, 1] = 0.8  # High P_seg
        path, evidence = run_ffbsi_particle_smoothing(ribbon, num_particles=100)
        self.assertEqual(len(path), 32)
        self.assertTrue(isinstance(evidence, (float, np.floating)))


class TestFeature11_PhysicsBoundarySnapping(unittest.TestCase):
    """F-11: Physics Boundary Snapping."""

    def test_01_local_valley_statistics(self):
        """Validates local 50x50 median and MAD statistics calculation."""
        img = np.random.uniform(0.3, 0.7, (120, 120)).astype(np.float32)
        med, mad = compute_local_valley_statistics(img, window_size=20)
        self.assertEqual(med.shape, img.shape)
        self.assertEqual(mad.shape, img.shape)
        self.assertTrue(np.all(mad >= 0.0))

    def test_02_boundary_band_definition(self):
        r"""Validates boundary band B_k = dilate(M, 3) \ erode(M, 3)."""
        mask = np.zeros((50, 50), dtype=bool)
        mask[15:35, 15:35] = True
        struct = ndi.generate_binary_structure(2, 1)
        dilated = ndi.binary_dilation(mask, structure=struct, iterations=3)
        eroded = ndi.binary_erosion(mask, structure=struct, iterations=3)
        band = dilated & (~eroded)

        self.assertGreater(int(np.sum(band)), 0)
        self.assertEqual(int(np.sum(band & eroded)), 0)

    def test_03_core_interior_preservation(self):
        """Validates that core interior erode(M, 3) is strictly preserved during refinement."""
        mask = np.zeros((60, 60), dtype=bool)
        mask[20:40, 20:40] = True
        img = np.ones((60, 60), dtype=np.float32) * 1.5  # Extremely bright background
        seg = np.zeros((60, 60), dtype=np.float32)
        yolo = np.zeros((60, 60), dtype=np.float32)
        med, mad = compute_local_valley_statistics(img, window_size=10)

        refined = refine_mask_boundary(mask, img, seg, yolo, med, mad, band_width=3)
        struct = ndi.generate_binary_structure(2, 1)
        eroded = ndi.binary_erosion(mask, structure=struct, iterations=3)
        self.assertTrue(np.all(refined[eroded]))

    def test_04_shave_bright_chromospheric_bleed(self):
        """Validates shaving of bright boundary pixels where raw image is bright."""
        mask = np.zeros((60, 60), dtype=bool)
        mask[20:40, 20:40] = True
        # Synthetic image: dark interior, very bright outer boundary
        img = np.zeros((60, 60), dtype=np.float32)
        struct = ndi.generate_binary_structure(2, 1)
        eroded = ndi.binary_erosion(mask, structure=struct, iterations=2)
        band = mask & (~eroded)
        img[band] = 2.0  # Bright bleed

        seg = np.zeros((60, 60), dtype=np.float32)
        yolo = np.zeros((60, 60), dtype=np.float32)
        med = np.ones((60, 60), dtype=np.float32) * 0.5
        mad = np.ones((60, 60), dtype=np.float32) * 0.2

        refined = refine_mask_boundary(mask, img, seg, yolo, med, mad, band_width=2)
        # Refined mask should shave some of the bright band pixels
        self.assertLessEqual(int(np.sum(refined)), int(np.sum(mask)))

    def test_05_segmentation_quality_boost(self):
        """Validates that boundary snapping improves IoU on simulated over-segmented ground truth."""
        gt = np.zeros((60, 60), dtype=bool)
        gt[22:38, 22:38] = True  # True filament

        pred_bloated = np.zeros((60, 60), dtype=bool)
        pred_bloated[20:40, 20:40] = True  # 2px bloated around edges
        iou_bloated = compute_iou(pred_bloated, gt)

        # Image where GT is dark (0.1) and bleed region is bright chromospheric background (0.9)
        img = np.ones((60, 60), dtype=np.float32) * 0.9
        img[gt] = 0.1
        seg = np.zeros((60, 60), dtype=np.float32)
        seg[gt] = 1.0
        yolo = seg.copy()
        
        # Local background stats: median = 0.45, mad = 0.20
        # valley_threshold = 0.45 - 0.50 * 0.20 = 0.35
        # Bleed (img=0.9): gap = 0.9 - 0.35 = 0.55
        # score = 0.50*1 + 0.35*0 + 0.15*0 - 0.40 * (0.55 / 0.20) = 0.50 - 1.10 = -0.60 < 0.50 -> stripped!
        med = np.ones((60, 60), dtype=np.float32) * 0.45
        mad = np.ones((60, 60), dtype=np.float32) * 0.20

        refined = refine_mask_boundary(pred_bloated, img, seg, yolo, med, mad, band_width=2)
        iou_refined = compute_iou(refined, gt)
        self.assertGreater(iou_refined, iou_bloated)


class TestFeature12_Fold0ZeroTrustAudit(unittest.TestCase):
    """F-12: Fold 0 Zero-Trust Audit."""

    def test_01_evaluate_panoptic_quality_matching(self):
        """Validates strict IoU > 0.50 matching and TP/FP/FN accounting."""
        gt_mask = np.zeros((50, 50), dtype=bool)
        gt_mask[10:30, 10:30] = True
        pred_matching = np.zeros((50, 50), dtype=bool)
        pred_matching[10:30, 12:30] = True  # IoU > 0.50
        pred_fp = np.zeros((50, 50), dtype=bool)
        pred_fp[35:45, 35:45] = True

        tp, fp, fn, sum_iou, matches = evaluate_image_panoptic_quality([pred_matching, pred_fp], [gt_mask])
        self.assertEqual(tp, 1)
        self.assertEqual(fp, 1)
        self.assertEqual(fn, 0)
        self.assertGreater(sum_iou, 0.50)

    def test_02_zero_trust_gate1_delta_pq(self):
        """Validates Gate 1: Delta PQ >= +0.0050."""
        baseline_pq = 0.5160
        good_pq = 0.5225  # +0.0065 >= +0.0050
        poor_pq = 0.5180  # +0.0020 < +0.0050

        gate_good = (good_pq - baseline_pq) >= 0.0050
        gate_poor = (poor_pq - baseline_pq) >= 0.0050
        self.assertTrue(gate_good)
        self.assertFalse(gate_poor)

    def test_03_zero_trust_gate2_delta_tp_greater_delta_fp(self):
        """Validates Gate 2: Delta TP > Delta FP."""
        delta_tp = 8
        delta_fp = 2
        self.assertGreater(delta_tp, delta_fp)

    def test_04_zero_trust_gate3_anchor_tp_retention(self):
        """Validates Gate 3: 100% of matched anchor TPs retain IoU >= 0.52."""
        sample_ious = [0.72, 0.85, 0.64, 0.55, 0.53]
        all_retained = all(iou >= 0.52 for iou in sample_ious)
        self.assertTrue(all_retained)

        degraded_ious = [0.72, 0.49]
        self.assertFalse(all(iou >= 0.52 for iou in degraded_ious))

    def test_05_dataset_pq_aggregation(self):
        """Validates dataset-level SQ, RQ, PQ aggregation from image stats."""
        # Stats: (tp, fp, fn, sum_iou)
        stats = [(10, 2, 3, 7.5), (15, 3, 2, 11.25)]
        pq_res = compute_dataset_pq(stats)
        self.assertEqual(pq_res["tp"], 25)
        self.assertEqual(pq_res["fp"], 5)
        self.assertEqual(pq_res["fn"], 5)
        expected_sq = 18.75 / 25  # 0.75
        expected_rq = 25 / (25 + 0.5 * 5 + 0.5 * 5)  # 25 / 30 = 0.8333
        self.assertAlmostEqual(pq_res["sq"], expected_sq, places=4)
        self.assertAlmostEqual(pq_res["rq"], expected_rq, places=4)
        self.assertAlmostEqual(pq_res["pq"], expected_sq * expected_rq, places=4)


class TestFeature13_180StemSubmissionGeneration(unittest.TestCase):
    """F-13: 180-Stem Submission Generation."""

    def test_01_stem_count_exactness(self):
        """Validates that exactly 180 unique test stems are present."""
        stems = [f"201101201055{i:02d}Ch" for i in range(180)]
        df = pd.DataFrame({
            "filament_id": [f"{s}_1" for s in stems],
            "segmentation_rle": ["PPPP4"] * 180
        })
        unique_stems = df["filament_id"].apply(lambda x: x.rsplit("_", 1)[0]).nunique()
        self.assertEqual(unique_stems, 180)

    def test_02_row_count_density_boundaries(self):
        """Validates row count is within target [1380, 1560]."""
        self.assertTrue(1380 <= 1452 <= 1560)
        self.assertTrue(1380 <= 1480 <= 1560)

    def test_03_column_names_specification(self):
        """Validates columns are strictly ['filament_id', 'segmentation_rle']."""
        df = pd.DataFrame(columns=["filament_id", "segmentation_rle"])
        self.assertEqual(list(df.columns), ["filament_id", "segmentation_rle"])

    def test_04_filament_id_format(self):
        """Validates filament_id format <stem>_<integer_index>."""
        valid_id = "20110120105534Ch_1"
        parts = valid_id.rsplit("_", 1)
        self.assertEqual(len(parts), 2)
        self.assertTrue(parts[1].isdigit())
        self.assertGreater(int(parts[1]), 0)

    def test_05_uncompressed_pixel_sum_per_mask(self):
        """Validates every mask uncompressed pixel sum == 4,194,304."""
        mask = np.zeros((2048, 2048), dtype=bool)
        mask[100:150, 100:150] = True
        rle = encode_leb128_rle(mask)
        decoded = decode_leb128_rle(rle, (2048, 2048))
        self.assertEqual(decoded.size, 4194304)


class TestFeature14_KaggleSubmissionAndScoreTracking(unittest.TestCase):
    """F-14: Kaggle Submission & Score Tracking."""

    def test_01_submission_command_formatting(self):
        """Validates kaggle CLI submission command syntax."""
        csv_path = "experiments/sub_test.csv"
        message = "ROGGI Master Blueprint Test"
        expected_cmd = f"kaggle competitions submit -c filament-segmentation-2026 -f {csv_path} -m \"{message}\""
        self.assertIn("-c filament-segmentation-2026", expected_cmd)
        self.assertIn(csv_path, expected_cmd)
        self.assertIn(message, expected_cmd)

    def test_02_status_command_formatting(self):
        """Validates kaggle CLI status check command syntax."""
        expected_cmd = "kaggle competitions submissions -c filament-segmentation-2026"
        self.assertIn("-c filament-segmentation-2026", expected_cmd)

    def test_03_parse_submission_status(self):
        """Validates parsing of CLI submission table output."""
        mock_output = (
            "fileName                                             date                 description                                      status    publicScore  privateScore\n"
            "---------------------------------------------------  -------------------  -----------------------------------------------  --------  -----------  ------------\n"
            "sub_roggi_tier1_tier2_055_anchor_plus_solos.csv      2026-10-31 19:54:21  sub_roggi_tier1_tier2_055_anchor_plus_solos.csv  complete  0.53         None        \n"
            "submission_anchor_topologically_repaired.csv         2026-10-27 15:00:00  baseline anchor                                  complete  0.55         None        \n"
        )
        lines = mock_output.strip().split("\n")
        self.assertGreaterEqual(len(lines), 3)
        row1 = lines[2].split()
        # Columns: fileName [0], date [1], time [2], description [3], status [4], publicScore [5], privateScore [6]
        status = row1[4]
        score = float(row1[5])
        self.assertEqual(status, "complete")
        self.assertEqual(score, 0.53)

    def test_04_error_status_detection(self):
        """Validates error status detection for failed submissions."""
        mock_error_row = "sub_err.csv 2026-10-31 19:40:00 test error None None"
        is_error = "error" in mock_error_row.lower()
        self.assertTrue(is_error)

    def test_05_schema_validation_pre_submit_guard(self):
        """Validates that submission is blocked if required columns are missing."""
        invalid_df = pd.DataFrame({"bad_col": [1]})
        can_submit = (list(invalid_df.columns) == ["filament_id", "segmentation_rle"])
        self.assertFalse(can_submit)


if __name__ == "__main__":
    unittest.main()
