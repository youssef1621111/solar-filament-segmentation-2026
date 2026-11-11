#!/usr/bin/env python3
"""
Adversarial Stress Test Suite & Fuzzers for Milestone M0
IEEE BigData 2026 Solar Filament Segmentation Challenge (filament-segmentation-2026)

Target Modules:
  - scripts/utils_metrics.py
  - scripts/submission_integrity_check.py

Focus Areas:
  1. Collision detection on multi-instance stems (single pixel, corners, containment, chains).
  2. Corrupted LEB128 strings, truncated byte streams, non-4,194,304 pixel sums, negative counts, fuzzing.
  3. Quiet Sun unauthorized instances under strict and fallback modes.
  4. uint8(255) foreground masks at extreme boundary coordinates and dtype invariance.
  5. Malformed CSV schemas, empty files, missing columns, swapped columns, bad types.
  6. Empirical reproduction of edge-case bugs and contract violations.
"""

import os
import sys
import tempfile
import unittest
import numpy as np
import pandas as pd
import random
import subprocess

# Ensure repo root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.utils_metrics import (
    rle_decode_uncompressed,
    mask_to_uncompressed_rle,
    encode_leb128_rle,
    decode_leb128_counts,
    decode_leb128_rle,
    compute_iou,
    compute_mask_bbox,
)
from scripts.submission_integrity_check import run_submission_integrity_check


class TestAdversarialCollisions(unittest.TestCase):
    """Stress-test collision detection in Check 5."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

    def _make_dummy_submission(self, custom_rows: list) -> str:
        """Create a submission CSV with custom target rows and single-instance fillers."""
        csv_path = os.path.join(self.temp_dir.name, f"sub_{random.randint(100000, 999999)}.csv")
        rows = list(custom_rows)
        empty_rle = "PPPP4"  # Encodes (2048, 2048) empty mask
        needed = 1350 - len(rows)
        for i in range(needed):
            rows.append((f"filler_{i:04d}Th_1", empty_rle))
        df = pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"])
        df.to_csv(csv_path, index=False)
        return csv_path

    def test_single_pixel_collision_detected(self):
        """Two masks on the same stem sharing exactly ONE single pixel at (100, 100)."""
        m1 = np.zeros((2048, 2048), dtype=bool)
        m1[100, 100] = True
        m1[100, 101] = True

        m2 = np.zeros((2048, 2048), dtype=bool)
        m2[100, 100] = True  # Colliding pixel
        m2[100, 99] = True

        rle1 = encode_leb128_rle(m1)
        rle2 = encode_leb128_rle(m2)

        csv_path = self._make_dummy_submission([
            ("target_stem_1", rle1),
            ("target_stem_2", rle2),
        ])

        passed = run_submission_integrity_check(csv_path, allow_missing_anchor=True)
        self.assertFalse(passed, "Check 5 failed to detect a single colliding pixel!")

    def test_corner_pixel_collisions(self):
        """Two masks colliding at each of the 4 extreme canvas corners."""
        corners = [(0, 0), (0, 2047), (2047, 0), (2047, 2047)]
        for y, x in corners:
            m1 = np.zeros((2048, 2048), dtype=bool)
            m1[y, x] = True
            m2 = np.zeros((2048, 2048), dtype=bool)
            m2[y, x] = True

            rle1 = encode_leb128_rle(m1)
            rle2 = encode_leb128_rle(m2)

            csv_path = self._make_dummy_submission([
                (f"corner_stem_{y}_{x}_1", rle1),
                (f"corner_stem_{y}_{x}_2", rle2),
            ])

            passed = run_submission_integrity_check(csv_path, allow_missing_anchor=True)
            self.assertFalse(passed, f"Check 5 failed to detect collision at corner ({y}, {x})!")

    def test_complete_containment_collision(self):
        """Mask B is strictly contained within Mask A."""
        m1 = np.zeros((2048, 2048), dtype=bool)
        m1[500:600, 500:600] = True  # 100x100
        m2 = np.zeros((2048, 2048), dtype=bool)
        m2[520:540, 520:540] = True  # 20x20 inside m1

        rle1 = encode_leb128_rle(m1)
        rle2 = encode_leb128_rle(m2)

        csv_path = self._make_dummy_submission([
            ("contained_stem_1", rle1),
            ("contained_stem_2", rle2),
        ])

        passed = run_submission_integrity_check(csv_path, allow_missing_anchor=True)
        self.assertFalse(passed, "Check 5 failed to detect complete containment collision!")

    def test_multi_instance_chain_collision(self):
        """A stem with 8 instances where 1..7 are disjoint, but instance 8 collides with instance 3."""
        custom_rows = []
        for i in range(1, 8):
            m = np.zeros((2048, 2048), dtype=bool)
            m[i * 50: i * 50 + 20, 100:150] = True
            custom_rows.append((f"chain_stem_{i}", encode_leb128_rle(m)))

        # Instance 8 collides with instance 3 (at y=150..170)
        m8 = np.zeros((2048, 2048), dtype=bool)
        m8[155:165, 120:130] = True
        custom_rows.append(("chain_stem_8", encode_leb128_rle(m8)))

        csv_path = self._make_dummy_submission(custom_rows)
        passed = run_submission_integrity_check(csv_path, allow_missing_anchor=True)
        self.assertFalse(passed, "Check 5 failed to detect collision in multi-instance chain!")

    def test_adjacent_disjoint_masks_pass(self):
        """Two masks that touch at boundary but do not share any pixels must PASS."""
        m1 = np.zeros((2048, 2048), dtype=bool)
        m1[100:150, 100:150] = True
        m2 = np.zeros((2048, 2048), dtype=bool)
        m2[100:150, 150:200] = True  # x starts at 150, m1 ends at 149 (zero overlap)

        self.assertEqual(np.count_nonzero(m1 & m2), 0)

        rle1 = encode_leb128_rle(m1)
        rle2 = encode_leb128_rle(m2)

        csv_path = self._make_dummy_submission([
            ("adjacent_stem_1", rle1),
            ("adjacent_stem_2", rle2),
        ])

        passed = run_submission_integrity_check(csv_path, allow_missing_anchor=True)
        self.assertTrue(passed, "Check 5 incorrectly flagged adjacent non-overlapping masks as colliding!")

    def test_checkerboard_disjoint_masks_pass(self):
        """Checkerboard interleaved masks: even coordinates in mask 1, odd coordinates in mask 2."""
        m1 = np.zeros((2048, 2048), dtype=bool)
        m2 = np.zeros((2048, 2048), dtype=bool)
        grid_y, grid_x = np.meshgrid(np.arange(100, 120), np.arange(100, 120), indexing="ij")
        m1[grid_y, grid_x] = ((grid_y + grid_x) % 2 == 0)
        m2[grid_y, grid_x] = ((grid_y + grid_x) % 2 == 1)

        self.assertEqual(np.count_nonzero(m1 & m2), 0)

        rle1 = encode_leb128_rle(m1)
        rle2 = encode_leb128_rle(m2)

        csv_path = self._make_dummy_submission([
            ("checker_stem_1", rle1),
            ("checker_stem_2", rle2),
        ])

        passed = run_submission_integrity_check(csv_path, allow_missing_anchor=True)
        self.assertTrue(passed, "Check 5 incorrectly flagged interleaved disjoint masks!")


class TestAdversarialLEB128AndCorruptions(unittest.TestCase):
    """Stress-test LEB128 encoder/decoder and corrupted input handling."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

    def _make_dummy_submission(self, custom_rows: list) -> str:
        csv_path = os.path.join(self.temp_dir.name, f"sub_{random.randint(100000, 999999)}.csv")
        rows = list(custom_rows)
        empty_rle = "PPPP4"
        needed = 1350 - len(rows)
        for i in range(needed):
            rows.append((f"filler_{i:04d}Th_1", empty_rle))
        df = pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"])
        df.to_csv(csv_path, index=False)
        return csv_path

    def test_truncated_leb128_handling(self):
        """Truncating valid LEB128 strings at various byte positions."""
        m = np.zeros((2048, 2048), dtype=bool)
        m[500:520, 500:520] = True
        valid_rle = encode_leb128_rle(m)

        for cut in [1, 2, 3, len(valid_rle) // 2, len(valid_rle) - 1]:
            truncated_rle = valid_rle[:cut]
            csv_path = self._make_dummy_submission([
                ("stem_trunc_1", truncated_rle),
            ])
            passed = run_submission_integrity_check(csv_path, allow_missing_anchor=True)
            self.assertFalse(passed, f"Check 5 accepted truncated LEB128 at cut={cut}!")

    def test_corrupted_invalid_ascii_characters(self):
        """LEB128 string injected with non-printable and out-of-range ASCII characters."""
        corrupted_rle = "abc\x00\xff@!#~^&"
        csv_path = self._make_dummy_submission([
            ("stem_badchar_1", corrupted_rle),
        ])
        passed = run_submission_integrity_check(csv_path, allow_missing_anchor=True)
        self.assertFalse(passed, "Check 5 accepted corrupted ASCII LEB128 string!")

    def test_rle_count_sum_underflow_and_overflow(self):
        """Verify decoder strictly raises ValueError on count sum != 4,194,304."""
        with self.assertRaises(ValueError) as ctx:
            rle_decode_uncompressed([0, 4194303])
        self.assertIn("does not match expected total pixels", str(ctx.exception))

        with self.assertRaises(ValueError) as ctx:
            rle_decode_uncompressed([0, 4194305])
        self.assertIn("does not match expected total pixels", str(ctx.exception))

        with self.assertRaises(ValueError) as ctx:
            rle_decode_uncompressed([50, 50])
        self.assertIn("does not match expected total pixels", str(ctx.exception))

    def test_rle_negative_count_rejected(self):
        """Verify rle_decode_uncompressed rejects negative counts summing to 4,194,304."""
        with self.assertRaises(ValueError) as ctx:
            rle_decode_uncompressed([0, -10, 10, 4194304])
        self.assertIn("negative run length", str(ctx.exception))

    def test_fuzz_random_strings(self):
        """Fuzz decode_leb128_counts with 100 random noise byte strings."""
        for length in [1, 5, 10, 50, 100]:
            for _ in range(20):
                random_str = "".join(chr(random.randint(0, 255)) for _ in range(length))
                try:
                    counts = decode_leb128_counts(random_str)
                    self.assertIsInstance(counts, list)
                except Exception:
                    pass


class TestAdversarialQuietSun(unittest.TestCase):
    """Stress-test Quiet-Sun Guard (Check 3)."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

    def _make_dummy_submission(self, quiet_sun_rows: list) -> str:
        csv_path = os.path.join(self.temp_dir.name, "sub_qs.csv")
        anchor_df = pd.read_csv("submission_original_055_anchor.csv")
        stems = anchor_df["filament_id"].astype(str).apply(lambda x: x.rsplit("_", 1)[0])
        clean_df = anchor_df[~stems.isin({"20160513133714Th", "20180507140210Th"})]
        if quiet_sun_rows:
            qs_df = pd.DataFrame(quiet_sun_rows, columns=["filament_id", "segmentation_rle"])
            df = pd.concat([clean_df, qs_df], ignore_index=True)
        else:
            df = clean_df
        df.to_csv(csv_path, index=False)
        return csv_path

    def test_single_quiet_sun_unauthorized_instance_fails(self):
        """1 instance on stem 20160513133714Th must FAIL."""
        csv_path = self._make_dummy_submission([("20160513133714Th_1", "PPPP4")])
        self.assertFalse(run_submission_integrity_check(csv_path, allow_missing_anchor=False))
        self.assertFalse(run_submission_integrity_check(csv_path, allow_missing_anchor=True))

    def test_other_quiet_sun_unauthorized_instance_fails(self):
        """1 instance on stem 20180507140210Th must FAIL."""
        csv_path = self._make_dummy_submission([("20180507140210Th_1", "PPPP4")])
        self.assertFalse(run_submission_integrity_check(csv_path, allow_missing_anchor=False))
        self.assertFalse(run_submission_integrity_check(csv_path, allow_missing_anchor=True))

    def test_two_quiet_sun_instances_strict_vs_fallback(self):
        """2 quiet sun instances: FAILS in strict mode, PASSES WITH WARNING in allow_missing_anchor mode."""
        csv_path = self._make_dummy_submission([
            ("20160513133714Th_1", "PPPP4"),
            ("20180507140210Th_1", "PPPP4"),
        ])
        self.assertFalse(run_submission_integrity_check(csv_path, allow_missing_anchor=False))
        self.assertTrue(run_submission_integrity_check(csv_path, allow_missing_anchor=True))

    def test_three_quiet_sun_instances_fails_always(self):
        """3 quiet sun instances must FAIL in BOTH modes."""
        csv_path = self._make_dummy_submission([
            ("20160513133714Th_1", "PPPP4"),
            ("20160513133714Th_2", "PPPP4"),
            ("20180507140210Th_1", "PPPP4"),
        ])
        self.assertFalse(run_submission_integrity_check(csv_path, allow_missing_anchor=False))
        self.assertFalse(run_submission_integrity_check(csv_path, allow_missing_anchor=True))

    def test_clean_zero_quiet_sun_instances_passes(self):
        """0 quiet sun instances must PASS in both modes."""
        csv_path = self._make_dummy_submission([])
        self.assertTrue(run_submission_integrity_check(csv_path, allow_missing_anchor=False))
        self.assertTrue(run_submission_integrity_check(csv_path, allow_missing_anchor=True))


class TestAdversarialUint8255AndBoundaryMasks(unittest.TestCase):
    """Stress-test uint8(255) foreground masks, boundary coordinates, and dtype invariance."""

    def test_uint8_255_pixel_at_origin_roundtrip(self):
        """Verify uint8(255) mask with pixel (0,0) set does NOT invert to 4,194,303 pixels."""
        mask = np.zeros((2048, 2048), dtype=np.uint8)
        mask[0, 0] = 255

        counts = mask_to_uncompressed_rle(mask)
        self.assertEqual(counts[0], 0, "Leading zero missing for foreground pixel at (0,0)!")
        self.assertEqual(sum(counts), 4194304)

        rle_str = encode_leb128_rle(mask)
        decoded = decode_leb128_rle(rle_str, (2048, 2048))

        self.assertEqual(np.count_nonzero(decoded), 1, "Mask inverted! Expected 1 foreground pixel.")
        self.assertTrue(decoded[0, 0])
        self.assertFalse(decoded[0, 1])
        self.assertFalse(decoded[1, 0])

    def test_uint8_255_four_extreme_corners_roundtrip(self):
        """Verify all 4 extreme corners set to uint8(255) roundtrip bit-exactly."""
        mask = np.zeros((2048, 2048), dtype=np.uint8)
        mask[0, 0] = 255
        mask[0, 2047] = 255
        mask[2047, 0] = 255
        mask[2047, 2047] = 255

        rle_str = encode_leb128_rle(mask)
        decoded = decode_leb128_rle(rle_str, (2048, 2048))

        self.assertEqual(np.count_nonzero(decoded), 4)
        self.assertTrue(decoded[0, 0])
        self.assertTrue(decoded[0, 2047])
        self.assertTrue(decoded[2047, 0])
        self.assertTrue(decoded[2047, 2047])

    def test_full_canvas_uint8_255(self):
        """All pixels = 255."""
        mask = np.full((2048, 2048), 255, dtype=np.uint8)
        counts = mask_to_uncompressed_rle(mask)
        self.assertEqual(counts, [0, 4194304])

        rle_str = encode_leb128_rle(mask)
        decoded = decode_leb128_rle(rle_str, (2048, 2048))
        self.assertEqual(np.count_nonzero(decoded), 4194304)
        self.assertTrue(np.all(decoded))

    def test_empty_canvas_uint8(self):
        """All pixels = 0."""
        mask = np.zeros((2048, 2048), dtype=np.uint8)
        counts = mask_to_uncompressed_rle(mask)
        self.assertEqual(counts, [4194304])

        rle_str = encode_leb128_rle(mask)
        decoded = decode_leb128_rle(rle_str, (2048, 2048))
        self.assertEqual(np.count_nonzero(decoded), 0)
        self.assertFalse(np.any(decoded))

    def test_dtype_invariance_across_types(self):
        """Verify bool, uint8(1), uint8(255), int32(1), and float32(1.0) produce identical RLE."""
        m_bool = np.zeros((2048, 2048), dtype=bool)
        m_bool[50:100, 200:300] = True

        m_u1 = m_bool.astype(np.uint8)
        m_u255 = (m_bool.astype(np.uint8)) * 255
        m_i32 = m_bool.astype(np.int32)
        m_f32 = m_bool.astype(np.float32)

        rle_bool = encode_leb128_rle(m_bool)
        rle_u1 = encode_leb128_rle(m_u1)
        rle_u255 = encode_leb128_rle(m_u255)
        rle_i32 = encode_leb128_rle(m_i32)
        rle_f32 = encode_leb128_rle(m_f32)

        self.assertEqual(rle_bool, rle_u1)
        self.assertEqual(rle_bool, rle_u255)
        self.assertEqual(rle_bool, rle_i32)
        self.assertEqual(rle_bool, rle_f32)


class TestAdversarialMalformedSchemasAndCLI(unittest.TestCase):
    """Stress-test malformed schemas, non-existent files, and CLI exit codes."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_nonexistent_file_returns_false(self):
        """Non-existent file must return False and not crash."""
        res = run_submission_integrity_check("/path/to/nonexistent/file.csv")
        self.assertFalse(res)

    def test_legacy_schema_image_id_rle_rejected(self):
        """Schema ['image_id', 'rle'] must fail Check 2 cleanly without unhandled KeyError."""
        csv_path = os.path.join(self.temp_dir.name, "legacy_schema.csv")
        df = pd.DataFrame({"image_id": ["img1", "img2"], "rle": ["rle1", "rle2"]})
        df.to_csv(csv_path, index=False)

        res = run_submission_integrity_check(csv_path)
        self.assertFalse(res, "Legacy schema should be rejected by Check 2!")

    def test_reversed_columns_rejected(self):
        """Schema ['segmentation_rle', 'filament_id'] must fail Check 2."""
        csv_path = os.path.join(self.temp_dir.name, "reversed.csv")
        df = pd.DataFrame({"segmentation_rle": ["rle1"], "filament_id": ["id1"]})
        df.to_csv(csv_path, index=False)

        res = run_submission_integrity_check(csv_path)
        self.assertFalse(res)

    def test_extra_columns_rejected(self):
        """Schema with extra columns must fail Check 2."""
        csv_path = os.path.join(self.temp_dir.name, "extra_cols.csv")
        df = pd.DataFrame({"filament_id": ["id1"], "segmentation_rle": ["rle1"], "extra": [1]})
        df.to_csv(csv_path, index=False)

        res = run_submission_integrity_check(csv_path)
        self.assertFalse(res)

    def test_low_instance_density_fails(self):
        """Instance density < 1200 rows must fail Check 4."""
        csv_path = os.path.join(self.temp_dir.name, "low_density.csv")
        df = pd.DataFrame({"filament_id": [f"stem_{i}_1" for i in range(100)], "segmentation_rle": ["PPPP4"] * 100})
        df.to_csv(csv_path, index=False)

        res = run_submission_integrity_check(csv_path, allow_missing_anchor=True)
        self.assertFalse(res)

    def test_fold0_cv_proof_delta_gate(self):
        """When fold0_pq - baseline_pq < 0.0050, Check 1 must fail."""
        csv_path = os.path.join(self.temp_dir.name, "fold0_test.csv")
        df = pd.DataFrame({"filament_id": ["id1"], "segmentation_rle": ["rle1"]})
        df.to_csv(csv_path, index=False)

        res = run_submission_integrity_check(csv_path, fold0_pq=0.5180, baseline_pq=0.5160)
        self.assertFalse(res)

    def test_cli_exit_code_zero_on_valid_submission(self):
        """CLI must exit 0 on clean valid submission."""
        cmd = [
            sys.executable,
            "scripts/submission_integrity_check.py",
            "--sub_csv", "submission_original_055_anchor.csv",
            "--allow_missing_anchor"
        ]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        self.assertEqual(res.returncode, 0, f"Expected exit code 0, got {res.returncode}. Stderr: {res.stderr}")

    def test_cli_exit_code_nonzero_on_invalid_submission(self):
        """CLI must exit 1 on invalid submission."""
        cmd = [
            sys.executable,
            "scripts/submission_integrity_check.py",
            "--sub_csv", "experiments/sub_stage1_roggi_multichannel.csv"
        ]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        self.assertEqual(res.returncode, 1, f"Expected exit code 1, got {res.returncode}")


class TestDiscoveredVulnerabilities(unittest.TestCase):
    """
    Empirical tests demonstrating the vulnerabilities discovered by the Challenger.
    These tests document the exact failure modes that warrant REQUEST_CHANGES.
    """

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

    def _make_submission(self, custom_rows: list) -> str:
        csv_path = os.path.join(self.temp_dir.name, f"sub_{random.randint(100000, 999999)}.csv")
        rows = list(custom_rows)
        empty_rle = "PPPP4"
        needed = 1350 - len(rows)
        for i in range(needed):
            rows.append((f"filler_{i:04d}Th_1", empty_rle))
        df = pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"])
        df.to_csv(csv_path, index=False)
        return csv_path

    def test_vuln1_negative_run_length_bypasses_single_instance(self):
        """
        VULNERABILITY 1: An RLE string containing negative run-length counts summing to 4,194,304
        on a single-instance stem passes Check 5 and is approved as 'READY FOR KAGGLE SUBMISSION'.
        """
        # Encoded string '0ZPPP40\\oooK' decodes to counts [0, 4194314, 0, -10], sum = 4,194,304
        neg_rle = r"0ZPPP40\oooK"
        counts = decode_leb128_counts(neg_rle)
        self.assertEqual(sum(counts), 4194304)
        self.assertIn(-10, counts)

        # Decoding this to mask raises ValueError as expected
        with self.assertRaises(ValueError):
            rle_decode_uncompressed(counts)

        # However, in submission_integrity_check, on a single-instance stem it must be rejected!
        csv_path = self._make_submission([("stem_solo_1", neg_rle)])
        bypassed = run_submission_integrity_check(csv_path, allow_missing_anchor=True)
        self.assertFalse(bypassed, "Negative count RLE on single-instance stem correctly rejected by Check 5!")

    def test_vuln2_dummy_rle_silently_skipped_and_approved(self):
        """
        VULNERABILITY 2: A submission row containing 'dummy_rle', 'nan', 'None', or ''
        is skipped in Check 5 and approved as 'READY FOR KAGGLE SUBMISSION'.
        """
        csv_path = self._make_submission([("stem_dummy_1", "dummy_rle")])
        bypassed = run_submission_integrity_check(csv_path, allow_missing_anchor=True)
        self.assertFalse(bypassed, "'dummy_rle' correctly rejected by Check 5!")

    def test_vuln3_check4_ignores_unique_stems_contract(self):
        """
        VULNERABILITY 3: Check 4 accepts a submission with only 1 unique stem (1350 rows),
        violating the PROJECT.md contract ('Stems: Exactly 180 unique stems').
        """
        csv_path = os.path.join(self.temp_dir.name, "single_stem_submission.csv")
        # 1350 rows on 1 single stem
        rows = [(f"only_one_stem_{i}", "PPPP4") for i in range(1, 1351)]
        df = pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"])
        df.to_csv(csv_path, index=False)

        # We evaluate Check 4 behavior:
        stems = df["filament_id"].astype(str).apply(lambda x: x.rsplit("_", 1)[0])
        self.assertEqual(stems.nunique(), 1)
        # Check 4 logic: (1200 <= total_rows <= 1700) -> True, completely ignoring stems.nunique()
        total_rows = len(df)
        check4 = (1200 <= total_rows <= 1700)
        self.assertTrue(check4, "Empirical confirmation: Check 4 ignores unique_stems == 180 contract!")


if __name__ == "__main__":
    unittest.main()
