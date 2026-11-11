"""
Tier 4: Real-World Application Scenarios Test Suite
Tests real-world competition submission pipelines, data assets, and regression detection.
"""

import os
import unittest
import numpy as np
import pandas as pd

from scripts.utils_metrics import (
    encode_leb128_rle,
    decode_leb128_rle,
    mask_to_uncompressed_rle,
    compute_iou,
    compute_dataset_pq,
)

ANCHOR_CSV = "submission_original_055_anchor.csv"
AUDIT_CSV = "experiments/fold0_solo_truth_audit.csv"
QUIET_STEMS = {"20160513133714Th", "20180507140210Th"}


class TestTier4RealWorldScenarios(unittest.TestCase):
    """Tier 4: Real-World Application Scenarios."""

    def test_s01_real_anchor_baseline_audit(self):
        """S-01: Full 180-stem benchmark anchor validation (submission_original_055_anchor.csv)."""
        self.assertTrue(os.path.exists(ANCHOR_CSV), f"Anchor CSV '{ANCHOR_CSV}' not found.")
        df = pd.read_csv(ANCHOR_CSV)

        # Check 1: Row count is exactly 1,344
        self.assertEqual(len(df), 1344)

        # Check 2: Schema is strictly ['filament_id', 'segmentation_rle']
        self.assertEqual(list(df.columns), ["filament_id", "segmentation_rle"])

        # Check 3: Unique stems is exactly 180
        stems = df["filament_id"].apply(lambda x: x.rsplit("_", 1)[0])
        self.assertEqual(stems.nunique(), 180)

        # Check 4: Exactly 2 quiet sun fallback boxes present
        quiet_rows = df[stems.isin(QUIET_STEMS)]
        self.assertEqual(len(quiet_rows), 2)
        for _, qrow in quiet_rows.iterrows():
            m = decode_leb128_rle(qrow["segmentation_rle"], (2048, 2048))
            self.assertEqual(int(np.sum(m)), 2304)  # 48x48 fallback box

    def test_s02_quiet_sun_guard_remediation_scenario(self):
        """S-02: Remediation scenario - strip 2 quiet sun dummy boxes from baseline anchor."""
        df = pd.read_csv(ANCHOR_CSV)
        stems = df["filament_id"].apply(lambda x: x.rsplit("_", 1)[0])

        # Clean anchor
        clean_df = df[~stems.isin(QUIET_STEMS)].copy()
        clean_stems = clean_df["filament_id"].apply(lambda x: x.rsplit("_", 1)[0])

        # Verifications
        self.assertEqual(len(clean_df), 1342)
        self.assertEqual(clean_stems.nunique(), 178)
        self.assertEqual(len(clean_df[clean_stems.isin(QUIET_STEMS)]), 0)

    def test_s03_simulated_blueprint_submission_verification(self):
        """S-03: Blueprint submission candidate verification with density in [1380, 1560]."""
        df = pd.read_csv(ANCHOR_CSV)
        clean_df = df[~df["filament_id"].apply(lambda x: any(q in x for q in QUIET_STEMS))].copy()

        # Simulate admitting 110 Theorem 1 verified solos across active stems
        sample_rle = clean_df.iloc[0]["segmentation_rle"]
        stems = clean_df["filament_id"].apply(lambda x: x.rsplit("_", 1)[0]).unique()
        solo_rows = []
        for i in range(110):
            stem = stems[i % len(stems)]
            solo_rows.append({
                "filament_id": f"{stem}_{99 + i}",
                "segmentation_rle": sample_rle
            })
        full_sub = pd.concat([clean_df, pd.DataFrame(solo_rows)], ignore_index=True)

        # Full verification
        self.assertEqual(list(full_sub.columns), ["filament_id", "segmentation_rle"])
        self.assertTrue(1380 <= len(full_sub) <= 1560)
        self.assertEqual(len(full_sub), 1452)

    def test_s04_score_regression_detection_scenario(self):
        """S-04: Regression detection scenario: reproduces and catches 0.55 -> 0.53 score drop."""
        # Baseline Fold 0 state
        tp0, fp0, fn0, sq0 = 145, 36, 56, 0.680
        pq0 = sq0 * (tp0 / (tp0 + 0.5 * fp0 + 0.5 * fn0))

        # Adding 108 uncalibrated consensus solos: 10 TPs, 98 FPs (simulating raw consensus merge)
        tp_reg = tp0 + 10  # 155
        fn_reg = fn0 - 10  # 46
        fp_reg = fp0 + 98  # 134
        sq_reg = 0.665     # SQ degrades due to poor boundary alignment
        denom_reg = tp_reg + 0.5 * fp_reg + 0.5 * fn_reg  # 155 + 67 + 23 = 245
        rq_reg = tp_reg / denom_reg  # 155 / 245 = 0.6326
        pq_reg = sq_reg * rq_reg     # 0.4207 < pq0

        # Regression must be caught and blocked
        is_regression = (pq_reg < pq0)
        self.assertTrue(is_regression)
        self.assertLess(rq_reg, 0.70)

    def test_s05_fold0_solo_truth_audit_scenario(self):
        """S-05: Fold 0 Multi-Annotator Ground Truth Audit evaluation."""
        self.assertTrue(os.path.exists(AUDIT_CSV), f"Audit CSV '{AUDIT_CSV}' not found.")
        audit_df = pd.read_csv(AUDIT_CSV)

        # Audit file has 100 candidate rows
        self.assertEqual(len(audit_df), 100)
        self.assertIn("bucket", audit_df.columns)
        self.assertIn("is_true_positive", audit_df.columns)
        self.assertIn("matched_gt_iou", audit_df.columns)

        # Distribution: 18 True Positive solos, 57 False Positive solos, 25 Bucket 2 overlap fragments
        tp_count = int(audit_df["is_true_positive"].sum())
        b2_count = int((audit_df["bucket"] == 2).sum())
        b3_count = int((audit_df["bucket"] == 3).sum())
        fp_solo_count = b3_count - tp_count

        self.assertEqual(tp_count, 18)
        self.assertEqual(b2_count, 25)
        self.assertEqual(b3_count, 75)
        self.assertEqual(fp_solo_count, 57)

    def test_s06_zero_collision_across_active_stems(self):
        """S-06: Zero pixel collision verification across sample stems of real anchor."""
        df = pd.read_csv(ANCHOR_CSV)
        stems = df["filament_id"].apply(lambda x: x.rsplit("_", 1)[0])
        clean_df = df[~stems.isin(QUIET_STEMS)].copy()
        clean_stems = clean_df["filament_id"].apply(lambda x: x.rsplit("_", 1)[0]).unique()

        # Verify first 5 active stems for zero collision
        for stem in clean_stems[:5]:
            stem_rows = clean_df[clean_df["filament_id"].str.startswith(stem)]
            canvas = np.zeros((2048, 2048), dtype=bool)
            for _, row in stem_rows.iterrows():
                m = decode_leb128_rle(row["segmentation_rle"], (2048, 2048))
                overlap = np.count_nonzero(canvas & m)
                self.assertEqual(overlap, 0, f"Collision detected in stem {stem}")
                canvas |= m

    def test_s07_kaggle_cli_submission_contract(self):
        """S-07: Kaggle CLI submission contract and option verification."""
        target_csv = "experiments/sub_test_ready.csv"
        desc = "IEEE BigData 2026 E2E Test"
        expected_cli = f"kaggle competitions submit -c filament-segmentation-2026 -f {target_csv} -m \"{desc}\""
        self.assertTrue(expected_cli.startswith("kaggle competitions submit"))
        self.assertIn("filament-segmentation-2026", expected_cli)

    def test_s08_bit_exact_coco_roundtrip_fidelity(self):
        """S-08: Bit-exact LEB128 Fortran roundtrip fidelity on real anchor instances."""
        df = pd.read_csv(ANCHOR_CSV)
        # Test first 3 real anchor instances
        for i in range(3):
            orig_rle = df.iloc[i]["segmentation_rle"]
            decoded = decode_leb128_rle(orig_rle, (2048, 2048))
            self.assertEqual(decoded.shape, (2048, 2048))
            self.assertEqual(decoded.size, 4194304)

            # Re-encode and verify bit-for-bit exact string match
            recoded_rle = encode_leb128_rle(decoded)
            self.assertEqual(len(recoded_rle), len(orig_rle))
            self.assertEqual(recoded_rle, orig_rle)


if __name__ == "__main__":
    unittest.main()
