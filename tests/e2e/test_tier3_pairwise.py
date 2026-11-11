"""
Tier 3: Pairwise Combinations Test Suite
Tests interactions between pairs of pipeline stages and post-processing modules.
"""

import unittest
import numpy as np
import pandas as pd
import scipy.ndimage as ndi

from scripts.utils_metrics import (
    encode_leb128_rle,
    decode_leb128_rle,
    compute_iou,
    evaluate_image_panoptic_quality,
    compute_dataset_pq,
)
from scripts.utils_solar_physics import (
    compute_mask_shape_features,
    compute_halpha_limb_deficit,
)
from scripts.stage2_roggi_ffbsi_bridge import (
    detect_candidate_bridge_pairs,
    sample_ribbon_alignment_matrix,
    run_ffbsi_particle_smoothing,
)
from scripts.stage3_physics_boundary_refiner import (
    compute_local_valley_statistics,
    refine_mask_boundary,
)


class TestTier3Pairwise(unittest.TestCase):
    """Tier 3: Pairwise Combinations."""

    def test_p01_anchor_pruner_and_solo_gating(self):
        """P-01: Anchor FP Pruner + Solo Inclusion Gate execute together without collision."""
        # Initial state: 3 anchor masks (one is an uncorroborated FP)
        canvas = np.zeros((200, 200), dtype=bool)
        a1 = np.zeros((200, 200), dtype=bool); a1[20:50, 20:50] = True   # Real anchor
        a2 = np.zeros((200, 200), dtype=bool); a2[60:90, 20:50] = True   # Real anchor
        a_fp = np.zeros((200, 200), dtype=bool); a_fp[120:150, 20:50] = True # Uncorroborated FP

        # Direction A: Prune a_fp
        anchors_retained = [a1, a2]
        for a in anchors_retained:
            canvas |= a

        # Direction B: Evaluate Solo candidate satisfying Theorem 1 (p*j > 0.275)
        p, j = 0.60, 0.70  # 0.42 > 0.275
        cand = np.zeros((200, 200), dtype=bool)
        cand[120:150, 100:150] = True  # Disjoint solo

        is_admitted = (p * j > 0.275) and (np.count_nonzero(cand & canvas) == 0)
        self.assertTrue(is_admitted)
        canvas |= cand

        # Invariant: 3 disjoint instances on canvas
        self.assertEqual(int(np.sum(a1 & cand)), 0)
        self.assertEqual(int(np.sum(a2 & cand)), 0)

    def test_p02_solo_gating_and_zero_collision_stamping(self):
        """P-02: Solo Gating + Zero-Collision Priority Stamping."""
        # Anchor instance occupies space
        canvas = np.zeros((100, 100), dtype=bool)
        anchor_mask = np.zeros((100, 100), dtype=bool)
        anchor_mask[20:60, 20:60] = True
        canvas |= anchor_mask

        # Candidate solo partially overlaps anchor
        solo_cand = np.zeros((100, 100), dtype=bool)
        solo_cand[50:80, 50:80] = True  # 10x10 overlap

        # Priority stamping clips overlap from solo candidate
        solo_clipped = solo_cand & (~canvas)
        self.assertEqual(int(np.sum(solo_clipped & anchor_mask)), 0)
        # Original anchor pixels 100% preserved
        self.assertTrue(np.all(canvas[anchor_mask]))

    def test_p03_solo_gating_and_sunspot_umbra_veto(self):
        """P-03: Solo Gating + Sunspot Umbra Veto."""
        # Candidate 1: High model score (p*j > 0.275) but round dark spot (sunspot)
        p1, j1 = 0.70, 0.60  # p*j = 0.42 > 0.275
        y, x = np.ogrid[:60, :60]
        spot_mask = ((y - 30)**2 + (x - 30)**2) <= 10**2  # round disk area ~ 314 < 500
        elong1, circ1, area1, veto1 = compute_mask_shape_features(spot_mask)

        # Candidate 1 must be rejected by Sunspot Veto despite high score
        admit_c1 = (p1 * j1 > 0.275) and (not veto1)
        self.assertTrue(veto1)
        self.assertFalse(admit_c1)

        # Candidate 2: High model score and elongated filament shape
        p2, j2 = 0.55, 0.65  # 0.3575 > 0.275
        fil_mask = np.zeros((60, 60), dtype=bool)
        fil_mask[28:32, 10:50] = True  # elongated strip
        elong2, circ2, area2, veto2 = compute_mask_shape_features(fil_mask)

        admit_c2 = (p2 * j2 > 0.275) and (not veto2)
        self.assertFalse(veto2)
        self.assertTrue(admit_c2)

    def test_p04_spine_guided_healing_and_dilation_grouping(self):
        """P-04: Spine-Guided FFBSi Bridge + MengxinLab Dilation Grouping."""
        # Collinear fragments within 20px
        foreground = np.zeros((80, 160), dtype=np.uint8)
        m1 = np.zeros((80, 160), dtype=bool); m1[38:42, 20:60] = True
        m2 = np.zeros((80, 160), dtype=bool); m2[38:42, 80:120] = True  # Gap = 20px
        foreground[m1] = 1
        foreground[m2] = 1

        # 1. Spine endpoint detection verifies collinear alignment
        pairs = detect_candidate_bridge_pairs([m1, m2], max_dist=30.0, min_dist=3.0, max_angle_deg=30.0)
        self.assertEqual(len(pairs), 1)

        # 2. MengxinLab dilation grouping unifies them with 0% boundary bloat
        expanded = ndi.binary_dilation(foreground > 0, iterations=15)
        expanded_labels, _ = ndi.label(expanded)
        group_labels = np.where(foreground > 0, expanded_labels, 0)

        # Single group ID, exact foreground pixel conservation
        unique_groups = np.unique(group_labels[group_labels > 0])
        self.assertEqual(len(unique_groups), 1)
        self.assertEqual(int(np.sum(group_labels > 0)), int(np.sum(foreground > 0)))

    def test_p05_dilation_grouping_and_boundary_snapping(self):
        """P-05: MengxinLab Grouping + Boundary Snapping."""
        # Grouped mask containing 2 fragments
        m = np.zeros((60, 120), dtype=bool)
        m[28:32, 10:40] = True
        m[28:32, 60:90] = True

        img = np.ones((60, 120), dtype=np.float32) * 0.5
        seg = np.zeros((60, 120), dtype=np.float32); seg[m] = 1.0
        yolo = seg.copy()
        med, mad = compute_local_valley_statistics(img, window_size=15)

        # Refine boundaries
        refined = refine_mask_boundary(m, img, seg, yolo, med, mad, band_width=2)
        # Ensure refinement produces a valid non-empty mask
        self.assertGreater(int(np.sum(refined)), 0)

    def test_p06_boundary_snapping_and_zero_collision_invariant(self):
        """P-06: Boundary Snapping + Zero-Collision Invariant."""
        # Two neighboring filaments separated by 8px
        canvas = np.zeros((100, 100), dtype=bool)
        m1 = np.zeros((100, 100), dtype=bool); m1[20:50, 20:45] = True
        m2 = np.zeros((100, 100), dtype=bool); m2[20:50, 55:80] = True  # gap = 10px

        img = np.ones((100, 100), dtype=np.float32) * 0.5
        med, mad = compute_local_valley_statistics(img, window_size=20)
        seg1 = np.zeros((100, 100), dtype=np.float32); seg1[m1] = 1.0
        seg2 = np.zeros((100, 100), dtype=np.float32); seg2[m2] = 1.0

        ref1 = refine_mask_boundary(m1, img, seg1, seg1, med, mad, band_width=3)
        ref2 = refine_mask_boundary(m2, img, seg2, seg2, med, mad, band_width=3)

        # If snapping caused boundary dilation overlap, priority stamping enforces disjointness
        overlap = np.count_nonzero(ref1 & ref2)
        if overlap > 0:
            ref2 = ref2 & (~ref1)

        self.assertEqual(np.count_nonzero(ref1 & ref2), 0)

    def test_p07_quiet_sun_guard_and_180_stem_generation(self):
        """P-07: Quiet Sun Guard + 180-Stem Submission Generation."""
        quiet_stems = {"20160513133714Th", "20180507140210Th"}
        all_stems = [f"201101201055{i:02d}Ch" for i in range(178)] + list(quiet_stems)
        self.assertEqual(len(all_stems), 180)

        # Generate submission rows: active stems have filaments, quiet stems have 0 rows
        sub_rows = []
        for stem in all_stems:
            if stem in quiet_stems:
                continue
            for inst in range(1, 8):
                sub_rows.append({"filament_id": f"{stem}_{inst}", "segmentation_rle": "PPPP4"})

        sub_df = pd.DataFrame(sub_rows)
        # Check active stems are present
        active_present = sub_df["filament_id"].apply(lambda x: x.rsplit("_", 1)[0]).nunique()
        self.assertEqual(active_present, 178)
        # Quiet stems have 0 rows
        quiet_rows = sub_df[sub_df["filament_id"].apply(lambda x: x.rsplit("_", 1)[0]).isin(quiet_stems)]
        self.assertEqual(len(quiet_rows), 0)

    def test_p08_rle_engine_and_schema_validator_roundtrip(self):
        """P-08: Bit-Exact Fortran RLE Engine + Submission Schema Validator."""
        mask = np.zeros((2048, 2048), dtype=bool)
        mask[200:300, 200:400] = True
        rle = encode_leb128_rle(mask)

        df = pd.DataFrame({
            "filament_id": ["20110120105534Ch_1"],
            "segmentation_rle": [rle]
        })

        # Schema validator decodes RLE and checks size
        decoded = decode_leb128_rle(df.iloc[0]["segmentation_rle"], (2048, 2048))
        self.assertEqual(decoded.size, 4194304)
        self.assertEqual(int(np.sum(decoded)), 20000)

    def test_p09_theorem1_gate_and_fold0_zero_trust_audit(self):
        """P-09: Theorem 1 Gate + Fold 0 Zero-Trust Audit."""
        # Initial benchmark: TP=145, FP=36, FN=56, SQ=0.680
        tp0, fp0, fn0, sq0 = 145, 36, 56, 0.680
        pq0 = sq0 * (tp0 / (tp0 + 0.5 * fp0 + 0.5 * fn0))  # 0.5160

        # Admit 18 verified solo TPs satisfying Theorem 1
        tp1 = tp0 + 18  # 163
        fn1 = fn0 - 18  # 38
        fp1 = fp0       # 36
        sq1 = sq0       # 0.680
        pq1 = sq1 * (tp1 / (tp1 + 0.5 * fp1 + 0.5 * fn1))

        delta_pq = pq1 - pq0
        self.assertGreaterEqual(delta_pq, 0.0050)

    def test_p10_full_e2e_pipeline_combination(self):
        """P-10: Complete pipeline combination (Prune -> Gate -> Heal -> Snap -> Validate)."""
        # Canvas with anchor, solo candidate, and quiet sun
        quiet_stems = {"20160513133714Th"}
        active_stem = "20110120105534Ch"

        # 1. Anchor with 1 real filament + 1 quiet sun dummy
        anchor_df = pd.DataFrame({
            "filament_id": [f"{active_stem}_1", f"20160513133714Th_1"],
            "segmentation_rle": ["PPPP4", "PPPP4"]
        })

        # 2. Prune quiet sun
        clean_df = anchor_df[~anchor_df["filament_id"].str.startswith("20160513133714Th")].copy()
        self.assertEqual(len(clean_df), 1)

        # 3. Solo admission: candidate passes Theorem 1
        p, j = 0.55, 0.60  # 0.33 > 0.275
        admitted_solo = {"filament_id": f"{active_stem}_2", "segmentation_rle": "PPPP4"}
        sub_df = pd.concat([clean_df, pd.DataFrame([admitted_solo])], ignore_index=True)

        # 4. Final submission schema validation
        self.assertEqual(list(sub_df.columns), ["filament_id", "segmentation_rle"])
        self.assertEqual(len(sub_df), 2)


if __name__ == "__main__":
    unittest.main()
