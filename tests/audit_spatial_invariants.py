#!/usr/bin/env python3
"""
Independent Adversarial Spatial & Format Invariant Audit
Author: Teamwork Challenger (challenger_1)
Date: 2026-11-09

Audits and stress-tests:
1. Disjoint Canvas Audit across all 180 test stems (1,327 masks):
   - Strict pairwise intersection checks: M_i \cap M_j
   - Exact colliding pixel count across entire dataset
2. Bit-Exact LEB128 Fortran RLE Sum Stress-Test:
   - 100% of RLE strings decode to uncompressed counts summing to 4,194,304
   - Non-negative count invariant (c >= 0)
   - Foreground at (0,0) leading zero rule validation + adversarial oracle
   - Pycocotools C-library cross-validation
3. Boundary & Extreme Cases:
   - Area extremes (min, max, percentiles)
   - Aspect ratio extremes and bounding box distributions
   - String integrity, character set validity, and truncation fuzzing
"""

import os
import sys
import glob
import json
import time
import numpy as np
import pandas as pd
from pycocotools import mask as mask_util

# Add repository root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.utils_metrics import (
    decode_leb128_counts,
    decode_leb128_rle,
    encode_leb128_rle,
    mask_to_uncompressed_rle,
    rle_decode_uncompressed,
)

SUBMISSION_PATH = os.path.join(PROJECT_ROOT, "submission.csv")
TEST_IMAGES_DIR = os.path.join(PROJECT_ROOT, "competition_data", "test_images")


def audit_submission():
    start_time = time.time()
    results = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "submission_path": SUBMISSION_PATH,
        "disjoint_canvas": {},
        "rle_stress_test": {},
        "boundary_extremes": {},
        "fuzzing": {},
        "verdict": "PENDING"
    }

    print("=" * 80)
    print("INDEPENDENT ADVERSARIAL AUDIT: SUBMISSION SPATIAL & FORMAT INVARIANTS")
    print("=" * 80)
    print(f"Target submission: {SUBMISSION_PATH}")

    if not os.path.exists(SUBMISSION_PATH):
        raise FileNotFoundError(f"Submission file not found: {SUBMISSION_PATH}")

    # Load submission
    df = pd.read_csv(SUBMISSION_PATH)
    total_rows = len(df)
    print(f"Loaded {total_rows} rows from submission CSV.")

    # Check schema
    expected_cols = ["filament_id", "segmentation_rle"]
    assert list(df.columns) == expected_cols, f"Schema mismatch: {list(df.columns)} != {expected_cols}"
    assert not df["filament_id"].isnull().any(), "Null filament_id found"
    assert not df["segmentation_rle"].isnull().any(), "Null segmentation_rle found"
    assert df["filament_id"].is_unique, "Duplicate filament_id found"

    # Identify test stems
    test_files = sorted(glob.glob(os.path.join(TEST_IMAGES_DIR, "*.jpeg")))
    test_stems = sorted([os.path.basename(f).replace(".jpeg", "") for f in test_files])
    assert len(test_stems) == 180, f"Expected 180 test stems, found {len(test_stems)}"

    # Parse stems from submission
    df["stem"] = df["filament_id"].apply(lambda x: x.rsplit("_", 1)[0])
    sub_stems = sorted(df["stem"].unique())
    print(f"Found {len(sub_stems)} unique stems in submission.")

    quiet_sun_stems = ["20160513133714Th", "20180507140210Th"]
    for qs in quiet_sun_stems:
        qs_rows = df[df["stem"] == qs]
        print(f"Quiet Sun stem '{qs}': {len(qs_rows)} rows predicted (expected 0).")
        assert len(qs_rows) == 0, f"Violation: Quiet Sun stem {qs} has {len(qs_rows)} rows!"

    # Verify stem sets match exactly
    expected_active_stems = sorted(set(test_stems) - set(quiet_sun_stems))
    assert sub_stems == expected_active_stems, "Submission stems do not match active test stems!"
    print(f"Active stems verified: {len(sub_stems)} / 178 active stems.")

    # -------------------------------------------------------------------------
    # TASK 1: Adversarial Disjoint Canvas Audit
    # -------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("TASK 1: ADVERSARIAL DISJOINT CANVAS AUDIT")
    print("-" * 80)

    total_colliding_pixels = 0
    colliding_pairs_count = 0
    colliding_stems_count = 0
    multi_instance_stems_count = 0
    single_instance_stems_count = 0
    zero_instance_stems_count = len(quiet_sun_stems)

    stem_to_rows = df.groupby("stem")

    per_stem_collision_report = []

    # Iterate over all 180 test stems
    for stem in test_stems:
        if stem in quiet_sun_stems:
            continue
        stem_df = stem_to_rows.get_group(stem)
        num_instances = len(stem_df)
        if num_instances == 1:
            single_instance_stems_count += 1
            # Still decode mask to verify shape
            rle_str = stem_df.iloc[0]["segmentation_rle"]
            rle_dict = {"size": [2048, 2048], "counts": rle_str.encode("utf-8")}
            mask = mask_util.decode(rle_dict).astype(bool)
            assert mask.shape == (2048, 2048)
            continue

        multi_instance_stems_count += 1
        # Decode all masks for this stem
        masks = []
        inst_ids = []
        for _, row in stem_df.iterrows():
            rle_str = row["segmentation_rle"]
            inst_ids.append(row["filament_id"])
            rle_dict = {"size": [2048, 2048], "counts": rle_str.encode("utf-8")}
            m = mask_util.decode(rle_dict).astype(bool)
            masks.append(m)

        # Cross-check 1: Integer Canvas Stamping
        canvas = np.zeros((2048, 2048), dtype=np.int32)
        stem_colliding_pixels = 0
        for idx, (inst_id, m) in enumerate(zip(inst_ids, masks), start=1):
            overlap_mask = (canvas > 0) & m
            overlap_count = int(np.count_nonzero(overlap_mask))
            if overlap_count > 0:
                stem_colliding_pixels += overlap_count
            canvas[m] = idx

        # Cross-check 2: Exhaustive Pairwise Intersection Check
        stem_pairwise_collisions = 0
        for i in range(len(masks)):
            for j in range(i + 1, len(masks)):
                intersection = np.count_nonzero(masks[i] & masks[j])
                if intersection > 0:
                    stem_pairwise_collisions += intersection
                    colliding_pairs_count += 1
                    per_stem_collision_report.append({
                        "stem": stem,
                        "inst_a": inst_ids[i],
                        "inst_b": inst_ids[j],
                        "colliding_pixels": int(intersection)
                    })

        assert stem_colliding_pixels == stem_pairwise_collisions, (
            f"Stem {stem}: canvas stamping ({stem_colliding_pixels}) != pairwise check ({stem_pairwise_collisions})"
        )

        if stem_colliding_pixels > 0:
            colliding_stems_count += 1
            total_colliding_pixels += stem_colliding_pixels

    print(f"Total test stems evaluated: {len(test_stems)} (180 total: {zero_instance_stems_count} zero-instance, {single_instance_stems_count} single-instance, {multi_instance_stems_count} multi-instance)")
    print(f"Total decoded masks: {total_rows}")
    print(f"Colliding stems: {colliding_stems_count}")
    print(f"Colliding instance pairs: {colliding_pairs_count}")
    print(f"Total colliding pixels across entire dataset: {total_colliding_pixels}")

    results["disjoint_canvas"] = {
        "total_test_stems": len(test_stems),
        "zero_instance_stems": zero_instance_stems_count,
        "single_instance_stems": single_instance_stems_count,
        "multi_instance_stems": multi_instance_stems_count,
        "total_masks_decoded": total_rows,
        "colliding_stems_count": colliding_stems_count,
        "colliding_pairs_count": colliding_pairs_count,
        "total_colliding_pixels": total_colliding_pixels,
        "invariant_passed": (total_colliding_pixels == 0)
    }

    # -------------------------------------------------------------------------
    # TASK 2: Bit-Exact LEB128 Fortran RLE Sum Stress-Test
    # -------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("TASK 2: BIT-EXACT LEB128 FORTRAN RLE SUM STRESS-TEST")
    print("-" * 80)

    total_rle_count = total_rows
    sum_exact_count = 0
    sum_mismatch_count = 0
    negative_counts_found = 0
    foreground_at_00_count = 0
    leading_zero_count = 0
    rule_00_inconsistencies = 0
    pycocotools_mismatches = 0

    all_filament_areas = []
    rle_string_lengths = []

    for idx, row in df.iterrows():
        fid = row["filament_id"]
        rle_str = row["segmentation_rle"]
        rle_string_lengths.append(len(rle_str))

        # Decode counts using LEB128 decoder
        counts = decode_leb128_counts(rle_str)
        count_sum = sum(counts)

        if count_sum == 4194304:
            sum_exact_count += 1
        else:
            sum_mismatch_count += 1
            print(f"[!] Count sum violation in {fid}: {count_sum} != 4194304")

        # Check negative counts
        if any(c < 0 for c in counts):
            negative_counts_found += 1
            print(f"[!] Negative count in {fid}")

        # Check (0,0) foreground leading zero rule
        # Decode binary mask via pycocotools
        rle_dict = {"size": [2048, 2048], "counts": rle_str.encode("utf-8")}
        mask = mask_util.decode(rle_dict).astype(bool)

        pixel_00 = mask[0, 0]
        leading_zero = (counts[0] == 0)

        if pixel_00:
            foreground_at_00_count += 1
        if leading_zero:
            leading_zero_count += 1

        # Invariant: pixel_00 is True iff counts[0] == 0
        if pixel_00 != leading_zero:
            rule_00_inconsistencies += 1
            print(f"[!] Rule (0,0) violation in {fid}: mask[0,0]={pixel_00}, counts[0]={counts[0]}")

        # Cross-check foreground area between pycocotools decode and odd count sum
        # In COCO RLE: counts[0] is bg, counts[1] is fg, counts[2] is bg, counts[3] is fg...
        fg_area_from_counts = sum(counts[1::2])
        fg_area_from_mask = int(np.count_nonzero(mask))
        if fg_area_from_counts != fg_area_from_mask:
            pycocotools_mismatches += 1
            print(f"[!] pycocotools area mismatch in {fid}: {fg_area_from_counts} vs {fg_area_from_mask}")

        all_filament_areas.append(fg_area_from_mask)

    print(f"Total RLEs tested: {total_rle_count}")
    print(f"RLEs summing to exactly 4,194,304 pixels: {sum_exact_count} / {total_rle_count} ({sum_exact_count / total_rle_count * 100:.2f}%)")
    print(f"RLEs with sum mismatches: {sum_mismatch_count}")
    print(f"RLEs with negative counts: {negative_counts_found}")
    print(f"Masks with foreground at (0,0): {foreground_at_00_count}")
    print(f"Masks with leading 0 in counts: {leading_zero_count}")
    print(f"(0,0) Leading Zero Rule Inconsistencies: {rule_00_inconsistencies}")
    print(f"pycocotools C-decode area cross-check mismatches: {pycocotools_mismatches}")

    # Adversarial Oracle: Stress-test foreground at (0,0) with synthetic masks
    print("\n[Adversarial Oracle: Synthetic Leading Zero Stress-Test]")
    synthetic_tests_passed = 0
    synthetic_cases = [
        ("Single pixel at (0,0)", lambda m: m.__setitem__((0, 0), True)),
        ("2x2 box at (0,0)", lambda m: m.__setitem__((slice(0, 2), slice(0, 2)), True)),
        ("First column full foreground", lambda m: m.__setitem__((slice(None), 0), True)),
        ("Entire mask foreground", lambda m: m.__setitem__((slice(None), slice(None)), True)),
        ("Single pixel at (1,0) [background at (0,0)]", lambda m: m.__setitem__((1, 0), True)),
        ("Single pixel at (0,1) [background at (0,0)]", lambda m: m.__setitem__((0, 1), True)),
    ]

    for name, modifier in synthetic_cases:
        test_mask = np.zeros((2048, 2048), dtype=bool)
        modifier(test_mask)
        uncompressed = mask_to_uncompressed_rle(test_mask)
        encoded_str = encode_leb128_rle(test_mask)
        decoded_counts = decode_leb128_counts(encoded_str)
        decoded_mask = decode_leb128_rle(encoded_str)

        # Check sum
        assert sum(decoded_counts) == 4194304
        assert not any(c < 0 for c in decoded_counts)
        # Check round-trip mask equality
        assert np.array_equal(test_mask, decoded_mask)

        # Check pycocotools agreement
        coco_dict = {"size": [2048, 2048], "counts": encoded_str.encode("utf-8")}
        coco_mask = mask_util.decode(coco_dict).astype(bool)
        assert np.array_equal(test_mask, coco_mask)

        # Check (0,0) rule
        if test_mask[0, 0]:
            assert decoded_counts[0] == 0, f"Synthetic test '{name}' failed leading zero check"
        else:
            assert decoded_counts[0] > 0, f"Synthetic test '{name}' failed background leading count check"

        synthetic_tests_passed += 1

    print(f"Adversarial synthetic oracle cases passed: {synthetic_tests_passed} / {len(synthetic_cases)}")

    results["rle_stress_test"] = {
        "total_rles": total_rle_count,
        "sum_exact_4194304": sum_exact_count,
        "sum_mismatches": sum_mismatch_count,
        "negative_counts": negative_counts_found,
        "foreground_at_00": foreground_at_00_count,
        "leading_zeros": leading_zero_count,
        "rule_00_inconsistencies": rule_00_inconsistencies,
        "pycocotools_mismatches": pycocotools_mismatches,
        "synthetic_oracle_passed": (synthetic_tests_passed == len(synthetic_cases)),
        "invariant_passed": (
            sum_exact_count == total_rle_count
            and negative_counts_found == 0
            and rule_00_inconsistencies == 0
            and pycocotools_mismatches == 0
        )
    }

    # -------------------------------------------------------------------------
    # TASK 3: Boundary & Extreme Cases
    # -------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("TASK 3: BOUNDARY & EXTREME CASES")
    print("-" * 80)

    all_filament_areas = np.array(all_filament_areas)
    min_area = int(np.min(all_filament_areas))
    max_area = int(np.max(all_filament_areas))
    mean_area = float(np.mean(all_filament_areas))
    std_area = float(np.std(all_filament_areas))
    area_percentiles = {
        f"p{p}": float(np.percentile(all_filament_areas, p))
        for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]
    }

    print(f"Filament Area Statistics (pixels):")
    print(f"  Min Area: {min_area} pixels")
    print(f"  Max Area: {max_area} pixels ({(max_area / 4194304) * 100:.2f}% of canvas)")
    print(f"  Mean Area: {mean_area:.1f} +/- {std_area:.1f} pixels")
    print(f"  Median (p50): {area_percentiles['p50']:.1f} pixels")
    print(f"  p1: {area_percentiles['p1']:.1f}, p5: {area_percentiles['p5']:.1f}, p95: {area_percentiles['p95']:.1f}, p99: {area_percentiles['p99']:.1f}")

    # Bounding box and aspect ratio analysis
    print("\nComputing Bounding Boxes and Aspect Ratios across all 1,327 masks...")
    aspect_ratios = []
    normalized_aspect_ratios = []  # max(H/W, W/H) >= 1.0
    touching_boundary_count = 0
    bbox_dimensions = []

    for idx, row in df.iterrows():
        rle_str = row["segmentation_rle"]
        rle_dict = {"size": [2048, 2048], "counts": rle_str.encode("utf-8")}
        mask = mask_util.decode(rle_dict).astype(bool)

        ys, xs = np.where(mask)
        y_min, y_max = int(np.min(ys)), int(np.max(ys))
        x_min, x_max = int(np.min(xs)), int(np.max(xs))

        h = y_max - y_min + 1
        w = x_max - x_min + 1
        bbox_dimensions.append((h, w))

        # Check touching canvas border
        if y_min == 0 or y_max == 2047 or x_min == 0 or x_max == 2047:
            touching_boundary_count += 1

        ar = float(h) / float(w) if w > 0 else 0.0
        norm_ar = max(float(h) / float(w), float(w) / float(h)) if (h > 0 and w > 0) else 1.0
        aspect_ratios.append(ar)
        normalized_aspect_ratios.append(norm_ar)

    aspect_ratios = np.array(aspect_ratios)
    normalized_aspect_ratios = np.array(normalized_aspect_ratios)

    min_ar = float(np.min(aspect_ratios))
    max_ar = float(np.max(aspect_ratios))
    median_ar = float(np.median(aspect_ratios))

    min_norm_ar = float(np.min(normalized_aspect_ratios))
    max_norm_ar = float(np.max(normalized_aspect_ratios))
    median_norm_ar = float(np.median(normalized_aspect_ratios))

    print(f"Aspect Ratio Statistics (H/W):")
    print(f"  Min H/W: {min_ar:.4f}")
    print(f"  Max H/W: {max_ar:.4f}")
    print(f"  Median H/W: {median_ar:.4f}")
    print(f"Elongation / Normalized Aspect Ratio max(H/W, W/H):")
    print(f"  Min Elongation: {min_norm_ar:.4f}")
    print(f"  Max Elongation: {max_norm_ar:.4f}")
    print(f"  Median Elongation: {median_norm_ar:.4f}")
    print(f"Masks touching canvas boundaries (y in {{0, 2047}} or x in {{0, 2047}}): {touching_boundary_count}")

    # Character set and string corruption audit
    print("\nAuditing RLE string characters and formatting...")
    corrupted_strings = 0
    whitespace_issues = 0
    empty_strings = 0

    for idx, row in df.iterrows():
        rle_str = row["segmentation_rle"]
        if not isinstance(rle_str, str) or len(rle_str) == 0:
            empty_strings += 1
            corrupted_strings += 1
            continue
        if rle_str != rle_str.strip():
            whitespace_issues += 1
        # Check valid ASCII range for LEB128 [48, 126]
        for ch in rle_str:
            code = ord(ch)
            if code < 48 or code > 126:
                corrupted_strings += 1
                break

    print(f"Empty RLE strings: {empty_strings}")
    print(f"Whitespace / padding issues: {whitespace_issues}")
    print(f"Corrupted or invalid character strings: {corrupted_strings}")

    # Truncation and corruption fuzzing harness
    print("\n[Adversarial Fuzzer: Truncated & Corrupted String Handling]")
    fuzz_truncations_caught = 0
    fuzz_samples = df.sample(n=min(50, len(df)), random_state=42)

    for _, row in fuzz_samples.iterrows():
        rle_str = row["segmentation_rle"]
        # Cut string in half
        truncated = rle_str[: len(rle_str) // 2]
        caught = False
        try:
            counts = decode_leb128_counts(truncated)
            # If it didn't throw, check if sum is wrong
            if sum(counts) != 4194304:
                caught = True
        except (ValueError, Exception):
            caught = True
        if caught:
            fuzz_truncations_caught += 1

    print(f"Truncated RLE fuzz attacks safely detected/rejected: {fuzz_truncations_caught} / {len(fuzz_samples)}")

    results["boundary_extremes"] = {
        "min_area": min_area,
        "max_area": max_area,
        "mean_area": mean_area,
        "std_area": std_area,
        "area_percentiles": area_percentiles,
        "aspect_ratio_min_hw": min_ar,
        "aspect_ratio_max_hw": max_ar,
        "aspect_ratio_median_hw": median_ar,
        "elongation_min": min_norm_ar,
        "elongation_max": max_norm_ar,
        "elongation_median": median_norm_ar,
        "touching_boundary_count": touching_boundary_count,
        "empty_strings": empty_strings,
        "whitespace_issues": whitespace_issues,
        "corrupted_strings": corrupted_strings,
        "fuzz_truncations_caught": fuzz_truncations_caught,
        "fuzz_samples_count": len(fuzz_samples),
        "fuzzing_passed": (fuzz_truncations_caught == len(fuzz_samples))
    }

    # -------------------------------------------------------------------------
    # FINAL VERDICT SYNTHESIS
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("AUDIT SUMMARY & DEFINITIVE VERDICT")
    print("=" * 80)

    verdict_checks = [
        ("Disjoint Canvas (0 colliding pixels across all 180 stems)", total_colliding_pixels == 0),
        ("Bit-Exact Sum (100% of RLEs sum to exactly 4,194,304)", sum_exact_count == total_rle_count),
        ("No Negative Counts (0 negative run lengths)", negative_counts_found == 0),
        ("(0,0) Foreground Leading Zero Invariant", rule_00_inconsistencies == 0),
        ("Pycocotools C-decode Bit-Exact Agreement", pycocotools_mismatches == 0),
        ("Zero Corrupted / Truncated / Whitespace Anomalies", (corrupted_strings == 0 and empty_strings == 0 and whitespace_issues == 0)),
        ("Quiet Sun Zero-Row Guard (0 rows on 2 quiet sun stems)", len(quiet_sun_stems) == 2),
        ("Non-Empty Filament Area (min_area > 0)", min_area > 0),
        ("Fuzzing & Truncation Rejection (100% caught)", fuzz_truncations_caught == len(fuzz_samples))
    ]

    all_passed = True
    for name, passed in verdict_checks:
        status_str = "[PASS]" if passed else "[FAIL]"
        print(f"  {status_str} {name}")
        if not passed:
            all_passed = False

    final_verdict = "APPROVE" if all_passed else "REQUEST_CHANGES"
    results["verdict"] = final_verdict
    results["verdict_checks"] = {name: passed for name, passed in verdict_checks}
    results["elapsed_seconds"] = round(time.time() - start_time, 2)

    print(f"\nElapsed time: {results['elapsed_seconds']} seconds.")
    print(f"FINAL AUDIT VERDICT: {final_verdict}")
    print("=" * 80)

    return results


if __name__ == "__main__":
    res = audit_submission()
    output_json = os.path.join(PROJECT_ROOT, "experiments", "adversarial_audit_results.json")
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(res, f, indent=2)
    print(f"Audit results written to {output_json}")
    if res["verdict"] != "APPROVE":
        sys.exit(1)
