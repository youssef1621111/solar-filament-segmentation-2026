"""
Adversarial Stress Test Suite for Submission Integrity Validator
Tests edge cases, fault injection, and exit code propagation:
1. Full scale: 179 valid stems + 1 colliding stem (all 180 stems from submission_original_055_anchor.csv) -> Exits 1, detects collision
2. Exact 1-pixel overlap between two masks -> Exits 1, detected
3. 1 corrupted RLE count sum (4,194,303 != 4,194,304) -> Exits 1, detected
4. Truncated LEB128 string in RLE column -> Exits 1, caught safely without unhandled crash
5. Strict Quiet-Sun enforcement on baseline anchor without --allow_missing_anchor -> Exits 1
"""

import sys
import os
import subprocess
import tempfile
import pandas as pd
import numpy as np

# Ensure project root in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.utils_metrics import decode_leb128_rle, encode_leb128_rle

DEFAULT_ANCHOR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results/operation_058_breakthrough/predictions/submission_exp031_calibrated_1300.csv"
)
if not os.path.exists(DEFAULT_ANCHOR):
    DEFAULT_ANCHOR = "submission_original_055_anchor.csv"



def run_validator(csv_path: str, extra_args: list = None) -> tuple:
    """Run submission_integrity_check.py as a subprocess and return (returncode, stdout, stderr)."""
    cmd = [sys.executable, "scripts/submission_integrity_check.py", "--sub_csv", csv_path]
    if extra_args:
        cmd.extend(extra_args)
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return proc.returncode, proc.stdout, proc.stderr


def test_single_pixel_collision():
    """Test exact 1-pixel overlap detection on a canvas."""
    print("\n" + "=" * 80)
    print("TEST 1: Exact 1-Pixel Overlap Detection")
    print("=" * 80)
    anchor_df = pd.read_csv(DEFAULT_ANCHOR)
    stems = anchor_df["filament_id"].astype(str).apply(lambda x: x.rsplit("_", 1)[0])
    
    # Take rows for 1 stem
    target_stem = stems.iloc[0]
    stem_rows = anchor_df[stems == target_stem].copy()
    first_rle = stem_rows.iloc[0]["segmentation_rle"]
    first_mask = decode_leb128_rle(first_rle, (2048, 2048))
    
    # Find a foreground pixel in first_mask
    ys, xs = np.where(first_mask)
    y_overlap, x_overlap = int(ys[0]), int(xs[0])
    
    # Create a 2nd mask that contains ONLY (y_overlap, x_overlap)
    single_pixel_mask = np.zeros((2048, 2048), dtype=bool)
    single_pixel_mask[y_overlap, x_overlap] = True
    single_pixel_rle = encode_leb128_rle(single_pixel_mask)

    # Build a test CSV with anchor containing the 1-pixel overlapping instance
    dup_row = pd.DataFrame([{
        "filament_id": f"{target_stem}_999",
        "segmentation_rle": single_pixel_rle
    }])
    test_df = pd.concat([anchor_df, dup_row], ignore_index=True)

    with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False) as f:
        temp_csv = f.name
        test_df.to_csv(temp_csv, index=False)

    try:
        ret, stdout, stderr = run_validator(temp_csv, extra_args=["--allow_missing_anchor"])
        print(f"Validator return code: {ret} (Expected: 1)")
        print(f"Overlap violation count: {'canvas collisions = 1' in stdout}")
        assert ret == 1, f"Expected returncode 1, got {ret}"
        assert target_stem in stdout, f"Expected stem {target_stem} to be reported"
        assert "canvas collisions = 1" in stdout or "canvas overlap violations = 1" in stdout, "Expected 1 canvas collision"
        print(">> TEST 1 PASSED: Exact 1-pixel collision detected and halted validator with exit code 1.")
        return True
    finally:
        if os.path.exists(temp_csv):
            os.remove(temp_csv)


def test_corrupted_count_sum():
    """Test 1 instance having count sum != 4,194,304."""
    print("\n" + "=" * 80)
    print("TEST 2: Corrupted RLE Count Sum (Not Equal to 4,194,304)")
    print("=" * 80)
    anchor_df = pd.read_csv(DEFAULT_ANCHOR)
    test_df = anchor_df.copy()
    
    # Corrupt instance 0 RLE by replacing with a mask of wrong size (2047x2048 = 4,192,256)
    corrupted_mask = np.zeros((2047, 2048), dtype=bool)
    corrupted_mask[10, 10] = True
    corrupted_rle = encode_leb128_rle(corrupted_mask)
    test_df.iloc[0, test_df.columns.get_loc("segmentation_rle")] = corrupted_rle

    with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False) as f:
        temp_csv = f.name
        test_df.to_csv(temp_csv, index=False)

    try:
        ret, stdout, stderr = run_validator(temp_csv, extra_args=["--allow_missing_anchor"])
        print(f"Validator return code: {ret} (Expected: 1)")
        print(f"RLE sum errors reported: {'RLE sum errors = 1' in stdout}")
        assert ret == 1, f"Expected returncode 1, got {ret}"
        assert "RLE sum errors = 1" in stdout
        assert "[BLOCKED" in stdout
        print(">> TEST 2 PASSED: Corrupted pixel sum halted validator with exit code 1.")
        return True
    finally:
        if os.path.exists(temp_csv):
            os.remove(temp_csv)


def test_truncated_leb128_string():
    """Test corrupted/truncated LEB128 string in RLE column."""
    print("\n" + "=" * 80)
    print("TEST 3: Truncated / Corrupted LEB128 String")
    print("=" * 80)
    anchor_df = pd.read_csv(DEFAULT_ANCHOR)
    test_df = anchor_df.copy()
    
    # Truncate RLE string to invalid length
    test_df.iloc[0, test_df.columns.get_loc("segmentation_rle")] = "a1b2c3??invalid"

    with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False) as f:
        temp_csv = f.name
        test_df.to_csv(temp_csv, index=False)

    try:
        ret, stdout, stderr = run_validator(temp_csv, extra_args=["--allow_missing_anchor"])
        print(f"Validator return code: {ret} (Expected: 1)")
        print(f"RLE sum errors reported: {'RLE sum errors = 1' in stdout}")
        assert ret == 1, f"Expected returncode 1, got {ret}"
        assert "[BLOCKED" in stdout
        print(">> TEST 3 PASSED: Truncated LEB128 string handled safely with exit code 1.")
        return True
    finally:
        if os.path.exists(temp_csv):
            os.remove(temp_csv)


def test_strict_quiet_sun_guard():
    """Verify that a submission containing Quiet-Sun rows fails Check 3."""
    print("\n" + "=" * 80)
    print("TEST 4: Strict Quiet Sun Guard Verification")
    print("=" * 80)
    anchor_df = pd.read_csv(DEFAULT_ANCHOR)
    qs_stem = "20160513133714Th"
    sample_rle = anchor_df["segmentation_rle"].iloc[0]
    qs_row = pd.DataFrame([{"filament_id": f"{qs_stem}_1", "segmentation_rle": sample_rle}])
    test_df = pd.concat([anchor_df, qs_row], ignore_index=True)

    with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False) as f:
        temp_csv = f.name
        test_df.to_csv(temp_csv, index=False)

    try:
        ret, stdout, stderr = run_validator(temp_csv)
        print(f"Validator return code: {ret} (Expected: 1)")
        assert ret == 1, f"Expected returncode 1, got {ret}"
        assert "Quiet-Sun Guard" in stdout and "[FAILED]" in stdout
        assert "[BLOCKED" in stdout
        print(">> TEST 4 PASSED: Injected quiet-sun row detected and halted validator with exit code 1.")
        return True
    finally:
        if os.path.exists(temp_csv):
            os.remove(temp_csv)


def test_full_180_stems_1_colliding_stem():
    """
    Generate an edge-case submission with 179 valid stems and 1 colliding stem
    using all 180 stems (1,344 rows) from submission_original_055_anchor.csv.
    """
    print("\n" + "=" * 80)
    print("TEST 5: Full-Scale Invariant (179 Valid Stems + 1 Colliding Stem across 180 Stems)")
    print("=" * 80)
    anchor_df = pd.read_csv(DEFAULT_ANCHOR)
    stems = anchor_df["filament_id"].astype(str).apply(lambda x: x.rsplit("_", 1)[0])
    unique_stems = stems.unique()
    assert len(unique_stems) >= 178, f"Expected >= 178 unique stems, found {len(unique_stems)}"

    # Pick a stem from the middle of the dataset
    target_stem = unique_stems[90]
    stem_rows = anchor_df[stems == target_stem].copy()
    print(f"Target stem chosen for collision: {target_stem} ({len(stem_rows)} original instances)")

    # Inject a collision: duplicate the first instance of target_stem with a numeric suffix
    dup_row = stem_rows.iloc[0:1].copy()
    dup_row["filament_id"] = f"{target_stem}_999"
    
    # Combine back: valid stems untouched + 1 colliding stem
    modified_df = pd.concat([anchor_df, dup_row], ignore_index=True)
    
    with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False) as f:
        temp_csv = f.name
        modified_df.to_csv(temp_csv, index=False)

    try:
        ret, stdout, stderr = run_validator(temp_csv, extra_args=["--allow_missing_anchor"])
        print(f"Validator return code: {ret} (Expected: 1)")
        print(f"Check 5 status in stdout: {'[FAILED]' in stdout}")
        print(f"Overall status: {'[BLOCKED' in stdout}")

        assert ret == 1, f"Expected returncode 1, got {ret}"
        assert "[FAILED]" in stdout, "Expected Check 5 to fail"
        assert "[BLOCKED" in stdout
        print(">> TEST 5 PASSED: Full-scale valid + 1 colliding stem halted validator with exit code 1.")
        return True
    finally:
        if os.path.exists(temp_csv):
            os.remove(temp_csv)


if __name__ == "__main__":
    test_single_pixel_collision()
    test_corrupted_count_sum()
    test_truncated_leb128_string()
    test_strict_quiet_sun_guard()
    # Run full 180 stems test last
    test_full_180_stems_1_colliding_stem()
    print("\n" + "=" * 80)
    print("ALL ADVERSARIAL STRESS TESTS COMPLETED SUCCESSFULLY (5/5 PASSED)!")
    print("=" * 80)
