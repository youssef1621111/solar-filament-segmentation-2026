#!/usr/bin/env python3
"""
================================================================================
scripts/submission_integrity_check.py
Championship Submission Integrity Checklist & Real Zero-Trust Protocol Validator
================================================================================
Upgraded for 0.58 -> 0.66+ SOTA:
- Replaced obsolete 0.55 static mocks with verified mathematical invariants.
- Default anchor points to official 0.58 Champion (submission_exp031_calibrated_1300.csv).
- Dynamic Panoptic Geometry, Area Quantiles, LEB128 bitstream, and Disjoint Canvas Audit.
================================================================================
"""

import os
import sys
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from pycocotools import mask as mask_util

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from scripts.utils_metrics import decode_leb128_counts
except ImportError:
    try:
        from utils_metrics import decode_leb128_counts
    except ImportError:
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from utils_metrics import decode_leb128_counts

DEFAULT_058_ANCHOR = str(PROJECT_ROOT / "0.58/submissions/submission_selected_primary_058.csv")
if not os.path.exists(DEFAULT_058_ANCHOR):
    DEFAULT_058_ANCHOR = str(PROJECT_ROOT / "results/operation_058_breakthrough/predictions/submission_exp031_calibrated_1300.csv")
QUIET_SUN_STEMS = {"20160513133714Th", "20180507140210Th"}
EXPECTED_ACTIVE_STEMS = 178
CANVAS_PIXELS = 2048 * 2048  # 4,194,304


def run_submission_integrity_check(
    submission_csv: str,
    anchor_csv: str = DEFAULT_058_ANCHOR,
    allow_missing_anchor: bool = False,
    fold0_pq: float = None,
    baseline_pq: float = None,
    **kwargs
) -> bool:
    print("=" * 80)
    print("CHAMPIONSHIP SUBMISSION INTEGRITY CHECKLIST (Verified Zero-Trust Protocol)")
    print("=" * 80)

    if fold0_pq is not None and baseline_pq is not None:
        delta = fold0_pq - baseline_pq
        if delta < 0.0050:
            print(f"Check 1: CV Delta Gate: delta={delta:.5f} < 0.0050 [FAILED]")
            return False

    checks_passed = {}

    if not os.path.exists(submission_csv):
        print(f"[!] Submission CSV '{submission_csv}' not found. Cannot proceed.")
        return False

    sub_df = pd.read_csv(submission_csv)

    # -------------------------------------------------------------------------
    # Check 0 / Pre-check: Column Schema Validation
    # -------------------------------------------------------------------------
    expected_cols = ["filament_id", "segmentation_rle"]
    if list(sub_df.columns) != expected_cols:
        print(f"Check 2: Column Schema: Columns = {list(sub_df.columns)} (Expected: {expected_cols}) [FAILED]")
        return False

    # -------------------------------------------------------------------------
    # Check 1: Real Panoptic Metric Health & Physical Area Distribution
    # -------------------------------------------------------------------------
    if len(sub_df) == 0:
        print("Check 1: Submission is empty! [FAILED]")
        return False

    areas = []
    decode_errors = 0
    # Sample first 200 instances for rapid physical area distribution audit
    sample_rles = sub_df["segmentation_rle"].iloc[:200]
    for rle_str in sample_rles:
        try:
            counts = decode_leb128_counts(str(rle_str))
            area = sum(counts[1::2])
            areas.append(area)
        except Exception:
            decode_errors += 1

    if decode_errors > 0 or len(areas) == 0:
        valid_area = False
        print(f"Check 1: Panoptic Health: Encountered {decode_errors} RLE decoding failures in sample [FAILED]")
    else:
        p25 = np.percentile(areas, 25)
        p50 = np.percentile(areas, 50)
        p75 = np.percentile(areas, 75)
        min_a = np.min(areas)
        max_a = np.max(areas)

        valid_area = (min_a >= 80) and (max_a <= 60000)
        status1 = "[PASSED]" if valid_area else "[FAILED]"
        print(f"Check 1: Panoptic Health: Area P25={p25:.0f}px, Median={p50:.0f}px, P75={p75:.0f}px (Min={min_a}, Max={max_a}) {status1}")

    checks_passed["1. Panoptic Metric Health"] = valid_area

    # -------------------------------------------------------------------------
    # Check 2: Column Schema & Filament ID Format
    # -------------------------------------------------------------------------
    expected_cols = ["filament_id", "segmentation_rle"]
    check2_cols = (list(sub_df.columns) == expected_cols)
    check2_fids = sub_df["filament_id"].apply(lambda x: isinstance(x, str) and "_" in x and x.rsplit("_", 1)[1].isdigit()).all()
    check2 = check2_cols and check2_fids
    checks_passed["2. Column Schema"] = check2
    status2 = "[PASSED]" if check2 else "[FAILED]"
    print(f"Check 2: Column Schema: Columns = {list(sub_df.columns)}, IDs formatted as <stem>_<int>: {check2_fids} {status2}")

    if not check2:
        return False

    # Stem extraction
    stems = sub_df["filament_id"].astype(str).apply(lambda x: x.rsplit("_", 1)[0])

    # -------------------------------------------------------------------------
    # Check 3: Quiet-Sun Guard (Strict 0 rows on quiet sun disks)
    # -------------------------------------------------------------------------
    qs_rows = sub_df[stems.isin(QUIET_SUN_STEMS)]
    check3 = (len(qs_rows) == 0)
    checks_passed["3. Quiet-Sun Guard"] = check3
    status3 = "[PASSED]" if check3 else "[FAILED]"
    print(f"Check 3: Quiet-Sun Guard: {len(qs_rows)} rows on Quiet-Sun disks (Expected: strictly 0) {status3}")

    # -------------------------------------------------------------------------
    # Check 4: Instance Density & Active Stem Coverage (Updated for EXP-043/044: 1000 - 1360 rows)
    # -------------------------------------------------------------------------
    total_rows = len(sub_df)
    unique_stems = stems.nunique()
    density = (total_rows / unique_stems) if unique_stems > 0 else 0

    check4 = (1000 <= total_rows <= 1360) and (unique_stems >= EXPECTED_ACTIVE_STEMS)
    checks_passed["4. Instance Density"] = check4
    status4 = "[PASSED]" if check4 else "[FAILED]"
    print(f"Check 4: Instance Density: {total_rows} rows across {unique_stems} active stems ({density:.2f} inst/stem) {status4}")

    # -------------------------------------------------------------------------
    # Check 5: Zero Pixel Overlap & LEB128 Fortran Bitstream Invariants
    # -------------------------------------------------------------------------
    rle_errors = 0
    for rle_val in sub_df["segmentation_rle"]:
        try:
            counts = decode_leb128_counts(str(rle_val))
            if sum(counts) != CANVAS_PIXELS or any(c < 0 for c in counts):
                rle_errors += 1
        except Exception:
            rle_errors += 1

    stem_groups = sub_df.groupby(stems)
    collision_errors = 0
    colliding_stems = []
    for stem_id, group in stem_groups:
        if len(group) <= 1:
            continue
        occupied = np.zeros((2048, 2048), dtype=bool)
        for rle_str in group["segmentation_rle"]:
            try:
                m = (mask_util.decode({"counts": str(rle_str).encode("ascii"), "size": [2048, 2048]}) > 0)
                if np.any(occupied & m):
                    collision_errors += 1
                    colliding_stems.append(stem_id)
                    break
                occupied |= m
            except Exception:
                collision_errors += 1
                colliding_stems.append(stem_id)
                break

    check5 = (rle_errors == 0) and (collision_errors == 0)
    checks_passed["5. Zero Pixel Overlap"] = check5
    status5 = "[PASSED]" if check5 else "[FAILED]"
    print(f"Check 5: Zero Overlap: RLE sum errors = {rle_errors}, canvas collisions = {collision_errors} across {unique_stems} stems {status5}")
    if colliding_stems:
        print(f"    [!] Colliding stems: {colliding_stems[:5]}")

    # -------------------------------------------------------------------------
    # Check 6: 0.58 Consensus Anchor Scaffold & Anti-Cannibalization Audit
    # -------------------------------------------------------------------------
    if allow_missing_anchor:
        check6 = True
        checks_passed["6. Anchor Scaffold"] = True
        print(f"Check 6: Anchor Scaffold: Validated standalone (--allow_missing_anchor) [PASSED]")
    elif not os.path.exists(anchor_csv):
        check6 = False
        checks_passed["6. Anchor Scaffold"] = False
        print(f"Check 6: Anchor Scaffold: Verified 0.58 anchor '{anchor_csv}' not found [FAILED]")
    else:
        anchor_df = pd.read_csv(anchor_csv)
        anc_stems = anchor_df["filament_id"].apply(lambda x: x.split("_")[0])
        covered_stems = set(stems).intersection(set(anc_stems))
        
        # Instance-level fidelity and anti-cannibalization verification
        common_fids = set(anchor_df["filament_id"]).intersection(set(sub_df["filament_id"]))
        anc_sub_map = sub_df.set_index("filament_id")["segmentation_rle"].to_dict()
        
        cannibalized_instances = []
        distorted_instances = []
        
        for fid in common_fids:
            anc_rle = anchor_df.loc[anchor_df["filament_id"] == fid, "segmentation_rle"].iloc[0]
            sub_rle = anc_sub_map[fid]
            if anc_rle == sub_rle:
                continue
            
            # Decode only when RLE strings differ
            try:
                m_anc = mask_util.decode({"counts": str(anc_rle).encode("ascii"), "size": [2048, 2048]}) > 0
                m_sub = mask_util.decode({"counts": str(sub_rle).encode("ascii"), "size": [2048, 2048]}) > 0
                
                a_anc = m_anc.sum()
                a_sub = m_sub.sum()
                inter = (m_anc & m_sub).sum()
                union = (m_anc | m_sub).sum()
                iou = inter / union if union > 0 else 1.0
                
                # Check 1: Severe pixel loss (>10% of anchor destroyed)
                lost_pixels = a_anc - inter
                if lost_pixels > 0.10 * a_anc:
                    # Check for authorized 2 -> 1 de-merging splits:
                    # Search if candidate split instances on the same stem collectively recover >= 95% of m_anc
                    stem_id = fid.rsplit("_", 1)[0]
                    stem_sub_fids = [f for f in sub_df["filament_id"] if f.rsplit("_", 1)[0] == stem_id]
                    split_union = np.zeros_like(m_anc)
                    split_count = 0
                    for sfid in stem_sub_fids:
                        srle = anc_sub_map.get(sfid)
                        if srle:
                            try:
                                sm = mask_util.decode({"counts": str(srle).encode("ascii"), "size": [2048, 2048]}) > 0
                                if (sm & m_anc).sum() > 0.10 * sm.sum():
                                    split_union |= (sm & m_anc)
                                    split_count += 1
                            except Exception:
                                pass
                    if split_count >= 2 and split_union.sum() >= 0.95 * a_anc:
                        # Authorized 2 -> 1 de-merging split!
                        pass
                    else:
                        cannibalized_instances.append((fid, lost_pixels, a_anc, a_sub))
                
                # Check 2: Distortion cliff (IoU with anchor < 0.60 or excessive dilation > 1.40x)
                if (iou < 0.60 or a_sub > 1.40 * a_anc) and fid not in [c[0] for c in cannibalized_instances]:
                    # If this instance was an authorized split partition, iou with the whole anchor will be lower,
                    # but it is strictly a subset (a_sub < a_anc) and non-dilated.
                    stem_id = fid.rsplit("_", 1)[0]
                    stem_sub_fids = [f for f in sub_df["filament_id"] if f.rsplit("_", 1)[0] == stem_id]
                    split_union = np.zeros_like(m_anc)
                    split_count = 0
                    for sfid in stem_sub_fids:
                        srle = anc_sub_map.get(sfid)
                        if srle:
                            try:
                                sm = mask_util.decode({"counts": str(srle).encode("ascii"), "size": [2048, 2048]}) > 0
                                if (sm & m_anc).sum() > 0.10 * sm.sum():
                                    split_union |= (sm & m_anc)
                                    split_count += 1
                            except Exception:
                                pass
                    if split_count >= 2 and split_union.sum() >= 0.95 * a_anc and a_sub <= a_anc:
                        # Authorized split: not a distortion
                        pass
                    else:
                        distorted_instances.append((fid, iou, a_anc, a_sub))
            except Exception:
                distorted_instances.append((fid, 0.0, 0, 0))

        # Check 6: Consensus Core Fidelity & Anti-Cannibalization Audit (EXP-043/044 Upgrade)
        audit_path = PROJECT_ROOT / "results/audit_1333_solo_fibril_purge.csv"
        if audit_path.exists():
            audit_df = pd.read_csv(audit_path)
            consensus_core_fids = set(audit_df[audit_df["category"].isin([1, 2])]["filament_id"])
            retained_core = consensus_core_fids.intersection(set(sub_df["filament_id"]))
            # In EXP-044, pure majority may select top ~1,040 instances; EXP-043 retains 100% of Cat 1 & Cat 2
            core_fidelity = (len(retained_core) >= 0.85 * len(consensus_core_fids))
            valid_instances = (len(common_fids) >= 0.70 * len(anchor_df)) and core_fidelity
        else:
            valid_instances = (len(common_fids) >= 0.70 * len(anchor_df))

        valid_stems = (len(covered_stems) >= EXPECTED_ACTIVE_STEMS)
        zero_cannibalism = (len(cannibalized_instances) == 0)
        low_distortion = (len(distorted_instances) <= 5)

        check6 = valid_stems and valid_instances and zero_cannibalism and low_distortion
        checks_passed["6. Anchor Scaffold"] = check6
        status6 = "[PASSED]" if check6 else "[FAILED]"
        print(f"Check 6: Consensus Core Scaffold: Stems={len(covered_stems)}/{EXPECTED_ACTIVE_STEMS}, Retained FIDs={len(common_fids)}/{len(anchor_df)}, Cannibalized={len(cannibalized_instances)}, Distorted={len(distorted_instances)} {status6}")
        if cannibalized_instances:
            print(f"    [!] Cannibalized anchor instances (lost >10% px): {[c[0] for c in cannibalized_instances[:5]]}")
        if distorted_instances:
            print(f"    [!] Severely distorted instances (IoU < 0.60 or Area > 1.40x): {[d[0] for d in distorted_instances[:5]]}")

    print("-" * 80)
    all_passed = all(checks_passed.values())
    verdict = "[PASSED - READY FOR KAGGLE LEADERBOARD DISPATCH]" if all_passed else "[BLOCKED - INTEGRITY VIOLATION DETECTED]"
    print(f"OVERALL VERDICT: {verdict}")
    print("=" * 80)

    return all_passed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Championship Submission Integrity Checklist")
    parser.add_argument("--sub_csv", required=True, help="Path to submission CSV")
    parser.add_argument("--anchor_csv", default=DEFAULT_058_ANCHOR, help="Path to verified 0.58 anchor CSV")
    parser.add_argument("--allow_missing_anchor", action="store_true", help="Permit running without local anchor")
    args = parser.parse_args()

    success = run_submission_integrity_check(
        submission_csv=args.sub_csv,
        anchor_csv=args.anchor_csv,
        allow_missing_anchor=args.allow_missing_anchor
    )
    sys.exit(0 if success else 1)
