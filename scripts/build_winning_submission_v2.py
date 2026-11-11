"""
Build Winning Submission v2: Optimized - Prune Anchor FPs + Add Solos
Uses fast bounding-box-based shape features (no scipy).
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from collections import defaultdict
from scripts.utils_metrics import (
    decode_leb128_rle, encode_leb128_rle, decode_leb128_counts,
    compute_mask_bbox, compute_iou
)
import time


def fast_shape(mask):
    """Fast shape features: area, elongation, circularity (bbox-based, no scipy)."""
    area = int(np.count_nonzero(mask))
    if area < 5:
        return area, 1.0, 1.0
    ys, xs = np.where(mask > 0)
    h = int(ys.max() - ys.min()) + 1
    w = int(xs.max() - xs.min()) + 1
    elongation = max(h, w) / max(min(h, w), 1)
    # Approximate circularity from bbox fill ratio
    bbox_area = h * w
    fill = area / bbox_area if bbox_area > 0 else 1.0
    # For a true circle, fill ~ pi/4 ~ 0.785; for thin filaments fill < 0.3
    # Map: circ = fill (higher fill = more circular)
    circularity = fill
    return area, elongation, circularity


def fast_area_from_rle(rle_str):
    """Get pixel area from RLE without full mask decode."""
    counts = decode_leb128_counts(rle_str)
    return sum(counts[i] for i in range(1, len(counts), 2))


def main():
    t0 = time.time()
    ANCHOR_CSV = "submission_original_055_anchor.csv"
    CONSENSUS_CSV = "experiments/consensus_v3/submission.csv"
    OUTPUT_CSV = "experiments/sub_pruned_merged.csv"
    QUIET_SUN_STEMS = {"20160513133714Th", "20180507140210Th"}

    # Pruning thresholds (conservative: only prune clearly bad + uncorroborated)
    PRUNE_AREA_HARD = 100       # Always prune uncorroborated below this
    PRUNE_AREA_SOFT = 300       # Prune uncorroborated below this IF also bad shape
    PRUNE_ELONG_MIN = 1.8       # Filaments should be elongated
    PRUNE_FILL_MAX = 0.50       # Filaments have low bbox fill (thin, not blobby)
    
    # Solo acceptance thresholds (strict: only accept clearly filament-like)
    SOLO_AREA_MIN = 200
    SOLO_ELONG_MIN = 2.0
    SOLO_FILL_MAX = 0.35
    CORR_IOU = 0.10

    print("=" * 80)
    print("BUILDING WINNING SUBMISSION v2 (OPTIMIZED)")
    print("=" * 80)

    anchor_df = pd.read_csv(ANCHOR_CSV)
    consensus_df = pd.read_csv(CONSENSUS_CSV)

    anchor_df['stem'] = anchor_df['filament_id'].str.rsplit('_', n=1).str[0]
    anchor_df['inst'] = anchor_df['filament_id'].str.rsplit('_', n=1).str[1].astype(int)
    consensus_df['stem'] = consensus_df['filament_id'].str.rsplit('_', n=1).str[0]
    consensus_df['inst'] = consensus_df['filament_id'].str.rsplit('_', n=1).str[1].astype(int)

    quiet_df = anchor_df[anchor_df['stem'].isin(QUIET_SUN_STEMS)].copy()
    active_anchor = anchor_df[~anchor_df['stem'].isin(QUIET_SUN_STEMS)].copy()

    print(f"Anchor: {len(anchor_df)} total ({len(quiet_df)} quiet sun, {len(active_anchor)} active)")
    print(f"Consensus: {len(consensus_df)} total")

    anchor_by_stem = defaultdict(list)
    for _, row in active_anchor.iterrows():
        anchor_by_stem[row['stem']].append(row.to_dict())

    consensus_by_stem = defaultdict(list)
    for _, row in consensus_df.iterrows():
        consensus_by_stem[row['stem']].append(row.to_dict())

    all_stems = sorted(set(list(anchor_by_stem.keys()) + list(consensus_by_stem.keys())))

    kept_rows = []  # (filament_id, segmentation_rle) tuples
    pruned_info = []
    solo_info = []
    
    for si, stem in enumerate(all_stems):
        if (si + 1) % 30 == 0:
            elapsed = time.time() - t0
            print(f"  [{elapsed:.0f}s] Processing stem {si+1}/{len(all_stems)}")
        
        a_rows = anchor_by_stem.get(stem, [])
        c_rows = consensus_by_stem.get(stem, [])
        
        # Fast pass: skip mask decode if no consensus for this stem
        if not c_rows:
            # Keep all anchors for stems with no consensus
            for r in a_rows:
                kept_rows.append((r['filament_id'], r['segmentation_rle']))
            continue
        
        # Decode all masks for this stem
        a_data = []  # (mask, bbox, area, elong, fill, row_dict)
        for r in a_rows:
            area = fast_area_from_rle(r['segmentation_rle'])
            if area < 5:
                a_data.append((None, None, area, 1.0, 1.0, r))
            else:
                mask = decode_leb128_rle(r['segmentation_rle'])
                bbox = compute_mask_bbox(mask)
                _, elong, fill = fast_shape(mask)
                a_data.append((mask, bbox, area, elong, fill, r))
        
        c_data = []  # (mask, bbox, area, elong, fill, row_dict)
        for r in c_rows:
            area = fast_area_from_rle(r['segmentation_rle'])
            if area < 5:
                c_data.append((None, None, area, 1.0, 1.0, r))
            else:
                mask = decode_leb128_rle(r['segmentation_rle'])
                bbox = compute_mask_bbox(mask)
                _, elong, fill = fast_shape(mask)
                c_data.append((mask, bbox, area, elong, fill, r))
        
        # Check anchor corroboration
        anchor_keep = []
        for ai, (a_mask, a_bbox, a_area, a_elong, a_fill, a_row) in enumerate(a_data):
            if a_mask is None:
                anchor_keep.append(ai)
                kept_rows.append((a_row['filament_id'], a_row['segmentation_rle']))
                continue
            
            # Check corroboration
            corroborated = False
            for ci, (c_mask, c_bbox, c_area, _, _, _) in enumerate(c_data):
                if c_mask is None:
                    continue
                iou = compute_iou(a_mask, c_mask, a_bbox, c_bbox)
                if iou >= CORR_IOU:
                    corroborated = True
                    break
            
            # Pruning decision
            is_bad_shape = (a_area < PRUNE_AREA_SOFT and (a_elong < PRUNE_ELONG_MIN or a_fill > PRUNE_FILL_MAX))
            is_tiny = a_area < PRUNE_AREA_HARD
            
            if not corroborated and (is_bad_shape or is_tiny):
                pruned_info.append({
                    'fid': a_row['filament_id'], 'area': a_area,
                    'elong': a_elong, 'fill': a_fill,
                    'reason': 'bad_shape' if is_bad_shape else 'tiny'
                })
            else:
                anchor_keep.append(ai)
                kept_rows.append((a_row['filament_id'], a_row['segmentation_rle']))
        
        # Find consensus solos
        for ci, (c_mask, c_bbox, c_area, c_elong, c_fill, c_row) in enumerate(c_data):
            if c_mask is None or c_area < SOLO_AREA_MIN:
                continue
            
            # Check overlap with ANY kept anchor
            is_solo = True
            for ai in anchor_keep:
                a_mask_i, a_bbox_i = a_data[ai][0], a_data[ai][1]
                if a_mask_i is None:
                    continue
                iou = compute_iou(c_mask, a_mask_i, c_bbox, a_bbox_i)
                if iou >= 0.05:
                    is_solo = False
                    break
            
            if is_solo and c_elong >= SOLO_ELONG_MIN and c_fill < SOLO_FILL_MAX:
                solo_info.append({
                    'stem': stem, 'area': c_area, 'elong': c_elong, 'fill': c_fill,
                    'mask': c_mask
                })
    
    elapsed = time.time() - t0
    print(f"\n[{elapsed:.0f}s] Processing complete.")
    print(f"  Anchors kept: {len(kept_rows)}")
    print(f"  Anchors pruned: {len(pruned_info)}")
    print(f"  Consensus solos accepted: {len(solo_info)}")
    
    if pruned_info:
        print(f"\nPruned details:")
        for p in pruned_info[:10]:
            print(f"  {p['fid']}: area={p['area']}, elong={p['elong']:.2f}, fill={p['fill']:.2f} ({p['reason']})")
        if len(pruned_info) > 10:
            print(f"  ... and {len(pruned_info) - 10} more")
    
    # Build final submission
    output_rows = []
    
    # Add quiet sun
    for _, row in quiet_df.iterrows():
        output_rows.append((row['filament_id'], row['segmentation_rle']))
    
    # Add kept anchors
    output_rows.extend(kept_rows)
    
    # Add solos with new instance IDs and pixel deconfliction
    solo_by_stem = defaultdict(list)
    for s in solo_info:
        solo_by_stem[s['stem']].append(s)
    
    for stem in sorted(solo_by_stem.keys()):
        # Find max existing instance ID
        existing = [int(fid.rsplit('_', 1)[1]) for fid, _ in output_rows 
                    if fid.rsplit('_', 1)[0] == stem]
        max_id = max(existing) if existing else 0
        
        # Build occupied canvas from kept anchors
        occupied = np.zeros((2048, 2048), dtype=bool)
        for fid, rle in output_rows:
            if fid.rsplit('_', 1)[0] == stem:
                area = fast_area_from_rle(rle)
                if area > 0:
                    occupied |= decode_leb128_rle(rle)
        
        for i, solo in enumerate(solo_by_stem[stem]):
            new_id = max_id + 1 + i
            fid = f"{stem}_{new_id}"
            mask = solo['mask'].copy()
            mask[occupied] = False
            remaining = np.count_nonzero(mask)
            if remaining < 50:
                print(f"  [SKIP] {fid}: {remaining}px after deconfliction")
                continue
            rle = encode_leb128_rle(mask)
            output_rows.append((fid, rle))
            occupied |= mask
            print(f"  [ADD] {fid}: {remaining}px, elong={solo['elong']:.1f}")
    
    # Write output
    out_df = pd.DataFrame(output_rows, columns=['filament_id', 'segmentation_rle'])
    out_df = out_df.sort_values('filament_id').reset_index(drop=True)
    
    stems_present = out_df['filament_id'].str.rsplit('_', n=1).str[0].nunique()
    
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    out_df.to_csv(OUTPUT_CSV, index=False)
    
    elapsed = time.time() - t0
    print(f"\n{'=' * 80}")
    print(f"FINAL SUBMISSION: {OUTPUT_CSV}")
    print(f"  Rows: {len(out_df)}, Stems: {stems_present}")
    print(f"  Pruned: {len(pruned_info)} FPs, Added: {len(solo_info)} solos")
    print(f"  Time: {elapsed:.0f}s")
    print(f"{'=' * 80}")
    
    # Quick RLE validation
    print("\nRLE validation (first 5 + last 5)...")
    for idx in list(range(min(5, len(out_df)))) + list(range(max(0, len(out_df)-5), len(out_df))):
        row = out_df.iloc[idx]
        counts = decode_leb128_counts(row['segmentation_rle'])
        total = sum(counts)
        ok = "OK" if total == 4194304 else f"FAIL({total})"
        print(f"  {row['filament_id']}: {ok}")


if __name__ == "__main__":
    main()
