"""
Build Winning Submission: Prune Anchor FPs + Add Gate-Passed Solos
Strategy:
  1. Load anchor (1344 rows) and consensus (2019 rows)
  2. For each anchor instance, compute shape features (area, elongation, circularity)
  3. For each anchor instance, check corroboration against consensus (IoU >= 0.10)
  4. Prune anchor instances that are BOTH uncorroborated AND have bad shape
  5. Identify consensus solos (not overlapping any surviving anchor) that pass gate
  6. Merge pruned anchor + accepted solos with anchor-first pixel priority
  7. Write submission CSV with LEB128 RLE encoding
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


def compute_shape_features(mask):
    """Compute area, elongation, circularity from binary mask."""
    area = int(np.count_nonzero(mask))
    if area < 5:
        return area, 1.0, 1.0
    ys, xs = np.where(mask > 0)
    # Elongation from bounding box
    h = ys.max() - ys.min() + 1
    w = xs.max() - xs.min() + 1
    elongation = max(h, w) / max(min(h, w), 1)
    # Circularity = 4*pi*area / perimeter^2
    # Approximate perimeter by counting boundary pixels
    from scipy import ndimage
    eroded = ndimage.binary_erosion(mask)
    perimeter = np.count_nonzero(mask & ~eroded)
    if perimeter > 0:
        circularity = (4 * np.pi * area) / (perimeter ** 2)
    else:
        circularity = 1.0
    return area, elongation, circularity


def fast_area_from_rle(rle_str):
    """Get pixel area from RLE without full mask decode."""
    counts = decode_leb128_counts(rle_str)
    # Foreground is at odd indices (0-indexed)
    return sum(counts[i] for i in range(1, len(counts), 2))


def main():
    ANCHOR_CSV = "submission_original_055_anchor.csv"
    CONSENSUS_CSV = "experiments/consensus_v3/submission.csv"
    OUTPUT_CSV = "experiments/sub_pruned_merged.csv"
    QUIET_SUN_STEMS = {"20160513133714Th", "20180507140210Th"}

    print("=" * 80)
    print("BUILDING WINNING SUBMISSION: PRUNE + MERGE")
    print("=" * 80)

    # Load data
    anchor_df = pd.read_csv(ANCHOR_CSV)
    consensus_df = pd.read_csv(CONSENSUS_CSV)

    anchor_df['stem'] = anchor_df['filament_id'].str.rsplit('_', n=1).str[0]
    anchor_df['inst'] = anchor_df['filament_id'].str.rsplit('_', n=1).str[1].astype(int)
    consensus_df['stem'] = consensus_df['filament_id'].str.rsplit('_', n=1).str[0]
    consensus_df['inst'] = consensus_df['filament_id'].str.rsplit('_', n=1).str[1].astype(int)

    # Separate quiet sun (preserve as-is)
    quiet_df = anchor_df[anchor_df['stem'].isin(QUIET_SUN_STEMS)].copy()
    active_anchor = anchor_df[~anchor_df['stem'].isin(QUIET_SUN_STEMS)].copy()

    print(f"Anchor total: {len(anchor_df)} rows ({len(quiet_df)} quiet sun, {len(active_anchor)} active)")
    print(f"Consensus total: {len(consensus_df)} rows")

    # Group by stem
    anchor_by_stem = defaultdict(list)
    for _, row in active_anchor.iterrows():
        anchor_by_stem[row['stem']].append(row)

    consensus_by_stem = defaultdict(list)
    for _, row in consensus_df.iterrows():
        consensus_by_stem[row['stem']].append(row)

    # Process each stem
    all_stems = sorted(set(list(anchor_by_stem.keys()) + list(consensus_by_stem.keys())))
    
    kept_anchors = []
    pruned_anchors = []
    accepted_solos = []
    
    total_stems = len(all_stems)
    
    for si, stem in enumerate(all_stems):
        if (si + 1) % 20 == 0 or si == 0:
            print(f"  Processing stem {si+1}/{total_stems}: {stem}")
        
        a_rows = anchor_by_stem.get(stem, [])
        c_rows = consensus_by_stem.get(stem, [])
        
        # Decode anchor masks
        a_masks = []
        a_bboxes = []
        a_features = []
        for r in a_rows:
            area = fast_area_from_rle(r['segmentation_rle'])
            if area < 10:
                # Tiny/empty mask — keep it (it won't hurt, it's just noise)
                a_masks.append(None)
                a_bboxes.append(None)
                a_features.append((area, 1.0, 1.0))
            else:
                mask = decode_leb128_rle(r['segmentation_rle'])
                bbox = compute_mask_bbox(mask)
                a_masks.append(mask)
                a_bboxes.append(bbox)
                a_features.append(compute_shape_features(mask))
        
        # Decode consensus masks
        c_masks = []
        c_bboxes = []
        for r in c_rows:
            area = fast_area_from_rle(r['segmentation_rle'])
            if area < 10:
                c_masks.append(None)
                c_bboxes.append(None)
            else:
                mask = decode_leb128_rle(r['segmentation_rle'])
                bbox = compute_mask_bbox(mask)
                c_masks.append(mask)
                c_bboxes.append(bbox)
        
        # Step 1: For each anchor, check corroboration against consensus
        anchor_corroborated = [False] * len(a_rows)
        for ai in range(len(a_rows)):
            if a_masks[ai] is None:
                anchor_corroborated[ai] = True  # Don't prune empty masks
                continue
            for ci in range(len(c_rows)):
                if c_masks[ci] is None:
                    continue
                iou = compute_iou(a_masks[ai], c_masks[ci], a_bboxes[ai], c_bboxes[ci])
                if iou >= 0.10:
                    anchor_corroborated[ai] = True
                    break
        
        # Step 2: Decide which anchors to keep vs prune
        for ai in range(len(a_rows)):
            area, elong, circ = a_features[ai]
            is_bad_shape = (area < 300 and (elong < 1.8 or circ > 0.45))
            is_uncorroborated = not anchor_corroborated[ai]
            
            # Prune if BOTH bad shape AND uncorroborated
            # Also prune if very tiny AND uncorroborated
            if is_uncorroborated and (is_bad_shape or area < 100):
                pruned_anchors.append({
                    'stem': stem,
                    'filament_id': a_rows[ai]['filament_id'],
                    'area': area,
                    'elongation': elong,
                    'circularity': circ,
                    'reason': 'uncorroborated+bad_shape' if is_bad_shape else 'uncorroborated+tiny'
                })
            else:
                kept_anchors.append(a_rows[ai])
        
        # Step 3: Find consensus solos (not overlapping any KEPT anchor)
        kept_indices = [ai for ai in range(len(a_rows)) 
                       if a_rows[ai]['filament_id'] in {k['filament_id'] for k in kept_anchors if k['stem'] == stem}]
        
        for ci in range(len(c_rows)):
            if c_masks[ci] is None:
                continue
            c_area = fast_area_from_rle(c_rows[ci]['segmentation_rle'])
            if c_area < 80:
                continue  # Skip tiny consensus instances
            
            # Check if this consensus instance overlaps any kept anchor
            is_solo = True
            for ai in kept_indices:
                if a_masks[ai] is None:
                    continue
                iou = compute_iou(c_masks[ci], a_masks[ai], c_bboxes[ci], a_bboxes[ai])
                if iou >= 0.05:  # Very low threshold — if any overlap, not solo
                    is_solo = False
                    break
            
            if is_solo:
                # Apply acceptance gate: require elongation > 2.0 and area > 200
                c_feat = compute_shape_features(c_masks[ci])
                c_elong = c_feat[1]
                c_circ = c_feat[2]
                # Filament-like: elongated, not circular
                if c_elong >= 2.0 and c_circ < 0.40 and c_area >= 200:
                    accepted_solos.append({
                        'stem': stem,
                        'mask': c_masks[ci],
                        'area': c_area,
                        'elongation': c_elong,
                        'circularity': c_circ,
                    })
    
    print(f"\n{'=' * 80}")
    print(f"PRUNING & MERGING RESULTS")
    print(f"{'=' * 80}")
    print(f"Anchors kept: {len(kept_anchors)}")
    print(f"Anchors pruned: {len(pruned_anchors)}")
    print(f"Consensus solos accepted: {len(accepted_solos)}")
    
    if pruned_anchors:
        prune_df = pd.DataFrame(pruned_anchors)
        print(f"\nPruned anchor breakdown:")
        print(prune_df['reason'].value_counts().to_string())
        print(f"Pruned area stats: mean={prune_df['area'].mean():.0f}, median={prune_df['area'].median():.0f}")
    
    if accepted_solos:
        solo_areas = [s['area'] for s in accepted_solos]
        solo_elongs = [s['elongation'] for s in accepted_solos]
        print(f"\nAccepted solo stats: areas={np.mean(solo_areas):.0f} avg, "
              f"elongation={np.mean(solo_elongs):.1f} avg")
    
    # Step 4: Build final submission
    # Start with kept anchors (preserve original RLE)
    output_rows = []
    
    # Add quiet sun stems first
    for _, row in quiet_df.iterrows():
        output_rows.append({
            'filament_id': row['filament_id'],
            'segmentation_rle': row['segmentation_rle']
        })
    
    # Add kept anchors (preserve original encoding)
    for row in kept_anchors:
        output_rows.append({
            'filament_id': row['filament_id'],
            'segmentation_rle': row['segmentation_rle']
        })
    
    # Add accepted solos — need to stamp onto canvas without overlapping kept anchors
    # Group by stem, assign new instance IDs
    solo_by_stem = defaultdict(list)
    for s in accepted_solos:
        solo_by_stem[s['stem']].append(s)
    
    for stem in sorted(solo_by_stem.keys()):
        # Find max existing instance ID for this stem
        existing_ids = [r['inst'] for r in kept_anchors if r['stem'] == stem]
        quiet_ids = []
        for _, qr in quiet_df.iterrows():
            qs = qr['filament_id'].rsplit('_', 1)[0]
            if qs == stem:
                quiet_ids.append(int(qr['filament_id'].rsplit('_', 1)[1]))
        all_ids = existing_ids + quiet_ids
        max_id = max(all_ids) if all_ids else 0
        
        # Decode all kept anchor masks for this stem to build occupied canvas
        kept_stem_masks = []
        for row in kept_anchors:
            if row['stem'] == stem:
                area = fast_area_from_rle(row['segmentation_rle'])
                if area > 0:
                    kept_stem_masks.append(decode_leb128_rle(row['segmentation_rle']))
        
        occupied = np.zeros((2048, 2048), dtype=bool)
        for m in kept_stem_masks:
            occupied |= m
        
        for si, solo in enumerate(solo_by_stem[stem]):
            new_id = max_id + 1 + si
            fid = f"{stem}_{new_id}"
            
            # Remove any pixels that overlap with occupied canvas
            solo_mask = solo['mask'].copy()
            solo_mask[occupied] = False
            
            # Check if still has enough pixels after deconfliction
            remaining = np.count_nonzero(solo_mask)
            if remaining < 50:
                print(f"  [SKIP] Solo {fid}: only {remaining} pixels after deconfliction")
                continue
            
            rle = encode_leb128_rle(solo_mask)
            output_rows.append({
                'filament_id': fid,
                'segmentation_rle': rle
            })
            
            # Update occupied canvas
            occupied |= solo_mask
    
    # Build output DataFrame
    out_df = pd.DataFrame(output_rows)
    
    # Sort by filament_id for consistency
    out_df = out_df.sort_values('filament_id').reset_index(drop=True)
    
    # Verify stem coverage
    out_df['stem_check'] = out_df['filament_id'].str.rsplit('_', n=1).str[0]
    stems_present = out_df['stem_check'].nunique()
    out_df.drop('stem_check', axis=1, inplace=True)
    
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    out_df.to_csv(OUTPUT_CSV, index=False)
    
    print(f"\nFinal submission: {OUTPUT_CSV}")
    print(f"  Total rows: {len(out_df)}")
    print(f"  Stems: {stems_present}")
    print(f"  Expected delta: pruned {len(pruned_anchors)} FPs, added {len(accepted_solos)} solos")
    
    # Validate RLE sums for a sample
    print("\nValidating RLE sums (sample of 10)...")
    import random
    sample = random.sample(range(len(out_df)), min(10, len(out_df)))
    for idx in sample:
        row = out_df.iloc[idx]
        counts = decode_leb128_counts(row['segmentation_rle'])
        total = sum(counts)
        status = "OK" if total == 4194304 else f"FAIL ({total})"
        print(f"  {row['filament_id']}: {status}")
    
    return OUTPUT_CSV


if __name__ == "__main__":
    output = main()
