import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
import time
from scripts.utils_metrics import decode_leb128_counts, decode_leb128_rle, compute_mask_bbox

# Read first 5 rows of anchor
df = pd.read_csv("submission_original_055_anchor.csv")

def fast_rle_stats(counts, H=2048):
    curr = 0
    val = False
    fg_runs = []
    for c in counts:
        if val and c > 0:
            fg_runs.append((curr, curr + c))
        curr += c
        val = not val

    if not fg_runs:
        return 0, (0, 0, 0, 0), 1.0, 1.0

    area = sum(end - start for start, end in fg_runs)
    min_x = min(start // H for start, _ in fg_runs)
    max_x = max((end - 1) // H for _, end in fg_runs)
    
    y_min = H
    y_max = 0
    for start, end in fg_runs:
        s_col = start // H
        e_col = (end - 1) // H
        if s_col == e_col:
            y1 = start % H
            y2 = (end - 1) % H
            if y1 < y_min: y_min = y1
            if y2 > y_max: y_max = y2
        else:
            # Spans column
            # For start col: start % H to H-1
            # For end col: 0 to (end-1) % H
            # For intermediate cols: 0 to H-1
            y_min = 0
            y_max = H - 1
            break
            
    h = y_max - y_min + 1
    w = max_x - min_x + 1
    elong = max(h, w) / max(min(h, w), 1)
    bbox_area = h * w
    fill = area / bbox_area if bbox_area > 0 else 1.0
    return area, (y_min, y_max, min_x, max_x), elong, fill

print("Testing first 10 rows:")
for i in range(10):
    rle = df.iloc[i]["segmentation_rle"]
    counts = decode_leb128_counts(rle)
    t0 = time.time()
    area, bbox, elong, fill = fast_rle_stats(counts)
    t_fast = time.time() - t0
    
    t0 = time.time()
    mask = decode_leb128_rle(rle)
    true_bbox = compute_mask_bbox(mask)
    t_slow = time.time() - t0
    
    print(f"Row {i}: area={area}, bbox={bbox}, true_bbox={true_bbox}, match={bbox==true_bbox}")
    assert bbox == true_bbox, f"Mismatch: {bbox} vs {true_bbox}"

print("ALL 10 MATCHED EXACTLY!")
