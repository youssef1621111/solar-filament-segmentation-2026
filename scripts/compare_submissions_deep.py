import pandas as pd
import numpy as np
from pycocotools import mask as mask_util
from pathlib import Path
import os

submissions = {
    "Primary 0.58": "final_submission/submission_selected_primary_058.csv",
    "Secondary 0.58": "final_submission/submission_selected_secondary_058.csv",
    "EXP-046 De-Merged": "final_submission/submission_exp046_pure_demerged.csv",
    "EXP-047 Dilated": "final_submission/submission_exp047_demerged_dilated.csv",
    "EXP-048 Skel-Adaptive": "final_submission/submission_exp048_skeleton_adaptive_sq.csv",
    "EXP-049 Unified SOTA": "final_submission/submission_exp049_unified_sota.csv",
}

print(f"{'Submission':<25} | {'Rows':<6} | {'Stems':<6} | {'P25':<6} | {'P50':<6} | {'P75':<6} | {'Mean Area':<10} | {'Growth vs Pri':<14}")
print("-" * 95)

primary_df = pd.read_csv(submissions["Primary 0.58"])
# Calculate areas from counts
def get_areas(df):
    areas = []
    for rle in df["segmentation_rle"]:
        # Decode first 2 counts or sum
        m = mask_util.decode({"counts": str(rle).encode("ascii"), "size": [2048, 2048]})
        areas.append(int(m.sum()))
    return np.array(areas)

# Fast approximation of areas using pycocotools area
def get_fast_areas(df):
    rles = [{"counts": str(r).encode("ascii"), "size": [2048, 2048]} for r in df["segmentation_rle"]]
    batches = [mask_util.area(rles[i:i+100]) for i in range(0, len(rles), 100)]
    return np.concatenate(batches)

pri_areas = get_fast_areas(primary_df)
pri_total = pri_areas.sum()

for name, path in submissions.items():
    if not os.path.exists(path):
        continue
    df = pd.read_csv(path)
    stems = df["filament_id"].apply(lambda x: x.split("_")[0]).nunique()
    areas = get_fast_areas(df)
    p25, p50, p75 = np.percentile(areas, [25, 50, 75])
    mean_a = areas.mean()
    total_a = areas.sum()
    growth = f"+{(total_a - pri_total)/pri_total*100:.2f}%" if name != "Primary 0.58" else "0.00% (Base)"
    print(f"{name:<25} | {len(df):<6} | {stems:<6} | {p25:<6.0f} | {p50:<6.0f} | {p75:<6.0f} | {mean_a:<10.1f} | {growth:<14}")
