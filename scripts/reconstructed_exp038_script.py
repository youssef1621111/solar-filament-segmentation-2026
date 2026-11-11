python3 -c "
import pandas as pd
from pycocotools import mask as mask_util
import numpy as np

sub_anchor = pd.read_csv('/home/fathyyoussef162/results/operation_058_breakthrough/predictions/submission_exp031_calibrated_1300.csv')
sub_exp037 = pd.read_csv('/home/fathyyoussef162/results/operation_058_breakthrough/predictions/submission_exp037_soft_consensus_1240.csv')

stems_anchor = sub_anchor['filament_id'].apply(lambda x: x.split('_')[0])
stems_exp037 = sub_exp037['filament_id'].apply(lambda x: x.split('_')[0])

unique_stems = stems_anchor.unique()
print(f'Processing {len(unique_stems)} stems...')

blended_rows = []
collisions_resolved = 0
matched_instances = 0

for s_idx, stem in enumerate(unique_stems):
    anc_rows = sub_anchor[stems_anchor == stem]
    exp_rows = sub_exp037[stems_exp037 == stem]
    
    anc_masks = [mask_util.decode({'counts': r.encode('ascii'), 'size': [2048, 2048]}) > 0 for r in anc_rows['segmentation_rle']]
    anc_ids = anc_rows['filament_id'].tolist()
    
    exp_masks = [mask_util.decode({'counts': r.encode('ascii'), 'size': [2048, 2048]}) > 0 for r in exp_rows['segmentation_rle']]
    
    occupied = np.zeros((2048, 2048), dtype=bool)
    stem_blended = []
    
    for a_id, a_m in zip(anc_ids, anc_masks):
        # Find best matching exp mask
        best_e = None
        best_iou = 0.0
        for e_m in exp_masks:
            inter = (a_m & e_m).sum()
            if inter > 0:
                iou = inter / (a_m | e_m).sum()
                if iou > best_iou:
                    best_iou = iou
                    best_e = e_m
                    
        # If high-confidence match, blend anchor with consensus ribbon
        if best_iou >= 0.50 and best_e is not None:
            # Anchor-Guided Hysteresis: union anchor with consensus ribbon
            blended_m = a_m | best_e
            matched_instances += 1
        else:
            blended_m = a_m
            
        stem_blended.append((a_id, blended_m))
        
    # Enforce strict zero collision across the stem canvas
    for a_id, b_m in stem_blended:
        clean_m = b_m & (~occupied)
        if clean_m.sum() < 20: # fallback to original anchor if overlap shrank it
            # find original anchor
            orig_m = anc_masks[anc_ids.index(a_id)]
            clean_m = orig_m & (~occupied)
        occupied |= clean_m
        
        rle = mask_util.encode(np.asfortranarray(clean_m.astype(np.uint8)))
        rle_str = rle['counts'].decode('ascii') if isinstance(rle['counts'], bytes) else str(rle['counts'])
        blended_rows.append({'filament_id': a_id, 'segmentation_rle': rle_str})

out_df = pd.DataFrame(blended_rows)
out_path = '/tmp/submission_exp038_test.csv'
out_df.to_csv(out_path, index=False)
print(f'Done! Total instances: {len(out_df)}, Matched & Blended: {matched_instances}/{len(sub_anchor)}')
"