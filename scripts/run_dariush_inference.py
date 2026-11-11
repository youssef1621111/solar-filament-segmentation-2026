#!/usr/bin/env python3
"""EXP-031: Dariush 10-model ensemble inference + consensus with EXP-024 anchor."""
import os, sys, glob, time, gc
import numpy as np
import torch
import segmentation_models_pytorch as smp
from PIL import Image
from scipy import ndimage
import pandas as pd
import cv2

# Paths
WEIGHTS_DIR = "/home/fathyyoussef162/dariush_weights"
TEST_DIR = "/home/fathyyoussef162/competition_data/test_images"
ANCHOR_CSV = "/home/fathyyoussef162/results/operation_058_breakthrough/predictions/submission_exp024_wsp_consensus_ensemble.csv"
OUTPUT_DIR = "/home/fathyyoussef162/results/exp031"
os.makedirs(OUTPUT_DIR, exist_ok=True)

H, W = 2048, 2048
QUIET_STEMS = {"20160513133714Th", "20180507140210Th"}

from scripts.utils_metrics import encode_leb128_rle as rle_encode, decode_leb128_rle as rle_decode


# ── 8-view TTA ──
def tta_predict(model, img_tensor):
    """8-view TTA: 4 rotations × 2 flips."""
    preds = []
    with torch.no_grad():
        for k in range(4):
            x = torch.rot90(img_tensor, k, [2, 3])
            logit = model(x)
            prob = torch.sigmoid(logit)
            prob = torch.rot90(prob, -k, [2, 3])
            preds.append(prob)
            # horizontal flip
            x_flip = torch.flip(x, [3])
            logit_flip = model(x_flip)
            prob_flip = torch.sigmoid(logit_flip)
            prob_flip = torch.flip(prob_flip, [3])
            prob_flip = torch.rot90(prob_flip, -k, [2, 3])
            preds.append(prob_flip)
    return torch.stack(preds).mean(dim=0)

# ── Load all 10 models one at a time, accumulate probabilities ──
def run_ensemble_inference():
    test_images = sorted(glob.glob(os.path.join(TEST_DIR, "*.jpeg")))
    stems = [os.path.splitext(os.path.basename(f))[0] for f in test_images]
    print(f"Found {len(test_images)} test images")
    
    # Initialize probability accumulator
    prob_maps = {}
    for stem in stems:
        prob_maps[stem] = np.zeros((H, W), dtype=np.float32)
    
    model_configs = []
    for fold in range(5):
        model_configs.append(("resnet34", f"unet_resnet34_fold{fold}.pt"))
        model_configs.append(("tu-tf_efficientnetv2_s", f"unet_tu-tf_efficientnetv2_s_fold{fold}.pt"))
    
    n_models = len(model_configs)
    
    for midx, (enc_name, weight_file) in enumerate(model_configs):
        t0 = time.time()
        print(f"\n[{midx+1}/{n_models}] Loading {weight_file}...")
        model = smp.Unet(encoder_name=enc_name, encoder_weights=None, in_channels=1, classes=1)
        state = torch.load(os.path.join(WEIGHTS_DIR, weight_file), map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        model.eval()
        del state
        gc.collect()
        
        for iidx, (img_path, stem) in enumerate(zip(test_images, stems)):
            img = np.array(Image.open(img_path).convert("L"), dtype=np.float32) / 255.0
            
            # Normalize: mean/std inside 950px circle around center
            cy, cx = H // 2, W // 2
            yy, xx = np.ogrid[:H, :W]
            disk_mask = ((yy - cy)**2 + (xx - cx)**2) <= 950**2
            mean_val = img[disk_mask].mean()
            std_val = img[disk_mask].std()
            if std_val < 1e-6:
                std_val = 1.0
            img_norm = (img - mean_val) / std_val
            
            tensor = torch.from_numpy(img_norm).unsqueeze(0).unsqueeze(0).float()
            
            # Run TTA
            prob = tta_predict(model, tensor)
            prob_np = prob.squeeze().numpy()
            prob_maps[stem] += prob_np
            
            if (iidx + 1) % 30 == 0:
                print(f"  Processed {iidx+1}/{len(test_images)} images...")
        
        del model
        gc.collect()
        elapsed = time.time() - t0
        print(f"  Model done in {elapsed:.1f}s")
    
    # Average over 10 models
    for stem in stems:
        prob_maps[stem] /= n_models
    
    return prob_maps, stems

def extract_instances(prob_map, thresh=0.35, min_peak=0.99, min_mean_p=0.85, min_area=200):
    """Apply Dariush's keep rule to extract filament instances."""
    binary = (prob_map > thresh).astype(np.uint8)
    labeled, n_components = ndimage.label(binary)
    
    instances = []
    for comp_id in range(1, n_components + 1):
        component_mask = (labeled == comp_id)
        area = component_mask.sum()
        if area < min_area:
            continue
        peak = prob_map[component_mask].max()
        if peak < min_peak:
            continue
        mean_p = prob_map[component_mask].mean()
        if mean_p < min_mean_p:
            continue
        instances.append(component_mask.astype(np.uint8))
    
    return instances

def compute_iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    if union == 0:
        return 0.0
    return intersection / union

def build_consensus_submission(prob_maps, stems):
    """Option A: Use Dariush ensemble as consensus validator for EXP-024 anchor."""
    print("\n=== Building Consensus Submission ===")
    
    # Load anchor
    anchor_df = pd.read_csv(ANCHOR_CSV)
    print(f"Anchor has {len(anchor_df)} rows")
    
    # Extract Dariush instances per stem
    dariush_instances = {}
    for stem in stems:
        if stem in QUIET_STEMS:
            continue
        instances = extract_instances(prob_maps[stem])
        dariush_instances[stem] = instances
        if len(instances) > 0:
            print(f"  {stem}: {len(instances)} Dariush instances")
    
    # For each anchor instance, check if Dariush also sees it
    kept_rows = []
    dropped_count = 0
    
    for _, row in anchor_df.iterrows():
        stem = row['id'].rsplit('_', 1)[0]
        
        # Always keep quiet-sun stems as empty
        if stem in QUIET_STEMS:
            continue
            
        anchor_mask = rle_decode(row['annotation'], (H, W))
        anchor_area = anchor_mask.sum()
        
        # Check against Dariush instances
        dar_insts = dariush_instances.get(stem, [])
        
        best_iou = 0.0
        for dar_mask in dar_insts:
            iou = compute_iou(anchor_mask, dar_mask)
            best_iou = max(best_iou, iou)
        
        # Also check against the full Dariush probability map
        prob = prob_maps.get(stem)
        if prob is not None:
            # What fraction of the anchor's pixels have high Dariush probability?
            if anchor_area > 0:
                overlap_prob = prob[anchor_mask > 0].mean()
            else:
                overlap_prob = 0.0
        else:
            overlap_prob = 0.0
        
        # Keep if: Dariush IoU > 0.30 OR Dariush mean prob on anchor pixels > 0.50
        # (relaxed threshold — we want to keep real filaments, only drop clear FPs)
        if best_iou > 0.30 or overlap_prob > 0.50:
            kept_rows.append(row)
        else:
            dropped_count += 1
            print(f"  DROPPED: {row['id']} (best_iou={best_iou:.3f}, overlap_prob={overlap_prob:.3f}, area={anchor_area})")
    
    print(f"\nKept {len(kept_rows)}/{len(anchor_df)} instances (dropped {dropped_count})")
    
    # Build submission DataFrame
    result_df = pd.DataFrame(kept_rows)
    
    # Also build pure Dariush submission for comparison
    dariush_rows = []
    inst_counter = {}
    for stem in stems:
        if stem in QUIET_STEMS:
            continue
        instances = dariush_instances.get(stem, [])
        for inst_mask in instances:
            count = inst_counter.get(stem, 0)
            inst_counter[stem] = count + 1
            rle = rle_encode(inst_mask)
            dariush_rows.append({
                'id': f"{stem}_{count}",
                'annotation': rle
            })
    
    dariush_df = pd.DataFrame(dariush_rows)
    
    return result_df, dariush_df

def verify_submission(df, label=""):
    """Zero-trust verification."""
    print(f"\n=== Verifying {label} ===")
    stems_seen = set()
    total_pixels = 0
    collision_map = {}
    
    for _, row in df.iterrows():
        stem = row['id'].rsplit('_', 1)[0]
        stems_seen.add(stem)
        mask = rle_decode(row['annotation'], (H, W))
        px = mask.sum()
        total_pixels += px
        
        if stem not in collision_map:
            collision_map[stem] = np.zeros((H, W), dtype=np.uint8)
        collision_map[stem] += mask
    
    max_overlap = max(cm.max() for cm in collision_map.values()) if collision_map else 0
    quiet_rows = sum(1 for _, r in df.iterrows() if r['id'].rsplit('_', 1)[0] in QUIET_STEMS)
    
    print(f"  Stems: {len(stems_seen)}")
    print(f"  Instances: {len(df)}")
    print(f"  Quiet-sun rows: {quiet_rows}")
    print(f"  Max pixel overlap: {max_overlap}")
    print(f"  Total foreground pixels: {total_pixels}")
    
    if quiet_rows > 0:
        print("  ❌ FAIL: quiet-sun rows present!")
    if max_overlap > 1:
        print("  ❌ FAIL: pixel collisions detected!")
    else:
        print("  ✅ PASS: no collisions")

def main():
    t_start = time.time()
    
    # Step 1: Run ensemble inference
    prob_maps, stems = run_ensemble_inference()
    
    # Save probability maps for later analysis
    np.savez_compressed(os.path.join(OUTPUT_DIR, "dariush_prob_maps.npz"),
                       **{s: prob_maps[s] for s in stems[:5]})  # Save first 5 for debugging
    
    # Step 2: Build consensus submission
    consensus_df, dariush_df = build_consensus_submission(prob_maps, stems)
    
    # Step 3: Save submissions
    consensus_path = os.path.join(OUTPUT_DIR, "submission_exp031_consensus.csv")
    dariush_path = os.path.join(OUTPUT_DIR, "submission_dariush_raw.csv")
    
    consensus_df.to_csv(consensus_path, index=False)
    dariush_df.to_csv(dariush_path, index=False)
    
    print(f"\nSaved consensus: {consensus_path} ({len(consensus_df)} rows)")
    print(f"Saved Dariush raw: {dariush_path} ({len(dariush_df)} rows)")
    
    # Step 4: Verify
    verify_submission(consensus_df, "EXP-031 Consensus")
    verify_submission(dariush_df, "Dariush Raw")
    
    # Step 5: Copy best to submission_output
    import shutil
    os.makedirs("/home/fathyyoussef162/submission_output", exist_ok=True)
    shutil.copy2(consensus_path, "/home/fathyyoussef162/submission_output/submission.csv")
    print(f"\n✅ Staged at submission_output/submission.csv")
    
    elapsed = time.time() - t_start
    print(f"\nTotal time: {elapsed/60:.1f} minutes")

if __name__ == "__main__":
    main()
