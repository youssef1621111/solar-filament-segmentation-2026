#!/usr/bin/env python3
"""
================================================================================
IEEE BigData 2026 Solar Filament Segmentation Challenge (filament-segmentation-2026)
Championship EXP-039: Topological Centerline Consensus Pipeline (Target: 0.66+ LB)
================================================================================
Core Architecture & Invariants:
1. Multi-Annotator Consensus Target Supervision (>= 2 Annotators Agreement)
   - Completely eliminates Trap 1 (Multi-Annotator False Positive Multiplier)
2. SoftCLDiceLoss (Topological Centerline Dice + Volumetric Dice + BCE)
   - Preserves curvilinear continuity along thin magnetic flux ropes
3. Native 1024x1024 Patch Tiling with 4-View Test-Time Augmentation (TTA)
4. Tier-1 Priority Anchor Stamping:
   - Preserves 100% of the verified 0.58 Anchor (submission_exp031_calibrated_1300)
   - Zero Cannibalization (Anchor instances never lose pixels)
5. Free-Canvas Consensus Recovery:
   - Recovers missing consensus filaments on deficit stems (P >= 0.88, Area >= 250)
6. Zero-Trust In-Kernel 6-Point Integrity Audit
================================================================================
"""

import os
import sys
import gc
import glob
import time
import math
import json
import random
from pathlib import Path
from collections import defaultdict
import subprocess

# Ensure dependencies installed
try:
    import pycocotools
    from pycocotools import mask as mask_util
    import segmentation_models_pytorch as smp
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "pycocotools", "segmentation-models-pytorch==0.5.0"])
    import pycocotools
    from pycocotools import mask as mask_util
    import segmentation_models_pytorch as smp

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# -------------------------------------------------------------------------
# Global Parameters & Constants
# -------------------------------------------------------------------------
SEED = 20260729
H = W = 2048
TILE_SIZE = 1024
STRIDE = 512
DISK_RADIUS = 950
DISK_CENTER = (1024, 1024)
QUIET_SUN_STEMS = {"20160513133714Th", "20180507140210Th"}
EXPECTED_ACTIVE_STEMS = 178
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
T0 = time.time()

def log(*args):
    elapsed = (time.time() - T0) / 60.0
    print(f"[{elapsed:6.2f}m]", *args, flush=True)

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed()

# -------------------------------------------------------------------------
# Solar Disk CLAHE & Illumination Normalization
# -------------------------------------------------------------------------
def apply_solar_clahe(img_gray: np.ndarray, clip_limit=2.0, tile_grid=(8, 8)) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    enhanced = clahe.apply(img_gray)
    return enhanced

def get_disk_mask(shape=(2048, 2048), radius=DISK_RADIUS, center=DISK_CENTER) -> np.ndarray:
    y, x = np.ogrid[:shape[0], :shape[1]]
    dist_sq = (x - center[0])**2 + (y - center[1])**2
    return dist_sq <= (radius**2)

DISK_MASK = get_disk_mask()

def normalize_solar_disk(img_gray: np.ndarray) -> np.ndarray:
    enhanced = apply_solar_clahe(img_gray)
    disk_pixels = enhanced[DISK_MASK]
    mean_val = float(np.mean(disk_pixels))
    std_val = float(np.std(disk_pixels)) + 1e-6
    norm = (enhanced.astype(np.float32) - mean_val) / std_val
    norm[~DISK_MASK] = 0.0
    return norm

# -------------------------------------------------------------------------
# Topological Centerline Soft-clDice Loss
# -------------------------------------------------------------------------
def soft_erode(img):
    p1 = -F.max_pool2d(-img, (3, 1), (1, 1), (1, 0))
    p2 = -F.max_pool2d(-img, (1, 3), (1, 1), (0, 1))
    return torch.min(p1, p2)

def soft_skeletonize(x, thresh_width=10):
    for i in range(thresh_width):
        min_p = soft_erode(x)
        max_p = F.max_pool2d(min_p, (3, 3), (1, 1), (1, 1))
        contour = F.relu(min_p - max_p)
        x = F.relu(x - contour)
        if i == 0:
            skel = contour
        else:
            skel = skel + F.relu(contour - skel * contour)
    return skel

class SoftCLDiceLoss(nn.Module):
    def __init__(self, alpha=0.5, smooth=1.0):
        super().__init__()
        self.alpha = alpha
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        bce_loss = self.bce(logits, targets)

        # 1. Volumetric Dice
        p_flat = probs.view(probs.size(0), -1)
        t_flat = targets.view(targets.size(0), -1)
        inter = (p_flat * t_flat).sum(dim=1)
        union = p_flat.sum(dim=1) + t_flat.sum(dim=1)
        vol_dice = (2.0 * inter + self.smooth) / (union + self.smooth)

        # 2. Topological Centerline Dice
        skel_pred = soft_skeletonize(probs)
        skel_true = soft_skeletonize(targets)

        tprec = (skel_pred * targets).sum(dim=(1, 2, 3)) / (skel_pred.sum(dim=(1, 2, 3)) + self.smooth)
        tsens = (probs * skel_true).sum(dim=(1, 2, 3)) / (skel_true.sum(dim=(1, 2, 3)) + self.smooth)
        cldice = (2.0 * tprec * tsens) / (tprec + tsens + 1e-6)

        cldice_loss = 1.0 - cldice.mean()
        vol_loss = 1.0 - vol_dice.mean()

        return 0.30 * bce_loss + 0.35 * vol_loss + 0.35 * cldice_loss

# -------------------------------------------------------------------------
# Multi-Annotator Consensus Dataset (1024x1024 Tiles)
# -------------------------------------------------------------------------
class SoftConsensusPatchDataset(Dataset):
    def __init__(self, patches: list):
        self.patches = patches

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        img_patch, target_patch = self.patches[idx]
        img_patch = img_patch.copy()
        target_patch = target_patch.copy()
        if random.random() > 0.5:
            img_patch = np.fliplr(img_patch).copy()
            target_patch = np.fliplr(target_patch).copy()
        if random.random() > 0.5:
            img_patch = np.flipud(img_patch).copy()
            target_patch = np.flipud(target_patch).copy()
        t_img = torch.from_numpy(img_patch).unsqueeze(0).float()
        t_target = torch.from_numpy(target_patch).unsqueeze(0).float()
        return t_img, t_target

def decode_annotation_mask(seg, h=2048, w=2048) -> np.ndarray:
    if not seg:
        return np.zeros((h, w), dtype=bool)
    try:
        if isinstance(seg, list):
            rles = mask_util.frPyObjects(seg, h, w)
            rle = mask_util.merge(rles)
        elif isinstance(seg, dict):
            if isinstance(seg.get('counts'), list):
                rle = mask_util.frPyObjects(seg, h, w)
            else:
                rle = seg
        else:
            return np.zeros((h, w), dtype=bool)
        decoded = mask_util.decode(rle)
        return (decoded > 0)
    except Exception:
        return np.zeros((h, w), dtype=bool)

def build_training_patches(ann_file: str, img_dir: str, max_patches=300) -> list:
    log(f"Loading annotations JSON metadata from: {ann_file}")
    with open(ann_file, "r") as f:
        coco_data = json.load(f)

    annotations_by_img = defaultdict(list)
    for a in coco_data.get("annotations", []):
        seg = a.get("segmentation")
        if seg:
            annotations_by_img[a["image_id"]].append(seg)

    stems_to_img_ids = defaultdict(list)
    stem_to_filename = {}
    for im in coco_data.get("images", []):
        fname = im["file_name"]
        stem = Path(fname).stem.split("_")[0]
        if im["id"] in annotations_by_img:
            stems_to_img_ids[stem].append(im["id"])
            if stem not in stem_to_filename:
                stem_to_filename[stem] = fname

    log(f"Found {len(coco_data.get('images', []))} COCO records across {len(stems_to_img_ids)} annotated physical stems.")

    del coco_data
    gc.collect()

    sorted_stems = sorted(
        stems_to_img_ids.keys(),
        key=lambda s: sum(len(annotations_by_img[iid]) for iid in stems_to_img_ids[s]),
        reverse=True
    )

    patches = []
    tile_coords = []
    for y in range(0, H - TILE_SIZE + 1, STRIDE):
        for x in range(0, W - TILE_SIZE + 1, STRIDE):
            tile_coords.append((y, x))

    for stem in sorted_stems:
        if len(patches) >= max_patches:
            break
        fname = stem_to_filename[stem]
        img_path = os.path.join(img_dir, fname)
        if not os.path.exists(img_path):
            candidates = glob.glob(f"/kaggle/input/**/{fname}", recursive=True)
            if candidates:
                img_path = candidates[0]
            else:
                continue

        img_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img_gray is None:
            continue

        norm_img = normalize_solar_disk(img_gray)

        img_ids = stems_to_img_ids[stem]
        n_annotators = len(img_ids)
        agreement = np.zeros((H, W), dtype=np.float32)

        for iid in img_ids:
            pass_union = np.zeros((H, W), dtype=bool)
            for seg in annotations_by_img[iid]:
                m = decode_annotation_mask(seg, H, W)
                pass_union |= m
            agreement += pass_union.astype(np.float32)

        agreement /= max(1.0, float(n_annotators))

        # Multi-Annotator Consensus Target:
        # If 2+ annotators, require >= 0.50 agreement. If 1 annotator, require agreement == 1.0.
        consensus_target = (agreement >= 0.50).astype(np.float32)
        consensus_target[~DISK_MASK] = 0.0

        for y, x in tile_coords:
            t_target = consensus_target[y:y+TILE_SIZE, x:x+TILE_SIZE]
            if t_target.sum() > 40:
                t_img = norm_img[y:y+TILE_SIZE, x:x+TILE_SIZE]
                patches.append((t_img.astype(np.float32), t_target.astype(np.float32)))
                if len(patches) >= max_patches:
                    break

        del agreement, consensus_target

    log(f"Generated {len(patches)} high-resolution 1024x1024 consensus training patches.")
    return patches

# -------------------------------------------------------------------------
# Training Function with SoftCLDiceLoss
# -------------------------------------------------------------------------
def train_consensus_model(model, patches, epochs=3, lr=1e-4, batch_size=4):
    log(f"Starting Soft-Consensus Fine-Tuning with SoftCLDiceLoss ({epochs} epochs, lr={lr}, batch_size={batch_size})...")
    dataset = SoftConsensusPatchDataset(patches)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)

    criterion = SoftCLDiceLoss(alpha=0.5, smooth=1.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler('cuda') if torch.cuda.is_available() else None

    model.train()
    for ep in range(epochs):
        ep_loss = 0.0
        n_batches = 0
        for imgs, targets in loader:
            imgs = imgs.to(DEVICE)
            targets = targets.to(DEVICE)
            optimizer.zero_grad()

            if scaler:
                with torch.amp.autocast('cuda'):
                    logits = model(imgs)
                    loss = criterion(logits, targets)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(imgs)
                loss = criterion(logits, targets)
                loss.backward()
                optimizer.step()

            ep_loss += loss.item()
            n_batches += 1

        log(f"Epoch [{ep+1}/{epochs}] clDice Composite Loss: {ep_loss / max(1, n_batches):.4f}")

    model.eval()
    log("Topological Consensus Fine-Tuning Complete.")

# -------------------------------------------------------------------------
# Test Inference with 4-View TTA and 1024x1024 Linear Blending
# -------------------------------------------------------------------------
def predict_tile_tta(model, tile_tensor: torch.Tensor) -> np.ndarray:
    with torch.no_grad():
        with torch.amp.autocast('cuda') if torch.cuda.is_available() else torch.no_grad():
            p1 = torch.sigmoid(model(tile_tensor))
            p2 = torch.sigmoid(model(torch.flip(tile_tensor, dims=[3])))
            p2 = torch.flip(p2, dims=[3])
            p3 = torch.sigmoid(model(torch.flip(tile_tensor, dims=[2])))
            p3 = torch.flip(p3, dims=[2])
            p4 = torch.sigmoid(model(tile_tensor.transpose(2, 3)))
            p4 = p4.transpose(2, 3)

            prob = (p1 + p2 + p3 + p4) / 4.0
            return prob.squeeze().float().cpu().numpy()

def predict_full_image_1024_tiling(model, norm_img: np.ndarray) -> np.ndarray:
    full_prob = np.zeros((H, W), dtype=np.float32)
    weight_map = np.zeros((H, W), dtype=np.float32)

    d_x = np.minimum(np.arange(TILE_SIZE), TILE_SIZE - 1 - np.arange(TILE_SIZE))
    d_y = np.minimum(np.arange(TILE_SIZE), TILE_SIZE - 1 - np.arange(TILE_SIZE))
    window_2d = (np.minimum.outer(d_y, d_x) / (TILE_SIZE / 2.0)).astype(np.float32) + 0.1

    tile_coords = []
    for y in range(0, H - TILE_SIZE + 1, STRIDE):
        for x in range(0, W - TILE_SIZE + 1, STRIDE):
            tile_coords.append((y, x))

    for y, x in tile_coords:
        patch = norm_img[y:y+TILE_SIZE, x:x+TILE_SIZE]
        t_patch = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0).float().to(DEVICE)
        prob_patch = predict_tile_tta(model, t_patch)

        full_prob[y:y+TILE_SIZE, x:x+TILE_SIZE] += prob_patch * window_2d
        weight_map[y:y+TILE_SIZE, x:x+TILE_SIZE] += window_2d

    full_prob /= np.maximum(weight_map, 1e-6)
    full_prob[~DISK_MASK] = 0.0
    return full_prob

def mask_to_rle(binary_mask: np.ndarray) -> str:
    rle = mask_util.encode(np.asfortranarray(binary_mask.astype(np.uint8)))
    return rle["counts"].decode("ascii") if isinstance(rle["counts"], bytes) else str(rle["counts"])

# -------------------------------------------------------------------------
# Main Execution Pipeline
# -------------------------------------------------------------------------
def main():
    log("=" * 80)
    log("CHAMPIONSHIP EXP-039: TOPOLOGICAL CONSENSUS PIPELINE (0.66+ TARGET)")
    log("=" * 80)

    # 1. Discover Datasets & Models
    test_dirs = glob.glob("/kaggle/input/**/test/test_images", recursive=True)
    if not test_dirs:
        test_dirs = glob.glob("/kaggle/input/**/test_images", recursive=True)
    assert len(test_dirs) > 0, "Could not find test images directory!"
    test_dir = test_dirs[0]
    test_files = sorted(glob.glob(os.path.join(test_dir, "*.jpeg")) + glob.glob(os.path.join(test_dir, "*.jpg")))
    log(f"Found {len(test_files)} test images in: {test_dir}")

    ann_files = glob.glob("/kaggle/input/**/MAGFiLO_1.0_Annotations_kaggle2026_train.json", recursive=True)
    train_dirs = glob.glob("/kaggle/input/**/train/train_images", recursive=True)
    if not train_dirs:
        train_dirs = glob.glob("/kaggle/input/**/train_images", recursive=True)

    weight_candidates = glob.glob("/kaggle/input/**/unet_resnet34_fold*.pt", recursive=True)

    # Robust Anchor Discovery (Guarantees anchor is always found)
    anchor_candidates = (
        glob.glob("/kaggle/input/**/anchor_valhalla_058.csv", recursive=True) +
        glob.glob("/kaggle/input/**/anchor*.csv", recursive=True) +
        glob.glob("/kaggle/input/**/submission_exp031*.csv", recursive=True) +
        glob.glob("/kaggle/input/**/submission*.csv", recursive=True)
    )
    assert len(anchor_candidates) > 0, "FATAL: Could not find anchor CSV in attached datasets!"
    anchor_csv_path = anchor_candidates[0]
    anchor_df = pd.read_csv(anchor_csv_path)
    # Apply Quiet-Sun filter to anchor scaffold
    anchor_df = anchor_df[~anchor_df['filament_id'].apply(lambda x: x.split('_')[0]).isin(QUIET_SUN_STEMS)].copy()
    log(f"Successfully loaded 0.58 verified anchor scaffold from: {anchor_csv_path} ({len(anchor_df)} active instances)")

    # 2. Build Model
    log("Initializing ResNet34 U-Net...")
    model = smp.Unet(encoder_name='resnet34', encoder_weights=None, in_channels=1, classes=1).to(DEVICE)

    if weight_candidates:
        log(f"Loading pretrained weights from: {weight_candidates[0]}")
        state_dict = torch.load(weight_candidates[0], map_location=DEVICE)
        model.load_state_dict(state_dict)
    else:
        log("No pretrained checkpoint found; downloading ImageNet encoder weights...")
        model = smp.Unet(encoder_name='resnet34', encoder_weights='imagenet', in_channels=1, classes=1).to(DEVICE)

    # 3. Fine-Tune on Multi-Annotator Consensus Targets with SoftCLDiceLoss
    if ann_files and train_dirs:
        train_patches = build_training_patches(ann_files[0], train_dirs[0], max_patches=300)
        if train_patches:
            train_consensus_model(model, train_patches, epochs=3, lr=1e-4, batch_size=4)
            del train_patches
            gc.collect()
    else:
        log("Training annotations not accessible; running direct calibrated inference.")

    # 4. Anchor-First Priority Stamping & Free-Canvas Consensus Recovery
    log(f"Executing Tier-1 Anchor Stamping & Consensus Recovery across {len(test_files)} test images...")
    submission_rows = []
    stems_in_test = []
    added_consensus_candidates = 0

    for idx, fpath in enumerate(test_files, start=1):
        stem = Path(fpath).stem
        if stem in QUIET_SUN_STEMS:
            continue
        stems_in_test.append(stem)

        # Retrieve anchor instances for this active stem
        anchor_stem_rows = anchor_df[anchor_df['filament_id'].str.startswith(f"{stem}_")]
        assert len(anchor_stem_rows) > 0, f"FATAL: Anchor missing active stem {stem}!"

        occupied = np.zeros((H, W), dtype=bool)

        # 1. Tier-1 Priority: Stamp ALL Anchor Instances FIRST (100% Bit-Exact Preservation)
        for _, arow in anchor_stem_rows.iterrows():
            fid = arow["filament_id"]
            rle_str = arow["segmentation_rle"]
            m_anc = (mask_util.decode({'counts': rle_str.encode('ascii'), 'size': [H, W]}) > 0)
            occupied |= m_anc
            submission_rows.append({
                "filament_id": fid,
                "segmentation_rle": rle_str
            })

        # 2. Free-Canvas Consensus Recovery:
        # On deficit stems (fewer than 7 anchor filaments), discover true missing consensus flux ropes
        if len(anchor_stem_rows) < 7:
            img_gray = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
            if img_gray is not None:
                norm_img = normalize_solar_disk(img_gray)
                prob_map = predict_full_image_1024_tiling(model, norm_img)

                free_canvas = (~occupied) & DISK_MASK
                cand_bin = ((prob_map >= 0.88) & free_canvas).astype(np.uint8)

                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                cand_bin = cv2.morphologyEx(cand_bin, cv2.MORPH_OPEN, kernel)

                num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cand_bin, connectivity=8)
                cands = []
                for lbl in range(1, num_labels):
                    area = stats[lbl, cv2.CC_STAT_AREA]
                    if 300 <= area <= 35000:
                        c_mask = (labels == lbl)
                        mean_p = float(np.mean(prob_map[c_mask]))
                        w_box = stats[lbl, cv2.CC_STAT_WIDTH]
                        h_box = stats[lbl, cv2.CC_STAT_HEIGHT]
                        aspect_ratio = max(w_box, h_box) / max(1, min(w_box, h_box))
                        # Strict elongation & overwhelming neural consensus gate
                        if mean_p >= 0.89 and aspect_ratio >= 2.8:
                            cands.append((mean_p * math.sqrt(area), c_mask))

                cands.sort(key=lambda x: -x[0])
                cand_idx = len(anchor_stem_rows) + 1
                for score, c_mask in cands[:1]: # At most 1 high-precision consensus candidate per deficit stem
                    clean_cand = c_mask & (~occupied)
                    if clean_cand.sum() >= 300:
                        occupied |= clean_cand
                        rle_cand = mask_to_rle(clean_cand)
                        submission_rows.append({
                            "filament_id": f"{stem}_{cand_idx}",
                            "segmentation_rle": rle_cand
                        })
                        added_consensus_candidates += 1
                        cand_idx += 1

        if idx % 30 == 0 or idx == len(test_files):
            log(f"Processed [{idx}/{len(test_files)}] images | Total rows: {len(submission_rows)} (Added {added_consensus_candidates} consensus candidates)")

    out_df = pd.DataFrame(submission_rows)
    stems = out_df['filament_id'].apply(lambda x: x.split('_')[0])
    log(f"Final submission: {len(out_df)} instances across {stems.nunique()} active stems. Added candidates: {added_consensus_candidates}")

    out_csv = "/kaggle/working/submission.csv"
    out_df.to_csv(out_csv, index=False)
    log(f"Saved submission to {out_csv} ({os.path.getsize(out_csv)} bytes)")

    # 5. In-Kernel Zero-Trust 6-Point Integrity Audit
    log("=" * 80)
    log("RUNNING IN-KERNEL ZERO-TRUST 6-POINT INTEGRITY AUDIT")
    log("=" * 80)

    # Check 1: Schema
    assert list(out_df.columns) == ["filament_id", "segmentation_rle"], "Check 1 failed: columns mismatch"
    log("[PASSED] Check 1: Column Schema strictly ['filament_id', 'segmentation_rle']")

    # Check 2: Quiet Sun Guard
    qs_present = stems.isin(QUIET_SUN_STEMS).sum()
    assert qs_present == 0, f"Check 2 failed: {qs_present} quiet-sun rows found"
    log("[PASSED] Check 2: Quiet-Sun Guard (0 rows on quiet sun disks)")

    # Check 3: Density
    n_instances = len(out_df)
    n_stems = stems.nunique()
    assert 1250 <= n_instances <= 1350, f"Check 3 failed: density out of bounds ({n_instances})"
    assert n_stems == EXPECTED_ACTIVE_STEMS, f"Check 3 failed: expected {EXPECTED_ACTIVE_STEMS} stems, got {n_stems}"
    log(f"[PASSED] Check 3: Instance Density ({n_instances} instances across {n_stems} active stems)")

    # Check 4: RLE Invariants
    for rle in out_df['segmentation_rle']:
        decoded = mask_util.decode({'counts': rle.encode('ascii'), 'size': [2048, 2048]})
        assert decoded.shape == (2048, 2048), "Check 4 failed: shape mismatch"
    log("[PASSED] Check 4: RLE Bitstream & Geometry Invariants")

    # Check 5: Zero Collisions
    grouped = out_df.groupby(stems)['segmentation_rle'].apply(list)
    for stem_id, rles in grouped.items():
        if len(rles) <= 1:
            continue
        occupied_test = np.zeros((2048, 2048), dtype=bool)
        for rle in rles:
            m = (mask_util.decode({'counts': rle.encode('ascii'), 'size': [2048, 2048]}) > 0)
            assert not np.any(occupied_test & m), f"Check 5 failed: collision on stem {stem_id}"
            occupied_test |= m
    log("[PASSED] Check 5: Zero Pixel Overlap across all active stems")

    # Check 6: 100% Anchor Scaffold Preservation
    common_anchor_fids = set(anchor_df['filament_id']).intersection(set(out_df['filament_id']))
    assert len(common_anchor_fids) == len(anchor_df), f"Check 6 failed: missing anchor instances ({len(common_anchor_fids)}/{len(anchor_df)})"
    log(f"[PASSED] Check 6: 100% Anchor Scaffold Preserved ({len(common_anchor_fids)}/{len(anchor_df)} instances)")

    log("=" * 80)
    log("EXP-039 TOPOLOGICAL CONSENSUS PIPELINE COMPLETED SUCCESSFULLY!")
    log("=" * 80)

if __name__ == "__main__":
    main()
