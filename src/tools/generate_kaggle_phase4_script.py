#!/usr/bin/env python3
"""
src/tools/generate_kaggle_phase4_script.py
Generates the self-contained Kaggle kernel script for Phase 4:
- Experiment 1: High-Resolution Baseline (1024x1024 native tiles, 5 folds)
- Experiment 2: Pretrained Encoder (EfficientNet-B0 U-Net, 5 folds)
- Experiment 3: Loss Function Ablation (BCE + Dice vs BCE + Dice + clDice)
- Experiment 4: Calibrated Instance Extraction Sweeps (Seed & Expand, Area Filtering, Gap Bridging)
Strictly compliant, zero test-set leakage, frozen 5-fold evaluation.
"""

import json
from pathlib import Path
import pandas as pd

def generate():
    manifest_path = Path("/home/fathyyoussef162/data/train_folds_manifest.csv")
    df = pd.read_csv(manifest_path)
    s2f = df.groupby("stem")["fold"].first().to_dict()
    s2f_json = json.dumps(s2f)

    header = '"""\nSolar Filament Segmentation Challenge 2026 (IEEE BigData Cup)\nPhase 4: High-Resolution Architecture & Backbone Scaling\n- Exp 1: 1024x1024 High-Res Baseline (5 Folds)\n- Exp 2: Pretrained EfficientNet-B0 U-Net (5 Folds)\n- Exp 3: Topological clDice Loss Ablation\n- Exp 4: Calibrated Instance Extraction Sweeps\n"""\n\n'

    body = f'''import os
import sys
import glob
import json
import time
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Literal, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from pycocotools import mask as mask_util

# -------------------------------------------------------------------------
# 1. Global Setup & Frozen Split Manifest
# -------------------------------------------------------------------------

SEED = 42
TILE_SIZE = 1024
STRIDE = 512
ORIGINAL_H = 2048
ORIGINAL_W = 2048
IOU_THRESHOLD = 0.50

def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
t0 = time.time()

def log(*args):
    print(f"[{{(time.time()-t0)/60:6.1f}}m]", *args, flush=True)

# Frozen 5-Fold Grouped Split
FROZEN_STEM_TO_FOLD = json.loads({repr(s2f_json)})

log(f"Active Compute Device: {{DEVICE}}")
if torch.cuda.is_available():
    log(f"GPU Model: {{torch.cuda.get_device_name(0)}} (Total VRAM: {{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}} GB)")
log(f"Frozen split loaded: {{len(FROZEN_STEM_TO_FOLD)}} physical observation stems.")


# -------------------------------------------------------------------------
# 2. Architectures: Baseline U-Net & Pretrained EfficientNet-B0 U-Net
# -------------------------------------------------------------------------

class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Down(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.down = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_ch, out_ch))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(x)


class Up(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch)
    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class BaselineUNet(nn.Module):
    """Native U-Net (7.76M parameters)"""
    def __init__(self, in_ch: int = 1, out_ch: int = 1, base_f: int = 32):
        super().__init__()
        self.inc = DoubleConv(in_ch, base_f)
        self.down1 = Down(base_f, base_f * 2)
        self.down2 = Down(base_f * 2, base_f * 4)
        self.down3 = Down(base_f * 4, base_f * 8)
        self.down4 = Down(base_f * 8, base_f * 16)
        self.up1 = Up(base_f * 16, base_f * 8)
        self.up2 = Up(base_f * 8, base_f * 4)
        self.up3 = Up(base_f * 4, base_f * 2)
        self.up4 = Up(base_f * 2, base_f)
        self.outc = nn.Conv2d(base_f, out_ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)


class PretrainedDecoderBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv((in_ch // 2) + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=True)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class PretrainedEfficientNetB0UNet(nn.Module):
    """U-Net with ImageNet Pretrained EfficientNet-B0 Encoder (10.2M parameters)"""
    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        backbone = models.efficientnet_b0(weights=weights)
        self.features = backbone.features
        
        # Decoder stages
        self.dec4 = PretrainedDecoderBlock(in_ch=1280, skip_ch=112, out_ch=256)
        self.dec3 = PretrainedDecoderBlock(in_ch=256, skip_ch=40, out_ch=128)
        self.dec2 = PretrainedDecoderBlock(in_ch=128, skip_ch=24, out_ch=64)
        self.dec1 = PretrainedDecoderBlock(in_ch=64, skip_ch=16, out_ch=32)
        
        self.final_up = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.final_conv = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Replicate 1-channel grayscale to 3 channels for ImageNet filter compatibility
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        s0 = self.features[0](x)
        s1 = self.features[1](s0)   # 16 ch, 1/2
        s2 = self.features[2](s1)   # 24 ch, 1/4
        s3 = self.features[3](s2)   # 40 ch, 1/8
        s4 = self.features[4](s3)
        s5 = self.features[5](s4)   # 112 ch, 1/16
        s6 = self.features[6](s5)
        s7 = self.features[7](s6)
        s8 = self.features[8](s7)   # 1280 ch, 1/32

        d4 = self.dec4(s8, s5)
        d3 = self.dec3(d4, s3)
        d2 = self.dec2(d3, s2)
        d1 = self.dec1(d2, s1)
        out = self.final_up(d1)
        return self.final_conv(out)


# -------------------------------------------------------------------------
# 3. Losses (BCE, Soft Dice, Topological clDice)
# -------------------------------------------------------------------------

def soft_dice_loss(pred_sig: torch.Tensor, target: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    pred_flat = pred_sig.view(pred_sig.size(0), -1)
    target_flat = target.view(target.size(0), -1)
    intersection = (pred_flat * target_flat).sum(dim=1)
    cardinality = pred_flat.sum(dim=1) + target_flat.sum(dim=1)
    dice = (2.0 * intersection + smooth) / (cardinality + smooth)
    return 1.0 - dice.mean()


def soft_skeletonize(x: torch.Tensor, iters: int = 3) -> torch.Tensor:
    for _ in range(iters):
        min_pool = -F.max_pool2d(-x, kernel_size=3, stride=1, padding=1)
        max_pool = F.max_pool2d(min_pool, kernel_size=3, stride=1, padding=1)
        skeleton = F.relu(x - max_pool)
        x = min_pool
    return skeleton


def soft_cldice_loss(pred_sig: torch.Tensor, target: torch.Tensor, iters: int = 3, smooth: float = 1.0) -> torch.Tensor:
    skel_pred = soft_skeletonize(pred_sig, iters=iters)
    skel_target = soft_skeletonize(target, iters=iters)
    tprec = (torch.sum(pred_sig * skel_target, dim=(1,2,3)) + smooth) / (torch.sum(skel_target, dim=(1,2,3)) + smooth)
    tsens = (torch.sum(skel_pred * target, dim=(1,2,3)) + smooth) / (torch.sum(skel_pred, dim=(1,2,3)) + smooth)
    cl_dice = (2.0 * tprec * tsens) / (tprec + tsens)
    return 1.0 - torch.mean(cl_dice)


# -------------------------------------------------------------------------
# 4. Official Evaluator & Blending Matrix
# -------------------------------------------------------------------------

def mask_to_rle(binary_mask: np.ndarray) -> str:
    rle = mask_util.encode(np.asfortranarray(binary_mask.astype(np.uint8)))
    counts = rle["counts"].decode("ascii") if isinstance(rle["counts"], bytes) else str(rle["counts"])
    return counts


def get_gaussian_weight_matrix(size: int = TILE_SIZE, sigma_scale: float = 0.25) -> torch.Tensor:
    sigma = size * sigma_scale
    k1d = cv2.getGaussianKernel(size, sigma)
    k2d = np.outer(k1d, k1d)
    k2d = k2d / k2d.max()
    return torch.from_numpy(k2d.astype(np.float32)).to(DEVICE)


def compute_official_pq(gt_df: pd.DataFrame, pred_df: pd.DataFrame, iou_threshold: float = IOU_THRESHOLD) -> Dict:
    if len(pred_df) == 0:
        return {{"pq": 0.0, "sq": 0.0, "rq": 0.0, "tp": 0, "fp": 0, "fn": len(gt_df), "num_annotator_views": len(gt_df["filament_id"].str.split("_").str[0].unique())}}

    gt_annotator_image_ids = gt_df["filament_id"].str.split("_", n=1).str[0]
    pred_image_ids = pred_df["filament_id"].str.rsplit("_", n=1).str[0]
    annotator_views = gt_annotator_image_ids.unique()

    tp_iou_scores = []
    fp_count = 0
    fn_count = 0

    pred_by_stem = {{}}
    for stem, group in pred_df.groupby(pred_image_ids):
        pred_by_stem[stem] = group["segmentation_rle"].tolist()

    for annotator_view in annotator_views:
        stem = annotator_view.split("-", 1)[1] if "-" in annotator_view else annotator_view
        gt_rles = gt_df.loc[gt_annotator_image_ids == annotator_view, "segmentation_rle"].tolist()
        pred_rles = pred_by_stem.get(stem, [])

        n_gt = len(gt_rles)
        n_pred = len(pred_rles)

        if n_gt == 0:
            fp_count += n_pred
            continue
        if n_pred == 0:
            fn_count += n_gt
            continue

        gt_dicts = [{{"size": [ORIGINAL_H, ORIGINAL_W], "counts": r.encode("ascii") if isinstance(r, str) else r}} for r in gt_rles]
        dt_dicts = [{{"size": [ORIGINAL_H, ORIGINAL_W], "counts": r.encode("ascii") if isinstance(r, str) else r}} for r in pred_rles]
        iou_matrix = mask_util.iou(dt_dicts, gt_dicts, [0] * n_gt).T
        hit_matrix = iou_matrix > iou_threshold

        tp_iou_scores.extend(iou_matrix[hit_matrix].tolist())
        fp_count += int((hit_matrix.sum(axis=0) == 0).sum())
        fn_count += int((hit_matrix.sum(axis=1) == 0).sum())

    tp_count = len(tp_iou_scores)
    denom = tp_count + 0.5 * fp_count + 0.5 * fn_count
    sq = float(np.mean(tp_iou_scores)) if tp_count > 0 else 0.0
    rq = float(tp_count / denom) if denom > 0 else 0.0
    pq = float(sum(tp_iou_scores) / denom) if denom > 0 else 0.0

    return {{
        "pq": pq,
        "sq": sq,
        "rq": rq,
        "tp": tp_count,
        "fp": fp_count,
        "fn": fn_count,
        "num_annotator_views": len(annotator_views)
    }}


# -------------------------------------------------------------------------
# 5. Native 1024x1024 Tile Dataset (Strategy A)
# -------------------------------------------------------------------------

class SolarTileDataset(Dataset):
    """
    Samples native 1024x1024 patches centered on filaments with jitter and random background tiles.
    Strategy A: Each annotator pass is an independent sample.
    """
    def __init__(self, samples: List[Dict], train_img_dir: str, augment: bool = True):
        self.samples = samples
        self.train_img_dir = train_img_dir
        self.augment = augment

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        item = self.samples[idx]
        img_p = os.path.join(self.train_img_dir, item["file_name"])
        if not os.path.exists(img_p):
            for ext in [".jpeg", ".jpg", ".png"]:
                c = os.path.join(self.train_img_dir, item["stem"] + ext)
                if os.path.exists(c):
                    img_p = c
                    break

        img = cv2.imread(img_p, cv2.IMREAD_GRAYSCALE) if os.path.exists(img_p) else np.zeros((ORIGINAL_H, ORIGINAL_W), dtype=np.uint8)
        
        # Render full 2048x2048 mask for this pass
        mask = np.zeros((ORIGINAL_H, ORIGINAL_W), dtype=np.uint8)
        for ann in item["annotations"]:
            seg = ann.get("segmentation", [])
            if not seg:
                continue
            rles = mask_util.frPyObjects(seg, ORIGINAL_H, ORIGINAL_W)
            rle = mask_util.merge(rles)
            m = mask_util.decode(rle)
            if m.ndim == 3:
                m = m.squeeze(-1)
            mask |= m.astype(np.uint8)

        cx, cy = item.get("center", (1024, 1024))
        if self.augment:
            cx += random.randint(-64, 64)
            cy += random.randint(-64, 64)

        x0 = max(0, min(ORIGINAL_W - TILE_SIZE, cx - TILE_SIZE // 2))
        y0 = max(0, min(ORIGINAL_H - TILE_SIZE, cy - TILE_SIZE // 2))

        patch_img = img[y0:y0+TILE_SIZE, x0:x0+TILE_SIZE].astype(np.float32) / 255.0
        patch_mask = mask[y0:y0+TILE_SIZE, x0:x0+TILE_SIZE].astype(np.float32)

        if self.augment:
            if random.random() > 0.5:
                patch_img = np.ascontiguousarray(np.fliplr(patch_img))
                patch_mask = np.ascontiguousarray(np.fliplr(patch_mask))
            if random.random() > 0.5:
                patch_img = np.ascontiguousarray(np.flipud(patch_img))
                patch_mask = np.ascontiguousarray(np.flipud(patch_mask))

        return (
            torch.from_numpy(patch_img).unsqueeze(0),
            torch.from_numpy(patch_mask).unsqueeze(0)
        )


def build_tile_samples(coco: Dict, stem_to_fold: Dict[str, int], fold: int, is_train: bool) -> List[Dict]:
    images_by_id = {{im["id"]: im for im in coco["images"]}}
    stems_to_image_ids = defaultdict(list)
    for im in coco["images"]:
        stem = im["file_name"].split(".")[0]
        assigned_fold = stem_to_fold[stem]
        if (assigned_fold != fold) if is_train else (assigned_fold == fold):
            stems_to_image_ids[stem].append(im["id"])

    anns_by_image_id = defaultdict(list)
    for a in coco["annotations"]:
        anns_by_image_id[a["image_id"]].append(a)

    tile_samples = []
    for stem, iids in stems_to_image_ids.items():
        for iid in iids:
            im_info = images_by_id[iid]
            anns = anns_by_image_id[iid]
            
            # Find filament centers
            centers = []
            for a in anns:
                bbox = a.get("bbox", [])
                if len(bbox) == 4:
                    cx = int(bbox[0] + bbox[2] / 2)
                    cy = int(bbox[1] + bbox[3] / 2)
                    centers.append((cx, cy))
                    
            if len(centers) > 0:
                # Sample 2 primary filament centers per image record
                sampled_centers = random.sample(centers, min(len(centers), 2))
                for c in sampled_centers:
                    tile_samples.append({{
                        "stem": stem,
                        "file_name": im_info["file_name"],
                        "annotations": anns,
                        "center": c
                    }})
            else:
                tile_samples.append({{
                    "stem": stem,
                    "file_name": im_info["file_name"],
                    "annotations": anns,
                    "center": (1024, 1024)
                }})
    return tile_samples


# -------------------------------------------------------------------------
# 6. Extraction Methods (Standard, Seed-and-Expand, Gap Bridging)
# -------------------------------------------------------------------------

def extract_instances_calibrated(
    full_prob: np.ndarray,
    stem: str,
    method: Literal["standard", "seed_expand"] = "standard",
    seed_thresh: float = 0.70,
    expand_thresh: float = 0.40,
    min_area: int = 100,
    gap_bridge: bool = False,
) -> List[Dict]:
    if method == "standard":
        binary = (full_prob >= expand_thresh).astype(np.uint8)
    elif method == "seed_expand":
        seed = full_prob >= seed_thresh
        cand = full_prob >= expand_thresh
        if gap_bridge:
            k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            cand = cv2.morphologyEx(cand.astype(np.uint8), cv2.MORPH_CLOSE, k).astype(bool)
        num_labels, labels = cv2.connectedComponents(cand.astype(np.uint8), connectivity=8)
        binary = np.zeros_like(cand, dtype=np.uint8)
        for lbl in range(1, num_labels):
            comp = (labels == lbl)
            if np.any(comp & seed):
                binary[comp] = 1

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary.astype(np.uint8), connectivity=8)
    rows = []
    inst_count = 0
    for lbl in range(1, num_labels):
        if stats[lbl, cv2.CC_STAT_AREA] >= min_area:
            inst_count += 1
            inst_mask = (labels == lbl).astype(np.uint8)
            rows.append({{
                "filament_id": f"{{stem}}_{{inst_count}}",
                "segmentation_rle": mask_to_rle(inst_mask)
            }})
    return rows


# -------------------------------------------------------------------------
# 7. Experiment Runner Engine
# -------------------------------------------------------------------------

def run_highres_fold(
    fold: int,
    model_type: Literal["baseline_unet", "efficientnet_b0"],
    use_cldice: bool,
    coco: Dict,
    img_id_to_stem: Dict[str, str],
    train_img_dir: str,
    epochs: int = 3,
    batch_size: int = 8,
    lr: float = 5e-4,
) -> Tuple[Dict, Dict[str, np.ndarray]]:
    """Runs high-resolution training and validation inference. Returns metrics and raw probability maps."""
    set_seed(SEED + fold * 10)
    exp_name = f"{{model_type}}_{{'cldice' if use_cldice else 'dice'}}_fold{{fold}}"
    log(f"--- Starting High-Res Run: {{exp_name}} ---")

    train_samples = build_tile_samples(coco, FROZEN_STEM_TO_FOLD, fold=fold, is_train=True)
    val_stems = sorted([s for s, f in FROZEN_STEM_TO_FOLD.items() if f == fold])
    log(f"Prepared {{len(train_samples)}} native 1024x1024 tile samples across training folds.")

    dataset = SolarTileDataset(train_samples, train_img_dir=train_img_dir, augment=True)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)

    if model_type == "baseline_unet":
        model = BaselineUNet(in_ch=1, out_ch=1, base_f=32).to(DEVICE)
    elif model_type == "efficientnet_b0":
        model = PretrainedEfficientNetB0UNet(pretrained=True).to(DEVICE)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion_bce = nn.BCEWithLogitsLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs * len(loader)), eta_min=1e-5)
    scaler = torch.amp.GradScaler("cuda") if DEVICE.type == "cuda" else None

    # Reset peak memory tracker
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(0)

    # Training Loop
    t_train_start = time.time()
    model.train()
    for ep in range(1, epochs + 1):
        ep_loss = 0.0
        for imgs, masks in loader:
            imgs = imgs.to(DEVICE, non_blocking=True)
            masks = masks.to(DEVICE, non_blocking=True)
            optimizer.zero_grad()
            with torch.amp.autocast("cuda"):
                logits = model(imgs)
                sig = torch.sigmoid(logits)
                loss = criterion_bce(logits, masks) + soft_dice_loss(sig, masks)
                if use_cldice:
                    loss += 0.5 * soft_cldice_loss(sig, masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            ep_loss += loss.item()
        log(f"Fold {{fold}} [{{model_type}}] Ep {{ep}}/{{epochs}} | Loss: {{ep_loss/len(loader):.4f}} | LR: {{scheduler.get_last_lr()[0]:.2e}}")
    train_time = time.time() - t_train_start
    peak_gpu_mb = torch.cuda.max_memory_allocated(0) / 1e6 if torch.cuda.is_available() else 0.0

    ckpt_path = f"/kaggle/working/{{exp_name}}.pt"
    torch.save(model.state_dict(), ckpt_path)

    # 1024x1024 Overlapping Tile Inference with 2D Gaussian Window
    gaussian_weight = get_gaussian_weight_matrix(TILE_SIZE)
    t_infer_start = time.time()
    model.eval()

    val_prob_maps = {{}}
    pred_rows_default = []

    with torch.no_grad():
        for stem in val_stems:
            img_p = os.path.join(train_img_dir, stem + ".jpeg")
            if not os.path.exists(img_p):
                for ext in [".jpg", ".png"]:
                    cand = os.path.join(train_img_dir, stem + ext)
                    if os.path.exists(cand):
                        img_p = cand
                        break
            img = cv2.imread(img_p, cv2.IMREAD_GRAYSCALE) if os.path.exists(img_p) else np.zeros((ORIGINAL_H, ORIGINAL_W), dtype=np.uint8)
            img_norm = img.astype(np.float32) / 255.0

            prob_accum = torch.zeros((ORIGINAL_H, ORIGINAL_W), device=DEVICE, dtype=torch.float32)
            weight_accum = torch.zeros((ORIGINAL_H, ORIGINAL_W), device=DEVICE, dtype=torch.float32)

            for y0 in range(0, ORIGINAL_H - TILE_SIZE + 1, STRIDE):
                for x0 in range(0, ORIGINAL_W - TILE_SIZE + 1, STRIDE):
                    patch = np.ascontiguousarray(img_norm[y0:y0+TILE_SIZE, x0:x0+TILE_SIZE])
                    p_tensor = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0).to(DEVICE)
                    with torch.amp.autocast("cuda"):
                        logits = model(p_tensor)
                    prob = torch.sigmoid(logits).squeeze()
                    prob_accum[y0:y0+TILE_SIZE, x0:x0+TILE_SIZE] += prob * gaussian_weight
                    weight_accum[y0:y0+TILE_SIZE, x0:x0+TILE_SIZE] += gaussian_weight

            full_prob = (prob_accum / torch.clamp(weight_accum, min=1e-6)).to(torch.float32).cpu().numpy()
            val_prob_maps[stem] = full_prob

            # Extract instances with default standard threshold (expand=0.50, min_area=50)
            rows = extract_instances_calibrated(full_prob, stem=stem, method="standard", expand_thresh=0.50, min_area=50)
            pred_rows_default.extend(rows)

    infer_time = time.time() - t_infer_start
    pred_df_default = pd.DataFrame(pred_rows_default, columns=["filament_id", "segmentation_rle"])

    # Build exact GT DataFrame for this fold
    val_stems_set = set(val_stems)
    gt_rows = []
    for ann in coco["annotations"]:
        im_stem = img_id_to_stem.get(ann["image_id"])
        if im_stem in val_stems_set:
            seg = ann.get("segmentation", [])
            if not seg:
                continue
            rles = mask_util.frPyObjects(seg, ORIGINAL_H, ORIGINAL_W)
            rle = mask_util.merge(rles)
            counts = rle["counts"].decode("ascii") if isinstance(rle["counts"], bytes) else str(rle["counts"])
            gt_rows.append({{
                "filament_id": f"{{ann['image_id']}}_{{ann['id']}}",
                "segmentation_rle": counts
            }})
    gt_df = pd.DataFrame(gt_rows, columns=["filament_id", "segmentation_rle"])

    metrics_default = compute_official_pq(gt_df, pred_df_default, iou_threshold=0.50)
    result = {{
        "fold": fold,
        "model_type": model_type,
        "use_cldice": use_cldice,
        "parameters": num_params,
        "peak_gpu_mb": round(peak_gpu_mb, 1),
        "pq": metrics_default["pq"],
        "sq": metrics_default["sq"],
        "rq": metrics_default["rq"],
        "tp": metrics_default["tp"],
        "fp": metrics_default["fp"],
        "fn": metrics_default["fn"],
        "train_time_seconds": round(train_time, 2),
        "infer_time_seconds": round(infer_time, 2),
        "checkpoint_path": ckpt_path
    }}
    log(f"Fold {{fold}} [{{model_type}}] PQ: {{result['pq']:.5f}} (SQ: {{result['sq']:.5f}}, RQ: {{result['rq']:.5f}} | TP: {{result['tp']}}, FP: {{result['fp']}}, FN: {{result['fn']}})")
    return result, val_prob_maps


# -------------------------------------------------------------------------
# 8. Master Phase 4 Execution Flow
# -------------------------------------------------------------------------

def main():
    log("=== Starting Phase 4: High-Resolution Architecture & Backbone Scaling ===")
    ann_paths = glob.glob("/kaggle/input/**/MAGFiLO_1.0_Annotations_kaggle2026_train.json", recursive=True)
    if not ann_paths:
        ann_paths = glob.glob("data/**/MAGFiLO_1.0_Annotations_kaggle2026_train.json", recursive=True)
    assert len(ann_paths) > 0, "Could not find training annotations!"
    
    comp_root = os.path.dirname(os.path.dirname(ann_paths[0]))
    train_img_dir = os.path.join(comp_root, "train", "train_images")
    with open(ann_paths[0], "r", encoding="utf-8") as f:
        coco = json.load(f)
    img_id_to_stem = {{im["id"]: im["file_name"].split(".")[0] for im in coco["images"]}}

    phase4_summary = {{
        "phase": 4,
        "baseline_512_ref": {{"mean_pq": 0.08377, "std_pq": 0.01817, "tp": 1843, "fp": 15579, "fn": 6356}},
        "exp1_highres_1024_baseline": {{}},
        "exp2_pretrained_efficientnet": {{}},
        "exp3_cldice_ablation": {{}},
        "exp4_extraction_sweeps": {{}}
    }}

    # ---------------------------------------------------------------------
    # EXPERIMENT 1: High-Resolution 1024x1024 Baseline U-Net (5 Folds)
    # ---------------------------------------------------------------------
    log("=======================================================")
    log("EXPERIMENT 1: 1024x1024 Native Tile Baseline U-Net (5 Folds)")
    log("=======================================================")
    exp1_fold_results = []
    exp1_prob_maps_fold0 = None
    for f in range(5):
        res, prob_maps = run_highres_fold(
            fold=f, model_type="baseline_unet", use_cldice=False,
            coco=coco, img_id_to_stem=img_id_to_stem, train_img_dir=train_img_dir, epochs=3
        )
        exp1_fold_results.append(res)
        if f == 0:
            exp1_prob_maps_fold0 = prob_maps

    pqs_1 = [r["pq"] for r in exp1_fold_results]
    sqs_1 = [r["sq"] for r in exp1_fold_results]
    rqs_1 = [r["rq"] for r in exp1_fold_results]
    phase4_summary["exp1_highres_1024_baseline"] = {{
        "description": "Native 1024x1024 overlapping tiles, Baseline U-Net (7.76M params)",
        "mean_pq": float(np.mean(pqs_1)),
        "std_pq": float(np.std(pqs_1)),
        "mean_sq": float(np.mean(sqs_1)),
        "mean_rq": float(np.mean(rqs_1)),
        "total_tp": int(sum(r["tp"] for r in exp1_fold_results)),
        "total_fp": int(sum(r["fp"] for r in exp1_fold_results)),
        "total_fn": int(sum(r["fn"] for r in exp1_fold_results)),
        "total_train_time_seconds": round(sum(r["train_time_seconds"] for r in exp1_fold_results), 2),
        "total_infer_time_seconds": round(sum(r["infer_time_seconds"] for r in exp1_fold_results), 2),
        "folds": exp1_fold_results
    }}
    log(f">>> Exp 1 High-Res 1024 Baseline: Mean PQ = {{phase4_summary['exp1_highres_1024_baseline']['mean_pq']:.5f}} +/- {{phase4_summary['exp1_highres_1024_baseline']['std_pq']:.5f}}")

    # ---------------------------------------------------------------------
    # EXPERIMENT 2: Pretrained EfficientNet-B0 U-Net (5 Folds)
    # ---------------------------------------------------------------------
    log("=======================================================")
    log("EXPERIMENT 2: Pretrained EfficientNet-B0 U-Net (5 Folds)")
    log("=======================================================")
    exp2_fold_results = []
    exp2_prob_maps_fold0 = None
    for f in range(5):
        res, prob_maps = run_highres_fold(
            fold=f, model_type="efficientnet_b0", use_cldice=False,
            coco=coco, img_id_to_stem=img_id_to_stem, train_img_dir=train_img_dir, epochs=3
        )
        exp2_fold_results.append(res)
        if f == 0:
            exp2_prob_maps_fold0 = prob_maps

    pqs_2 = [r["pq"] for r in exp2_fold_results]
    sqs_2 = [r["sq"] for r in exp2_fold_results]
    rqs_2 = [r["rq"] for r in exp2_fold_results]
    phase4_summary["exp2_pretrained_efficientnet"] = {{
        "description": "Pretrained EfficientNet-B0 U-Net (10.2M params) on 1024x1024 tiles",
        "mean_pq": float(np.mean(pqs_2)),
        "std_pq": float(np.std(pqs_2)),
        "mean_sq": float(np.mean(sqs_2)),
        "mean_rq": float(np.mean(rqs_2)),
        "total_tp": int(sum(r["tp"] for r in exp2_fold_results)),
        "total_fp": int(sum(r["fp"] for r in exp2_fold_results)),
        "total_fn": int(sum(r["fn"] for r in exp2_fold_results)),
        "total_train_time_seconds": round(sum(r["train_time_seconds"] for r in exp2_fold_results), 2),
        "total_infer_time_seconds": round(sum(r["infer_time_seconds"] for r in exp2_fold_results), 2),
        "folds": exp2_fold_results
    }}
    log(f">>> Exp 2 Pretrained EfficientNet: Mean PQ = {{phase4_summary['exp2_pretrained_efficientnet']['mean_pq']:.5f}} +/- {{phase4_summary['exp2_pretrained_efficientnet']['std_pq']:.5f}}")

    # ---------------------------------------------------------------------
    # EXPERIMENT 3: Topological clDice Loss Ablation (Fold 0)
    # ---------------------------------------------------------------------
    log("=======================================================")
    log("EXPERIMENT 3: Topological clDice Loss Ablation (Fold 0)")
    log("=======================================================")
    res_cldice, _ = run_highres_fold(
        fold=0, model_type="efficientnet_b0", use_cldice=True,
        coco=coco, img_id_to_stem=img_id_to_stem, train_img_dir=train_img_dir, epochs=3
    )
    phase4_summary["exp3_cldice_ablation"] = {{
        "fold_0_without_cldice": exp2_fold_results[0],
        "fold_0_with_cldice": res_cldice,
        "pq_delta": round(res_cldice["pq"] - exp2_fold_results[0]["pq"], 5),
        "tp_delta": res_cldice["tp"] - exp2_fold_results[0]["tp"],
        "fp_delta": res_cldice["fp"] - exp2_fold_results[0]["fp"]
    }}
    log(f">>> Exp 3 clDice Ablation: Without clDice PQ = {{exp2_fold_results[0]['pq']:.5f}} -> With clDice PQ = {{res_cldice['pq']:.5f}} (Delta: {{phase4_summary['exp3_cldice_ablation']['pq_delta']:+.5f}})")

    # ---------------------------------------------------------------------
    # EXPERIMENT 4: Calibrated Instance Extraction Sweeps (Fold 0)
    # ---------------------------------------------------------------------
    log("=======================================================")
    log("EXPERIMENT 4: Calibrated Instance Extraction Sweeps (Fold 0)")
    log("=======================================================")
    # Evaluate extraction parameter sweeps on out-of-fold probability maps
    val_stems_f0 = sorted([s for s, f in FROZEN_STEM_TO_FOLD.items() if f == 0])
    val_stems_set_f0 = set(val_stems_f0)
    gt_rows_f0 = []
    for ann in coco["annotations"]:
        im_stem = img_id_to_stem.get(ann["image_id"])
        if im_stem in val_stems_set_f0:
            seg = ann.get("segmentation", [])
            if not seg:
                continue
            rles = mask_util.frPyObjects(seg, ORIGINAL_H, ORIGINAL_W)
            rle = mask_util.merge(rles)
            counts = rle["counts"].decode("ascii") if isinstance(rle["counts"], bytes) else str(rle["counts"])
            gt_rows_f0.append({{"filament_id": f"{{ann['image_id']}}_{{ann['id']}}", "segmentation_rle": counts}})
    gt_df_f0 = pd.DataFrame(gt_rows_f0, columns=["filament_id", "segmentation_rle"])

    sweep_configs = [
        {{"name": "Standard (0.50, min50)", "method": "standard", "expand": 0.50, "seed": 0.50, "min_area": 50, "bridge": False}},
        {{"name": "Standard (0.50, min150)", "method": "standard", "expand": 0.50, "seed": 0.50, "min_area": 150, "bridge": False}},
        {{"name": "Standard (0.60, min100)", "method": "standard", "expand": 0.60, "seed": 0.60, "min_area": 100, "bridge": False}},
        {{"name": "Seed-Expand (0.70/0.40, min100)", "method": "seed_expand", "seed": 0.70, "expand": 0.40, "min_area": 100, "bridge": False}},
        {{"name": "Seed-Expand (0.70/0.40, min150)", "method": "seed_expand", "seed": 0.70, "expand": 0.40, "min_area": 150, "bridge": False}},
        {{"name": "Seed-Expand (0.80/0.40, min150)", "method": "seed_expand", "seed": 0.80, "expand": 0.40, "min_area": 150, "bridge": False}},
        {{"name": "Seed-Expand + GapBridge (0.70/0.40, min150)", "method": "seed_expand", "seed": 0.70, "expand": 0.40, "min_area": 150, "bridge": True}},
    ]

    extraction_results = []
    for cfg in sweep_configs:
        sweep_rows = []
        for stem in val_stems_f0:
            full_prob = exp2_prob_maps_fold0[stem]
            rows = extract_instances_calibrated(
                full_prob, stem=stem, method=cfg["method"],
                seed_thresh=cfg["seed"], expand_thresh=cfg["expand"],
                min_area=cfg["min_area"], gap_bridge=cfg["bridge"]
            )
            sweep_rows.extend(rows)
        sweep_pred_df = pd.DataFrame(sweep_rows, columns=["filament_id", "segmentation_rle"])
        m = compute_official_pq(gt_df_f0, sweep_pred_df, iou_threshold=0.50)
        res_cfg = {{
            "config": cfg["name"],
            "pq": m["pq"],
            "sq": m["sq"],
            "rq": m["rq"],
            "tp": m["tp"],
            "fp": m["fp"],
            "fn": m["fn"],
            "total_predicted": len(sweep_rows)
        }}
        extraction_results.append(res_cfg)
        log(f"Extraction Sweep [{{cfg['name']}}]: PQ = {{m['pq']:.5f}} (SQ: {{m['sq']:.5f}}, RQ: {{m['rq']:.5f}} | TP: {{m['tp']}}, FP: {{m['fp']}}, FN: {{m['fn']}})")

    phase4_summary["exp4_extraction_sweeps"] = extraction_results

    out_file = "/kaggle/working/phase4_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(phase4_summary, f, indent=2)
    log(f"Exported Phase 4 results to: {{out_file}}")
    log("=== Phase 4 Execution Finished Successfully! ===")


if __name__ == "__main__":
    main()
'''

    out_script = Path("/home/fathyyoussef162/kaggle_kernel/solar_filament_phase4_highres.py")
    out_script.write_text(header + body)
    print(f"Generated {out_script} ({out_script.stat().st_size:,} bytes).")

if __name__ == "__main__":
    generate()
