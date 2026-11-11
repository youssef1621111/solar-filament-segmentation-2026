#!/usr/bin/env python3
"""
src/training/train_eval_fold.py
Training and evaluation pipeline for a single fold and supervision strategy.
Reports exact official Panoptic Quality (PQ, SQ, RQ, TP, FP, FN), training time, and inference time.
"""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Literal, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from pycocotools import mask as mask_util
from torch.utils.data import DataLoader

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.coco_to_rle_gt import build_gt_dataframe_for_fold
from src.data.dataset import SolarFilamentDataset, build_samples_for_strategy
from src.eval.panoptic_quality import compute_panoptic_quality_exact
from src.models.unet import BaselineUNet

HEIGHT = 2048
WIDTH = 2048


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def soft_dice_loss(pred_sig: torch.Tensor, target: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    pred_flat = pred_sig.view(pred_sig.size(0), -1)
    target_flat = target.view(target.size(0), -1)
    intersection = (pred_flat * target_flat).sum(dim=1)
    cardinality = pred_flat.sum(dim=1) + target_flat.sum(dim=1)
    dice = (2.0 * intersection + smooth) / (cardinality + smooth)
    return 1.0 - dice.mean()


def mask_to_rle(binary_mask: np.ndarray) -> str:
    rle = mask_util.encode(np.asfortranarray(binary_mask.astype(np.uint8)))
    counts = rle["counts"].decode("ascii") if isinstance(rle["counts"], bytes) else str(rle["counts"])
    return counts


def run_fold(
    fold: int,
    strategy: Literal["A", "B", "C"],
    train_json_path: str,
    manifest_csv_path: str,
    images_dir: str,
    checkpoint_dir: str = "checkpoints",
    results_dir: str = "results",
    epochs: int = 5,
    batch_size: int = 8,
    lr: float = 5e-4,
    target_size: int = 512,
    device_str: str = "cuda" if torch.cuda.is_available() else "cpu",
    seed: int = 42,
    num_workers: int = 2,
    min_area: int = 50,
    prob_threshold: float = 0.50,
) -> Dict:
    set_seed(seed)
    device = torch.device(device_str)
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    manifest_df = pd.read_csv(manifest_csv_path)
    
    print(f"\n=======================================================")
    print(f"Starting Fold {fold} | Strategy {strategy} | Target Size {target_size}x{target_size} | Device {device}")
    print(f"=======================================================")
    
    # 1. Prepare Datasets
    train_samples = build_samples_for_strategy(
        train_json_path, manifest_df, fold=fold, is_train=True, strategy=strategy
    )
    val_stems = sorted(list(set(manifest_df[manifest_df["fold"] == fold]["stem"])))
    
    print(f"Train samples count: {len(train_samples)}")
    print(f"Validation stems count: {len(val_stems)}")
    
    train_dataset = SolarFilamentDataset(
        train_samples, images_dir=images_dir, target_size=target_size, augment=True
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )
    
    # 2. Initialize Model & Optimization
    model = BaselineUNet(in_channels=1, out_channels=1, base_f=32).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion_bce = nn.BCEWithLogitsLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, epochs * len(train_loader)), eta_min=1e-5
    )
    
    use_amp = (device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    
    # 3. Training Loop
    t_train_start = time.time()
    model.train()
    for ep in range(1, epochs + 1):
        t_ep_start = time.time()
        running_loss = 0.0
        for imgs, masks in train_loader:
            imgs = imgs.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            if use_amp:
                with torch.amp.autocast("cuda"):
                    logits = model(imgs)
                    loss_bce = criterion_bce(logits, masks)
                    loss_dice = soft_dice_loss(torch.sigmoid(logits), masks)
                    total_loss = loss_bce + loss_dice
                scaler.scale(total_loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(imgs)
                loss_bce = criterion_bce(logits, masks)
                loss_dice = soft_dice_loss(torch.sigmoid(logits), masks)
                total_loss = loss_bce + loss_dice
                total_loss.backward()
                optimizer.step()
                
            scheduler.step()
            running_loss += total_loss.item()
            
        epoch_loss = running_loss / max(1, len(train_loader))
        ep_duration = time.time() - t_ep_start
        print(f"Epoch {ep:2d}/{epochs:2d} | Loss: {epoch_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.2e} | Time: {ep_duration:.1f}s")
        
    total_train_time = time.time() - t_train_start
    checkpoint_path = os.path.join(checkpoint_dir, f"baseline_unet_fold{fold}_strat{strategy}.pt")
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Saved model checkpoint to: {checkpoint_path} (Training time: {total_train_time:.1f}s)")
    
    # 4. Out-of-Fold Validation Inference
    print(f"\nRunning out-of-fold inference on {len(val_stems)} validation stems...")
    model.eval()
    pred_rows = []
    t_infer_start = time.time()
    
    with torch.no_grad():
        for stem in val_stems:
            img_path = None
            for ext in [".jpeg", ".jpg", ".png"]:
                candidate = os.path.join(images_dir, stem + ext)
                if os.path.exists(candidate):
                    img_path = candidate
                    break
                    
            if img_path and os.path.exists(img_path):
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            else:
                img = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
                
            orig_h, orig_w = img.shape
            if (orig_h, orig_w) != (target_size, target_size):
                img_in = cv2.resize(img, (target_size, target_size), interpolation=cv2.INTER_AREA)
            else:
                img_in = img
                
            t_in = torch.from_numpy(img_in.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0).to(device)
            if use_amp:
                with torch.amp.autocast("cuda"):
                    logits = model(t_in)
            else:
                logits = model(t_in)
                
            prob_map = torch.sigmoid(logits).squeeze().cpu().numpy()
            
            # Upsample back to 2048x2048 if needed
            if (target_size, target_size) != (HEIGHT, WIDTH):
                full_prob = cv2.resize(prob_map, (WIDTH, HEIGHT), interpolation=cv2.INTER_LINEAR)
            else:
                full_prob = prob_map
                
            # Instance Extraction
            binary_mask = (full_prob >= prob_threshold).astype(np.uint8)
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
            
            inst_count = 0
            for lbl in range(1, num_labels):
                area = stats[lbl, cv2.CC_STAT_AREA]
                if area >= min_area:
                    inst_mask = (labels == lbl).astype(np.uint8)
                    rle_str = mask_to_rle(inst_mask)
                    inst_count += 1
                    pred_rows.append({
                        "filament_id": f"{stem}_{inst_count}",
                        "segmentation_rle": rle_str
                    })
                    
    total_infer_time = time.time() - t_infer_start
    avg_infer_per_img = total_infer_time / max(1, len(val_stems))
    print(f"Inference complete: {len(pred_rows)} instances extracted across {len(val_stems)} stems.")
    print(f"Total inference time: {total_infer_time:.1f}s ({avg_infer_per_img*1000:.1f}ms per image)")
    
    pred_df = pd.DataFrame(pred_rows)
    pred_csv_path = os.path.join(results_dir, f"preds_fold{fold}_strat{strategy}.csv")
    pred_df.to_csv(pred_csv_path, index=False)
    
    # 5. Exact Metric Evaluation Against Ground Truth
    print("\nComputing exact Panoptic Quality metric...")
    gt_df = build_gt_dataframe_for_fold(Path(train_json_path), Path(manifest_csv_path), fold=fold)
    eval_metrics = compute_panoptic_quality_exact(gt_df, pred_df, iou_threshold=0.50, fast_coco=True)
    
    results = {
        "fold": fold,
        "strategy": strategy,
        "pq": eval_metrics["pq"],
        "sq": eval_metrics["sq"],
        "rq": eval_metrics["rq"],
        "tp": eval_metrics["tp"],
        "fp": eval_metrics["fp"],
        "fn": eval_metrics["fn"],
        "num_annotator_views": eval_metrics["num_annotator_views"],
        "total_predicted_instances": len(pred_rows),
        "total_gt_instances": len(gt_df),
        "train_time_seconds": round(total_train_time, 2),
        "infer_time_seconds": round(total_infer_time, 2),
        "infer_ms_per_image": round(avg_infer_per_img * 1000, 2),
        "model_config": {
            "architecture": "BaselineUNet",
            "base_f": 32,
            "target_size": target_size,
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "min_area": min_area,
            "prob_threshold": prob_threshold,
            "seed": seed,
            "device": str(device),
        },
        "checkpoint_path": checkpoint_path,
        "pred_csv_path": pred_csv_path,
    }
    
    print("\n--- Validation Results ---")
    print(f"Panoptic Quality (PQ):     {results['pq']:.5f}")
    print(f"Segmentation Quality (SQ): {results['sq']:.5f}")
    print(f"Recognition Quality (RQ):  {results['rq']:.5f}")
    print(f"True Positives (TP):       {results['tp']}")
    print(f"False Positives (FP):      {results['fp']}")
    print(f"False Negatives (FN):      {results['fn']}")
    print(f"Training Time:             {results['train_time_seconds']}s")
    print(f"Inference Time:            {results['infer_time_seconds']}s")
    print(f"--------------------------\n")
    
    results_json_path = os.path.join(results_dir, f"eval_fold{fold}_strat{strategy}.json")
    with open(results_json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
        
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--strategy", type=str, default="A", choices=["A", "B", "C"])
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--target_size", type=int, default=512)
    parser.add_argument("--train_json", type=str, default="data/annotations/MAGFiLO_1.0_Kaggle_2026/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json")
    parser.add_argument("--manifest", type=str, default="data/train_folds_manifest.csv")
    parser.add_argument("--images_dir", type=str, default="data/train_images")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    run_fold(
        fold=args.fold,
        strategy=args.strategy,
        train_json_path=args.train_json,
        manifest_csv_path=args.manifest,
        images_dir=args.images_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        target_size=args.target_size,
        device_str=args.device,
        seed=args.seed
    )
