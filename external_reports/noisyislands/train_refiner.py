"""Train the SeedRefiner (mask-conditioned U-Net) on per-instance crops.

Stage 2 of the two-stage pipeline: detector → refiner.
Trains on single-view GT (medoid annotator) crops.
Uses BCE + soft-Dice loss, AdamW, cosine LR schedule.
"""

import os
import argparse
import time
import math
import json
import numpy as np
import torch
import cv2
import pandas as pd
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from dataset_refiner import RefinerDataset, build_coco_indices
from model_seed_refiner import SeedRefiner, BCEPlusDiceLoss
from single_view_gt import select_medoid_annotators, build_single_view_targets
from instance_wise_dice import evaluate_instance_wise_dice
from panoptic_painter import masks_to_submission_rows


def train_epoch(model, loader, criterion, optimizer, scaler, accum_steps, device):
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()
    pbar = tqdm(loader, desc="Training", leave=False)
    for step, batch in enumerate(pbar):
        features, targets = batch
        features = features.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.amp.autocast(device_type="cuda"):
            logits = model(features)
            loss = criterion(logits, targets) / accum_steps

        scaler.scale(loss).backward()

        if (step + 1) % accum_steps == 0 or (step + 1) == len(loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_loss += loss.item() * accum_steps
        pbar.set_postfix(loss=f"{loss.item() * accum_steps:.4f}")

    return total_loss / max(len(loader), 1)


@torch.no_grad()
def evaluate_refiner(model, val_loader, device):
    model.eval()
    total_dice = 0.0
    total_samples = 0
    pbar = tqdm(val_loader, desc="Validating", leave=False)
    for features, targets in pbar:
        features = features.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with torch.amp.autocast(device_type="cuda"):
            logits = model(features)
        probs = logits.sigmoid()
        dice_scores = []
        for i in range(probs.shape[0]):
            p = probs[i, 0].cpu().numpy()
            t = targets[i, 0].cpu().numpy()
            inter = np.logical_and(p > 0.5, t > 0.5).sum()
            dice = (2 * inter + 1e-7) / (p.sum() + t.sum() + 1e-7)
            dice_scores.append(dice)
        total_dice += sum(dice_scores)
        total_samples += len(dice_scores)
    return total_dice / max(total_samples, 1)


def square_bounds(mask: np.ndarray, context: float = 1.8, minimum: int = 96) -> tuple[int, int, int, int]:
    height, width = mask.shape
    ys, xs = np.where(mask)
    if len(xs) == 0:
        cx, cy = width // 2, height // 2
        side = min(min(height, width), minimum)
    else:
        cx = (xs.min() + xs.max() + 1) / 2
        cy = (ys.min() + ys.max() + 1) / 2
        side = int(math.ceil(max(xs.max() - xs.min() + 1, ys.max() - ys.min() + 1) * context))
    side = min(max(side, minimum), min(height, width))
    x0 = int(round(cx - side / 2))
    y0 = int(round(cy - side / 2))
    x1, y1 = x0 + side, y0 + side
    if x0 < 0:
        x1 -= x0; x0 = 0
    if y0 < 0:
        y1 -= y0; y0 = 0
    if x1 > width:
        x0 -= x1 - width; x1 = width
    if y1 > height:
        y0 -= y1 - height; y1 = height
    return int(x0), int(y0), int(x1), int(y1)


def run_refiner_val(
    model, val_stems, gt_val_df, image_dir, indices, selected_image_ids,
    crop_size=256, context=1.8, min_crop=96, device="cuda",
    refiner_threshold=0.70,
):
    """Run refiner-based inference on validation images and compute instance_wise_dice.

    This simulates test-time behavior: for each GT instance, we create a seed mask
    (jittered version of GT), run the refiner, then threshold and paint.
    """
    from dataset_refiner import jitter_seed_mask
    import random
    import pycocotools.mask as mask_utils
    from scipy.optimize import linear_sum_assignment

    model.eval()
    pred_rows = []

    rng = random.Random(42)

    for stem in val_stems:
        filename = f"{stem}.jpeg"
        img_path = os.path.join(image_dir, filename)
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue

        h, w = image.shape

        # Get GT instances for this image
        gt_instances = gt_val_df[gt_val_df["stem"] == stem]
        if len(gt_instances) == 0:
            continue

        # For validation, use exact GT masks as seeds (best case) with small jitter
        seeds = []
        for rle_str in gt_instances["segmentation_rle"]:
            rle_dict = {"size": [h, w], "counts": rle_str.encode("ascii") if isinstance(rle_str, str) else rle_str}
            gt_mask = mask_utils.decode(rle_dict).astype(bool)
            j_rng = np.random.default_rng(rng.randint(0, 1_000_000))
            seed = jitter_seed_mask(gt_mask, j_rng)
            seeds.append(seed)

        if not seeds:
            continue

        # Extract crops
        crops = []
        bounds_list = []
        for seed in seeds:
            bounds = square_bounds(seed, context, min_crop)
            x0, y0, x1, y1 = bounds
            crop = cv2.resize(
                image[y0:y1, x0:x1], (crop_size, crop_size), interpolation=cv2.INTER_LINEAR
            )
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(crop)
            seed_crop = cv2.resize(
                seed[y0:y1, x0:x1].astype(np.uint8), (crop_size, crop_size), interpolation=cv2.INTER_NEAREST
            )
            features = np.stack([crop, clahe, seed_crop * 255], axis=0).astype(np.float32) / 255.0
            crops.append(features)
            bounds_list.append(bounds)

        if not crops:
            continue

        # Flip-4 TTA on crops
        crops_arr = np.stack(crops)
        tta_probs = np.stack([crops_arr, crops_arr[:, :, :, ::-1], crops_arr[:, :, ::-1, :], crops_arr[:, :, ::-1, ::-1]])

        with torch.no_grad():
            batch = torch.from_numpy(
                np.ascontiguousarray(tta_probs.reshape(-1, 3, crop_size, crop_size))
            ).to(device, non_blocking=True)
            with torch.amp.autocast(device_type="cuda"):
                logits = model(batch)
            probs = logits.sigmoid().cpu().numpy().reshape(
                4, len(crops), 1, crop_size, crop_size
            )[:, :, 0]

        # Invert TTA
        probs = (
            probs[0]
            + np.ascontiguousarray(probs[1, :, :, ::-1])
            + np.ascontiguousarray(probs[2, :, ::-1, :])
            + np.ascontiguousarray(probs[3, :, ::-1, ::-1])
        ) / 4.0

        # Decode each crop back to full image
        candidates = []
        for i, (bounds, prob) in enumerate(zip(bounds_list, probs)):
            x0, y0, x1, y1 = bounds
            prob_resized = cv2.resize(
                prob.astype(np.float32), (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR
            )
            crop_mask = prob_resized >= refiner_threshold
            if crop_mask.sum() == 0:
                continue
            full = np.zeros((h, w), dtype=np.uint8)
            full[y0:y1, x0:x1] = crop_mask.astype(np.uint8)
            mask_bool = full.astype(bool)
            score = float(prob_resized[crop_mask].mean())
            candidates.append((score, mask_bool))

        # Score-ordered panoptic painting
        candidates.sort(key=lambda x: -x[0])
        occupied = np.zeros((h, w), dtype=bool)
        inst_idx = 1
        for score, mask in candidates:
            bool_m = mask & (~occupied)
            if int(bool_m.sum()) >= 96:
                occupied |= bool_m
                pred_rows.append({
                    "filament_id": f"{stem}_{inst_idx}",
                    "segmentation_rle": "",
                    "stem": stem,
                })
                # Store mask reference for RLE encoding
                pred_rows[-1]["_mask"] = bool_m
                inst_idx += 1

    # Encode masks to RLE
    H, W = 2048, 2048
    for row in pred_rows:
        if "_mask" in row:
            m = row.pop("_mask")
            rle = mask_utils.encode(np.asfortranarray(m.astype(np.uint8)))
            row["segmentation_rle"] = rle["counts"].decode("ascii") if isinstance(rle["counts"], bytes) else rle["counts"]

    pred_df = pd.DataFrame(pred_rows)
    if "_mask" in pred_df.columns:
        del pred_df["_mask"]

    return pred_df


def run_training(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data_dir = "/home/dead/filament-segmentation/data/MAGFiLO_1.0_Kaggle_2026"
    coco_path = os.path.join(data_dir, "train/MAGFiLO_1.0_Annotations_kaggle2026_train.json")
    image_dir = os.path.join(data_dir, "train/train_images")

    indices = build_coco_indices(coco_path)

    # Load folds and GT
    folds_df = pd.read_csv("processed/folds_single_view.csv")
    gt_df = pd.read_parquet("processed/gt_single_view.parquet")

    # Select medoid annotators
    selected_image_ids = select_medoid_annotators(
        indices["image_ids_by_file"],
        indices["annotations_by_image"],
        indices["images_by_id"],
    )

    train_files = folds_df[folds_df["fold"] != args.fold]["file_name"].tolist()
    val_files = folds_df[folds_df["fold"] == args.fold]["file_name"].tolist()
    val_stems = [os.path.splitext(f)[0] for f in val_files]

    gt_val_df = gt_df[gt_df["fold"] == args.fold].reset_index(drop=True)
    print(f"  Train files: {len(train_files)}, Val files: {len(val_files)}, Val GT instances: {len(gt_val_df)}")

    train_dataset = RefinerDataset(
        filenames=train_files,
        image_dir=image_dir,
        annotations_by_image_id=indices["annotations_by_image"],
        images_by_id=indices["images_by_id"],
        selected_image_ids=selected_image_ids,
        crop_size=args.crop_size,
        context=args.context,
        min_crop=args.min_crop,
        augment=True,
        seed=args.seed,
    )
    val_dataset = RefinerDataset(
        filenames=val_files,
        image_dir=image_dir,
        annotations_by_image_id=indices["annotations_by_image"],
        images_by_id=indices["images_by_id"],
        selected_image_ids=selected_image_ids,
        crop_size=args.crop_size,
        context=args.context,
        min_crop=args.min_crop,
        augment=False,
        seed=args.seed,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=4, pin_memory=True,
    )

    model = SeedRefiner(pretrained=True).to(device)
    criterion = BCEPlusDiceLoss(bce_weight=1.0, dice_weight=1.0).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler(device="cuda")

    os.makedirs(args.save_dir, exist_ok=True)
    checkpoint_path = os.path.join(args.save_dir, f"seed_refiner_fold{args.fold}.pt")

    best_iwd = 0.0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_epoch(model, train_loader, criterion, optimizer, scaler, args.accum_steps, device)
        scheduler.step()

        val_dice = evaluate_refiner(model, val_loader, device)

        # Run full instance-wise Dice evaluation
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            pred_df = run_refiner_val(
                model, val_stems, gt_val_df, image_dir, indices, selected_image_ids,
                crop_size=args.crop_size, context=args.context, min_crop=args.min_crop,
                device=device, refiner_threshold=args.threshold,
            )
            if len(pred_df) > 0:
                pred_df["stem"] = pred_df["filament_id"].str.rsplit("_", n=1).str[0]
            results = evaluate_instance_wise_dice(gt_val_df, pred_df)
            iwd = results["instance_wise_dice"]
        else:
            iwd = val_dice

        dt = time.time() - t0
        tp_str = str(results.get("tp", 0)) if "results" in dir() else "n/a"
        fp_str = str(results.get("fp", 0)) if "results" in dir() else "n/a"
        fn_str = str(results.get("fn", 0)) if "results" in dir() else "n/a"
        print(
            f"Epoch [{epoch:02d}/{args.epochs:02d}] ({dt:.1f}s) - "
            f"Loss: {train_loss:.4f} | Val Dice: {val_dice:.4f} | "
            f"IWD: {iwd:.4f} (TP: {tp_str}, FP: {fp_str}, FN: {fn_str})"
        )

        if epoch % args.eval_every == 0 or epoch == args.epochs:
            if iwd > best_iwd:
                best_iwd = iwd
                torch.save({
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "instance_wise_dice": iwd,
                    "fold": args.fold,
                    "crop_size": args.crop_size,
                }, checkpoint_path)
                print(f"  >>> New Best Instance-wise Dice: {best_iwd:.4f}")

    print(f"\nTraining Complete. Best IWD on Fold {args.fold}: {best_iwd:.4f}")
    return best_iwd


def parse_args():
    parser = argparse.ArgumentParser(description="Train SeedRefiner on filament crops")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--accum_steps", type=int, default=1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--crop_size", type=int, default=256)
    parser.add_argument("--context", type=float, default=1.8)
    parser.add_argument("--min_crop", type=int, default=96)
    parser.add_argument("--threshold", type=float, default=0.70)
    parser.add_argument("--eval_every", type=int, default=2)
    parser.add_argument("--save_dir", type=str, default="/home/dead/filament-segmentation/checkpoints")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    run_training(parse_args())
