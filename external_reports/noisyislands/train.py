import os
import argparse
import time
import pandas as pd
import numpy as np
import torch
import cv2
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from dataset import SolarFilamentDataset
from model_unet_spine import FilamentSpineUNet, FilamentCompositeLoss, spine_watershed_postprocess
from panoptic_painter import masks_to_submission_rows, check_no_overlap
from metrics import evaluate_predictions
from instance_wise_dice import evaluate_instance_wise_dice

def train_epoch(model, loader, criterion, optimizer, scaler, accum_steps=2, device="cuda"):
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()

    pbar = tqdm(loader, desc="Training", leave=False)
    for step, batch in enumerate(pbar):
        images = batch["image"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)

        with torch.amp.autocast(device_type="cuda"):
            preds = model(images)
            loss, stats = criterion(preds, targets)
            loss = loss / accum_steps

        scaler.scale(loss).backward()

        if (step + 1) % accum_steps == 0 or (step + 1) == len(loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_loss += loss.item() * accum_steps
        pbar.set_postfix(loss=f"{loss.item() * accum_steps:.4f}")

    return total_loss / len(loader)

@torch.no_grad()
def evaluate_validation(model, loader, gt_val_df, device="cuda", min_area=150, metric="pq", image_size=1024):
    model.eval()
    pred_rows = []

    for batch in tqdm(loader, desc="Validating", leave=False):
        images = batch["image"].to(device, non_blocking=True)
        stems = batch["stem"]

        with torch.amp.autocast(device_type="cuda"):
            preds = model(images)

        for i in range(len(images)):
            stem = stems[i]
            mask_prob = torch.sigmoid(preds[i, 0]).float().cpu().numpy()
            spine_prob = torch.sigmoid(preds[i, 1]).float().cpu().numpy()
            dist_prob = torch.sigmoid(preds[i, 2]).float().cpu().numpy()

            # Mask out dark space outside the solar disk
            y_grid, x_grid = np.ogrid[:mask_prob.shape[0], :mask_prob.shape[1]]
            dist_center = np.sqrt(
                (x_grid - mask_prob.shape[1] / 2) ** 2 + (y_grid - mask_prob.shape[0] / 2) ** 2
            )
            solar_disk = dist_center <= (0.49 * mask_prob.shape[0])
            mask_prob = mask_prob * solar_disk
            spine_prob = spine_prob * solar_disk

            # Extract disjoint instances via spine-watershed
            kept_instances = spine_watershed_postprocess(
                mask_prob=mask_prob,
                spine_prob=spine_prob,
                dist_prob=dist_prob,
                mask_threshold=0.48,
                spine_threshold=0.40,
                min_area=max(min_area // 4, 45),
            )

            # Upsample to 2048x2048
            kept_instances_2048 = []
            for score, mask_1024, _ in kept_instances:
                mask_2048 = cv2.resize(mask_1024, (2048, 2048), interpolation=cv2.INTER_NEAREST)
                kept_instances_2048.append((score, mask_2048, int(mask_2048.sum())))

            img_rows = masks_to_submission_rows(stem, kept_instances_2048)
            pred_rows.extend(img_rows)

    pred_df = pd.DataFrame(pred_rows)
    if pred_df.empty:
        if metric == "iwd":
            return {"instance_wise_dice": 0.0, "tp": 0, "fp": 0, "fn": len(gt_val_df)}, pred_df
        return {"pq": 0.0, "sq": 0.0, "rq": 0.0, "tp": 0, "fp": 0, "fn": len(gt_val_df)}, pred_df

    if metric == "iwd":
        if "stem" not in pred_df.columns:
            pred_df["stem"] = pred_df["filament_id"].str.rsplit("_", n=1).str[0]
        metrics = evaluate_instance_wise_dice(gt_val_df, pred_df)
    else:
        metrics = evaluate_predictions(gt_val_df, pred_df)

    return metrics, pred_df

def run_training(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    os.makedirs(args.save_dir, exist_ok=True)

    # Use single-view GT if requested
    if args.single_view:
        folds_path = os.path.join(args.data_dir, "processed/folds_single_view.csv")
        gt_path = os.path.join(args.data_dir, "processed/gt_single_view.parquet")
        from single_view_gt import select_medoid_annotators
        import json
        from collections import defaultdict

        coco_path = os.path.join(args.data_dir, "data/MAGFiLO_1.0_Kaggle_2026/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json")
        with open(coco_path, "r") as f:
            coco = json.load(f)
        images_by_id = {str(img["id"]): img for img in coco["images"]}
        annotations_by_image = defaultdict(list)
        for ann in coco["annotations"]:
            annotations_by_image[str(ann["image_id"])].append(ann)
        image_ids_by_file = defaultdict(list)
        for img in coco["images"]:
            image_ids_by_file[img["file_name"]].append(img["id"])
        for fname in image_ids_by_file:
            image_ids_by_file[fname].sort()

        selected_image_ids = select_medoid_annotators(
            image_ids_by_file, annotations_by_image, images_by_id
        )
        print(f"Single-view GT: selected {len(selected_image_ids)} medoid annotators")
    else:
        folds_path = os.path.join(args.data_dir, "processed/folds.csv")
        gt_path = os.path.join(args.data_dir, "processed/gt_validation_df.parquet")
        selected_image_ids = None

    folds_df = pd.read_csv(folds_path)
    gt_df = pd.read_parquet(gt_path)

    train_meta = folds_df[folds_df["fold"] != args.fold].reset_index(drop=True)
    val_meta = folds_df[folds_df["fold"] == args.fold].reset_index(drop=True)
    gt_val_df = gt_df[gt_df["fold"] == args.fold].reset_index(drop=True)

    print(f"--- FOLD {args.fold} ---")
    print(f"Train images: {len(train_meta)}, Val images: {len(val_meta)}, Val annotations: {len(gt_val_df)}")

    # Create Datasets
    train_dataset = SolarFilamentDataset(
        train_meta,
        data_dir=os.path.join(args.data_dir, "data/MAGFiLO_1.0_Kaggle_2026"),
        image_size=args.image_size,
        is_train=True,
        single_view=args.single_view,
        selected_image_ids=selected_image_ids,
    )
    val_dataset = SolarFilamentDataset(
        val_meta,
        data_dir=os.path.join(args.data_dir, "data/MAGFiLO_1.0_Kaggle_2026"),
        image_size=args.image_size,
        is_train=False,
        single_view=args.single_view,
        selected_image_ids=selected_image_ids,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # Initialize Model & Optimizer
    model = FilamentSpineUNet(backbone=args.backbone, pretrained=True, in_channels=1, classes=3).to(device)
    criterion = FilamentCompositeLoss().to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler(device="cuda")

    best_pq = 0.0
    sv_tag = "_single_view" if args.single_view else ""
    metric_tag = "_iwd" if args.metric == "iwd" else ""
    checkpoint_path = os.path.join(args.save_dir, f"best_{args.backbone}{sv_tag}{metric_tag}_fold{args.fold}.pt")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_epoch(
            model, train_loader, criterion, optimizer, scaler, accum_steps=args.accum_steps, device=device
        )
        scheduler.step()

        val_metrics, val_pred_df = evaluate_validation(
            model, val_loader, gt_val_df, device=device, min_area=args.min_area,
            metric=args.metric, image_size=args.image_size,
        )
        dt = time.time() - t0

        if args.metric == "iwd":
            primary = val_metrics["instance_wise_dice"]
            tp = val_metrics["tp"]
            fp = val_metrics["fp"]
            fn = val_metrics["fn"]
            print(
                f"Epoch [{epoch:02d}/{args.epochs:02d}] ({dt:.1f}s) - "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Instance-wise Dice: {primary:.4f} | "
                f"TP: {tp}, FP: {fp}, FN: {fn}"
            )
            if primary > best_pq:
                best_pq = primary
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "val_metrics": val_metrics,
                        "backbone": args.backbone,
                        "fold": args.fold,
                        "image_size": args.image_size,
                        "single_view": args.single_view,
                        "metric": args.metric,
                    },
                    checkpoint_path,
                )
                val_pred_df.to_parquet(
                    os.path.join(args.save_dir, f"oof_pred_{args.backbone}_fold{args.fold}.parquet"),
                    index=False,
                )
                print(f"  >>> New Best IWD: {best_pq:.4f}! Saved checkpoint to {checkpoint_path}")
        else:
            pq = val_metrics["pq"]
            sq = val_metrics["sq"]
            rq = val_metrics["rq"]
            tp = val_metrics["tp"]
            fp = val_metrics["fp"]
            fn = val_metrics["fn"]
            print(
                f"Epoch [{epoch:02d}/{args.epochs:02d}] ({dt:.1f}s) - "
                f"Train Loss: {train_loss:.4f} | "
                f"Val PQ: {pq:.4f} (SQ: {sq:.4f}, RQ: {rq:.4f}) | "
                f"TP: {tp}, FP: {fp}, FN: {fn}"
            )
            if pq > best_pq:
                best_pq = pq
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "val_metrics": val_metrics,
                        "backbone": args.backbone,
                        "fold": args.fold,
                        "image_size": args.image_size,
                        "single_view": args.single_view,
                        "metric": args.metric,
                    },
                    checkpoint_path,
                )
                val_pred_df.to_parquet(
                    os.path.join(args.save_dir, f"oof_pred_{args.backbone}_fold{args.fold}.parquet"),
                    index=False,
                )
                print(f"  >>> New Best PQ: {best_pq:.4f}! Saved checkpoint to {checkpoint_path}")

    metric_name = "Instance-wise Dice" if args.metric == "iwd" else "PQ"
    print(f"\nTraining Complete. Best Validation {metric_name} on Fold {args.fold}: {best_pq:.4f}")
    return best_pq

def parse_args():
    parser = argparse.ArgumentParser(description="Train FilamentSpineUNet on GONG Solar Images")
    parser.add_argument("--fold", type=int, default=0, help="Validation fold index (0-4)")
    parser.add_argument("--backbone", type=str, default="resnet34", help="Encoder backbone name")
    parser.add_argument("--epochs", type=int, default=12, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size per step")
    parser.add_argument("--accum_steps", type=int, default=2, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=3e-4, help="Initial learning rate")
    parser.add_argument("--image_size", type=int, default=1024, help="Input image size")
    parser.add_argument("--min_area", type=int, default=150, help="Minimum filament area in pixels")
    parser.add_argument("--data_dir", type=str, default="/home/dead/filament-segmentation", help="Project root dir")
    parser.add_argument("--save_dir", type=str, default="/home/dead/filament-segmentation/checkpoints", help="Save dir")
    parser.add_argument("--single_view", action="store_true", help="Use single-view (medoid) GT instead of consensus")
    parser.add_argument("--metric", type=str, default="pq", choices=["pq", "iwd"], help="Validation metric")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run_training(args)
