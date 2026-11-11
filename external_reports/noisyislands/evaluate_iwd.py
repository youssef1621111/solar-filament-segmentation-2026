"""Evaluate detector and/or refiner model on validation fold using instance-wise Dice.

Usage:
  python evaluate_iwd.py --checkpoint checkpoints/best_resnet34_single_view_fold0.pt
  python evaluate_iwd.py --checkpoint checkpoints/best_resnet34_single_view_iwd_fold0.pt --single-view
  python evaluate_iwd.py --detector checkpoints/best_resnet34_single_view_iwd_fold0.pt --refiner checkpoints/seed_refiner_fold0.pt
"""

import os
import json
import argparse
import numpy as np
import cv2
import torch
from torch.utils.data import DataLoader
import pandas as pd
from tqdm import tqdm

from dataset import SolarFilamentDataset
from model_unet_spine import FilamentSpineUNet, spine_watershed_postprocess
from model_seed_refiner import SeedRefiner
from panoptic_painter import masks_to_submission_rows
from instance_wise_dice import evaluate_instance_wise_dice
from infer_two_stage import (
    square_bounds, merge_overlapping_masks, run_refiner_on_crops,
    select_tta_probability, REFINER_THRESHOLD, REFINER_CROP_SIZE,
)


@torch.no_grad()
def run_detector_validation(model, loader, device, min_area, mask_threshold, spine_threshold):
    """Run detector-only inference and return prediction DataFrame."""
    model.eval()
    pred_rows = []
    for batch in tqdm(loader, desc="Detector inference", leave=False):
        images = batch["image"].to(device, non_blocking=True)
        stems = batch["stem"]
        with torch.amp.autocast(device_type="cuda"):
            preds = model(images)
        for i in range(len(images)):
            stem = stems[i]
            mask_prob = torch.sigmoid(preds[i, 0]).float().cpu().numpy()
            spine_prob = torch.sigmoid(preds[i, 1]).float().cpu().numpy()
            dist_prob = torch.sigmoid(preds[i, 2]).float().cpu().numpy()

            y_grid, x_grid = np.ogrid[:mask_prob.shape[0], :mask_prob.shape[1]]
            dist_center = np.sqrt(
                (x_grid - mask_prob.shape[1] / 2) ** 2 + (y_grid - mask_prob.shape[0] / 2) ** 2
            )
            solar_disk = dist_center <= (0.49 * mask_prob.shape[0])
            mask_prob *= solar_disk
            spine_prob *= solar_disk

            kept = spine_watershed_postprocess(
                mask_prob=mask_prob, spine_prob=spine_prob, dist_prob=dist_prob,
                mask_threshold=mask_threshold, spine_threshold=spine_threshold,
                min_area=max(min_area // 4, 24),
            )
            kept_2048 = [
                (s, cv2.resize(m, (2048, 2048), interpolation=cv2.INTER_NEAREST), int(cv2.resize(m, (2048, 2048), interpolation=cv2.INTER_NEAREST).sum()))
                for s, m, _ in kept
            ]
            img_rows = masks_to_submission_rows(stem, kept_2048)
            pred_rows.extend(img_rows)
    pred_df = pd.DataFrame(pred_rows)
    if len(pred_df) > 0:
        pred_df["stem"] = pred_df["filament_id"].str.rsplit("_", n=1).str[0]
    return pred_df


@torch.no_grad()
def run_two_stage_validation(
    detector, refiner, folds_df, gt_val_df, image_dir, device,
    image_size=1024, min_area=96, mask_threshold=0.48, spine_threshold=0.40,
    refiner_threshold=REFINER_THRESHOLD, batch_size=32, use_tta=True,
):
    """Run two-stage inference on validation fold and compute instance_wise_dice."""
    detector.eval()
    refiner.eval()
    transforms = __import__("solar_transforms").get_valid_transforms(image_size)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    val_files = folds_df[folds_df["fold"] == 0]["file_name"].tolist()  # Caller passes fold-filtered
    # Actually use the stems from gt_val_df
    val_stems = sorted(gt_val_df["stem"].unique())

    pred_rows = []
    stats = {"images": 0, "instances": 0, "refiner_tasks": 0, "tta_accepted": 0, "tta_rejected": 0}

    for stem in tqdm(val_stems, desc="Two-stage validation"):
        filename = f"{stem}.jpeg"
        img_path = os.path.join(image_dir, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        h, w = img.shape

        # Stage 1: Detector (use enhanced image for detector input)
        enhanced = enhance_solar_contrast_fn(img)
        augmented = transforms(image=enhanced)
        input_t = augmented["image"].unsqueeze(0).to(device)
        with torch.amp.autocast(device_type="cuda"):
            preds = detector(input_t)[0]

        mask_prob = torch.sigmoid(preds[0]).float().cpu().numpy()
        spine_prob = torch.sigmoid(preds[1]).float().cpu().numpy()
        dist_prob = torch.sigmoid(preds[2]).float().cpu().numpy()

        y_grid, x_grid = np.ogrid[:mask_prob.shape[0], :mask_prob.shape[1]]
        dist_center = np.sqrt(
            (x_grid - mask_prob.shape[1] / 2) ** 2 + (y_grid - mask_prob.shape[0] / 2) ** 2
        )
        solar_disk = dist_center <= (0.49 * mask_prob.shape[0])
        mask_prob *= solar_disk
        spine_prob *= solar_disk

        kept = spine_watershed_postprocess(
            mask_prob=mask_prob, spine_prob=spine_prob, dist_prob=dist_prob,
            mask_threshold=mask_threshold, spine_threshold=spine_threshold,
            min_area=max(min_area // 4, 24),
        )

        coarse_instances = []
        for score, mask_1024, _ in kept:
            mask_2048 = cv2.resize(mask_1024, (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
            if mask_2048.sum() >= min_area:
                coarse_instances.append((score, mask_2048))

        if not coarse_instances:
            continue

        merged_masks = merge_overlapping_masks([m for _, m in coarse_instances])
        crops = []
        crop_info = []
        for mask in merged_masks:
            bounds = square_bounds(mask.astype(np.uint8))
            x0, y0, x1, y1 = bounds
            image_crop = cv2.resize(img[y0:y1, x0:x1], (REFINER_CROP_SIZE, REFINER_CROP_SIZE), interpolation=cv2.INTER_LINEAR)
            clahe_crop = clahe.apply(image_crop)
            seed_crop = cv2.resize(mask[y0:y1, x0:x1].astype(np.uint8), (REFINER_CROP_SIZE, REFINER_CROP_SIZE), interpolation=cv2.INTER_NEAREST)
            features = np.stack([image_crop, clahe_crop, seed_crop * 255], axis=0).astype(np.float32) / 255.0
            crops.append(features)
            crop_info.append((bounds, mask))

        if not crops:
            continue

        stats["refiner_tasks"] += len(crops)
        crops_arr = np.stack(crops)

        if use_tta:
            single_probs = run_refiner_on_crops(refiner, crops_arr, device, use_tta=False)
            tta_probs = run_refiner_on_crops(refiner, crops_arr, device, use_tta=True)
            final_probs = np.stack([
                select_tta_probability(single, tta, refiner_threshold)
                for single, tta in zip(single_probs, tta_probs)
            ])
            for single, tta in zip(single_probs, tta_probs):
                if np.array_equal(single, tta):
                    stats["tta_rejected"] += 1
                else:
                    stats["tta_accepted"] += 1
        else:
            final_probs = run_refiner_on_crops(refiner, crops_arr, device, use_tta=False)

        candidates = []
        for i, (bounds, seed_mask) in enumerate(crop_info):
            x0, y0, x1, y1 = bounds
            prob = final_probs[i]
            crop_mask = cv2.resize(prob.astype(np.float32), (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR) >= refiner_threshold
            if crop_mask.sum() == 0:
                continue
            full = np.zeros((h, w), dtype=bool)
            full[y0:y1, x0:x1] = crop_mask
            refined = full & seed_mask
            if refined.sum() == 0:
                refined = full
            score = float(prob[crop_mask > 0].mean()) if crop_mask.sum() > 0 else 0.0
            candidates.append((score, refined))

        # Single CC enforcement (keep largest component)
        cc_candidates = []
        for score, mask in candidates:
            num_labels, labels = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
            if num_labels <= 1:
                cc_candidates.append((score, mask))
            else:
                areas = [(labels == i).sum() for i in range(1, num_labels)]
                largest = np.argmax(areas) + 1
                cc_candidates.append((score, labels == largest))

        cc_candidates.sort(key=lambda x: -x[0])
        occupied = np.zeros((h, w), dtype=bool)
        inst_idx = 1
        for score, mask in cc_candidates:
            bool_m = mask & (~occupied)
            if int(bool_m.sum()) >= min_area:
                occupied |= bool_m
                from panoptic_painter import encode_mask_to_rle
                rle_str = encode_mask_to_rle(bool_m.astype(np.uint8))
                pred_rows.append({"filament_id": f"{stem}_{inst_idx}", "segmentation_rle": rle_str, "stem": stem})
                stats["instances"] += 1
                inst_idx += 1

        stats["images"] += 1

    pred_df = pd.DataFrame(pred_rows)
    print(f"\nValidation stats: {stats}")
    return pred_df


def enhance_solar_contrast_fn(img, clip_limit=2.0, tile_grid_size=(8, 8)):
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(img)


def main():
    parser = argparse.ArgumentParser(description="Evaluate instance-wise Dice on validation fold")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--detector_ckpt", type=str, default=None)
    parser.add_argument("--refiner_ckpt", type=str, default=None)
    parser.add_argument("--backbone", type=str, default="resnet34")
    parser.add_argument("--single_view", action="store_true")
    parser.add_argument("--image_size", type=int, default=1024)
    parser.add_argument("--min_area", type=int, default=96)
    parser.add_argument("--mask_threshold", type=float, default=0.48)
    parser.add_argument("--spine_threshold", type=float, default=0.40)
    parser.add_argument("--refiner_threshold", type=float, default=REFINER_THRESHOLD)
    parser.add_argument("--no_tta", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load GT
    if args.single_view:
        gt_df = pd.read_parquet("processed/gt_single_view.parquet")
        folds_df = pd.read_csv("processed/folds_single_view.csv")
        from single_view_gt import select_medoid_annotators
        coco_path = "data/MAGFiLO_1.0_Kaggle_2026/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json"
        with open(coco_path) as f:
            coco = json.load(f)
        from collections import defaultdict
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
    else:
        gt_df = pd.read_parquet("processed/gt_validation_df.parquet")
        folds_df = pd.read_csv("processed/folds.csv")
        selected_image_ids = None

    gt_val_df = gt_df[gt_df["fold"] == args.fold].reset_index(drop=True)
    val_meta = folds_df[folds_df["fold"] == args.fold].reset_index(drop=True)
    print(f"Fold {args.fold}: {len(val_meta)} images, {len(gt_val_df)} GT instances")

    # Load dataset for detector path
    val_dataset = SolarFilamentDataset(
        val_meta,
        image_size=args.image_size, is_train=False,
        single_view=args.single_view, selected_image_ids=selected_image_ids,
    )
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=4, pin_memory=True)

    if args.detector_ckpt and not args.refiner_ckpt:
        # Detector-only evaluation
        ckpt = torch.load(args.detector_ckpt, map_location=device, weights_only=False)
        model = FilamentSpineUNet(
            backbone=ckpt.get("backbone", args.backbone), pretrained=False, in_channels=1, classes=3
        ).to(device)
        model.load_state_dict(ckpt["model_state"])
        pred_df = run_detector_validation(
            model, val_loader, device, args.min_area,
            args.mask_threshold, args.spine_threshold,
        )
    elif args.detector_ckpt and args.refiner_ckpt:
        # Two-stage evaluation
        det_ckpt = torch.load(args.detector_ckpt, map_location=device, weights_only=False)
        detector = FilamentSpineUNet(
            backbone=det_ckpt.get("backbone", args.backbone), pretrained=False, in_channels=1, classes=3
        ).to(device)
        detector.load_state_dict(det_ckpt["model_state"])

        ref_ckpt = torch.load(args.refiner_ckpt, map_location=device, weights_only=False)
        refiner = SeedRefiner(pretrained=False).to(device)
        refiner.load_state_dict(ref_ckpt["model_state"])

        image_dir = "/home/dead/filament-segmentation/data/MAGFiLO_1.0_Kaggle_2026/train/train_images"
        pred_df = run_two_stage_validation(
            detector, refiner, val_meta, gt_val_df, image_dir, device,
            image_size=args.image_size, min_area=args.min_area,
            mask_threshold=args.mask_threshold, spine_threshold=args.spine_threshold,
            refiner_threshold=args.refiner_threshold,
            use_tta=not args.no_tta,
        )
    else:
        raise ValueError("Must specify at least --detector_ckpt or both --detector_ckpt and --refiner_ckpt")

    # Evaluate with instance_wise_dice
    if "stem" not in pred_df.columns and len(pred_df) > 0:
        pred_df["stem"] = pred_df["filament_id"].str.rsplit("_", n=1).str[0]
    if "stem" not in gt_val_df.columns:
        gt_val_df = gt_val_df.copy()
        if args.single_view:
            gt_val_df["stem"] = gt_val_df["filament_id"].str.split("-").str[1].str.rsplit("_", n=1).str[0]
        else:
            gt_val_df["stem"] = gt_val_df["filament_id"].str.split("_", n=1).str[0].str.split("-", n=1).str[1]

    results = evaluate_instance_wise_dice(gt_val_df, pred_df)
    print(f"\n=== Instance-Wise Dice Evaluation ===")
    print(f"  Instance-wise Dice: {results['instance_wise_dice']:.4f}")
    print(f"  Mean per-image Dice: {results['mean_per_image_dice']:.4f}")
    print(f"  TP: {results['tp']}, FP: {results['fp']}, FN: {results['fn']}")
    print(f"  Images: {results['n_images']}, Total GT: {results['n_gt']}")
    print(f"  Predictions: {len(pred_df)}")


if __name__ == "__main__":
    from solar_transforms import enhance_solar_contrast
    main()
