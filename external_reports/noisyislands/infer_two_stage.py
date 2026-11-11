"""Two-stage inference: Detector (UNet) → SeedRefiner (U-Net refiner) with flip-4 TTA.

Pipeline:
1. Run detector UNet to get coarse mask + spine probability maps
2. Spine-watershed to extract coarse instances (seeds)
3. For each instance crop: [image, CLAHE, seed_mask] → SeedRefiner
4. Flip-4 TTA (identity, hflip, vflip, hvflip) on refiner crops
5. Topology-safe acceptance (area/overlap/components guards)
6. Score-ordered panoptic painting → COCO RLE submission

This replaces the single-stage approach and adds:
- Mask-conditioned refiner for boundary refinement
- Flip-4 TTA on crops (not full image, for memory efficiency)
- Single-view GT-aware evaluation
"""

import os
import glob
import argparse
import math
import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import pycocotools.mask as mask_utils
from scipy.optimize import linear_sum_assignment
import pandas as pd

from solar_transforms import enhance_solar_contrast, get_valid_transforms
from model_unet_spine import FilamentSpineUNet, spine_watershed_postprocess
from model_seed_refiner import SeedRefiner
from panoptic_painter import masks_to_submission_rows, check_no_overlap, encode_mask_to_rle
from instance_wise_dice import evaluate_instance_wise_dice


DETECTOR_THRESHOLD = 0.48
SPINE_THRESHOLD = 0.40
DETECTOR_SCORE_THRESHOLD = 0.60
REFINER_CROP_SIZE = 256
REFINER_CONTEXT = 1.8
REFINER_MIN_CROP = 96
REFINER_THRESHOLD = 0.70
TTA_AREA_RATIO_MIN = 0.60
TTA_AREA_RATIO_MAX = 1.70
TTA_DICE_MIN = 0.62
TTA_MIN_CONTROL_AREA_FOR_DICE = 32
TTA_TRANSFORMS = ("identity", "hflip", "vflip", "hvflip")


def square_bounds(mask: np.ndarray, context: float = REFINER_CONTEXT, minimum: int = REFINER_MIN_CROP) -> tuple[int, int, int, int]:
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


def component_count(mask: np.ndarray) -> int:
    return max(cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)[0] - 1, 0)


def binary_dice(first: np.ndarray, second: np.ndarray) -> float:
    intersection = np.logical_and(first, second).sum()
    return float((2 * intersection + 1e-7) / (first.sum() + second.sum() + 1e-7))


def merge_overlapping_masks(masks: list[np.ndarray]) -> list[np.ndarray]:
    """Merge masks that overlap (IoU > 0) into groups, union each group."""
    if len(masks) <= 1:
        return masks
    rles = [
        mask_utils.encode(np.asfortranarray(m.astype(np.uint8)))
        for m in masks
    ]
    adjacency = np.asarray(
        mask_utils.iou(rles, rles, [0] * len(rles)),
        dtype=np.float64,
    ) > 0
    groups: list[list[int]] = []
    seen: set[int] = set()
    for start in range(len(masks)):
        if start in seen:
            continue
        stack = [start]
        group: list[int] = []
        while stack:
            idx = stack.pop()
            if idx in seen:
                continue
            seen.add(idx)
            group.append(idx)
            stack.extend(np.flatnonzero(adjacency[idx]).tolist())
        groups.append(group)
    result = []
    for group in groups:
        if len(group) == 1:
            result.append(masks[group[0]])
        else:
            merged = np.logical_or.reduce([masks[i] for i in group])
            result.append(merged.astype(bool))
    return result


def run_refiner_on_crops(
    refiner: nn.Module,
    crops: np.ndarray,
    device: torch.device,
    use_tta: bool = True,
) -> np.ndarray:
    """Run SeedRefiner on crop array with optional flip-4 TTA.

    Args:
        crops: [N, 3, 256, 256] float32 array (image, CLAHE, seed as 0-1)
        device: torch device
        use_tta: Whether to apply flip-4 TTA

    Returns:
        [N, 256, 256] float32 probability array
    """
    import torch.nn as nn  # noqa
    n = len(crops)
    if not use_tta:
        batch = torch.from_numpy(crops.astype(np.float32)).to(device)
        with torch.no_grad(), torch.amp.autocast(device_type="cuda"):
            probs = refiner(batch).sigmoid().cpu().numpy()[:, 0]
        return np.ascontiguousarray(probs, dtype=np.float32)

    # Flip-4 TTA
    transforms = ["identity", "hflip", "vflip", "hvflip"]
    all_probs = []
    for t in transforms:
        if t == "identity":
            t_crops = crops
        elif t == "hflip":
            t_crops = crops[:, :, :, ::-1]
        elif t == "vflip":
            t_crops = crops[:, :, ::-1, :]
        elif t == "hvflip":
            t_crops = crops[:, :, ::-1, ::-1]
        t_crops = np.ascontiguousarray(t_crops, dtype=np.float32)
        batch = torch.from_numpy(t_crops).to(device)
        with torch.no_grad(), torch.amp.autocast(device_type="cuda"):
            probs = refiner(batch).sigmoid().cpu().numpy()[:, 0]
        if t == "hflip":
            probs = probs[:, :, ::-1]
        elif t == "vflip":
            probs = probs[:, ::-1, :]
        elif t == "hvflip":
            probs = probs[:, ::-1, ::-1]
        all_probs.append(probs)

    mean_probs = np.mean(all_probs, axis=0)
    return np.ascontiguousarray(mean_probs, dtype=np.float32)


def select_tta_probability(
    single_prob: np.ndarray,
    tta_prob: np.ndarray,
    threshold: float = REFINER_THRESHOLD,
) -> np.ndarray:
    """Apply topology-safe TTA acceptance (from lb-0-70 notebook)."""
    single_mask = single_prob >= threshold
    tta_mask = tta_prob >= threshold
    single_area = int(single_mask.sum())
    tta_area = int(tta_mask.sum())
    if tta_area == 0:
        return single_prob
    if single_area == 0:
        return single_prob
    area_ratio = tta_area / max(single_area, 1)
    if area_ratio < TTA_AREA_RATIO_MIN:
        return single_prob
    if area_ratio > TTA_AREA_RATIO_MAX:
        return single_prob
    single_components = component_count(single_mask)
    tta_components = component_count(tta_mask)
    if tta_components > max(single_components + 1, 2):
        return single_prob
    if single_area >= TTA_MIN_CONTROL_AREA_FOR_DICE:
        if binary_dice(single_mask, tta_mask) < TTA_DICE_MIN:
            return single_prob
    return tta_prob


def run_two_stage_inference(
    detector_ckpt: str,
    refiner_ckpt: str,
    test_dir: str,
    output_path: str,
    detector_backbone: str = "resnet34",
    image_size: int = 1024,
    device: str = "cuda",
    use_tta: bool = True,
    refiner_threshold: float = REFINER_THRESHOLD,
    mask_threshold: float = DETECTOR_THRESHOLD,
    spine_threshold: float = SPINE_THRESHOLD,
    min_area: int = 96,
    batch_size: int = 32,
):
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load detector
    print(f"Loading detector: {detector_ckpt}")
    det_ckpt = torch.load(detector_ckpt, map_location=device, weights_only=False)
    detector = FilamentSpineUNet(
        backbone=detector_backbone, pretrained=False, in_channels=1, classes=3
    ).to(device)
    detector.load_state_dict(det_ckpt["model_state"])
    detector.eval()

    # Load refiner
    print(f"Loading refiner: {refiner_ckpt}")
    ref_ckpt = torch.load(refiner_ckpt, map_location=device, weights_only=False)
    refiner = SeedRefiner(pretrained=False).to(device)
    refiner.load_state_dict(ref_ckpt["model_state"])
    refiner.eval()

    test_files = sorted(glob.glob(os.path.join(test_dir, "*.jpeg")) + glob.glob(os.path.join(test_dir, "*.jpg")))
    print(f"Found {len(test_files)} test images")
    transforms = get_valid_transforms(image_size)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    submission_rows = []
    stats = {
        "total_images": len(test_files),
        "total_instances": 0,
        "det_skipped_empty": 0,
        "refiner_tasks": 0,
        "tta_accepted": 0,
        "tta_rejected": 0,
    }

    for img_path in tqdm(test_files, desc="Two-stage inference"):
        fname = os.path.basename(img_path)
        stem = os.path.splitext(fname)[0]
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        h, w = img.shape
        enhanced = enhance_solar_contrast(img)

        # Stage 1: Run detector
        augmented = transforms(image=enhanced)
        input_t = augmented["image"].unsqueeze(0).to(device)
        with torch.no_grad(), torch.amp.autocast(device_type="cuda"):
            preds = detector(input_t)[0]

        mask_prob = torch.sigmoid(preds[0]).float().cpu().numpy()
        spine_prob = torch.sigmoid(preds[1]).float().cpu().numpy()

        # Solar disk masking
        y_grid, x_grid = np.ogrid[:mask_prob.shape[0], :mask_prob.shape[1]]
        dist_center = np.sqrt(
            (x_grid - mask_prob.shape[1] / 2) ** 2 + (y_grid - mask_prob.shape[0] / 2) ** 2
        )
        solar_disk = dist_center <= (0.49 * mask_prob.shape[0])
        mask_prob = mask_prob * solar_disk
        spine_prob = spine_prob * solar_disk

        # Extract coarse instances via spine-watershed
        kept_instances_1024 = spine_watershed_postprocess(
            mask_prob=mask_prob,
            spine_prob=spine_prob,
            mask_threshold=mask_threshold,
            spine_threshold=spine_threshold,
            min_area=max(min_area // 4, 24),
        )

        # Upsample to native 2048x2048
        coarse_instances = []
        for score, mask_1024, area in kept_instances_1024:
            mask_2048 = cv2.resize(mask_1024.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
            if mask_2048.sum() >= min_area:
                coarse_instances.append((score, mask_2048.astype(bool)))

        if not coarse_instances:
            stats["det_skipped_empty"] += 1
            continue

        # Merge overlapping detector masks
        scored_masks = merge_overlapping_masks([m for _, m in coarse_instances])

        # Stage 2: Prepare refiner crops (channel 0 = raw image, channel 1 = CLAHE, channel 2 = seed)
        crops = []
        crop_info = []
        for mask in scored_masks:
            bounds = square_bounds(mask.astype(np.uint8))
            x0, y0, x1, y1 = bounds
            image_crop = cv2.resize(
                img[y0:y1, x0:x1], (REFINER_CROP_SIZE, REFINER_CROP_SIZE),
                interpolation=cv2.INTER_LINEAR,
            )
            clahe_crop = clahe.apply(image_crop)
            seed_crop = cv2.resize(
                mask[y0:y1, x0:x1].astype(np.uint8), (REFINER_CROP_SIZE, REFINER_CROP_SIZE),
                interpolation=cv2.INTER_NEAREST,
            )
            features = np.stack(
                [image_crop, clahe_crop, seed_crop * 255], axis=0
            ).astype(np.float32) / 255.0
            crops.append(features)
            crop_info.append((bounds, mask))

        if not crops:
            continue

        stats["refiner_tasks"] += len(crops)
        crops_arr = np.stack(crops)

        # Run refiner (with or without TTA)
        if use_tta:
            single_probs = run_refiner_on_crops(refiner, crops_arr, device, use_tta=False)
            tta_probs = run_refiner_on_crops(refiner, crops_arr, device, use_tta=True)
            final_probs = np.stack([
                select_tta_probability(single, tta, refiner_threshold)
                for single, tta in zip(single_probs, tta_probs)
            ])
            accepted = np.array([
                not np.array_equal(single, tta)
                for single, tta in zip(single_probs, tta_probs)
            ])
            stats["tta_accepted"] += int(accepted.sum())
            stats["tta_rejected"] += int((~accepted).sum())
        else:
            final_probs = run_refiner_on_crops(refiner, crops_arr, device, use_tta=False)

        # Decode refined masks to full image resolution
        candidates = []
        for i, (bounds, seed_mask) in enumerate(crop_info):
            x0, y0, x1, y1 = bounds
            prob = final_probs[i]
            prob_resized = cv2.resize(
                prob.astype(np.float32), (x1 - x0, y1 - y0),
                interpolation=cv2.INTER_LINEAR,
            )
            crop_mask = prob_resized >= refiner_threshold
            if crop_mask.sum() == 0:
                continue
            full = np.zeros((h, w), dtype=bool)
            full[y0:y1, x0:x1] = crop_mask
            refined = full & seed_mask
            if refined.sum() == 0:
                refined = full
            score = float(prob_resized[crop_mask].mean())
            candidates.append((score, refined))

        # Enforce single CC within each candidate (filaments are connected)
        cc_candidates = []
        for score, mask in candidates:
            num_labels, labels = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
            if num_labels <= 1:
                cc_candidates.append((score, mask))
            else:
                # Keep only the largest CC
                areas = [int((labels == i).sum()) for i in range(1, num_labels)]
                largest = np.argmax(areas) + 1
                cc_candidates.append((score, labels == largest))

        # Score-ordered panoptic painting
        cc_candidates.sort(key=lambda x: -x[0])
        occupied = np.zeros((h, w), dtype=bool)
        inst_idx = 1
        for score, mask in cc_candidates:
            bool_m = mask & (~occupied)
            clean_area = int(bool_m.sum())
            if clean_area >= min_area:
                occupied |= bool_m
                rle_str = encode_mask_to_rle(bool_m.astype(np.uint8))
                submission_rows.append({
                    "filament_id": f"{stem}_{inst_idx}",
                    "segmentation_rle": rle_str,
                })
                stats["total_instances"] += 1
                inst_idx += 1

    sub_df = pd.DataFrame(submission_rows)
    print(f"\n--- Two-Stage Inference Summary ---")
    print(f"Total images: {stats['total_images']}")
    print(f"Total instances: {stats['total_instances']} ({stats['total_instances']/max(stats['total_images'],1):.1f} per image)")
    print(f"Detector skipped (empty): {stats['det_skipped_empty']}")
    print(f"Refiner tasks: {stats['refiner_tasks']}")
    if use_tta:
        print(f"TTA accepted: {stats['tta_accepted']}, rejected: {stats['tta_rejected']}")

    check_no_overlap(sub_df)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    sub_df.to_csv(output_path, index=False)
    print(f"[SUCCESS] Submission saved to {output_path} ({os.path.getsize(output_path)/1024:.1f} KB)")
    return sub_df


def parse_args():
    parser = argparse.ArgumentParser(description="Two-stage detector+refiner inference")
    parser.add_argument("--test_dir", type=str, default="/home/dead/filament-segmentation/data/MAGFiLO_1.0_Kaggle_2026/test/test_images")
    parser.add_argument("--detector_ckpt", type=str, default="checkpoints/best_resnet34_fold0.pt")
    parser.add_argument("--refiner_ckpt", type=str, default="checkpoints/seed_refiner_fold0.pt")
    parser.add_argument("--output", type=str, default="submission_two_stage.csv")
    parser.add_argument("--backbone", type=str, default="resnet34")
    parser.add_argument("--image_size", type=int, default=1024)
    parser.add_argument("--refiner_threshold", type=float, default=REFINER_THRESHOLD)
    parser.add_argument("--mask_threshold", type=float, default=DETECTOR_THRESHOLD)
    parser.add_argument("--no_tta", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_two_stage_inference(
        detector_ckpt=args.detector_ckpt,
        refiner_ckpt=args.refiner_ckpt,
        test_dir=args.test_dir,
        output_path=args.output,
        detector_backbone=args.backbone,
        image_size=args.image_size,
        use_tta=not args.no_tta,
        refiner_threshold=args.refiner_threshold,
        mask_threshold=args.mask_threshold,
    )
