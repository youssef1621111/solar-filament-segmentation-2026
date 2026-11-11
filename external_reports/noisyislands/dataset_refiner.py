"""Refiner training dataset: per-instance crop extraction.

Each sample is a (256, 256) 3-channel crop: [image, CLAHE, seed_mask].
The seed_mask is a jittered version of the GT mask (for training) or the exact
GT mask (for inference). This trains the refiner to take an imperfect seed
and produce a refined binary mask.

This implements the approach from:
- lb-0-70-topology-safe-solar-filaments-handoff.ipynb (RefinerDataset)
- new-metric-solar-filament.ipynb (jitter_seed_mask, square_bounds)
"""

import math
import random
import json
import os
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
import pycocotools.mask as mask_utils
from solar_transforms import enhance_solar_contrast


def square_bounds(
    mask: np.ndarray,
    context: float = 1.8,
    minimum: int = 96,
) -> tuple[int, int, int, int]:
    """Compute square crop bounds around a mask with given context multiplier."""
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
        x1 -= x0
        x0 = 0
    if y0 < 0:
        y1 -= y0
        y0 = 0
    if x1 > width:
        x0 -= x1 - width
        x1 = width
    if y1 > height:
        y0 -= y1 - height
        y1 = height
    return int(x0), int(y0), int(x1), int(y1)


def jitter_seed_mask(mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Jitter a ground-truth mask to simulate detector noise for refiner training."""
    seed = mask.astype(np.uint8)
    radius = int(rng.integers(-5, 6))
    if radius != 0:
        kernel_size = max(1, 2 * abs(radius) + 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        seed = cv2.dilate(seed, kernel) if radius > 0 else cv2.erode(seed, kernel)
    shift_x = int(rng.integers(-8, 9))
    shift_y = int(rng.integers(-8, 9))
    if shift_x != 0 or shift_y != 0:
        matrix = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
        seed = cv2.warpAffine(seed, matrix, (seed.shape[1], seed.shape[0]), flags=cv2.INTER_NEAREST)
    if not seed.any():
        seed = mask.astype(np.uint8)
    return seed.astype(bool)


def annotation_to_mask(annotation: dict, h: int, w: int) -> np.ndarray:
    rles = mask_utils.frPyObjects(annotation["segmentation"], h, w)
    rle = mask_utils.merge(rles)
    return mask_utils.decode(rle).astype(bool)


class RefinerDataset(Dataset):
    """
    Crop-level dataset for the SeedRefiner.
    Each item is one instance crop with 3 channels: [image, CLAHE, seed_mask].
    Training ground truth is the exact GT mask within the crop.

    Args:
        filenames: List of physical image filenames
        image_dir: Directory containing the original images
        annotations_by_image_id: dict mapping image_id -> list of annotation dicts
        images_by_id: dict mapping image_id -> image metadata dict
        selected_image_ids: dict mapping filename -> selected (medoid) image_id
        crop_size: Refiner input resolution (default 256)
        augment: Whether to apply training augmentations
    """

    def __init__(
        self,
        filenames: list[str],
        image_dir: str,
        annotations_by_image_id: dict,
        images_by_id: dict,
        selected_image_ids: dict,
        crop_size: int = 256,
        context: float = 1.8,
        min_crop: int = 96,
        augment: bool = True,
        seed: int = 42,
    ):
        self.filenames = filenames
        self.image_dir = image_dir
        self.annotations_by_image_id = annotations_by_image_id
        self.images_by_id = images_by_id
        self.selected_image_ids = selected_image_ids
        self.crop_size = crop_size
        self.context = context
        self.min_crop = min_crop
        self.augment = augment
        self.seed = seed
        self.rng = random.Random(seed)

        self.samples = []
        for idx, filename in enumerate(filenames):
            image_id = selected_image_ids.get(filename)
            if image_id is None:
                continue
            anns = annotations_by_image_id.get(image_id, [])
            for ann in anns:
                if ann.get("iscrowd", 0):
                    continue
                self.samples.append((filename, image_id, ann["id"]))

        print(f"  RefinerDataset {'(train)' if augment else '(val)'}: {len(self.samples)} samples from {len(filenames)} images")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        filename, image_id, ann_id = self.samples[index]

        # Load original 2048x2048 grayscale image
        img_path = os.path.join(self.image_dir, filename)
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            image = np.zeros((2048, 2048), dtype=np.uint8)

        # Get exact annotation and render GT mask
        info = self.images_by_id[image_id]
        h, w = int(info["height"]), int(info["width"])
        ann = None
        for a in self.annotations_by_image_id.get(image_id, []):
            if a["id"] == ann_id:
                ann = a
                break
        if ann is None:
            mask = np.zeros((h, w), dtype=bool)
        else:
            mask = annotation_to_mask(ann, h, w)

        # Jitter seed mask for training (simulates detector noise)
        if self.augment:
            rng = np.random.default_rng(self.seed + index + self.rng.randint(0, 1_000_000))
            seed_mask = jitter_seed_mask(mask, rng)
        else:
            seed_mask = mask.copy()

        # Compute crop bounds from seed mask
        bounds = square_bounds(seed_mask, self.context, self.min_crop)
        x0, y0, x1, y1 = bounds

        # Extract crops
        image_crop = cv2.resize(
            image[y0:y1, x0:x1], (self.crop_size, self.crop_size), interpolation=cv2.INTER_LINEAR
        )
        target_crop = cv2.resize(
            mask[y0:y1, x0:x1].astype(np.uint8), (self.crop_size, self.crop_size), interpolation=cv2.INTER_NEAREST
        )
        seed_crop = cv2.resize(
            seed_mask[y0:y1, x0:x1].astype(np.uint8), (self.crop_size, self.crop_size), interpolation=cv2.INTER_NEAREST
        )

        # CLAHE enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(image_crop)

        # Build 3-channel input: [image, CLAHE, seed_mask]
        features = np.stack(
            [image_crop, clahe, seed_crop * 255], axis=0
        ).astype(np.float32) / 255.0

        # Augment with D4 rotations and flips
        if self.augment:
            k = random.randrange(4)
            features = np.rot90(features, k, axes=(1, 2)).copy()
            target_crop = np.rot90(target_crop, k).copy()
            if random.random() < 0.5:
                features = features[:, :, ::-1].copy()
                target_crop = target_crop[:, ::-1].copy()
            gain = random.uniform(0.9, 1.1)
            features[:2] = np.clip(features[:2] * gain, 0, 1)

        return (
            torch.from_numpy(np.ascontiguousarray(features)),
            torch.from_numpy(np.ascontiguousarray(target_crop[None].astype(np.float32))),
        )


def build_coco_indices(coco_path: str):
    """Load COCO JSON and build indices for the dataset."""
    with open(coco_path, "r") as f:
        coco = json.load(f)
    images_by_id = {str(img["id"]): img for img in coco["images"]}
    annotations_by_image = {}
    for ann in coco["annotations"]:
        annotations_by_image.setdefault(str(ann["image_id"]), []).append(ann)
    image_ids_by_file = {}
    for img_id, img in images_by_id.items():
        image_ids_by_file.setdefault(img["file_name"], []).append(img_id)
    for fname in image_ids_by_file:
        image_ids_by_file[fname].sort()
    return {
        "images_by_id": images_by_id,
        "annotations_by_image": annotations_by_image,
        "image_ids_by_file": image_ids_by_file,
    }


@torch.no_grad()
def test_dataset():
    import pandas as pd
    data_dir = "/home/dead/filament-segmentation/data/MAGFiLO_1.0_Kaggle_2026"
    coco_path = os.path.join(data_dir, "train/MAGFiLO_1.0_Annotations_kaggle2026_train.json")
    image_dir = os.path.join(data_dir, "train/train_images")

    indices = build_coco_indices(coco_path)

    # Use medoid selection from single_view_gt
    from single_view_gt import select_medoid_annotators
    selected = select_medoid_annotators(
        indices["image_ids_by_file"],
        indices["annotations_by_image"],
        indices["images_by_id"],
    )

    folds_df = pd.read_csv("processed/folds_single_view.csv")
    train_files = folds_df[folds_df["fold"] != 0]["file_name"].tolist()[:10]

    ds = RefinerDataset(
        filenames=train_files,
        image_dir=image_dir,
        annotations_by_image_id=indices["annotations_by_image"],
        images_by_id=indices["images_by_id"],
        selected_image_ids=selected,
        augment=True,
    )
    print(f"Dataset size: {len(ds)}")
    features, target = ds[0]
    print(f"Features shape: {features.shape}, dtype: {features.dtype}")
    print(f"Target shape: {target.shape}, dtype: {target.dtype}")
    print(f"Feature stats: min={features.min():.3f}, max={features.max():.3f}")
    print(f"Target stats: min={target.min():.3f}, max={target.max():.3f}, nonzero={target.sum():.1f}")
    print("✓ RefinerDataset test passed")


if __name__ == "__main__":
    test_dataset()
