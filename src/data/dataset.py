#!/usr/bin/env python3
"""
src/data/dataset.py
Dataset module for Solar Filament Segmentation supporting:
  - Strategy A: Individual annotator passes as separate training samples.
  - Strategy B: Union of all annotator masks per observation stem.
  - Strategy C: Agreement-based consensus masks (count >= 2 for multi-annotator; 100% kept for single-annotator).
"""

import json
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
from pycocotools import mask as mask_util
from torch.utils.data import Dataset

ORIGINAL_H = 2048
ORIGINAL_W = 2048


def build_samples_for_strategy(
    coco_json_path: str,
    manifest_df: pd.DataFrame,
    fold: int,
    is_train: bool = True,
    strategy: Literal["A", "B", "C"] = "A",
) -> List[Dict]:
    """
    Constructs the list of sample metadata dictionaries for training or validation.

    Strategies:
      - 'A': Each annotator pass (COCO image record) is an independent sample.
      - 'B': All annotator masks for a physical observation stem are merged into a union mask.
      - 'C': Agreement-based consensus mask:
             For single-annotator stems: 100% preserved.
             For multi-annotator stems: pixels/instances where agreement >= 2 annotators.
    """
    if is_train:
        allowed_stems = set(manifest_df[manifest_df["fold"] != fold]["stem"])
    else:
        allowed_stems = set(manifest_df[manifest_df["fold"] == fold]["stem"])

    with open(coco_json_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    # Index images by id and stem
    images_by_id = {im["id"]: im for im in coco["images"]}
    stems_to_image_ids = defaultdict(list)
    for im in coco["images"]:
        stem = im["file_name"].split(".")[0]
        if stem in allowed_stems:
            stems_to_image_ids[stem].append(im["id"])

    # Index annotations by image_id
    anns_by_image_id = defaultdict(list)
    for a in coco["annotations"]:
        anns_by_image_id[a["image_id"]].append(a)

    samples = []

    if strategy == "A":
        # Each annotator pass is an independent sample
        for stem, iids in stems_to_image_ids.items():
            for iid in iids:
                im_info = images_by_id[iid]
                anns = anns_by_image_id[iid]
                samples.append({
                    "sample_id": iid,
                    "stem": stem,
                    "file_name": im_info["file_name"],
                    "strategy": "A",
                    "n_annotators": 1,
                    "annotations": anns,
                })

    elif strategy == "B":
        # Union of all annotator masks for each stem
        for stem, iids in stems_to_image_ids.items():
            first_im_info = images_by_id[iids[0]]
            all_stem_anns = []
            for iid in iids:
                all_stem_anns.extend(anns_by_image_id[iid])
            samples.append({
                "sample_id": stem,
                "stem": stem,
                "file_name": first_im_info["file_name"],
                "strategy": "B",
                "n_annotators": len(iids),
                "annotations": all_stem_anns,
            })

    elif strategy == "C":
        # Agreement-based consensus
        for stem, iids in stems_to_image_ids.items():
            first_im_info = images_by_id[iids[0]]
            passes_anns = [anns_by_image_id[iid] for iid in iids]
            samples.append({
                "sample_id": stem,
                "stem": stem,
                "file_name": first_im_info["file_name"],
                "strategy": "C",
                "n_annotators": len(iids),
                "passes_annotations": passes_anns,
            })

    return samples


def render_binary_mask(
    sample: Dict, height: int = ORIGINAL_H, width: int = ORIGINAL_W
) -> np.ndarray:
    """
    Renders a 2D binary numpy mask (H, W) according to the sample's strategy.
    """
    strat = sample["strategy"]

    if strat in ("A", "B"):
        mask = np.zeros((height, width), dtype=np.uint8)
        for ann in sample["annotations"]:
            seg = ann.get("segmentation", [])
            if not seg:
                continue
            rles = mask_util.frPyObjects(seg, height, width)
            rle = mask_util.merge(rles)
            m = mask_util.decode(rle)
            if m.ndim == 3:
                m = m.squeeze(-1)
            mask |= m.astype(np.uint8)
        return mask

    elif strat == "C":
        n_passes = sample["n_annotators"]
        passes_anns = sample["passes_annotations"]

        if n_passes == 1:
            # Single-annotator stem: 100% preserved
            mask = np.zeros((height, width), dtype=np.uint8)
            for ann in passes_anns[0]:
                seg = ann.get("segmentation", [])
                if not seg:
                    continue
                rles = mask_util.frPyObjects(seg, height, width)
                rle = mask_util.merge(rles)
                m = mask_util.decode(rle)
                if m.ndim == 3:
                    m = m.squeeze(-1)
                mask |= m.astype(np.uint8)
            return mask
        else:
            # Multi-annotator: count pixel overlap across passes
            accum = np.zeros((height, width), dtype=np.int32)
            for pass_anns in passes_anns:
                pass_mask = np.zeros((height, width), dtype=np.uint8)
                for ann in pass_anns:
                    seg = ann.get("segmentation", [])
                    if not seg:
                        continue
                    rles = mask_util.frPyObjects(seg, height, width)
                    rle = mask_util.merge(rles)
                    m = mask_util.decode(rle)
                    if m.ndim == 3:
                        m = m.squeeze(-1)
                    pass_mask |= m.astype(np.uint8)
                accum += pass_mask

            # Consensus requirement: agreement >= 2
            consensus_mask = (accum >= 2).astype(np.uint8)
            return consensus_mask

    raise ValueError(f"Unknown strategy: {strat}")


class SolarFilamentDataset(Dataset):
    """
    PyTorch Dataset yielding image and mask tensors.
    Supports on-the-fly resizing or tile cropping, and spatial augmentations.
    """
    def __init__(
        self,
        samples: List[Dict],
        images_dir: str,
        target_size: int = 512,
        augment: bool = False,
    ):
        self.samples = samples
        self.images_dir = images_dir
        self.target_size = target_size
        self.augment = augment

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[idx]
        file_name = sample["file_name"]
        img_path = os.path.join(self.images_dir, file_name)

        if not os.path.exists(img_path):
            stem = sample["stem"]
            for ext in [".jpeg", ".jpg", ".png"]:
                candidate = os.path.join(self.images_dir, stem + ext)
                if os.path.exists(candidate):
                    img_path = candidate
                    break

        if os.path.exists(img_path):
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        else:
            img = np.zeros((ORIGINAL_H, ORIGINAL_W), dtype=np.uint8)

        mask = render_binary_mask(sample, height=ORIGINAL_H, width=ORIGINAL_W)

        # Resize to target size if needed
        if (ORIGINAL_H, ORIGINAL_W) != (self.target_size, self.target_size):
            img_resized = cv2.resize(img, (self.target_size, self.target_size), interpolation=cv2.INTER_AREA)
            # Nearest neighbor for mask to avoid blurring binary edges
            mask_resized = cv2.resize(mask, (self.target_size, self.target_size), interpolation=cv2.INTER_NEAREST)
        else:
            img_resized = img
            mask_resized = mask

        img_norm = img_resized.astype(np.float32) / 255.0
        mask_norm = mask_resized.astype(np.float32)

        if self.augment:
            if random.random() > 0.5:
                img_norm = np.ascontiguousarray(np.fliplr(img_norm))
                mask_norm = np.ascontiguousarray(np.fliplr(mask_norm))
            if random.random() > 0.5:
                img_norm = np.ascontiguousarray(np.flipud(img_norm))
                mask_norm = np.ascontiguousarray(np.flipud(mask_norm))

        t_img = torch.from_numpy(img_norm).unsqueeze(0)  # (1, H, W)
        t_mask = torch.from_numpy(mask_norm).unsqueeze(0)  # (1, H, W)

        return t_img, t_mask
