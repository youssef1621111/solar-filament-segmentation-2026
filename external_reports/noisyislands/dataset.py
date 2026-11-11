import os
import cv2
import json
import numpy as np
import torch
from torch.utils.data import Dataset
import pycocotools.mask as mask_util
from collections import defaultdict
from solar_transforms import enhance_solar_contrast, get_train_transforms, get_valid_transforms

class SolarFilamentDataset(Dataset):
    """
    Dataset loader for GONG H-alpha solar filament segmentation.
    Produces:
      - Enhanced solar image (1 channel)
      - Binary filament mask (1 channel)
      - Spine / skeleton heatmap (1 channel)
      - Normalized distance transform map (1 channel)

    Args:
        records_df: DataFrame with columns [file_name, stem, fold]
        data_dir: Path to MAGFiLO_1.0_Kaggle_2026
        coco_json_path: Path to COCO annotations JSON
        image_size: Target resize resolution
        is_train: Whether to apply training augmentations
        single_view: If True, use only the medoid annotator's masks (not consensus OR).
                     Requires selected_image_ids to be provided.
        selected_image_ids: dict mapping filename -> selected annotator image_id (for single_view mode)
    """
    def __init__(
        self,
        records_df,
        data_dir: str = "/home/dead/filament-segmentation/data/MAGFiLO_1.0_Kaggle_2026",
        coco_json_path: str = None,
        image_size: int = 1024,
        is_train: bool = True,
        single_view: bool = False,
        selected_image_ids: dict = None,
    ):
        self.records_df = records_df.reset_index(drop=True)
        self.data_dir = data_dir
        self.image_dir = os.path.join(data_dir, "train/train_images")
        self.image_size = image_size
        self.is_train = is_train
        self.single_view = single_view
        self.selected_image_ids = selected_image_ids or {}

        if is_train:
            self.transforms = get_train_transforms(image_size)
        else:
            self.transforms = get_valid_transforms(image_size)

        if coco_json_path is None:
            coco_json_path = os.path.join(data_dir, "train/MAGFiLO_1.0_Annotations_kaggle2026_train.json")

        # Load COCO annotations grouped by physical filename and annotator image_id
        with open(coco_json_path, "r") as f:
            coco = json.load(f)

        img_id_to_file = {img["id"]: img["file_name"] for img in coco["images"]}
        self.img_id_to_anns = defaultdict(list)
        self.file_to_anns = defaultdict(list)
        for ann in coco["annotations"]:
            self.img_id_to_anns[ann["image_id"]].append(ann)
            fname = img_id_to_file[ann["image_id"]]
            self.file_to_anns[fname].append(ann)

        # Build image_id lookup: file_name -> annotator image_id
        self.file_to_image_ids = defaultdict(list)
        for img in coco["images"]:
            self.file_to_image_ids[img["file_name"]].append(img["id"])

    def __len__(self):
        return len(self.records_df)

    def __getitem__(self, idx):
        row = self.records_df.iloc[idx]
        fname = row["file_name"]
        img_path = os.path.join(self.image_dir, fname)

        # 1. Read grayscale image
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Image not found at {img_path}")

        # 2. Contrast enhancement
        enhanced = enhance_solar_contrast(img)

        # 3. Render ground-truth mask and spine maps at native 2048x2048
        H, W = img.shape[:2]
        mask = np.zeros((H, W), dtype=np.uint8)
        spine = np.zeros((H, W), dtype=np.uint8)

        if self.single_view:
            # Use only the medoid annotator's annotations (single-view GT)
            selected_img_id = self.selected_image_ids.get(fname)
            if selected_img_id is not None:
                anns = self.img_id_to_anns.get(selected_img_id, [])
            else:
                anns = []
        else:
            # Consensus: OR all annotators' masks together
            anns = self.file_to_anns.get(fname, [])

        for ann in anns:
            # Render polygon mask
            try:
                rles = mask_util.frPyObjects(ann["segmentation"], H, W)
                poly_mask = mask_util.decode(mask_util.merge(rles))
                mask |= poly_mask.astype(np.uint8)
            except Exception:
                pass

            # Render spine polyline
            spine_pts = ann.get("spine", [])
            if spine_pts and len(spine_pts) >= 4:
                coords = np.array(spine_pts, dtype=np.float32).reshape(-1, 2)
                pts = np.round(coords).astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(spine, [pts], isClosed=False, color=1, thickness=3)

        # Distance transform inside the filaments
        dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        max_d = dist.max()
        if max_d > 0:
            dist = dist / max_d

        # 4. Apply augmentations
        augmented = self.transforms(
            image=enhanced,
            mask=mask,
            spine=spine,
            weight=dist,
        )

        image_t = augmented["image"].float() # [1, H, W]
        mask_t = augmented["mask"].unsqueeze(0).float() # [1, H, W]
        spine_t = augmented["spine"].unsqueeze(0).float() # [1, H, W]
        dist_t = augmented["weight"].unsqueeze(0).float() # [1, H, W]

        # Target tensor concatenates [mask, spine, distance]
        target_t = torch.cat([mask_t, spine_t, dist_t], dim=0) # [3, H, W]

        return {
            "image": image_t,
            "target": target_t,
            "file_name": fname,
            "stem": row["stem"],
        }
