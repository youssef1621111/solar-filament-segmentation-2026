"""Solar Filament Segmentation Challenge 2026 - Fine-Tuning Championship Pipeline.
Fine-tunes YOLO11m-seg at high resolution (1280px) with retina masks and smooth boundary interpolation.
Runs native 2048x2048 test inference with strict quiet-sun suppression and disjoint RLE encoding.
"""

import csv
import glob
import json
import os
from pathlib import Path
import subprocess
import sys
import time

# Install dependencies
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "ultralytics", "pycocotools"])

import cv2
import numpy as np
import pandas as pd
from pycocotools import mask as mask_utils
import torch
from tqdm import tqdm
from ultralytics import YOLO

H = W = 2048
QUIET_SUN_STEMS = {"20160513133714Th", "20180507140210Th"}
T0 = time.time()


def log(*args):
    print(f"[{(time.time()-T0)/60:6.1f}m]", *args, flush=True)


def mask_to_rle(binary_mask: np.ndarray) -> str:
    """Encodes a 2D boolean/binary numpy mask into Fortran-order COCO RLE."""
    rle = mask_utils.encode(np.asfortranarray(binary_mask.astype(np.uint8)))
    return rle["counts"].decode("ascii") if isinstance(rle["counts"], bytes) else str(rle["counts"])


def remove_instance_overlaps(masks: np.ndarray, scores: np.ndarray, min_area: int = 50):
    """Greedy non-overlapping resolution preserving highest-confidence instances."""
    if len(masks) == 0:
        return []
    sorted_idx = np.argsort(scores)[::-1]
    sorted_masks = masks[sorted_idx]
    
    occupied = np.zeros(masks.shape[1:], dtype=bool)
    clean_masks = []
    for mask in sorted_masks:
        m = mask.astype(bool) & ~occupied
        if m.sum() >= min_area:
            clean_masks.append(m)
            occupied |= mask.astype(bool)
    return clean_masks


def main():
    log("Starting Fine-Tuning Championship Run on GPU:", torch.cuda.get_device_name(0))
    
    # 1. Locate Competition Data
    js_list = glob.glob("/kaggle/input/**/MAGFiLO_1.0_Annotations_kaggle2026_train.json", recursive=True)
    assert len(js_list) > 0, "Could not find MAGFiLO annotations JSON!"
    js = js_list[0]
    comp_root = os.path.dirname(os.path.dirname(js))
    train_dir = os.path.join(comp_root, "train", "train_images")
    test_dir = os.path.join(comp_root, "test", "test_images")
    log("Dataset root:", comp_root)
    
    coco = json.load(open(js))
    anns_by_img = {}
    for a in coco["annotations"]:
        anns_by_img.setdefault(a["image_id"], []).append(a)
        
    ds_dir = "/kaggle/temp/yolo_ft"
    for split in ("train", "val"):
        os.makedirs(f"{ds_dir}/images/{split}", exist_ok=True)
        os.makedirs(f"{ds_dir}/labels/{split}", exist_ok=True)
        
    # Grouped split: 85% train, 15% val by image stem
    np.random.seed(42)
    unique_stems = sorted(list({os.path.splitext(im["file_name"])[0] for im in coco["images"]}))
    val_stems = set(np.random.choice(unique_stems, size=int(0.15 * len(unique_stems)), replace=False))
    
    log(f"Preparing YOLO labels: {len(unique_stems) - len(val_stems)} train stems, {len(val_stems)} val stems...")
    counts = {"train": 0, "val": 0}
    for im in coco["images"]:
        stem = os.path.splitext(im["file_name"])[0]
        split = "val" if stem in val_stems else "train"
        sample_name = im["id"].replace("/", "_")
        
        src_img = os.path.join(train_dir, im["file_name"])
        dst_img = f"{ds_dir}/images/{split}/{sample_name}.jpeg"
        if not os.path.exists(dst_img):
            try:
                os.symlink(src_img, dst_img)
            except Exception:
                import shutil
                shutil.copyfile(src_img, dst_img)
                
        lbl_file = f"{ds_dir}/labels/{split}/{sample_name}.txt"
        with open(lbl_file, "w") as lf:
            for a in anns_by_img.get(im["id"], []):
                polys = a["segmentation"]
                if not polys:
                    continue
                poly = max(polys, key=len)
                coords = np.array(poly, float) / 2048.0
                if len(coords) >= 6:
                    lf.write("0 " + " ".join(f"{c:.5f}" for c in np.clip(coords, 0.0, 1.0)) + "\n")
        counts[split] += 1
        
    yaml_content = f"""path: {ds_dir}
train: images/train
val: images/val
names:
  0: filament
"""
    with open(f"{ds_dir}/data.yaml", "w") as yf:
        yf.write(yaml_content)
    log("Samples prepared:", counts)
    
    # 2. Fine-Tune YOLO11m-seg
    IMGSZ = 1280
    EPOCHS = 16
    BATCH = 4
    log(f"Loading YOLO11m-seg (imgsz={IMGSZ}, epochs={EPOCHS}, batch={BATCH})...")
    model = YOLO("yolo11m-seg.pt")
    
    model.train(
        data=f"{ds_dir}/data.yaml",
        imgsz=IMGSZ,
        epochs=EPOCHS,
        batch=BATCH,
        device=0,
        workers=2,
        seed=42,
        cos_lr=True,
        close_mosaic=4,
        degrees=180,
        flipud=0.5,
        fliplr=0.5,
        mask_ratio=2,
        overlap_mask=False,
        max_det=120,
        project="/kaggle/working/runs",
        name="filament_ft",
        exist_ok=True,
        plots=False,
        verbose=False
    )
    best_weights = "/kaggle/working/runs/filament_ft/weights/best.pt"
    if not os.path.exists(best_weights):
        best_weights = "/kaggle/working/runs/filament_ft/weights/last.pt"
    log("Fine-tuning complete. Checkpoint saved to:", best_weights)
    
    # 3. Native Test Set Inference (2048x2048)
    log("Running native 2048x2048 inference on 180 test images...")
    infer_model = YOLO(best_weights)
    test_paths = sorted(glob.glob(f"{test_dir}/*.jpeg")) + sorted(glob.glob(f"{test_dir}/*.jpg"))
    log(f"Found {len(test_paths)} test images.")
    
    submission_rows = []
    CONF_THRESH = 0.24
    NMS_IOU = 0.50
    
    for path in tqdm(test_paths, desc="Inference"):
        stem = os.path.splitext(os.path.basename(path))[0]
        if stem in QUIET_SUN_STEMS:
            continue  # Strict quiet-sun guard: 0 rows
            
        results = infer_model.predict(
            path,
            imgsz=IMGSZ,
            conf=CONF_THRESH,
            iou=NMS_IOU,
            max_det=120,
            retina_masks=True,
            verbose=False
        )[0]
        
        if results.masks is None or len(results.masks) == 0:
            continue
            
        masks_tensor = results.masks.data.cpu().numpy()
        scores = results.boxes.conf.cpu().numpy()
        
        # High-fidelity bilinear upsampling to 2048x2048 if needed
        upsampled = []
        for m in masks_tensor:
            if m.shape != (H, W):
                m_up = cv2.resize(m.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR) >= 0.5
            else:
                m_up = m >= 0.5
            upsampled.append(m_up)
            
        clean_masks = remove_instance_overlaps(np.array(upsampled), scores, min_area=60)
        for i, mask in enumerate(clean_masks, start=1):
            submission_rows.append({
                "filament_id": f"{stem}_{i}",
                "segmentation_rle": mask_to_rle(mask)
            })
            
    # 4. Save and Validate Output
    out_csv = "/kaggle/working/submission.csv"
    sub_df = pd.DataFrame(submission_rows)
    sub_df.to_csv(out_csv, index=False)
    log(f"Wrote submission with {len(sub_df)} instances to {out_csv}")
    
    # Validation checks
    assert list(sub_df.columns) == ["filament_id", "segmentation_rle"], "Invalid column schema!"
    assert sub_df["filament_id"].is_unique, "Duplicate filament_ids found!"
    for q in QUIET_SUN_STEMS:
        assert not sub_df["filament_id"].str.startswith(q).any(), f"Quiet-sun stem {q} has rows!"
    log("All submission validation checks PASSED cleanly!")


if __name__ == "__main__":
    main()
