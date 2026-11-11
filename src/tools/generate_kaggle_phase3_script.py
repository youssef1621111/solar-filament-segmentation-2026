#!/usr/bin/env python3
"""
src/tools/generate_kaggle_phase3_script.py
Generates the self-contained, optimized Kaggle kernel script for Phase 3.
"""

import json
from pathlib import Path
import pandas as pd

def generate():
    manifest_path = Path("/home/fathyyoussef162/data/train_folds_manifest.csv")
    df = pd.read_csv(manifest_path)
    s2f = df.groupby("stem")["fold"].first().to_dict()
    s2f_json = json.dumps(s2f)

    # Use raw string or careful replacement to avoid escaping bugs
    header = '"""\nSolar Filament Segmentation Challenge 2026 (IEEE BigData Cup)\nPhase 3: Baseline U-Net 5-Fold Cross-Validation & Annotator Supervision Comparison (Strategies A, B, C)\nFully reproducible, strictly isolated, zero test-set leakage.\n"""\n\n'
    
    script_body = f'''import os
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
from torch.utils.data import Dataset, DataLoader
from pycocotools import mask as mask_util

# -------------------------------------------------------------------------
# 1. Global Setup & Frozen Manifest
# -------------------------------------------------------------------------

SEED = 42
TARGET_SIZE = 512
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

# Frozen 5-Fold Grouped Stem Assignment (StratifiedGroupKFold on stem)
FROZEN_STEM_TO_FOLD = json.loads({repr(s2f_json)})

log(f"Active Compute Device: {{DEVICE}}")
if torch.cuda.is_available():
    log(f"GPU Model: {{torch.cuda.get_device_name(0)}}")
log(f"Frozen split loaded: {{len(FROZEN_STEM_TO_FOLD)}} physical observation stems.")


# -------------------------------------------------------------------------
# 2. Baseline U-Net Architecture
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
        self.down = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch)
        )

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


# -------------------------------------------------------------------------
# 3. Loss & Metric Functions
# -------------------------------------------------------------------------

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


def compute_official_pq(
    gt_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    iou_threshold: float = IOU_THRESHOLD,
) -> Dict:
    gt_annotator_image_ids = gt_df["filament_id"].str.split("_", n=1).str[0]
    pred_image_ids = pred_df["filament_id"].str.rsplit("_", n=1).str[0]
    annotator_views = gt_annotator_image_ids.unique()

    tp_iou_scores = []
    fp_count = 0
    fn_count = 0

    pred_by_stem = {{}}
    if len(pred_df) > 0:
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
# 4. Dataset & Annotator Supervision Strategies (A, B, C)
# -------------------------------------------------------------------------

def build_samples(
    coco: Dict,
    stem_to_fold: Dict[str, int],
    fold: int,
    is_train: bool,
    strategy: Literal["A", "B", "C"]
) -> List[Dict]:
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

    samples = []
    if strategy == "A":
        # Each annotator pass is an individual training sample
        for stem, iids in stems_to_image_ids.items():
            for iid in iids:
                im_info = images_by_id[iid]
                samples.append({{
                    "stem": stem,
                    "file_name": im_info["file_name"],
                    "strategy": "A",
                    "n_annotators": 1,
                    "annotations": anns_by_image_id[iid]
                }})
    elif strategy == "B":
        # Union of all annotator masks per observation stem
        for stem, iids in stems_to_image_ids.items():
            first_im_info = images_by_id[iids[0]]
            all_anns = []
            for iid in iids:
                all_anns.extend(anns_by_image_id[iid])
            samples.append({{
                "stem": stem,
                "file_name": first_im_info["file_name"],
                "strategy": "B",
                "n_annotators": len(iids),
                "annotations": all_anns
            }})
    elif strategy == "C":
        # Agreement-based consensus (multi-annotator: agreement >= 2; single: 100% kept)
        for stem, iids in stems_to_image_ids.items():
            first_im_info = images_by_id[iids[0]]
            passes = [anns_by_image_id[iid] for iid in iids]
            samples.append({{
                "stem": stem,
                "file_name": first_im_info["file_name"],
                "strategy": "C",
                "n_annotators": len(iids),
                "passes_annotations": passes
            }})
    return samples


def render_mask(sample: Dict) -> np.ndarray:
    strat = sample["strategy"]
    if strat in ("A", "B"):
        mask = np.zeros((ORIGINAL_H, ORIGINAL_W), dtype=np.uint8)
        for ann in sample["annotations"]:
            seg = ann.get("segmentation", [])
            if not seg:
                continue
            rles = mask_util.frPyObjects(seg, ORIGINAL_H, ORIGINAL_W)
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
            mask = np.zeros((ORIGINAL_H, ORIGINAL_W), dtype=np.uint8)
            for ann in passes_anns[0]:
                seg = ann.get("segmentation", [])
                if not seg:
                    continue
                rles = mask_util.frPyObjects(seg, ORIGINAL_H, ORIGINAL_W)
                rle = mask_util.merge(rles)
                m = mask_util.decode(rle)
                if m.ndim == 3:
                    m = m.squeeze(-1)
                mask |= m.astype(np.uint8)
            return mask
        else:
            accum = np.zeros((ORIGINAL_H, ORIGINAL_W), dtype=np.int32)
            for pass_anns in passes_anns:
                pass_mask = np.zeros((ORIGINAL_H, ORIGINAL_W), dtype=np.uint8)
                for ann in pass_anns:
                    seg = ann.get("segmentation", [])
                    if not seg:
                        continue
                    rles = mask_util.frPyObjects(seg, ORIGINAL_H, ORIGINAL_W)
                    rle = mask_util.merge(rles)
                    m = mask_util.decode(rle)
                    if m.ndim == 3:
                        m = m.squeeze(-1)
                    pass_mask |= m.astype(np.uint8)
                accum += pass_mask
            return (accum >= 2).astype(np.uint8)
    raise ValueError(f"Unknown strategy: {{strat}}")


class SolarDataset(Dataset):
    def __init__(self, samples: List[Dict], train_img_dir: str, target_size: int = TARGET_SIZE, augment: bool = True):
        self.samples = samples
        self.train_img_dir = train_img_dir
        self.target_size = target_size
        self.augment = augment

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        
        # Lazy cached mask at target_size
        if "cached_mask" in s:
            mask = s["cached_mask"]
        else:
            m_raw = render_mask(s)
            mask = cv2.resize(m_raw, (self.target_size, self.target_size), interpolation=cv2.INTER_NEAREST)
            s["cached_mask"] = mask

        img_p = os.path.join(self.train_img_dir, s["file_name"])
        if not os.path.exists(img_p):
            stem = s["stem"]
            for ext in [".jpeg", ".jpg", ".png"]:
                cand = os.path.join(self.train_img_dir, stem + ext)
                if os.path.exists(cand):
                    img_p = cand
                    break

        img = cv2.imread(img_p, cv2.IMREAD_GRAYSCALE) if os.path.exists(img_p) else np.zeros((ORIGINAL_H, ORIGINAL_W), dtype=np.uint8)
        if (ORIGINAL_H, ORIGINAL_W) != (self.target_size, self.target_size):
            img = cv2.resize(img, (self.target_size, self.target_size), interpolation=cv2.INTER_AREA)

        img_norm = img.astype(np.float32) / 255.0
        mask_norm = mask.astype(np.float32)

        if self.augment:
            if random.random() > 0.5:
                img_norm = np.ascontiguousarray(np.fliplr(img_norm))
                mask_norm = np.ascontiguousarray(np.fliplr(mask_norm))
            if random.random() > 0.5:
                img_norm = np.ascontiguousarray(np.flipud(img_norm))
                mask_norm = np.ascontiguousarray(np.flipud(mask_norm))

        return (
            torch.from_numpy(img_norm).unsqueeze(0),
            torch.from_numpy(mask_norm).unsqueeze(0)
        )


# -------------------------------------------------------------------------
# 5. Training & Evaluation Engine
# -------------------------------------------------------------------------

def execute_fold_run(
    fold: int,
    strategy: Literal["A", "B", "C"],
    coco: Dict,
    img_id_to_stem: Dict[str, str],
    train_img_dir: str,
    epochs: int = 5,
    batch_size: int = 8,
    lr: float = 5e-4,
    min_area: int = 50,
    prob_threshold: float = 0.50,
) -> Dict:
    set_seed(SEED + fold * 10 + ord(strategy))
    log(f"--- Running Fold {{fold}} | Strategy {{strategy}} ---")

    train_samples = build_samples(coco, FROZEN_STEM_TO_FOLD, fold=fold, is_train=True, strategy=strategy)
    val_stems = sorted([s for s, f in FROZEN_STEM_TO_FOLD.items() if f == fold])
    log(f"Train samples: {{len(train_samples)}} | Validation stems: {{len(val_stems)}}")

    dataset = SolarDataset(train_samples, train_img_dir, target_size=TARGET_SIZE, augment=True)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)

    model = BaselineUNet(in_ch=1, out_ch=1, base_f=32).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion_bce = nn.BCEWithLogitsLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs * len(loader)), eta_min=1e-5)
    scaler = torch.amp.GradScaler("cuda") if DEVICE.type == "cuda" else None

    # Train
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
                loss = criterion_bce(logits, masks) + soft_dice_loss(torch.sigmoid(logits), masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            ep_loss += loss.item()
        log(f"Fold {{fold}} [{{strategy}}] Ep {{ep}}/{{epochs}} | Loss: {{ep_loss/len(loader):.4f}} | LR: {{scheduler.get_last_lr()[0]:.2e}}")
    train_time = time.time() - t_train_start

    # Checkpoint
    ckpt_path = f"/kaggle/working/baseline_unet_fold{{fold}}_strat{{strategy}}.pt"
    torch.save(model.state_dict(), ckpt_path)

    # Validation Inference
    t_infer_start = time.time()
    model.eval()
    pred_rows = []
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
            img_in = cv2.resize(img, (TARGET_SIZE, TARGET_SIZE), interpolation=cv2.INTER_AREA)
            t_in = torch.from_numpy(img_in.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0).to(DEVICE)
            with torch.amp.autocast("cuda"):
                logits = model(t_in)
            
            # Cast to float32 explicitly before numpy & OpenCV resize!
            prob_map = torch.sigmoid(logits).squeeze().to(torch.float32).cpu().numpy()
            full_prob = cv2.resize(prob_map, (ORIGINAL_W, ORIGINAL_H), interpolation=cv2.INTER_LINEAR)

            binary = (full_prob >= prob_threshold).astype(np.uint8)
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
            inst_idx = 0
            for lbl in range(1, num_labels):
                if stats[lbl, cv2.CC_STAT_AREA] >= min_area:
                    inst_idx += 1
                    pred_rows.append({{
                        "filament_id": f"{{stem}}_{{inst_idx}}",
                        "segmentation_rle": mask_to_rle((labels == lbl).astype(np.uint8))
                    }})
    infer_time = time.time() - t_infer_start
    pred_df = pd.DataFrame(pred_rows)

    # Build exact GT DataFrame for this fold using O(1) dict lookup
    val_stems_set = set(val_stems)
    gt_rows = []
    for ann in coco["annotations"]:
        iid = ann["image_id"]
        im_stem = img_id_to_stem.get(iid)
        if im_stem in val_stems_set:
            seg = ann.get("segmentation", [])
            if not seg:
                continue
            rles = mask_util.frPyObjects(seg, ORIGINAL_H, ORIGINAL_W)
            rle = mask_util.merge(rles)
            counts = rle["counts"].decode("ascii") if isinstance(rle["counts"], bytes) else str(rle["counts"])
            gt_rows.append({{
                "filament_id": f"{{iid}}_{{ann['id']}}",
                "segmentation_rle": counts
            }})
    gt_df = pd.DataFrame(gt_rows)

    metrics = compute_official_pq(gt_df, pred_df, iou_threshold=0.50)
    result = {{
        "fold": fold,
        "strategy": strategy,
        "pq": metrics["pq"],
        "sq": metrics["sq"],
        "rq": metrics["rq"],
        "tp": metrics["tp"],
        "fp": metrics["fp"],
        "fn": metrics["fn"],
        "num_annotator_views": metrics["num_annotator_views"],
        "total_predicted_instances": len(pred_rows),
        "total_gt_instances": len(gt_df),
        "train_time_seconds": round(train_time, 2),
        "infer_time_seconds": round(infer_time, 2),
        "infer_ms_per_image": round((infer_time / max(1, len(val_stems))) * 1000, 2),
        "checkpoint_path": ckpt_path
    }}

    log(f"Fold {{fold}} [{{strategy}}] PQ: {{result['pq']:.5f}} (SQ: {{result['sq']:.5f}}, RQ: {{result['rq']:.5f}} | TP: {{result['tp']}}, FP: {{result['fp']}}, FN: {{result['fn']}})")
    return result


# -------------------------------------------------------------------------
# 6. Main Master Execution Flow
# -------------------------------------------------------------------------

def main():
    log("=== Starting Phase 3 Baseline & Supervision Comparison Pipeline ===")
    ann_paths = glob.glob("/kaggle/input/**/MAGFiLO_1.0_Annotations_kaggle2026_train.json", recursive=True)
    if not ann_paths:
        ann_paths = glob.glob("data/**/MAGFiLO_1.0_Annotations_kaggle2026_train.json", recursive=True)
    assert len(ann_paths) > 0, "Could not find competition training annotations JSON!"
    
    comp_root = os.path.dirname(os.path.dirname(ann_paths[0]))
    train_img_dir = os.path.join(comp_root, "train", "train_images")
    log(f"Found annotations: {{ann_paths[0]}}")
    log(f"Found train images dir: {{train_img_dir}}")

    with open(ann_paths[0], "r", encoding="utf-8") as f:
        coco = json.load(f)
    log(f"Loaded COCO: {{len(coco['images'])}} images, {{len(coco['annotations'])}} annotations.")
    img_id_to_stem = {{im["id"]: im["file_name"].split(".")[0] for im in coco["images"]}}

    summary_output = {{
        "phase": 3,
        "model_architecture": "BaselineUNet (32 base_f, 7.76M params)",
        "resolution": f"{{TARGET_SIZE}}x{{TARGET_SIZE}}",
        "epochs_per_fold": 5,
        "batch_size": 8,
        "seed": SEED,
        "device": str(DEVICE),
        "step1_baseline_5fold": {{}},
        "step2_supervision_comparison": {{}}
    }}

    # ---------------------------------------------------------------------
    # STEP 1: Baseline 5-Fold Cross-Validation (Strategy A: Individual Passes)
    # ---------------------------------------------------------------------
    log("=======================================================")
    log("STEP 1: Executing 5-Fold Cross Validation for Baseline (Strategy A)")
    log("=======================================================")
    fold_results_strat_a = []
    for f in range(5):
        res = execute_fold_run(fold=f, strategy="A", coco=coco, img_id_to_stem=img_id_to_stem, train_img_dir=train_img_dir, epochs=5)
        fold_results_strat_a.append(res)

    pqs_a = [r["pq"] for r in fold_results_strat_a]
    sqs_a = [r["sq"] for r in fold_results_strat_a]
    rqs_a = [r["rq"] for r in fold_results_strat_a]

    summary_output["step1_baseline_5fold"] = {{
        "strategy": "A (Individual passes as separate samples)",
        "mean_pq": float(np.mean(pqs_a)),
        "std_pq": float(np.std(pqs_a)),
        "min_pq": float(np.min(pqs_a)),
        "max_pq": float(np.max(pqs_a)),
        "mean_sq": float(np.mean(sqs_a)),
        "mean_rq": float(np.mean(rqs_a)),
        "total_tp": int(sum(r["tp"] for r in fold_results_strat_a)),
        "total_fp": int(sum(r["fp"] for r in fold_results_strat_a)),
        "total_fn": int(sum(r["fn"] for r in fold_results_strat_a)),
        "total_train_time_seconds": round(sum(r["train_time_seconds"] for r in fold_results_strat_a), 2),
        "total_infer_time_seconds": round(sum(r["infer_time_seconds"] for r in fold_results_strat_a), 2),
        "folds": fold_results_strat_a
    }}

    log(f">>> Baseline 5-Fold (Strategy A) Results:")
    log(f"Mean PQ: {{summary_output['step1_baseline_5fold']['mean_pq']:.5f}} +/- {{summary_output['step1_baseline_5fold']['std_pq']:.5f}}")
    log(f"Mean SQ: {{summary_output['step1_baseline_5fold']['mean_sq']:.5f}} | Mean RQ: {{summary_output['step1_baseline_5fold']['mean_rq']:.5f}}")

    # ---------------------------------------------------------------------
    # STEP 2: Annotator Supervision Experiments Across All 5 Folds
    # ---------------------------------------------------------------------
    log("=======================================================")
    log("STEP 2: Annotator Supervision Experiments Across All 5 Folds")
    log("=======================================================")
    
    # Run Strategy B across all 5 folds
    fold_results_strat_b = []
    for f in range(5):
        res_b = execute_fold_run(fold=f, strategy="B", coco=coco, img_id_to_stem=img_id_to_stem, train_img_dir=train_img_dir, epochs=5)
        fold_results_strat_b.append(res_b)
        
    pqs_b = [r["pq"] for r in fold_results_strat_b]
    sqs_b = [r["sq"] for r in fold_results_strat_b]
    rqs_b = [r["rq"] for r in fold_results_strat_b]
    
    # Run Strategy C across all 5 folds
    fold_results_strat_c = []
    for f in range(5):
        res_c = execute_fold_run(fold=f, strategy="C", coco=coco, img_id_to_stem=img_id_to_stem, train_img_dir=train_img_dir, epochs=5)
        fold_results_strat_c.append(res_c)
        
    pqs_c = [r["pq"] for r in fold_results_strat_c]
    sqs_c = [r["sq"] for r in fold_results_strat_c]
    rqs_c = [r["rq"] for r in fold_results_strat_c]

    summary_output["step2_supervision_comparison"] = {{
        "strategy_A_individual": summary_output["step1_baseline_5fold"],
        "strategy_B_union": {{
            "strategy": "B (Union of all annotator masks per stem)",
            "mean_pq": float(np.mean(pqs_b)),
            "std_pq": float(np.std(pqs_b)),
            "mean_sq": float(np.mean(sqs_b)),
            "mean_rq": float(np.mean(rqs_b)),
            "total_tp": int(sum(r["tp"] for r in fold_results_strat_b)),
            "total_fp": int(sum(r["fp"] for r in fold_results_strat_b)),
            "total_fn": int(sum(r["fn"] for r in fold_results_strat_b)),
            "total_train_time_seconds": round(sum(r["train_time_seconds"] for r in fold_results_strat_b), 2),
            "folds": fold_results_strat_b
        }},
        "strategy_C_consensus": {{
            "strategy": "C (Consensus voting: count >= 2 for multi-annotator, 100% kept for single)",
            "mean_pq": float(np.mean(pqs_c)),
            "std_pq": float(np.std(pqs_c)),
            "mean_sq": float(np.mean(sqs_c)),
            "mean_rq": float(np.mean(rqs_c)),
            "total_tp": int(sum(r["tp"] for r in fold_results_strat_c)),
            "total_fp": int(sum(r["fp"] for r in fold_results_strat_c)),
            "total_fn": int(sum(r["fn"] for r in fold_results_strat_c)),
            "total_train_time_seconds": round(sum(r["train_time_seconds"] for r in fold_results_strat_c), 2),
            "folds": fold_results_strat_c
        }}
    }}

    log(">>> Strategy Comparison Summary (All 5 Folds):")
    log(f"Strategy A (Individual Passes): Mean PQ = {{summary_output['step1_baseline_5fold']['mean_pq']:.5f}} +/- {{summary_output['step1_baseline_5fold']['std_pq']:.5f}} (TP: {{summary_output['step1_baseline_5fold']['total_tp']}}, FP: {{summary_output['step1_baseline_5fold']['total_fp']}}, FN: {{summary_output['step1_baseline_5fold']['total_fn']}})")
    log(f"Strategy B (Mask Union):        Mean PQ = {{summary_output['step2_supervision_comparison']['strategy_B_union']['mean_pq']:.5f}} +/- {{summary_output['step2_supervision_comparison']['strategy_B_union']['std_pq']:.5f}} (TP: {{summary_output['step2_supervision_comparison']['strategy_B_union']['total_tp']}}, FP: {{summary_output['step2_supervision_comparison']['strategy_B_union']['total_fp']}}, FN: {{summary_output['step2_supervision_comparison']['strategy_B_union']['total_fn']}})")
    log(f"Strategy C (Consensus Voting):  Mean PQ = {{summary_output['step2_supervision_comparison']['strategy_C_consensus']['mean_pq']:.5f}} +/- {{summary_output['step2_supervision_comparison']['strategy_C_consensus']['std_pq']:.5f}} (TP: {{summary_output['step2_supervision_comparison']['strategy_C_consensus']['total_tp']}}, FP: {{summary_output['step2_supervision_comparison']['strategy_C_consensus']['total_fp']}}, FN: {{summary_output['step2_supervision_comparison']['strategy_C_consensus']['total_fn']}})")

    out_file = "/kaggle/working/phase3_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary_output, f, indent=2)
    log(f"Exported Phase 3 results to: {{out_file}}")
    log("=== Phase 3 Execution Finished Successfully! ===")


if __name__ == "__main__":
    main()
'''

    out_script = Path("/home/fathyyoussef162/kaggle_kernel/solar_filament_phase3_baseline.py")
    out_script.write_text(header + script_body)
    print(f"Generated {out_script} ({out_script.stat().st_size:,} bytes).")

if __name__ == "__main__":
    generate()
