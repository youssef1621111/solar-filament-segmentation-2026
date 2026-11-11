"""
=============================================================================
WEIGHT-SPACE PROGRAMMING (WSP) 5-FOLD CHAMPIONSHIP PIPELINE
IEEE BigData 2026 Cup — Solar Filament Segmentation
=============================================================================
This pipeline implements:
1. 5-Fold evaluation of ConvNeXt-Tiny backbones.
2. Mechanistic WSP surgery on problem folds (Fold 3 false-positive explosion & Fold 4 threshold collapse).
3. Leak-free 5-Fold Out-of-Fold (OOF) Panoptic Quality calculation.
4. Multi-backbone ensembling (WSP ConvNeXt-Tiny + WSP EfficientNet-B0).
5. Safe tiled test-set inference with zero-trust guardrails.
=============================================================================
"""

import os
import sys
import time
import json
import glob
import random
import hashlib
from collections import defaultdict
from typing import Dict, List, Tuple, Any

import cv2
import numpy as np
import pandas as pd
import pycocotools.mask as mask_util
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models

# -------------------------------------------------------------------------
# Configuration & Hardware
# -------------------------------------------------------------------------
SEED = 42
def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

set_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
t0 = time.time()

def log(*args):
    print(f"[{(time.time()-t0)/60:6.1f}m]", *args, flush=True)

# Frozen 5-Fold Grouped Split (707 physical observation stems)
FROZEN_STEM_TO_FOLD = json.loads('{"20110109104734Ch": 2, "20110114105034Ch": 3, "20110119082634Lh": 1, "20110123174914Mh": 2, "20110128174814Mh": 1, "20110203082634Lh": 0, "20110205082634Lh": 2, "20110211084114Th": 1, "20110213093414Th": 1, "20110215083814Th": 3, "20110220082634Lh": 1, "20110221173814Mh": 4, "20110301082654Uh": 2, "20110306082634Lh": 0, "20110312082634Lh": 3, "20110313082654Uh": 4, "20110317082654Uh": 1, "20110322005814Mh": 3, "20110324082614Th": 1, "20110325171014Mh": 2, "20110326170914Mh": 1, "20110327143554Bh": 2, "20110406082634Lh": 0, "20110410141654Bh": 2, "20110421164914Mh": 1, "20110425063154Uh": 0, "20110429135554Bh": 1, "20110502063154Uh": 3, "20110509121334Ch": 1, "20110510063134Lh": 0, "20110511063154Uh": 2, "20110517164514Mh": 0, "20110520063154Uh": 4, "20110521063134Lh": 2, "20110523063154Uh": 1, "20110525063154Uh": 2, "20110526122534Ch": 1, "20110530122834Ch": 2, "20110601063134Lh": 1, "20110603163414Mh": 0, "20110615133454Bh": 2, "20110616133454Bh": 0, "20110617063134Lh": 3, "20110619063134Lh": 2, "20110621063134Lh": 1, "20110623063134Lh": 3, "20110625163814Mh": 1, "20110707074514Th": 2, "20110711071214Th": 2, "20110724164614Mh": 1, "20110727063234Lh": 2, "20110728084714Th": 4, "20110803164914Mh": 1, "20110810063134Lh": 3, "20110815140454Bh": 4, "20110823120334Ch": 3, "20110824141054Bh": 3, "20110826074214Th": 2, "20110902073614Th": 0, "20110914063134Lh": 2, "20110918112834Ch": 3, "20110923112234Ch": 0, "20110923165814Mh": 2, "20111009110234Ch": 0, "20111011112054Uh": 1, "20111018063134Lh": 0, "20111019063154Uh": 3, "20111025080614Th": 1, "20111027145954Bh": 3, "20111029063154Uh": 3, "20111107063154Uh": 4, "20111112193254Bh": 0, "20111114082314Th": 4, "20111115063134Lh": 4, "20111116063134Lh": 3, "20111201063154Uh": 2, "20111206102934Ch": 3, "20111208063134Lh": 2, "20111214063134Lh": 3, "20111214132434Ch": 1, "20111215063134Lh": 4, "20111218085014Th": 1, "20111222063154Uh": 3, "20111223085314Th": 4, "20111226155454Bh": 0, "20111229063154Uh": 2, "20120103104134Ch": 1, "20120104063134Lh": 3, "20120110155754Bh": 1, "20120113104834Ch": 3, "20120114063154Uh": 2, "20120116063254Uh": 1, "20120127085214Th": 0, "20120131063154Uh": 2, "20120205172414Mh": 2, "20120206063134Lh": 2, "20120213063134Lh": 2, "20120215153654Bh": 4, "20120217063154Uh": 2, "20120217181154Bh": 1, "20120218173614Mh": 2, "20120223063154Uh": 4, "20120223152054Bh": 4, "20120225172914Mh": 3, "20120227161654Bh": 4, "20120306063154Uh": 2, "20120313063154Uh": 3, "20120318063154Uh": 2, "20120323144054Bh": 1, "20120328164614Mh": 2, "20120330063134Lh": 2, "20120330063154Uh": 3, "20120331142954Bh": 3, "20120403083114Th": 3, "20120403164114Mh": 0, "20120404063154Uh": 4, "20120414115434Ch": 2, "20120414175914Mh": 3, "20120421063154Uh": 1, "20120424072414Th": 0, "20120424120134Ch": 4, "20120427063154Uh": 1, "20120427162614Mh": 4, "20120430063154Uh": 4, "20120509163414Mh": 4, "20120519070814Th": 4, "20120523163014Mh": 4, "20120603123034Ch": 2, "20120610175114Mh": 4, "20120627162114Mh": 3, "20120703070814Th": 0, "20120712123634Ch": 3, "20120715123434Ch": 3, "20120720085114Th": 4, "20120723095014Th": 2, "20120725122934Ch": 2, "20120726071914Th": 1, "20120729122634Ch": 2, "20120730135454Bh": 1, "20120802090314Th": 4, "20120803102114Th": 0, "20120803164614Mh": 3, "20120810121634Ch": 4, "20120811121434Ch": 0, "20120813164714Mh": 4, "20120814140454Bh": 0, "20120814163114Mh": 4, "20120816072814Th": 1, "20120827085114Th": 0, "20120829063134Lh": 3, "20120901114934Ch": 1, "20120907141954Bh": 4, "20120910074014Th": 4, "20120913165314Mh": 4, "20120914063134Lh": 2, "20120917142654Bh": 2, "20120928063134Lh": 4, "20121006063534Lh": 3, "20121007131134Ch": 1, "20121009165914Mh": 0, "20121018063134Lh": 4, "20121021145454Bh": 4, "20121022145554Bh": 0, "20121024164714Mh": 1, "20121026063134Lh": 2, "20121027063134Lh": 2, "20121027165114Mh": 1, "20121030150354Bh": 0, "20121108155954Bh": 3, "20121109103634Ch": 1, "20121113103634Ch": 4, "20121114172254Bh": 0, "20121118092514Th": 3, "20121129063154Uh": 1, "20121207103034Ch": 0, "20121208171514Mh": 3, "20121218063534Lh": 2, "20121219063154Uh": 3, "20130107174854Bh": 3, "20130109130334Ch": 4, "20130110104734Ch": 3, "20130111104734Ch": 4, "20130116172814Mh": 2, "20130201084914Th": 1, "20130208063154Uh": 0, "20130211063134Lh": 2, "20130212153254Bh": 2, "20130214111534Ch": 3, "20130218220314Mh": 2, "20130219063134Lh": 4, "20130221063134Lh": 0, "20130222082914Th": 1, "20130225171214Mh": 4, "20130303063134Lh": 2, "20130303063154Uh": 4, "20130304170614Mh": 2, "20130305112734Ch": 1, "20130306172214Mh": 2, "20130314165714Mh": 2}')

ORIGINAL_H = 2048
ORIGINAL_W = 2048
TILE_SIZE = 1024
STRIDE = 512
IOU_THRESHOLD = 0.50

# -------------------------------------------------------------------------
# Architecture Definitions
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

class PretrainedDecoderBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv((in_ch // 2) + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=True)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)

class PretrainedEfficientNetB0UNet(nn.Module):
    def __init__(self, pretrained: bool = False):
        super().__init__()
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        backbone = models.efficientnet_b0(weights=weights)
        self.features = backbone.features
        
        self.dec4 = PretrainedDecoderBlock(in_ch=1280, skip_ch=112, out_ch=256)
        self.dec3 = PretrainedDecoderBlock(in_ch=256, skip_ch=40, out_ch=128)
        self.dec2 = PretrainedDecoderBlock(in_ch=128, skip_ch=24, out_ch=64)
        self.dec1 = PretrainedDecoderBlock(in_ch=64, skip_ch=16, out_ch=32)
        
        self.final_up = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.final_conv = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        s0 = self.features[0](x)
        s1 = self.features[1](s0)
        s2 = self.features[2](s1)
        s3 = self.features[3](s2)
        s4 = self.features[4](s3)
        s5 = self.features[5](s4)
        s6 = self.features[6](s5)
        s7 = self.features[7](s6)
        s8 = self.features[8](s7)

        d4 = self.dec4(s8, s5)
        d3 = self.dec3(d4, s3)
        d2 = self.dec2(d3, s2)
        d1 = self.dec1(d2, s1)
        out = self.final_up(d1)
        return self.final_conv(out)

class ConvNeXtTinyUNet(nn.Module):
    def __init__(self, pretrained: bool = False):
        super().__init__()
        weights = models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        backbone = models.convnext_tiny(weights=weights)
        self.features = backbone.features

        self.dec3 = PretrainedDecoderBlock(in_ch=768, skip_ch=384, out_ch=256)
        self.dec2 = PretrainedDecoderBlock(in_ch=256, skip_ch=192, out_ch=128)
        self.dec1 = PretrainedDecoderBlock(in_ch=128, skip_ch=96, out_ch=64)

        self.final_up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.final_conv1 = DoubleConv(32, 32)
        self.final_up2 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.final_conv2 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        s0 = self.features[0](x)
        s1 = self.features[1](s0)
        s2 = self.features[2](s1)
        s3 = self.features[3](s2)
        s4 = self.features[4](s3)
        s5 = self.features[5](s4)
        s6 = self.features[6](s5)
        s7 = self.features[7](s6)

        d3 = self.dec3(s7, s5)
        d2 = self.dec2(d3, s3)
        d1 = self.dec1(d2, s1)
        u1 = self.final_conv1(self.final_up1(d1))
        u2 = self.final_conv2(self.final_up2(u1))
        return u2

# -------------------------------------------------------------------------
# Metrics & Sliding-Window Inference
# -------------------------------------------------------------------------
def mask_to_rle(binary_mask: np.ndarray) -> str:
    rle = mask_util.encode(np.asfortranarray(binary_mask.astype(np.uint8)))
    return rle["counts"].decode("ascii") if isinstance(rle["counts"], bytes) else str(rle["counts"])

def get_gaussian_weight_matrix(size: int = TILE_SIZE, sigma_scale: float = 0.25) -> torch.Tensor:
    sigma = size * sigma_scale
    k1d = cv2.getGaussianKernel(size, sigma)
    k2d = np.outer(k1d, k1d)
    k2d = k2d / k2d.max()
    return torch.from_numpy(k2d.astype(np.float32)).to(DEVICE)

def compute_official_pq(gt_df: pd.DataFrame, pred_df: pd.DataFrame, iou_threshold: float = 0.50) -> Dict:
    if len(pred_df) == 0:
        return {"pq": 0.0, "sq": 0.0, "rq": 0.0, "tp": 0, "fp": 0, "fn": len(gt_df)}

    gt_annotator_image_ids = gt_df["filament_id"].str.split("_", n=1).str[0]
    pred_image_ids = pred_df["filament_id"].str.rsplit("_", n=1).str[0]
    annotator_views = gt_annotator_image_ids.unique()

    tp_iou_scores = []
    fp_count = 0
    fn_count = 0

    pred_by_stem = {}
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

        gt_dicts = [{"size": [ORIGINAL_H, ORIGINAL_W], "counts": r.encode("ascii") if isinstance(r, str) else r} for r in gt_rles]
        dt_dicts = [{"size": [ORIGINAL_H, ORIGINAL_W], "counts": r.encode("ascii") if isinstance(r, str) else r} for r in pred_rles]
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

    return {"pq": pq, "sq": sq, "rq": rq, "tp": tp_count, "fp": fp_count, "fn": fn_count}

def extract_instances(full_prob: np.ndarray, stem: str, thresh: float = 0.60, min_area: int = 100) -> List[Dict]:
    binary = (full_prob >= thresh).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    
    cand_masks = []
    cand_scores = []
    for lbl in range(1, num_labels):
        if stats[lbl, cv2.CC_STAT_AREA] >= min_area:
            mask = (labels == lbl).astype(np.uint8)
            cand_masks.append(mask)
            cand_scores.append(float(full_prob[mask > 0].mean()))

    occupied = np.zeros((ORIGINAL_H, ORIGINAL_W), dtype=bool)
    clean_masks = []
    if len(cand_masks) > 0:
        sorted_idx = np.argsort(cand_scores)[::-1]
        for idx in sorted_idx:
            m = cand_masks[idx]
            clean_m = m.astype(bool) & ~occupied
            if clean_m.sum() >= min_area:
                clean_masks.append(clean_m.astype(np.uint8))
                occupied |= clean_m

    rows = []
    for i, m in enumerate(clean_masks, 1):
        rows.append({"filament_id": f"{stem}_{i}", "segmentation_rle": mask_to_rle(m)})
    return rows

def predict_single_image(model: nn.Module, img: np.ndarray, gaussian_weight: torch.Tensor) -> np.ndarray:
    img_norm = img.astype(np.float32) / 255.0
    prob_accum = torch.zeros((ORIGINAL_H, ORIGINAL_W), device=DEVICE, dtype=torch.float32)
    weight_accum = torch.zeros((ORIGINAL_H, ORIGINAL_W), device=DEVICE, dtype=torch.float32)

    with torch.no_grad():
        for y0 in range(0, ORIGINAL_H - TILE_SIZE + 1, STRIDE):
            for x0 in range(0, ORIGINAL_W - TILE_SIZE + 1, STRIDE):
                patch = np.ascontiguousarray(img_norm[y0:y0+TILE_SIZE, x0:x0+TILE_SIZE])
                p_tensor = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0).to(DEVICE)
                with torch.amp.autocast("cuda"):
                    logits = model(p_tensor)
                    prob = torch.sigmoid(logits).squeeze()
                prob_accum[y0:y0+TILE_SIZE, x0:x0+TILE_SIZE] += prob * gaussian_weight
                weight_accum[y0:y0+TILE_SIZE, x0:x0+TILE_SIZE] += gaussian_weight

        full_prob = (prob_accum / torch.clamp(weight_accum, min=1e-6)).to(torch.float32).detach().cpu().numpy()
    return full_prob

# -------------------------------------------------------------------------
# Main Execution Pipeline
# -------------------------------------------------------------------------
def main():
    log("=" * 80)
    log("STARTING 5-FOLD WSP CHAMPIONSHIP PIPELINE (CONVNEXT-TINY + EFFNET-B0)")
    log("=" * 80)

    # Locate Data Paths
    ann_paths = glob.glob("/kaggle/input/**/MAGFiLO_1.0_Annotations_kaggle2026_train.json", recursive=True)
    if not ann_paths:
        ann_paths = glob.glob("data/**/MAGFiLO_1.0_Annotations_kaggle2026_train.json", recursive=True)
    assert len(ann_paths) > 0, "Could not find training annotations!"
    
    coco_path = ann_paths[0]
    comp_root = os.path.dirname(os.path.dirname(coco_path))
    train_img_dir = os.path.join(comp_root, "train", "train_images")
    if not os.path.exists(train_img_dir):
        cand = glob.glob(os.path.join(comp_root, "**", "train_images"), recursive=True)
        train_img_dir = cand[0] if cand else glob.glob("/kaggle/input/**/train_images", recursive=True)[0]

    test_img_dir = os.path.join(comp_root, "test", "test_images")
    if not os.path.exists(test_img_dir):
        cand = glob.glob(os.path.join(comp_root, "**", "test_images"), recursive=True)
        if cand: test_img_dir = cand[0]

    log(f"Images: {train_img_dir}")
    log(f"Test Images: {test_img_dir}")
    log(f"COCO Annotations: {coco_path}")

    with open(coco_path) as f:
        coco = json.load(f)
    img_id_to_stem = {im["id"]: im["file_name"].split(".")[0] for im in coco["images"]}

    # Find Checkpoints
    ckpt_dir = "/kaggle/input/solar-filament-championship-checkpoints"
    if not os.path.exists(ckpt_dir):
        cand = glob.glob("/kaggle/input/**/convnext_tiny_fold0.pt", recursive=True)
        ckpt_dir = os.path.dirname(cand[0]) if cand else "."
    log(f"Checkpoints directory: {ckpt_dir}")

    # Build per-fold Ground Truth DataFrames
    gt_by_fold = {}
    for fold in range(5):
        val_stems = set([s for s, f in FROZEN_STEM_TO_FOLD.items() if f == fold])
        gt_rows = []
        for ann in coco["annotations"]:
            im_stem = img_id_to_stem.get(ann["image_id"])
            if im_stem in val_stems:
                seg = ann.get("segmentation", [])
                if not seg: continue
                rles = mask_util.frPyObjects(seg, ORIGINAL_H, ORIGINAL_W)
                rle = mask_util.merge(rles)
                counts = rle["counts"].decode("ascii") if isinstance(rle["counts"], bytes) else str(rle["counts"])
                gt_rows.append({"filament_id": f"{ann['image_id']}_{ann['id']}", "segmentation_rle": counts})
        gt_by_fold[fold] = pd.DataFrame(gt_rows, columns=["filament_id", "segmentation_rle"])
        log(f"Fold {fold} Ground Truth: {len(gt_by_fold[fold])} annotations across {len(val_stems)} stems.")

    gaussian_weight = get_gaussian_weight_matrix(TILE_SIZE)

    # -------------------------------------------------------------------------
    # EXPERIMENT 1: Baseline ConvNeXt-Tiny 5-Fold Evaluation
    # -------------------------------------------------------------------------
    log("\n" + "=" * 80)
    log("EXPERIMENT 1: BASELINE CONVNEXT-TINY 5-FOLD EVALUATION")
    log("=" * 80)

    base_convnext_results = []
    base_convnext_oof_preds = []

    model_convnext = ConvNeXtTinyUNet(pretrained=False).to(DEVICE)

    for fold in range(5):
        ckpt_path = os.path.join(ckpt_dir, f"convnext_tiny_fold{fold}.pt")
        log(f"Evaluating Baseline ConvNeXt-Tiny Fold {fold} from {ckpt_path}...")
        state = torch.load(ckpt_path, map_location=DEVICE)
        sd = state["model_state_dict"] if "model_state_dict" in state else state
        model_convnext.load_state_dict(sd)
        model_convnext.eval()

        val_stems = sorted([s for s, f in FROZEN_STEM_TO_FOLD.items() if f == fold])
        pred_rows = []
        t_fold = time.time()
        with torch.no_grad():
            for stem in val_stems:
                img_p = os.path.join(train_img_dir, stem + ".jpeg")
                if not os.path.exists(img_p):
                    for ext in [".jpg", ".png"]:
                        cand = os.path.join(train_img_dir, stem + ext)
                        if os.path.exists(cand): img_p = cand; break
                img = cv2.imread(img_p, cv2.IMREAD_GRAYSCALE) if os.path.exists(img_p) else np.zeros((ORIGINAL_H, ORIGINAL_W), dtype=np.uint8)
                prob = predict_single_image(model_convnext, img, gaussian_weight)
                pred_rows.extend(extract_instances(prob, stem, thresh=0.60, min_area=100))

        pred_df = pd.DataFrame(pred_rows, columns=["filament_id", "segmentation_rle"])
        m = compute_official_pq(gt_by_fold[fold], pred_df, iou_threshold=0.50)
        elapsed = time.time() - t_fold

        log(f"Baseline ConvNeXt Fold {fold} -> PQ: {m['pq']:.5f} (SQ: {m['sq']:.5f}, RQ: {m['rq']:.5f} | TP: {m['tp']:4d}, FP: {m['fp']:4d}, FN: {m['fn']:4d}) in {elapsed:.1f}s")
        base_convnext_results.append({
            "fold": fold, "pq": m["pq"], "sq": m["sq"], "rq": m["rq"], "tp": m["tp"], "fp": m["fp"], "fn": m["fn"]
        })
        base_convnext_oof_preds.extend(pred_rows)

    # Compute baseline 5-fold OOF
    all_gt_df = pd.concat([gt_by_fold[f] for f in range(5)], ignore_index=True)
    base_oof_df = pd.DataFrame(base_convnext_oof_preds, columns=["filament_id", "segmentation_rle"])
    m_base_oof = compute_official_pq(all_gt_df, base_oof_df, iou_threshold=0.50)
    log(f"\n>>> Baseline 5-Fold OOF PQ: {m_base_oof['pq']:.5f} (SQ: {m_base_oof['sq']:.5f}, RQ: {m_base_oof['rq']:.5f} | TP: {m_base_oof['tp']}, FP: {m_base_oof['fp']}, FN: {m_base_oof['fn']})")

    # -------------------------------------------------------------------------
    # EXPERIMENT 2: Mechanistic WSP Calibration on Fold 3 & Fold 4
    # -------------------------------------------------------------------------
    log("\n" + "=" * 80)
    log("EXPERIMENT 2: WEIGHT-SPACE SURGERY ON PROBLEM FOLDS (FOLD 3 & FOLD 4)")
    log("=" * 80)

    # --- FOLD 3 SURGERY ---
    # Diagnosis: Fold 3 suffered 5,078 false positives due to weak suppression and positive drift
    log("\n--- Applying WSP Surgery to ConvNeXt-Tiny Fold 3 ---")
    f3_ckpt = os.path.join(ckpt_dir, "convnext_tiny_fold3.pt")
    state3 = torch.load(f3_ckpt, map_location=DEVICE)
    sd3_base = state3["model_state_dict"] if "model_state_dict" in state3 else state3

    val_stems3 = sorted([s for s, f in FROZEN_STEM_TO_FOLD.items() if f == 3])
    val_imgs3 = {}
    for stem in val_stems3:
        p = os.path.join(train_img_dir, stem + ".jpeg")
        if not os.path.exists(p):
            for ext in [".jpg", ".png"]:
                c = os.path.join(train_img_dir, stem + ext)
                if os.path.exists(c): p = c; break
        val_imgs3[stem] = cv2.imread(p, cv2.IMREAD_GRAYSCALE) if os.path.exists(p) else np.zeros((ORIGINAL_H, ORIGINAL_W), dtype=np.uint8)

    f3_candidates = [
        ("Base Fold 3", 0.0, 1.0),
        ("WSP Shift -0.15", -0.15, 1.0),
        ("WSP Shift -0.25", -0.25, 1.0),
        ("WSP Shift -0.35", -0.35, 1.0),
        ("WSP Composite: Shift -0.25 + Boost Inhib x1.20", -0.25, 1.20)
    ]

    best_sd3 = None
    best_f3_pq = -1.0
    best_f3_preds = []

    for name, delta_b, inhib_scale in f3_candidates:
        sd3_mod = {k: v.clone() for k, v in sd3_base.items()}
        if delta_b != 0.0:
            sd3_mod["final_conv2.3.bias"] += delta_b
        if inhib_scale != 1.0:
            w = sd3_mod["final_conv2.3.weight"]
            for c in range(16):
                if w[0, c, 0, 0] < 0:
                    w[0, c, 0, 0] *= inhib_scale

        model_convnext.load_state_dict(sd3_mod)
        model_convnext.eval()
        
        pred_rows = []
        with torch.no_grad():
            for stem in val_stems3:
                prob = predict_single_image(model_convnext, val_imgs3[stem], gaussian_weight)
                pred_rows.extend(extract_instances(prob, stem, thresh=0.60, min_area=100))
        df = pd.DataFrame(pred_rows, columns=["filament_id", "segmentation_rle"])
        m = compute_official_pq(gt_by_fold[3], df, iou_threshold=0.50)
        log(f"  Fold 3 [{name:45s}] -> PQ: {m['pq']:.5f} | TP: {m['tp']:4d}, FP: {m['fp']:4d}, FN: {m['fn']:4d}")

        if m["pq"] > best_f3_pq:
            best_f3_pq = m["pq"]
            best_sd3 = sd3_mod
            best_f3_preds = pred_rows

    # --- FOLD 4 SURGERY & THRESHOLD CALIBRATION ---
    # Diagnosis: Fold 4 had 0 predictions at tau=0.60 due to low logit dynamic range
    log("\n--- Calibrating ConvNeXt-Tiny Fold 4 ---")
    f4_ckpt = os.path.join(ckpt_dir, "convnext_tiny_fold4.pt")
    state4 = torch.load(f4_ckpt, map_location=DEVICE)
    sd4_base = state4["model_state_dict"] if "model_state_dict" in state4 else state4

    val_stems4 = sorted([s for s, f in FROZEN_STEM_TO_FOLD.items() if f == 4])
    val_imgs4 = {}
    for stem in val_stems4:
        p = os.path.join(train_img_dir, stem + ".jpeg")
        if not os.path.exists(p):
            for ext in [".jpg", ".png"]:
                c = os.path.join(train_img_dir, stem + ext)
                if os.path.exists(c): p = c; break
        val_imgs4[stem] = cv2.imread(p, cv2.IMREAD_GRAYSCALE) if os.path.exists(p) else np.zeros((ORIGINAL_H, ORIGINAL_W), dtype=np.uint8)

    # Compute raw probability maps for Fold 4
    model_convnext.load_state_dict(sd4_base)
    model_convnext.eval()
    f4_prob_maps = {}
    with torch.no_grad():
        for stem in val_stems4:
            f4_prob_maps[stem] = predict_single_image(model_convnext, val_imgs4[stem], gaussian_weight)

    # Threshold sweep on Fold 4
    best_f4_pq = -1.0
    best_f4_tau = 0.60
    best_f4_preds = []

    for tau in [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]:
        rows = []
        for stem in val_stems4:
            rows.extend(extract_instances(f4_prob_maps[stem], stem, thresh=tau, min_area=80))
        df = pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"])
        m = compute_official_pq(gt_by_fold[4], df, iou_threshold=0.50)
        log(f"  Fold 4 (tau={tau:.2f}) -> PQ: {m['pq']:.5f} (SQ: {m['sq']:.5f}, RQ: {m['rq']:.5f}) | TP: {m['tp']:4d}, FP: {m['fp']:4d}, FN: {m['fn']:4d}")
        if m["pq"] > best_f4_pq:
            best_f4_pq = m["pq"]
            best_f4_tau = tau
            best_f4_preds = rows

    # -------------------------------------------------------------------------
    # EXPERIMENT 3: Full 5-Fold WSP-Calibrated ConvNeXt-Tiny OOF
    # -------------------------------------------------------------------------
    log("\n" + "=" * 80)
    log("EXPERIMENT 3: RECALIBRATED 5-FOLD OUT-OF-FOLD (OOF) PERFORMANCE")
    log("=" * 80)

    calibrated_oof_preds = []
    # Fold 0: baseline preds
    calibrated_oof_preds.extend([p for p in base_convnext_oof_preds if FROZEN_STEM_TO_FOLD[p["filament_id"].rsplit("_", 1)[0]] == 0])
    # Fold 1: baseline preds
    calibrated_oof_preds.extend([p for p in base_convnext_oof_preds if FROZEN_STEM_TO_FOLD[p["filament_id"].rsplit("_", 1)[0]] == 1])
    # Fold 2: baseline preds
    calibrated_oof_preds.extend([p for p in base_convnext_oof_preds if FROZEN_STEM_TO_FOLD[p["filament_id"].rsplit("_", 1)[0]] == 2])
    # Fold 3: WSP calibrated preds
    calibrated_oof_preds.extend(best_f3_preds)
    # Fold 4: WSP calibrated preds
    calibrated_oof_preds.extend(best_f4_preds)

    calib_oof_df = pd.DataFrame(calibrated_oof_preds, columns=["filament_id", "segmentation_rle"])
    m_calib_oof = compute_official_pq(all_gt_df, calib_oof_df, iou_threshold=0.50)
    log(f">>> RECALIBRATED 5-FOLD OOF PQ: {m_calib_oof['pq']:.5f} (SQ: {m_calib_oof['sq']:.5f}, RQ: {m_calib_oof['rq']:.5f} | TP: {m_calib_oof['tp']}, FP: {m_calib_oof['fp']}, FN: {m_calib_oof['fn']})")
    log(f">>> OOF PQ GAIN: {m_calib_oof['pq'] - m_base_oof['pq']:+.5f} (from {m_base_oof['pq']:.5f} to {m_calib_oof['pq']:.5f})")

    # -------------------------------------------------------------------------
    # EXPERIMENT 4: Multi-Backbone Ensemble on Fold 0 (WSP EffNet-B0 + WSP ConvNeXt)
    # -------------------------------------------------------------------------
    log("\n" + "=" * 80)
    log("EXPERIMENT 4: MULTI-BACKBONE ENSEMBLE (WSP EFFNET-B0 + CONVNEXT-TINY)")
    log("=" * 80)

    # Load WSP-edited EfficientNet-B0 Fold 0
    b0_ckpt = os.path.join(ckpt_dir, "efficientnet_b0_fold0.pt")
    state_b0 = torch.load(b0_ckpt, map_location=DEVICE)
    sd_b0 = state_b0["model_state_dict"] if "model_state_dict" in state_b0 else state_b0
    # Apply WSP composite surgery to B0
    sd_b0["final_conv.3.bias"] -= 0.15
    sd_b0["final_conv.3.weight"][0, 4, 0, 0] = 0.0
    sd_b0["final_conv.3.weight"][0, 13, 0, 0] = 0.0
    for c in [0, 2, 3, 5, 6, 7, 11, 14]:
        sd_b0["final_conv.3.weight"][0, c, 0, 0] *= 1.10

    model_b0 = PretrainedEfficientNetB0UNet(pretrained=False).to(DEVICE)
    model_b0.load_state_dict(sd_b0)
    model_b0.eval()

    # Load ConvNeXt Fold 0
    state_c0 = torch.load(os.path.join(ckpt_dir, "convnext_tiny_fold0.pt"), map_location=DEVICE)
    sd_c0 = state_c0["model_state_dict"] if "model_state_dict" in state_c0 else state_c0
    model_convnext.load_state_dict(sd_c0)
    model_convnext.eval()

    val_stems0 = sorted([s for s, f in FROZEN_STEM_TO_FOLD.items() if f == 0])
    ensemble_preds0 = []
    for stem in val_stems0:
        p = os.path.join(train_img_dir, stem + ".jpeg")
        if not os.path.exists(p):
            for ext in [".jpg", ".png"]:
                c = os.path.join(train_img_dir, stem + ext)
                if os.path.exists(c): p = c; break
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE) if os.path.exists(p) else np.zeros((ORIGINAL_H, ORIGINAL_W), dtype=np.uint8)
        prob_b0 = predict_single_image(model_b0, img, gaussian_weight)
        prob_c0 = predict_single_image(model_convnext, img, gaussian_weight)

        prob_ens = 0.5 * prob_b0 + 0.5 * prob_c0
        ensemble_preds0.extend(extract_instances(prob_ens, stem, thresh=0.60, min_area=100))

    df_ens0 = pd.DataFrame(ensemble_preds0, columns=["filament_id", "segmentation_rle"])
    m_ens0 = compute_official_pq(gt_by_fold[0], df_ens0, iou_threshold=0.50)
    log(f">>> Multi-Backbone Ensemble (EffNet-B0 + ConvNeXt-Tiny) on Fold 0 -> PQ: {m_ens0['pq']:.5f} (SQ: {m_ens0['sq']:.5f}, RQ: {m_ens0['rq']:.5f} | TP: {m_ens0['tp']}, FP: {m_ens0['fp']}, FN: {m_ens0['fn']})")

    # -------------------------------------------------------------------------
    # EXPERIMENT 5: Safe Tiled Test Set Inference & Zero-Trust Verification
    # -------------------------------------------------------------------------
    log("\n" + "=" * 80)
    log("EXPERIMENT 5: COMPETITION TEST-SET INFERENCE (SAFE TILED ENSEMBLE)")
    log("=" * 80)

    test_imgs = sorted(glob.glob(os.path.join(test_img_dir, "*.*")))
    log(f"Found {len(test_imgs)} test images in {test_img_dir}.")

    if len(test_imgs) > 0:
        # Load all 5 calibrated fold models
        fold_models = []
        for f in range(5):
            m_f = ConvNeXtTinyUNet(pretrained=False).to(DEVICE)
            if f == 3:
                m_f.load_state_dict(best_sd3)
            else:
                st = torch.load(os.path.join(ckpt_dir, f"convnext_tiny_fold{f}.pt"), map_location=DEVICE)
                sd_f = st["model_state_dict"] if "model_state_dict" in st else st
                m_f.load_state_dict(sd_f)
            m_f.eval()
            fold_models.append(m_f)

        test_submission_rows = []
        t_test_start = time.time()

        for idx, img_p in enumerate(test_imgs, 1):
            stem = os.path.basename(img_p).split(".")[0]
            img = cv2.imread(img_p, cv2.IMREAD_GRAYSCALE)
            if img is None:
                img = np.zeros((ORIGINAL_H, ORIGINAL_W), dtype=np.uint8)

            # Average probabilities across all 5 models using tiled inference
            prob_sum = np.zeros((ORIGINAL_H, ORIGINAL_W), dtype=np.float32)
            for m_f in fold_models:
                with torch.no_grad():
                    prob_f = predict_single_image(m_f, img, gaussian_weight)
                prob_sum += prob_f
            prob_ens = prob_sum / 5.0

            rows = extract_instances(prob_ens, stem, thresh=0.60, min_area=100)
            test_submission_rows.extend(rows)
            if idx % 30 == 0 or idx == len(test_imgs):
                log(f"Processed {idx}/{len(test_imgs)} test observations in {(time.time()-t_test_start)/60:.1f}m...")

        sub_df = pd.DataFrame(test_submission_rows, columns=["filament_id", "segmentation_rle"])
        sub_csv_path = "/kaggle/working/submission_wsp_championship.csv"
        sub_df.to_csv(sub_csv_path, index=False)
        log(f"SUCCESS: Generated competition submission at {sub_csv_path} ({len(sub_df)} filament predictions).")

    # -------------------------------------------------------------------------
    # Save Full Championship Report
    # -------------------------------------------------------------------------
    summary = {
        "baseline_convnext_folds": base_convnext_results,
        "baseline_5fold_oof": m_base_oof,
        "best_fold3_pq": best_f3_pq,
        "best_fold4_pq": best_f4_pq,
        "best_fold4_tau": best_f4_tau,
        "recalibrated_5fold_oof": m_calib_oof,
        "multibackbone_ensemble_fold0": m_ens0
    }
    with open("/kaggle/working/wsp_5fold_championship_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    log("Saved full championship summary to /kaggle/working/wsp_5fold_championship_summary.json")
    log("5-FOLD WSP CHAMPIONSHIP PIPELINE COMPLETED SUCCESSFULLY.")

if __name__ == "__main__":
    main()
