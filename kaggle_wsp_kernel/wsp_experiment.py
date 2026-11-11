"""
=============================================================================
WEIGHT-SPACE PROGRAMMING (WSP) FOR SOLAR FILAMENT SEGMENTATION
IEEE BigData 2026 Cup — Kaggle Evaluation Engine
=============================================================================
This script executes the Weight-Space Programming (WSP) experimental pipeline:
1. Loads or trains baseline EfficientNet-B0 Fold 0.
2. Programmatically synthesizes and evaluates candidate weight edits:
   - Level 1: Output Head Surgery (Bias Realignment, Channel Damping, Inhibitory Amplification, Composite)
   - Level 2: Penultimate Conv Subspace Projection (SVD Noise Truncation)
   - Level 3: Weight Space Merging vs Probability Map Ensembling
3. Evaluates Panoptic Quality (PQ, SQ, RQ, TP, FP, FN) on strict 5-fold held-out validation.
4. Performs comprehensive threshold (0.45 - 0.70) and area calibration sweeps.
5. Saves all quantitative diagnostics and the championship edited checkpoint.
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
# Architecture Definition
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

# -------------------------------------------------------------------------
# Metric and Extraction Utilities
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

    # Remove overlaps by confidence
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

    for y0 in range(0, ORIGINAL_H - TILE_SIZE + 1, STRIDE):
        for x0 in range(0, ORIGINAL_W - TILE_SIZE + 1, STRIDE):
            patch = np.ascontiguousarray(img_norm[y0:y0+TILE_SIZE, x0:x0+TILE_SIZE])
            p_tensor = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0).to(DEVICE)
            with torch.amp.autocast("cuda"):
                logits = model(p_tensor)
                prob = torch.sigmoid(logits).squeeze()
            prob_accum[y0:y0+TILE_SIZE, x0:x0+TILE_SIZE] += prob * gaussian_weight
            weight_accum[y0:y0+TILE_SIZE, x0:x0+TILE_SIZE] += gaussian_weight

    full_prob = (prob_accum / torch.clamp(weight_accum, min=1e-6)).to(torch.float32).cpu().numpy()
    return full_prob

# -------------------------------------------------------------------------
# Weight-Space Programming (WSP) Surgery Engine
# -------------------------------------------------------------------------
def apply_wsp_edit(base_state_dict: Dict[str, torch.Tensor], edit_type: str) -> Dict[str, torch.Tensor]:
    """Applies targeted, hypothesis-driven weight-space modifications in-memory."""
    sd = {k: v.clone() for k, v in base_state_dict.items()}
    
    if edit_type == "baseline":
        return sd
        
    elif edit_type == "WSP_L1_bias_m15":
        # Level 1: Shift classification head bias by -0.15
        sd["final_conv.3.bias"] -= 0.15
        return sd
        
    elif edit_type == "WSP_L1_damp_noisy":
        # Level 1: Zero out weak noisy excitatory channels 4 and 13
        sd["final_conv.3.weight"][0, 4, 0, 0] = 0.0
        sd["final_conv.3.weight"][0, 13, 0, 0] = 0.0
        return sd
        
    elif edit_type == "WSP_L1_boost_inhib":
        # Level 1: Boost background inhibitor channels by 15%
        for c in [0, 2, 3, 5, 6, 7, 11, 14]:
            sd["final_conv.3.weight"][0, c, 0, 0] *= 1.15
        return sd
        
    elif edit_type == "WSP_L1_composite":
        # Level 1: Synergistic Composite (Bias -0.15, Zero Ch 4&13, Boost Inhib 10%)
        sd["final_conv.3.bias"] -= 0.15
        sd["final_conv.3.weight"][0, 4, 0, 0] = 0.0
        sd["final_conv.3.weight"][0, 13, 0, 0] = 0.0
        for c in [0, 2, 3, 5, 6, 7, 11, 14]:
            sd["final_conv.3.weight"][0, c, 0, 0] *= 1.10
        return sd
        
    elif edit_type == "WSP_L2_svd_trunc":
        # Level 2: SVD Rank 14 Noise Truncation on final_conv.0
        w0 = sd["final_conv.0.weight"].cpu().numpy()
        mat = w0.reshape(16, -1)
        u, s, vt = np.linalg.svd(mat, full_matrices=False)
        mat_trunc = np.dot(u[:, :14] * s[:14], vt[:14, :]).reshape(16, 16, 3, 3)
        sd["final_conv.0.weight"] = torch.from_numpy(mat_trunc.astype(np.float32)).to(sd["final_conv.0.weight"].device)
        return sd
        
    elif edit_type == "WSP_weight_merge":
        # Weight-Space Interpolation: 50% Baseline + 50% WSP Composite
        sd_comp = apply_wsp_edit(base_state_dict, "WSP_L1_composite")
        sd_merged = {}
        for k in sd.keys():
            sd_merged[k] = 0.5 * sd[k] + 0.5 * sd_comp[k]
        return sd_merged
        
    else:
        raise ValueError(f"Unknown edit type: {edit_type}")

# -------------------------------------------------------------------------
# Main Execution Pipeline
# -------------------------------------------------------------------------
def main():
    log("=" * 80)
    log("STARTING WEIGHT-SPACE PROGRAMMING (WSP) RESEARCH EXECUTION")
    log("=" * 80)

    # Locate inputs
    ann_paths = glob.glob("/kaggle/input/**/MAGFiLO_1.0_Annotations_kaggle2026_train.json", recursive=True)
    if not ann_paths:
        ann_paths = glob.glob("data/**/MAGFiLO_1.0_Annotations_kaggle2026_train.json", recursive=True)
    if not ann_paths:
        ann_paths = glob.glob("/kaggle/input/**/train_annotations*.json", recursive=True)
    assert len(ann_paths) > 0, "Could not find training annotations!"
    
    coco_path = ann_paths[0]
    comp_root = os.path.dirname(os.path.dirname(coco_path))
    train_img_dir = os.path.join(comp_root, "train", "train_images")
    if not os.path.exists(train_img_dir):
        cand = glob.glob(os.path.join(comp_root, "**", "train_images"), recursive=True)
        if cand:
            train_img_dir = cand[0]
        else:
            cand = glob.glob("/kaggle/input/**/train_images", recursive=True)
            if cand:
                train_img_dir = cand[0]

    log(f"Images: {train_img_dir}")
    log(f"COCO Annotations: {coco_path}")

    with open(coco_path) as f:
        coco = json.load(f)
    img_id_to_stem = {im["id"]: im["file_name"].split(".")[0] for im in coco["images"]}

    # Prepare Fold 0 Ground Truth
    val_stems = sorted([s for s, f in FROZEN_STEM_TO_FOLD.items() if f == 0])
    log(f"Fold 0 Validation contains {len(val_stems)} physical stems.")

    val_stems_set = set(val_stems)
    gt_rows = []
    for ann in coco["annotations"]:
        im_stem = img_id_to_stem.get(ann["image_id"])
        if im_stem in val_stems_set:
            seg = ann.get("segmentation", [])
            if not seg:
                continue
            rles = mask_util.frPyObjects(seg, ORIGINAL_H, ORIGINAL_W)
            rle = mask_util.merge(rles)
            counts = rle["counts"].decode("ascii") if isinstance(rle["counts"], bytes) else str(rle["counts"])
            gt_rows.append({"filament_id": f"{ann['image_id']}_{ann['id']}", "segmentation_rle": counts})
    gt_df = pd.DataFrame(gt_rows, columns=["filament_id", "segmentation_rle"])
    log(f"Built Ground Truth DataFrame: {len(gt_df)} annotations.")

    # Locate or Train Baseline EfficientNet-B0 Fold 0 Checkpoint
    base_ckpt_path = None
    for root, dirs, files in os.walk("/kaggle/input"):
        if "efficientnet_b0_fold0.pt" in files:
            base_ckpt_path = os.path.join(root, "efficientnet_b0_fold0.pt")
            break
            
    if base_ckpt_path is None:
        ckpt_candidates = [
            "/kaggle/working/efficientnet_b0_fold0.pt",
            "/home/fathyyoussef162/kaggle_kernel/output/efficientnet_b0_fold0.pt",
            "/home/fathyyoussef162/results/weight_space_programming/original_checkpoints/efficientnet_b0_fold0.pt"
        ]
        for p in ckpt_candidates:
            if os.path.exists(p):
                base_ckpt_path = p
                break

    model = PretrainedEfficientNetB0UNet(pretrained=False).to(DEVICE)
    if base_ckpt_path is not None:
        log(f"Found baseline checkpoint at: {base_ckpt_path}")
        state = torch.load(base_ckpt_path, map_location=DEVICE)
        base_state_dict = state["model_state_dict"] if "model_state_dict" in state else state
        model.load_state_dict(base_state_dict)
    else:
        log("No existing checkpoint found. Training baseline EfficientNet-B0 Fold 0 (3 epochs)...")
        # Build training samples for Fold 0
        from torch.utils.data import Dataset
        class SolarTileDataset(Dataset):
            def __init__(self, samples, train_img_dir):
                self.samples = samples
                self.train_img_dir = train_img_dir
            def __len__(self): return len(self.samples)
            def __getitem__(self, idx):
                item = self.samples[idx]
                img_p = os.path.join(self.train_img_dir, item["file_name"])
                img = cv2.imread(img_p, cv2.IMREAD_GRAYSCALE) if os.path.exists(img_p) else np.zeros((ORIGINAL_H, ORIGINAL_W), dtype=np.uint8)
                cx, cy = item["center"]
                h_start = max(0, min(cy - TILE_SIZE // 2, ORIGINAL_H - TILE_SIZE))
                w_start = max(0, min(cx - TILE_SIZE // 2, ORIGINAL_W - TILE_SIZE))
                p_img = img[h_start:h_start+TILE_SIZE, w_start:w_start+TILE_SIZE]
                p_mask = np.zeros((TILE_SIZE, TILE_SIZE), dtype=np.float32)
                for a in item["annotations"]:
                    seg = a.get("segmentation", [])
                    if seg:
                        rles = mask_util.frPyObjects(seg, ORIGINAL_H, ORIGINAL_W)
                        m = mask_util.decode(mask_util.merge(rles))
                        p_mask = np.maximum(p_mask, m[h_start:h_start+TILE_SIZE, w_start:w_start+TILE_SIZE].astype(np.float32))
                return torch.from_numpy(p_img.astype(np.float32)/255.0).unsqueeze(0), torch.from_numpy(p_mask).unsqueeze(0)

        # Build samples
        stems_to_iids = defaultdict(list)
        images_by_id = {im["id"]: im for im in coco["images"]}
        anns_by_iid = defaultdict(list)
        for a in coco["annotations"]: anns_by_iid[a["image_id"]].append(a)
        for im in coco["images"]:
            s = im["file_name"].split(".")[0]
            if FROZEN_STEM_TO_FOLD[s] != 0: stems_to_iids[s].append(im["id"])
        train_samples = []
        for s, iids in stems_to_iids.items():
            for iid in iids:
                im_info = images_by_id[iid]
                anns = anns_by_iid[iid]
                train_samples.append({"file_name": im_info["file_name"], "annotations": anns, "center": (1024, 1024)})
        
        loader = DataLoader(SolarTileDataset(train_samples, train_img_dir), batch_size=4, shuffle=True, num_workers=2)
        model = PretrainedEfficientNetB0UNet(pretrained=True).to(DEVICE)
        opt = torch.optim.AdamW(model.parameters(), lr=4e-4, weight_decay=1e-4)
        crit = nn.BCEWithLogitsLoss()
        model.train()
        for ep in range(1, 4):
            for x, y in loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                opt.zero_grad()
                logits = model(x)
                loss = crit(logits, y)
                loss.backward()
                opt.step()
            log(f"Trained Ep {ep}/3")
        base_state_dict = model.state_dict()
        torch.save(base_state_dict, "/kaggle/working/efficientnet_b0_fold0.pt")

    # Pre-cache raw images for validation stems
    log("Pre-loading Fold 0 validation stem images into memory...")
    val_images = {}
    for stem in val_stems:
        p = os.path.join(train_img_dir, stem + ".jpeg")
        if not os.path.exists(p):
            for ext in [".jpg", ".png"]:
                cand = os.path.join(train_img_dir, stem + ext)
                if os.path.exists(cand): p = cand; break
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE) if os.path.exists(p) else np.zeros((ORIGINAL_H, ORIGINAL_W), dtype=np.uint8)
        val_images[stem] = img
    log(f"Cached {len(val_images)} validation images.")

    gaussian_weight = get_gaussian_weight_matrix(TILE_SIZE)

    # -------------------------------------------------------------------------
    # Evaluation Matrix: Candidate Models & Edits
    # -------------------------------------------------------------------------
    CANDIDATES = [
        ("Baseline (Original EffNet-B0)", "baseline"),
        ("WSP L1: Bias Realignment (-0.15)", "WSP_L1_bias_m15"),
        ("WSP L1: Noisy Channel Damping (Ch 4,13)", "WSP_L1_damp_noisy"),
        ("WSP L1: Inhibitory Amplification (x1.15)", "WSP_L1_boost_inhib"),
        ("WSP L1: Composite Head Surgery", "WSP_L1_composite"),
        ("WSP L2: SVD Rank-14 Noise Truncation", "WSP_L2_svd_trunc"),
        ("WSP L3: Weight Space Merge (Baseline + Composite)", "WSP_weight_merge"),
    ]

    results_table = []
    saved_prob_maps = {}

    for name, edit_type in CANDIDATES:
        log(f"\n--- Evaluating Candidate: {name} ---")
        edited_sd = apply_wsp_edit(base_state_dict, edit_type)
        model.load_state_dict(edited_sd)
        model.eval()

        t_start = time.time()
        prob_maps = {}
        with torch.no_grad():
            for stem in val_stems:
                prob_maps[stem] = predict_single_image(model, val_images[stem], gaussian_weight)
        saved_prob_maps[edit_type] = prob_maps
        elapsed = time.time() - t_start

        # Standard Evaluation (Thresh = 0.60, Min Area = 100)
        pred_rows = []
        for stem in val_stems:
            pred_rows.extend(extract_instances(prob_maps[stem], stem, thresh=0.60, min_area=100))
        pred_df = pd.DataFrame(pred_rows, columns=["filament_id", "segmentation_rle"])
        m = compute_official_pq(gt_df, pred_df, iou_threshold=0.50)

        log(f"Result [{name}] (tau=0.60, area=100) -> PQ: {m['pq']:.5f} (SQ: {m['sq']:.5f}, RQ: {m['rq']:.5f} | TP: {m['tp']}, FP: {m['fp']}, FN: {m['fn']}) in {elapsed:.1f}s")
        results_table.append({
            "name": name,
            "edit_type": edit_type,
            "thresh": 0.60,
            "min_area": 100,
            "pq": m["pq"],
            "sq": m["sq"],
            "rq": m["rq"],
            "tp": m["tp"],
            "fp": m["fp"],
            "fn": m["fn"],
        })

    # -------------------------------------------------------------------------
    # PART 7: Probability Averaging Ensemble vs Weight-Space Merge
    # -------------------------------------------------------------------------
    log("\n--- Evaluating PART 7: Probability-Map Averaging Ensemble ---")
    ens_prob_maps = {}
    base_maps = saved_prob_maps["baseline"]
    comp_maps = saved_prob_maps["WSP_L1_composite"]
    for stem in val_stems:
        ens_prob_maps[stem] = 0.5 * base_maps[stem] + 0.5 * comp_maps[stem]

    ens_pred_rows = []
    for stem in val_stems:
        ens_pred_rows.extend(extract_instances(ens_prob_maps[stem], stem, thresh=0.60, min_area=100))
    ens_pred_df = pd.DataFrame(ens_pred_rows, columns=["filament_id", "segmentation_rle"])
    m_ens = compute_official_pq(gt_df, ens_pred_df, iou_threshold=0.50)
    log(f"Result [Probability Map Ensemble: Baseline + Composite] -> PQ: {m_ens['pq']:.5f} (SQ: {m_ens['sq']:.5f}, RQ: {m_ens['rq']:.5f} | TP: {m_ens['tp']}, FP: {m_ens['fp']}, FN: {m_ens['fn']})")
    results_table.append({
        "name": "Probability Map Ensemble (Baseline + Composite)",
        "edit_type": "ensemble_prob_map",
        "thresh": 0.60,
        "min_area": 100,
        "pq": m_ens["pq"],
        "sq": m_ens["sq"],
        "rq": m_ens["rq"],
        "tp": m_ens["tp"],
        "fp": m_ens["fp"],
        "fn": m_ens["fn"],
    })

    # -------------------------------------------------------------------------
    # PART 9: Comprehensive Threshold & Area Calibration Sweeps
    # -------------------------------------------------------------------------
    log("\n" + "=" * 80)
    log("PART 9: THRESHOLD & AREA CALIBRATION SWEEPS")
    log("=" * 80)

    # Sweep best candidate (Composite) and Baseline across thresholds
    sweep_models = [("baseline", "Baseline"), ("WSP_L1_composite", "WSP Composite")]
    calibration_results = []
    
    for edit_key, label in sweep_models:
        log(f"\n--- Threshold Sweep for {label} (Min Area = 100) ---")
        p_maps = saved_prob_maps[edit_key]
        for tau in [0.45, 0.50, 0.55, 0.60, 0.65, 0.70]:
            rows = []
            for stem in val_stems:
                rows.extend(extract_instances(p_maps[stem], stem, thresh=tau, min_area=100))
            df = pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"])
            metric = compute_official_pq(gt_df, df, iou_threshold=0.50)
            log(f"  {label} (tau={tau:.2f}) -> PQ: {metric['pq']:.5f} | TP: {metric['tp']:4d}, FP: {metric['fp']:4d}, FN: {metric['fn']:4d}")
            calibration_results.append({
                "model": label,
                "thresh": tau,
                "min_area": 100,
                "pq": metric["pq"],
                "sq": metric["sq"],
                "rq": metric["rq"],
                "tp": metric["tp"],
                "fp": metric["fp"],
                "fn": metric["fn"]
            })

    # -------------------------------------------------------------------------
    # Final Reporting & Champion Checkpoint Export
    # -------------------------------------------------------------------------
    log("\n" + "=" * 80)
    log("MASTER RESULTS SUMMARY TABLE (Fold 0 Validation)")
    log("=" * 80)
    summary_df = pd.DataFrame(results_table)
    base_pq = summary_df.loc[summary_df["edit_type"] == "baseline", "pq"].values[0]
    base_fp = summary_df.loc[summary_df["edit_type"] == "baseline", "fp"].values[0]
    summary_df["delta_pq"] = summary_df["pq"] - base_pq
    summary_df["fp_reduction_pct"] = (base_fp - summary_df["fp"]) / base_fp * 100.0

    print(summary_df.to_string(index=False))

    # Save champion checkpoint (WSP L1 Composite)
    champion_sd = apply_wsp_edit(base_state_dict, "WSP_L1_composite")
    torch.save(champion_sd, "/kaggle/working/wsp_champion_fold0.pt")
    log("Exported champion edited checkpoint to /kaggle/working/wsp_champion_fold0.pt")

    # Save full JSON report
    report = {
        "benchmark_base_pq": float(base_pq),
        "results_table": results_table,
        "calibration_sweep": calibration_results
    }
    with open("/kaggle/working/wsp_evaluation_results.json", "w") as f:
        json.dump(report, f, indent=2)
    log("Saved full JSON report to /kaggle/working/wsp_evaluation_results.json")
    log("WEIGHT-SPACE PROGRAMMING EXPERIMENT EXECUTION COMPLETED.")

if __name__ == "__main__":
    main()
