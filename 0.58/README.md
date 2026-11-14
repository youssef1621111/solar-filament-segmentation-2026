# 🏆 Solar Filament Segmentation 2026: 0.58 Championship Solution Pack

**Competition:** IEEE BigData 2026 Solar Filament Segmentation Challenge (`filament-segmentation-2026`)  
**Official Kaggle Verified Public LB:** **`0.58`** (Submission Ref: `58349276`, Status: `SubmissionStatus.COMPLETE`)  
**Timestamp:** November 14, 2026  
**Evaluation Canvas:** 2048 × 2048 Full-Disk Solar $H\alpha$ GONG Imagery (180 Test Stems: 178 Active, 2 Quiescent)  
**Primary Submission:** `submissions/submission_selected_primary_058.csv` (1,300 instances, 0 Collisions, 0 Quiet-Sun rows)

---

## 1. Directory Structure

```
0.58/
├── README.md                                # This definitive documentation
├── submissions/
│   ├── submission_selected_primary_058.csv  # 1,300 instances (Verified 0.58 LB on Kaggle, Ref 58349276)
│   └── submission_selected_secondary_058.csv# 1,333 instances (Verified 0.58 LB on Kaggle, Ref 58234966)
├── matrices/
│   ├── 7family_prediction_matrix.csv        # Cross-evaluation matrix with 7 model family votes and expected IoU
│   ├── audit_1333_solo_fibril_purge.csv     # Morphological & physical audit of all 1,333 candidate filaments
│   └── multimodel_consensus_candidates.csv  # High-confidence multi-model consensus candidate pool
├── code/
│   ├── solar_filament_championship.py       # Full-resolution U-Net training, SoftCLDiceLoss, and 4-view TTA inference
│   ├── wsp_5fold_pipeline.py                # 5-fold ConvNeXt-Tiny pipeline + Weight-Space Programming (WSP) surgery
│   ├── phase1_build_7family_matrix.py       # Cross-architecture jury consensus builder
│   ├── phase2_panoptic_surgical_optimizer.py# 3 surgical operations (de-merging, pruning, free-canvas rescue)
│   └── submission_integrity_check.py        # Zero-trust 6-point integrity verifier
└── weights/                                 # 1.53 GB of Binary Model Weights & Checkpoints
    ├── dariush_10_unets/                    # 10 PyTorch U-Net models (936 MB total, extracted & ready)
    │   ├── unet_resnet34_fold0.pt ... fold4.pt               (93.4 MB each)
    │   └── unet_tu-tf_efficientnetv2_s_fold0.pt ... fold4.pt (85.2 MB each)
    └── None00000_dinov3/                    # DINOv3 ViT-B/16 Refiners & LightGBM Models (664 MB total)
        ├── weights/refiner/fold0.pt ... fold4.pt             (132.9 MB each)
        ├── weights/ml/balance_fold0.joblib ... fold4.joblib
        └── reference/submission.csv
```

---

## 2. Complete Model Zoo & Architecture Breakdown

The 0.58 championship submission is powered by an ensemble of **18 models** spanning 7 diverse architectures:

| Model ID | Architecture / Backbone | Resolution & Tiling | Loss Function | Role in 0.58 Solution |
| :--- | :--- | :--- | :--- | :--- |
| **Model 1** | **ResNet-34 U-Net** | $1024 \times 1024$ (stride 512) | Boundary-Weighted BCE + Dice | Global chromospheric baseline |
| **Model 2** | **ResNeXt-50 (32x4d) U-Net** | $1024 \times 1024$ (stride 512) | Focal Loss + Lovász-Softmax | Deep feature representation of faint ribbons |
| **Model 3** | **EfficientNet-B4 U-Net** | $1024 \times 1024$ (stride 512) | BCE + Tversky Loss ($\beta = 0.7$) | High-recall filament edge localization |
| **Models 4–8** | **Dariush ResNet-34 U-Net (5 Folds)** | Multi-Scale Sliding Windows | Boundary Cross-Entropy | Cross-validated ribbon localization |
| **Models 9–13**| **Dariush EfficientNetV2-S U-Net (5 Folds)**| Multi-Scale Sliding Windows | Boundary Cross-Entropy | Sharp edge refinement and contrast calibration |
| **Model 14** | **Ultralytics YOLOv8x-seg / YOLO26x-seg** | Native $2048 \times 2048$ (Retina Masks) | Box CIoU + Mask BCE | Top-down instance seeds & bounding boxes |
| **Model 15** | **Two-Stage Mask R-CNN ResNet-50 FPN v2** | Full disk ($2048 \times 2048$) | Region Proposal Network + RoIAlign | Candidate RoI proposal generator |
| **Model 16** | **4-View TTA ResNet-18 Crop U-Net (Fususu)** | $512 \times 512$ RoI ribbon crops | Dice + BCE with Bilinear Resampling | High-precision boundary refinement inside crops |
| **Model 17** | **5-Fold ConvNeXt-Tiny Dual-Head U-Net** | $1024 \times 1024$ Gaussian overlap | **Soft-clDice + Medial Centerline Loss** | Captures thin serpentine ribbons via $7 \times 7$ kernels |
| **Model 18** | **Weight-Space Programmed EfficientNet-B0** | $1024 \times 1024$ | Parametric Weight Surgery | Background noise suppression in quiet-Sun regions |

---

## 3. The Mathematics of the 0.58 Score

### 3.1 The Exact Panoptic Quality Identity
For a prediction set $P$ evaluated against ground truth instances $G$ annotated by 3 solar physicists:

$$\text{PQ}(P, G) = \underbrace{\frac{\sum_{(p, g) \in \text{TP}} \text{IoU}(p, g)}{|\text{TP}|}}_{\text{SQ}} \times \underbrace{\frac{|\text{TP}|}{|\text{TP}| + \frac{1}{2}|\text{FP}| + \frac{1}{2}|\text{FN}|}}_{\text{RQ}}$$

By substituting $|\text{FN}| = |G| - |\text{TP}|$ and $|\text{FP}| = |P| - |\text{TP}|$, the $|\text{TP}|$ terms in the denominator cancel identically:

$$\boxed{\text{PQ}(P, G) \equiv \frac{2 \sum_{p \in P} \text{IoU}_{\text{matched}}(p, G)}{|P| + |G|}}$$

### 3.2 Marginal Inclusion / Pruning Rule
Every single row $p \in P$ in `submission.csv` adds exactly $+\frac{1}{2}$ to the denominator regardless of whether it is a True Positive or a False Positive:
* **Pruning a mask $p$** strictly increases $\text{PQ}$ if and only if:
  $$2 \cdot \mathbb{E}[\text{IoU}_{\text{matched}}(p)] < \text{PQ}_{\text{current}} \iff \mathbb{E}[\text{IoU}_{\text{matched}}(p)] < 0.29 \quad (\text{at } 0.58)$$
* **Adding a mask $p$** strictly increases $\text{PQ}$ if and only if:
  $$2 \cdot \mathbb{E}[\text{IoU}_{\text{matched}}(p)] > 0.58 \iff \mathbb{E}[\text{IoU}_{\text{matched}}(p)] > 0.29$$

### 3.3 Why Pruning Achieved 0.58
In initial ensembles (`submission_original_055_anchor.csv`, 1,344 instances), raw model predictions included faint single-annotator fibrils and quiet-sun background artifacts.
When a model predicts a filament marked by only 1 of the 3 solar physicists:
* Matches Annotator 1: $+1\text{ TP}$.
* Against Annotators 2 and 3: **$+2\text{ False Positives}$**!
* Net marginal score: $\frac{0.75}{1 + 0.5(2)} = \mathbf{0.375} \ll 0.58$ (severely penalizes Panoptic Quality).

Purging 44 uncorroborated single-annotator fibrils and quiet-sun fallback boxes dropped instance count from 1,344 to **1,300 instances**, elevating the score from **0.55 to 0.58**.

---

## 4. Verification & Reproduction Protocol

### Step 1: Run Integrity Checklist
Verify that the primary 0.58 submission complies with all competition invariants:
```bash
python code/submission_integrity_check.py --sub_csv submissions/submission_selected_primary_058.csv --allow_missing_anchor
```
Expected output:
* Check 1: Panoptic Health: Area P25=607px, Median=1180px, P75=2528px [PASSED]
* Check 2: Column Schema: Columns = ['filament_id', 'segmentation_rle'], IDs formatted as `<stem>_<int>` [PASSED]
* Check 3: Quiet-Sun Guard: 0 rows on Quiet-Sun disks [PASSED]
* Check 4: Instance Density: 1300 rows across 178 active stems [PASSED]
* Check 5: Zero Overlap: Canvas collisions = 0 across 178 stems [PASSED]
* Check 6: Consensus Core Scaffold: Retained FIDs=1300/1300, Cannibalized=0 [PASSED]
* **OVERALL VERDICT: [PASSED - READY FOR KAGGLE LEADERBOARD DISPATCH]**

### Step 2: Submit to Kaggle via CLI
```bash
$env:KAGGLE_API_TOKEN="<YOUR_TOKEN>"; kaggle competitions submit -c filament-segmentation-2026 -f submissions/submission_selected_primary_058.csv -m "Championship Primary Selected 0.58 Baseline (1300 inst)"
```

### Step 3: Check Evaluated Score
```bash
$env:KAGGLE_API_TOKEN="<YOUR_TOKEN>"; kaggle competitions submissions -c filament-segmentation-2026
```
Verified result: **Public Score: `0.58`** (Status: `SubmissionStatus.COMPLETE`).
