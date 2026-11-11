# Project: IEEE BigData 2026 Solar Filament Segmentation Winning Blueprint

## Architecture
- **Pipeline Flow**:
  1. `Anchor Baseline` (`submission_original_055_anchor.csv`, 1,344 rows, 180 stems, LB 0.55).
  2. `Direction A (Anchor FP Pruner)`: Prune anchor instances with YOLO conf == 0 and H-alpha absorption deficit < 1.0 sigma. Remove 2 quiet-sun dummy boxes.
  3. `Direction B (Solo Inclusion Gate)`: Admit solo candidates satisfying Theorem 1 ($p \cdot j > 0.275$, H-alpha deficit $\ge 1.8\sigma$, elongation $\ge 2.0$, circularity $\le 0.45$, Sunspot Umbra Veto) via zero-collision priority stamping.
  4. `Spine-Guided Healing & MengxinLab Dilation Grouping`: Reconnect $1 \to M$ fractured filaments across bright fibrils within 30px without boundary bloat (`group_labels = np.where(foreground > 0, expanded_labels, 0)`).
  5. `Physics Boundary Snapping`: Local valley thresholding on 3px boundary band $B_k = \text{dilate}(M, 3) \setminus \text{erode}(M, 3)$ to shave chromospheric bleed and boost SQ.
  6. `Validation & Submission`: Fold 0 Zero-Trust Audit ($\Delta PQ \ge +0.0050$), generate full 180-stem submission CSV with exact COCO LEB128 Fortran RLE format, submit to Kaggle CLI, and track score.

## Feature Inventory
| # | Feature | Description | Milestone | Source |
|---|---------|-------------|-----------|--------|
| 1 | Submission Schema Validator Fix | Fix `submission_integrity_check.py` to validate `['filament_id', 'segmentation_rle']` and zero collisions | M0 | survey |
| 2 | Bit-Exact LEB128 Fortran RLE Engine | Pure-Python Fortran-order RLE encoder/decoder summing to 4,194,304 px | M0 | survey |
| 3 | Direction A: Anchor FP Pruner | Prune anchor masks with YOLO conf == 0 and deficit < 1.0 sigma | M1 | request & survey |
| 4 | Quiet Sun Guard | Remove 2 dummy fallback $48 \times 48$ boxes on quiet sun stems | M1 | survey |
| 5 | Direction B: Solo Inclusion Gate | Admit solo candidates satisfying Theorem 1 ($p \cdot j > 0.275$) | M2 | request & survey |
| 6 | Shape & Contrast Filters | PCA Elongation $\ge 2.0$, Circularity $\le 0.45$, Deficit $\ge 1.8\sigma$ | M2 | request & survey |
| 7 | Sunspot Umbra Veto | Reject dark circular absorbers ($\text{area} < 500 \land \text{elong} < 1.38 \land \text{circ} > 0.68$) | M2 | survey |
| 8 | Zero-Collision Priority Stamping | Disjoint canvas allocation ensuring $\forall i \ne j, M_i \cap M_j = \emptyset$ | M2 | survey |
| 9 | MengxinLab Dilation Grouping | Connect collinear fragments within 30px with 0% boundary bloat | M3 | request & survey |
| 10 | Spine-Guided FFBSi Bridge | Medial spine endpoint alignment and particle smoothing for fractures | M3 | request & survey |
| 11 | Physics Boundary Snapping | Local 50x50 MAD valley thresholding on 3px boundary band | M4 | request & survey |
| 12 | Fold 0 Zero-Trust Audit | Benchmark evaluation against Fold 0 multi-annotator ground truth | M5 | request & survey |
| 13 | 180-Stem Submission Generation | Build full 180-stem submission file in exact competition format | M5 | request & survey |
| 14 | Kaggle Submission & Score Tracking | Submit via Kaggle CLI and verify leaderboard evaluation status | M5 | request & survey |

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| M0 | Infrastructure & Schema Fix | Fix `scripts/submission_integrity_check.py`, verify LEB128 Fortran engine, verify anchor baseline | none | IN_PROGRESS |
| M1 | Direction A: Anchor FP Pruner | Prune uncorroborated anchor masks (YOLO conf == 0 & deficit < 1.0 sigma) and quiet-sun dummy boxes | M0 | PLANNED |
| M2 | Direction B: Solo Inclusion Gate | Implement Theorem 1 gate ($p \cdot j > 0.275$), deficit $\ge 1.8\sigma$, elongation, circularity, sunspot veto, zero-collision stamping | M1 | PLANNED |
| M3 | Spine-Guided Healing & Dilation Grouping | Bridge $1 \to M$ fractures across bright chromospheric fibrils within 30px with 0% boundary bloat | M2 | PLANNED |
| M4 | Physics Boundary Snapping | Local valley thresholding on 3px boundary band to shave bleed and boost SQ | M3 | PLANNED |
| M5 | Final E2E Pass, Audit & Kaggle Submission | 100% E2E test pass, Fold 0 zero-trust validation audit, 180-stem CSV generation, Kaggle submit & track | M4, E2E_TEST | PLANNED |

## Interface Contracts
### M0 ↔ All Modules
- `scripts/submission_integrity_check.py --sub_csv <csv_path>`:
  - Must return exit code 0 on valid submission.
  - Required columns: `filament_id,segmentation_rle`.
  - Stems: Exactly 180 unique stems.
  - Instances: Non-overlapping ($M_i \cap M_j = \emptyset$).
  - Quiet sun stems (`20160513133714Th`, `20180507140210Th`): 0 rows.
  - RLE: decodes to $(2048, 2048)$ with 4,194,304 pixels.

### M1 ↔ M2 (Direction A Output)
- Output CSV: `experiments/sub_m1_anchor_pruned.csv`
- Schema: `filament_id,segmentation_rle`
- Invariant: Subset of anchor masks; 2 quiet sun dummy boxes pruned; valid instances collision-free.

### M2 ↔ M3 (Direction B Output)
- Output CSV: `experiments/sub_m2_two_tier_gated.csv`
- Schema: `filament_id,segmentation_rle`
- Solos added: Only those satisfying Theorem 1 ($p \cdot j > 0.275$, deficit $\ge 1.8\sigma$, elongation $\ge 2.0$, circularity $\le 0.45$, not sunspot).

### M3 ↔ M4 (Healing Output)
- Output CSV: `experiments/sub_m3_healed.csv`
- Schema: `filament_id,segmentation_rle`
- Invariant: Zero boundary bloat (foreground pixels strictly conserved). Fractured instances within 30px merged into common IDs.

### M4 ↔ M5 (Refinement Output)
- Output CSV: `experiments/sub_m4_refined.csv`
- Schema: `filament_id,segmentation_rle`
- Shaves 1–3px chromospheric boundary bleed on $B_k = \text{dilate}(M, 3) \setminus \text{erode}(M, 3)$.

### M5 ↔ Kaggle
- Command: `kaggle competitions submit -c filament-segmentation-2026 -f <csv_path> -m "<description>"`
- Followed by tracking: `kaggle competitions submissions -c filament-segmentation-2026` until completed.

## Code Layout
- `scripts/`: Production execution modules
  - `submission_integrity_check.py`: Official integrity verification script
  - `utils_metrics.py`: Bit-exact COCO LEB128 Fortran RLE and PQ metrics
  - `utils_solar_physics.py`: Limb-darkening, PCA elongation, circularity, sunspot veto
  - `stage1_anchor_fp_pruner.py`: Direction A pruner
  - `stage1_solo_inclusion_gate.py`: Direction B Theorem 1 inclusion gate
  - `stage2_roggi_ffbsi_bridge.py`: Spine-guided healing & MengxinLab grouping
  - `stage3_physics_boundary_refiner.py`: Local valley boundary snapping
  - `step0_fold0_solo_truth_audit.py`: Fold 0 zero-trust validation audit
- `tests/e2e/`: Opaque-box E2E test suite (Tiers 1-4)
- `experiments/`: Intermediate outputs, audits, and submission candidates
- `submission_original_055_anchor.csv`: Champion baseline anchor
