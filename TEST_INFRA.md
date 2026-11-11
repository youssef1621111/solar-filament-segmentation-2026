# Comprehensive Test Infrastructure Specification (TEST_INFRA.md)
## IEEE BigData 2026 Solar Filament Segmentation Challenge (`filament-segmentation-2026`)

**Author**: `teamwork_test_writer_e2e_1`  
**Scope**: 4-Tier Opaque-Box E2E Test Suite & Quality Infrastructure  
**Target Root**: `/home/fathyyoussef162/`

---

## 1. Test Philosophy & Principles

The Solar Filament Segmentation test infrastructure is governed by five foundational testing principles:

1. **Opaque-Box Architectural Decoupling**:
   Tests treat the model pipelines, segmentation networks, and post-processing filters as black/opaque boxes. Tests evaluate interface contracts, mathematical invariants, input/output schemas, and metric properties rather than internal private variables.

2. **Progressive Testability & Zero-Trust Verification**:
   Every test milestone must be verifiable using only features from the current milestone and completed dependencies. Every claimed performance improvement must satisfy the strict Fold 0 Zero-Trust Gate:
   $$\Delta PQ \ge +0.0050, \quad \Delta TP > \Delta FP, \quad \forall m \in \text{TP}_{\text{anchor}}, \text{IoU}(m, \text{GT}) \ge 0.52$$

3. **Mathematical & Physics Invariant Enforcement**:
   - **Theorem 1 Admission Gate**: Solo detection candidates must satisfy $p \cdot j > 0.275$ to guarantee $\mathbb{E}[PQ_{\text{new}}] > 0.55$.
   - **Boundary Bloat Zero-Tolerance**: MengxinLab dilation grouping must satisfy $\Delta \text{Foreground Pixels} = 0$. Dilation changes instance membership, never mask boundaries.
   - **Disjoint Canvas Invariant**: Across all 180 observation stems, instances must satisfy $\forall i \ne j, M_i \cap M_j = \emptyset$.
   - **Solar Physics Constraints**: Filaments must exhibit $H\alpha$ absorption contrast $\ge 1.8\sigma$ relative to background, elongation $\ge 2.0$, circularity $\le 0.45$, and pass the Sunspot Umbra Veto.

4. **Bit-Exact Standard Compliance**:
   All RLE representations must comply bit-for-bit with the official COCO LEB128 Fortran-order compressed RLE standard (`maskApi.c`). Every 2048x2048 canvas must sum to exactly $4,194,304$ uncompressed pixels in column-major order.

5. **Self-Contained & Isolated Execution**:
   Each test must create and manage its own state, execute without order dependencies, and clean up transient resources. The test suite operates with zero external network or GPU requirements, executing via standard Python testing frameworks.

---

## 2. Feature Inventory (14 Features)

All 14 features from `PROJECT.md` § Feature Inventory are covered by the test infrastructure:

| Feature ID | Feature Name | Description | Source / Milestone | Primary Contract / Mathematical Invariant |
|---|---|---|---|---|
| **F-01** | Submission Schema Validator Fix | Correct column verification in `submission_integrity_check.py` | M0 (survey) | Columns: `['filament_id', 'segmentation_rle']`; 0 pixel collisions |
| **F-02** | Bit-Exact LEB128 Fortran RLE Engine | Pure-Python Fortran-order RLE encoder/decoder | M0 (survey) | Sum of counts $\equiv 4,194,304$; exact COCO LEB128 compliance |
| **F-03** | Direction A: Anchor FP Pruner | Prune uncorroborated anchor false positives | M1 (request/survey) | Prune iff $(\text{YOLO\_conf} == 0) \land (\text{Deficit} < 1.0\sigma)$ |
| **F-04** | Quiet Sun Guard | Remove 2 dummy fallback $48\times 48$ boxes | M1 (survey) | Quiet-sun stems (`20160513133714Th`, `20180507140210Th`) have 0 rows |
| **F-05** | Direction B: Solo Inclusion Gate | Admit solo candidates satisfying Theorem 1 | M2 (request/survey) | Admit iff $p \cdot j > 0.275 \implies \mathbb{E}[\Delta PQ] > 0$ |
| **F-06** | Shape & Contrast Filters | Morphological and radiometric quality filters | M2 (request/survey) | $\text{Elongation} \ge 2.0$, $\text{Circularity} \le 0.45$, $\text{Deficit} \ge 1.8\sigma$ |
| **F-07** | Sunspot Umbra Veto | Reject dark circular absorbers | M2 (survey) | Veto iff $\text{area} < 500 \land \text{elong} < 1.38 \land \text{circ} > 0.68$ |
| **F-08** | Zero-Collision Priority Stamping | Disjoint canvas allocation across tiers | M2 (survey) | Disjoint canvas: $\forall i \ne j, M_i \cap M_j = \emptyset$; Tier 1 priority |
| **F-09** | MengxinLab Dilation Grouping | Bridge collinear fractures within 30px | M3 (request/survey) | `group_labels = np.where(foreground > 0, expanded_labels, 0)`; 0% bloat |
| **F-10** | Spine-Guided FFBSi Bridge | Medial spine endpoint alignment & particle bridge | M3 (request/survey) | Gap $\le 30\text{px}$, tangent alignment $\ge \cos(30^\circ)$, evidence $\ge \tau_{\text{bridge}}$ |
| **F-11** | Physics Boundary Snapping | Local 50x50 MAD valley boundary refinement | M4 (request/survey) | Refine $B_k = \text{dilate}(M, 3) \setminus \text{erode}(M, 3)$; preserve core |
| **F-12** | Fold 0 Zero-Trust Audit | Benchmark evaluation against multi-annotator GT | M5 (request/survey) | Strict $\text{IoU} > 0.50$ matching; $\Delta PQ \ge +0.0050$; $\Delta TP > \Delta FP$ |
| **F-13** | 180-Stem Submission Generation | Build full 180-stem submission file | M5 (request/survey) | Exactly 180 stems; rows $\in [1380, 1560]$; valid RLE strings |
| **F-14** | Kaggle Submission & Score Tracking | Kaggle CLI submit and status verification | M5 (request/survey) | Valid submission command, CLI tracking, error recovery |

---

## 3. 4-Tier Test Architecture

The test suite in `tests/e2e/` is structured into four progressive tiers:

```
tests/e2e/
├── __init__.py
├── test_tier1_features.py       # Tier 1: Feature Coverage (>= 5 tests per feature, >= 70 total)
├── test_tier2_boundaries.py     # Tier 2: Boundary & Corner Cases (15+ tests)
├── test_tier3_pairwise.py       # Tier 3: Pairwise Combinations (10+ tests)
├── test_tier4_scenarios.py      # Tier 4: Real-World Scenarios (8+ tests)
└── run_all_e2e_tests.py         # Unified Test Runner & Tier Summary Reporter
```

### Tier 1: Feature Coverage
- **Objective**: Exhaustively exercise the primary behavior, contracts, and error paths of all 14 features.
- **Coverage Requirement**: $\ge 5$ distinct test cases per feature $\implies \ge 70$ test cases total.
- **Scope**:
  - F-01: Schema validation, column checking, collision checking, stem extraction, density bounds.
  - F-02: Roundtrip RLE encoding/decoding, pixel sum exactness, empty mask, origin pixel, Fortran order.
  - F-03: Pruning uncorroborated FPs, retaining corroborated instances, mathematical RQ proof, high-confidence anchor retention.
  - F-04: Quiet sun identification, 48x48 fallback box pruning, 0-row verification, active stem protection, PQ impact proof.
  - F-05: Theorem 1 admission ($p \cdot j > 0.275$), rejection ($p \cdot j \le 0.275$), expectation proof, regression prevention, inflection boundary.
  - F-06: Elongation calculation, circularity calculation, deficit calculation, non-filament blob rejection, thin filament acceptance.
  - F-07: Sunspot veto triggering, elongated filament protection, large feature protection, area boundary, anchor zero-veto guarantee.
  - F-08: Disjoint canvas stamping, Tier 1 priority preservation, Tier 2 residual clipping, minimum area discard, empty mask handling.
  - F-09: Collinear fracture grouping within 30px, 0% boundary bloat invariant, distant fragment separation, single fragment preservation, shape fidelity.
  - F-10: Skeleton ridge extraction, tangent vector calculation, bridge candidate detection, misaligned pair rejection, FFBSi particle smoothing.
  - F-11: Local 50x50 MAD statistics, boundary band definition, core erosion preservation, chromospheric bleed trimming, SQ boost.
  - F-12: Multi-annotator IoU > 0.50 matching, PQ/SQ/RQ derivation, Zero-Trust Gate 1 ($\Delta PQ \ge 0.0050$), Gate 2 ($\Delta TP > \Delta FP$), Gate 3 ($\text{IoU} \ge 0.52$).
  - F-13: 180 stems exact count, row count in $[1380, 1560]$, filament_id format, RLE validity, quiet-sun 0 rows.
  - F-14: Kaggle CLI submission command formatting, status tracking options, output parsing, score extraction, schema error catching.

### Tier 2: Boundary & Corner Cases
- **Objective**: Stress-test extreme mathematical and spatial boundary conditions.
- **Scope**:
  - B-01: All-zero empty mask ($2048 \times 2048 \implies$ count `[4194304]`, RLE `'PPPP4'`).
  - B-02: All-ones full mask ($2048 \times 2048 \implies$ count `[0, 4194304]`).
  - B-03: Foreground at origin $(0, 0) == 1$ (requires leading zero in COCO specification).
  - B-04: Foreground at extreme corner $(2047, 2047) == 1$.
  - B-05: Memory layout contrast: C-order (transposition, IoU 0.0) vs Fortran-order (IoU 1.0).
  - B-06: Quiet sun stems with 0 rows (handles empty dataframe without crashing).
  - B-07: Solo candidate area at boundary: area 699 (rejected) vs 700 (evaluated).
  - B-08: Sunspot veto threshold limits: area 499 vs 500, elong 1.37 vs 1.39, circ 0.67 vs 0.69.
  - B-09: Single-pixel fracture (1px gap bridged with 0% bloat).
  - B-10: Exact 30px fracture distance (bridged) vs 31px distance (not bridged).
  - B-11: Narrow 2px filament barb where erosion core is empty (maintains continuity).
  - B-12: Tangent angle alignment at exact threshold $\cos(30^\circ) \approx 0.866$.
  - B-13: Theorem 1 threshold at exact value $p \cdot j = 0.275000$.
  - B-14: Single instance stem canvas (no collision partner).
  - B-15: Extreme aspect ratio filament ($\text{elongation} > 15.0$).

### Tier 3: Pairwise Combinations
- **Objective**: Verify that pipeline stages interact without destructive side-effects or contract violations.
- **Scope**:
  - P-01: Anchor Pruner + Solo Inclusion Gate (simultaneous pruning and admission; no slot collision).
  - P-02: Solo Gating + Zero-Collision Stamping (solos stamped only into free space; anchor unaffected).
  - P-03: Solo Gating + Sunspot Umbra Veto (two-stage filter: candidate must pass both).
  - P-04: Spine-Guided Healing + MengxinLab Dilation Grouping (FFBSi confirms bridge, MengxinLab unifies ID without boundary bloat).
  - P-05: MengxinLab Grouping + Boundary Snapping (grouped instances refined without re-splitting).
  - P-06: Boundary Snapping + Zero-Collision Stamping (3px dilation band does not invade neighboring instances).
  - P-07: Quiet Sun Guard + 180-Stem Submission Generation (quiet sun stripped; 180 stems maintained).
  - P-08: Fortran RLE Engine + Submission Schema Validator (validator decodes RLE and checks overlap).
  - P-09: Theorem 1 Gate + Fold 0 Zero-Trust Audit (admitted solos strictly improve Fold 0 benchmark PQ).
  - P-10: Full E2E Pipeline Combination: Prune $\to$ Gate $\to$ Heal $\to$ Snap $\to$ Validate.

### Tier 4: Real-World Scenarios
- **Objective**: Test against real-world competition data assets and end-to-end execution scenarios.
- **Scope**:
  - S-01: Real 0.55 Anchor Baseline Audit (`submission_original_055_anchor.csv`): 1,344 rows, 180 stems, 0 collisions on active stems, 2 quiet-sun dummy boxes.
  - S-02: Quiet Sun Guard Remediation: complete removal of 2 dummy boxes, yielding 1,342 real rows and 0 rows on quiet sun.
  - S-03: Real-World 180-Stem Blueprint Submission Candidate: full submission evaluation passing all 6 integrity checks.
  - S-04: Real-World Regression Detection: simulates the uncalibrated consensus merge that degraded LB from 0.55 to 0.53; verifies test suite detects RQ drop and blocks submission.
  - S-05: Fold 0 Multi-Annotator Ground Truth Audit: evaluates `experiments/fold0_solo_truth_audit.csv` against Zero-Trust gates.
  - S-06: Zero Pixel Collision across full 180-stem canvas.
  - S-07: Competition Submission Schema & Kaggle CLI submission packaging.
  - S-08: Bit-Exact pycocotools Interoperability on real anchor masks.

---

## 4. Coverage Thresholds and Quality Gates

| Metric | Required Threshold | Verification Method |
|---|---|---|
| **Tier 1 Feature Coverage** | 100% (14 / 14 features, $\ge 5$ tests each) | `tests/e2e/run_all_e2e_tests.py` |
| **Tier 2 Boundary Tests** | 100% pass rate (15 / 15 tests) | `tests/e2e/run_all_e2e_tests.py` |
| **Tier 3 Pairwise Tests** | 100% pass rate (10 / 10 tests) | `tests/e2e/run_all_e2e_tests.py` |
| **Tier 4 Scenario Tests** | 100% pass rate (8 / 8 tests) | `tests/e2e/run_all_e2e_tests.py` |
| **Zero-Collision Invariant** | 0 overlapping pixels across all stems | Decoded boolean canvas intersection |
| **LEB128 Fortran Pixel Sum** | Exactly 4,194,304 uncompressed pixels | `sum(mask_to_uncompressed_rle(mask)) == 4194304` |
| **Quiet Sun Instance Count** | Exactly 0 rows for stems `20160513133714Th` & `20180507140210Th` | Schema validator Check 3 |
| **Instance Density Target** | Total rows $\in [1380, 1560]$ | Count of submission dataframe rows |
| **Zero Boundary Bloat** | $\Delta \text{Foreground Area} \equiv 0$ on dilation grouping | Dilation labels projected to foreground |

---

## 5. Test Execution Protocol

To run the complete test suite:
```bash
python3 tests/e2e/run_all_e2e_tests.py
```

Expected output:
- Individual test progress across all 4 tiers
- Detailed per-tier summary table (Tests Run, Passed, Failed, Errored, Success Rate)
- Final aggregate status and exit code 0 on complete pass.
