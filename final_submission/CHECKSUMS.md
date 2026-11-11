# Final Selected Submissions — Checksum & Metadata Manifest
## Competition: Solar Filament Segmentation Challenge 2026 (IEEE BigData Cup)
## Team: cracker_1_war

---

### Primary Final Selection: Ref `58203935`
- **File Name**: `submission_selected_primary_058.csv`
- **Source**: `results/operation_058_breakthrough/predictions/submission_exp031_calibrated_1300.csv`
- **Kaggle Evaluation**: **0.58 Public Leaderboard**
- **Instance Count**: Exactly 1,300 instances
- **Active Stems**: 178 / 180 (7.30 instances / stem)
- **Quiet-Sun Rows**: Exactly 0 rows on `20160513133714Th` and `20180507140210Th`
- **Collisions**: Exactly 0 pixel collisions ($\forall i \ne j, M_i \cap M_j = \emptyset$)
- **File Size**: 372,491 bytes
- **Cryptographic Hashes**:
  - **SHA-256**: `0d527e838724cf8b57fde09babd2db13f8264de14e33bd8ae9c776299c4b47a3`
  - **MD5**: `b89e97ef67d6c50e85e96bcd472a1741`

---

### Secondary Final Selection: Ref `58234966`
- **File Name**: `submission_selected_secondary_058.csv`
- **Source**: `results/operation_058_breakthrough/predictions/submission_exp039_topological_consensus_1333.csv`
- **Kaggle Evaluation**: **0.58 Public Leaderboard**
- **Instance Count**: 1,333 instances
- **Active Stems**: 178 / 180 (7.49 instances / stem)
- **Quiet-Sun Rows**: Exactly 0 rows
- **Collisions**: Exactly 0 pixel collisions
- **Anchor Retention**: 100.0% (1,300/1,300 anchor instances preserved bit-for-bit)
- **New Candidates**: 9 soft-clDice multi-annotator consensus candidates stamped on free canvas
- **File Size**: 377,478 bytes
- **Cryptographic Hashes**:
  - **SHA-256**: `ad99e3fdfe189c3ce3ca761bfb9dd61a357446f6a4e3fac00135185cdb6b2b1f`
  - **MD5**: `738922078835a3181da97b2e81c714a8`

---

### Official Master Technical Handoff Document (4-Page PDF)
- **File Name**: `CHAMPIONSHIP_TECHNICAL_HANDOFF_4PAGE.pdf`
- **Location**: `final_submission/CHAMPIONSHIP_TECHNICAL_HANDOFF_4PAGE.pdf`
- **Page Count**: Exactly 4 pages (Letter format, high-density vector layout)
- **Scope**: Complete technical documentation, MAGFiLO PQ metric mathematics, multi-annotator disagreement penalty proof, end-to-end architecture, Soft-clDice loss formulas, 42-experiment forensic matrix, 5 failure modes root-cause analysis, 3 sacred invariants, 6-point integrity audit table, 5-tier adversarial stress test results, and Kaggle replication/selection instructions.
- **Cryptographic Hashes**:
  - **SHA-256**: `23d3624ccc24549045b1978280395a359d8ea0ee38a93c1fd134d2538e4173c7`
  - **MD5**: `79e58683d53f13d3d27d51ec4c6d5160`

---

### Zero-Trust 6-Point Audit Checklist:
Both submissions have been passed through `scripts/submission_integrity_check.py` and scored 6/6 `[PASSED]`:
1. [PASSED] Check 1: Panoptic Health (Area distributions within natural solar filament bounds)
2. [PASSED] Check 2: Column Schema strictly `['filament_id', 'segmentation_rle']`
3. [PASSED] Check 3: Quiet-Sun Guard (Strictly 0 rows on quiescent disks)
4. [PASSED] Check 4: Instance Density (7.30 - 7.49 instances / stem)
5. [PASSED] Check 5: Zero Overlap (Zero negative runs, sum to 4,194,304, zero collisions)
6. [PASSED] Check 6: Anchor Scaffold (1,300/1,300 FIDs retained, 0 cannibalized, 0 distorted)
