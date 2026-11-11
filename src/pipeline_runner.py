#!/usr/bin/env python3
"""
src/pipeline_runner.py
Master runner for Phase 3:
1. Baseline 5-fold cross-validation
2. Annotator supervision comparison (Strategies A, B, C)
Outputs comprehensive structured reports and metrics.
"""

import argparse
import glob
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training.train_eval_fold import run_fold


def run_5fold_baseline(
    strategy: str = "A",
    train_json: str = "data/annotations/MAGFiLO_1.0_Kaggle_2026/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json",
    manifest: str = "data/train_folds_manifest.csv",
    images_dir: str = "data/train_images",
    epochs: int = 5,
    batch_size: int = 8,
    target_size: int = 512,
    device: str = "cuda",
    seed: int = 42,
) -> Dict:
    """Runs all 5 folds for a given strategy and aggregates metrics."""
    fold_results = []
    
    print(f"\n=======================================================")
    print(f"Executing 5-Fold Cross Validation | Strategy: {strategy}")
    print(f"=======================================================")
    
    for f in range(5):
        res = run_fold(
            fold=f,
            strategy=strategy,
            train_json_path=train_json,
            manifest_csv_path=manifest,
            images_dir=images_dir,
            epochs=epochs,
            batch_size=batch_size,
            target_size=target_size,
            device_str=device,
            seed=seed,
        )
        fold_results.append(res)
        
    pqs = [r["pq"] for r in fold_results]
    sqs = [r["sq"] for r in fold_results]
    rqs = [r["rq"] for r in fold_results]
    tps = sum(r["tp"] for r in fold_results)
    fps = sum(r["fp"] for r in fold_results)
    fns = sum(r["fn"] for r in fold_results)
    total_train_time = sum(r["train_time_seconds"] for r in fold_results)
    total_infer_time = sum(r["infer_time_seconds"] for r in fold_results)
    
    summary = {
        "experiment": f"5fold_baseline_strat_{strategy}",
        "strategy": strategy,
        "n_folds": 5,
        "mean_pq": float(np.mean(pqs)),
        "std_pq": float(np.std(pqs)),
        "min_pq": float(np.min(pqs)),
        "max_pq": float(np.max(pqs)),
        "mean_sq": float(np.mean(sqs)),
        "mean_rq": float(np.mean(rqs)),
        "total_tp": int(tps),
        "total_fp": int(fps),
        "total_fn": int(fns),
        "total_train_time_seconds": round(total_train_time, 2),
        "total_infer_time_seconds": round(total_infer_time, 2),
        "fold_results": fold_results,
    }
    
    out_path = f"results/summary_5fold_strat_{strategy}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        
    print(f"\n=== 5-Fold Summary for Strategy {strategy} ===")
    print(f"Mean PQ: {summary['mean_pq']:.5f} +/- {summary['std_pq']:.5f}")
    print(f"Mean SQ: {summary['mean_sq']:.5f} | Mean RQ: {summary['mean_rq']:.5f}")
    print(f"Total TP: {summary['total_tp']} | Total FP: {summary['total_fp']} | Total FN: {summary['total_fn']}")
    print(f"Total Train Time: {summary['total_train_time_seconds']:.1f}s | Infer Time: {summary['total_infer_time_seconds']:.1f}s")
    print(f"================================================\n")
    return summary


def run_supervision_comparison(
    folds: List[int] = [0],
    train_json: str = "data/annotations/MAGFiLO_1.0_Kaggle_2026/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json",
    manifest: str = "data/train_folds_manifest.csv",
    images_dir: str = "data/train_images",
    epochs: int = 5,
    batch_size: int = 8,
    target_size: int = 512,
    device: str = "cuda",
    seed: int = 42,
) -> Dict:
    """Runs Strategies A, B, and C across the designated folds for identical comparison."""
    comparison = {}
    for strat in ["A", "B", "C"]:
        strat_results = []
        for f in folds:
            res = run_fold(
                fold=f,
                strategy=strat,
                train_json_path=train_json,
                manifest_csv_path=manifest,
                images_dir=images_dir,
                epochs=epochs,
                batch_size=batch_size,
                target_size=target_size,
                device_str=device,
                seed=seed,
            )
            strat_results.append(res)
            
        pqs = [r["pq"] for r in strat_results]
        sqs = [r["sq"] for r in strat_results]
        rqs = [r["rq"] for r in strat_results]
        comparison[strat] = {
            "strategy": strat,
            "mean_pq": float(np.mean(pqs)),
            "std_pq": float(np.std(pqs)),
            "mean_sq": float(np.mean(sqs)),
            "mean_rq": float(np.mean(rqs)),
            "total_tp": int(sum(r["tp"] for r in strat_results)),
            "total_fp": int(sum(r["fp"] for r in strat_results)),
            "total_fn": int(sum(r["fn"] for r in strat_results)),
            "results_by_fold": strat_results,
        }
        
    out_path = "results/supervision_comparison.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)
        
    return comparison


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="5fold", choices=["5fold", "compare"])
    parser.add_argument("--strategy", type=str, default="A", choices=["A", "B", "C"])
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--target_size", type=int, default=512)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--train_json", type=str, default="data/annotations/MAGFiLO_1.0_Kaggle_2026/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json")
    parser.add_argument("--manifest", type=str, default="data/train_folds_manifest.csv")
    parser.add_argument("--images_dir", type=str, default="data/train_images")
    args = parser.parse_args()
    
    if args.mode == "5fold":
        run_5fold_baseline(
            strategy=args.strategy,
            train_json=args.train_json,
            manifest=args.manifest,
            images_dir=args.images_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            target_size=args.target_size,
            device=args.device,
        )
    elif args.mode == "compare":
        run_supervision_comparison(
            folds=[0],
            train_json=args.train_json,
            manifest=args.manifest,
            images_dir=args.images_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            target_size=args.target_size,
            device=args.device,
        )
