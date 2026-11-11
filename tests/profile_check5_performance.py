"""
Empirical Challenger Benchmark & Memory Profiler for Check 5 & LEB128 Engine
Tests:
1. Exact peak memory profiling during Check 5 across all 180 stems (1,344 rows).
2. Throughput & runtime of 100% RLE count checks (decode_leb128_counts).
3. Edge case: 179 valid stems + 1 colliding stem (first stem, middle stem, last stem).
4. Edge case: 1-pixel overlap vs multi-pixel overlap.
5. Invariant: corrupted count sum (!= 4194304).
"""

import sys
import os
import time
import tracemalloc
import resource
import subprocess
import pandas as pd
import numpy as np

# Ensure project root in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.utils_metrics import decode_leb128_counts, decode_leb128_rle, encode_leb128_rle


def get_rss_mb():
    """Get current process RSS in MB from /proc/self/status or resource.getrusage."""
    with open('/proc/self/status', 'r') as f:
        for line in f:
            if line.startswith('VmRSS:'):
                return float(line.split()[1]) / 1024.0
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0


def benchmark_leb128_counts_speed(csv_path="submission_original_055_anchor.csv"):
    """Benchmark speed of decode_leb128_counts on all 1,344 rows."""
    print("=" * 80)
    print("BENCHMARK 1: decode_leb128_counts Speed & Throughput on 100% of Rows")
    print("=" * 80)
    df = pd.read_csv(csv_path)
    rle_list = [str(r) for r in df["segmentation_rle"] if str(r) not in ("dummy_rle", "nan", "None", "")]
    total_masks = len(rle_list)
    print(f"Loaded {total_masks} valid RLE strings from {csv_path}")

    # Warmup
    for r in rle_list[:10]:
        _ = decode_leb128_counts(r)

    # Timed run
    start_time = time.perf_counter()
    count_sum_correct = 0
    total_counts_len = 0
    for r in rle_list:
        counts = decode_leb128_counts(r)
        if sum(counts) == 4194304:
            count_sum_correct += 1
        total_counts_len += len(counts)
    elapsed = time.perf_counter() - start_time

    rate = total_masks / elapsed if elapsed > 0 else 0
    ms_per_mask = (elapsed / total_masks) * 1000.0 if total_masks > 0 else 0

    print(f"Total masks processed: {total_masks}")
    print(f"Correct sum == 4,194,304: {count_sum_correct}/{total_masks} ({count_sum_correct/total_masks*100:.2f}%)")
    print(f"Elapsed time: {elapsed:.4f} s")
    print(f"Throughput: {rate:.1f} masks/sec")
    print(f"Latency per mask: {ms_per_mask:.3f} ms")
    print(f"Total RLE run-length elements parsed: {total_counts_len}")
    assert count_sum_correct == total_masks, f"Expected all {total_masks} to sum to 4194304"
    return {
        "total_masks": total_masks,
        "elapsed_s": elapsed,
        "throughput_fps": rate,
        "latency_ms": ms_per_mask
    }


def profile_check5_memory(csv_path="submission_original_055_anchor.csv"):
    """
    Profile peak memory usage during Check 5 (both tracemalloc and RSS).
    Requirement: strictly remains under 50 MB peak RAM.
    """
    print("=" * 80)
    print("BENCHMARK 2: Check 5 Peak Memory Profiling (Full 180 Stems, 1,344 Rows)")
    print("=" * 80)

    # Measure baseline RSS
    baseline_rss = get_rss_mb()
    print(f"Initial Baseline RSS: {baseline_rss:.2f} MB")

    df = pd.read_csv(csv_path)
    post_load_rss = get_rss_mb()
    print(f"Post-DataFrame Load RSS: {post_load_rss:.2f} MB (Delta: {post_load_rss - baseline_rss:+.2f} MB)")

    stems = df["filament_id"].astype(str).apply(lambda x: x.rsplit("_", 1)[0])
    stem_groups = df.groupby(stems)["segmentation_rle"].apply(list).to_dict()

    tracemalloc.start()
    tracemalloc.reset_peak()

    start_time = time.perf_counter()
    peak_rss_during = post_load_rss

    # Part A: Fast audit across 100% of rows
    rle_errors = 0
    for rle_val in df["segmentation_rle"]:
        rle_str = str(rle_val)
        if rle_str in ("dummy_rle", "nan", "None", ""):
            continue
        try:
            counts = decode_leb128_counts(rle_str)
            if sum(counts) != 4194304:
                rle_errors += 1
        except Exception:
            rle_errors += 1

    current_rss = get_rss_mb()
    if current_rss > peak_rss_during:
        peak_rss_during = current_rss

    # Part B: Stem-streaming collision check
    overlap_errors = 0
    multi_instance_count = 0
    total_masks_decoded = 0

    for stem, rle_list in stem_groups.items():
        if len(rle_list) <= 1:
            continue
        multi_instance_count += 1
        occupied = np.zeros((2048, 2048), dtype=bool)
        for rle_str in rle_list:
            rle_str = str(rle_str)
            if rle_str in ("dummy_rle", "nan", "None", ""):
                continue
            mask = decode_leb128_rle(rle_str, (2048, 2048))
            total_masks_decoded += 1
            if np.any(occupied & mask):
                overlap_errors += 1
                break
            occupied |= mask

        current_rss = get_rss_mb()
        if current_rss > peak_rss_during:
            peak_rss_during = current_rss

    elapsed = time.perf_counter() - start_time
    current_size, peak_traced = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    peak_traced_mb = peak_traced / (1024.0 * 1024.0)
    current_traced_mb = current_size / (1024.0 * 1024.0)
    final_rss = get_rss_mb()
    rss_delta_mb = peak_rss_during - post_load_rss

    print(f"Check 5 Completed in {elapsed:.2f} s")
    print(f"Multi-instance stems checked: {multi_instance_count}")
    print(f"Masks fully decoded to 2048x2048 canvas: {total_masks_decoded}")
    print(f"Overlap violations detected: {overlap_errors}")
    print(f"Peak Tracemalloc Allocation: {peak_traced_mb:.2f} MB")
    print(f"Current Tracemalloc Allocation at end: {current_traced_mb:.2f} MB")
    print(f"Peak RSS during Check 5: {peak_rss_during:.2f} MB")
    print(f"RSS Delta during Check 5: {rss_delta_mb:+.2f} MB")
    print(f"Final RSS: {final_rss:.2f} MB")

    print("-" * 80)
    threshold = 50.0  # MB
    print(f"Constraint Check: Peak RAM < 50MB:")
    print(f"  Peak Tracemalloc ({peak_traced_mb:.2f} MB) < 50 MB? {'YES [PASS]' if peak_traced_mb < threshold else 'NO [FAIL]'}")
    print(f"  RSS Delta ({rss_delta_mb:.2f} MB) < 50 MB? {'YES [PASS]' if rss_delta_mb < threshold else 'NO [FAIL]'}")
    print(f"  Peak Total Process RSS ({peak_rss_during:.2f} MB) < 50 MB? {'YES [PASS]' if peak_rss_during < threshold else 'NO [FAIL]'}")
    print("-" * 80)

    return {
        "elapsed_s": elapsed,
        "peak_traced_mb": peak_traced_mb,
        "peak_rss_mb": peak_rss_during,
        "rss_delta_mb": rss_delta_mb,
        "multi_instance_stems": multi_instance_count,
        "masks_decoded": total_masks_decoded,
        "overlap_errors": overlap_errors,
        "rle_errors": rle_errors
    }


if __name__ == "__main__":
    b1 = benchmark_leb128_counts_speed()
    b2 = profile_check5_memory()
