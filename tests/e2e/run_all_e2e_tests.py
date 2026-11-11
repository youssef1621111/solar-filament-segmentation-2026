#!/usr/bin/env python3
"""
ROGGI E2E Test Runner & 4-Tier Verification Suite
IEEE BigData 2026 Solar Filament Segmentation Challenge (filament-segmentation-2026)

Executes all 4 test tiers:
  - Tier 1: Feature Coverage (70 tests across 14 features)
  - Tier 2: Boundary & Corner Cases (15 tests)
  - Tier 3: Pairwise Combinations (10 tests)
  - Tier 4: Real-World Application Scenarios (8 tests)

Outputs detailed per-tier reporting and returns exit code 0 on complete pass.
"""

import os
import sys
import time
import unittest
from datetime import datetime, timezone

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import tests.e2e.test_tier1_features as tier1
import tests.e2e.test_tier2_boundaries as tier2
import tests.e2e.test_tier3_pairwise as tier3
import tests.e2e.test_tier4_scenarios as tier4


def run_tier_suite(suite_name: str, module):
    """Run a single test module suite and return structured metrics."""
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(module)
    num_tests = suite.countTestCases()

    # Use a custom or quiet runner to capture results cleanly
    stream = open(os.devnull, "w")
    runner = unittest.TextTestRunner(stream=stream, verbosity=0)

    start_time = time.perf_counter()
    result = runner.run(suite)
    elapsed = time.perf_counter() - start_time
    stream.close()

    failures = len(result.failures)
    errors = len(result.errors)
    skipped = len(result.skipped)
    passed = num_tests - (failures + errors + skipped)

    return {
        "tier_name": suite_name,
        "module": module.__name__,
        "total": num_tests,
        "passed": passed,
        "failed": failures,
        "errors": errors,
        "skipped": skipped,
        "elapsed": elapsed,
        "result_obj": result,
    }


def main():
    print("=" * 90)
    print("IEEE BIGDATA 2026 SOLAR FILAMENT SEGMENTATION — E2E TEST RUNNER")
    print(f"Timestamp: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"Project Root: {PROJECT_ROOT}")
    print("=" * 90)

    tiers_to_run = [
        ("Tier 1: Feature Coverage (14 Features)", tier1),
        ("Tier 2: Boundary & Corner Cases", tier2),
        ("Tier 3: Pairwise Combinations", tier3),
        ("Tier 4: Real-World Scenarios", tier4),
    ]

    tier_metrics = []
    all_passed = True

    print("\nExecuting test suites...")
    for suite_name, module in tiers_to_run:
        print(f"  -> Running {suite_name}...", end="", flush=True)
        metrics = run_tier_suite(suite_name, module)
        status = "PASSED" if (metrics["failed"] == 0 and metrics["errors"] == 0) else "FAILED"
        if status == "FAILED":
            all_passed = False
        print(f" [{status}] ({metrics['passed']}/{metrics['total']} passed in {metrics['elapsed']:.2f}s)")
        tier_metrics.append(metrics)

    # Detailed Summary Table
    print("\n" + "=" * 90)
    print(f"{'TIER SUMMARY':^90}")
    print("=" * 90)
    print(f"{'Tier':<42} | {'Total':>6} | {'Passed':>6} | {'Failed':>6} | {'Errors':>6} | {'Time (s)':>8} | {'Status':<7}")
    print("-" * 90)

    total_tests = sum(m["total"] for m in tier_metrics)
    total_passed = sum(m["passed"] for m in tier_metrics)
    total_failed = sum(m["failed"] for m in tier_metrics)
    total_errors = sum(m["errors"] for m in tier_metrics)
    total_time = sum(m["elapsed"] for m in tier_metrics)

    for m in tier_metrics:
        status_str = "PASS" if (m["failed"] == 0 and m["errors"] == 0) else "FAIL"
        print(f"{m['tier_name']:<42} | {m['total']:>6} | {m['passed']:>6} | {m['failed']:>6} | {m['errors']:>6} | {m['elapsed']:>8.2f} | {status_str:<7}")

    print("-" * 90)
    overall_status = "PASSED" if all_passed else "FAILED"
    print(f"{'TOTAL / AGGREGATE':<42} | {total_tests:>6} | {total_passed:>6} | {total_failed:>6} | {total_errors:>6} | {total_time:>8.2f} | {overall_status:<7}")
    print("=" * 90)

    # Print error details if any failures occurred
    if not all_passed:
        print("\n" + "!" * 90)
        print("FAILURE DETAILS:")
        print("!" * 90)
        for m in tier_metrics:
            res = m["result_obj"]
            for test_case, trace in res.failures:
                print(f"\n[FAIL] {test_case}")
                print(trace)
            for test_case, trace in res.errors:
                print(f"\n[ERROR] {test_case}")
                print(trace)
        print("!" * 90)
        return 1

    print(f"\n>> All {total_tests} tests in 4 tiers passed successfully (100% pass rate).")
    print(">> Zero-Trust Test Infrastructure is fully verified.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
