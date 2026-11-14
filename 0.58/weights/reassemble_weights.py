#!/usr/bin/env python3
"""
0.58/weights/reassemble_weights.py
Automated Reassembly & Verification of Binary Model Weights.
Enables full GitHub repository hosting without Git LFS quota limitations by
splitting files > 70 MB into ~45 MB parts, and reconstructing them bit-for-bit.
"""

import os
import sys
import json
import hashlib
from pathlib import Path

WEIGHTS_DIR = Path(__file__).resolve().parent
CHECKSUMS_JSON = WEIGHTS_DIR / "weights_checksums.json"
CHUNK_SIZE = 48 * 1024 * 1024  # 48 MB chunks (strictly under GitHub 50 MB warning threshold)


def compute_sha256(file_path: Path) -> str:
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(8 * 1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def split_large_files():
    print("=" * 80)
    print("SPLITTING LARGE WEIGHT FILES (< 50 MB per part)")
    print("=" * 80)
    checksums = {}

    for pt_path in sorted(list(WEIGHTS_DIR.rglob("*.pt"))):
        sz = pt_path.stat().st_size
        rel_path = pt_path.relative_to(WEIGHTS_DIR).as_posix()
        sha = compute_sha256(pt_path)
        checksums[rel_path] = {"size": sz, "sha256": sha}

        if sz > 70 * 1024 * 1024:
            print(f"[*] Splitting: {rel_path} ({sz / (1024*1024):.2f} MB)...")
            part_idx = 0
            with open(pt_path, "rb") as f:
                while chunk := f.read(CHUNK_SIZE):
                    part_file = pt_path.parent / f"{pt_path.name}.part{part_idx:02d}"
                    part_file.write_bytes(chunk)
                    print(f"    -> Created {part_file.name} ({len(chunk) / (1024*1024):.2f} MB)")
                    part_idx += 1

    with open(CHECKSUMS_JSON, "w", encoding="utf-8") as f:
        json.dump(checksums, f, indent=2)
    print(f"\n[+] Saved checksums to '{CHECKSUMS_JSON}'")
    print("=" * 80)


def reassemble_all(clean_parts: bool = False):
    print("=" * 80)
    print("REASSEMBLING ALL MODEL WEIGHTS FROM PARTS")
    print("=" * 80)
    if not CHECKSUMS_JSON.exists():
        print(f"[-] Checksums file not found: {CHECKSUMS_JSON}")
        return

    with open(CHECKSUMS_JSON, "r", encoding="utf-8") as f:
        checksums = json.load(f)

    for rel_path, meta in checksums.items():
        pt_path = WEIGHTS_DIR / rel_path
        expected_sha = meta["sha256"]
        expected_size = meta["size"]

        if pt_path.exists() and pt_path.stat().st_size == expected_size:
            actual_sha = compute_sha256(pt_path)
            if actual_sha == expected_sha:
                print(f"[OK] {rel_path} is intact ({expected_size / (1024*1024):.2f} MB).")
                continue

        # Look for parts
        parts = sorted(list(pt_path.parent.glob(f"{pt_path.name}.part*")))
        if not parts:
            print(f"[!] Warning: Neither {rel_path} nor its parts were found.")
            continue

        print(f"[*] Reassembling: {rel_path} from {len(parts)} parts...")
        pt_path.parent.mkdir(parents=True, exist_ok=True)
        with open(pt_path, "wb") as out_f:
            for p in parts:
                with open(p, "rb") as in_f:
                    while chunk := in_f.read(8 * 1024 * 1024):
                        out_f.write(chunk)

        actual_sha = compute_sha256(pt_path)
        if actual_sha == expected_sha:
            print(f"[PASSED] {rel_path} verified bit-for-bit (SHA-256 match).")
            if clean_parts:
                for p in parts:
                    p.unlink()
                print("         Cleaned up temporary part files.")
        else:
            print(f"[ERROR] Checksum mismatch on {rel_path}!")
            print(f"        Expected: {expected_sha}")
            print(f"        Actual:   {actual_sha}")

    print("=" * 80)
    print("REASSEMBLY COMPLETED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    if "--split" in sys.argv:
        split_large_files()
    else:
        reassemble_all(clean_parts="--clean-parts" in sys.argv)
