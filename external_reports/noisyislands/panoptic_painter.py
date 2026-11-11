import numpy as np
import pandas as pd
import pycocotools.mask as mask_util
import cv2

def paint_panoptic(
    candidates: list[tuple[float, np.ndarray]],
    min_area: int = 150,
    H: int = 2048,
    W: int = 2048,
) -> list[tuple[float, np.ndarray, int]]:
    """
    Score-ordered panoptic painting to resolve pixel overlap.
    
    Args:
        candidates: List of (confidence_score: float, binary_mask: np.ndarray uint8 [H, W])
        min_area: Minimum pixel area to retain an instance after subtraction.
        H, W: Image height and width (default 2048).

    Returns:
        List of (score, disjoint_binary_mask, area)
    """
    # Sort candidates by score descending
    candidates = sorted(candidates, key=lambda x: -x[0])
    claimed = np.zeros((H, W), dtype=np.uint8)
    kept = []

    for score, binary in candidates:
        if binary.dtype != np.uint8:
            binary = binary.astype(np.uint8)

        # Subtract pixels already claimed by higher-scoring detections
        remaining = binary & (~claimed)
        area = int(remaining.sum())

        if area < min_area:
            continue

        # Optional: check morphological elongation (filaments are elongated, sunspots are compact)
        # We can compute connected components on remaining in case subtraction split it
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(remaining, connectivity=8)
        valid_components = []
        for l in range(1, num_labels):
            comp_area = stats[l, cv2.CC_STAT_AREA]
            if comp_area >= min_area:
                comp_mask = (labels == l).astype(np.uint8)
                valid_components.append((comp_area, comp_mask))

        if not valid_components:
            continue

        # Re-merge valid components or keep largest
        final_mask = np.zeros((H, W), dtype=np.uint8)
        for _, comp_mask in valid_components:
            final_mask |= comp_mask

        final_area = int(final_mask.sum())
        if final_area < min_area:
            continue

        claimed |= final_mask
        kept.append((score, final_mask, final_area))

    return kept

def encode_mask_to_rle(mask: np.ndarray) -> str:
    """
    Encodes a 2D binary uint8 mask into a COCO RLE string.
    CRITICAL: Mask must be in Fortran-contiguous memory layout.
    """
    if not mask.flags["F_CONTIGUOUS"]:
        mask = np.asfortranarray(mask, dtype=np.uint8)
    else:
        mask = mask.astype(np.uint8)
    
    rle = mask_util.encode(mask)
    return rle["counts"].decode("utf-8")

def decode_rle_to_mask(rle_str: str, H: int = 2048, W: int = 2048) -> np.ndarray:
    """
    Decodes a COCO RLE string back to a (H, W) uint8 binary mask.
    """
    rle_dict = {"size": [H, W], "counts": rle_str.encode("utf-8") if isinstance(rle_str, str) else rle_str}
    return mask_util.decode(rle_dict).astype(np.uint8)

def masks_to_submission_rows(
    image_id: str,
    kept_instances: list[tuple[float, np.ndarray, int]],
) -> list[dict]:
    """
    Converts kept instances into formatted submission rows.
    """
    rows = []
    for idx, (score, mask, area) in enumerate(kept_instances, start=1):
        rle_str = encode_mask_to_rle(mask)
        rows.append({
            "filament_id": f"{image_id}_{idx}",
            "segmentation_rle": rle_str,
        })
    return rows

def check_no_overlap(csv_path_or_df, H: int = 2048, W: int = 2048) -> bool:
    """
    Strict validation to guarantee that no pixels are shared by two instances
    in any image. Raises ValueError if overlap is found.
    """
    if isinstance(csv_path_or_df, str):
        df = pd.read_csv(csv_path_or_df)
    else:
        df = csv_path_or_df.copy()

    if df.empty:
        print("[CHECK] DataFrame is empty.")
        return True

    df["stem"] = df["filament_id"].str.rsplit("_", n=1).str[0]
    total_checked = 0
    multi_instance_images = 0

    for stem, group in df.groupby("stem"):
        total_checked += 1
        if len(group) < 2:
            continue
        multi_instance_images += 1

        rle_dicts = [
            {"size": [H, W], "counts": r.encode("utf-8") if isinstance(r, str) else r}
            for r in group["segmentation_rle"]
        ]
        masks = mask_util.decode(rle_dicts)
        acc = masks.sum(axis=2)

        max_overlap = int(acc.max())
        if max_overlap > 1:
            overlap_count = int((acc > 1).sum())
            raise ValueError(
                f"[FAIL] Image stem '{stem}' has {overlap_count} overlapping pixels (max count {max_overlap})!"
            )

    print(f"[PASS] Zero pixel overlaps found across {total_checked} images ({multi_instance_images} multi-instance).")
    return True
