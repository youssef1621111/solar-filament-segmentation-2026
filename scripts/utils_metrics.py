"""
Evaluation Metrics & COCO RLE Utilities
Panoptic Quality (PQ), Segmentation Quality (SQ), Recognition Quality (RQ)
Strict COCO LEB128 compressed RLE encoder/decoder matching pycocotools maskApi.c
"""

import numpy as np


def rle_decode_uncompressed(counts, shape=(2048, 2048)):
    """
    Decode uncompressed count list to binary mask (Fortran order).
    Validates that total decoded pixels strictly match shape[0] * shape[1].
    """
    total_pixels = shape[0] * shape[1]
    actual_pixels = sum(counts)
    if actual_pixels != total_pixels:
        raise ValueError(
            f"Decoded RLE pixel sum ({actual_pixels}) does not match expected total pixels ({total_pixels}) for shape {shape}"
        )
    mask_flat = np.zeros(total_pixels, dtype=bool)
    curr = 0
    val = False
    for c in counts:
        if c < 0:
            raise ValueError(f"Encountered negative run length count ({c}) in RLE")
        if val:
            mask_flat[curr: curr + c] = True
        curr += c
        val = not val
    return mask_flat.reshape(shape, order='F')


def mask_to_uncompressed_rle(mask: np.ndarray):
    """
    Convert binary mask (2048x2048) to uncompressed count list (Fortran order).
    Properly handles bool, uint8 (0/1 or 0/255), and float representations.
    """
    flat_F = (mask > 0).flatten(order='F')
    n = len(flat_F)
    if n == 0:
        return []
    # Find change points
    changes = np.where(flat_F[1:] != flat_F[:-1])[0] + 1
    split_indices = np.concatenate([[0], changes, [n]])
    counts = np.diff(split_indices)

    # If first pixel is foreground, COCO requires leading 0 for background
    if flat_F[0]:
        counts = np.concatenate([[0], counts])
    return [int(c) for c in counts]


def encode_leb128_rle(mask: np.ndarray) -> str:
    """
    Encode binary mask to COCO LEB128 compressed RLE string.
    Uses exact COCO maskApi.c rleToString specification.
    Guaranteed bit-for-bit equivalence with pycocotools.mask.encode.
    """
    counts = mask_to_uncompressed_rle(mask)
    chars = []
    for j, count in enumerate(counts):
        x = count
        if j > 2:
            x -= counts[j - 2]
        more = True
        while more:
            c = x & 0x1F
            x >>= 5
            more = (x != -1) if (c & 0x10) else (x != 0)
            if more:
                c |= 0x20
            chars.append(chr(c + 48))
    return "".join(chars)


def decode_leb128_counts(rle_str: str) -> list:
    """
    Decode COCO LEB128 compressed RLE string to uncompressed count list.
    Uses exact COCO maskApi.c rleFrString specification.
    Fast count extraction without full boolean mask allocation (~1.5s for 1,344 instances).
    """
    counts = []
    j = 0
    k = 0
    s_len = len(rle_str)
    while k < s_len:
        x = 0
        more = 1
        m = 0
        while more:
            if k >= s_len:
                raise ValueError("Truncated LEB128 string")
            c = ord(rle_str[k]) - 48
            k += 1
            x |= (c & 0x1F) << (m * 5)
            more = (c & 0x20)
            m += 1
            if not more and (c & 0x10):
                x |= (~0 << (m * 5))
        if j > 2:
            x += counts[j - 2]
        counts.append(x)
        j += 1
    return counts


def decode_leb128_rle(rle_str: str, shape=(2048, 2048)) -> np.ndarray:
    """
    Decode COCO LEB128 compressed RLE string to binary mask.
    Uses exact COCO maskApi.c rleFrString specification.
    """
    counts = decode_leb128_counts(rle_str)
    return rle_decode_uncompressed(counts, shape)


def compute_mask_bbox(mask: np.ndarray):
    """Compute bounding box [y_min, y_max, x_min, x_max] of binary mask."""
    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        return 0, 0, 0, 0
    return int(np.min(ys)), int(np.max(ys)), int(np.min(xs)), int(np.max(xs))


def compute_iou(mask_a: np.ndarray, mask_b: np.ndarray, bbox_a=None, bbox_b=None) -> float:
    """Compute Intersection over Union between two binary masks with bounding-box pre-filtering."""
    if bbox_a is not None and bbox_b is not None:
        y1_a, y2_a, x1_a, x2_a = bbox_a
        y1_b, y2_b, x1_b, x2_b = bbox_b
        if x2_a < x1_b or x2_b < x1_a or y2_a < y1_b or y2_b < y1_a:
            return 0.0

    intersection = np.count_nonzero(mask_a & mask_b)
    if intersection == 0:
        return 0.0
    union = np.count_nonzero(mask_a | mask_b)
    return float(intersection) / float(union) if union > 0 else 0.0


def evaluate_image_panoptic_quality(pred_masks: list, gt_masks: list, iou_thresh: float = 0.50):
    """
    Evaluate Panoptic Quality for a single image against ground truth.
    Strict IoU > 0.50 matching: each prediction can match at most 1 GT, and vice versa.
    Returns: tp_count, fp_count, fn_count, sum_iou, matches (list of (pred_idx, gt_idx, iou))
    """
    num_preds = len(pred_masks)
    num_gts = len(gt_masks)

    if num_preds == 0 and num_gts == 0:
        return 0, 0, 0, 0.0, []
    if num_preds == 0:
        return 0, 0, num_gts, 0.0, []
    if num_gts == 0:
        return 0, num_preds, 0, 0.0, []

    pred_bboxes = [compute_mask_bbox(p) for p in pred_masks]
    gt_bboxes = [compute_mask_bbox(g) for g in gt_masks]

    matched_gt = set()
    matched_pred = set()
    matches = []
    sum_iou = 0.0

    candidates = []
    for p_idx, (p_mask, p_box) in enumerate(zip(pred_masks, pred_bboxes)):
        for g_idx, (g_mask, g_box) in enumerate(zip(gt_masks, gt_bboxes)):
            iou = compute_iou(p_mask, g_mask, p_box, g_box)
            if iou > iou_thresh:
                candidates.append((iou, p_idx, g_idx))

    candidates.sort(key=lambda x: x[0], reverse=True)

    for iou, p_idx, g_idx in candidates:
        if p_idx not in matched_pred and g_idx not in matched_gt:
            matched_pred.add(p_idx)
            matched_gt.add(g_idx)
            matches.append((p_idx, g_idx, iou))
            sum_iou += iou

    tp = len(matches)
    fp = num_preds - tp
    fn = num_gts - tp

    return tp, fp, fn, sum_iou, matches


def compute_dataset_pq(all_image_stats: list):
    """
    Aggregate image-level statistics and compute dataset-level SQ, RQ, PQ.
    all_image_stats: list of tuples (tp, fp, fn, sum_iou)
    """
    total_tp = sum(s[0] for s in all_image_stats)
    total_fp = sum(s[1] for s in all_image_stats)
    total_fn = sum(s[2] for s in all_image_stats)
    total_sum_iou = sum(s[3] for s in all_image_stats)

    sq = (total_sum_iou / total_tp) if total_tp > 0 else 0.0
    denominator = total_tp + 0.5 * total_fp + 0.5 * total_fn
    rq = (total_tp / denominator) if denominator > 0 else 0.0
    pq = sq * rq

    return {
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "sq": sq,
        "rq": rq,
        "pq": pq
    }
