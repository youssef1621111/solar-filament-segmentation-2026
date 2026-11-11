import cv2
import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

def enhance_solar_contrast(img: np.ndarray, clip_limit: float = 2.0, tile_grid_size: tuple = (8, 8)) -> np.ndarray:
    """
    Applies CLAHE (Contrast Limited Adaptive Histogram Equalization) to reveal fine
    chromospheric fibrils and solar filament structures.
    """
    if len(img.shape) == 3:
        gray = img[:, :, 0]
    else:
        gray = img

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    enhanced = clahe.apply(gray)
    return enhanced

def get_train_transforms(image_size: int = 1024) -> A.Compose:
    """
    Data augmentations respecting solar rotational and reflectional physics (D4 group).
    """
    return A.Compose(
        [
            A.Resize(image_size, image_size, interpolation=cv2.INTER_LINEAR),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.04,
                scale_limit=0.08,
                rotate_limit=15,
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                mask_value=0,
                p=0.5,
            ),
            A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.5),
            A.Normalize(mean=[0.35], std=[0.25]),
            ToTensorV2(),
        ],
        additional_targets={"spine": "mask", "weight": "mask"},
    )

def get_valid_transforms(image_size: int = 1024) -> A.Compose:
    """
    Deterministic preprocessing for validation and test inference.
    """
    return A.Compose(
        [
            A.Resize(image_size, image_size, interpolation=cv2.INTER_LINEAR),
            A.Normalize(mean=[0.35], std=[0.25]),
            ToTensorV2(),
        ],
        additional_targets={"spine": "mask", "weight": "mask"},
    )

def split_into_tiles(
    img: np.ndarray,
    tile_size: int = 1024,
    overlap: int = 256,
) -> tuple[list[np.ndarray], list[tuple[int, int, int, int]]]:
    """
    Splits a (H, W) or (H, W, C) image into overlapping tiles.
    Returns (tiles, coords) where coords is (y1, y2, x1, x2).
    """
    H, W = img.shape[:2]
    stride = tile_size - overlap
    tiles = []
    coords = []

    y_starts = list(range(0, H - tile_size + 1, stride))
    if y_starts[-1] + tile_size < H:
        y_starts.append(H - tile_size)

    x_starts = list(range(0, W - tile_size + 1, stride))
    if x_starts[-1] + tile_size < W:
        x_starts.append(W - tile_size)

    for y in y_starts:
        for x in x_starts:
            y1, y2 = y, y + tile_size
            x1, x2 = x, x + tile_size
            tile = img[y1:y2, x1:x2]
            tiles.append(tile)
            coords.append((y1, y2, x1, x2))

    return tiles, coords

def merge_tiles(
    tiles: list[np.ndarray],
    coords: list[tuple[int, int, int, int]],
    out_shape: tuple[int, int] = (2048, 2048),
    tile_size: int = 1024,
) -> np.ndarray:
    """
    Seamlessly merges predicted tile probability maps using 2D Cosine window weighting.
    """
    H, W = out_shape
    output = np.zeros((H, W), dtype=np.float32)
    weight_map = np.zeros((H, W), dtype=np.float32)

    # Generate 2D Cosine window
    win1d = np.sin(np.linspace(0, np.pi, tile_size))
    window2d = np.outer(win1d, win1d).astype(np.float32)
    window2d = np.clip(window2d, 1e-4, 1.0)

    for tile, (y1, y2, x1, x2) in zip(tiles, coords):
        output[y1:y2, x1:x2] += tile * window2d
        weight_map[y1:y2, x1:x2] += window2d

    weight_map = np.maximum(weight_map, 1e-6)
    return output / weight_map
