"""
ROGGI Physics-Guided Utilities for Solar Filament Analysis
IEEE BigData 2026 Cup — Solar Filament Segmentation Challenge
"""

import re
from datetime import datetime
import numpy as np
import scipy.ndimage as ndi


def parse_timestamp_from_stem(stem: str):
    """
    Parse observation UTC timestamp YYYYMMDDHHMMSS from filename stem.
    Example: 20160513133714Th -> datetime(2016, 5, 13, 13, 37, 14)
    """
    match = re.search(r"(\d{14})", stem)
    if match:
        dt_str = match.group(1)
        try:
            return datetime.strptime(dt_str, "%Y%m%d%H%M%S")
        except ValueError:
            pass
    return None


def estimate_solar_disk_parameters(image: np.ndarray, thresh_percentile: float = 20.0):
    """
    Estimate the solar disk center (x0, y0) and radius R_sun from a 2048x2048 H-alpha image.
    Uses intensity thresholding and center of mass/equivalent radius.
    """
    H, W = image.shape
    # If standard 2048x2048 GONG disk, default nominal values:
    default_x0, default_y0 = W / 2.0, H / 2.0
    default_r = min(H, W) * 0.44  # ~900 px

    # Solar disk is brighter than space background in H-alpha GONG
    t_val = np.percentile(image, thresh_percentile)
    disk_mask = image > t_val

    # Remove outer border noise
    disk_mask[:30, :] = 0
    disk_mask[-30:, :] = 0
    disk_mask[:, :30] = 0
    disk_mask[:, -30:] = 0

    area = np.sum(disk_mask)
    if area < 100000:
        return default_x0, default_y0, default_r

    cy, cx = ndi.center_of_mass(disk_mask)
    r_est = np.sqrt(area / np.pi)
    return float(cx), float(cy), float(r_est)


def compute_limb_darkening_profile(image: np.ndarray, x0: float, y0: float, r_sun: float, ring_width: float = 10.0):
    """
    Fit radial solar limb-darkening profile Median_ring(r) and MAD_ring(r)
    in concentric radial rings from disk center (x0, y0, R_sun).
    Returns normalized absorption deficit map:
    Delta I_mu(x, y) = (Median_ring(r) - I(x, y)) / (MAD_ring(r) + 1e-5)
    """
    H, W = image.shape
    y_idx, x_idx = np.ogrid[:H, :W]
    r_map = np.sqrt((x_idx - x0) ** 2 + (y_idx - y0) ** 2)

    max_ring = int(np.ceil(r_sun / ring_width)) + 1
    ring_idx = np.floor(r_map / ring_width).astype(np.int32)

    delta_i_mu = np.zeros_like(image, dtype=np.float32)

    # Compute ring statistics within the solar disk
    for ring in range(max_ring):
        mask = (ring_idx == ring) & (r_map <= r_sun * 1.02)
        if np.count_nonzero(mask) < 20:
            continue
        ring_vals = image[mask]
        med = float(np.median(ring_vals))
        mad = float(np.median(np.abs(ring_vals - med)))
        delta_i_mu[mask] = (med - ring_vals) / (mad + 1e-5)

    return delta_i_mu, r_map


def compute_halpha_limb_deficit(delta_i_mu: np.ndarray, mask: np.ndarray, dilate_px: int = 15) -> float:
    """
    Record difference between the interior of candidate mask Y_k and a 15px surrounding background ring:
    deficit = mean_{interior}(Delta I_mu) - mean_{background_ring}(Delta I_mu)
    """
    interior_pixels = delta_i_mu[mask > 0]
    if len(interior_pixels) == 0:
        return 0.0
    mean_interior = float(np.mean(interior_pixels))

    # Background ring: dilate(mask, dilate_px) \ mask
    struct = ndi.generate_binary_structure(2, 1)
    dilated = ndi.binary_dilation(mask > 0, structure=struct, iterations=dilate_px)
    bg_ring = dilated & (~(mask > 0))

    bg_pixels = delta_i_mu[bg_ring]
    if len(bg_pixels) == 0:
        return mean_interior
    mean_bg = float(np.mean(bg_pixels))

    return mean_interior - mean_bg


def compute_mask_shape_features(mask: np.ndarray):
    """
    Compute elongation and circularity for Sunspot Umbra Veto:
    - PCA covariance eigenvalues (lambda_1, lambda_2) of mask coordinates
    - elongation = sqrt(lambda_1 / (lambda_2 + 1e-5))
    - circularity = 4 * pi * Area / (Perimeter^2)
    - is_sunspot_veto = (area < 500) and (elongation < 1.38) and (circularity > 0.68)
    """
    area = int(np.sum(mask > 0))
    if area < 10:
        return 1.0, 0.0, area, False

    coords = np.column_stack(np.where(mask > 0))  # shape: (N, 2) -> (y, x)
    # Center coordinates
    centered = coords - np.mean(coords, axis=0, keepdims=True)
    # Covariance matrix (2x2)
    cov = np.cov(centered, rowvar=False)

    if cov.ndim < 2 or np.isnan(cov).any():
        eigenvals = np.array([1.0, 1.0])
    else:
        eigenvals = np.linalg.eigvalsh(cov)
        eigenvals = np.maximum(eigenvals, 1e-6)

    lambda_2, lambda_1 = sorted(eigenvals)  # ascending: lambda_2 <= lambda_1
    elongation = float(np.sqrt(lambda_1 / (lambda_2 + 1e-5)))

    # Estimate perimeter via binary morphological boundary
    struct = ndi.generate_binary_structure(2, 1)
    eroded = ndi.binary_erosion(mask > 0, structure=struct)
    perimeter = int(np.sum((mask > 0) & (~eroded)))
    if perimeter == 0:
        perimeter = 1

    circularity = float((4.0 * np.pi * area) / (perimeter ** 2))

    is_sunspot_veto = bool((area < 500) and (elongation < 1.38) and (circularity > 0.68))
    return elongation, circularity, area, is_sunspot_veto


def solar_differential_rotation_rate(phi_rad: np.ndarray) -> np.ndarray:
    """
    Physical solar differential rotation law (Snodgrass & Ulrich 1990):
    omega(phi) = 14.713 - 2.396 * sin^2(phi) - 1.787 * sin^4(phi) [deg/day]
    phi_rad: heliographic latitude in radians
    """
    sin_phi = np.sin(phi_rad)
    sin2 = sin_phi ** 2
    sin4 = sin2 ** 2
    return 14.713 - 2.396 * sin2 - 1.787 * sin4


def project_disk_by_differential_rotation(
    source_mask_or_prob: np.ndarray,
    delta_t_hours: float,
    x0: float,
    y0: float,
    r_sun: float
) -> np.ndarray:
    """
    Project source disk foreground probability/mask onto target disk grid
    separated by delta_t_hours using solar differential rotation.
    delta_t = t_target - t_source (hours).
    """
    H, W = source_mask_or_prob.shape
    y_grid, x_grid = np.mgrid[:H, :W]

    # Target normalized coordinates
    u = (x_grid - x0) / r_sun
    v = (y_grid - y0) / r_sun
    rho2 = u ** 2 + v ** 2
    valid_disk = rho2 < 0.98

    # Heliographic latitude phi and longitude lambda on target disk
    v_clipped = np.clip(v, -0.999, 0.999)
    phi = np.arcsin(v_clipped)

    cos_phi = np.cos(phi)
    u_rel = np.clip(u / (cos_phi + 1e-7), -0.999, 0.999)
    lam = np.arcsin(u_rel)

    # Differential rotation longitudinal shift between source and target
    # omega is deg/day -> convert to radians: (omega * delta_t / 24) * (pi / 180)
    omega_deg_per_day = solar_differential_rotation_rate(phi)
    delta_lam_rad = np.radians(omega_deg_per_day * (delta_t_hours / 24.0))

    # Longitude on source disk: lam_source = lam_target - delta_lam
    lam_source = lam - delta_lam_rad

    # Source disk projected coordinates
    u_source = np.sin(lam_source) * cos_phi
    v_source = np.sin(phi)

    x_source = x0 + u_source * r_sun
    y_source = y0 + v_source * r_sun

    # Invalid points where source rotated around limb or outside disk
    source_valid = valid_disk & (np.abs(lam_source) < (np.pi / 2.0))

    coords = np.array([y_source, x_source])
    warped = ndi.map_coordinates(
        source_mask_or_prob.astype(np.float32),
        coords,
        order=1,
        mode="constant",
        cval=0.0
    )
    warped[~source_valid] = 0.0
    return warped
