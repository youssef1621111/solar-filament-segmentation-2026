import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp
import numpy as np
import cv2
from panoptic_painter import paint_panoptic

class FilamentSpineUNet(nn.Module):
    """
    Dual-Head UNet predicting:
      - Channel 0: Filament binary segmentation logits
      - Channel 1: Spine / skeleton heatmap logits
      - Channel 2: Internal distance transform regression
    """
    def __init__(
        self,
        backbone: str = "resnet34",
        pretrained: bool = True,
        in_channels: int = 1,
        classes: int = 3,
    ):
        super().__init__()
        weights = "imagenet" if pretrained else None
        self.model = smp.Unet(
            encoder_name=backbone,
            encoder_weights=weights,
            in_channels=in_channels,
            classes=classes,
        )

    def forward(self, x):
        return self.model(x)

class FilamentCompositeLoss(nn.Module):
    """
    Multi-task loss balancing instance boundary accuracy and spine topological localization:
      L = L_bce(mask) + L_dice(mask) + 2.0 * L_focal(spine) + 1.0 * L_smooth_l1(dist)
    """
    def __init__(self, mask_pos_weight: float = 3.0, spine_pos_weight: float = 6.0):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([mask_pos_weight]))
        self.spine_bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([spine_pos_weight]))
        self.smooth_l1 = nn.SmoothL1Loss()

    def dice_loss(self, logits, targets, smooth=1.0):
        probs = torch.sigmoid(logits)
        intersection = (probs * targets).sum(dim=(-2, -1))
        union = probs.sum(dim=(-2, -1)) + targets.sum(dim=(-2, -1))
        dice = (2.0 * intersection + smooth) / (union + smooth)
        return 1.0 - dice.mean()

    def forward(self, preds, targets):
        # preds: [B, 3, H, W]
        # targets: [B, 3, H, W] (mask, spine, dist)
        mask_pred = preds[:, 0:1]
        spine_pred = preds[:, 1:2]
        dist_pred = preds[:, 2:3]

        mask_target = targets[:, 0:1]
        spine_target = targets[:, 1:2]
        dist_target = targets[:, 2:3]

        # 1. Mask loss
        if self.bce.pos_weight.device != mask_pred.device:
            self.bce.pos_weight = self.bce.pos_weight.to(mask_pred.device)
        l_mask_bce = self.bce(mask_pred, mask_target)
        l_mask_dice = self.dice_loss(mask_pred, mask_target)
        loss_mask = l_mask_bce + l_mask_dice

        # 2. Spine loss
        if self.spine_bce.pos_weight.device != spine_pred.device:
            self.spine_bce.pos_weight = self.spine_bce.pos_weight.to(spine_pred.device)
        loss_spine = self.spine_bce(spine_pred, spine_target) + self.dice_loss(spine_pred, spine_target)

        # 3. Distance regression loss (sigmoid on dist_pred)
        dist_probs = torch.sigmoid(dist_pred)
        loss_dist = self.smooth_l1(dist_probs, dist_target)

        total_loss = loss_mask + 2.0 * loss_spine + 1.0 * loss_dist
        return total_loss, {
            "loss_mask": loss_mask.item(),
            "loss_spine": loss_spine.item(),
            "loss_dist": loss_dist.item(),
            "total_loss": total_loss.item(),
        }

def spine_watershed_postprocess(
    mask_prob: np.ndarray,
    spine_prob: np.ndarray,
    dist_prob: np.ndarray = None,
    mask_threshold: float = 0.40,
    spine_threshold: float = 0.35,
    min_area: int = 150,
) -> list[tuple[float, np.ndarray, int]]:
    """
    Extracts individual filament instances using Spine-Seeded Watershed disambiguation,
    followed by score-ordered panoptic painting.
    """
    H, W = mask_prob.shape[:2]
    binary_mask = (mask_prob > mask_threshold).astype(np.uint8)

    if binary_mask.sum() < min_area:
        return []

    # 1. Extract spine markers inside the binary mask
    spine_markers = (spine_prob > spine_threshold) & (binary_mask > 0)

    # 2. Label seed components
    num_seeds, seed_labels = cv2.connectedComponents(spine_markers.astype(np.uint8), connectivity=8)

    candidates = []

    if num_seeds <= 1:
        # Fallback to connected components of binary mask directly if no spine seeds detected
        num_comps, comp_labels = cv2.connectedComponents(binary_mask, connectivity=8)
        for c in range(1, num_comps):
            c_mask = (comp_labels == c).astype(np.uint8)
            area = int(c_mask.sum())
            if area >= min_area:
                score = float(mask_prob[c_mask > 0].mean())
                candidates.append((score, c_mask))
    else:
        # 3. Spine-Seeded Watershed segmentation to separate close/parallel filaments
        # Invert distance or use 1 - mask_prob as elevation surface
        elevation = (255 * (1.0 - mask_prob)).astype(np.uint8)
        elevation = cv2.cvtColor(elevation, cv2.COLOR_GRAY2BGR)

        # Watershed requires markers with background = 0, boundaries = -1
        markers = seed_labels.astype(np.int32)
        # Background pixels outside binary mask set to 1, seed markers offset by 1
        markers[binary_mask == 0] = 0
        markers[seed_labels > 0] += 1
        markers[binary_mask == 0] = 1

        cv2.watershed(elevation, markers)

        # Extract segmented instances (markers > 1)
        unique_markers = np.unique(markers)
        for m in unique_markers:
            if m <= 1:  # 0 is unknown, 1 is background, -1 is boundary
                continue
            inst_mask = ((markers == m) & (binary_mask > 0)).astype(np.uint8)
            area = int(inst_mask.sum())
            if area >= min_area:
                # Score combines mask confidence and spine presence
                mean_conf = float(mask_prob[inst_mask > 0].mean())
                max_spine = float(spine_prob[inst_mask > 0].max()) if (inst_mask > 0).any() else 0.0
                score = 0.7 * mean_conf + 0.3 * max_spine
                candidates.append((score, inst_mask))

        # Check for any truly separated unseeded mask regions (e.g. isolated filaments without a strong spine)
        assigned = np.zeros((H, W), dtype=np.uint8)
        for _, c_mask in candidates:
            assigned |= c_mask
        # Only consider unassigned pixels that are not adjacent (erode 2px) to existing assigned instances
        unassigned = binary_mask & (~assigned)
        unassigned = cv2.erode(unassigned, np.ones((3, 3), np.uint8))
        if unassigned.sum() >= min_area:
            num_unassigned, unassigned_labels = cv2.connectedComponents(unassigned, connectivity=8)
            for c in range(1, num_unassigned):
                c_mask = (unassigned_labels == c).astype(np.uint8)
                c_mask = cv2.dilate(c_mask, np.ones((3, 3), np.uint8)) & binary_mask
                area = int(c_mask.sum())
                if area >= min_area:
                    score = float(mask_prob[c_mask > 0].mean())
                    candidates.append((score, c_mask))

    # 4. Enforce strict non-overlapping constraint via Score-Ordered Panoptic Painting
    kept_instances = paint_panoptic(candidates, min_area=min_area, H=H, W=W)
    return kept_instances
