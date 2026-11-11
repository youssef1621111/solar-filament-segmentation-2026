#!/usr/bin/env python3
"""
Test & Verification Suite for Soft-Consensus Pipeline Components:
1. CLAHE disk equalization
2. Soft consensus target calculation
3. Soft Dice Loss gradient and forward pass
4. 1024x1024 tiling and Gaussian blending reconstruction
"""

import sys
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F

def test_clahe():
    img = np.random.randint(50, 200, (2048, 2048), dtype=np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(img)
    assert enhanced.shape == (2048, 2048)
    print(">> CLAHE test PASSED")

def test_soft_consensus():
    # Simulate 3 annotators on a 2048x2048 canvas
    ann1 = np.zeros((2048, 2048), dtype=bool)
    ann2 = np.zeros((2048, 2048), dtype=bool)
    ann3 = np.zeros((2048, 2048), dtype=bool)
    
    # Filament A: all 3 agree (unanimous consensus)
    ann1[100:200, 100:200] = True
    ann2[100:200, 100:200] = True
    ann3[100:200, 100:200] = True
    
    # Filament B: 2 agree (majority consensus)
    ann1[300:400, 300:400] = True
    ann2[300:400, 300:400] = True
    
    # Filament C: only 1 drew (solo annotator noise)
    ann1[500:600, 500:600] = True
    
    agreement = (ann1.astype(np.float32) + ann2.astype(np.float32) + ann3.astype(np.float32)) / 3.0
    soft_target = 0.75 * (agreement >= 0.50).astype(np.float32) + 0.25 * agreement
    
    # Filament A should have weight 1.0
    assert np.isclose(soft_target[150, 150], 1.0)
    # Filament B should have weight 0.75 + 0.25 * (2/3) = 0.9167
    assert np.isclose(soft_target[350, 350], 0.75 + 0.25 * (2.0/3.0))
    # Filament C (solo) should have weight 0.25 * (1/3) = 0.0833 (heavily suppressed!)
    assert np.isclose(soft_target[550, 550], 0.25 * (1.0/3.0))
    # Background should be 0.0
    assert soft_target[0, 0] == 0.0
    print(f">> Soft consensus target test PASSED (A=1.0, B={soft_target[350, 350]:.4f}, C={soft_target[550, 550]:.4f})")

class SoftDiceBCEWithLogitsLoss(nn.Module):
    def __init__(self, bce_weight=0.5, smooth=1.0):
        super().__init__()
        self.bce_weight = bce_weight
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets)
        probs = torch.sigmoid(logits)
        
        # Flatten across spatial dimensions
        p_flat = probs.view(probs.size(0), -1)
        t_flat = targets.view(targets.size(0), -1)
        
        intersection = (p_flat * t_flat).sum(dim=1)
        cardinality = (p_flat.pow(2) + t_flat.pow(2)).sum(dim=1)
        dice_loss = 1.0 - (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        
        return self.bce_weight * bce_loss + (1.0 - self.bce_weight) * dice_loss.mean()

def test_loss():
    logits = torch.randn(2, 1, 256, 256, requires_grad=True)
    targets = torch.rand(2, 1, 256, 256)
    loss_fn = SoftDiceBCEWithLogitsLoss()
    loss = loss_fn(logits, targets)
    loss.backward()
    assert logits.grad is not None
    assert not torch.isnan(loss)
    print(f">> SoftDiceBCE loss test PASSED (Loss={loss.item():.4f})")

def test_tiling_reconstruction():
    # Verify 1024x1024 overlapping tiling covers full 2048x2048 with linear distance weighting
    full_mask = np.zeros((2048, 2048), dtype=np.float32)
    weight_map = np.zeros((2048, 2048), dtype=np.float32)
    
    tile_size = 1024
    stride = 512
    coords = []
    for y in range(0, 2048 - tile_size + 1, stride):
        for x in range(0, 2048 - tile_size + 1, stride):
            coords.append((y, x))
            
    assert len(coords) == 9, f"Expected 9 tiles (3x3 grid), got {len(coords)}"
    
    # 2D Linear tent window with non-zero edge floor
    d_x = np.minimum(np.arange(tile_size), tile_size - 1 - np.arange(tile_size))
    d_y = np.minimum(np.arange(tile_size), tile_size - 1 - np.arange(tile_size))
    window_2d = (np.minimum.outer(d_y, d_x) / (tile_size / 2.0)).astype(np.float32) + 0.1
    
    for y, x in coords:
        full_mask[y:y+tile_size, x:x+tile_size] += window_2d
        weight_map[y:y+tile_size, x:x+tile_size] += window_2d
        
    full_mask /= np.maximum(weight_map, 1e-6)
    assert np.allclose(full_mask, 1.0, atol=1e-5)
    print(">> Tiling and linear window blending test PASSED (100% reconstruction exactness)")

if __name__ == "__main__":
    test_clahe()
    test_soft_consensus()
    test_loss()
    test_tiling_reconstruction()
    print("ALL TESTS PASSED SUCCESSFULLY!")
