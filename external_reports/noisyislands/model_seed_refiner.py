"""Mask-conditioned SeedRefiner: ResNet18 U-Net with 3-channel input.

Input channels: [0]=image crop, [1]=CLAHE-enhanced image crop, [2]=seed/detector mask.
This is the refiner from the winning Felipe Castro / Nomannic19 approach.
The seed mask channel guides boundary refinement and instance separation.

Architecture follows lb-0-70 notebook CropUNet / new-metric notebook SeedRefiner pattern:
- ResNet18 encoder (ImageNet-pretrained optional)
- Symmetric U-Net decoder with skip connections
- ImageNet mean/std normalization built-in
- Single output channel (refined mask probability)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18


class ConvBlock(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class SeedRefiner(nn.Module):
    """ResNet18 U-Net refiner that takes image + CLAHE + seed mask as input."""

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        encoder = resnet18(weights="DEFAULT" if pretrained else None)
        self.stem = nn.Sequential(encoder.conv1, encoder.bn1, encoder.relu)
        self.pool, self.layer1, self.layer2, self.layer3, self.layer4 = (
            encoder.maxpool,
            encoder.layer1,
            encoder.layer2,
            encoder.layer3,
            encoder.layer4,
        )
        self.up4, self.dec4 = nn.ConvTranspose2d(512, 256, 2, 2), ConvBlock(512, 256)
        self.up3, self.dec3 = nn.ConvTranspose2d(256, 128, 2, 2), ConvBlock(256, 128)
        self.up2, self.dec2 = nn.ConvTranspose2d(128, 64, 2, 2), ConvBlock(128, 64)
        self.up1, self.dec1 = nn.ConvTranspose2d(64, 32, 2, 2), ConvBlock(96, 32)
        self.up0, self.head = nn.ConvTranspose2d(32, 16, 2, 2), nn.Conv2d(16, 1, 1)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        tensor = (tensor - self.mean) / self.std
        x0 = self.stem(tensor)
        x1 = self.layer1(self.pool(x0))
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        x = self.dec4(torch.cat([self.up4(x4), x3], dim=1))
        x = self.dec3(torch.cat([self.up3(x), x2], dim=1))
        x = self.dec2(torch.cat([self.up2(x), x1], dim=1))
        x = self.dec1(torch.cat([self.up1(x), x0], dim=1))
        return self.head(self.up0(x))


class SoftDiceLoss(nn.Module):
    """Computes 1 - soft Dice coefficient (lower is better)."""

    def __init__(self, smooth: float = 1.0) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = logits.sigmoid()
        intersection = (probs * targets).sum(dim=(1, 2, 3))
        denominator = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
        dice = (2.0 * intersection + self.smooth) / (denominator + self.smooth)
        return (1.0 - dice).mean()


class BCEPlusDiceLoss(nn.Module):
    """Combined BCE + soft Dice loss for refiner training."""

    def __init__(self, bce_weight: float = 1.0, dice_weight: float = 1.0) -> None:
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = SoftDiceLoss()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.bce_weight * self.bce(logits, targets) + self.dice_weight * self.dice(logits, targets)


@torch.no_grad()
def test_refiner():
    """Quick smoke test."""
    model = SeedRefiner(pretrained=False)
    x = torch.randn(2, 3, 256, 256)
    out = model(x)
    assert out.shape == (2, 1, 256, 256), f"Expected (2, 1, 256, 256), got {out.shape}"
    print(f"SeedRefiner output shape: {out.shape} ✓")

    loss_fn = BCEPlusDiceLoss()
    target = torch.randint(0, 2, (2, 1, 256, 256)).float()
    loss = loss_fn(out, target)
    print(f"Loss value: {loss.item():.4f} ✓")


if __name__ == "__main__":
    test_refiner()
