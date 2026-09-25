from __future__ import annotations

from typing import Any


def build_model(architecture: str, num_classes: int) -> Any:
    import torch.nn as nn

    if architecture == "resnet18_cifar":
        from torchvision.models import resnet18

        model = resnet18(weights=None, num_classes=num_classes)
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
        return model
    if architecture == "tiny_cnn":
        return nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(16, num_classes),
        )
    raise ValueError(f"Unknown architecture: {architecture}")

