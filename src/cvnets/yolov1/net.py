from typing import Self

import torch
import torch.nn as nn
from torch.types import Tensor
from torchvision import models


class YOLOv1(nn.Module):
    def __init__(self, imgsz: int, S: int, B: int, C: int) -> None:
        super().__init__()
        self.imgsz = int(imgsz)
        self.S = int(S)
        self.B = int(B)
        self.C = int(C)

        self.backbone = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])  # Without avgpool and fc.

        self.neck = nn.Sequential(
            nn.LazyConv2d(1024, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1),
            nn.Conv2d(1024, 1024, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1),
            nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1),
            nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1),
        )

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(4096),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5),
            nn.Linear(4096, S * S * (B * 5 + C)),
        )

        self.warmup()

    def forward(self, x: Tensor) -> Tensor:
        x = self.backbone(x)
        x = self.neck(x)
        x = self.head(x)
        return x

    def warmup(self) -> Self:
        x = torch.randn(1, 3, self.imgsz, self.imgsz, generator=torch.manual_seed(42))
        self.forward(x)
        return self
