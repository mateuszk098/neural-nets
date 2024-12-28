from typing import Self

import torch
import torch.nn as nn
from torch.types import Tensor
from torchvision import models


class YOLOv1(nn.Module):
    def __init__(self, imgsz: int, S: int, B: int, C: int, use_sigmoid: bool = False) -> None:
        super().__init__()
        self.imgsz = int(imgsz)
        self.S = int(S)
        self.B = int(B)
        self.C = int(C)
        self.use_sigmoid = bool(use_sigmoid)

        self.backbone = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])  # Without avgpool and fc.

        self.neck = nn.Sequential(
            nn.LazyConv2d(1024, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.Mish(),
            nn.Conv2d(1024, 1024, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.Mish(),
            nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.Mish(),
            nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.Mish(),
        )

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(4096),
            nn.Mish(),
            nn.Dropout(0.5),
            nn.Linear(4096, self.S * self.S * (self.B * 5 + self.C)),
        )

        self.warmup()  # Required for LazyLinear and LazyConv2d to intialize dimensions.

    def forward(self, x: Tensor) -> Tensor:
        x = self.backbone(x)
        x = self.neck(x)
        x = self.head(x)

        if self.use_sigmoid:
            # Sigmoid activation for x, y, sqrt(w), sqrt(h), and objectness.
            # Faster convergence (?) since that output should be between 0 and 1.
            x = x.view(-1, self.S, self.S, self.B * 5 + self.C)
            torch.sigmoid_(x[..., : self.B * 5])
            x = x.view(-1, self.S * self.S * (self.B * 5 + self.C))

        return x

    def warmup(self) -> Self:
        x = torch.randn(1, 3, self.imgsz, self.imgsz, generator=torch.manual_seed(42))
        self.forward(x)
        return self
