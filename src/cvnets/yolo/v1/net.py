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

        self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*self.backbone.children())[:-2]  # Without avgpool and fc.

        self.neck = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1, bias=False),
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
            nn.AdaptiveAvgPool2d((7, 7)),
        )

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(7 * 7 * 1024, 4096),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5),
            nn.Linear(4096, S * S * (B * 5 + C)),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.backbone(x)
        x = self.neck(x)
        x = self.head(x)
        return x

    def is_backbone_trainable(self) -> bool:
        return all(p.requires_grad for p in self.backbone.parameters())
