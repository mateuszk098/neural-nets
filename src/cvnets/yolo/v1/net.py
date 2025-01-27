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
            nn.LeakyReLU(0.1),
        )

        # Convolution as head since I don't have enough computional power
        # to use fully connected layers as in the original paper.
        self.head = nn.Conv2d(1024, self.B * 5 + self.C, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        x = self.backbone(x)
        x = self.neck(x)
        x = self.head(x)

        x = x.permute(0, 2, 3, 1)
        # Sigmoid for x, y, w, h, conf and Softmax for classes.
        x[..., : self.B * 5] = torch.sigmoid(x[..., : self.B * 5])
        x[..., self.B * 5 :] = torch.softmax(x[..., self.B * 5 :], dim=-1)

        return x

    def is_backbone_trainable(self) -> bool:
        return all(p.requires_grad for p in self.backbone.parameters())
