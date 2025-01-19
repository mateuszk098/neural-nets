import torch
import torch.nn as nn
from torch.types import Tensor
from torchvision import models


class YOLOv2(nn.Module):
    def __init__(self, B: int, C: int) -> None:
        super(YOLOv2, self).__init__()
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.backbone = nn.ModuleDict(dict(self.backbone.named_children()))

        self.backbone.pop("avgpool")
        self.backbone.pop("fc")

        self.neck = nn.ModuleDict(
            {
                "layer1": nn.Sequential(
                    nn.Conv2d(512, 1024, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(1024),
                    nn.LeakyReLU(0.1),
                ),
                "layer2": nn.Sequential(
                    nn.Conv2d(1024, 1024, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(1024),
                    nn.LeakyReLU(0.1),
                ),
                "layer3": nn.Sequential(
                    nn.Conv2d(2048, 1024, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(1024),
                    nn.LeakyReLU(0.1),
                ),
                "layer4": nn.Conv2d(1024, B * (5 + C), kernel_size=1),
            }
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)

        x_fine_grained = self.backbone.layer3(x)
        x_full = self.backbone.layer4(x_fine_grained)

        x_full = self.neck.layer1(x_full)
        x_full = self.neck.layer2(x_full)

        batch_size = x_fine_grained.size(0)
        spatial_size = (x_fine_grained.size(2) // 2, x_fine_grained.size(3) // 2)
        x_fine_grained = x_fine_grained.view(batch_size, -1, *spatial_size)

        x_cat = torch.cat((x_fine_grained, x_full), dim=1)

        x_cat = self.neck.layer3(x_cat)
        x_cat = self.neck.layer4(x_cat)

        return x_cat
