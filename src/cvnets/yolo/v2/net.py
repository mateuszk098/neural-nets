import torch
import torch.nn as nn
from torch.types import Tensor
from torchvision import models


class YOLOv2(nn.Module):
    def __init__(self, num_anchors: int, num_classes: int) -> None:
        super().__init__()
        self.num_anchors = int(num_anchors)
        self.num_classes = int(num_classes)

        self.backbone = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        self.backbone = nn.ModuleDict(dict(self.backbone.named_children()))

        self.backbone.pop("avgpool")
        self.backbone.pop("fc")

        self.neck = nn.ModuleDict(
            {
                "layer1": nn.Sequential(
                    nn.LazyConv2d(1024, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(1024),
                    nn.LeakyReLU(0.1, inplace=True),
                ),
                "layer2": nn.Sequential(
                    nn.Conv2d(1024, 1024, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(1024),
                    nn.LeakyReLU(0.1, inplace=True),
                ),
                "layer3": nn.Sequential(
                    nn.LazyConv2d(1024, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(1024),
                    nn.LeakyReLU(0.1, inplace=True),
                ),
            }
        )

        self.head = nn.Conv2d(1024, self.num_anchors * (5 + self.num_classes), kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)

        x_fgr = self.backbone.layer3(x)
        x_full = self.backbone.layer4(x_fgr)

        x_full = self.neck.layer1(x_full)
        x_full = self.neck.layer2(x_full)

        x_fgr = x_fgr.view(x_fgr.size(0), -1, x_fgr.size(2) // 2, x_fgr.size(3) // 2)
        x_cat = torch.cat((x_fgr, x_full), dim=1)

        x_cat = self.neck.layer3(x_cat)
        x_cat = self.head(x_cat)

        # (N, B * (5 + C), S, S) -> (N, S, S, B, 5 + C)
        x_cat = x_cat.permute(0, 2, 3, 1)  # Channels last.
        x_cat = x_cat.view(x_cat.size(0), x_cat.size(1), x_cat.size(2), self.num_anchors, 5 + self.num_classes)

        # Sigmoid for x, y, w, h, conf and Softmax for classes.
        x_scale = x_cat.clone()
        x_scale[..., :5] = torch.sigmoid(x_scale[..., :5])
        x_scale[..., 5:] = torch.softmax(x_scale[..., 5:], dim=-1)

        return x_scale
