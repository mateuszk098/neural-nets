from enum import Enum
from itertools import repeat

import torch
import torch.nn as nn
from torch.types import Tensor


class LazyBasicBlock(nn.Module):
    def __init__(self, filters: int, kernel: int, stride: int) -> None:
        super().__init__()
        self.padding = int(kernel // 2)
        self.stride = int(stride)

        self.conv1 = nn.LazyConv2d(filters, kernel, stride, self.padding, bias=False)
        self.bn1 = nn.LazyBatchNorm2d()
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.LazyConv2d(filters, kernel, 1, self.padding, bias=False)
        self.bn2 = nn.LazyBatchNorm2d()

        self.downsample: nn.Identity | nn.Sequential | None = None

    def forward(self, x: Tensor) -> Tensor:
        x_residual = self.conv1(x)
        x_residual = self.bn1(x_residual)
        x_residual = self.relu(x_residual)
        x_residual = self.conv2(x_residual)
        x_residual = self.bn2(x_residual)

        if self.downsample is None:
            self.downsample = self._create_downsample(x.shape[1], x_residual.shape[1], self.stride)
        x_shortcut = self.downsample(x)

        return self.relu(x_residual + x_shortcut)

    def _create_downsample(self, in_filters: int, out_filters: int, stride: int) -> nn.Sequential | nn.Identity:
        shortcut_connection = nn.Identity()
        if in_filters != out_filters or stride > 1:
            return nn.Sequential(nn.LazyConv2d(out_filters, 1, stride, bias=False), nn.LazyBatchNorm2d())
        return shortcut_connection


class LazyBottleneck(nn.Module):
    def __init__(self, filters: int, kernel: int, stride: int) -> None:
        super().__init__()
        self.padding = int(kernel // 2)
        self.stride = int(stride)

        self.conv1 = nn.LazyConv2d(filters, 1, 1, 0, bias=False)
        self.bn1 = nn.LazyBatchNorm2d()
        self.conv2 = nn.LazyConv2d(filters, kernel, stride, self.padding, bias=False)
        self.bn2 = nn.LazyBatchNorm2d()
        self.conv3 = nn.LazyConv2d(filters * 4, 1, 1, 0, bias=False)
        self.bn3 = nn.LazyBatchNorm2d()
        self.relu = nn.ReLU(inplace=True)

        self.downsample: nn.Identity | nn.Sequential | None = None

    def forward(self, x: Tensor) -> Tensor:
        x_residual = self.conv1(x)
        x_residual = self.bn1(x_residual)
        x_residual = self.relu(x_residual)
        x_residual = self.conv2(x_residual)
        x_residual = self.bn2(x_residual)
        x_residual = self.relu(x_residual)
        x_residual = self.conv3(x_residual)
        x_residual = self.bn3(x_residual)

        if self.downsample is None:
            self.downsample = self._create_downsample(x.shape[1], x_residual.shape[1], self.stride)
        x_shortcut = self.downsample(x)

        return self.relu(x_residual + x_shortcut)

    def _create_downsample(self, in_filters: int, out_filters: int, stride: int) -> nn.Sequential | nn.Identity:
        shortcut_connection = nn.Identity()
        if in_filters != out_filters or stride > 1:
            return nn.Sequential(nn.LazyConv2d(out_filters, 1, stride, 0, bias=False), nn.LazyBatchNorm2d())
        return shortcut_connection


class Structure(Enum):
    BASIC = LazyBasicBlock
    BOTTLENECK = LazyBottleneck


class ResNet(nn.Module):
    def __init__(self, structure: Structure, repeats: list[int], fc_units: int) -> None:
        super().__init__()
        self.conv1 = nn.LazyConv2d(64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.LazyBatchNorm2d()
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self._build_layers(structure, repeats)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.LazyLinear(fc_units)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def _build_layers(self, structure: Structure, repeats: list[int]) -> None:
        prev_filters = 64
        filters = (64, 128, 256, 512)

        for i, k in enumerate(repeats):
            self.__setattr__(f"layer{i+1}", nn.Sequential())
            for n_filters in repeat(filters[i], k):
                stride = 1 if n_filters == prev_filters else 2
                self.__getattr__(f"layer{i+1}").append(structure.value(n_filters, 3, stride))
                prev_filters = n_filters
