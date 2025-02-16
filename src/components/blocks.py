import torch
import torch.nn as nn
from torch.types import Tensor


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.residual = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.shortcut = nn.Identity()
        if in_channels != out_channels or stride > 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        residual = self.residual(x)
        shortcut = self.shortcut(x)
        return self.activation(residual + shortcut)


class SqueezeExcitationBlock(nn.Module):
    def __init__(self, in_channels: int, factor: int = 8) -> None:
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(in_channels, in_channels // factor),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // factor, in_channels),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        se = self.se(x)
        return x * se.view(x.size(0), x.size(1), 1, 1)


class MaxDepthPool(nn.Module):
    def __init__(self, pool_size: int) -> None:
        super().__init__()
        self.pool_size = int(pool_size)

    def forward(self, x: Tensor) -> Tensor:
        x = x.view(x.size(0), x.size(1) // self.pool_size, self.pool_size, x.size(2), x.size(3))
        return torch.amax(x, dim=2)


class AvgDepthPool(nn.Module):
    def __init__(self, pool_size: int) -> None:
        super().__init__()
        self.pool_size = int(pool_size)

    def forward(self, x: Tensor) -> Tensor:
        x = x.view(x.size(0), x.size(1) // self.pool_size, self.pool_size, x.size(2), x.size(3))
        return torch.mean(x, dim=2)


class SpatialCBAMBlock(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.max_pool = MaxDepthPool(in_channels)
        self.avg_pool = AvgDepthPool(in_channels)
        self.spatial = nn.Sequential(nn.Conv2d(2, 1, kernel_size=7, padding=3), nn.Sigmoid())

    def forward(self, x: Tensor) -> Tensor:
        x_max = self.max_pool(x)
        x_avg = self.avg_pool(x)
        stack = torch.cat((x_max, x_avg), dim=1)
        return x * self.spatial(stack)


class SpatialFusionBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=stride, padding=2, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.conv7 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=7, stride=stride, padding=3, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.max_pool = MaxDepthPool(3)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        x3 = self.conv3(x)
        x5 = self.conv5(x)
        x7 = self.conv7(x)
        stack = torch.cat((x3, x5, x7), dim=1)
        x_max = self.max_pool(stack)
        return self.activation(x_max)
