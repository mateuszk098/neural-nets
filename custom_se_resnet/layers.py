import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class MaxDepthPool2d(nn.Module):
    """Max pooling op over the channel dimension."""

    def __init__(self, pool_size: int) -> None:
        super().__init__()
        self.pool_size = int(pool_size)

    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)

    def forward(self, x: Tensor) -> Tensor:
        old_shape = x.shape
        new_shape = (old_shape[0], old_shape[1] // self.pool_size, self.pool_size, *old_shape[2:])
        return torch.amax(x.view(new_shape), dim=2)


class SqueezeExcitation(nn.Module):
    """Squeeze and Excitation block."""

    def __init__(self, in_channels: int, factor: int) -> None:
        super().__init__()
        self.squeeze_channels = int(in_channels // factor)
        self.feed_forward = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, self.squeeze_channels),
            nn.Mish(),
            nn.Linear(self.squeeze_channels, in_channels),
            nn.Sigmoid(),
        )

    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)

    def forward(self, x: Tensor) -> Tensor:
        return x * self.feed_forward(x).view(x.shape[0], x.shape[1], 1, 1)


class ResidualBlock(nn.Module):
    """Residual connection block."""

    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1
    ) -> None:
        super().__init__()
        self.padding = int(kernel_size // 2)
        self.feed_forward = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, self.padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Mish(),
            nn.Conv2d(out_channels, out_channels, kernel_size, 1, self.padding, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.shortcut_connection = nn.Sequential()
        if in_channels != out_channels or stride > 1:
            self.shortcut_connection = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)

    def forward(self, x: Tensor) -> Tensor:
        x_residual = self.feed_forward(x)
        x_shortcut = self.shortcut_connection(x)
        return F.mish(x_residual + x_shortcut)


class SEResidualBlock(nn.Module):
    """Residual connection with Squeeze and Excitation block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        squeeze_factor: int = 8,
        squeeze_active: bool = True,
    ) -> None:
        super().__init__()
        self.residual_block = ResidualBlock(in_channels, out_channels, kernel_size, stride)
        self.squeeze_excitation = SqueezeExcitation(out_channels, squeeze_factor)
        self.squeeze_active = bool(squeeze_active)

    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)

    def forward(self, x: Tensor) -> Tensor:
        x_residual = self.residual_block.feed_forward(x)
        x_shortcut = self.residual_block.shortcut_connection(x)
        residual_output = F.mish(x_residual + x_shortcut)
        if self.squeeze_active:
            return x_shortcut + self.squeeze_excitation(residual_output)
        return residual_output
