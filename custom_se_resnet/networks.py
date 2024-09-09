import torch
import torch.nn as nn
from torch import Tensor

import layers


class SEResNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.feed_forward = nn.Sequential(
            #
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(num_features=32),
            nn.Mish(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            #
            layers.SEResidualBlock(32, 64, kernel_size=3, stride=1, squeeze_active=True),
            layers.SEResidualBlock(64, 64, kernel_size=3, stride=1, squeeze_active=True),
            layers.MaxDepthPool2d(pool_size=2),
            nn.MaxPool2d(kernel_size=2, stride=2),
            #
            layers.SEResidualBlock(32, 96, kernel_size=5, stride=1, squeeze_active=True),
            layers.SEResidualBlock(96, 96, kernel_size=5, stride=1, squeeze_active=True),
            layers.MaxDepthPool2d(pool_size=2),
            nn.MaxPool2d(kernel_size=2, stride=2),
            #
            layers.SEResidualBlock(48, 128, kernel_size=3, stride=1, squeeze_active=True),
            layers.SEResidualBlock(128, 128, kernel_size=3, stride=1, squeeze_active=True),
            layers.MaxDepthPool2d(pool_size=4),
            nn.MaxPool2d(kernel_size=2, stride=2),
            #
            nn.Flatten(),
            #
            nn.Linear(32 * 8 * 8, 256, bias=False),
            nn.BatchNorm1d(num_features=256),
            nn.Mish(),
            nn.Dropout1d(0.4),
            #
            nn.Linear(256, 256, bias=False),
            nn.BatchNorm1d(num_features=256),
            nn.Mish(),
            nn.Dropout1d(0.4),
            #
            nn.Linear(256, 1),
        )

    def __call__(self, x: Tensor) -> Tensor:
        return self.predict(x)

    def forward(self, x: Tensor) -> Tensor:
        return self.feed_forward(x)

    @torch.inference_mode()
    def predict(self, x: Tensor) -> Tensor:
        return self.feed_forward(x)
