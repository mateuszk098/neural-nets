from collections import namedtuple

import torch
import torch.nn as nn
from torch.types import Tensor
from torchvision.ops import sigmoid_focal_loss

NamedLoss = namedtuple("NamedLoss", ("total", "focal", "dice"))


class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0) -> None:
        super().__init__()
        self.alpha = float(alpha)
        self.gamma = float(gamma)

    def forward(self, inputs: Tensor, targets: Tensor) -> Tensor:
        return sigmoid_focal_loss(inputs, targets, self.alpha, self.gamma, reduction="mean")


class DiceLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.eps = 1e-9

    def forward(self, inputs: Tensor, targets: Tensor) -> Tensor:
        inputs = torch.sigmoid(inputs).clamp(self.eps, 1 - self.eps)
        numerator = 2 * (inputs * targets).sum()
        denominator = inputs.sum() + targets.sum()
        return 1 - (numerator + self.eps) / (denominator + self.eps)


class ComboLoss(nn.Module):
    def __init__(self, focal_alpha: float = 0.25, focal_gamma: float = 2.0, beta: float = 0.5) -> None:
        super().__init__()
        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.dice_loss = DiceLoss()
        self.beta = float(beta)

    def forward(self, inputs: Tensor, targets: Tensor) -> NamedLoss:
        focal_loss = self.beta * self.focal_loss(inputs, targets)
        dice_loss = (1 - self.beta) * self.dice_loss(inputs, targets)
        total_loss = focal_loss + dice_loss
        return NamedLoss(total=total_loss, focal=focal_loss, dice=dice_loss)
