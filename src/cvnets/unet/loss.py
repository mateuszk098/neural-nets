from collections import namedtuple

import torch
import torch.nn as nn
from torch.types import Tensor
from torchvision.ops import sigmoid_focal_loss

NamedLoss = namedtuple("NamedLoss", ("total", "focal", "dice"))


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0) -> None:
        super().__init__()
        self.alpha = float(alpha)
        self.gamma = float(gamma)

    def forward(self, inputs: Tensor, targets: Tensor) -> Tensor:
        return sigmoid_focal_loss(inputs, targets, self.alpha, self.gamma, reduction="mean")


class DiceLoss(nn.Module):
    def __init__(self, eps: float = 1e-9) -> None:
        super().__init__()
        self.eps = float(eps)

    def forward(self, inputs: Tensor, targets: Tensor) -> Tensor:
        inputs = torch.sigmoid(inputs)
        numerator = 2 * (inputs * targets).mean()
        denominator = inputs.mean() + targets.mean()
        return 1 - (numerator + self.eps) / (denominator + self.eps)


class ComboLoss(nn.Module):
    def __init__(self, focal_alpha=0.25, focal_gamma=2.0, dice_eps=1e-9, beta: float = 0.5) -> None:
        super().__init__()
        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.dice_loss = DiceLoss(eps=dice_eps)
        self.beta = float(beta)

    def forward(self, inputs: Tensor, targets: Tensor) -> NamedLoss:
        focal_loss = self.beta * self.focal_loss(inputs, targets)
        dice_loss = (1 - self.beta) * self.dice_loss(inputs, targets)
        total_loss = focal_loss + dice_loss
        return NamedLoss(total=total_loss, focal=focal_loss, dice=dice_loss)
