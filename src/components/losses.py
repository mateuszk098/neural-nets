from typing import Literal

import torch
import torch.nn as nn
from torch.types import Tensor
from torchmetrics.functional.image import learned_perceptual_image_patch_similarity as lpips
from torchmetrics.functional.image import structural_similarity_index_measure as ssim
from torchvision.ops import sigmoid_focal_loss


class SSIMLoss(nn.Module):
    def __init__(self, sigma: float = 1.5, kernel_size: int = 11) -> None:
        super().__init__()
        self.sigma = float(sigma)
        self.kernel_size = int(kernel_size)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        # Higher SSIM is better so the loss is the opposite.
        return 1 - ssim(
            preds=input,
            target=target,
            sigma=self.sigma,
            kernel_size=self.kernel_size,
            reduction="elementwise_mean",
        )  # type: ignore


class LPIPSLoss(nn.Module):
    def __init__(
        self,
        net_type: Literal["alex", "vgg", "squeeze"] = "alex",
        reduction: Literal["sum", "mean"] = "mean",
        normalize: bool = True,
    ) -> None:
        super().__init__()
        self.net_type = str(net_type)
        self.reduction = str(reduction)
        self.normalize = bool(normalize)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return lpips(
            img1=input,
            img2=target,
            net_type=self.net_type,  # type: ignore
            reduction=self.reduction,  # type: ignore
            normalize=self.normalize,
        )


class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0) -> None:
        super().__init__()
        self.alpha = float(alpha)
        self.gamma = float(gamma)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return sigmoid_focal_loss(input, target, alpha=self.alpha, gamma=self.gamma, reduction="mean")


class DiceLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.eps = 1e-9

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        probas = torch.sigmoid(input).clamp(self.eps, 1 - self.eps)
        intersection = torch.sum(probas * target)
        union = probas.sum() + target.sum()
        return 1 - (2 * intersection + self.eps) / (union + self.eps)


class TverskyLoss(nn.Module):
    def __init__(self, alpha: float = 0.3, beta: float = 0.7, eps: float = 1e-6) -> None:
        super().__init__()
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.eps = float(eps)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        probas = torch.sigmoid(input).clamp(self.eps, 1 - self.eps)
        intersection = torch.sum(probas * target)
        fps = torch.sum(probas * (1 - target))
        fns = torch.sum((1 - probas) * target)
        return 1 - (intersection + self.eps) / (intersection + self.alpha * fps + self.beta * fns + self.eps)
