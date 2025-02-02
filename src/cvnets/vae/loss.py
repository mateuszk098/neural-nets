import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.types import Tensor


class VAELoss(nn.Module):
    def __init__(self, beta: float = 1.0) -> None:
        super().__init__()
        self.beta = float(beta)

    def forward(self, input: Tensor, target: Tensor, mu: Tensor, logvar: Tensor) -> Tensor:
        reconst_loss = self.mse_loss(input, target)
        latent_loss = self.latent_loss(mu, logvar) * self.beta
        return reconst_loss + latent_loss

    def mse_loss(self, input: Tensor, target: Tensor) -> Tensor:
        # https://stats.stackexchange.com/questions/502314/variational-autoencoder-balance-kl-divergence-and-reconstructionloss
        return F.mse_loss(input, target, reduction="sum") / input.size(0)

    def latent_loss(self, mu: Tensor, logvar: Tensor) -> Tensor:
        return -0.5 * torch.mean(1 + logvar - logvar.exp() - mu.square())
