import torch
import torch.nn as nn
from torch.types import Tensor


class VAELoss(nn.Module):
    def __init__(self, reconst_alpha=0.7, kl_alpha: float = 1.0, mu_alpha: float = 0.01) -> None:
        super().__init__()
        self.rt_alpha = float(reconst_alpha)
        self.kl_alpha = float(kl_alpha)
        self.mu_alpha = float(mu_alpha)

    def forward(
        self,
        input: Tensor,
        target: Tensor,
        masked: Tensor,
        mu: Tensor | None = None,
        logvar: Tensor | None = None,
    ) -> Tensor:
        reconst_loss = self.reconst_loss(input, target, masked)
        if not self.training:
            return reconst_loss

        assert mu is not None and logvar is not None

        latent_loss = self.latent_loss(mu, logvar)
        mu_var_loss = self.mu_var_loss(mu)

        return reconst_loss + latent_loss + mu_var_loss

    def reconst_loss(self, input: Tensor, target: Tensor, masked: Tensor) -> Tensor:
        no_kpts_mask = masked.eq(0).float()
        is_kpts_mask = masked.ne(0).float()

        no_kpts_loss = (input - target).mul(no_kpts_mask).square().sum()
        is_kpts_loss = (input - target).mul(is_kpts_mask).square().sum()

        reconst_loss = self.rt_alpha * no_kpts_loss + (1 - self.rt_alpha) * is_kpts_loss
        reconst_loss = reconst_loss.div(input.size(0))

        return reconst_loss

    def latent_loss(self, mu: Tensor, logvar: Tensor) -> Tensor:
        loss = -0.5 * torch.mean(1 + logvar - logvar.exp() - mu.square())
        return loss * self.kl_alpha

    def mu_var_loss(self, mu: Tensor) -> Tensor:
        return torch.mean(mu.var(dim=0) - 1).square() * self.mu_alpha
