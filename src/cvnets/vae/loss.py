import torch
import torch.nn as nn
from torch.types import Tensor


class VAELoss(nn.Module):
    def __init__(
        self,
        rt_alpha=0.6,
        kl_alpha: float = 0.01,
        kl_alpha_multiplier: float = 1.01,
        latent_alpha: float = 0.1,
        latent_alpha_multiplier: float = 1.01,
    ) -> None:
        super().__init__()
        self.rt_alpha = float(rt_alpha)
        self.kl_alpha = float(kl_alpha)
        self.kl_alpha_multiplier = float(kl_alpha_multiplier)
        self.latent_alpha = float(latent_alpha)
        self.latent_alpha_multiplier = float(latent_alpha_multiplier)

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
        latent_var_loss = self.latent_var_loss(mu, logvar)

        print(
            f"reconst_loss: {reconst_loss.item():.5f}, latent_loss: {latent_loss.item():.5f}, latent_var_loss: {latent_var_loss.item():.5f}"
        )

        return reconst_loss + latent_loss + latent_var_loss

    def step(self) -> None:
        self.kl_alpha = max(0.0, self.kl_alpha * self.kl_alpha_multiplier)
        self.latent_alpha = max(0.0, self.latent_alpha * self.latent_alpha_multiplier)

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

    def latent_var_loss(self, mu: Tensor, logvar: Tensor) -> Tensor:
        mu_var_loss = torch.mean(mu.var(dim=0) - 1).square()
        logvar_var_loss = torch.mean(logvar.var(dim=0) - 1).square()
        return self.latent_alpha * (mu_var_loss + logvar_var_loss)
