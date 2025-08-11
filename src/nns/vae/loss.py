import torch
import torch.nn as nn
from torch.types import Tensor


class VAELoss(nn.Module):
    def __init__(
        self,
        reconst_alpha: float = 0.6,
        kl_alpha: float = 0.01,
        kl_alpha_multiplier: float = 1.01,
        latent_alpha: float = 0.1,
        latent_alpha_multiplier: float = 1.01,
    ) -> None:
        super().__init__()
        self.reconst_alpha = float(reconst_alpha)
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

        assert mu is not None and logvar is not None, "mu and logvar must be provided during training"

        latent_loss = self.latent_loss(mu, logvar)
        latent_var_loss = self.latent_var_loss(mu, logvar)

        return reconst_loss + latent_loss + latent_var_loss

    def step(self) -> None:
        # Initially we mostly focus on reconstruction loss, and then we gradually
        # increase the importance of the KL and latent variance to encourage more diversity.
        self.kl_alpha = max(0.0, self.kl_alpha * self.kl_alpha_multiplier)
        self.latent_alpha = max(0.0, self.latent_alpha * self.latent_alpha_multiplier)

    def reconst_loss(self, input: Tensor, target: Tensor, masked: Tensor) -> Tensor:
        # What to do with no visible keypoints? Good question, but here I just don't care.
        is_visible = target.ne(0).float()

        is_masked = masked.eq(0).float().mul(is_visible)
        no_masked = masked.ne(0).float().mul(is_visible)

        is_masked_loss = (input - target).mul(is_masked).square().sum()
        no_masked_loss = (input - target).mul(no_masked).square().sum()

        reconst_loss = self.reconst_alpha * is_masked_loss + (1 - self.reconst_alpha) * no_masked_loss
        reconst_loss = reconst_loss.div(input.size(0))

        return reconst_loss

    def latent_loss(self, mu: Tensor, logvar: Tensor) -> Tensor:
        loss = -0.5 * torch.mean(1 + logvar - logvar.exp() - mu.square())
        return loss * self.kl_alpha

    def latent_var_loss(self, mu: Tensor, logvar: Tensor) -> Tensor:
        # This is a helpful regularization term to keep means and variances of the
        # latent space appropriately distributed (spreaded).
        mu_var_loss = torch.mean(mu.var(dim=0) - 1).square()
        logvar_var_loss = torch.mean(logvar.var(dim=0) - 1).square()
        return self.latent_alpha * (mu_var_loss + logvar_var_loss)
