from typing import Any

import torch
import torch.nn as nn
from torch.types import Tensor


class VariationalEncoder(nn.Module):
    def __init__(self, in_features: int, hidden_features: list[int]) -> None:
        super().__init__()
        self.fc = nn.Sequential()
        self.fc.append(nn.Linear(in_features, hidden_features[0]))
        self.fc.append(nn.LeakyReLU(0.1, inplace=True))

        for in_feats, out_feats in zip(hidden_features[:-1], hidden_features[1:]):
            self.fc.append(nn.Linear(in_feats, out_feats))
            self.fc.append(nn.LeakyReLU(0.1, inplace=True))

    def forward(self, x: Tensor) -> Tensor:
        z = self.fc(x)
        return z


class LatentSampler(nn.Module):
    def __init__(self, in_features: int, latent_features: int) -> None:
        super().__init__()
        self.mu = nn.Linear(in_features, latent_features)
        self.logvar = nn.Linear(in_features, latent_features)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        mu = self.mu(x)
        logvar = self.logvar(x)
        z = torch.randn_like(mu) * torch.exp(0.5 * logvar) + mu
        return z, mu, logvar


class VariationalDecoder(nn.Module):
    def __init__(self, out_features: int, latent_features: int, hidden_features: list[int]) -> None:
        super().__init__()
        self.fc = nn.Sequential()
        self.fc.append(nn.Linear(latent_features, hidden_features[0]))
        self.fc.append(nn.LeakyReLU(0.1, inplace=True))

        for in_feats, out_feats in zip(hidden_features[:-1], hidden_features[1:]):
            self.fc.append(nn.Linear(in_feats, out_feats))
            self.fc.append(nn.LeakyReLU(0.1, inplace=True))

        self.fc.append(nn.Linear(hidden_features[-1], out_features))

    def forward(self, x: Tensor) -> Tensor:
        z = self.fc(x)
        return z


class VAENet(nn.Module):
    def __init__(self, input_shape: tuple[int, ...], latent_features: int, hidden_units: list[int]) -> None:
        super().__init__()
        self.input_shape = tuple(input_shape)
        self.n_features = int(torch.prod(torch.tensor(input_shape)))

        self.encoder = VariationalEncoder(self.n_features, hidden_units)
        self.sampler = LatentSampler(hidden_units[-1], latent_features)
        self.decoder = VariationalDecoder(self.n_features, latent_features, hidden_units[::-1])

    def forward(self, x: Tensor) -> Any:
        z = torch.flatten(x, start_dim=1)

        z = self.encoder(z)
        z, mu, logvar = self.sampler(z)
        z = self.decoder(z)
        z = z.reshape(-1, *self.input_shape)

        if self.training:
            return z, mu, logvar

        return z
