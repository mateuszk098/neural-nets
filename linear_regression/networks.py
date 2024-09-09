import torch
import torch.nn as nn
from torch import Tensor


class LinearRegression(nn.Module):
    def __init__(self, seed: int) -> None:
        super().__init__()
        self.gen = torch.manual_seed(seed)
        self.weight = nn.Parameter(torch.randn(1, generator=self.gen, requires_grad=True))
        self.bias = nn.Parameter(torch.randn(1, generator=self.gen, requires_grad=True))

    def __call__(self, x: Tensor) -> Tensor:
        return self.predict(x)

    def forward(self, x: Tensor) -> Tensor:
        return self.weight * x + self.bias

    @torch.inference_mode()
    def predict(self, x: Tensor) -> Tensor:
        return self.forward(x)
