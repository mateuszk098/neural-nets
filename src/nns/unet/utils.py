import random
from os import PathLike
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.types import Tensor


def initialize_seed(seed: int = 42) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def worker_init_fn(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is not None:
        worker_info.dataset.set_seed(worker_seed)  # type: ignore


def load_yaml(file: str | PathLike) -> dict[str, Any]:
    with open(file, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class DeNormalizer(nn.Module):
    def __init__(
        self,
        mean: tuple[float, ...] = (0.485, 0.456, 0.406),
        std: tuple[float, ...] = (0.229, 0.224, 0.225),
        max_pixel_value: float = 1.0,
    ) -> None:
        super().__init__()
        self.mean = tuple(mean)
        self.std = tuple(std)
        self.max_pixel_value = float(max_pixel_value)

    def forward(self, x: Tensor) -> Tensor:
        mean = torch.as_tensor(self.mean, dtype=x.dtype, device=x.device)
        std = torch.as_tensor(self.std, dtype=x.dtype, device=x.device)
        x = x * std.view(3, 1, 1) + mean.view(3, 1, 1)
        return x.clamp(0, 1).mul(self.max_pixel_value)


class EarlyStopping:
    def __init__(self, patience: int = 10, min_delta: float = 0.0) -> None:
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.counter = 0
        self.min_value = float("inf")

    def __call__(self, loss: float) -> bool:
        return self.should_stop(loss)

    def should_stop(self, loss: float) -> bool:
        if loss < self.min_value - self.min_delta:
            self.min_value = loss
            self.counter = 0
            return False

        self.counter += 1
        if self.counter == self.patience:
            return True

        return False
