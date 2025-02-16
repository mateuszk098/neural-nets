import random
from os import PathLike
from typing import Any

import numpy as np
import torch
import yaml


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
