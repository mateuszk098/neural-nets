import random
import time
from os import PathLike
from pathlib import Path
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
        worker_info.dataset.transform.set_random_seed(worker_seed)  # type: ignore


def xyxy2xywh(bboxes: Tensor) -> Tensor:
    xywh = torch.zeros_like(bboxes)
    xywh[..., :2] = bboxes[..., :2] + 0.5 * (bboxes[..., 2:] - bboxes[..., :2])
    xywh[..., 2:] = bboxes[..., 2:] - bboxes[..., :2]
    return xywh


def xywh2xyxy(bboxes: Tensor) -> Tensor:
    xyxy = torch.zeros_like(bboxes)
    xyxy[..., :2] = bboxes[..., :2] - 0.5 * bboxes[..., 2:]
    xyxy[..., 2:] = bboxes[..., :2] + 0.5 * bboxes[..., 2:]
    return xyxy


def iou(bboxes1: Tensor, bboxes2: Tensor) -> Tensor:
    x1 = torch.max(bboxes1[..., 0], bboxes2[..., 0])
    y1 = torch.max(bboxes1[..., 1], bboxes2[..., 1])
    x2 = torch.min(bboxes1[..., 2], bboxes2[..., 2])
    y2 = torch.min(bboxes1[..., 3], bboxes2[..., 3])

    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    area1 = (bboxes1[..., 2] - bboxes1[..., 0]) * (bboxes1[..., 3] - bboxes1[..., 1])
    area2 = (bboxes2[..., 2] - bboxes2[..., 0]) * (bboxes2[..., 3] - bboxes2[..., 1])

    return intersection / (area1 + area2 - intersection + 1e-6)


def create_offsets(S: int, num_anchors: int) -> tuple[Tensor, Tensor]:
    cys, cxs = torch.meshgrid(torch.arange(S) / S, torch.arange(S) / S, indexing="ij")
    cys = cys.reshape(1, S, S, 1).repeat(1, 1, 1, num_anchors)
    cxs = cxs.reshape(1, S, S, 1).repeat(1, 1, 1, num_anchors)
    return cys, cxs


def dfs_freeze(module: nn.Module) -> None:
    for param in module.parameters():
        param.requires_grad_(False)


def dfs_unfreeze(module: nn.Module) -> None:
    for param in module.parameters():
        param.requires_grad_(True)


def load_yaml(file: str | PathLike) -> dict[str, Any]:
    with open(file, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def create_current_run_dir(root="./checkpoints/") -> Path:
    current_run_dir = Path(root).joinpath(time.strftime("run-%Y-%m-%d-%H-%M-%S"))
    current_run_dir.mkdir(parents=True, exist_ok=True)
    return current_run_dir
