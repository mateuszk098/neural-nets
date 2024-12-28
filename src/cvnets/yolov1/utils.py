from os import PathLike
from typing import Any

import torch
import torch.nn as nn
import yaml
from torch.types import Tensor


def iou(bboxes1: Tensor, bboxes2: Tensor) -> Tensor:
    x1 = torch.max(bboxes1[..., 0], bboxes2[..., 0])
    y1 = torch.max(bboxes1[..., 1], bboxes2[..., 1])
    x2 = torch.min(bboxes1[..., 2], bboxes2[..., 2])
    y2 = torch.min(bboxes1[..., 3], bboxes2[..., 3])

    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    area1 = (bboxes1[..., 2] - bboxes1[..., 0]) * (bboxes1[..., 3] - bboxes1[..., 1])
    area2 = (bboxes2[..., 2] - bboxes2[..., 0]) * (bboxes2[..., 3] - bboxes2[..., 1])

    return intersection / (area1 + area2 - intersection + 1e-6)


# Also remember to set optimizer to only update parameters that require grad. E.g.:
# torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))
def dfs_freeze(model: nn.Module) -> None:
    for name, child in model.named_children():
        for param in child.parameters():
            param.requires_grad_(False)
        dfs_freeze(child)


def dfs_unfreeze(model: nn.Module) -> None:
    for name, child in model.named_children():
        for param in child.parameters():
            param.requires_grad_(True)
        dfs_unfreeze(child)


def load_yaml(file: str | PathLike) -> dict[str, Any]:
    with open(file, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
