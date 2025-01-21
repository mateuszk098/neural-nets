from os import PathLike

import numpy as np
import torch
from torch.types import Tensor

from cvnets.yolo.utils import create_offsets, xywh2xyxy


def anchor_iou(whs1: Tensor, whs2: Tensor) -> Tensor:
    area1 = whs1[..., 0] * whs1[..., 1]
    area2 = whs2[..., 0] * whs2[..., 1]
    intersection = torch.min(whs1[..., 0], whs2[..., 0]) * torch.min(whs1[..., 1], whs2[..., 1])
    return intersection / (area1 + area2 - intersection + 1e-6)


def load_anchor_bboxes(path: str | PathLike) -> Tensor:
    return torch.from_numpy(np.load(path)).float()


def decode_preds(preds: Tensor, anchor_bboxes: Tensor, downsample: int, imgsz: int) -> Tensor:
    S = int(imgsz) // int(downsample)
    cys, cxs = create_offsets(S)
    anchors = anchor_bboxes.reshape(1, 1, 1, 5, 2).repeat(1, S, S, 1, 1)
    decoded = torch.zeros_like(preds)

    decoded[..., 0] = cxs + preds[..., 0] / S
    decoded[..., 1] = cys + preds[..., 1] / S
    decoded[..., 2] = anchors[..., 0] * torch.exp(preds[..., 2])
    decoded[..., 3] = anchors[..., 1] * torch.exp(preds[..., 3])

    decoded[..., :4] = xywh2xyxy(decoded[..., :4]) * imgsz
    decoded[..., 4:] = preds[..., 4:]

    return decoded


def decode_target(target: Tensor, anchor_bboxes: Tensor, downsample: int, imgsz: int) -> Tensor:
    S = int(imgsz) // int(downsample)
    decoded = decode_preds(target, anchor_bboxes, S, imgsz)
    decoded[..., :4] = torch.masked_fill(decoded[..., :4], target[..., :4] == 0, 0)
    return decoded
