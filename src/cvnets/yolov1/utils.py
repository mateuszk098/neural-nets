from os import PathLike
from typing import Any

import torch
import torch.nn as nn
import yaml
from torch.types import Tensor
from torchvision.ops import nms


def iou(bboxes1: Tensor, bboxes2: Tensor) -> Tensor:
    x1 = torch.max(bboxes1[..., 0], bboxes2[..., 0])
    y1 = torch.max(bboxes1[..., 1], bboxes2[..., 1])
    x2 = torch.min(bboxes1[..., 2], bboxes2[..., 2])
    y2 = torch.min(bboxes1[..., 3], bboxes2[..., 3])

    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    area1 = (bboxes1[..., 2] - bboxes1[..., 0]) * (bboxes1[..., 3] - bboxes1[..., 1])
    area2 = (bboxes2[..., 2] - bboxes2[..., 0]) * (bboxes2[..., 3] - bboxes2[..., 1])

    return intersection / (area1 + area2 - intersection + 1e-6)


# def nms(detections: Tensor, iou_threshold: float = 0.3) -> Tensor:
#     sorted_dets = sorted(detections, key=lambda x: x[-1])  # type: ignore
#     keep = list()

#     while sorted_dets:
#         current_det = sorted_dets.pop()
#         sorted_dets = list(det for det in sorted_dets if iou(current_det, det[:-1]) < iou_threshold)
#         keep.append(current_det)

#     return torch.stack(keep, dim=0)


def decode_yolo_output(prediction: Tensor, imgsz: int, S: int, B: int, C: int) -> tuple[Tensor, Tensor, Tensor]:
    prediction = prediction.reshape(-1, S, S, 5 * B + C).clamp(0, 1)
    device = prediction.device

    class_scores, class_ids = prediction[..., 5 * B :].max(dim=-1, keepdim=True)

    offset_y, offset_x = torch.meshgrid(torch.arange(S) / S, torch.arange(S) / S, indexing="ij")
    offset_y = offset_y.reshape(1, S, S, 1).repeat(1, 1, 1, B).to(device)
    offset_x = offset_x.reshape(1, S, S, 1).repeat(1, 1, 1, B).to(device)

    pred_bboxes = prediction[..., : 5 * B].reshape(-1, S, S, B, 5)

    xyxys = torch.stack(
        [
            pred_bboxes[..., 0] / S + offset_x - 0.5 * pred_bboxes[..., 2].square(),
            pred_bboxes[..., 1] / S + offset_y - 0.5 * pred_bboxes[..., 3].square(),
            pred_bboxes[..., 0] / S + offset_x + 0.5 * pred_bboxes[..., 2].square(),
            pred_bboxes[..., 1] / S + offset_y + 0.5 * pred_bboxes[..., 3].square(),
        ],
        dim=-1,
    ).mul(imgsz)
    confs = pred_bboxes[..., 4] * class_scores
    class_ids = class_ids.repeat(1, 1, 1, B)

    xyxys = xyxys.reshape(-1, S * S * B, 4)
    confs = confs.reshape(-1, S * S * B)
    class_ids = class_ids.reshape(-1, S * S * B)

    return xyxys, confs, class_ids


def filter_detections(
    xyxys: Tensor,
    confs: Tensor,
    labels: Tensor,
    conf_threshold: float = 0.05,
    nms_threshold: float = 0.3,
) -> tuple[Tensor, Tensor, Tensor]:
    assert xyxys.dim() == 2 and confs.dim() == 1 and labels.dim() == 1, "Input tensors must be 2D, 1D, and 1D"

    labels = labels[confs > conf_threshold]
    xyxys = xyxys[confs > conf_threshold]
    confs = confs[confs > conf_threshold]

    keep = nms(xyxys, confs, iou_threshold=nms_threshold)

    return xyxys[keep], confs[keep], labels[keep]


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
