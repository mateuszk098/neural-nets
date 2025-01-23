from collections import namedtuple
from os import PathLike

import numpy as np
import torch
from torch.types import Tensor
from torchvision.ops import nms

from cvnets.yolo.utils import create_offsets, xywh2xyxy

Prediction = namedtuple("Prediction", ["xyxys", "confs", "labels"])


def anchor_iou(whs1: Tensor, whs2: Tensor) -> Tensor:
    area1 = whs1[..., 0] * whs1[..., 1]
    area2 = whs2[..., 0] * whs2[..., 1]
    intersection = torch.min(whs1[..., 0], whs2[..., 0]) * torch.min(whs1[..., 1], whs2[..., 1])
    return intersection / (area1 + area2 - intersection + 1e-6)


def load_anchor_bboxes(path: str | PathLike) -> Tensor:
    return torch.from_numpy(np.load(path)).float()


def decode_predictions(predictions: Tensor, anchors: Tensor, downsample: int, imgsz: int) -> Tensor:
    predictions = predictions.cpu()

    S = int(imgsz) // int(downsample)
    cys, cxs = create_offsets(S, anchors.size(0))
    anchors = anchors.reshape(1, 1, 1, anchors.size(0), 2).repeat(1, S, S, 1, 1)
    decoded = torch.zeros_like(predictions)

    decoded[..., 0] = cxs + predictions[..., 0] / S
    decoded[..., 1] = cys + predictions[..., 1] / S
    decoded[..., 2] = anchors[..., 0] * torch.exp(predictions[..., 2])
    decoded[..., 3] = anchors[..., 1] * torch.exp(predictions[..., 3])

    decoded[..., :4] = xywh2xyxy(decoded[..., :4]).clamp(0, 1).mul(imgsz)
    decoded[..., 4:] = predictions[..., 4:]

    return decoded


def decode_target(target: Tensor, anchor_bboxes: Tensor, downsample: int, imgsz: int) -> Tensor:
    S = int(imgsz) // int(downsample)
    decoded = decode_predictions(target, anchor_bboxes, S, imgsz)
    decoded[..., :4] = torch.masked_fill(decoded[..., :4], target[..., :4] == 0, 0)
    return decoded


def postprocess_predictions(
    predictions: Tensor,
    anchors: Tensor,
    *,
    downsample: int = 32,
    imgsz: int = 416,
    num_classes: int = 20,
    conf_thresh: float = 0.2,
    nms_thresh: float = 0.4,
) -> list[Prediction]:
    decoded_preds = decode_predictions(predictions, anchors, downsample, imgsz)
    decoded_preds = decoded_preds.reshape(decoded_preds.size(0), -1, 5 + num_classes)
    filtered = list()

    for pred in decoded_preds:
        labels = pred[:, 5:].argmax(dim=1)
        xyxys = pred[:, :4]
        # Prediction confidence is the product of objectness and class confidence.
        confs = pred[:, 4] * pred[:, 5:].amax(dim=1)

        labels = labels[confs > conf_thresh]
        xyxys = xyxys[confs > conf_thresh]
        confs = confs[confs > conf_thresh]

        keep = nms(xyxys, confs, iou_threshold=nms_thresh)

        labels = labels[keep]
        confs = confs[keep]
        xyxys = xyxys[keep]

        filtered.append(Prediction(xyxys, confs, labels))

    return filtered
