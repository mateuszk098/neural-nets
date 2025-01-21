from collections import namedtuple

import torch
import torch.nn as nn
from torch.types import Tensor

from cvnets.yolo.utils import iou

NamedLoss = namedtuple("NamedLoss", ["total", "coord", "itobj", "noobj", "label"])


class YOLOv1Loss(nn.Module):
    def __init__(self, S: int, B: int, C: int, lambda_coord: float = 5.0, lambda_noobj: float = 0.5) -> None:
        super().__init__()
        self.S = int(S)
        self.B = int(B)
        self.C = int(C)
        self.lambda_coord = float(lambda_coord)
        self.lambda_noobj = float(lambda_noobj)

    def forward(self, y_pred: Tensor, y_true: Tensor) -> NamedLoss:
        y_pred = y_pred.reshape(-1, self.S, self.S, 5 * self.B + self.C)
        y_true = y_true.reshape(-1, self.S, self.S, 5 * self.B + self.C)
        device = y_pred.device

        # Extract the bounding boxes from the predictions and targets
        pred_bboxes = y_pred[..., : 5 * self.B].reshape(-1, self.S, self.S, self.B, 5)
        true_bboxes = y_true[..., : 5 * self.B].reshape(-1, self.S, self.S, self.B, 5)

        # Create the offset grid to extract the bounding box coordinates.
        # From x_offset, y_offset, sqrt(w), sqrt(h) -> x1, y1, x2, y2 (normalized to 0 - 1).
        offset_y, offset_x = torch.meshgrid(torch.arange(self.S) / self.S, torch.arange(self.S) / self.S, indexing="ij")
        # (batch_size, S, S, B).
        offset_y = offset_y.reshape(1, self.S, self.S, 1).repeat(1, 1, 1, self.B).to(device)
        offset_x = offset_x.reshape(1, self.S, self.S, 1).repeat(1, 1, 1, self.B).to(device)

        # Extract the bounding box coordinates -> (batch_size, S, S, B, 4).
        pred_bboxes_xyxy = torch.stack(
            [
                # Predefined offsets are in 0 - 1, so we must normalize predicted offset by S
                # to be able to add it.
                pred_bboxes[..., 0] / self.S + offset_x - 0.5 * pred_bboxes[..., 2].square(),
                pred_bboxes[..., 1] / self.S + offset_y - 0.5 * pred_bboxes[..., 3].square(),
                pred_bboxes[..., 0] / self.S + offset_x + 0.5 * pred_bboxes[..., 2].square(),
                pred_bboxes[..., 1] / self.S + offset_y + 0.5 * pred_bboxes[..., 3].square(),
            ],
            dim=-1,
        )
        true_bboxes_xyxy = torch.stack(
            [
                true_bboxes[..., 0] / self.S + offset_x - 0.5 * true_bboxes[..., 2].square(),
                true_bboxes[..., 1] / self.S + offset_y - 0.5 * true_bboxes[..., 3].square(),
                true_bboxes[..., 0] / self.S + offset_x + 0.5 * true_bboxes[..., 2].square(),
                true_bboxes[..., 1] / self.S + offset_y + 0.5 * true_bboxes[..., 3].square(),
            ],
            dim=-1,
        )

        # Compute the Intersection over Union (IoU) between the predicted and true bounding boxes.
        bboxes_iou = iou(pred_bboxes_xyxy, true_bboxes_xyxy)

        # Only the bounding boxes with the maximum IoU are responsible for the prediction.
        max_iou_vals, max_iou_ids = bboxes_iou.max(dim=-1, keepdim=True)
        max_iou_vals = max_iou_vals.repeat(1, 1, 1, self.B).to(device)
        max_iou_ids = max_iou_ids.repeat(1, 1, 1, self.B).to(device)
        bboxes_ids = torch.arange(self.B).reshape(1, 1, 1, self.B).expand_as(max_iou_ids).to(device)

        # Object appears in the cell?
        objectness_mask = y_true[..., 4:5].bool()
        # The jth bounding box predictor in cell i is responsible for that prediction?
        predictors_mask = (max_iou_ids == bboxes_ids) & objectness_mask

        pred_classes = y_pred[..., 5 * self.B :]
        true_classes = y_true[..., 5 * self.B :]

        coord_loss = (
            (pred_bboxes[..., :4] - true_bboxes[..., :4])
            .square()
            .mul(predictors_mask.reshape(-1, self.S, self.S, self.B, 1))
            .sum()
        )
        itobj_loss = (pred_bboxes[..., 4] - max_iou_vals).square().mul(predictors_mask).sum()
        noobj_loss = (pred_bboxes[..., 4]).square().mul(predictors_mask.bitwise_not()).sum()
        label_loss = (pred_classes - true_classes).square().mul(objectness_mask).sum()
        total_loss = coord_loss * self.lambda_coord + itobj_loss + noobj_loss * self.lambda_noobj + label_loss

        return NamedLoss(
            total_loss / len(y_true),
            coord_loss * self.lambda_coord / len(y_true),
            itobj_loss / len(y_true),
            noobj_loss * self.lambda_noobj / len(y_true),
            label_loss / len(y_true),
        )
