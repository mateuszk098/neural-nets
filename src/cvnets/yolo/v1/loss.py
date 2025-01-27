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
        device = y_pred.device
        batch_size = y_pred.size(0)

        pred_bboxes = y_pred[..., : self.B * 5].reshape(-1, self.S, self.S, self.B, 5)
        true_bboxes = y_true[..., :5].repeat(1, 1, 1, self.B).reshape(-1, self.S, self.S, self.B, 5)

        # Create the offset grid to extract the bounding box coordinates.
        # From x_offset, y_offset, sqrt(w), sqrt(h) -> x1, y1, x2, y2 (normalized to 0 - 1).
        steps = torch.arange(self.S) / self.S
        offset_y, offset_x = torch.meshgrid(steps, steps, indexing="ij")
        # (batch_size, S, S, B).
        offset_y = offset_y.reshape(-1, self.S, self.S, 1).repeat(1, 1, 1, self.B).to(device)
        offset_x = offset_x.reshape(-1, self.S, self.S, 1).repeat(1, 1, 1, self.B).to(device)

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

        bboxes_iou = iou(pred_bboxes_xyxy, true_bboxes_xyxy)
        max_iou_vals, max_iou_ids = torch.max(bboxes_iou, dim=-1, keepdim=True)
        max_iou_vals = max_iou_vals.expand(-1, -1, -1, self.B).to(device)
        max_iou_ids = max_iou_ids.expand(-1, -1, -1, self.B).to(device)

        itobj_mask = y_true[..., 4].reshape(-1, self.S, self.S, 1)
        predictors = torch.arange(self.B).reshape(1, 1, 1, self.B).expand(-1, self.S, self.S, -1).to(device)
        liability_mask = (predictors == max_iou_ids).float() * itobj_mask.expand(-1, -1, -1, self.B)

        coord_loss = pred_bboxes[..., :4] - true_bboxes[..., :4]
        coord_loss = coord_loss.mul(liability_mask.reshape(-1, self.S, self.S, self.B, 1)).square().sum()
        coord_loss = coord_loss * self.lambda_coord

        itobj_loss = max_iou_vals - true_bboxes[..., 4]
        itobj_loss = itobj_loss.mul(liability_mask).square().sum()

        noobj_loss = pred_bboxes[..., 4] - true_bboxes[..., 4]
        noobj_loss = noobj_loss.mul(1 - liability_mask).square().sum()
        noobj_loss = noobj_loss * self.lambda_noobj

        label_loss = y_pred[..., self.B * 5 :] - y_true[..., 5:]
        label_loss = label_loss.mul(itobj_mask).square().sum()

        total_loss = coord_loss + itobj_loss + noobj_loss + label_loss

        return NamedLoss(
            total_loss / batch_size,
            coord_loss / batch_size,
            itobj_loss / batch_size,
            noobj_loss / batch_size,
            label_loss / batch_size,
        )
