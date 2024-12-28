import torch
import torch.nn as nn
from torch.types import Tensor

from .utils import iou


class YOLOv1Loss(nn.Module):
    def __init__(self, S: int, B: int, C: int, lambda_coord: float = 5.0, lambda_noobj: float = 0.5) -> None:
        super().__init__()
        self.S = int(S)
        self.B = int(B)
        self.C = int(C)
        self.lambda_coord = float(lambda_coord)
        self.lambda_noobj = float(lambda_noobj)

    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        y_pred = y_pred.reshape(-1, self.S, self.S, 5 * self.B + self.C)
        y_true = y_true.reshape(-1, self.S, self.S, 5 * self.B + self.C)

        # Extract the bounding boxes from the predictions and targets
        pred_bboxes = y_pred[..., : 5 * self.B].reshape(-1, self.S, self.S, self.B, 5)
        true_bboxes = y_true[..., : 5 * self.B].reshape(-1, self.S, self.S, self.B, 5)

        # Create the offset grid to extract the bounding box coordinates.
        # From x_offset, y_offset, sqrt(w), sqrt(h) -> x1, y1, x2, y2 (normalized to 0 - 1).
        offset_y, offset_x = torch.meshgrid(torch.arange(self.S) / self.S, torch.arange(self.S) / self.S, indexing="ij")
        # (batch_size, S, S, B).
        offset_y = offset_y.reshape(1, self.S, self.S, 1).repeat(1, 1, 1, self.B)
        offset_x = offset_x.reshape(1, self.S, self.S, 1).repeat(1, 1, 1, self.B)

        offset_y = offset_y.to(y_pred.device)
        offset_x = offset_x.to(y_pred.device)

        # Extract the bounding box coordinates -> (batch_size, S, S, B, 4).
        pred_bboxes_xyxy = torch.stack(
            [
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
        max_iou_values, max_iou_indices = bboxes_iou.max(dim=-1, keepdim=True)
        max_iou_values = max_iou_values.repeat(1, 1, 1, self.B)
        max_iou_indices = max_iou_indices.repeat(1, 1, 1, self.B)
        bboxes_indices = torch.arange(self.B).reshape(1, 1, 1, self.B).expand_as(max_iou_indices)

        max_iou_values = max_iou_values.to(y_pred.device)
        max_iou_indices = max_iou_indices.to(y_pred.device)
        bboxes_indices = bboxes_indices.to(y_pred.device)

        # Object appears in the cell?
        objectness_indicator = y_true[..., 4:5].bool()
        # The jth bounding box predictor in cell i is responsible for that prediction?
        predictors_indicator = (max_iou_indices == bboxes_indices) & objectness_indicator

        pred_classes = y_pred[..., 5 * self.B :]
        true_classes = y_true[..., 5 * self.B :]

        x_loss = ((pred_bboxes[..., 0] - true_bboxes[..., 0]).square()).mul(predictors_indicator).sum()
        y_loss = ((pred_bboxes[..., 1] - true_bboxes[..., 1]).square()).mul(predictors_indicator).sum()
        w_sqrt_loss = ((pred_bboxes[..., 2] - true_bboxes[..., 2]).square()).mul(predictors_indicator).sum()
        h_sqrt_loss = ((pred_bboxes[..., 3] - true_bboxes[..., 3]).square()).mul(predictors_indicator).sum()

        localization_loss = x_loss + y_loss + w_sqrt_loss + h_sqrt_loss
        objectness_loss = (pred_bboxes[..., 4] - max_iou_values).square().mul(predictors_indicator).sum()
        no_objectness_loss = pred_bboxes[..., 4].square().mul(predictors_indicator.bitwise_not()).sum()
        classification_loss = (pred_classes - true_classes).square().mul(objectness_indicator).sum()

        total_loss = (
            localization_loss * self.lambda_coord
            + objectness_loss
            + no_objectness_loss * self.lambda_noobj
            + classification_loss
        )

        # print(
        #     localization_loss.detach().cpu() * self.lambda_coord,
        #     objectness_loss.detach().cpu(),
        #     no_objectness_loss.detach().cpu() * self.lambda_noobj,
        #     classification_loss.detach().cpu(),
        # )

        return total_loss / len(y_true)
