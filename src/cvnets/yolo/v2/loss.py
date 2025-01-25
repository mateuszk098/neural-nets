from collections import namedtuple

import torch.nn as nn
from torch.types import Tensor

NamedLoss = namedtuple("NamedLoss", ["total", "coord", "itobj", "noobj", "label"])


class YOLOv2Loss(nn.Module):
    def __init__(self, lambda_coord: float = 5.0, lambda_noobj: float = 0.5) -> None:
        super().__init__()
        self.lambda_coord = float(lambda_coord)
        self.lambda_noobj = float(lambda_noobj)

    def forward(self, y_pred: Tensor, y_true: Tensor) -> NamedLoss:
        batch_size = y_pred.size(0)
        itobj_mask = y_true[..., 4]
        noobj_mask = 1 - itobj_mask

        coord_loss = (y_pred[..., :4] - y_true[..., :4]).mul(itobj_mask.unsqueeze(-1)).square().sum()
        label_loss = (y_pred[..., 5:] - y_true[..., 5:]).mul(itobj_mask.unsqueeze(-1)).square().sum()
        itobj_loss = (y_pred[..., 4] - y_true[..., 4]).mul(itobj_mask).square().sum()
        noobj_loss = (y_pred[..., 4] - y_true[..., 4]).mul(noobj_mask).square().sum()

        coord_loss = coord_loss * self.lambda_coord
        noobj_loss = noobj_loss * self.lambda_noobj

        total_loss = coord_loss + itobj_loss + noobj_loss + label_loss

        return NamedLoss(
            total_loss / batch_size,
            coord_loss / batch_size,
            itobj_loss / batch_size,
            noobj_loss / batch_size,
            label_loss / batch_size,
        )
