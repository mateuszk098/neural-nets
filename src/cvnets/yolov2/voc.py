from torch.types import Tensor

from cvnets.yolov1.voc import VOCDataset as YOLOv1VOCDataset


class VOCDataset(YOLOv1VOCDataset):
    def _create_yolo_target(self, bboxes: Tensor, labels: Tensor) -> Tensor:
        raise NotImplementedError("Implement this for YOLOv2 loss.")
