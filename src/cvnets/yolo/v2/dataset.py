import os
import random
from collections import namedtuple
from os import PathLike
from typing import Self

os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"

import albumentations as A
import cv2 as cv
import torch
from torch.types import Tensor
from torch.utils.data import Dataset

from cvnets.yolo.utils import xyxy2xywh
from cvnets.yolo.v2.utils import anchor_iou
from cvnets.yolo.voc import VOCSplit, load_voc_dataset

YOLOSample = namedtuple("YOLOSample", ("image", "bboxes", "labels", "target"))
YOLOSampleBatch = namedtuple("YOLOSampleBatch", ("images", "bboxes", "labels", "targets"))


class VOCDataset(Dataset):
    def __init__(
        self,
        root: str | PathLike,
        /,
        *,
        anchors: Tensor,
        split: VOCSplit,
        downsample: int = 2**5,
        imgsz: int = 416,
        normalize: bool = True,
    ) -> None:
        dataset, classes = load_voc_dataset(root, split)

        self.dataset = dataset
        self.class_to_idx = {cls: idx for idx, cls in enumerate(sorted(classes))}
        self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}

        self.anchors = anchors.clone()
        self.downsample = int(downsample)

        self.num_anchors = anchors.size(0)
        self.num_classes = len(self.class_to_idx)
        self.normalize = bool(normalize)

        self._imgsz = int(imgsz)
        self.S = self.imgsz // self.downsample

        self._train_transform = A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.CropAndPad(percent=[-0.25, 0.25], p=1),
                A.Affine(
                    scale=(0.75, 1.25),
                    border_mode=cv.BORDER_REPLICATE,
                    fit_output=True,
                    keep_ratio=True,
                    p=1,
                ),
                A.ColorJitter(
                    brightness=(0.75, 1.25),
                    contrast=(0.75, 1.25),
                    saturation=(0.75, 1.25),
                    hue=(-0.25, 0.25),
                    p=1,
                ),
                A.ToGray(num_output_channels=3, p=0.1),
                A.Resize(self.imgsz, self.imgsz),
            ],
            bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"], min_visibility=0.25),
        )

        self._eval_transform = A.Compose(
            [
                A.Resize(self.imgsz, self.imgsz),
            ],
            bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"]),
        )

        if self.normalize:  # Useful when using non-normalized images for testing.
            self._train_transform.transforms.append(A.Normalize())
            self._eval_transform.transforms.append(A.Normalize())

        self.transform = self._eval_transform

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> YOLOSample:
        image_meta = self.dataset[index]

        image = cv.imread(str(image_meta.path), cv.IMREAD_COLOR)
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        bboxes = list(d.bbox for d in image_meta.detections)
        labels = list(self.class_to_idx[d.label] for d in image_meta.detections)

        transformed = self.transform(image=image, bboxes=bboxes, labels=labels)

        image = torch.from_numpy(transformed["image"]).permute(2, 0, 1).float()  # C, H, W
        bboxes = torch.as_tensor(transformed["bboxes"]).float()
        labels = torch.as_tensor(transformed["labels"]).int()

        if not bboxes.any():  # If all bboxes are removed by augmentations, try again.
            return self.__getitem__(index)

        target = self._create_yolo_target(bboxes, labels)

        return YOLOSample(image, bboxes, labels, target)

    def train(self) -> Self:
        self.transform = self._train_transform
        return self

    def eval(self) -> Self:
        self.transform = self._eval_transform
        return self

    @property
    def imgsz(self) -> int:
        return self._imgsz

    @imgsz.setter
    def imgsz(self, value: int) -> None:
        self._imgsz = int(value)
        self.S = self.imgsz // self.downsample

        for transform in (self._train_transform, self._eval_transform):
            for t in transform.transforms:
                if isinstance(t, A.Resize):
                    t.width, t.height = self.imgsz, self.imgsz
                    break

    def _create_yolo_target(self, bboxes: Tensor, labels: Tensor) -> Tensor:
        target = torch.zeros((self.S, self.S, self.num_anchors, 5 + self.num_classes))
        bboxes_xywh = xyxy2xywh(bboxes)

        cell_size = self.imgsz // self.S
        ns, ms = (bboxes_xywh[:, :2] / cell_size).int().t()
        txys = (bboxes_xywh[:, :2] % cell_size) / cell_size
        whs = bboxes_xywh[:, 2:] / self.imgsz

        for idx, (n, m, txy, wh) in enumerate(zip(ns, ms, txys, whs)):
            k = torch.argmax(anchor_iou(wh, self.anchors))
            twh = torch.log(wh / self.anchors[k])

            target[m, n, k, 0:2] = txy
            target[m, n, k, 2:4] = twh
            target[m, n, k, 4] = 1.0  # IoU of the predicted bbox with the ground truth bbox.
            target[m, n, k, 5 + labels[idx]] = 1.0  # One-hot encode the class label.

        return target


def collate_fn(samples: list[YOLOSample]) -> YOLOSampleBatch:
    batch_images = list()
    batch_labels = list()
    batch_bboxes = list()
    batch_targets = list()

    for image, bboxes, labels, target in samples:
        batch_images.append(image)
        batch_labels.append(labels)
        batch_bboxes.append(bboxes)
        batch_targets.append(target)

    batch_images = torch.stack(batch_images, dim=0)
    batch_targets = torch.stack(batch_targets, dim=0)

    return YOLOSampleBatch(batch_images, batch_bboxes, batch_labels, batch_targets)


def sampled_collate_fn(samples: list[YOLOSample], img_sizes: list[int]) -> YOLOSampleBatch:
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is not None:
        worker_info.dataset.imgsz = random.choice(img_sizes)  # type: ignore
    return collate_fn(samples)
