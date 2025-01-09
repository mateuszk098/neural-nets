import os
from collections import namedtuple
from os import PathLike
from pathlib import Path
from typing import Literal, Self
from xml.etree import ElementTree

os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"

import albumentations as A
import cv2 as cv
import torch
from torch.types import Tensor
from torch.utils.data import Dataset

Detection = namedtuple("Detection", ("label", "bbox"))
ImageMeta = namedtuple("ImageMeta", ("path", "width", "height", "detections"))
YOLOSample = namedtuple("YOLOSample", ("image", "bboxes", "labels", "target"))


def collate_fn(samples: list[YOLOSample]) -> tuple[Tensor, Tensor, list[Tensor], list[Tensor]]:
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

    return batch_images, batch_targets, batch_bboxes, batch_labels


class VOC2012Dataset(Dataset):
    def __init__(
        self,
        root: str | PathLike,
        /,
        *,
        imgsz: int,
        S: int,
        B: int,
        split: Literal["train", "val"],
        normalize: bool = True,
    ) -> None:
        dataset, classes = self.load(root, split)

        self.dataset = dataset
        self.class_to_idx = {cls: idx for idx, cls in enumerate(sorted(classes))}
        self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}

        self.imgsz = int(imgsz)
        self.S = int(S)
        self.B = int(B)
        self.C = len(self.class_to_idx)
        self.normalize = bool(normalize)

        self._train_transform = A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.Affine(
                    scale=(0.8, 1.2),
                    translate_percent=(-0.2, 0.2),
                    border_mode=cv.BORDER_REPLICATE,
                    always_apply=True,
                ),
                A.ColorJitter(
                    brightness=(0.8, 1.2),
                    contrast=(0.8, 1.2),
                    saturation=(0.8, 1.2),
                    hue=(-0.2, 0.2),
                    always_apply=True,
                ),
                A.Resize(self.imgsz, self.imgsz, always_apply=True),
            ],
            bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"]),
            seed=42,
        )

        self._eval_transform = A.Compose(
            [
                A.Resize(self.imgsz, self.imgsz, always_apply=True),
            ],
            bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"]),
            seed=42,
        )

        if self.normalize:  # Useful when using non-normalized images for testing.
            self._train_transform.transforms.append(A.Normalize(always_apply=True))
            self._eval_transform.transforms.append(A.Normalize(always_apply=True))

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

        if bboxes.any():
            target = self._create_yolo_target(bboxes, labels)
        else:
            target = torch.zeros((self.S, self.S, self.B * 5 + self.C))

        return YOLOSample(image, bboxes, labels, target)

    def _create_yolo_target(self, bboxes: Tensor, labels: Tensor) -> Tensor:
        target = torch.zeros((self.S, self.S, self.B * 5 + self.C))
        cell_size = self.imgsz // self.S

        bboxes_xywh = torch.zeros_like(bboxes)
        bboxes_xywh[:, :2] = bboxes[:, :2] + 0.5 * (bboxes[:, 2:] - bboxes[:, :2])
        bboxes_xywh[:, 2:] = bboxes[:, 2:] - bboxes[:, :2]

        # Get the cell indices for each bbox.
        # Transpose because, when unpacking there must be a tensor of shape (2, N),
        # where N is the number of bboxes. So, ns is a tensor of shape (N,) - the cell column index
        # for each bbox and ms is a tensor of shape (N,) - the cell row index for each bbox.
        ns, ms = (bboxes_xywh[:, :2] // cell_size).int().t()
        # Calculate offset of bbox center from cell top-left corner and normalize it.
        offsets = (bboxes_xywh[:, :2] % cell_size) / cell_size
        # Normalize bbox width and height by image size.
        whs = bboxes_xywh[:, 2:] / self.imgsz

        # Target on the grid cell is [[offset_x, offset_y, sqrt(w), sqrt(h), confidence] * B, class_1, class_2, ...]
        for idx, (n, m, offset, wh) in enumerate(zip(ns, ms, offsets, whs)):
            for k in range(self.B):
                target[m, n, k * 5 : k * 5 + 2] = offset
                target[m, n, k * 5 + 2 : k * 5 + 4] = wh.sqrt()
                target[m, n, k * 5 + 4] = 1.0
            # One-hot encode the class label.
            target[m, n, self.B * 5 + labels[idx]] = 1.0

        return target

    def train(self) -> Self:
        self.transform = self._train_transform
        return self

    def eval(self) -> Self:
        self.transform = self._eval_transform
        return self

    @staticmethod
    def load(root: str | PathLike, split: Literal["train", "val"]) -> tuple[list[ImageMeta], set[str]]:
        data_root = Path(root).expanduser()
        image_set = data_root.joinpath("ImageSets", "Main", f"{split}.txt")

        dataset = list()
        classes = set()

        with open(image_set) as f:
            for name in (name.strip() for name in f):
                image_path = data_root.joinpath("JPEGImages", f"{name}.jpg")
                annot_path = data_root.joinpath("Annotations", f"{name}.xml")

                annot_tree = ElementTree.parse(annot_path)
                annot_root = annot_tree.getroot()

                detections = list()
                # Following elements are always present, so don't mind about the type ignore.
                for obj in annot_root.findall("object"):
                    if obj.find("difficult").text == "1":  # type: ignore
                        continue

                    label = obj.find("name").text  # type: ignore
                    bndbox = obj.find("bndbox")  # type: ignore
                    bbox = (
                        int(bndbox.find("xmin").text),  # type: ignore
                        int(bndbox.find("ymin").text),  # type: ignore
                        int(bndbox.find("xmax").text),  # type: ignore
                        int(bndbox.find("ymax").text),  # type: ignore
                    )
                    detections.append(Detection(label, bbox))
                    classes.add(label)

                if detections:
                    width = int(annot_root.find("size").find("width").text)  # type: ignore
                    height = int(annot_root.find("size").find("height").text)  # type: ignore
                    dataset.append(ImageMeta(image_path, width, height, detections))

        return dataset, classes
