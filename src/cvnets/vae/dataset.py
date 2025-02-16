import json
from dataclasses import dataclass
from itertools import chain
from os import PathLike
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from torch.utils.data import Dataset


@dataclass(init=False, frozen=True)
class NamedKeypoints:
    LINKS: tuple[tuple[int, int], ...] = (
        (13, 11),
        (11, 9),
        (14, 12),
        (12, 10),
        (9, 10),
        (3, 9),
        (4, 10),
        (3, 5),
        (4, 6),
        (5, 7),
        (6, 8),
        (2, 3),
        (1, 2),
        (2, 4),
    )
    NAMES: tuple[str, ...] = (
        "head",
        "neck",
        "left_shoulder",
        "right_shoulder",
        "left_elbow",
        "right_elbow",
        "left_wrist",
        "right_wrist",
        "left_hip",
        "right_hip",
        "left_knee",
        "right_knee",
        "left_ankle",
        "right_ankle",
    )
    NAMES_XY: tuple[str, ...] = tuple(chain.from_iterable(f"{n}_x {n}_y".split() for n in NAMES))
    NAMES_XYC: tuple[str, ...] = tuple(chain.from_iterable(f"{n}_x {n}_y {n}_c".split() for n in NAMES))


class COCOKeypointsParser:
    def __init__(self, root: str | PathLike, min_keypoints: int = 17, with_image: bool = False) -> None:
        self.min_keypoints = int(min_keypoints)
        self.with_image = bool(with_image)

        self.alldata = self._load_coco_keypoints(root)
        self.dataset = self._parse_coco_keypoints(self.alldata)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.dataset[idx]

    def _load_coco_keypoints(self, file: str | PathLike) -> dict[str, Any]:
        file = Path(file).expanduser()
        with open(file, "r", encoding="utf-8") as f:
            return json.load(f)

    def _parse_coco_keypoints(self, data: dict[str, Any]) -> list[dict]:
        dataset = list()
        for annotation in data["annotations"]:
            if annotation["num_keypoints"] < self.min_keypoints:
                continue

            image_file = ""
            if self.with_image:
                for image in data["images"]:
                    if image["id"] == annotation["image_id"]:
                        image_file = image["file_name"]
                        break

            keypoints = self._parse_keypoints(annotation["keypoints"])
            bbox_xywh = self._extend_bbox(np.asarray(annotation["bbox"]), keypoints)

            dataset.append(
                {
                    "keypoints": keypoints,
                    "bbox_xywh": bbox_xywh,
                    "image_file": image_file,
                }
            )

        return dataset

    def _parse_keypoints(self, keypoints: list[int]) -> NDArray:
        kpts = np.array(keypoints).reshape(-1, 3)
        joints = np.zeros((14, 3))
        # Some keypoints may be zero, so we need to mask them out to not affect the mean.
        joints[0] = np.ma.masked_where(kpts[:5] == 0, kpts[:5]).mean(axis=0).data.round()
        joints[1] = np.mean(kpts[5:7], axis=0).round()
        joints[2:] = kpts[5:17]
        return joints.astype(np.float32)

    def _extend_bbox(self, bbox: NDArray, keypoints: NDArray) -> NDArray:
        x1, y1 = bbox[:2]
        x2, y2 = bbox[:2] + bbox[2:]

        x_min, y_min = keypoints[:, :2].min(axis=0)
        x_max, y_max = keypoints[:, :2].max(axis=0)

        x1, y1 = min(x1, x_min), min(y1, y_min)
        x2, y2 = max(x2, x_max), max(y2, y_max)

        return np.array((x1, y1, x2 - x1, y2 - y1), dtype=np.float32)


class COCOKeypointsDataset(Dataset):
    def __init__(
        self,
        root: str | PathLike,
        min_keypoints: int = 17,
        max_kpts_to_mask: int = 5,
        seed: int = 42,
        with_visibility: bool = False,
        with_image: bool = False,
    ) -> None:
        self.parser = COCOKeypointsParser(root, min_keypoints, with_image)
        self.max_kpts_to_mask = int(max_kpts_to_mask)
        self.gen = np.random.Generator(np.random.MT19937(seed))
        self.with_visibility = bool(with_visibility)

    def __len__(self) -> int:
        return len(self.parser)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        keypoints = self.parser[idx]["keypoints"]
        bbox_xywh = self.parser[idx]["bbox_xywh"]
        image_file = self.parser[idx]["image_file"]

        normal_keypoints = keypoints.copy()
        normal_keypoints[:, :2] = (keypoints[:, :2] - bbox_xywh[:2]) / bbox_xywh[2:]

        num_kpts_to_mask = self.gen.integers(1, self.max_kpts_to_mask, endpoint=True)
        kpts_to_mask_out = self.gen.choice(range(len(keypoints)), num_kpts_to_mask, replace=False)
        masked_keypoints = normal_keypoints.copy()
        masked_keypoints[kpts_to_mask_out] = 0

        if not self.with_visibility:
            normal_keypoints = normal_keypoints[:, :2]
            masked_keypoints = masked_keypoints[:, :2]

        return {
            "normal_keypoints": normal_keypoints,
            "masked_keypoints": masked_keypoints,
            "bbox_xywh": bbox_xywh,
            "image_file": image_file,
        }

    def set_seed(self, seed: int) -> None:
        self.gen = np.random.Generator(np.random.MT19937(seed))
