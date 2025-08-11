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
class COCOKeypoints:
    LINKS: tuple[tuple[int, int], ...] = (
        (16, 14),
        (14, 12),
        (17, 15),
        (15, 13),
        (12, 13),
        (6, 12),
        (7, 13),
        (6, 7),
        (6, 8),
        (7, 9),
        (8, 10),
        (9, 11),
        (2, 3),
        (1, 2),
        (1, 3),
        (2, 4),
        (3, 5),
        (4, 6),
        (5, 7),
    )
    NAMES: tuple[str, ...] = (
        "nose",
        "left_eye",
        "right_eye",
        "left_ear",
        "right_ear",
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


@dataclass(init=False, frozen=True)
class IntermediateKeypoints:
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
    def __init__(self, root: str | PathLike, min_keypoints: int, with_image: bool = False) -> None:
        coco_headlike_names = ("nose", "left_eye", "right_eye", "left_ear", "right_ear")
        coco_torso_names = ("left_shoulder", "right_shoulder", "left_hip", "right_hip")

        self._headlike_ids = list(COCOKeypoints.NAMES.index(k) for k in coco_headlike_names)
        self._torso_ids = list(COCOKeypoints.NAMES.index(k) for k in coco_torso_names)

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

            keypoints = np.asarray(annotation["keypoints"]).reshape(-1, 3)
            if not self._check_keypoints(keypoints):
                continue

            keypoints = self._parse_keypoints(keypoints)
            bbox_xywh = self._extend_bbox(np.asarray(annotation["bbox"]), keypoints)

            image_file = ""
            if self.with_image:
                for image in data["images"]:
                    if image["id"] == annotation["image_id"]:
                        image_file = image["file_name"]
                        break

            dataset.append(
                {
                    "keypoints": keypoints[:, :2],
                    "visibility": keypoints[:, 2],
                    "bbox_xywh": bbox_xywh,
                    "image_file": image_file,
                }
            )

        return dataset

    def _check_keypoints(self, keypoints: NDArray) -> bool:
        # Only one headlike keypoint is enough to simulate head presence.
        head_check = keypoints[self._headlike_ids, 2].any().item()
        # Suppose all torso keypoints must be visible to provide fine-enough person representation.
        torso_check = keypoints[self._torso_ids, 2].all().item()
        return head_check and torso_check

    def _parse_keypoints(self, keypoints: NDArray) -> NDArray:
        kpts = np.zeros((14, 3))
        kpts[2:] = keypoints[5:17]

        # Some headlike keypoints may be zero, so mask them out to not affect the mean.
        headlike_mask = np.equal(keypoints[:5, 2], 0).reshape(-1, 1)
        headlike_mask = np.repeat(headlike_mask, 3, axis=1)
        kpts[0] = np.ma.masked_array(keypoints[:5], mask=headlike_mask).mean(axis=0).data.round()

        # Shoulders always exists because of the torso check.
        kpts[1] = np.mean(keypoints[5:7], axis=0).round()

        return kpts.astype(np.float32)

    def _extend_bbox(self, bbox: NDArray, keypoints: NDArray) -> NDArray:
        x1, y1 = bbox[:2]
        x2, y2 = bbox[:2] + bbox[2:]

        # We cannot take non-visible keypoints into account.
        mask = np.equal(keypoints[:, 2], 0).reshape(-1, 1)
        mask = np.repeat(mask, repeats=3, axis=1)
        masked = np.ma.masked_array(keypoints, mask=mask)

        x_min, y_min = masked[:, :2].min(axis=0)
        x_max, y_max = masked[:, :2].max(axis=0)

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
        normalize: bool = True,
        with_image: bool = False,
    ) -> None:
        self.parser = COCOKeypointsParser(root, min_keypoints, with_image)
        self.max_kpts_to_mask = int(max_kpts_to_mask)
        self.gen = np.random.Generator(np.random.MT19937(seed))
        self.normalize = bool(normalize)

    def __len__(self) -> int:
        return len(self.parser)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        origin_kpts = self.parser[idx]["keypoints"]
        origin_visb = self.parser[idx]["visibility"]
        bbox_xywh = self.parser[idx]["bbox_xywh"]
        image_file = self.parser[idx]["image_file"]

        if self.normalize:
            mask = np.equal(origin_visb, 0).reshape(-1, 1)
            origin_kpts = np.where(mask, origin_kpts, (origin_kpts - bbox_xywh[:2]) / bbox_xywh[2:])

        num_kpts_to_mask = self.gen.integers(1, self.max_kpts_to_mask, endpoint=True)
        kpts_to_mask_out = self.gen.choice(range(len(origin_kpts)), num_kpts_to_mask, replace=False)

        masked_kpts = origin_kpts.copy()
        masked_kpts[kpts_to_mask_out] = 0

        masked_visb = origin_visb.copy()
        masked_visb[kpts_to_mask_out] = 0

        return {
            "origin_kpts": origin_kpts,
            "origin_visb": origin_visb,
            "masked_kpts": masked_kpts,
            "masked_visb": masked_visb,
            "bbox_xywh": bbox_xywh,
            "image_file": image_file,
        }

    def set_seed(self, seed: int) -> None:
        self.gen = np.random.Generator(np.random.MT19937(seed))
