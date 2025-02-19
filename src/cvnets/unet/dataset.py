from os import PathLike
from pathlib import Path

import albumentations as A
import albumentations.pytorch as AP
import cv2 as cv
import numpy as np
from torch.types import Tensor
from torch.utils.data import Dataset


class ISICDataset(Dataset):
    def __init__(
        self,
        root: str | PathLike,
        imgsz: int,
        normalize: bool = True,
        as_tensor: bool = True,
        seed: int = 42,
    ) -> None:
        self.data = self._load_isic_data(root)

        self._train_transform = A.Compose(
            transforms=[
                A.Resize(imgsz, imgsz),
                A.ColorJitter(
                    brightness=(0.8, 1.2),
                    contrast=(0.8, 1.2),
                    saturation=(0.8, 1.2),
                    hue=(-0.1, 0.1),
                    p=0.5,
                ),
                A.Blur(blur_limit=(3, 11), p=0.1),
                A.GaussNoise(std_range=(0.1, 0.2), p=0.1),
                A.ToGray(p=0.1),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.ElasticTransform(alpha=100, sigma=100, p=0.5),
                A.Affine(
                    scale=(0.8, 1.2),
                    translate_percent=(-0.1, 0.1),
                    rotate=(-10, 10),
                    border_mode=cv.BORDER_CONSTANT,
                    p=0.5,
                ),
            ],
            seed=seed,
        )

        self._valid_transform = A.Compose(
            transforms=[
                A.Resize(imgsz, imgsz),
            ],
            seed=seed,
        )

        if bool(normalize):
            self._train_transform.transforms.append(A.Normalize())
            self._valid_transform.transforms.append(A.Normalize())

        if bool(as_tensor):
            self._train_transform.transforms.append(AP.ToTensorV2(transpose_mask=True))
            self._valid_transform.transforms.append(AP.ToTensorV2(transpose_mask=True))

        self.train()

    def train(self) -> None:
        self.transform = self._train_transform

    def eval(self) -> None:
        self.transform = self._valid_transform

    def set_seed(self, seed: int) -> None:
        self.transform.set_random_seed(seed)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx) -> tuple[Tensor, Tensor]:
        image_fle = self.data[idx]["image_file"]
        mask_file = self.data[idx]["mask_file"]

        image = cv.imread(str(image_fle), cv.IMREAD_COLOR)
        mask = cv.imread(str(mask_file), cv.IMREAD_GRAYSCALE)

        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        mask = np.expand_dims(mask, axis=-1)

        augmented = self.transform(image=image, mask=mask)

        return augmented["image"], augmented["mask"]

    def _load_isic_data(self, root: str | PathLike) -> list[dict[str, Path]]:
        root = Path(root).expanduser()
        images_root = root.joinpath("images")
        masks_root = root.joinpath("masks")

        data = list()
        for image_file in images_root.iterdir():
            data.append(
                {
                    "image_file": image_file,
                    "mask_file": masks_root.joinpath(image_file.stem + "_Segmentation.png"),
                }
            )

        return data
