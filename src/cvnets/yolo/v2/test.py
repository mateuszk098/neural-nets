import logging
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import cv2
from torch.utils.data import DataLoader

from cvnets.yolo.utils import initialize_seed, load_yaml, worker_init_fn
from cvnets.yolo.v2.dataset import VOCDataset, collate_fn, sampled_collate_fn
from cvnets.yolo.v2.utils import load_anchor_bboxes

initialize_seed()

FORMAT = "[%(asctime)s - %(module)s/%(levelname)s]: %(message)s"
DATE_FMT = "%Y-%m-%d %H:%M:%S"

logging.basicConfig(level=logging.INFO, format=FORMAT, datefmt=DATE_FMT)


if __name__ == "__main__":
    config_file = "config.yaml"
    config = SimpleNamespace(**load_yaml(config_file))
    anchor_bboxes = load_anchor_bboxes(config.ANCHOR_BBOXES)
    img_sizes = [320, 352, 384, 416, 448, 480, 512, 544, 576, 608]

    train_dataset = VOCDataset(
        config.DATASET,
        anchors=anchor_bboxes,
        downsample=config.DOWNSAMPLE,
        split="train",
        normalize=False,
    )

    eval_dataset = VOCDataset(
        config.DATASET,
        anchors=anchor_bboxes,
        downsample=config.DOWNSAMPLE,
        split="train",
        normalize=False,
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=False,
        collate_fn=partial(sampled_collate_fn, img_sizes=img_sizes),
        num_workers=4,
        persistent_workers=config.PERSISTENT_WORKERS,
        pin_memory=config.PIN_MEMORY,
        worker_init_fn=worker_init_fn,
    )

    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=32,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
        persistent_workers=config.PERSISTENT_WORKERS,
        pin_memory=config.PIN_MEMORY,
        worker_init_fn=worker_init_fn,
    )

    train_dataloader.dataset.train()  # type: ignore
    eval_dataloader.dataset.eval()  # type: ignore

    print("Here we go!")

    train1_dir = Path("train1")
    train1_dir.mkdir(exist_ok=True)

    for k, batch in enumerate(train_dataloader, start=1):
        print(batch.images.shape)  # type: ignore
        image = batch.images[0].permute(1, 2, 0).numpy().astype("uint8")
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(train1_dir / f"{k:03d}.jpg"), image)

    eval1_dir = Path("eval1")
    eval1_dir.mkdir(exist_ok=True)

    print(eval_dataloader.dataset.transform)  # type: ignore
    for k, batch in enumerate(eval_dataloader, start=1):
        print(batch.images.shape)  # type: ignore
        image = batch.images[0].permute(1, 2, 0).numpy().astype("uint8")
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(eval1_dir / f"{k:03d}.jpg"), image)

    train2_dir = Path("train2")
    train2_dir.mkdir(exist_ok=True)

    for k, batch in enumerate(train_dataloader, start=1):
        print(batch.images.shape)  # type: ignore
        image = batch.images[0].permute(1, 2, 0).numpy().astype("uint8")
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(train2_dir / f"{k:03d}.jpg"), image)
