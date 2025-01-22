import logging
import math
import time
from argparse import ArgumentParser
from os import PathLike
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader

from cvnets.yolo.utils import load_yaml
from cvnets.yolo.v2.dataset import VOCDataset, collate_fn
from cvnets.yolo.v2.loss import NamedLoss, YOLOv2Loss
from cvnets.yolo.v2.net import YOLOv2
from cvnets.yolo.v2.utils import load_anchor_bboxes

SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FORMAT = "[%(asctime)s - %(module)s/%(levelname)s]: %(message)s"
DATE_FMT = "%Y-%m-%d %H:%M:%S"

torch.backends.cudnn.benchmark = True
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

logging.basicConfig(level=logging.INFO, format=FORMAT, datefmt=DATE_FMT)


def train_step(
    *, model: nn.Module, loader: DataLoader, loss_fn: nn.Module, optim: Optimizer, scheduler: LRScheduler
) -> NamedLoss:
    model.train()
    loader.dataset.train()  # type: ignore
    partial_loss = torch.zeros(5)

    for batch in loader:
        x, y = batch.images.to(DEVICE), batch.targets.to(DEVICE)

        loss = loss_fn(model(x), y)
        loss.total.backward()

        optim.step()
        scheduler.step()
        optim.zero_grad()

        partial_loss += torch.as_tensor(loss)

    return NamedLoss(*partial_loss.cpu().detach().div(len(loader)).tolist())


def valid_step(*, model: nn.Module, loader: DataLoader, loss_fn: nn.Module) -> NamedLoss:
    model.eval()
    loader.dataset.eval()  # type: ignore
    partial_loss = torch.zeros(5)

    with torch.inference_mode():
        for batch in loader:
            x, y = batch.images.to(DEVICE), batch.targets.to(DEVICE)
            loss = loss_fn(model(x), y)
            partial_loss += torch.as_tensor(loss)

    return NamedLoss(*partial_loss.cpu().detach().div(len(loader)).tolist())


def main(*, config_file: str | PathLike) -> None:
    logging.info(f"Loading configuration from {config_file!s}...")
    config = SimpleNamespace(**load_yaml(config_file))

    anchors = load_anchor_bboxes(config.ANCHOR_BBOXES)

    train_dataset = VOCDataset(
        config.DATASET,
        anchors=anchors,
        downsample=config.DOWNSAMPLE,
        imgsz=config.IMGSZ,
        split="train",
    )
    valid_dataset = VOCDataset(
        config.DATASET,
        anchors=anchors,
        downsample=config.DOWNSAMPLE,
        imgsz=config.IMGSZ,
        split="val",
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        generator=torch.manual_seed(SEED),
        num_workers=config.NUM_WORKERS,
        collate_fn=collate_fn,
        persistent_workers=config.PERSISTENT_WORKERS,
        pin_memory=config.PIN_MEMORY,
    )
    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        collate_fn=collate_fn,
        persistent_workers=config.PERSISTENT_WORKERS,
        pin_memory=config.PIN_MEMORY,
    )

    model = YOLOv2(num_anchors=train_dataset.num_anchors, num_classes=train_dataset.num_classes)
    model = model.to(DEVICE)

    loss_fn = YOLOv2Loss(lambda_coord=config.LAMBDA_COORD, lambda_noobj=config.LAMBDA_NOOBJ)
    optim = torch.optim.AdamW(
        params=model.parameters(),
        weight_decay=config.WEIGHT_DECAY,
        amsgrad=True,
        fused=True,
    )
    scheduler_steps = config.EPOCHS * int(math.ceil(len(train_dataset) / config.BATCH_SIZE))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer=optim,
        max_lr=config.MAX_LR,
        div_factor=config.DIV_FACTOR,
        final_div_factor=config.FINAL_DIV_FACTOR,
        three_phase=config.THREE_PHASE,
        total_steps=scheduler_steps,
        anneal_strategy=config.ANNEAL_STRATEGY,
        cycle_momentum=config.CYCLE_MOMENTUM,
        base_momentum=config.BASE_MOMENTUM,
        max_momentum=config.MAX_MOMENTUM,
    )

    logging.info(f"Start training on {torch.cuda.get_device_name(DEVICE)}...")

    log = (
        "Epoch: {:3d}/{} | {} Time: {:7.2f} s | Total Loss: {:7.4f} | "
        "Localization Loss: {:7.4f} | Objectness Loss: {:7.4f} | Noobjectness Loss: {:7.4f} | "
        "Classification Loss: {:7.4f}"
    )

    checkpoints = Path("./checkpoints/")
    checkpoints.mkdir(exist_ok=True)

    logging.info("Training backbone, neck, and head...")

    best_loss = float("inf")

    for epoch in range(1, config.EPOCHS + 1):
        t0 = time.perf_counter()
        train_loss = train_step(
            model=model,
            loader=train_loader,
            loss_fn=loss_fn,
            optim=optim,
            scheduler=scheduler,
        )
        t1 = time.perf_counter()
        logging.info(log.format(epoch, config.EPOCHS, "Train", t1 - t0, *train_loss))

        if config.VALID_STEP:
            t0 = time.perf_counter()
            valid_loss = valid_step(model=model, loader=valid_loader, loss_fn=loss_fn)
            t1 = time.perf_counter()
            logging.info(log.format(epoch, config.EPOCHS, "Valid", t1 - t0, *valid_loss))

        if train_loss.total < best_loss:
            best_loss = train_loss.total
            torch.save(model.state_dict(), checkpoints.joinpath("yolov2-voc.pt"))


if __name__ == "__main__":
    p = ArgumentParser()
    p.add_argument("--config-file", type=Path, required=True)
    kwargs = vars(p.parse_args())
    main(**kwargs)
