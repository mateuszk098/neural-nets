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

from cvnets.yolov1.loss import YOLOv1Loss
from cvnets.yolov1.net import YOLOv1
from cvnets.yolov1.utils import dfs_freeze, dfs_unfreeze, load_yaml
from cvnets.yolov1.voc import VOC2012Dataset, collate_fn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FORMAT = "[%(asctime)s] [%(module)s] [%(levelname)s]: %(message)s"
DATE_FMT = "%Y-%m-%d %H:%M:%S"

logging.basicConfig(level=logging.INFO, format=FORMAT, datefmt=DATE_FMT)


def train_step(
    model: nn.Module, loader: DataLoader, loss: nn.Module, optimizer: Optimizer, scheduler: LRScheduler
) -> float:
    model.train()
    loader.dataset.train()  # type: ignore
    total_loss = torch.tensor(0.0)

    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)

        model_loss = loss(model(x), y)
        optimizer.zero_grad()
        model_loss.backward()

        optimizer.step()
        scheduler.step()

        total_loss += model_loss.cpu().detach()

    return total_loss.item() / len(loader)


def valid_step(model: nn.Module, loader: DataLoader, loss: nn.Module) -> float:
    model.eval()
    loader.dataset.eval()  # type: ignore
    total_loss = torch.tensor(0.0)

    with torch.inference_mode():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)

            model_loss = loss(model(x), y)
            total_loss += model_loss.cpu().detach()

    return total_loss.item() / len(loader)


def main(*, config_file: str | PathLike) -> None:
    logging.info(f"Loading configuration from {config_file!s}...")
    config = SimpleNamespace(**load_yaml(config_file))

    train_dataset = VOC2012Dataset(config.DATASET, imgsz=config.IMGSZ, S=config.S, B=config.B, split="train")
    valid_dataset = VOC2012Dataset(config.DATASET, imgsz=config.IMGSZ, S=config.S, B=config.B, split="val")

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
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

    model = YOLOv1(imgsz=config.IMGSZ, S=config.S, B=config.B, C=config.C).to(DEVICE)
    dfs_freeze(model.backbone)

    optimizer = torch.optim.AdamW(
        params=filter(lambda p: p.requires_grad, model.parameters()),
        amsgrad=True,
        fused=True,
        weight_decay=config.WEIGHT_DECAY,
    )
    loss = YOLOv1Loss(
        S=config.S,
        B=config.B,
        C=config.C,
        lambda_coord=config.LAMBDA_COORD,
        lambda_noobj=config.LAMBDA_NOOBJ,
    )
    scheduler_steps = config.EPOCHS * int(math.ceil(len(train_dataset) / config.BATCH_SIZE))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer=optimizer,
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

    for epoch in range(1, config.EPOCHS + 1):
        t0 = time.perf_counter()
        train_loss = train_step(model, train_loader, loss, optimizer, scheduler)
        torch.save(model.state_dict(), f"yolov1-voc2012-epoch-{epoch:03d}.pt")
        t1 = time.perf_counter()
        logging.info(f"Epoch: {epoch:3d} | Train Time: {(t1-t0):3.2f} s | Train Loss: {train_loss:6.4f}")

        if epoch == 20:
            logging.info("Unfreezing the backbone...")
            dfs_unfreeze(model.backbone)
            optimizer.add_param_group({"backbone_params": model.backbone.parameters()})


if __name__ == "__main__":
    p = ArgumentParser()
    p.add_argument("--config-file", type=Path, required=True)
    kwargs = vars(p.parse_args())
    main(**kwargs)
