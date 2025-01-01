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

from cvnets.yolov1.loss import NamedLoss, YOLOv1Loss
from cvnets.yolov1.net import YOLOv1
from cvnets.yolov1.utils import dfs_freeze, dfs_unfreeze, load_yaml
from cvnets.yolov1.voc import VOC2012Dataset, collate_fn

SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FORMAT = "[%(asctime)s - %(module)s - %(levelname)s]: %(message)s"
DATE_FMT = "%Y-%m-%d %H:%M:%S"

torch.backends.cudnn.benchmark = True
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

logging.basicConfig(level=logging.INFO, format=FORMAT, datefmt=DATE_FMT)


def train_step(
    model: nn.Module, loader: DataLoader, loss: nn.Module, optimizer: Optimizer, scheduler: LRScheduler
) -> NamedLoss:
    model.train()
    loader.dataset.train()  # type: ignore
    partial_loss = torch.zeros(5)

    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)

        model_loss = loss(model(x), y)
        optimizer.zero_grad()
        model_loss.total.backward()

        optimizer.step()
        scheduler.step()

        partial_loss += torch.as_tensor(
            (
                model_loss.total,
                model_loss.coord,
                model_loss.itobj,
                model_loss.noobj,
                model_loss.label,
            )
        )

    return NamedLoss(*partial_loss.cpu().detach().div(len(loader)).tolist())


def valid_step(model: nn.Module, loader: DataLoader, loss: nn.Module) -> NamedLoss:
    model.eval()
    loader.dataset.eval()  # type: ignore
    partial_loss = torch.zeros(5)

    with torch.inference_mode():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            model_loss = loss(model(x), y)
            partial_loss += torch.as_tensor(
                (
                    model_loss.total,
                    model_loss.coord,
                    model_loss.itobj,
                    model_loss.noobj,
                    model_loss.label,
                )
            )

    return NamedLoss(*partial_loss.cpu().detach().div(len(loader)).tolist())


def main(*, config_file: str | PathLike) -> None:
    logging.info(f"Loading configuration from {config_file!s}...")
    config = SimpleNamespace(**load_yaml(config_file))

    train_dataset = VOC2012Dataset(
        config.DATASET, imgsz=config.IMGSZ, S=config.S, B=config.B, split="train", normalize=True
    )
    valid_dataset = VOC2012Dataset(
        config.DATASET, imgsz=config.IMGSZ, S=config.S, B=config.B, split="val", normalize=True
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

    model = YOLOv1(imgsz=config.IMGSZ, S=config.S, B=config.B, C=config.C)
    model = model.to(DEVICE)
    dfs_freeze(model.backbone)  # Freeze the backbone to stabilize the head and neck first.

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

    log = (
        "Epoch: {:3d}/{} | {} Time: {:7.2f} s | Total Loss: {:7.4f} | "
        "Localization Loss: {:7.4f} | Objectness Loss: {:7.4f} | Noobjectness Loss: {:7.4f} | "
        "Classification Loss: {:7.4f}"
    )

    checkpoints = Path("./checkpoints/")
    checkpoints.mkdir(exist_ok=True)

    for epoch in range(1, config.EPOCHS + 1):
        t0 = time.perf_counter()
        train_loss = train_step(model, train_loader, loss, optimizer, scheduler)
        valid_loss = valid_step(model, valid_loader, loss)
        t1 = time.perf_counter()
        logging.info(
            log.format(
                epoch,
                "Train",
                config.EPOCHS,
                t1 - t0,
                train_loss.total,
                train_loss.coord,
                train_loss.itobj,
                train_loss.noobj,
                train_loss.label,
            )
        )
        logging.info(
            log.format(
                epoch,
                "Valid",
                config.EPOCHS,
                t1 - t0,
                valid_loss.total,
                valid_loss.coord,
                valid_loss.itobj,
                valid_loss.noobj,
                valid_loss.label,
            )
        )

        torch.save(model.state_dict(), checkpoints.joinpath("yolov1-voc2012.pt"))

        if config.UNFREEZE_BACKBONE and epoch == config.EPOCH_TO_UNFREEZE_BACKBONE:
            logging.info("Unfreezing the backbone...")
            dfs_unfreeze(model.backbone)
            # Maybe two different optimizers and schedulers for the backbone and detector?
            optimizer.add_param_group(
                {
                    "params": model.backbone.parameters(),
                    **{key: value for key, value in optimizer.param_groups[0].items() if key != "params"},
                }
            )


if __name__ == "__main__":
    p = ArgumentParser()
    p.add_argument("--config-file", type=Path, required=True)
    kwargs = vars(p.parse_args())
    main(**kwargs)
