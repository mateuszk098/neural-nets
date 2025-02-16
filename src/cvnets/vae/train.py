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

from cvnets.vae.dataset import COCOKeypointsDataset
from cvnets.vae.loss import VAELoss
from cvnets.vae.net import VAENet
from cvnets.vae.utils import initialize_seed, load_yaml, worker_init_fn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FORMAT = "[%(asctime)s - %(module)s/%(levelname)s]: %(message)s"
DATE_FMT = "%Y-%m-%d %H:%M:%S"

initialize_seed()
torch.backends.cudnn.benchmark = True
logging.basicConfig(level=logging.INFO, format=FORMAT, datefmt=DATE_FMT)


def train_step(
    *, model: nn.Module, loader: DataLoader, loss_fn: VAELoss, optim: Optimizer, scheduler: LRScheduler
) -> float:
    model.train()
    loss_fn.train()
    partial_loss = torch.tensor(0.0)

    for batch in loader:
        origin_kpts = batch["origin_kpts"].to(DEVICE)
        masked_kpts = batch["masked_kpts"].to(DEVICE)

        pred_kpts, mu, logvar = model(masked_kpts)

        loss = loss_fn(pred_kpts, origin_kpts, masked_kpts, mu, logvar)
        loss.backward()

        optim.step()
        scheduler.step()
        optim.zero_grad()

        partial_loss += torch.as_tensor(loss).detach().cpu()

    loss_fn.step()

    return partial_loss.div(len(loader)).item()


def valid_step(*, model: nn.Module, loader: DataLoader, loss_fn: nn.Module) -> float:
    model.eval()
    loss_fn.eval()
    partial_loss = torch.tensor(0.0)

    with torch.inference_mode():
        for batch in loader:
            origin_kpts = batch["origin_kpts"].to(DEVICE)
            masked_kpts = batch["masked_kpts"].to(DEVICE)

            pred_kpts = model(masked_kpts)
            loss = loss_fn(pred_kpts, origin_kpts, masked_kpts)

            partial_loss += torch.as_tensor(loss).detach().cpu()

    return partial_loss.div(len(loader)).item()


def main(*, config_file: str | PathLike) -> None:
    logging.info(f"Loading configuration from {config_file!s}...")
    config = SimpleNamespace(**load_yaml(config_file))
    for k, v in config.__dict__.items():
        logging.info(f"{k + ':':<25} {v}")

    train_dataset = COCOKeypointsDataset(
        config.TRAIN_DATASET,
        min_keypoints=config.MIN_KEYPOINTS,
        max_kpts_to_mask=config.MAX_KPTS_TO_MASK,
    )
    valid_dataset = COCOKeypointsDataset(
        config.VALID_DATASET,
        min_keypoints=config.MIN_KEYPOINTS,
        max_kpts_to_mask=config.MAX_KPTS_TO_MASK,
    )

    logging.info(f"{'TRAIN DATASET LENGTH:':<25} {len(train_dataset)}")
    logging.info(f"{'VALID DATASET LENGTH:':<25} {len(valid_dataset)}")

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        worker_init_fn=worker_init_fn,
        persistent_workers=config.PERSISTENT_WORKERS,
        pin_memory=config.PIN_MEMORY,
    )
    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        worker_init_fn=worker_init_fn,
        persistent_workers=config.PERSISTENT_WORKERS,
        pin_memory=config.PIN_MEMORY,
    )

    model = VAENet(input_shape=(14, 2), latent_features=16, hidden_units=[32, 64, 96])
    model = model.to(DEVICE)

    loss_fn = VAELoss(
        reconst_alpha=config.RECONST_ALPHA,
        kl_alpha=config.KL_ALPHA,
        kl_alpha_multiplier=config.KL_ALPHA_MULTIPLIER,
        latent_alpha=config.LATENT_ALPHA,
        latent_alpha_multiplier=config.LATENT_ALPHA_MULTIPLIER,
    )
    optim = torch.optim.NAdam(
        params=model.parameters(),
        weight_decay=config.WEIGHT_DECAY,
        decoupled_weight_decay=config.DECOUPLED_WEIGHT_DECAY,
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
    log = "Epoch: {:3d}/{} | Time: {:6.2f} s | Train Loss: {:7.4f} | Valid Loss: {:7.4f}"
    best_loss = float("inf")

    for epoch in range(1, config.EPOCHS + 1):
        t0 = time.perf_counter()
        train_loss = train_step(model=model, loader=train_loader, loss_fn=loss_fn, optim=optim, scheduler=scheduler)
        valid_loss = valid_step(model=model, loader=valid_loader, loss_fn=loss_fn)
        t1 = time.perf_counter()
        logging.info(log.format(epoch, config.EPOCHS, t1 - t0, train_loss, valid_loss))

        if valid_loss < best_loss:
            best_loss = valid_loss
            torch.save(model.state_dict(), "vae.pt")


if __name__ == "__main__":
    p = ArgumentParser()
    p.add_argument("--config-file", type=Path, required=True)
    kwargs = vars(p.parse_args())
    main(**kwargs)
