import logging
import time
from argparse import ArgumentParser
from collections import namedtuple
from os import PathLike
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from torchmetrics.classification import BinaryJaccardIndex

from cvnets.unet.dataset import ISICDataset
from cvnets.unet.loss import ComboLoss, NamedLoss
from cvnets.unet.network import ResUNetPP
from cvnets.unet.utils import DeNormalizer, EarlyStopping, initialize_seed, load_yaml, worker_init_fn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FORMAT = "[%(asctime)s - %(module)s/%(levelname)s]: %(message)s"
DATE_FMT = "%Y-%m-%d %H:%M:%S"

initialize_seed()
torch.backends.cudnn.benchmark = True
logging.basicConfig(level=logging.INFO, format=FORMAT, datefmt=DATE_FMT)

SummaryImages = namedtuple("SummaryImages", ("images", "true_masks", "pred_masks"))


def train_step(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: ComboLoss,
    optimizer: Optimizer,
    denormalizer: DeNormalizer,
) -> tuple[NamedLoss, SummaryImages]:
    model.train()
    accumulated_loss = torch.zeros(len(NamedLoss._fields))
    summary_images = SummaryImages(None, None, None)

    for step, (images, masks) in enumerate(loader, start=1):
        images, masks = images.to(DEVICE), masks.to(DEVICE)
        logits = model(images)
        loss = loss_fn(logits, masks)
        loss.total.backward()
        optimizer.step()
        optimizer.zero_grad()
        accumulated_loss += torch.as_tensor(loss).detach().cpu()

        if step == len(loader):
            images_denorm = denormalizer(images)
            pred_masks = torch.sigmoid(logits)
            summary_images = SummaryImages(images_denorm, masks, pred_masks)

    accumulated_loss = accumulated_loss.div(len(loader)).tolist()
    accumulated_loss = NamedLoss(*accumulated_loss)

    return accumulated_loss, summary_images


def eval_step(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: ComboLoss,
    metric_fn: BinaryJaccardIndex,
    denormalizer: DeNormalizer,
) -> tuple[NamedLoss, SummaryImages, float]:
    model.eval()
    metric_fn.reset()
    accumulated_loss = torch.zeros(len(NamedLoss._fields))
    summary_images = SummaryImages(None, None, None)

    with torch.inference_mode():
        for step, (images, masks) in enumerate(loader, start=1):
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            logits = model(images)
            loss = loss_fn(logits, masks)
            accumulated_loss += torch.as_tensor(loss).detach().cpu()
            metric_fn.update(logits, masks)

            if step == len(loader):
                images_denorm = denormalizer(images)
                pred_masks = torch.sigmoid(logits)
                summary_images = SummaryImages(images_denorm, masks, pred_masks)

    accumulated_loss = accumulated_loss.div(len(loader)).tolist()
    accumulated_loss = NamedLoss(*accumulated_loss)
    metric = metric_fn.compute().item()

    return accumulated_loss, summary_images, metric


def main(*, config_file: str | PathLike) -> None:
    logging.info(f"Loading configuration from {config_file!s}...")
    config = SimpleNamespace(**load_yaml(config_file))
    for k, v in config.__dict__.items():
        logging.info(f"{k + ':':<25} {v}")

    train_dataset = ISICDataset(root=config.TRAIN_DIR, imgsz=config.IMGSZ)
    train_dataset.train()

    eval_dataset = ISICDataset(root=config.EVAL_DIR, imgsz=config.IMGSZ)
    eval_dataset.eval()

    logging.info(f"{'TRAIN DATASET LENGTH:':<25} {len(train_dataset)}")
    logging.info(f"{'EVAL DATASET LENGTH:':<25} {len(eval_dataset)}")

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        worker_init_fn=worker_init_fn,
        persistent_workers=config.PERSISTENT_WORKERS,
    )

    eval_loader = DataLoader(
        dataset=eval_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,  # Shuffle for visualization purposes.
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        worker_init_fn=worker_init_fn,
        persistent_workers=config.PERSISTENT_WORKERS,
    )

    model = ResUNetPP(in_channels=3, out_channels=1, base=config.NET_BASE).to(DEVICE)
    loss_fn = ComboLoss(
        focal_alpha=config.FOCAL_ALPHA,
        focal_gamma=config.FOCAL_GAMMA,
        beta=config.LOSS_BETA,
    )
    metric_fn = BinaryJaccardIndex().to(DEVICE)
    early_stopping = EarlyStopping(
        patience=config.EARLY_STOPPING_PATIENCE,
        min_delta=config.EARLY_STOPPING_DELTA,
    )
    optimizer = torch.optim.NAdam(
        model.parameters(),
        lr=config.LR,
        weight_decay=config.WEIGHT_DECAY,
        decoupled_weight_decay=config.DECOUPLED_WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer,
        factor=config.LR_FACTOR,
        patience=config.LR_PATIENCE,
        threshold=config.LR_THRESHOLD,
    )
    denormalizer = DeNormalizer()
    writer = SummaryWriter()

    logging.info(f"Start training on {torch.cuda.get_device_name(DEVICE)}...")
    log = (
        "Epoch: {:3d}/{} | Time: {:6.2f} s | Total Loss: {:6.4f} | Focal Loss: {:6.4f} | Dice Loss: {:6.4f} | "
        "Eval Total Loss: {:6.4f} | Eval Focal Loss: {:6.4f} | Eval Dice Loss: {:6.4f} | Eval Jaccard Index: {:6.4f}"
    )
    current_run_dir = Path(writer.get_logdir())

    for epoch in range(1, config.EPOCHS + 1):
        t0 = time.perf_counter()
        train_loss, train_images = train_step(model, train_loader, loss_fn, optimizer, denormalizer)
        eval_loss, eval_images, eval_metric = eval_step(model, eval_loader, loss_fn, metric_fn, denormalizer)
        t1 = time.perf_counter()

        logging.info(log.format(epoch, config.EPOCHS, t1 - t0, *train_loss, *eval_loss, eval_metric))

        torch.save(model.state_dict(), current_run_dir.joinpath(model.__class__.__name__).with_suffix(".pt"))

        if config.VISUALIZE:
            writer.add_scalars("Train Loss", train_loss._asdict(), epoch)
            writer.add_scalars("Eval Loss", eval_loss._asdict(), epoch)
            writer.add_scalar("Eval Jaccard Index", eval_metric, epoch)
            writer.add_images("Train Images", train_images.images, epoch)
            writer.add_images("Train True Masks", train_images.true_masks, epoch)
            writer.add_images("Train Pred Masks", train_images.pred_masks, epoch)
            writer.add_images("Eval Images", eval_images.images, epoch)
            writer.add_images("Eval True Masks", eval_images.true_masks, epoch)
            writer.add_images("Eval Pred Masks", eval_images.pred_masks, epoch)

        if early_stopping(eval_loss.total):
            logging.info("Early stopping...")
            break

        scheduler.step(eval_loss.total)


if __name__ == "__main__":
    p = ArgumentParser()
    p.add_argument("--config-file", type=Path, required=True)
    kwargs = vars(p.parse_args())
    main(**kwargs)
