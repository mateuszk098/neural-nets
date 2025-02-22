import logging
from argparse import ArgumentParser
from os import PathLike
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchmetrics.classification import BinaryJaccardIndex

from cvnets.unet.dataset import ISICDataset
from cvnets.unet.loss import ComboLoss, NamedLoss
from cvnets.unet.network import UNet
from cvnets.unet.utils import initialize_seed, load_yaml

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FORMAT = "[%(asctime)s - %(module)s/%(levelname)s]: %(message)s"
DATE_FMT = "%Y-%m-%d %H:%M:%S"

initialize_seed()
logging.basicConfig(level=logging.INFO, format=FORMAT, datefmt=DATE_FMT)


def eval_step(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: ComboLoss,
    metric_fn: BinaryJaccardIndex,
) -> tuple[NamedLoss, float]:
    accumulated_loss = torch.zeros(len(NamedLoss._fields))

    with torch.inference_mode():
        for step, (images, masks) in enumerate(loader, start=1):
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            logits = model(images)
            loss = loss_fn(logits, masks)
            accumulated_loss += torch.as_tensor(loss).detach().cpu()
            metric_fn.update(logits, masks)

    accumulated_loss = accumulated_loss.div(len(loader)).tolist()
    accumulated_loss = NamedLoss(*accumulated_loss)
    metric = metric_fn.compute().item()

    return accumulated_loss, metric


def main(*, config_file: str | PathLike) -> None:
    logging.info(f"Loading configuration from {config_file!s}...")
    config = SimpleNamespace(**load_yaml(config_file))

    dataset = ISICDataset(root=config.TEST_DIR, imgsz=config.IMGSZ)
    dataset.eval()

    loader = DataLoader(dataset=dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    model = UNet(in_channels=3, out_channels=1, base=config.NET_BASE)
    model.load_state_dict(torch.load(config.MODEL_FILE, map_location=DEVICE, weights_only=True))
    model.to(DEVICE)
    model.eval()

    loss_fn = ComboLoss(focal_alpha=config.FOCAL_ALPHA, focal_gamma=config.FOCAL_GAMMA, beta=config.LOSS_BETA)
    metric_fn = BinaryJaccardIndex().to(DEVICE)

    logging.info(f"Start evaluation on {torch.cuda.get_device_name(DEVICE)}...")
    log = "Total Loss: {:6.4f} | Focal Loss: {:6.4f} | Dice Loss: {:6.4f} | Jaccard Index: {:6.4f}"

    loss, metric = eval_step(model, loader, loss_fn, metric_fn)
    logging.info(log.format(*loss, metric))


if __name__ == "__main__":
    p = ArgumentParser()
    p.add_argument("--config-file", type=Path, required=True)
    kwargs = vars(p.parse_args())
    main(**kwargs)
