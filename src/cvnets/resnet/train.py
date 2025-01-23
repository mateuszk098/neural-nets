import logging

import torch
import torch.nn as nn
import torchmetrics.classification as metrics
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torchmetrics import Metric

from cvnets.resnet.net import ResNet, Structure
from cvnets.resnet.utils import MyImageFolder

SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FORMAT = "[%(asctime)s - %(module)s - %(levelname)s]: %(message)s"
DATE_FMT = "%Y-%m-%d %H:%M:%S"

torch.backends.cudnn.benchmark = True
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

logging.basicConfig(level=logging.INFO, format=FORMAT, datefmt=DATE_FMT)


def train_step(model: nn.Module, loader: DataLoader, loss_fn: nn.Module, optimizer: Optimizer) -> None:
    model.train()
    loader.dataset.train()  # type: ignore

    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()

        loss = loss_fn(model(x), y)
        loss.backward()

        optimizer.step()


def valid_step(model: nn.Module, loader: DataLoader, loss_fn: nn.Module, metric: Metric) -> tuple[float, float]:
    model.eval()
    loader.dataset.eval()  # type: ignore
    metric.reset()

    total_loss = 0.0

    with torch.inference_mode():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)

            y_logit = model(x)
            y_proba = torch.softmax(model(x), dim=-1)

            loss = loss_fn(y_logit, y)
            total_loss += loss.item()

            metric.update(y_proba, y)

    total_loss = total_loss / len(loader)
    metric_val = metric.compute().item()

    return total_loss, metric_val


def main() -> None:
    train_dataset = MyImageFolder("~/Downloads/cars/train")
    valid_dataset = MyImageFolder("~/Downloads/cars/valid")

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=128,
        shuffle=True,
        num_workers=4,
        persistent_workers=True,
        pin_memory=True,
    )
    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=128,
        shuffle=False,
        num_workers=4,
        persistent_workers=True,
        pin_memory=True,
    )

    n_classes = len(train_dataset.classes)
    model = ResNet(Structure.BASIC, repeats=[2, 2, 2, 2], fc_units=n_classes)
    _ = model(torch.randn(1, 3, 224, 224))  # Forward pass to initialize lazy layers.
    model = model.to(DEVICE)

    loss_fn = nn.CrossEntropyLoss()
    metric = metrics.MulticlassAccuracy(num_classes=n_classes, average="weighted")
    metric = metric.to(DEVICE)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=5)

    logging.info(f"Start training on {torch.cuda.get_device_name(DEVICE)}...")

    for epoch in range(1, 101):
        train_step(model, train_loader, loss_fn, optimizer)
        train_loss, train_acc = valid_step(model, train_loader, loss_fn, metric)
        valid_loss, valid_acc = valid_step(model, valid_loader, loss_fn, metric)

        scheduler.step(valid_loss)

        logging.info(
            f"Epoch: {epoch:03d} | "
            f"Train Loss: {train_loss:8.4f} | "
            f"Train Acc: {train_acc:8.4f} | "
            f"Valid Loss: {valid_loss:8.4f} | "
            f"Valid Acc: {valid_acc:8.4f}"
        )


if __name__ == "__main__":
    main()
