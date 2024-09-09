import logging
import time
from collections import defaultdict

import torch
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer  # type: ignore
from torch.utils.data import DataLoader, TensorDataset
from torcheval.metrics import Metric
from utils import EarlyStopping


def create_dataloaders(
    x_train: Tensor, y_train: Tensor, x_valid: Tensor, y_valid: Tensor, batch_size: int = 16
) -> tuple[DataLoader, DataLoader]:
    train_dataset = TensorDataset(x_train, y_train)
    valid_dataset = TensorDataset(x_valid, y_valid)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    return train_dataloader, valid_dataloader


def train_step(model: Module, loader: DataLoader, loss: Module, optimizer: Optimizer) -> None:
    model.train()

    for x, y in loader:
        y_pred = model.forward(x).squeeze()
        loss_value = loss.forward(y_pred, y)

        optimizer.zero_grad()
        loss_value.backward()
        optimizer.step()


def valid_step(
    model: Module, loader: DataLoader, loss: Module, metric: Metric
) -> tuple[float, float]:
    model.eval()
    metric.reset()
    loss_value = torch.tensor(0.0)

    with torch.inference_mode():
        for x, y in loader:
            y_pred = model.forward(x).squeeze()
            loss_value += loss.forward(y_pred, y)
            metric.update(y, y_pred)

    return loss_value.item() / len(loader), metric.compute().item()


def train(
    model: Module,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    loss: Module,
    metric: Metric,
    optimizer: Optimizer,
    early_stopping: EarlyStopping,
    n_epochs: int = 1000,
    verbose_step: int = 50,
) -> dict[str, list[float]]:
    history = defaultdict(list)
    log = "Epoch: {:3d} | Time: {:3.2f}s | Loss: {:8.5f} | MSE: {:8.5f} | Val Loss: {:8.5f} | Val MSE: {:8.5f}"

    for epoch in range(n_epochs):
        t0 = time.perf_counter()
        train_step(model, train_loader, loss, optimizer)
        t1 = time.perf_counter()

        train_loss, train_mse = valid_step(model, train_loader, loss, metric)
        valid_loss, valid_mse = valid_step(model, valid_loader, loss, metric)

        if early_stopping(valid_loss):
            logging.info("Early Stopping...")
            break

        if (epoch + 1) % verbose_step == 0 or epoch == 0:
            info = log.format(epoch + 1, t1 - t0, train_loss, train_mse, valid_loss, valid_mse)
            logging.info(info)

        history["Loss"].append(train_loss)
        history["MSE"].append(train_mse)
        history["Val Loss"].append(valid_loss)
        history["Val MSE"].append(valid_mse)

    return history
