import logging
import math
from argparse import ArgumentParser
from os import PathLike
from pathlib import Path

import tiktoken
import torch
import torch.nn as nn
import torch.nn.functional as F
from tiktoken import Encoding
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader

import llms.utils as utils
from llms.dataset import GPT2Dataset
from llms.models import GPT2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FORMAT = "[%(asctime)s - %(module)s/%(levelname)s]: %(message)s"
DATE_FMT = "%Y-%m-%d %H:%M:%S"


def setup() -> None:
    utils.initialize_seed()
    torch.backends.cudnn.benchmark = True
    logging.basicConfig(level=logging.INFO, format=FORMAT, datefmt=DATE_FMT)


def calc_loss_batch(inputs: Tensor, targets: Tensor, model: nn.Module, device: torch.device = DEVICE) -> Tensor:
    inputs = inputs.to(device)
    targets = targets.to(device)
    logits = model(inputs)
    return F.cross_entropy(logits.flatten(0, 1), targets.flatten(), reduction="mean")


def calc_loss_loader(model: nn.Module, loader: DataLoader, num_batches: int | None = None) -> float:
    total_loss = 0.0
    num_batches = len(loader) if num_batches is None else min(num_batches, len(loader))

    for i, (inputs, targets) in enumerate(loader, start=1):
        loss = calc_loss_batch(inputs, targets, model)
        total_loss += loss.item()
        if i == num_batches:
            break

    return total_loss / num_batches


def evaluate_model(
    model: nn.Module,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    eval_iter: int | None = None,
) -> tuple[float, float]:
    with torch.inference_mode():
        train_loss = calc_loss_loader(model, train_loader, eval_iter)
        valid_loss = calc_loss_loader(model, valid_loader, eval_iter)
    return train_loss, valid_loss


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    optimizer: Optimizer,
    scheduler: LRScheduler,
    tokenizer: Encoding,
    num_epochs: int,
    warmup_steps: int | None = None,
    eval_iter: int | None = None,
    device: torch.device = DEVICE,
) -> None:
    token_seen, train_step = 0, 0
    train_loss_hist, valid_loss_hist, token_seen_hist = [], [], []

    for epoch in range(1, num_epochs + 1):
        model.train()

        for inputs, targets in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(inputs, targets, model)
            loss.backward()
            optimizer.step()
            scheduler.step()
            train_step += 1
            token_seen += inputs.numel()

            if warmup_steps is not None and train_step > warmup_steps:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        model.eval()
        train_loss, valid_loss = evaluate_model(model, train_loader, valid_loader, eval_iter=eval_iter)
        train_loss_hist.append(train_loss)
        valid_loss_hist.append(valid_loss)
        token_seen_hist.append(token_seen)

        logging.info(f"Epoch: {epoch:3d}/{num_epochs} Train: {train_loss:.4f} Valid: {valid_loss:.4f}")

        input_context = "Every effort moves you"
        context_len = model.pos_emb.weight.shape[0]

        tokens = utils.text_to_tokens(input_context, tokenizer).to(device)
        tokens = utils.generate_text(model=model, tokens=tokens, max_new_tokens=50, context_len=context_len)
        text = utils.tokens_to_text(tokens, tokenizer)
        print(text.replace("\n", " "))


def main(*, config_file: str | PathLike) -> None:
    logging.info(f"Loading configuration from {config_file!s}...")
    config = utils.load_yaml(config_file)

    tokenizer = tiktoken.get_encoding("gpt2")
    vocab_size = tokenizer.n_vocab

    with open("./the-verdict.txt") as f:
        text = f.read()

    train_ratio = 0.9
    split_id = int(len(text) * train_ratio)
    train_text = text[:split_id]
    valid_text = text[split_id:]

    train_ds = GPT2Dataset(tokenizer, train_text, context_len=config["CONTEXT_LEN"], stride=config["CONTEXT_LEN"])
    valid_ds = GPT2Dataset(tokenizer, valid_text, context_len=config["CONTEXT_LEN"], stride=config["CONTEXT_LEN"])

    train_loader = DataLoader(
        dataset=train_ds,
        batch_size=config["BATCH_SIZE"],
        shuffle=True,
        num_workers=config["NUM_WORKERS"],
        pin_memory=True,
        drop_last=True,
        persistent_workers=config["PERSISTENT_WORKERS"],
    )
    valid_loader = DataLoader(
        dataset=valid_ds,
        batch_size=config["BATCH_SIZE"],
        shuffle=False,
        num_workers=config["NUM_WORKERS"],
        pin_memory=True,
        drop_last=False,
        persistent_workers=config["PERSISTENT_WORKERS"],
    )

    architecture = config["ARCHITECTURES"][config["ARCHITECTURE"]]
    model = GPT2(
        vocab_size=vocab_size,
        context_len=config["CONTEXT_LEN"],
        num_layers=architecture["NUM_LAYERS"],
        embedding_dim=architecture["EMBEDDING_DIM"],
        num_heads=architecture["NUM_HEADS"],
        dropout=config["DROPOUT"],
        qkv_bias=config["QKV_BIAS"],
    )
    model.to(DEVICE)

    optimizer = torch.optim.NAdam(
        params=model.parameters(),
        weight_decay=config["WEIGHT_DECAY"],
        decoupled_weight_decay=config["DECOUPLED_WEIGHT_DECAY"],
    )
    scheduler_steps = config["NUM_EPOCHS"] * int(math.ceil(len(train_ds) / config["BATCH_SIZE"]))
    warmup_steps = int(config["PCT_START"] * scheduler_steps)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer=optimizer,
        max_lr=config["MAX_LR"],
        div_factor=config["DIV_FACTOR"],
        final_div_factor=config["FINAL_DIV_FACTOR"],
        three_phase=config["THREE_PHASE"],
        total_steps=scheduler_steps,
        pct_start=config["PCT_START"],
        anneal_strategy=config["ANNEAL_STRATEGY"],
        cycle_momentum=config["CYCLE_MOMENTUM"],
        base_momentum=config["BASE_MOMENTUM"],
        max_momentum=config["MAX_MOMENTUM"],
    )

    train_model(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        tokenizer=tokenizer,
        warmup_steps=warmup_steps,
        num_epochs=config["NUM_EPOCHS"],
        eval_iter=config["EVAL_ITER"],
    )


if __name__ == "__main__":
    setup()

    p = ArgumentParser()
    p.add_argument("--config-file", type=Path, required=False, default=Path("./config.yaml"))
    kwargs = vars(p.parse_args())
    main(**kwargs)
