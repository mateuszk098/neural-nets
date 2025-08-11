import random
import time
from os import PathLike
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import yaml
from tiktoken import Encoding
from torch import Tensor


def load_yaml(file: str | PathLike) -> dict[str, Any]:
    with open(file, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def create_current_run_dir(root: str | PathLike = "./checkpoints/") -> Path:
    current_run_dir = Path(root).joinpath(time.strftime("run-%Y-%m-%d-%H-%M-%S"))
    current_run_dir.mkdir(parents=True, exist_ok=True)
    return current_run_dir


def initialize_seed(seed: int = 42) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def text_to_tokens(text: str, tokenizer: Encoding) -> Tensor:
    tokens = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    return torch.tensor(tokens).unsqueeze(0)


def tokens_to_text(tokens: Tensor, tokenizer: Encoding) -> str:
    tokens = tokens.squeeze(0)
    return tokenizer.decode(tokens.tolist())


def generate_text(
    model: nn.Module,
    tokens: Tensor,
    max_new_tokens: int,
    context_len: int,
    eof_token: int | None = None,
    temperature: float | None = None,
    topk: int | None = None,
) -> Tensor:
    for _ in range(max_new_tokens):
        x = tokens[:, -context_len:]  # Keep only the last context_len tokens.
        with torch.no_grad():
            logits = model(x)
        logits = logits[:, -1, :]  # Only the latest token.

        if topk is not None:
            top_logits, top_indices = torch.topk(logits, topk)
            new_logits = torch.full_like(logits, -torch.inf)
            new_logits.scatter_(1, top_indices, top_logits)

        if temperature is not None:
            logits = logits / temperature
            probas = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probas, num_samples=1)
        else:
            next_token = torch.argmax(logits, dim=-1, keepdim=True)

        if eof_token is not None and next_token == eof_token:
            break

        tokens = torch.cat((tokens, next_token), dim=1)

    return tokens
