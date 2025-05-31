import torch
from tiktoken import Encoding
from torch import Tensor
from torch.utils.data import Dataset


class GPTDataset(Dataset):
    def __init__(self, tokenizer: Encoding, text: str, context_len: int, stride: int = 1) -> None:
        self.inp_ids = []
        self.des_ids = []

        tokens = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
        for i in range(0, len(tokens) - context_len, stride):
            inp_ids = tokens[i : i + context_len]
            des_ids = tokens[i + 1 : i + context_len + 1]
            self.inp_ids.append(torch.tensor(inp_ids))
            self.des_ids.append(torch.tensor(des_ids))

    def __len__(self) -> int:
        return len(self.inp_ids)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        return self.inp_ids[index], self.des_ids[index]
