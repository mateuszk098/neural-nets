import torch
import torch.nn as nn
from torch import Tensor

from nns.gpt.layers import LayerNorm, Transformer


class GPT2(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_len: int,
        num_layers: int,
        embedding_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        qkv_bias: bool = False,
    ) -> None:
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, embedding_dim)
        self.pos_emb = nn.Embedding(context_len, embedding_dim)
        self.dropout = nn.Dropout(dropout)

        self.trf_blocks = nn.Sequential(
            *[
                Transformer(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    qkv_bias=qkv_bias,
                )
                for _ in range(num_layers)
            ]
        )

        self.norm = LayerNorm(embedding_dim)
        self.out_head = nn.Linear(embedding_dim, vocab_size, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        batch_size, num_tokens = x.size()

        tok_emb = self.tok_emb(x)
        pos_emb = self.pos_emb(torch.arange(num_tokens, device=x.device))

        x = tok_emb + pos_emb
        x = self.dropout(x)
        x = self.trf_blocks(x)
        x = self.norm(x)

        return self.out_head(x)
