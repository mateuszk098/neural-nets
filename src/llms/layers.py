import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim: int, num_heads: int, dropout: float = 0.0, qkv_bias: bool = False) -> None:
        assert embedding_dim % num_heads == 0, "embedding_dim must be divisible by num_heads"
        super().__init__()

        self.dropout = float(dropout)
        self.num_heads = int(num_heads)
        self.heads_dim = int(embedding_dim / num_heads)

        self.qkv = nn.Linear(embedding_dim, embedding_dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, x: Tensor) -> Tensor:
        batch_size, num_tokens, embedding_dim = x.size()

        qkv = self.qkv(x).contiguous()
        qkv = qkv.view(batch_size, num_tokens, 3, self.num_heads, self.heads_dim)
        # 3 x (batch_size, num_heads, num_tokens, heads_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)

        dropout = self.dropout if self.training else 0.0
        # (batch_size, num_heads, num_tokens, heads_dim)
        context_vectors = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout, is_causal=True)
        # Go back to (batch_size, num_tokens, num_heads, heads_dim) and then to original shape.
        context_vectors = context_vectors.transpose(1, 2).contiguous()
        context_vectors = context_vectors.view(batch_size, num_tokens, embedding_dim)

        return self.proj(context_vectors)


class FeedForward(nn.Module):
    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.ff = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 4),
            nn.GELU(approximate="tanh"),
            nn.Linear(embedding_dim * 4, embedding_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.ff(x)


class LayerNorm(nn.Module):
    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(embedding_dim))
        self.shift = nn.Parameter(torch.zeros(embedding_dim))
        self.eps = 1e-6

    def forward(self, x: Tensor) -> Tensor:
        mean = torch.mean(x, dim=-1, keepdim=True)
        std = torch.std(x, dim=-1, keepdim=True, unbiased=False)
        return self.scale * (x - mean) / (std + self.eps) + self.shift


class Transformer(nn.Module):
    def __init__(self, embedding_dim: int, num_heads: int, dropout: float = 0.0, qkv_bias: bool = False) -> None:
        super().__init__()
        self.attn = MultiHeadAttention(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            dropout=dropout,
            qkv_bias=qkv_bias,
        )
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.ff = FeedForward(embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        shortcut = x

        x = self.norm1(x)
        x = self.attn(x)
        x = self.dropout(x)
        x = shortcut + x

        shortcut = x

        x = self.norm2(x)
        x = self.ff(x)
        x = self.dropout(x)
        x = shortcut + x

        return x
