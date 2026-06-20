"""
TLOB — dual-attention transformer for LOB (Berti & Kasneci, 2025).

Vendored (thin wrapper) from the official implementation
    https://github.com/LeonardoBerti00/TLOB  (models/tlob.py, models/bin.py, models/mlplob.py)
adapted to this repo's single data contract and made self-contained:
  - removed the `constants as cst` / `cst.DEVICE` coupling (device is taken from the
    input tensor; the positional encoding is a registered buffer),
  - removed `einops` (replaced rearrange with permute/reshape),
  - removed the LOBSTER `dataset_type` branch (we always feed continuous (B, T, F)).
Architecture is otherwise unchanged: BiN over (features, time) → linear embed + positional
encoding → `num_layers` pairs of {temporal self-attention, feature self-attention} with a
permute between them → progressive MLP head → 3 logits.

Original authors retain credit; see the repo LICENSE.
"""

import torch
from torch import nn


class BiN(nn.Module):
    """Bilinear Normalization — normalizes along both the temporal and feature axes
    and learns to weight the two. `d1` = num_features, `t1` = seq length.
    Input/output shape: (B, d1, t1)."""

    def __init__(self, d1, t1):
        super().__init__()
        self.t1 = t1
        self.d1 = d1

        self.B1 = nn.Parameter(torch.zeros(t1, 1))
        self.l1 = nn.Parameter(torch.empty(t1, 1))
        nn.init.xavier_normal_(self.l1)

        self.B2 = nn.Parameter(torch.zeros(d1, 1))
        self.l2 = nn.Parameter(torch.empty(d1, 1))
        nn.init.xavier_normal_(self.l2)

        self.y1 = nn.Parameter(torch.full((1,), 0.5))
        self.y2 = nn.Parameter(torch.full((1,), 0.5))

    def forward(self, x):
        # keep the two mixing scalars non-negative (as in the original)
        if self.y1.item() < 0:
            nn.init.constant_(self.y1, 0.01)
        if self.y2.item() < 0:
            nn.init.constant_(self.y2, 0.01)

        device = x.device

        # normalization along the temporal dimension
        T2 = torch.ones([self.t1, 1], device=device)
        x2 = torch.mean(x, dim=2).reshape(x.shape[0], x.shape[1], 1)
        std = torch.std(x, dim=2).reshape(x.shape[0], x.shape[1], 1)
        std[std < 1e-4] = 1
        diff = x - (x2 @ T2.T)
        Z2 = diff / (std @ T2.T)
        X2 = self.l2 @ T2.T
        X2 = X2 * Z2
        X2 = X2 + (self.B2 @ T2.T)

        # normalization along the feature dimension
        T1 = torch.ones([self.d1, 1], device=device)
        x1 = torch.mean(x, dim=1).reshape(x.shape[0], x.shape[2], 1)
        std = torch.std(x, dim=1).reshape(x.shape[0], x.shape[2], 1)
        op1 = torch.permute(x1 @ T1.T, (0, 2, 1))
        op2 = torch.permute(std @ T1.T, (0, 2, 1))
        z1 = (x - op1) / op2
        X1 = T1 @ self.l1.T
        X1 = X1 * z1
        X1 = X1 + (T1 @ self.B1.T)

        return self.y1 * X1 + self.y2 * X2


class MLP(nn.Module):
    """Residual MLP block (start_dim → hidden_dim → final_dim) with LayerNorm + GELU."""

    def __init__(self, start_dim: int, hidden_dim: int, final_dim: int):
        super().__init__()
        self.layer_norm = nn.LayerNorm(final_dim)
        self.fc = nn.Linear(start_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, final_dim)
        self.gelu = nn.GELU()

    def forward(self, x):
        residual = x
        x = self.fc(x)
        x = self.gelu(x)
        x = self.fc2(x)
        if x.shape[2] == residual.shape[2]:
            x = x + residual
        x = self.layer_norm(x)
        x = self.gelu(x)
        return x


class ComputeQKV(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int):
        super().__init__()
        self.q = nn.Linear(hidden_dim, hidden_dim * num_heads)
        self.k = nn.Linear(hidden_dim, hidden_dim * num_heads)
        self.v = nn.Linear(hidden_dim, hidden_dim * num_heads)

    def forward(self, x):
        return self.q(x), self.k(x), self.v(x)


class TransformerLayer(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, final_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.qkv = ComputeQKV(hidden_dim, num_heads)
        self.attention = nn.MultiheadAttention(
            hidden_dim * num_heads, num_heads, batch_first=True
        )
        self.mlp = MLP(hidden_dim, hidden_dim * 4, final_dim)
        self.w0 = nn.Linear(hidden_dim * num_heads, hidden_dim)

    def forward(self, x):
        res = x
        q, k, v = self.qkv(x)
        x, _ = self.attention(q, k, v, need_weights=False)
        x = self.w0(x)
        x = x + res
        x = self.norm(x)
        x = self.mlp(x)
        if x.shape[-1] == res.shape[-1]:
            x = x + res
        return x


def sinusoidal_positional_embedding(
    token_sequence_size, token_embedding_dim, n=10000.0
):
    if token_embedding_dim % 2 != 0:
        raise ValueError(
            f"Sinusoidal positional embedding needs an even dim (got {token_embedding_dim})"
        )
    T, d = token_sequence_size, token_embedding_dim
    positions = torch.arange(0, T).unsqueeze(1)
    embeddings = torch.zeros(T, d)
    denom = torch.pow(n, 2 * torch.arange(0, d // 2) / d)
    embeddings[:, 0::2] = torch.sin(positions / denom)
    embeddings[:, 1::2] = torch.cos(positions / denom)
    return embeddings


class TLOB(nn.Module):
    """Dual-attention transformer (Berti & Kasneci 2025). Input (B, seq_size, num_features)
    of continuous LOB features → (B, 3) logits."""

    def __init__(
        self,
        hidden_dim: int,
        num_layers: int,
        seq_size: int,
        num_features: int,
        num_heads: int,
        is_sin_emb: bool = True,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.seq_size = seq_size

        self.norm_layer = BiN(num_features, seq_size)
        self.emb_layer = nn.Linear(num_features, hidden_dim)
        if is_sin_emb:
            self.register_buffer(
                "pos_encoder", sinusoidal_positional_embedding(seq_size, hidden_dim)
            )
        else:
            self.pos_encoder = nn.Parameter(torch.randn(1, seq_size, hidden_dim))

        self.layers = nn.ModuleList()
        for i in range(num_layers):
            if i != num_layers - 1:
                self.layers.append(TransformerLayer(hidden_dim, num_heads, hidden_dim))
                self.layers.append(TransformerLayer(seq_size, num_heads, seq_size))
            else:  # last pair reduces dimensionality before the head
                self.layers.append(
                    TransformerLayer(hidden_dim, num_heads, hidden_dim // 4)
                )
                self.layers.append(TransformerLayer(seq_size, num_heads, seq_size // 4))

        total_dim = (hidden_dim // 4) * (seq_size // 4)
        self.final_layers = nn.ModuleList()
        while total_dim > 128:
            self.final_layers.append(nn.Linear(total_dim, total_dim // 4))
            self.final_layers.append(nn.GELU())
            total_dim = total_dim // 4
        self.final_layers.append(nn.Linear(total_dim, 3))

    def forward(self, x):
        # x: (B, seq_size, num_features)
        x = x.permute(0, 2, 1)  # (B, F, S) for BiN
        x = self.norm_layer(x)
        x = x.permute(0, 2, 1)  # (B, S, F)
        x = self.emb_layer(x)  # (B, S, hidden)
        x = x + self.pos_encoder
        for layer in self.layers:
            x = layer(x)
            x = x.permute(0, 2, 1)  # alternate temporal <-> feature axis
        x = x.reshape(x.shape[0], -1)
        for layer in self.final_layers:
            x = layer(x)
        return x
