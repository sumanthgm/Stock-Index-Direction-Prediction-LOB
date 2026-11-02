"""
Model architectures for LOB mid-price direction prediction.

    DeepLOB  — Zhang 2019 (universal baseline)
    MLPLOB   — Berti & Kasneci 2025 (simple MLP + BiN, ablation floor)
    TLOB     — Berti & Kasneci 2025 (dual-attention transformer; vendored in tlob.py)
    MambaLOB — This work (Selective SSM for LOB)

All models share the same interface:
    model = DeepLOB()            # or MLPLOB(), TLOBModel(), MambaLOB()
    logits = model(x)            # x: (B, seq_len, 40)  →  logits: (B, 3)

Input convention (DeepLOB standard):
    x shape: (batch, seq_len=100, features=40)
    features ordered as: [ask_p1, ask_v1, bid_p1, bid_v1, ..., ×10 levels]
    Labels: 0=Down, 1=Stationary, 2=Up
"""

from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Shared utility: Bilinear Normalization (BiN) from Berti & Kasneci 2025
# ---------------------------------------------------------------------------
class BilinearNorm(nn.Module):
    """
    Bilinear Normalization (BiN) — Berti & Kasneci 2025 (TLOB paper).
    Normalises across the feature dimension using two learned weight vectors,
    shown to stabilise LOB training better than LayerNorm or BatchNorm.
    Applied to the flattened or pooled representation before the classifier head.
    """

    def __init__(self, d: int):
        super().__init__()
        self.w1 = nn.Parameter(torch.ones(d))
        self.w2 = nn.Parameter(torch.ones(d))
        self.b = nn.Parameter(torch.zeros(d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, d)
        return self.w1 * x * torch.sigmoid(self.w2 * x) + self.b


# ---------------------------------------------------------------------------
# DeepLOB — Zhang et al. 2019
# ---------------------------------------------------------------------------
class _InceptionModule(nn.Module):
    """Lightweight Inception block operating on the temporal axis."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        third = out_channels // 3
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, third, kernel_size=(1, 1)),
            nn.LeakyReLU(0.01),
            nn.Conv2d(third, third, kernel_size=(3, 1), padding=(1, 0)),
            nn.LeakyReLU(0.01),
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, third, kernel_size=(1, 1)),
            nn.LeakyReLU(0.01),
            nn.Conv2d(third, third, kernel_size=(5, 1), padding=(2, 0)),
            nn.LeakyReLU(0.01),
        )
        self.branch3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(3, 1), stride=1, padding=(1, 0)),
            nn.Conv2d(in_channels, out_channels - 2 * third, kernel_size=(1, 1)),
            nn.LeakyReLU(0.01),
        )

    def forward(self, x):
        return torch.cat([self.branch1(x), self.branch2(x), self.branch3(x)], dim=1)


class DeepLOB(nn.Module):
    """
    DeepLOB: Deep Learning for Limit Order Books — Zhang et al. 2019.
    Architecture: Conv feature extractor → Inception → LSTM → FC(3).

    Input:  (B, T=100, F=40)
    Output: (B, 3) logits
    """

    def __init__(self, seq_len: int = 100, n_features: int = 40, lstm_hidden: int = 64):
        super().__init__()
        self.seq_len = seq_len

        # CNN: extract local feature interactions across LOB levels
        # Reshape input to (B, 1, T, F) for Conv2d
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(1, 2), stride=(1, 2)),  # (B,32,T,20)
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=(4, 1)),  # (B,32,T-3,20)
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=(4, 1)),  # (B,32,T-6,20)
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(32),
        )

        # Inception: capture multi-scale temporal patterns
        self.inception = _InceptionModule(32, 32)

        # LSTM: sequence model over inception output
        self.lstm = nn.LSTM(
            input_size=32 * 20,  # channels × spatial
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
        )

        self.fc = nn.Linear(lstm_hidden, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F)
        x = x.unsqueeze(1)  # (B, 1, T, F)
        x = self.conv(x)  # (B, 32, T', F')
        x = self.inception(x)  # (B, 32, T', F')
        B, C, T, F = x.shape
        x = x.permute(0, 2, 1, 3)  # (B, T', C, F')
        x = x.reshape(B, T, C * F)  # (B, T', C*F)
        _, (h, _) = self.lstm(x)  # h: (1, B, hidden)
        x = h.squeeze(0)  # (B, hidden)
        return self.fc(x)  # (B, 3)


# ---------------------------------------------------------------------------
# MLPLOB — Berti & Kasneci 2025
# ---------------------------------------------------------------------------
class MLPLOB(nn.Module):
    """
    MLPLOB — Berti & Kasneci 2025 (from the TLOB paper).
    Simple MLP baseline with Bilinear Normalization.
    Establishes the floor: shows how much temporal modelling matters.

    Input:  (B, T=100, F=40)
    Output: (B, 3) logits
    """

    def __init__(self, seq_len: int = 100, n_features: int = 40, hidden: int = 128):
        super().__init__()
        in_dim = seq_len * n_features
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        self.bn = BilinearNorm(hidden)
        self.fc = nn.Linear(hidden, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x)  # (B, hidden)
        x = self.bn(x)
        return self.fc(x)  # (B, 3)


# ---------------------------------------------------------------------------
# MambaLOB — This work
# ---------------------------------------------------------------------------
class _MambaBlock(nn.Module):
    """
    Pure-PyTorch selective SSM block — no CUDA kernel dependency.
    Approximates Mamba's selective state update:
        Δ, B, C = linear projections of x  (input-dependent gating)
        h_t = diag(exp(Δ·A))·h_{t-1} + Δ·B·x_t   (selective forgetting)
        y_t = C·h_t
    Followed by a gated output projection (SiLU gate, as in Mamba paper).

    For the CUDA-optimised version on Colab A100:
        from mamba_ssm import Mamba
        block = Mamba(d_model=d_model, d_state=d_state, d_conv=4, expand=2)
    Drop-in replacement — same input/output shape.
    """

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = d_model * expand

        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)

        # Selective parameters: Δ (dt), B, C depend on the input x
        self.dt_proj = nn.Linear(self.d_inner, self.d_inner, bias=True)
        self.B_proj = nn.Linear(self.d_inner, d_state, bias=False)
        self.C_proj = nn.Linear(self.d_inner, d_state, bias=False)

        # Fixed A initialised log-uniformly (from original Mamba paper)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0).expand(self.d_inner, -1).contiguous()
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # Depthwise conv for local context (d_conv = 4 in Mamba)
        self.conv1d = nn.Conv1d(
            self.d_inner,
            self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner,
            bias=True,
        )

        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, d_model)
        residual = x
        x = self.norm(x)

        xz = self.in_proj(x)  # (B, L, 2*d_inner)
        x_in, z = xz.chunk(2, dim=-1)  # each (B, L, d_inner)

        # Local conv on the sequence axis
        x_in = self.conv1d(x_in.transpose(1, 2))  # (B, d_inner, L + pad)
        x_in = x_in[:, :, : x.shape[1]].transpose(1, 2)  # (B, L, d_inner)
        x_in = F.silu(x_in)

        # Selective parameters (input-dependent)
        dt = F.softplus(self.dt_proj(x_in))  # (B, L, d_inner)
        B = self.B_proj(x_in)  # (B, L, d_state)
        C = self.C_proj(x_in)  # (B, L, d_state)
        A = -torch.exp(self.A_log)  # (d_inner, d_state), negative

        # Discretised selective SSM scan (sequential, O(L) loop)
        B_size, L, _ = x_in.shape
        h = torch.zeros(B_size, self.d_inner, self.d_state, device=x.device, dtype=x.dtype)
        ys = []
        for t in range(L):
            dt_t = dt[:, t, :].unsqueeze(-1)  # (B, d_inner, 1)
            dA = torch.exp(dt_t * A.unsqueeze(0))  # (B, d_inner, d_state)
            dB = dt_t * B[:, t, :].unsqueeze(1)  # (B, d_inner, d_state)
            h = dA * h + dB * x_in[:, t, :].unsqueeze(-1)  # (B, d_inner, d_state)
            y_t = (h * C[:, t, :].unsqueeze(1)).sum(-1) + self.D * x_in[:, t, :]
            ys.append(y_t)
        y = torch.stack(ys, dim=1)  # (B, L, d_inner)

        # Gated output
        y = y * F.silu(z)
        return self.out_proj(y) + residual


class _BiMamba(nn.Module):
    """
    Bidirectional wrapper around any Mamba block (Tier-B novelty).

    A vanilla Mamba scan is strictly causal (left→right), which suits streaming
    forecasting but discards the fact that, *within a fixed look-back window*, the
    most recent events are as informative when read backwards. We run the same
    look-back window forwards and backwards through two independent blocks and
    average — letting each position attend to its full local neighbourhood while
    keeping the O(L) cost (2× constant). Works with both the CUDA kernel and the
    pure-PyTorch fallback since both map (B, L, d) → (B, L, d).
    """

    def __init__(self, mamba_cls, d_model: int, d_state: int = 16):
        super().__init__()
        self.fwd = mamba_cls(d_model, d_state=d_state)
        self.bwd = mamba_cls(d_model, d_state=d_state)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.fwd(x)
        b = torch.flip(self.bwd(torch.flip(x, dims=[1])), dims=[1])
        return 0.5 * (f + b)


def _resolve_mamba_cls(d_state: int, use_cuda_mamba: bool, who: str = "Mamba"):
    """Return a Mamba block constructor: CUDA kernel if available, else fallback."""
    mamba_cls = _MambaBlock
    if use_cuda_mamba:
        try:
            from mamba_ssm import Mamba as CudaMamba

            mamba_cls = lambda d, **kw: CudaMamba(  # noqa: E731
                d_model=d, d_state=d_state, d_conv=4, expand=2
            )
            print(f"{who}: using mamba-ssm CUDA kernel.")
        except ImportError:
            print(f"{who}: mamba-ssm not available, using pure-PyTorch fallback.")
    return mamba_cls


def _make_mamba_layers(mamba_cls, d_model: int, d_state: int, n_layers: int, bidirectional: bool) -> nn.ModuleList:
    def one():
        if bidirectional:
            return _BiMamba(mamba_cls, d_model, d_state=d_state)
        return mamba_cls(d_model, d_state=d_state)

    return nn.ModuleList([one() for _ in range(n_layers)])


class MambaLOB(nn.Module):
    """
    MambaLOB — Selective State Space Model for LOB mid-price direction prediction.
    Novel architecture; first application of Mamba-family SSMs to LOB prediction.

    Architecture:
        Linear projection (40 → d_model)
        → Mamba Block × n_layers       [selective temporal context; optionally bidirectional]
        → [optional] Spatial MHA       [feature-axis attention, from TLOB]
        → BiN                          [Bilinear Normalization, from TLOB]
        → FC(3) → logits

    To use the CUDA-optimised Mamba kernel instead of the pure-PyTorch fallback:
        1. pip install mamba-ssm causal-conv1d
        2. Set use_cuda_mamba=True in the constructor.
        Fallback (_MambaBlock) is used when mamba_ssm is unavailable or on CPU.

    bidirectional=True wraps each block in _BiMamba (Tier-B improved variant).

    Input:  (B, T=100, F=40)
    Output: (B, 3) logits
    """

    def __init__(
        self,
        seq_len: int = 100,
        n_features: int = 40,
        d_model: int = 64,
        d_state: int = 16,
        n_layers: int = 2,
        spatial_heads: int = 0,  # 0 = no spatial MHA (ablation flag)
        bidirectional: bool = False,
        use_cuda_mamba: bool = True,
    ):
        super().__init__()
        self.input_proj = nn.Linear(n_features, d_model)

        mamba_cls = _resolve_mamba_cls(d_state, use_cuda_mamba, who="MambaLOB")
        self.mamba_layers = _make_mamba_layers(mamba_cls, d_model, d_state, n_layers, bidirectional)

        # Optional spatial self-attention over the feature axis (from TLOB)
        self.spatial_attn = None
        if spatial_heads > 0:
            self.spatial_norm = nn.LayerNorm(seq_len)
            self.spatial_attn = nn.MultiheadAttention(embed_dim=seq_len, num_heads=spatial_heads, batch_first=True)

        self.bn = BilinearNorm(d_model)
        self.fc = nn.Linear(d_model, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F=40)
        x = self.input_proj(x)  # (B, T, d_model)

        for layer in self.mamba_layers:
            x = layer(x)  # (B, T, d_model)

        # Optional: spatial attention over feature axis
        if self.spatial_attn is not None:
            # Transpose: treat features as sequence, time as embedding
            xT = x.permute(0, 2, 1)  # (B, d_model, T)
            xT = self.spatial_norm(xT)
            xT, _ = self.spatial_attn(xT, xT, xT)
            x = xT.permute(0, 2, 1)  # (B, T, d_model)

        x = x[:, -1, :]  # last timestep  (B, d_model)
        x = self.bn(x)
        return self.fc(x)  # (B, 3)


# ---------------------------------------------------------------------------
# ConvMambaLOB — This work (Tier-B headline architecture)
# ---------------------------------------------------------------------------
class ConvMambaLOB(nn.Module):
    """
    ConvMambaLOB — hybrid CNN + Selective SSM (Tier-B novelty).

    Motivation: DeepLOB pairs a strong *spatial* feature extractor (Conv + Inception
    over LOB levels) with a *temporal* LSTM. MambaLOB drops the spatial stage and
    feeds raw features straight into the SSM. ConvMambaLOB keeps DeepLOB's spatial
    inductive bias but replaces its O(L) sequential LSTM with a selective SSM:

        (B,1,T,F) → Conv stack → Inception   [local price×volume / level interactions]
        → flatten channels×width per step     → (B, T', C·F')
        → Linear → d_model
        → Mamba Block × n_layers              [selective long-range temporal context]
        → BiN → FC(3)

    The conv front-end is layout-agnostic here (output dim discovered with a dummy
    pass), so it accepts any feature_set (base40 … all/66). Goal: match/beat TLOB at
    lower compute — i.e. "we designed a better LOB model", not "we applied Mamba".

    Input:  (B, T=100, F)
    Output: (B, 3) logits
    """

    def __init__(
        self,
        seq_len: int = 100,
        n_features: int = 40,
        d_model: int = 64,
        d_state: int = 16,
        n_layers: int = 2,
        bidirectional: bool = False,
        use_cuda_mamba: bool = True,
    ):
        super().__init__()
        self.seq_len = seq_len

        # DeepLOB-style spatial feature extractor (no temporal reduction in Inception)
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(1, 2), stride=(1, 2)),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=(4, 1)),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=(4, 1)),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(32),
        )
        self.inception = _InceptionModule(32, 32)

        # Discover conv output shape so any feature width works
        with torch.no_grad():
            dummy = torch.zeros(1, 1, seq_len, n_features)
            o = self.inception(self.conv(dummy))  # (1, C, T', F')
            _, c_out, _, f_out = o.shape
        self.proj = nn.Linear(c_out * f_out, d_model)

        mamba_cls = _resolve_mamba_cls(d_state, use_cuda_mamba, who="ConvMambaLOB")
        self.mamba_layers = _make_mamba_layers(mamba_cls, d_model, d_state, n_layers, bidirectional)

        self.bn = BilinearNorm(d_model)
        self.fc = nn.Linear(d_model, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F)
        x = x.unsqueeze(1)  # (B, 1, T, F)
        x = self.conv(x)  # (B, 32, T', F')
        x = self.inception(x)  # (B, 32, T', F')
        B, C, T, Fp = x.shape
        x = x.permute(0, 2, 1, 3).reshape(B, T, C * Fp)  # (B, T', C*F')
        x = self.proj(x)  # (B, T', d_model)
        for layer in self.mamba_layers:
            x = layer(x)
        x = x[:, -1, :]  # last timestep
        x = self.bn(x)
        return self.fc(x)  # (B, 3)


# ---------------------------------------------------------------------------
# TLOB — Berti & Kasneci 2025 (dual-attention transformer; vendored in tlob.py)
# ---------------------------------------------------------------------------
class TLOBModel(nn.Module):
    """Thin wrapper over the vendored official TLOB so it fits our contract
    (B, seq_len, n_features) -> (B, 3). FI-2010 paper settings: num_layers=4,
    num_heads=1, sinusoidal positional embedding. hidden_dim=128 (sweep range)."""

    def __init__(
        self,
        seq_len: int = 100,
        n_features: int = 40,
        hidden_dim: int = 128,
        num_layers: int = 4,
        num_heads: int = 1,
        is_sin_emb: bool = True,
    ):
        super().__init__()
        from tlob import TLOB

        self.net = TLOB(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            seq_size=seq_len,
            num_features=n_features,
            num_heads=num_heads,
            is_sin_emb=is_sin_emb,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)  # (B, 3)


# ---------------------------------------------------------------------------
# Registry — easy lookup by name string
# ---------------------------------------------------------------------------
MODEL_REGISTRY = {
    "deeplob": DeepLOB,
    "mlplob": MLPLOB,
    "mambalob": MambaLOB,
    "tlob": TLOBModel,
    # Tier-B improved MambaLOB variants (this work)
    "bimambalob": partial(MambaLOB, bidirectional=True),
    "convmambalob": ConvMambaLOB,
    "biconvmambalob": partial(ConvMambaLOB, bidirectional=True),
}


def build_model(name: str, **kwargs) -> nn.Module:
    """Instantiate a model by name. Extra kwargs forwarded to constructor."""
    key = name.lower()
    if key not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Choose from {list(MODEL_REGISTRY)}.")
    return MODEL_REGISTRY[key](**kwargs)
