"""
models.py — All 4 models in one file. Kept minimal.

1. LogisticBaseline  — flatten window, linear layer, sigmoid
2. SimpleTransformer — self-attention over time steps, classify
3. SimpleGAT         — graph attention over stocks, classify
4. TransformerGAT    — Transformer encodes time, GAT shares across stocks
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import config


# ═══════════════════════════════════════════════════════
# 1. LOGISTIC BASELINE
# ═══════════════════════════════════════════════════════

class LogisticBaseline(nn.Module):
    """
    Flatten the window of returns, apply one linear layer.
    Equivalent to logistic regression on raw features.
    """
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(config.WINDOW_SIZE, 1)

    def forward(self, x, edge_index=None):
        # x: [B, N, W]
        out = self.linear(x).squeeze(-1)    # [B, N]
        return out                           # raw logits (use BCEWithLogitsLoss)


# ═══════════════════════════════════════════════════════
# 2. TRANSFORMER
# ═══════════════════════════════════════════════════════

class SimpleTransformer(nn.Module):
    """
    Per-stock Transformer: processes each stock's time series independently.
    Projects scalar returns to d_model, adds positional encoding,
    runs through Transformer encoder, mean-pools, classifies.
    """
    def __init__(self):
        super().__init__()
        d = config.D_MODEL

        # Project scalar -> d_model
        self.input_proj = nn.Linear(1, d)

        # Positional encoding (fixed sinusoidal)
        pe = torch.zeros(config.WINDOW_SIZE, d)
        pos = torch.arange(config.WINDOW_SIZE, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d, 2, dtype=torch.float) * (-math.log(10000.0) / d))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))   # [1, W, d]

        # Transformer encoder
        layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=config.N_HEADS, dim_feedforward=d * 4,
            dropout=0.1, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=config.N_LAYERS)

        # Classification head
        self.head = nn.Linear(d, 1)

    def forward(self, x, edge_index=None):
        B, N, W = x.shape
        x = x.reshape(B * N, W, 1)                 # [B*N, W, 1]
        x = self.input_proj(x) + self.pe[:, :W, :]  # [B*N, W, d]
        x = self.encoder(x)                          # [B*N, W, d]
        x = x.mean(dim=1)                           # pool over time -> [B*N, d]
        out = self.head(x).squeeze(-1)               # [B*N]
        return out.reshape(B, N)                     # [B, N] logits


# ═══════════════════════════════════════════════════════
# 3. GAT (Graph Attention Network)
# ═══════════════════════════════════════════════════════

class GATLayer(nn.Module):
    """Single multi-head graph attention layer."""
    def __init__(self, d_in, d_out, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.d_out = d_out
        self.W = nn.Parameter(torch.randn(n_heads, d_in, d_out) * 0.01)
        self.a = nn.Parameter(torch.randn(n_heads, 2 * d_out) * 0.01)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, x, edge_index):
        N = x.size(0)
        K = self.n_heads
        Wh = torch.einsum("ni,kio->kno", x, self.W)        # [K, N, d_out]
        src, tgt = edge_index[0], edge_index[1]
        edge_feat = torch.cat([Wh[:, src], Wh[:, tgt]], dim=-1)
        e = self.leaky_relu(torch.einsum("kef,kf->ke", edge_feat, self.a))

        # Stable softmax per target node
        e_max = _scatter_max(e, tgt, N)
        e_exp = torch.exp(e - e_max[:, tgt])
        e_sum = torch.zeros(K, N, device=x.device).scatter_add(
            1, tgt.unsqueeze(0).expand(K, -1), e_exp)
        alpha = e_exp / (e_sum[:, tgt] + 1e-10)

        # Aggregate
        weighted = alpha.unsqueeze(-1) * Wh[:, src]
        out = torch.zeros(K, N, self.d_out, device=x.device).scatter_add(
            1, tgt.unsqueeze(0).unsqueeze(-1).expand(K, -1, self.d_out), weighted)
        return F.elu(out.permute(1, 0, 2).reshape(N, K * self.d_out))


def _scatter_max(src, index, N):
    K = src.shape[0]
    out = torch.full((K, N), float("-inf"), device=src.device)
    out = out.scatter_reduce(1, index.unsqueeze(0).expand(K, -1),
                             src, reduce="amax", include_self=True)
    return out.masked_fill(out == float("-inf"), 0.0)


class SimpleGAT(nn.Module):
    """
    GAT that operates on per-stock features.
    First compresses each stock's window into an embedding (via linear),
    then applies graph attention to share info between stocks.
    """
    def __init__(self):
        super().__init__()
        d = config.D_MODEL
        # Simple encoder: flatten window -> embedding
        self.encoder = nn.Sequential(
            nn.Linear(config.WINDOW_SIZE, d),
            nn.ReLU(),
        )
        # GAT layers
        self.gat1 = GATLayer(d, d // config.N_HEADS, config.N_HEADS)
        self.norm1 = nn.LayerNorm(d)
        self.gat2 = GATLayer(d, d, n_heads=1)
        self.norm2 = nn.LayerNorm(d)
        # Classifier
        self.head = nn.Linear(d, 1)

    def forward(self, x, edge_index):
        B, N, W = x.shape
        # Encode each stock's window
        x_flat = x.reshape(B * N, W)
        emb = self.encoder(x_flat).reshape(B, N, -1)   # [B, N, d]

        # GAT message passing (per sample)
        out = torch.zeros_like(emb)
        for b in range(B):
            h = self.gat1(emb[b], edge_index)
            h = self.norm1(h)
            h = self.gat2(h, edge_index)
            out[b] = self.norm2(h + emb[b])             # residual

        logits = self.head(out).squeeze(-1)             # [B, N]
        return logits


# ═══════════════════════════════════════════════════════
# 4. TRANSFORMER + GAT (Combined)
# ═══════════════════════════════════════════════════════

class TransformerGAT(nn.Module):
    """
    Stage 1: Transformer encodes each stock's time series (temporal)
    Stage 2: GAT shares information between correlated stocks (spatial)
    Stage 3: Classify UP/DOWN per stock
    """
    def __init__(self):
        super().__init__()
        d = config.D_MODEL

        # Temporal encoder (same as SimpleTransformer)
        self.input_proj = nn.Linear(1, d)
        pe = torch.zeros(config.WINDOW_SIZE, d)
        pos = torch.arange(config.WINDOW_SIZE, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d, 2, dtype=torch.float) * (-math.log(10000.0) / d))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

        layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=config.N_HEADS, dim_feedforward=d * 4,
            dropout=0.1, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=config.N_LAYERS)

        # Graph encoder
        self.gat1 = GATLayer(d, d // config.N_HEADS, config.N_HEADS)
        self.norm1 = nn.LayerNorm(d)
        self.gat2 = GATLayer(d, d, n_heads=1)
        self.norm2 = nn.LayerNorm(d)

        # Classifier
        self.head = nn.Linear(d, 1)

    def _encode_temporal(self, x):
        B, N, W = x.shape
        x = x.reshape(B * N, W, 1)
        x = self.input_proj(x) + self.pe[:, :W, :]
        x = self.encoder(x)
        x = x.mean(dim=1)
        return x.reshape(B, N, -1)

    def forward(self, x, edge_index):
        B, N, W = x.shape
        emb = self._encode_temporal(x)       # [B, N, d]

        # GAT message passing
        out = torch.zeros_like(emb)
        for b in range(B):
            h = self.gat1(emb[b], edge_index)
            h = self.norm1(h)
            h = self.gat2(h, edge_index)
            out[b] = self.norm2(h + emb[b])

        logits = self.head(out).squeeze(-1)
        return logits
