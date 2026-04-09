"""Minimal drop-in replacement for zeta.nn.FeedForward to avoid zetascale dependency."""

import torch.nn as nn
import torch.nn.functional as F


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, dropout=0.0, **kwargs):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = dim_out or dim
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, inner_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)
