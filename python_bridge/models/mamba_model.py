"""
=============================================================
  Python ML Bridge - Mamba (S6 Selective State Space Model)
  Gu & Dao, CMU / Princeton, 2023 — https://arxiv.org/abs/2312.00752

  Why Mamba is different from everything else in the stack:
  ──────────────────────────────────────────────────────────
  • Transformer: O(L²) attention — learns WHICH bars relate
  • LSTM: gated recurrence — learns SEQUENTIAL flow
  • TCN/PatchTST: convolutions — learns LOCAL patterns
  • Mamba (S6): selective state spaces — learns WHAT TO REMEMBER
    from the price stream and WHAT TO FORGET at each bar

  The key innovation is input-dependent state transitions:
    Δ_t, B_t, C_t = f(x_t)   ← all three depend on the current bar
  This means Mamba can selectively compress relevant price history
  (e.g., remember the last swing high, forget irrelevant noise).

  Pure PyTorch implementation — no CUDA kernels required.
  Sequential scan over seq_len=64 → negligible CPU overhead.

  Architecture:
    Input (batch, 64, features)
    → Input embedding → (batch, 64, d_model)
    → N × MambaBlock (SSM + SiLU gate + residual)
    → LayerNorm → global avg pool
    → Classifier → (batch, 3)
=============================================================
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import MambaConfig


# ─────────────────────────────────────────────
#  SELECTIVE SCAN (S6 core)
# ─────────────────────────────────────────────
def selective_scan(
    x: torch.Tensor,       # (B, L, d_inner)
    delta: torch.Tensor,   # (B, L, d_inner)  — input-dependent step size
    A: torch.Tensor,       # (d_inner, d_state) — negative diagonal
    B: torch.Tensor,       # (B, L, d_state)   — input-dependent
    C: torch.Tensor,       # (B, L, d_state)   — input-dependent
    D: torch.Tensor,       # (d_inner,)         — skip connection
) -> torch.Tensor:
    """
    Discretised S6 selective scan.

    Recurrence (Zero-Order Hold discretisation):
        Ā_t = exp(Δ_t ⊙ A)           diagonal, shape (d_inner, d_state)
        B̄_t = Δ_t ⊙ B_t              shape (d_inner, d_state)
        h_t = Ā_t ⊙ h_{t-1} + B̄_t ⊙ x_t    state update
        y_t = C_t · h_t + D ⊙ x_t            output

    For seq_len=64 the sequential loop is microseconds on CPU.
    Returns: y (B, L, d_inner)
    """
    B_sz, L, d_inner = x.shape
    d_state = A.shape[1]

    # Discretise A: (B, L, d_inner, d_state)
    A_bar = torch.exp(delta.unsqueeze(-1) * A)          # A is already negative

    # Discretise B (simplified ZOH): (B, L, d_inner, d_state)
    B_bar = delta.unsqueeze(-1) * B.unsqueeze(2)

    # Sequential scan
    h = x.new_zeros(B_sz, d_inner, d_state)
    ys = []
    for t in range(L):
        # State update: h = Ā ⊙ h + B̄ ⊙ x
        h = A_bar[:, t] * h + B_bar[:, t] * x[:, t].unsqueeze(-1)
        # Output: y = C · h   →   (B, d_inner)
        y_t = (h * C[:, t].unsqueeze(1)).sum(-1)
        ys.append(y_t)

    y = torch.stack(ys, dim=1)          # (B, L, d_inner)
    return y + D * x                    # skip connection


# ─────────────────────────────────────────────
#  MAMBA BLOCK
# ─────────────────────────────────────────────
class MambaBlock(nn.Module):
    """
    Single Mamba block with:
      1. Input projection → x branch + z gate branch
      2. Depthwise Conv1d for local context (4-bar window)
      3. SiLU activation
      4. S6 selective scan
      5. Multiplicative gating: y = SSM(x) * SiLU(z)
      6. Output projection + residual
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        d_inner  = expand * d_model
        dt_rank  = max(1, d_model // 16)
        self.d_inner  = d_inner
        self.d_state  = d_state
        self.dt_rank  = dt_rank

        self.norm     = nn.LayerNorm(d_model)

        # In-projection: produces x branch + z gate
        self.in_proj  = nn.Linear(d_model, 2 * d_inner, bias=False)

        # Short depthwise conv for local temporal context
        self.conv1d   = nn.Conv1d(
            d_inner, d_inner, kernel_size=d_conv,
            padding=d_conv - 1, groups=d_inner, bias=True,
        )

        # SSM parameter projections
        self.x_proj   = nn.Linear(d_inner, dt_rank + 2 * d_state, bias=False)
        self.dt_proj  = nn.Linear(dt_rank, d_inner, bias=True)

        # A: learnable log-diagonal of state matrix (initialised s.t. A < 0)
        A_init = torch.arange(1, d_state + 1, dtype=torch.float32)
        self.A_log = nn.Parameter(
            torch.log(A_init).unsqueeze(0).expand(d_inner, -1).clone()
        )
        self.D        = nn.Parameter(torch.ones(d_inner))

        # Out-projection
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)
        self.dropout  = nn.Dropout(dropout)

        # Init dt_proj bias for stable initial step sizes
        nn.init.uniform_(self.dt_proj.bias, -4, -1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, d_model) → (B, L, d_model)"""
        residual = x
        x = self.norm(x)
        B_sz, L, _ = x.shape

        # Split into x-branch and z-gate
        xz       = self.in_proj(x)                          # (B, L, 2*d_inner)
        x_branch = xz[..., :self.d_inner]
        z        = xz[..., self.d_inner:]

        # Depthwise conv (causal: trim right padding)
        x_conv = self.conv1d(x_branch.transpose(1, 2))[:, :, :L]
        x_branch = F.silu(x_conv.transpose(1, 2))           # (B, L, d_inner)

        # SSM parameters from input
        xz2       = self.x_proj(x_branch)                   # (B, L, dt_rank + 2*d_state)
        delta_raw = xz2[..., :self.dt_rank]
        B_mat     = xz2[..., self.dt_rank:self.dt_rank + self.d_state]
        C_mat     = xz2[..., self.dt_rank + self.d_state:]

        delta = F.softplus(self.dt_proj(delta_raw))         # (B, L, d_inner)
        A     = -torch.exp(self.A_log.float())              # (d_inner, d_state) < 0

        # Selective scan
        y = selective_scan(x_branch, delta, A, B_mat, C_mat, self.D)

        # Multiplicative gate
        y = y * F.silu(z)

        return self.dropout(self.out_proj(y)) + residual


# ─────────────────────────────────────────────
#  MARKET MAMBA
# ─────────────────────────────────────────────
class MarketMamba(nn.Module):
    """
    Mamba S6 model for BUY / SELL / HOLD classification.

    Stacks N MambaBlocks over the embedded price sequence.
    The selective state space learns to retain the most decision-
    relevant price history and discard noise — a fundamentally
    different inductive bias from all attention and conv models.
    """

    def __init__(self, config: Optional[MambaConfig] = None):
        super().__init__()
        self.config = config or MambaConfig()

        # Input embedding
        self.input_embed = nn.Sequential(
            nn.Linear(self.config.input_features, self.config.d_model),
            nn.LayerNorm(self.config.d_model),
        )

        # Mamba blocks
        self.blocks = nn.ModuleList([
            MambaBlock(
                d_model  = self.config.d_model,
                d_state  = self.config.d_state,
                d_conv   = self.config.d_conv,
                expand   = self.config.expand,
                dropout  = self.config.dropout,
            )
            for _ in range(self.config.n_layers)
        ])

        self.norm_f = nn.LayerNorm(self.config.d_model)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.config.d_model, self.config.d_model // 2),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.d_model // 2, self.config.num_classes),
        )

        # Confidence head
        self.confidence_head = nn.Sequential(
            nn.Linear(self.config.d_model, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, features) → (batch, d_model)"""
        h = self.input_embed(x)
        for block in self.blocks:
            h = block(h)
        h = self.norm_f(h)
        return h.mean(dim=1)            # global avg pool

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self._encode(x))

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.forward(x), dim=-1)

    def predict_with_confidence(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        pooled = self._encode(x)
        return F.softmax(self.classifier(pooled), dim=-1), self.confidence_head(pooled)
