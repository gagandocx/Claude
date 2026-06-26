"""
=============================================================
  Python ML Bridge - xLSTM  (Extended LSTM)
  Sepp Hochreiter et al., JKU Linz — https://arxiv.org/abs/2405.04517
  The original LSTM inventor rewriting LSTM from scratch (2024).

  Why xLSTM beats standard LSTM for XAUUSD scalping:
  ────────────────────────────────────────────────────
  Standard LSTM uses SCALAR memory cells and SIGMOID gates.
  Problems for financial time series:
    • Sigmoid saturates → vanishing gradient on long sequences
    • Scalar memory = limited capacity per cell
    • Cannot selectively "write" specific patterns to memory

  xLSTM introduces the mLSTM (matrix memory LSTM):
    • MATRIX memory cell  C ∈ ℝ^(d×d)  — exponentially more capacity
    • EXPONENTIAL gates   i_t = exp(·)  — never saturates, no vanishing
    • Covariance update:  C_t = f_t·C_{t-1} + i_t·(v_t⊗k_t^T)
      → stores an outer-product "key-value" association at each bar
      → retrieval:  h_t = C_t·q_t / max(|n_t^T·q_t|, 1)
    • Stabilised via log-space running max  (no overflow)

  In plain English:
    At bar 64, the matrix memory can simultaneously hold associations
    from bar 1 (a swing high), bar 32 (a breakout), and bar 58
    (a momentum shift). Standard LSTM would have overwritten most
    of those with intermediate noise.

  Architecture:
    Input (batch, 64, 46)
    → Linear embedding → (batch, 64, d_model)
    → N × mLSTMBlock  (mLSTMCell + LayerNorm + FFN + residual)
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
from config.settings import xLSTMConfig


# ─────────────────────────────────────────────
#  mLSTM CELL
# ─────────────────────────────────────────────
class mLSTMCell(nn.Module):
    """
    Multi-head matrix-memory LSTM cell.

    State per head:
        C  ∈ ℝ^(Hd×Hd)  — matrix memory (key-value associations)
        n  ∈ ℝ^Hd        — running normaliser
        m  ∈ ℝ            — running log-space max (numerical stability)

    Update equations (per head, per timestep):
        q_t = Wq · x_t / √Hd           query
        k_t = Wk · x_t / √Hd           key
        v_t = Wv · x_t                  value
        log_i_t = wi · x_t + bi         input gate (log-space)
        log_f_t = wf · x_t + bf         forget gate (log-space)
        m_t = max(log_f_t + m_{t-1}, log_i_t)   running max
        f̃_t = exp(log_f_t + m_{t-1} - m_t)       stabilised forget
        ĩ_t = exp(log_i_t - m_t)                  stabilised input
        C_t = f̃_t · C_{t-1} + ĩ_t · (v_t ⊗ k_t^T)   matrix update
        n_t = f̃_t · n_{t-1} + ĩ_t · k_t                normaliser
        h_t = C_t @ q_t / max(|n_t · q_t|, 1)          retrieval
        o_t = σ(Wo · x_t)               output gate
        y_t = o_t ⊙ h_t (reshaped to d_model)
    """

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model  = d_model
        self.n_heads  = n_heads
        self.head_dim = d_model // n_heads

        # QKV projections (shared across all heads via reshape)
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)

        # Gate projections (one scalar per head)
        self.w_i = nn.Linear(d_model, n_heads)   # input gate
        self.w_f = nn.Linear(d_model, n_heads)   # forget gate
        self.w_o = nn.Linear(d_model, d_model)   # output gate

        self._init_weights()

    def _init_weights(self):
        for lin in [self.w_q, self.w_k, self.w_v]:
            nn.init.xavier_uniform_(lin.weight)
        nn.init.zeros_(self.w_f.bias)
        # Initialise forget bias to ~3 so forget gates start near 1 (keep memory)
        nn.init.constant_(self.w_f.bias, 3.0)

    def forward(
        self, x: torch.Tensor
    ) -> torch.Tensor:
        """
        Process full sequence at once (sequential scan internally).

        Args:
            x: (B, L, d_model)
        Returns:
            y: (B, L, d_model)
        """
        B, L, d = x.shape
        H, Hd   = self.n_heads, self.head_dim
        scale   = Hd ** -0.5

        # Project QKV for the full sequence at once
        q = self.w_q(x).reshape(B, L, H, Hd)   # (B,L,H,Hd)
        k = self.w_k(x).reshape(B, L, H, Hd) * scale
        v = self.w_v(x).reshape(B, L, H, Hd)

        # Gate logits
        log_i = self.w_i(x)    # (B, L, H)
        log_f = self.w_f(x)    # (B, L, H)
        o     = torch.sigmoid(self.w_o(x))  # (B, L, d_model)

        # Sequential scan over time
        C = x.new_zeros(B, H, Hd, Hd)   # matrix memory
        n = x.new_zeros(B, H, Hd)        # normaliser
        m = x.new_full((B, H), -1e9)     # running log-max

        ys = []
        for t in range(L):
            # ── stabilised gates ──────────────────────────────────────────
            m     = torch.maximum(log_f[:, t] + m, log_i[:, t])
            f_t   = torch.exp(log_f[:, t] + (m - torch.maximum(log_f[:, t] + m, log_i[:, t])) + m - m)
            # Cleaner: recompute without mutation
            m_new = torch.maximum(log_f[:, t] + m.clone(), log_i[:, t])
            f_t   = torch.exp(log_f[:, t] + m - m_new)   # (B, H)
            i_t   = torch.exp(log_i[:, t] - m_new)        # (B, H)
            m     = m_new

            # ── matrix memory update ──────────────────────────────────────
            # outer product v ⊗ k^T : (B,H,Hd,Hd)
            outer = torch.einsum("bhd,bhe->bhde", v[:, t], k[:, t])
            C     = f_t[:, :, None, None] * C + i_t[:, :, None, None] * outer

            # ── normaliser update ─────────────────────────────────────────
            n = f_t.unsqueeze(-1) * n + i_t.unsqueeze(-1) * k[:, t]

            # ── retrieval: h = C @ q ──────────────────────────────────────
            h_raw  = torch.einsum("bhde,bhe->bhd", C, q[:, t])  # (B,H,Hd)
            denom  = torch.einsum("bhd,bhd->bh", n, q[:, t]).abs().clamp(min=1)
            h_norm = h_raw / denom.unsqueeze(-1)                 # (B,H,Hd)

            # ── output gate ───────────────────────────────────────────────
            y_t = o[:, t] * h_norm.reshape(B, d)   # (B, d_model)
            ys.append(y_t)

        return torch.stack(ys, dim=1)   # (B, L, d_model)


# ─────────────────────────────────────────────
#  mLSTM BLOCK
# ─────────────────────────────────────────────
class mLSTMBlock(nn.Module):
    """
    xLSTM block: mLSTMCell + pre-norm + FFN + residual connections.
    Same macro-structure as a Transformer block but with mLSTM
    replacing multi-head attention.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.cell  = mLSTMCell(d_model, n_heads)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn   = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.cell(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


# ─────────────────────────────────────────────
#  MARKET xLSTM
# ─────────────────────────────────────────────
class MarketXLSTM(nn.Module):
    """
    xLSTM model for BUY / SELL / HOLD classification.

    The matrix memory cells give this model significantly higher capacity
    than the standard BiLSTM in the stack. While BiLSTM processes the
    sequence forward+backward with scalar cells, xLSTM processes it
    once with matrix cells that hold exponentially more associations.
    """

    def __init__(self, config: Optional[xLSTMConfig] = None):
        super().__init__()
        self.config = config or xLSTMConfig()
        d = self.config.d_model

        self.embed = nn.Sequential(
            nn.Linear(self.config.input_features, d),
            nn.LayerNorm(d),
        )
        self.blocks = nn.ModuleList([
            mLSTMBlock(d, self.config.n_heads, self.config.dropout)
            for _ in range(self.config.n_layers)
        ])
        self.norm_f = nn.LayerNorm(d)

        self.classifier = nn.Sequential(
            nn.Linear(d, d // 2),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(d // 2, self.config.num_classes),
        )
        self.confidence_head = nn.Sequential(
            nn.Linear(d, 32), nn.GELU(), nn.Linear(32, 1), nn.Sigmoid(),
        )

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        h = self.embed(x)
        for block in self.blocks:
            h = block(h)
        return self.norm_f(h).mean(dim=1)   # global avg pool → (B, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self._encode(x))

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.forward(x), dim=-1)

    def predict_with_confidence(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        pooled = self._encode(x)
        return F.softmax(self.classifier(pooled), dim=-1), self.confidence_head(pooled)
