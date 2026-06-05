"""
models/transformer_adaln.py
============================
Transformer score network with Adaptive LayerNorm-Zero (AdaLN-Zero) timestep
conditioning, following the DiT architecture (Peebles & Xie, 2022).

Key difference from TransformerScoreNetwork in baseline.py:
  - baseline: adds t_emb once to token embeddings at the input
  - here:     t_emb is NOT added to tokens; instead each block derives its own
              scale/shift/gate parameters from t_emb via a small linear head,
              conditioning both LayerNorms independently at every layer

The gate parameters are zero-initialised, so every block starts as an identity
mapping and gradient signal flows cleanly from the start of training.

Usage:
    model_type: transformer_adaln   (in config yaml)
"""

import torch
import torch.nn as nn

from models.baseline import SinusoidalTimestepEmbedding


class AdaLNBlock(nn.Module):
    """
    Pre-norm Transformer block with AdaLN-Zero conditioning.

    Each block receives the global time conditioning vector c ~ (B, D) and
    computes six modulation scalars per token dimension:
        shift_attn, scale_attn, gate_attn,
        shift_ff,   scale_ff,   gate_ff

    The LayerNorms have no learned affine parameters (elementwise_affine=False)
    because AdaLN supplies them. Gates start at zero → block is identity at init.
    """

    def __init__(self, dim: int, n_heads: int, ff_mult: int = 4, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)
        self.attn  = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.ff    = nn.Sequential(
            nn.Linear(dim, dim * ff_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * ff_mult, dim),
            nn.Dropout(dropout),
        )
        # SiLU + linear → 6 * dim modulation params
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim, bias=True),
        )
        # zero-init: gates = 0 at start, block acts as identity residual
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        x : (B, N, D)  token sequence
        c : (B, D)     time conditioning (same for all tokens in a structure)
        """
        mods = self.adaLN_modulation(c)                         # (B, 6*D)
        shift_a, scale_a, gate_a, shift_f, scale_f, gate_f = mods.chunk(6, dim=-1)

        # attention branch
        h = self.norm1(x)
        h = h * (1 + scale_a.unsqueeze(1)) + shift_a.unsqueeze(1)
        h, _ = self.attn(h, h, h)
        x = x + gate_a.unsqueeze(1) * h

        # feed-forward branch
        h = self.norm2(x)
        h = h * (1 + scale_f.unsqueeze(1)) + shift_f.unsqueeze(1)
        h = self.ff(h)
        x = x + gate_f.unsqueeze(1) * h

        return x


class AdaLNTransformerScoreNetwork(nn.Module):
    """
    Drop-in replacement for TransformerScoreNetwork with AdaLN-Zero conditioning.

    Input:  x_t (B, N, 3),  t (B,)
    Output:     (B, N, 3)   predicted noise
    """

    def __init__(
        self,
        n_residues: int   = 10,
        hidden_dim: int   = 256,
        n_heads:    int   = 8,
        n_layers:   int   = 6,
        time_dim:   int   = 64,
        dropout:    float = 0.1,
    ):
        super().__init__()
        self.n_residues = n_residues

        # shared time MLP: scalar t → (B, hidden_dim) conditioning vector
        # output goes to each block's adaLN_modulation, NOT added to tokens
        self.time_mlp = nn.Sequential(
            SinusoidalTimestepEmbedding(time_dim),
            nn.Linear(time_dim, time_dim * 4),
            nn.SiLU(),
            nn.Linear(time_dim * 4, hidden_dim),
        )

        self.input_proj = nn.Linear(3, hidden_dim)
        self.pos_embed  = nn.Embedding(n_residues, hidden_dim)

        self.blocks = nn.ModuleList([
            AdaLNBlock(hidden_dim, n_heads, dropout=dropout)
            for _ in range(n_layers)
        ])

        # final modulated norm — same AdaLN pattern as each block
        self.norm_out          = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.final_modulation  = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 2 * hidden_dim, bias=True),
        )
        nn.init.zeros_(self.final_modulation[-1].weight)
        nn.init.zeros_(self.final_modulation[-1].bias)

        self.out_proj = nn.Linear(hidden_dim, 3)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        B, N, _ = x_t.shape
        device  = x_t.device

        h = self.input_proj(x_t)                               # (B, N, D)
        h = h + self.pos_embed(torch.arange(N, device=device)) # (B, N, D)

        c = self.time_mlp(t)                                   # (B, D) — not added to h

        for block in self.blocks:
            h = block(h, c)                                    # (B, N, D)

        # final norm with AdaLN
        shift, scale = self.final_modulation(c).chunk(2, dim=-1)
        h = self.norm_out(h)
        h = h * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

        return self.out_proj(h)                                # (B, N, 3)
