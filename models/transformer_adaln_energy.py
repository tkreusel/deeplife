"""
models/transformer_adaln_energy.py
====================================
AdaLN Transformer score network with energy-based Classifier-Free Guidance (CFG).

Combines two orthogonal improvements:
  - AdaLN-Zero timestep conditioning (better denoising at every noise level)
  - Energy CFG (steers generation toward compact/extended conformations)

The conditioning vector c fed to every AdaLN block carries both signals:

    c = time_emb(t) + energy_emb(energy_z)     shape (B, hidden_dim)

When energy_z is dropped (training CFG dropout) or absent (unconditional
inference), a learned null embedding is used instead of energy_emb.

Temperature mapping (same convention as EGNNEnergyScoreNetwork):
    τ = 0.0  →  e_z = -2.0   compact / folded   (~2nd energy percentile)
    τ = 0.5  →  e_z =  0.0   average
    τ = 1.0  →  e_z = +2.0   extended / transient (~98th percentile)

CFG guidance at inference:
    ε_guided = ε_uncond + guidance_scale * (ε_cond - ε_uncond)

    Call forward() twice per step: once with energy_z, once with energy_z=None.

Usage:
    model_type: transformer_adaln_energy   (in config yaml)
"""

import torch
import torch.nn as nn

from models.baseline import SinusoidalTimestepEmbedding
from models.transformer_adaln import AdaLNBlock


class AdaLNEnergyTransformerScoreNetwork(nn.Module):
    """
    AdaLN Transformer + energy CFG.

    Input:  x_t (B, N, 3), t (B,), energy_z (B,) or None
    Output: (B, N, 3)  predicted noise
    """

    def __init__(
        self,
        n_residues:       int   = 10,
        hidden_dim:       int   = 256,
        n_heads:          int   = 8,
        n_layers:         int   = 6,
        time_dim:         int   = 64,
        dropout:          float = 0.1,
        energy_drop_prob: float = 0.15,
    ):
        super().__init__()
        self.n_residues       = n_residues
        self.energy_drop_prob = energy_drop_prob

        # shared time MLP — output goes into AdaLN blocks as conditioning, not added to tokens
        self.time_mlp = nn.Sequential(
            SinusoidalTimestepEmbedding(time_dim),
            nn.Linear(time_dim, time_dim * 4),
            nn.SiLU(),
            nn.Linear(time_dim * 4, hidden_dim),
        )

        # energy MLP: scalar z-score → hidden_dim
        self.energy_mlp = nn.Sequential(
            nn.Linear(1, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

        # learned null embedding: used when energy_z=None or dropped by CFG
        # initialised to zeros; learns to represent the unconditional average
        self.null_energy_emb = nn.Parameter(torch.zeros(hidden_dim))

        self.input_proj = nn.Linear(3, hidden_dim)
        self.pos_embed  = nn.Embedding(n_residues, hidden_dim)

        self.blocks = nn.ModuleList([
            AdaLNBlock(hidden_dim, n_heads, dropout=dropout)
            for _ in range(n_layers)
        ])

        self.norm_out         = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.final_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 2 * hidden_dim, bias=True),
        )
        nn.init.zeros_(self.final_modulation[-1].weight)
        nn.init.zeros_(self.final_modulation[-1].bias)

        self.out_proj = nn.Linear(hidden_dim, 3)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    @staticmethod
    def temperature_to_energy_z(tau: float) -> float:
        """
        Map τ ∈ [0, 1] to z-score energy.
            τ=0 → e_z=-2  (compact/folded)
            τ=1 → e_z=+2  (extended/transient)
        """
        return 4.0 * tau - 2.0

    def forward(
        self,
        x_t:      torch.Tensor,
        t:        torch.Tensor,
        energy_z: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        x_t      : (B, N, 3)
        t        : (B,)        diffusion timestep
        energy_z : (B,) | None z-score-normalised energy; None → unconditional
        """
        B, N, _ = x_t.shape
        device  = x_t.device

        # ── Conditioning vector ───────────────────────────────────────────────
        c = self.time_mlp(t)                                     # (B, D)

        if energy_z is not None:
            e_emb = self.energy_mlp(energy_z.unsqueeze(-1))      # (B, D)
            if self.training and self.energy_drop_prob > 0.0:
                drop = torch.rand(B, device=device) < self.energy_drop_prob
                null = self.null_energy_emb.unsqueeze(0).expand(B, -1)
                e_emb = torch.where(drop.unsqueeze(-1), null, e_emb)
        else:
            e_emb = self.null_energy_emb.unsqueeze(0).expand(B, -1)

        c = c + e_emb                                            # (B, D)

        # ── Token sequence ────────────────────────────────────────────────────
        h = self.input_proj(x_t)                                 # (B, N, D)
        h = h + self.pos_embed(torch.arange(N, device=device))  # (B, N, D)

        for block in self.blocks:
            h = block(h, c)                                      # (B, N, D)

        # ── Final modulated norm + output projection ──────────────────────────
        shift, scale = self.final_modulation(c).chunk(2, dim=-1)
        h = self.norm_out(h)
        h = h * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

        return self.out_proj(h)                                  # (B, N, 3)
