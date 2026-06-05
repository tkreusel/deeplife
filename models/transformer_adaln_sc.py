"""
models/transformer_adaln_sc.py
================================
AdaLN Transformer with self-conditioning, energy CFG, and zero-CoM diffusion.

Extends AdaLNEnergyTransformerScoreNetwork with one addition:

    Self-conditioning
    -----------------
    At each denoising step the model receives its own x₀ prediction from the
    *previous* step as an additional input.  This lets the model iteratively
    refine its estimate rather than starting blind at every step.

    Implementation (Chen et al., "Analog Bits", 2022):
      - Extra projection:  self_cond_proj : Linear(3 → hidden_dim),  zero-init
      - forward() accepts  x0_self_cond (B, N, 3) | None
        When None (first step, or unconditional), zeros are used
      - Training: 50 % of batches use a stop-gradient preliminary x₀ as input;
        the other 50 % use zeros (teaches both code paths)

    Zero-CoM diffusion
    ------------------
    Pair this model with ZeroCoMGaussianDiffusion.  The noise schedule and loss
    are unchanged; zero-CoM projection is applied to the noise in q_sample and
    noise_pred in ddim_sample_sc (see diffusion_zerocom.py).

Usage:
    model_type: transformer_adaln_sc   (in config yaml)
"""

import torch
import torch.nn as nn

from models.transformer_adaln_energy import AdaLNEnergyTransformerScoreNetwork


class AdaLNSCScoreNetwork(AdaLNEnergyTransformerScoreNetwork):
    """
    AdaLN Transformer + energy CFG + self-conditioning.

    Input:  x_t          (B, N, 3)
            t            (B,)
            energy_z     (B,) | None
            x0_self_cond (B, N, 3) | None   ← previous x₀ prediction
    Output: (B, N, 3)  predicted noise
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # infer hidden_dim from the already-built input_proj weight
        hidden_dim = self.input_proj.out_features

        # projects previous x₀ estimate into token space — zero-init so the
        # model starts identically to the non-self-conditioned version
        self.self_cond_proj = nn.Linear(3, hidden_dim)
        nn.init.zeros_(self.self_cond_proj.weight)
        nn.init.zeros_(self.self_cond_proj.bias)

    def forward(
        self,
        x_t:          torch.Tensor,
        t:            torch.Tensor,
        energy_z:     torch.Tensor = None,
        x0_self_cond: torch.Tensor = None,
    ) -> torch.Tensor:
        B, N, _ = x_t.shape
        device  = x_t.device

        # ── Conditioning vector (time + energy) ───────────────────────────────
        c = self.time_mlp(t)

        if energy_z is not None:
            e_emb = self.energy_mlp(energy_z.unsqueeze(-1))
            if self.training and self.energy_drop_prob > 0.0:
                drop  = torch.rand(B, device=device) < self.energy_drop_prob
                null  = self.null_energy_emb.unsqueeze(0).expand(B, -1)
                e_emb = torch.where(drop.unsqueeze(-1), null, e_emb)
        else:
            e_emb = self.null_energy_emb.unsqueeze(0).expand(B, -1)

        c = c + e_emb

        # ── Token sequence ────────────────────────────────────────────────────
        h = self.input_proj(x_t)
        h = h + self.pos_embed(torch.arange(N, device=device))

        # self-conditioning: add projected x₀ from the previous denoising step
        if x0_self_cond is None:
            x0_self_cond = torch.zeros_like(x_t)
        h = h + self.self_cond_proj(x0_self_cond)

        # ── AdaLN transformer blocks ──────────────────────────────────────────
        for block in self.blocks:
            h = block(h, c)

        shift, scale = self.final_modulation(c).chunk(2, dim=-1)
        h = self.norm_out(h)
        h = h * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

        return self.out_proj(h)
