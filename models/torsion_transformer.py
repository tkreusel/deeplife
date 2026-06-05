"""
models/torsion_transformer.py
==============================
Transformer velocity network for TorsionFlow v2.

Replaces the global MLP (TorsionFlowNet) with a self-attention backbone that
models inter-residue correlations between dihedral angles.  This is the key
architectural change from v1: the MLP pooled all 15 DOFs into a single vector,
preventing the model from learning that e.g. dihedrals φ₃–φ₅ must co-vary to
form the β-hairpin turn.

Architecture
------------
15 tokens: 8 bond-angle tokens (θ₀…θ₇) followed by 7 dihedral tokens (φ₀…φ₆).
Each token is projected from its (sin, cos) representation to d_model.  A
learnable position embedding (per token index 0–14) encodes chain order.  Time
and energy signals are broadcast-added to every token before self-attention.

    tokens  ∈ (B, 15, 2)          (sin/cos of each DOF)
        ↓  token_proj
    h   ∈ (B, 15, d_model)
        + pos_emb[0..14]           learnable, per token
        + time_proj(time_emb(t))   broadcast over sequence
        + energy_proj2(e_emb)      broadcast over sequence; CFG dropout
        ↓  TransformerEncoder (pre-LN, GELU, n_layers × n_heads)
    h   ∈ (B, 15, d_model)
        ↓  out_theta(h[:, :8])     → (B, 8) bond-angle velocities
        ↓  out_phi  (h[:, 8:])     → (B, 7) dihedral velocities

Interface
---------
Identical to TorsionFlowNet so the same TorsionalFlowMatching framework and
train_torsion.py training script work without modification:

    pred_theta, pred_phi = model(theta, phi, t_scaled, energy_z=None)

Energy conditioning
-------------------
Identical CFG setup to TorsionFlowNet (and SE3FlowEnergyNet):
    - Learned null embedding replaces energy when dropped or None
    - Dropout probability: energy_drop_prob (default 0.15)
    - At inference: v_guided = v_uncond + scale * (v_cond - v_uncond)

model_type in config: "torsion_transformer_energy"
"""

import torch
import torch.nn as nn
from torch import Tensor

from models.egnn import SinusoidalTimestepEmbedding
from models.internal_coords import N_ANGLES, N_DIHEDRALS

N_TOKENS = N_ANGLES + N_DIHEDRALS   # 15


class TorsionTransformerNet(nn.Module):
    """
    Transformer velocity network for TorsionFlow.

    Parameters
    ----------
    d_model          : hidden dimension (default 256)
    n_heads          : number of attention heads (default 4)
    n_layers         : number of transformer encoder layers (default 6)
    time_dim         : sinusoidal time embedding dimension (default 64)
    energy_dim       : energy embedding dimension (default 32)
    energy_drop_prob : CFG dropout probability (default 0.15)
    dropout          : attention + FF dropout (default 0.1)
    """

    def __init__(
        self,
        d_model:          int   = 256,
        n_heads:          int   = 4,
        n_layers:         int   = 6,
        time_dim:         int   = 64,
        energy_dim:       int   = 32,
        energy_drop_prob: float = 0.15,
        dropout:          float = 0.1,
    ):
        super().__init__()
        self.energy_drop_prob = energy_drop_prob
        self.energy_dim       = energy_dim
        self.d_model          = d_model

        # ── Time embedding ──────────────────────────────────────────────────
        self.time_emb  = SinusoidalTimestepEmbedding(time_dim)
        self.time_proj = nn.Linear(time_dim, d_model)

        # ── Energy embedding ────────────────────────────────────────────────
        self.energy_proj  = nn.Linear(1, energy_dim)
        self.energy_proj2 = nn.Linear(energy_dim, d_model)
        self.null_energy  = nn.Parameter(torch.zeros(energy_dim))

        # ── Per-token: (sin, cos) → d_model ─────────────────────────────────
        self.token_proj = nn.Linear(2, d_model)

        # ── Positional embedding (one per token, 0..14) ─────────────────────
        self.pos_emb = nn.Embedding(N_TOKENS, d_model)

        # ── Transformer encoder (pre-LN for stable training) ────────────────
        encoder_layer = nn.TransformerEncoderLayer(
            d_model        = d_model,
            nhead          = n_heads,
            dim_feedforward = 4 * d_model,
            dropout        = dropout,
            activation     = 'gelu',
            batch_first    = True,
            norm_first     = True,    # pre-LN: more stable than post-LN
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers,
            enable_nested_tensor=False,   # avoid warning with small batch sizes
        )

        # ── Output heads (separate for θ and φ) ─────────────────────────────
        self.out_theta = nn.Linear(d_model, 1)   # (B, 8, d_model) → (B, 8, 1)
        self.out_phi   = nn.Linear(d_model, 1)   # (B, 7, d_model) → (B, 7, 1)

        self._init_weights()

    def _init_weights(self):
        # Zero-init output heads so the network starts near the identity flow
        nn.init.zeros_(self.out_theta.weight)
        nn.init.zeros_(self.out_theta.bias)
        nn.init.zeros_(self.out_phi.weight)
        nn.init.zeros_(self.out_phi.bias)

    # ── Forward ──────────────────────────────────────────────────────────────

    def forward(
        self,
        theta:    Tensor,                # (B, 8)  bond angles [rad]
        phi:      Tensor,                # (B, 7)  dihedral angles [rad]
        t_scaled: Tensor,                # (B,)    time in [0, 999]
        energy_z: Tensor | None = None,  # (B,)    z-normalised energy or None
    ) -> tuple[Tensor, Tensor]:
        """
        Returns:
            pred_theta : (B, 8)  bond-angle velocity [rad/time]
            pred_phi   : (B, 7)  dihedral angular velocity [rad/time]
        """
        B = theta.shape[0]

        # ── Build (sin, cos) token features ──────────────────────────────────
        # Both θ and φ are well-represented by (sin, cos); no discontinuity.
        theta_sc = torch.stack([theta.sin(), theta.cos()], dim=-1)  # (B, 8, 2)
        phi_sc   = torch.stack([phi.sin(),   phi.cos()],   dim=-1)  # (B, 7, 2)
        tokens   = torch.cat([theta_sc, phi_sc], dim=1)             # (B, 15, 2)

        # ── Project tokens to d_model ──────────────────────────────────────
        h = self.token_proj(tokens)   # (B, 15, d_model)

        # ── Positional embedding ───────────────────────────────────────────
        pos_ids = torch.arange(N_TOKENS, device=theta.device)
        h = h + self.pos_emb(pos_ids).unsqueeze(0)   # (B, 15, d_model)

        # ── Time conditioning (broadcast to all tokens) ────────────────────
        t_emb = self.time_proj(self.time_emb(t_scaled))   # (B, d_model)
        h = h + t_emb.unsqueeze(1)                         # (B, 15, d_model)

        # ── Energy conditioning (CFG dropout) ─────────────────────────────
        null = self.null_energy.unsqueeze(0).expand(B, -1)   # (B, energy_dim)
        if energy_z is None:
            e_emb = null
        else:
            e_emb = self.energy_proj(energy_z.unsqueeze(-1))   # (B, energy_dim)
            if self.training and self.energy_drop_prob > 0:
                drop_mask = torch.rand(B, device=theta.device) < self.energy_drop_prob
                e_emb = torch.where(drop_mask.unsqueeze(-1), null, e_emb)

        e_h = self.energy_proj2(e_emb)   # (B, d_model)
        h = h + e_h.unsqueeze(1)          # (B, 15, d_model)

        # ── Transformer encoder ────────────────────────────────────────────
        h = self.transformer(h)   # (B, 15, d_model)

        # ── Output heads ──────────────────────────────────────────────────
        pred_theta = self.out_theta(h[:, :N_ANGLES]).squeeze(-1)   # (B, 8)
        pred_phi   = self.out_phi(h[:, N_ANGLES:]).squeeze(-1)     # (B, 7)

        return pred_theta, pred_phi

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
