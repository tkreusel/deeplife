"""
models/torsion_net.py
=====================
MLP velocity network for TorsionFlow.

Predicts angular velocities (dθ/dt, dφ/dt) in internal coordinate space
from the current torsional state and time. SE(3) invariance is structural —
internal coordinates are invariant to rigid rotations and translations, so
no equivariant architecture is required.

Interface
---------
    pred_theta, pred_phi = model(theta, phi, t_scaled, energy_z=None)

    theta    : (B, 8)   bond angles [rad]
    phi      : (B, 7)   dihedral angles [rad]
    t_scaled : (B,)     time scaled to [0, 999] (matches SinusoidalTimestepEmbedding)
    energy_z : (B,)     z-normalised energy label, or None for unconditional

    pred_theta : (B, 8)  predicted bond-angle velocities [rad/time]
    pred_phi   : (B, 7)  predicted dihedral angular velocities [rad/time]

Internal coordinate features
-----------------------------
    theta : represented as scalars (no periodicity — range is (0, π))
    phi   : represented as (sin φ, cos φ) — unambiguous on S¹ for the network

Input vector: [theta(8) | sin(phi)(7) | cos(phi)(7)] = 22 geometric dims
Plus time embedding (64 dims) and energy embedding (32 dims).

Energy conditioning
-------------------
Identical CFG setup to SE3FlowEnergyNet:
    - Learned null embedding replaces energy when dropped or unconditional
    - Dropout probability: energy_drop_prob (default 0.15)
    - CFG guidance at sampling: v_guided = v_uncond + scale * (v_cond - v_uncond)

model_type in config: "torsion_flow_energy"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from models.egnn import SinusoidalTimestepEmbedding

N_ANGLES    = 8
N_DIHEDRALS = 7
IC_DIM      = N_ANGLES + 2 * N_DIHEDRALS   # 8 + 7 + 7 = 22


class TorsionFlowNet(nn.Module):
    """
    4-layer MLP predicting OT-CFM angular velocities in internal coordinate space.

    Parameters
    ----------
    hidden_dim       : width of each hidden layer (default 256)
    n_layers         : number of hidden layers (default 4)
    time_dim         : sinusoidal time embedding dimension (default 64)
    energy_dim       : energy embedding dimension (default 32)
    energy_drop_prob : CFG dropout probability — drop energy conditioning with this
                       probability during training to enable guidance at inference
    """

    def __init__(
        self,
        hidden_dim:       int   = 256,
        n_layers:         int   = 4,
        time_dim:         int   = 64,
        energy_dim:       int   = 32,
        energy_drop_prob: float = 0.15,
    ):
        super().__init__()
        self.energy_drop_prob = energy_drop_prob
        self.energy_dim       = energy_dim
        self.time_dim         = time_dim

        # ── Time embedding ──────────────────────────────────────────────────
        self.time_emb = SinusoidalTimestepEmbedding(time_dim)

        # ── Energy embedding ────────────────────────────────────────────────
        self.energy_proj = nn.Linear(1, energy_dim)
        self.null_energy = nn.Parameter(torch.zeros(energy_dim))

        # ── Input projection ────────────────────────────────────────────────
        in_dim = IC_DIM + time_dim + energy_dim   # 22 + 64 + 32 = 118
        self.input_proj = nn.Linear(in_dim, hidden_dim)

        # ── Hidden layers ───────────────────────────────────────────────────
        layers: list[nn.Module] = []
        for _ in range(n_layers):
            layers += [
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
            ]
        self.hidden = nn.Sequential(*layers)

        # ── Output heads ────────────────────────────────────────────────────
        self.out_theta = nn.Linear(hidden_dim, N_ANGLES)      # dθ/dt per bond angle
        self.out_phi   = nn.Linear(hidden_dim, N_DIHEDRALS)   # dφ/dt per dihedral

        self._init_weights()

    def _init_weights(self):
        # Zero-initialize output heads so the network starts near the identity flow
        nn.init.zeros_(self.out_theta.weight)
        nn.init.zeros_(self.out_theta.bias)
        nn.init.zeros_(self.out_phi.weight)
        nn.init.zeros_(self.out_phi.bias)

    # ── Forward ─────────────────────────────────────────────────────────────

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

        # ── Internal coordinate features ─────────────────────────────────
        # Bond angles as scalars; dihedrals as (sin, cos) to avoid discontinuity
        ic = torch.cat([theta, torch.sin(phi), torch.cos(phi)], dim=-1)  # (B, 22)

        # ── Time ──────────────────────────────────────────────────────────
        t_emb = self.time_emb(t_scaled)   # (B, time_dim)

        # ── Energy (CFG dropout) ──────────────────────────────────────────
        null = self.null_energy.unsqueeze(0).expand(B, -1)   # (B, energy_dim)
        if energy_z is None:
            e_emb = null
        else:
            e_emb = self.energy_proj(energy_z.unsqueeze(-1))   # (B, energy_dim)
            if self.training and self.energy_drop_prob > 0:
                drop_mask = torch.rand(B, device=theta.device) < self.energy_drop_prob
                e_emb = torch.where(drop_mask.unsqueeze(-1), null, e_emb)

        # ── MLP ───────────────────────────────────────────────────────────
        h = self.input_proj(torch.cat([ic, t_emb, e_emb], dim=-1))   # (B, hidden_dim)
        h = self.hidden(h)

        return self.out_theta(h), self.out_phi(h)   # (B, 8), (B, 7)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
