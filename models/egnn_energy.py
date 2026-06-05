"""
models/egnn_energy.py
=====================
SE(3)-equivariant velocity network with energy conditioning and
Classifier-Free Guidance (CFG) support.

Extends the EGNNScoreNetwork architecture to condition generation on a
per-structure energy label, enabling temperature-controlled sampling:

    τ = 0  →  stable,  compact conformations  (low energy)
    τ = 1  →  transient, extended conformations (high energy)

Architecture
------------
The key change relative to EGNNScoreNetwork is a combined conditioning vector:

    cond_emb = [ t_emb (time_dim) | e_emb (energy_dim) ]   shape (B, cond_dim)

This is passed to each EGNNLayer as the `t_emb` argument — the layer is
unchanged; it simply receives a wider conditioning vector.  The input
projection and EGNNLayer constructors are adjusted to use cond_dim.

Energy input
------------
energy_z : (B,) z-score-normalised energy (mean 0, std 1 over the training set).

    energy_z = (E_raw - E_mean) / E_std

Normalisation is performed *outside* the model (in the training script and
the analysis/sampling scripts).  The model stores no energy statistics.

Classifier-Free Guidance
------------------------
During training, energy conditioning is randomly dropped with probability
`p_drop` (default 0.15).  When dropped, a learned null embedding replaces
the energy embedding.  At inference the caller can either:

  - Pass `energy_z=None` for unconditional generation.
  - Pass energy_z and `guidance_scale > 1` in the sampling loop for
    amplified conditioning:
      v_guided = v_uncond + w * (v_cond - v_uncond)

Temperature mapping
-------------------
Use `EGNNEnergyScoreNetwork.temperature_to_energy_z(τ)` to convert a
user-facing temperature τ ∈ [0, 1] to the normalised energy value:

    e_z = 4.0 * τ - 2.0

This spans ±2σ of the training energy distribution:
    τ=0.0  →  e_z=-2.0  →  ~2nd percentile  (very stable)
    τ=0.5  →  e_z= 0.0  →  ~50th percentile (average)
    τ=1.0  →  e_z=+2.0  →  ~98th percentile (very transient)

Usage
-----
    model = EGNNEnergyScoreNetwork(...)
    # training (energy_z passed; model applies CFG dropout internally):
    v = model(x_t, t_scaled, energy_z=e_z_batch)
    # unconditional inference:
    v = model(x_t, t_scaled, energy_z=None)
    # conditional inference (no guidance amplification):
    e_z = model.temperature_to_energy_z(0.0)
    v = model(x_t, t_scaled, energy_z=torch.full((B,), e_z, device=dev))
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from models.egnn import EGNNLayer, SinusoidalTimestepEmbedding


class EGNNEnergyScoreNetwork(nn.Module):
    """
    SE(3)-equivariant velocity network conditioned on energy (CFG-enabled).

    Drop-in for EGNNScoreNetwork when used with ZeroCoMFlowMatching:
        model(x_t, t) -> (B, N, 3)   ← energy_z defaults to None (unconditional)

    Additional call signature for energy-conditioned generation:
        model(x_t, t, energy_z=e_z)  ← energy_z : (B,) z-normalised scalars

    Config keys (same block as egnn.yaml / flowmatch.yaml):
        model.hidden_dim      → node_dim
        model.n_layers        → number of EGNN layers
        model.time_dim        → time embedding dimension
        model.edge_dim        → edge message dimension  (default 64)
        model.energy_dim      → energy embedding dimension  (new, default 32)
        model.energy_drop_prob → CFG dropout probability    (new, default 0.15)
    """

    def __init__(
        self,
        n_residues:       int   = 10,
        node_dim:         int   = 128,
        edge_dim:         int   = 64,
        time_dim:         int   = 64,
        n_layers:         int   = 5,
        energy_dim:       int   = 32,
        energy_drop_prob: float = 0.15,
    ):
        super().__init__()
        self.n_residues       = n_residues
        self.energy_drop_prob = energy_drop_prob

        cond_dim = time_dim + energy_dim   # combined conditioning vector width

        # ── Time embedding (identical to EGNNScoreNetwork) ────────────────
        self.time_mlp = nn.Sequential(
            SinusoidalTimestepEmbedding(time_dim),
            nn.Linear(time_dim, time_dim * 4),
            nn.SiLU(),
            nn.Linear(time_dim * 4, time_dim),
        )

        # ── Energy embedding ──────────────────────────────────────────────
        # Maps z-score energy scalar → energy_dim vector.
        self.energy_mlp = nn.Sequential(
            nn.Linear(1, energy_dim * 2),
            nn.SiLU(),
            nn.Linear(energy_dim * 2, energy_dim),
        )

        # Learned null embedding: substituted during CFG dropout and
        # unconditional inference.  Initialised to zeros; learned during training
        # to capture the "average" (unconditioned) velocity direction.
        self.null_energy_emb = nn.Parameter(torch.zeros(energy_dim))

        # ── Residue position embedding ────────────────────────────────────
        self.res_embed = nn.Embedding(n_residues, node_dim)

        # ── Input projection — now uses cond_dim ──────────────────────────
        # Input: [ res_embed (node_dim) | cond_emb (cond_dim) ]
        self.input_proj = nn.Sequential(
            nn.Linear(node_dim + cond_dim, node_dim),
            nn.SiLU(),
            nn.Linear(node_dim, node_dim),
        )

        # ── EGNN layers — time_dim parameter = cond_dim ───────────────────
        # EGNNLayer is unchanged; it just receives a wider conditioning vector
        # in place of t_emb.
        self.layers = nn.ModuleList([
            EGNNLayer(node_dim=node_dim, edge_dim=edge_dim, time_dim=cond_dim)
            for _ in range(n_layers)
        ])

        # ── Output gate (identical to EGNNScoreNetwork) ───────────────────
        self.output_gate = nn.Sequential(
            nn.Linear(node_dim, node_dim // 2),
            nn.SiLU(),
            nn.Linear(node_dim // 2, 1),
            nn.Sigmoid(),
        )

        self._init_weights()

    # ── Initialisation ────────────────────────────────────────────────────

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.normal_(self.res_embed.weight, std=0.02)
        # null_energy_emb: keep zeros init (nn.Parameter default)

    # ── Temperature helpers ───────────────────────────────────────────────

    @staticmethod
    def temperature_to_energy_z(tau: float) -> float:
        """
        Map user-facing temperature τ ∈ [0, 1] to z-score-normalised energy.

            τ = 0.0  →  e_z = -2.0  (stable,   ~2nd percentile)
            τ = 0.5  →  e_z =  0.0  (average,  ~50th percentile)
            τ = 1.0  →  e_z = +2.0  (transient, ~98th percentile)

        The sign convention matches the dataset: higher raw energy
        (less negative) → more extended / transient conformation.
        """
        return 4.0 * tau - 2.0

    # ── Forward ───────────────────────────────────────────────────────────

    def forward(self, x_t: Tensor, t: Tensor, energy_z: Tensor = None) -> Tensor:
        """
        x_t      : (B, N, 3)   noisy Cα coordinates (scaled and zero-CoM)
        t        : (B,)         timestep scaled to [0, 999]
        energy_z : (B,) | None  z-score-normalised energy.
                                 None → use null embedding (unconditional).

        Returns v_pred : (B, N, 3)  predicted velocity (equivariant, zero-CoM)
        """
        B, N, _ = x_t.shape

        # ── Time embedding ────────────────────────────────────────────────
        t_emb = self.time_mlp(t)                                 # (B, time_dim)

        # ── Energy embedding (with CFG dropout during training) ───────────
        if energy_z is not None:
            e_emb = self.energy_mlp(energy_z.unsqueeze(-1))      # (B, energy_dim)

            if self.training and self.energy_drop_prob > 0.0:
                drop = torch.rand(B, device=x_t.device) < self.energy_drop_prob
                null = self.null_energy_emb.unsqueeze(0).expand(B, -1)
                e_emb = torch.where(drop.unsqueeze(-1), null, e_emb)
        else:
            # Unconditional: use learned null embedding broadcast over batch
            e_emb = self.null_energy_emb.unsqueeze(0).expand(B, -1)  # (B, energy_dim)

        # ── Combined conditioning vector ──────────────────────────────────
        cond_emb = torch.cat([t_emb, e_emb], dim=-1)             # (B, cond_dim)

        # ── Initial node features: residue embedding + conditioning ───────
        res_idx  = torch.arange(N, device=x_t.device)
        r_emb    = self.res_embed(res_idx)[None].expand(B, -1, -1)   # (B, N, node_dim)
        cond_exp = cond_emb[:, None, :].expand(B, N, -1)             # (B, N, cond_dim)
        h        = self.input_proj(torch.cat([r_emb, cond_exp], dim=-1))  # (B, N, node_dim)

        # ── EGNN forward — cond_emb replaces t_emb in each layer ─────────
        x    = x_t.clone()
        x_in = x_t.clone()

        for layer in self.layers:
            h, x = layer(h, x, cond_emb)

        # ── Output: displacement gated by per-node features ───────────────
        v_pred = (x - x_in) * self.output_gate(h)               # (B, N, 3)

        # Project to zero-CoM (equivariant + removes global translation drift)
        v_pred = v_pred - v_pred.mean(dim=1, keepdim=True)

        return v_pred

    # ── Utilities ─────────────────────────────────────────────────────────

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @torch.no_grad()
    def check_equivariance(self, tol: float = None, n_trials: int = 5) -> bool:
        """Rotation-equivariance sanity check (unconditional path)."""
        self.eval()
        device = next(self.parameters()).device
        if tol is None:
            tol = 0.01 * (self.n_residues / 10)   # scale with N for float32 accumulation
        B, N   = 2, self.n_residues
        errors = []

        for _ in range(n_trials):
            x = torch.randn(B, N, 3, device=device)
            x = x - x.mean(dim=1, keepdim=True)
            t = torch.zeros(B, dtype=torch.long, device=device)

            Q, R_mat = torch.linalg.qr(torch.randn(3, 3, device=device))
            Q = Q * torch.sign(torch.diag(R_mat)).unsqueeze(0)
            if torch.det(Q) < 0:
                Q[:, 0] *= -1

            Rx    = (Q @ x.reshape(B, N, 3, 1)).squeeze(-1)
            v_x   = self(x,  t, energy_z=None)
            v_Rx  = self(Rx, t, energy_z=None)
            R_v   = (Q @ v_x.reshape(B, N, 3, 1)).squeeze(-1)
            errors.append(((v_Rx - R_v).norm() / (v_x.norm() + 1e-8)).item())

        mean_err = sum(errors) / len(errors)
        ok       = mean_err < tol
        print(f"  Equivariance check ({n_trials} trials): mean_rel_err={mean_err:.2e}  "
              f"[{'PASS' if ok else 'FAIL'}]")
        return ok
