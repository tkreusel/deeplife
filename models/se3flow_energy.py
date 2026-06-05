"""
models/se3flow_energy.py
========================
SE(3)-equivariant flow matching velocity network with energy conditioning.

This is the velocity network used inside ZeroCoMFlowMatching (OT-CFM).
It predicts the velocity field  v_θ(x_t, t, e_z)  used to integrate the ODE:
    dx/dt = v_θ(x_t, t, e_z)   from t=0 (noise) to t=1 (data)

It is NOT a diffusion model and NOT a score network — it predicts velocities
for flow matching, which is a fundamentally different framework.

Architecture improvements over EGNNEnergyScoreNetwork (models/egnn_energy.py)
------------------------------------------------------------------------------
1. RBF distance encoding  — replaces the single d² scalar in edge messages
   with 16 Gaussian basis functions (σ=0.5 Å, centres 0.5–15.5 Å).
   Verified: cosine similarity between RBF vectors at ideal bond (3.832 Å)
   and ±0.5 Å violation is 0.67–0.90 — clearly discriminable vs d² scalar.

2. Sequence-separation embedding  — adds a 4-dim embedding of |i−j| to each
   edge message, encoding chain topology directly. Bonded pairs (|i−j|=1)
   get a distinct learned embedding from non-bonded pairs, giving the MLP
   an explicit prior that consecutive Cα atoms have a nearly rigid bond.

3. Larger default capacity  — hidden_dim=192, edge_dim=96, n_layers=7
   (~1.27M params) for better geometry learning.

4. Energy conditioning (identical to EGNNEnergyScoreNetwork):
   cond_emb = [t_emb | e_emb]  injected into every message-passing layer.
   Classifier-Free Guidance (CFG) dropout during training (p_drop=0.15).
   Temperature mapping: τ ∈ [0,1] → e_z = 4τ − 2 (±2σ of training dist).

SE(3) equivariance is preserved:
   - RBF depends only on ||xᵢ−xⱼ|| (rotation-invariant scalar)
   - sep_emb depends only on |i−j| (index-based, not coordinates)
   - coord update is a weighted sum of equivariant difference vectors (xᵢ−xⱼ)

Usage
-----
    # In flow matching training (velocity prediction):
    from models.se3flow_energy import SE3FlowEnergyNet
    from models.flow_matching  import ZeroCoMFlowMatching

    model     = SE3FlowEnergyNet(n_residues=10, ...)
    diffusion = ZeroCoMFlowMatching(sigma_min=1e-4)

    loss = diffusion.training_loss_energy(model, x0, energy_z, physics_weight=0.15)

    # Temperature-conditioned sampling:
    x = diffusion.ddim_sample_cfg(model, shape, device, tau=0.0, guidance_scale=2.0)

model_type in config: "se3flow_energy"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from models.egnn import SinusoidalTimestepEmbedding


# ─────────────────────────────────────────────────────────────────────────────
# FLOW MATCHING LAYER  (RBF distances + sequence-separation embedding)
# ─────────────────────────────────────────────────────────────────────────────

class SE3FlowLayer(nn.Module):
    """
    One message-passing layer for the SE(3)-equivariant flow matching network.

    Identical mechanics to EGNNLayer (models/egnn.py) with richer edge features:

      OLD edge input: [ hᵢ | hⱼ | d² (1)     | cond_emb ]
      NEW edge input: [ hᵢ | hⱼ | rbf(d) (n_rbf) | sep_emb (sep_dim) | cond_emb ]

    Coordinate update (equivariant):
        xᵢ' = xᵢ + (1/(N−1)) Σⱼ≠ᵢ (xᵢ−xⱼ) · φ_x(mᵢⱼ)

    SE(3) equivariance is preserved:
        - RBF(||Rxᵢ−Rxⱼ||) = RBF(||xᵢ−xⱼ||) — rotation-invariant distance
        - sep_emb is index-based, not coordinate-based
        → Δ(Rxᵢ) = R·Δxᵢ  ✓
    """

    def __init__(
        self,
        node_dim:       int,
        edge_dim:       int,
        cond_dim:       int,       # time_dim + energy_dim
        n_residues:     int = 10,
        n_rbf:          int = 16,
        rbf_dmax:       float = 15.5,
        sep_dim:        int = 4,
        cond_in_phi_e:  bool = True,
    ):
        super().__init__()
        self.n_rbf        = n_rbf
        self.sep_dim      = sep_dim
        self.cond_in_phi_e = cond_in_phi_e

        # RBF centres: linearly spaced 0.5–15.5 Å, σ = half-spacing = 0.5 Å
        centers = torch.linspace(0.5, rbf_dmax, n_rbf)
        self.register_buffer('rbf_centers', centers)
        self.rbf_sigma = (rbf_dmax - 0.5) / (n_rbf - 1) / 2.0

        # Sequence-separation embedding: |i−j| ∈ {0, …, n_residues−1}
        self.sep_embed = nn.Embedding(n_residues, sep_dim)

        # φ_e: edge message MLP.
        # cond_emb included alongside AdaLN (additive, not replacement):
        #   - cond_emb in phi_e → timestep modulates coordinate-update weights directly
        #   - AdaLN on h       → timestep modulates node feature extraction
        # Both serve different roles; removing either causes bond validity regression.
        # cond_in_phi_e=False for old checkpoints trained before cond_emb was restored.
        edge_in_dim = node_dim * 2 + n_rbf + sep_dim + (cond_dim if cond_in_phi_e else 0)
        self.phi_e = nn.Sequential(
            nn.Linear(edge_in_dim, edge_dim),
            nn.SiLU(),
            nn.Linear(edge_dim, edge_dim),
            nn.SiLU(),
        )

        # φ_x: coordinate weight scalar per edge
        self.phi_x = nn.Sequential(
            nn.Linear(edge_dim, edge_dim),
            nn.SiLU(),
            nn.Linear(edge_dim, 1),
            nn.Tanh(),
        )

        # φ_h: node update MLP
        self.phi_h = nn.Sequential(
            nn.Linear(node_dim + edge_dim, node_dim),
            nn.SiLU(),
            nn.Linear(node_dim, node_dim),
        )

        self.norm = nn.LayerNorm(node_dim)

        # AdaLN-Zero: per-layer scale/shift/gate derived from cond_emb.
        # Zero-init → gates=0, scale=0, shift=0 at init (identity mapping).
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 3 * node_dim, bias=True),
        )
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def _rbf(self, dist: Tensor) -> Tensor:
        """dist : (B, N, N) → (B, N, N, n_rbf) Gaussian RBF features."""
        d = dist.unsqueeze(-1)
        return torch.exp(-((d - self.rbf_centers) ** 2) / (2.0 * self.rbf_sigma ** 2))

    def forward(self, h: Tensor, x: Tensor, cond_emb: Tensor) -> tuple:
        """
        h        : (B, N, node_dim)
        x        : (B, N, 3)
        cond_emb : (B, cond_dim)   concatenated [time_emb | energy_emb]
        Returns (h', x')
        """
        B, N, _ = x.shape

        hi = h[:, :, None, :].expand(B, N, N, -1)
        hj = h[:, None, :, :].expand(B, N, N, -1)
        xi = x[:, :, None, :].expand(B, N, N, 3)
        xj = x[:, None, :, :].expand(B, N, N, 3)

        diff = xi - xj
        dist = diff.norm(dim=-1).clamp(min=1e-8)

        rbf_feat = self._rbf(dist)                                    # (B, N, N, n_rbf)

        idx     = torch.arange(N, device=x.device)
        sep_idx = (idx.unsqueeze(1) - idx.unsqueeze(0)).abs()         # (N, N)
        sep_e   = self.sep_embed(sep_idx).unsqueeze(0).expand(B, -1, -1, -1)

        phi_e_parts = [hi, hj, rbf_feat, sep_e]
        if self.cond_in_phi_e:
            t_e = cond_emb[:, None, None, :].expand(B, N, N, -1)
            phi_e_parts.append(t_e)
        m   = self.phi_e(torch.cat(phi_e_parts, dim=-1))

        eye = torch.eye(N, device=x.device, dtype=torch.bool)
        m   = m * (~eye)[None, :, :, None]

        w     = self.phi_x(m) * (~eye)[None, :, :, None]
        dx    = (diff * w).sum(dim=2) / (N - 1)
        x_new = x + dx

        agg   = m.sum(dim=2)
        h_raw = self.norm(h + self.phi_h(torch.cat([h, agg], dim=-1)))

        # AdaLN-Zero: (scale, shift, gate) from time+energy conditioning
        mods              = self.adaLN_modulation(cond_emb)           # (B, 3*node_dim)
        scale, shift, gate = mods.chunk(3, dim=-1)
        h_new = h + gate.unsqueeze(1) * (
            h_raw * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        )

        return h_new, x_new


# ─────────────────────────────────────────────────────────────────────────────
# SE(3)-EQUIVARIANT FLOW MATCHING VELOCITY NETWORK WITH ENERGY CONDITIONING
# ─────────────────────────────────────────────────────────────────────────────

class SE3FlowEnergyNet(nn.Module):
    """
    SE(3)-equivariant velocity network for energy-conditioned flow matching.

    Predicts the ODE velocity field v_θ(x_t, t, e_z) → (B, N, 3).
    Use with ZeroCoMFlowMatching (models/flow_matching.py).

    This is a FLOW MATCHING model (predicts velocities, integrated with Heun ODE).
    It is NOT a diffusion model, NOT a score network, and is distinct from
    EGNNScoreNetwork (models/egnn.py) which predicts noise for DDPM.

    Config keys (model section):
        hidden_dim       → node_dim        (default 192)
        edge_dim         → edge_dim        (default 96)
        n_layers         → n_layers        (default 7)
        time_dim         → sinusoidal emb  (default 64)
        energy_dim       → energy emb dim  (default 32)
        energy_drop_prob → CFG dropout     (default 0.15)
        n_rbf            → RBF features    (default 16)
        sep_dim          → seq-sep emb     (default 4)

    model_type in config: "se3flow_energy"
    """

    def __init__(
        self,
        n_residues:       int   = 10,
        node_dim:         int   = 192,
        edge_dim:         int   = 96,
        time_dim:         int   = 64,
        n_layers:         int   = 7,
        energy_dim:       int   = 32,
        energy_drop_prob: float = 0.15,
        n_rbf:            int   = 16,
        sep_dim:          int   = 4,
        x1_pred:          bool  = False,
        self_cond:        bool  = False,
        cond_in_phi_e:    bool  = True,
    ):
        super().__init__()
        self.n_residues       = n_residues
        self.energy_drop_prob = energy_drop_prob
        self.x1_pred          = x1_pred
        self.self_cond        = self_cond

        cond_dim = time_dim + energy_dim

        # ── Time embedding ────────────────────────────────────────────────
        self.time_mlp = nn.Sequential(
            SinusoidalTimestepEmbedding(time_dim),
            nn.Linear(time_dim, time_dim * 4),
            nn.SiLU(),
            nn.Linear(time_dim * 4, time_dim),
        )

        # ── Energy embedding + CFG null vector ───────────────────────────
        self.energy_mlp = nn.Sequential(
            nn.Linear(1, energy_dim * 2),
            nn.SiLU(),
            nn.Linear(energy_dim * 2, energy_dim),
        )
        self.null_energy_emb = nn.Parameter(torch.zeros(energy_dim))

        # ── Residue position embedding ────────────────────────────────────
        self.res_embed = nn.Embedding(n_residues, node_dim)

        # ── Input projection ──────────────────────────────────────────────
        self.input_proj = nn.Sequential(
            nn.Linear(node_dim + cond_dim, node_dim),
            nn.SiLU(),
            nn.Linear(node_dim, node_dim),
        )

        # ── Flow matching layers ──────────────────────────────────────────
        self.layers = nn.ModuleList([
            SE3FlowLayer(
                node_dim      = node_dim,
                edge_dim      = edge_dim,
                cond_dim      = cond_dim,
                n_residues    = n_residues,
                n_rbf         = n_rbf,
                sep_dim       = sep_dim,
                cond_in_phi_e = cond_in_phi_e,
            )
            for _ in range(n_layers)
        ])

        # ── Output gate: σ(MLP(h)) ∈ (0,1) scales velocity magnitude ────
        self.output_gate = nn.Sequential(
            nn.Linear(node_dim, node_dim // 2),
            nn.SiLU(),
            nn.Linear(node_dim // 2, 1),
            nn.Sigmoid(),
        )

        # ── x1 prediction head (always created; zero-init) ────────────────
        # Used when x1_pred=True (fine-tuning stage). Predicts the residual
        # δ = x₁ − x_t from final node features h and INPUT coordinates x_in.
        #
        # Design: equivariant message passing on x_in (not accumulated x),
        # zero-initialized final layer → δ=0 at init → x1_pred = x_t ✓
        # This is safe to load from a velocity-mode checkpoint (strict=False):
        # the head starts zero and quickly learns from pre-trained h features.
        x1_edge_in = node_dim * 2 + n_rbf + sep_dim
        self.x1_sep_embed = nn.Embedding(n_residues, sep_dim)
        self.x1_phi_e = nn.Sequential(
            nn.Linear(x1_edge_in, edge_dim), nn.SiLU(),
            nn.Linear(edge_dim, edge_dim),   nn.SiLU(),
        )
        self.x1_phi_x = nn.Sequential(
            nn.Linear(edge_dim, edge_dim // 2), nn.SiLU(),
            nn.Linear(edge_dim // 2, 1),
        )

        # ── Self-conditioning projection (optional) ───────────────────────────
        # When self_cond=True, injects per-node aggregated RBF features from x̂₁
        # (the previous ODE step's x₁ estimate) into node features before the
        # message-passing layers.
        #
        # RBF(||x̂₁_i − x̂₁_j||) is SE(3)-invariant (distances are rotation-
        # invariant).  Mean-aggregation over j gives per-node invariant features.
        # Zero-init ensures the model starts identically to the non-SC version.
        if self_cond:
            self.self_cond_proj = nn.Linear(n_rbf, node_dim)
            nn.init.zeros_(self.self_cond_proj.weight)
            nn.init.zeros_(self.self_cond_proj.bias)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.normal_(self.res_embed.weight, std=0.02)
        nn.init.normal_(self.x1_sep_embed.weight, std=0.02)
        # Zero-init x1 head output → δ=0 at init → x1_pred = x_t regardless of h
        nn.init.zeros_(self.x1_phi_x[-1].weight)
        nn.init.zeros_(self.x1_phi_x[-1].bias)

    @staticmethod
    def temperature_to_energy_z(tau: float) -> float:
        """
        Map temperature τ ∈ [0,1] to z-score energy for CFG sampling.
            τ=0.0 → e_z=−2.0  (~2nd percentile,  stable/folded)
            τ=0.5 → e_z= 0.0  (~50th percentile, average)
            τ=1.0 → e_z=+2.0  (~98th percentile, transient/extended)
        """
        return 4.0 * tau - 2.0

    def forward(
        self,
        x_t:      Tensor,
        t:        Tensor,
        energy_z: Tensor = None,
        sc_x1:    Tensor = None,
    ) -> Tensor:
        """
        x_t      : (B, N, 3)   interpolated coordinates (zero-CoM, model space)
        t        : (B,)         timestep scaled to [0, 999]
        energy_z : (B,) | None  z-score-normalised energy (None → unconditional/CFG null)
        sc_x1    : (B, N, 3) | None  self-conditioning: previous step's x̂₁ estimate.
                   When None, self-conditioning contribution is zero (zero-init weights).

        Returns velocity v_pred : (B, N, 3)  equivariant, zero-CoM
        """
        B, N, _ = x_t.shape

        t_emb = self.time_mlp(t)

        if energy_z is not None:
            e_emb = self.energy_mlp(energy_z.unsqueeze(-1))
            if self.training and self.energy_drop_prob > 0.0:
                drop = torch.rand(B, device=x_t.device) < self.energy_drop_prob
                null = self.null_energy_emb.unsqueeze(0).expand(B, -1)
                e_emb = torch.where(drop.unsqueeze(-1), null, e_emb)
        else:
            e_emb = self.null_energy_emb.unsqueeze(0).expand(B, -1)

        cond_emb = torch.cat([t_emb, e_emb], dim=-1)

        res_idx  = torch.arange(N, device=x_t.device)
        r_emb    = self.res_embed(res_idx)[None].expand(B, -1, -1)
        cond_exp = cond_emb[:, None, :].expand(B, N, -1)
        h        = self.input_proj(torch.cat([r_emb, cond_exp], dim=-1))

        # Self-conditioning: inject RBF(x̂₁) features into node embeddings.
        # Uses per-node mean-aggregated RBF distance features from the previous
        # step's x̂₁ estimate.  SE(3)-invariant: distances are rotation-invariant.
        if self.self_cond and sc_x1 is not None:
            diff_sc = sc_x1[:, :, None, :] - sc_x1[:, None, :, :]  # (B, N, N, 3)
            dist_sc = diff_sc.norm(dim=-1).clamp(min=1e-8)           # (B, N, N)
            rbf_sc  = self.layers[0]._rbf(dist_sc)                   # (B, N, N, n_rbf)
            h_sc    = rbf_sc.mean(dim=2)                             # (B, N, n_rbf)
            h       = h + self.self_cond_proj(h_sc)                  # (B, N, node_dim)

        x    = x_t.clone()
        x_in = x_t.clone()

        for layer in self.layers:
            h, x = layer(h, x, cond_emb)

        if self.x1_pred:
            # x1 prediction via dedicated zero-init equivariant head.
            # Uses INPUT coordinates x_in (not accumulated x) for diff vectors,
            # so δ=0 at init → x1_pred = x_t regardless of layer drift.
            # Pre-trained h features (from velocity stage) give a warm start.
            xi_in = x_in[:, :, None, :].expand(B, N, N, 3)
            xj_in = x_in[:, None, :, :].expand(B, N, N, 3)
            diff_in  = xi_in - xj_in
            dist_in  = diff_in.norm(dim=-1).clamp(min=1e-8)
            rbf_in   = self.layers[0]._rbf(dist_in)               # reuse RBF params

            hi_out   = h[:, :, None, :].expand(B, N, N, -1)
            hj_out   = h[:, None, :, :].expand(B, N, N, -1)
            idx      = torch.arange(N, device=x_in.device)
            sep_idx  = (idx.unsqueeze(1) - idx.unsqueeze(0)).abs()
            sep_out  = self.x1_sep_embed(sep_idx).unsqueeze(0).expand(B, -1, -1, -1)

            eye_out  = torch.eye(N, device=x_in.device, dtype=torch.bool)
            m_out    = self.x1_phi_e(torch.cat([hi_out, hj_out, rbf_in, sep_out], dim=-1))
            m_out    = m_out * (~eye_out)[None, :, :, None]
            w_out    = self.x1_phi_x(m_out) * (~eye_out)[None, :, :, None]
            delta    = (diff_in * w_out).sum(dim=2) / (N - 1)     # equivariant δ

            x_out    = x_in + delta
            x_out    = x_out - x_out.mean(dim=1, keepdim=True)    # zero-CoM
            return x_out

        v_pred = (x - x_in) * self.output_gate(h)
        v_pred = v_pred - v_pred.mean(dim=1, keepdim=True)  # zero-CoM

        return v_pred

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @torch.no_grad()
    def check_equivariance(self, tol: float = 1e-4) -> bool:
        """Rotation-equivariance sanity check (unconditional path)."""
        self.eval()
        B, N = 2, self.n_residues
        x = torch.randn(B, N, 3)
        x = x - x.mean(dim=1, keepdim=True)
        t = torch.zeros(B)

        Q, _ = torch.linalg.qr(torch.randn(3, 3))
        if torch.det(Q) < 0:
            Q[:, 0] *= -1

        Rx = (Q @ x.reshape(B, N, 3, 1)).squeeze(-1)

        v_x  = self(x,  t, energy_z=None, sc_x1=None)
        v_Rx = self(Rx, t, energy_z=None, sc_x1=None)
        R_v  = (Q @ v_x.reshape(B, N, 3, 1)).squeeze(-1)

        err = (v_Rx - R_v).abs().max().item()
        ok  = err < tol
        print(f"  Equivariance check: max_err={err:.2e}  {'✓ PASS' if ok else f'✗ FAIL (tol={tol})'}")
        return ok
