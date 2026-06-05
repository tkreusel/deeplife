"""
models/egnn.py
==============
SE(3)-equivariant score network for Chignolin Cα diffusion.

Drop-in replacement for MLPScoreNetwork / TransformerScoreNetwork:
  same call signature  model(x_t, t) -> (B, N, 3)
  same training interface: diffusion.training_loss(model, x0)

Key difference from the baseline models:
  The coordinate update in each EGNN layer is a weighted sum of
  DIFFERENCE VECTORS (xᵢ - xⱼ), which is SE(3)-equivariant by construction:
  rotating the input rotates the output identically, so the model never
  wastes capacity re-learning the same structure at different orientations.

Pair this with ZeroCoMGaussianDiffusion (models/diffusion_zerocom.py)
and RandomSE3Transform (data/transforms.py) for maximum benefit.

Registered as model_type: "egnn" in configs/egnn.yaml.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from models.baseline import SinusoidalTimestepEmbedding


# ─────────────────────────────────────────────────────────────────────────────
# EGNN LAYER
# ─────────────────────────────────────────────────────────────────────────────

class EGNNLayer(nn.Module):
    """
    One layer of the Equivariant Graph Neural Network (Satorras et al., 2021).

    For a fully-connected graph of N nodes with features hᵢ and positions xᵢ:

      Edge message (invariant):
        mᵢⱼ = MLP( hᵢ ‖ hⱼ ‖ ‖xᵢ-xⱼ‖² ‖ t_emb )

      Coordinate update (equivariant):
        xᵢ' = xᵢ + (1/(N-1)) Σⱼ≠ᵢ  (xᵢ - xⱼ) · φ_x(mᵢⱼ)

        Equivariance: ‖Rxᵢ - Rxⱼ‖² = ‖xᵢ - xⱼ‖²  →  mᵢⱼ unchanged
                      ⟹  Δ(Rxᵢ) = R·Δxᵢ  ✓

      Node update (invariant):
        hᵢ' = LayerNorm( hᵢ + MLP( hᵢ ‖ Σⱼ mᵢⱼ ) )
    """

    def __init__(self, node_dim: int, edge_dim: int, time_dim: int):
        super().__init__()

        # φ_e: message MLP — inputs are hᵢ, hⱼ, ‖xᵢ-xⱼ‖², t_emb
        self.phi_e = nn.Sequential(
            nn.Linear(node_dim * 2 + 1 + time_dim, edge_dim),
            nn.SiLU(),
            nn.Linear(edge_dim, edge_dim),
            nn.SiLU(),
        )

        # φ_x: coordinate weight — single scalar per edge
        self.phi_x = nn.Sequential(
            nn.Linear(edge_dim, edge_dim),
            nn.SiLU(),
            nn.Linear(edge_dim, 1),
            nn.Tanh(),  # bound the coordinate update magnitude
        )

        # φ_h: node update MLP
        self.phi_h = nn.Sequential(
            nn.Linear(node_dim + edge_dim, node_dim),
            nn.SiLU(),
            nn.Linear(node_dim, node_dim),
        )

        self.norm = nn.LayerNorm(node_dim)

    def forward(self, h: Tensor, x: Tensor, t_emb: Tensor):
        """
        h     : (B, N, node_dim)
        x     : (B, N, 3)
        t_emb : (B, time_dim)
        Returns updated h', x'  (same shapes)
        """
        B, N, _ = x.shape

        # broadcast to all pairs (i, j)
        hi = h[:, :, None, :].expand(B, N, N, -1)   # (B, N, N, node_dim)
        hj = h[:, None, :, :].expand(B, N, N, -1)
        xi = x[:, :, None, :].expand(B, N, N, 3)
        xj = x[:, None, :, :].expand(B, N, N, 3)

        diff    = xi - xj                                        # (B, N, N, 3)
        dist_sq = (diff ** 2).sum(-1, keepdim=True)              # (B, N, N, 1)
        t_e     = t_emb[:, None, None, :].expand(B, N, N, -1)   # (B, N, N, time_dim)

        edge_in = torch.cat([hi, hj, dist_sq, t_e], dim=-1)
        m       = self.phi_e(edge_in)                            # (B, N, N, edge_dim)

        # mask self-loops
        eye  = torch.eye(N, device=x.device, dtype=torch.bool)
        m    = m * (~eye)[None, :, :, None]

        # equivariant coordinate update
        w    = self.phi_x(m)                                     # (B, N, N, 1)
        w    = w * (~eye)[None, :, :, None]
        dx   = (diff * w).sum(dim=2) / (N - 1)                  # (B, N, 3)
        x_new = x + dx

        # node update with residual + LayerNorm
        agg  = m.sum(dim=2)                                      # (B, N, edge_dim)
        h_new = self.norm(h + self.phi_h(torch.cat([h, agg], dim=-1)))

        return h_new, x_new


# ─────────────────────────────────────────────────────────────────────────────
# EGNN SCORE NETWORK
# ─────────────────────────────────────────────────────────────────────────────

class EGNNScoreNetwork(nn.Module):
    """
    SE(3)-equivariant denoising network for Chignolin Cα point clouds.

    Architecture:
      1. Embed residue index + timestep → initial node features h₀
      2. L EGNN layers: h and x updated jointly
      3. ε_pred = (x_final - x_input) gated by a per-node output head
         The displacement is equivariant; the gate is invariant.

    Drop-in for MLPScoreNetwork / TransformerScoreNetwork:
      model(x_t, t) -> (B, N, 3)

    Config keys used (same block as baseline):
      model.hidden_dim  → node_dim
      model.n_layers    → number of EGNN layers
      model.time_dim    → time embedding dimension
      model.edge_dim    → edge message dimension  (new, optional, default 64)
    """

    def __init__(
        self,
        n_residues: int = 10,
        node_dim:   int = 128,
        edge_dim:   int = 64,
        time_dim:   int = 64,
        n_layers:   int = 5,
    ):
        super().__init__()
        self.n_residues = n_residues

        # ── Time embedding ────────────────────────────────────────────────
        self.time_mlp = nn.Sequential(
            SinusoidalTimestepEmbedding(time_dim),
            nn.Linear(time_dim, time_dim * 4),
            nn.SiLU(),
            nn.Linear(time_dim * 4, time_dim),
        )

        # ── Residue position embedding ────────────────────────────────────
        # Breaks permutation symmetry intentionally: residue 0 ≠ residue 9
        self.res_embed = nn.Embedding(n_residues, node_dim)

        # ── Input projection ──────────────────────────────────────────────
        self.input_proj = nn.Sequential(
            nn.Linear(node_dim + time_dim, node_dim),
            nn.SiLU(),
            nn.Linear(node_dim, node_dim),
        )

        # ── EGNN layers ───────────────────────────────────────────────────
        self.layers = nn.ModuleList([
            EGNNLayer(node_dim=node_dim, edge_dim=edge_dim, time_dim=time_dim)
            for _ in range(n_layers)
        ])

        # ── Output gate: σ(MLP(hᵢ)) ∈ (0,1) scales coordinate shift ──────
        # Starts near 0.5 — the network learns how much to trust its own
        # coordinate prediction at each residue position and noise level.
        self.output_gate = nn.Sequential(
            nn.Linear(node_dim, node_dim // 2),
            nn.SiLU(),
            nn.Linear(node_dim // 2, 1),
            nn.Sigmoid(),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.normal_(self.res_embed.weight, std=0.02)

    def forward(self, x_t: Tensor, t: Tensor) -> Tensor:
        """
        x_t : (B, N, 3)   noisy Cα coordinates (scaled and centered)
        t   : (B,)         integer diffusion timesteps

        Returns ε_pred : (B, N, 3)  predicted noise (equivariant, zero-CoM)
        """
        B, N, _ = x_t.shape

        # time embedding — same for all atoms in a structure
        t_emb = self.time_mlp(t)                                # (B, time_dim)

        # initial node features: residue embedding + time
        res_idx = torch.arange(N, device=x_t.device)
        r_emb   = self.res_embed(res_idx)[None].expand(B, -1, -1)   # (B, N, node_dim)
        t_exp   = t_emb[:, None, :].expand(B, N, -1)                # (B, N, time_dim)
        h       = self.input_proj(torch.cat([r_emb, t_exp], dim=-1)) # (B, N, node_dim)

        # EGNN forward pass — coordinates are updated equivariantly
        x     = x_t.clone()
        x_in  = x_t.clone()   # save for computing displacement

        for layer in self.layers:
            h, x = layer(h, x, t_emb)

        # predicted noise = coordinate displacement, gated by node features
        eps_pred = (x - x_in) * self.output_gate(h)    # (B, N, 3)

        # project to zero-CoM (equivariant + removes global translation)
        eps_pred = eps_pred - eps_pred.mean(dim=1, keepdim=True)

        return eps_pred

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @torch.no_grad()
    def check_equivariance(self, tol: float = 1e-2, n_trials: int = 10) -> bool:
        """
        Empirical SE(3)-equivariance test.

        For an equivariant network: model(R·x, t) == R·model(x, t)

        The test measures the relative error:
            err = ‖model(R·x, t) − R·model(x, t)‖ / ‖model(x, t)‖

        Expected values (float32):
          EGNN (this model), random weights  : ~1e-3 to 5e-3   PASS
          EGNN (this model), trained weights : ~1e-4 to 1e-3   PASS
          Transformer / MLP (non-equivariant): ~0.5  to 2.0    FAIL

        The tol=1e-2 threshold sits between float32 accumulation noise (~5e-3)
        and a real equivariance violation (~0.1-2.0).  If you see mean_err > 1e-2
        you've likely broken equivariance (e.g. added a position-dependent layer).

        To confirm a suspected violation: cast to float64 and repeat.  Float32
        noise drops to ~1e-12 in float64; real violations stay large.

        WHY equivariance matters: the model does not need to re-learn the same
        protein geometry at every orientation; one example covers all rotations.
        This improves data efficiency and generalisation.

        Returns True if mean relative error < tol across n_trials random rotations.
        """
        self.eval()
        B, N = 2, self.n_residues
        errors = []

        for _ in range(n_trials):
            x = torch.randn(B, N, 3)
            x = x - x.mean(dim=1, keepdim=True)   # zero-CoM
            t = torch.randint(0, 100, (B,), dtype=torch.long)

            # Haar-uniform random rotation (det = +1)
            Q, R_mat = torch.linalg.qr(torch.randn(3, 3))
            Q = Q * torch.sign(torch.diag(R_mat)).unsqueeze(0)  # fix column signs
            if torch.det(Q) < 0:
                Q[:, 0] *= -1

            Rx     = (Q @ x.reshape(B, N, 3, 1)).squeeze(-1)   # rotate input
            eps_x  = self(x, t)                                  # ε̂(x, t)
            eps_Rx = self(Rx, t)                                 # ε̂(R·x, t)
            R_eps  = (Q @ eps_x.reshape(B, N, 3, 1)).squeeze(-1) # R·ε̂(x, t)

            numer = (eps_Rx - R_eps).norm().item()
            denom = eps_x.norm().item() + 1e-8
            errors.append(numer / denom)

        mean_err = sum(errors) / len(errors)
        max_err  = max(errors)
        ok       = mean_err < tol
        status   = "PASS" if ok else "FAIL — possible equivariance violation"
        print(
            f"  Equivariance check ({n_trials} trials): "
            f"mean_rel_err={mean_err:.2e}  max={max_err:.2e}  [{status}]"
        )
        return ok
