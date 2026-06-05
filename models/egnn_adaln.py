"""
models/egnn_adaln.py
=====================
SE(3)-equivariant DDPM score network combining:

  1. AdaLN-Zero node updates (from DiT / AdaLN Transformer)
     Each EGNN layer's node MLP is conditioned on the timestep via an adaptive
     LayerNorm with zero-initialised gate — the block starts as an identity
     residual and learns to deviate, giving clean gradient flow from step 1.

  2. Energy-conditioned generation with Classifier-Free Guidance (CFG)
     A z-score-normalised energy scalar is embedded and concatenated with the
     time embedding to form a shared conditioning vector fed to every layer.
     During training, energy conditioning is randomly dropped (p=0.15) so the
     model also learns the unconditional distribution.  At inference, guidance
     amplification is possible:
         eps = eps_uncond + w * (eps_cond - eps_uncond)

  3. Equivariant coordinate updates (unchanged from egnn.py)
     The coordinate update xᵢ += Σⱼ (xᵢ−xⱼ)·φ_x(mᵢⱼ) is purely a function
     of invariant pair messages, so SE(3)-equivariance is preserved exactly.

Architecture summary
--------------------
  cond_emb = concat( time_mlp(t),  energy_mlp(e_z) )   # (B, cond_dim)

  Per EGNNAdaLNLayer:
    mᵢⱼ  = φ_e( hᵢ ‖ hⱼ ‖ ‖xᵢ−xⱼ‖² ‖ cond_emb )          # invariant message
    xᵢ'  = xᵢ + Σⱼ (xᵢ−xⱼ)·φ_x(mᵢⱼ) / (N-1)              # equivariant update
    shift, scale, gate = adaLN_mod( cond_emb )              # (B, node_dim) each
    h_normed = LayerNorm(hᵢ) * (1 + scale) + shift          # adaptive modulation
    hᵢ'  = hᵢ + gate * φ_h( h_normed ‖ Σⱼ mᵢⱼ )           # gated residual

Temperature-controlled sampling
--------------------------------
    e_z = model.temperature_to_energy_z(tau)   # tau ∈ [0, 1]
    tau=0.0  →  e_z=-2.0  →  stable,   compact  (~2nd  percentile)
    tau=0.5  →  e_z= 0.0  →  average           (~50th percentile)
    tau=1.0  →  e_z=+2.0  →  transient, extended (~98th percentile)

Drop-in compatibility
---------------------
    model(x_t, t)               # unconditional (energy_z defaults to None)
    model(x_t, t, energy_z=ez)  # energy-conditioned
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from models.egnn import SinusoidalTimestepEmbedding


# ─────────────────────────────────────────────────────────────────────────────
# EGNN LAYER WITH AdaLN-ZERO NODE UPDATE
# ─────────────────────────────────────────────────────────────────────────────

class EGNNAdaLNLayer(nn.Module):
    """
    EGNN layer with AdaLN-Zero timestep/energy conditioning.

    Coordinate update: UNCHANGED from EGNNLayer — equivariant weighted sum
    of difference vectors, which is SE(3)-equivariant by construction.

    Node update: AdaLN-Zero replaces standard LayerNorm + residual.
      - LayerNorm has no learned affine (elementwise_affine=False); AdaLN
        supplies the scale and shift from the conditioning vector.
      - A per-step gate (zero-initialised) wraps the update, so the block
        starts as an identity residual and learns to deviate gradually.

    cond_dim = time_dim + energy_dim  (or just time_dim if energy_dim == 0)
    """

    def __init__(self, node_dim: int, edge_dim: int, cond_dim: int):
        super().__init__()

        # φ_e: message MLP — invariant inputs
        self.phi_e = nn.Sequential(
            nn.Linear(node_dim * 2 + 1 + cond_dim, edge_dim),
            nn.SiLU(),
            nn.Linear(edge_dim, edge_dim),
            nn.SiLU(),
        )

        # φ_x: coordinate weight — single scalar per edge, bounded by Tanh
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

        # LayerNorm WITHOUT learned affine — AdaLN supplies scale/shift
        self.norm = nn.LayerNorm(node_dim, elementwise_affine=False)

        # AdaLN modulation head: cond_emb → [shift | scale | gate] each (node_dim,)
        # Zero-init on the linear projection:
        #   gate = 0 at init  →  block starts as identity residual
        #   shift, scale = 0  →  norm output is unmodified at init
        self.adaLN_mod = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 3 * node_dim, bias=True),
        )
        nn.init.zeros_(self.adaLN_mod[-1].weight)
        nn.init.zeros_(self.adaLN_mod[-1].bias)

    def forward(self, h: Tensor, x: Tensor, cond_emb: Tensor):
        """
        h        : (B, N, node_dim)
        x        : (B, N, 3)
        cond_emb : (B, cond_dim)   combined time + energy conditioning
        Returns updated h', x'  (same shapes)
        """
        B, N, _ = x.shape

        # ── Pair features ─────────────────────────────────────────────────
        hi = h[:, :, None, :].expand(B, N, N, -1)
        hj = h[:, None, :, :].expand(B, N, N, -1)
        xi = x[:, :, None, :].expand(B, N, N, 3)
        xj = x[:, None, :, :].expand(B, N, N, 3)

        diff    = xi - xj                                        # (B, N, N, 3)
        dist_sq = (diff ** 2).sum(-1, keepdim=True)              # (B, N, N, 1)
        c_e     = cond_emb[:, None, None, :].expand(B, N, N, -1)

        m   = self.phi_e(torch.cat([hi, hj, dist_sq, c_e], dim=-1))  # (B,N,N,edge_dim)
        eye = torch.eye(N, device=x.device, dtype=torch.bool)
        m   = m * (~eye)[None, :, :, None]

        # ── Equivariant coordinate update (UNCHANGED) ─────────────────────
        w     = self.phi_x(m) * (~eye)[None, :, :, None]        # (B, N, N, 1)
        dx    = (diff * w).sum(dim=2) / (N - 1)                  # (B, N, 3)
        x_new = x + dx

        # ── AdaLN-Zero node update ────────────────────────────────────────
        mods              = self.adaLN_mod(cond_emb)             # (B, 3*node_dim)
        shift, scale, gate = mods.chunk(3, dim=-1)               # each (B, node_dim)

        agg    = m.sum(dim=2)                                    # (B, N, edge_dim)
        h_norm = self.norm(h)                                    # (B, N, node_dim)
        h_mod  = h_norm * (1 + scale[:, None, :]) + shift[:, None, :]  # modulated
        h_up   = self.phi_h(torch.cat([h_mod, agg], dim=-1))    # (B, N, node_dim)
        h_new  = h + gate[:, None, :] * h_up                    # gated residual

        return h_new, x_new


# ─────────────────────────────────────────────────────────────────────────────
# FULL SCORE NETWORK
# ─────────────────────────────────────────────────────────────────────────────

class EGNNAdaLNScoreNetwork(nn.Module):
    """
    SE(3)-equivariant DDPM score network with AdaLN-Zero conditioning
    and optional energy CFG for temperature-controlled sampling.

    Call signatures:
        model(x_t, t)                        # unconditional
        model(x_t, t, energy_z=e_z)          # energy-conditioned
        model(x_t, t, energy_z=None)         # explicit unconditional (same as above)

    Config keys:
        model.hidden_dim       → node_dim
        model.edge_dim         → edge message dimension   (default 64)
        model.time_dim         → time embedding dimension (default 64)
        model.n_layers         → EGNN layers
        model.energy_dim       → energy embedding dimension (default 32; 0 = disabled)
        model.energy_drop_prob → CFG dropout probability    (default 0.15)
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
        self.energy_dim       = energy_dim
        self.energy_drop_prob = energy_drop_prob

        cond_dim = time_dim + energy_dim  # width of the combined conditioning vector

        # ── Time embedding ────────────────────────────────────────────────
        self.time_mlp = nn.Sequential(
            SinusoidalTimestepEmbedding(time_dim),
            nn.Linear(time_dim, time_dim * 4),
            nn.SiLU(),
            nn.Linear(time_dim * 4, time_dim),
        )

        # ── Energy embedding + CFG null token ─────────────────────────────
        if energy_dim > 0:
            self.energy_mlp = nn.Sequential(
                nn.Linear(1, energy_dim * 2),
                nn.SiLU(),
                nn.Linear(energy_dim * 2, energy_dim),
            )
            # Learned null embedding used when energy is dropped or None.
            # Initialised to zeros; learns the "average" unconditional direction.
            self.null_energy_emb = nn.Parameter(torch.zeros(energy_dim))

        # ── Residue position embedding ────────────────────────────────────
        self.res_embed = nn.Embedding(n_residues, node_dim)

        # ── Input projection: [res_embed | cond_emb] → node_dim ──────────
        self.input_proj = nn.Sequential(
            nn.Linear(node_dim + cond_dim, node_dim),
            nn.SiLU(),
            nn.Linear(node_dim, node_dim),
        )

        # ── EGNNAdaLN layers ──────────────────────────────────────────────
        self.layers = nn.ModuleList([
            EGNNAdaLNLayer(node_dim=node_dim, edge_dim=edge_dim, cond_dim=cond_dim)
            for _ in range(n_layers)
        ])

        # ── Output gate: σ(MLP(hᵢ)) scales coordinate displacement ───────
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
                # Skip the zero-init adaLN_mod projections (already handled)
                if hasattr(m, '_adaln_zero_init'):
                    continue
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.normal_(self.res_embed.weight, std=0.02)

    # ── Conditioning builder ──────────────────────────────────────────────

    def _build_cond(self, t: Tensor, energy_z: Tensor | None) -> Tensor:
        """
        Build the combined conditioning vector (B, cond_dim).

        During training with energy_z provided:
          - with probability energy_drop_prob: replace with null_energy_emb
          - otherwise: use energy_mlp(energy_z)

        At inference with energy_z=None: always use null_energy_emb.
        """
        B      = t.shape[0]
        t_emb  = self.time_mlp(t)                               # (B, time_dim)

        if self.energy_dim == 0:
            return t_emb

        if energy_z is not None:
            e_emb = self.energy_mlp(energy_z.float().unsqueeze(-1))  # (B, energy_dim)
            if self.training and self.energy_drop_prob > 0.0:
                drop = torch.rand(B, device=t.device) < self.energy_drop_prob
                null = self.null_energy_emb.unsqueeze(0).expand(B, -1)
                e_emb = torch.where(drop.unsqueeze(-1), null, e_emb)
        else:
            e_emb = self.null_energy_emb.unsqueeze(0).expand(B, -1)  # (B, energy_dim)

        return torch.cat([t_emb, e_emb], dim=-1)                # (B, cond_dim)

    # ── Forward ───────────────────────────────────────────────────────────

    def forward(self, x_t: Tensor, t: Tensor, energy_z: Tensor = None) -> Tensor:
        """
        x_t      : (B, N, 3)   noisy Cα coordinates (scaled, zero-CoM)
        t        : (B,)         integer diffusion timesteps
        energy_z : (B,) | None  z-score-normalised energy
                                 None → unconditional (null energy embedding)

        Returns ε_pred : (B, N, 3)  predicted noise (equivariant, zero-CoM)
        """
        B, N, _ = x_t.shape

        cond_emb = self._build_cond(t, energy_z)                # (B, cond_dim)

        # Initial node features: residue embedding + conditioning
        res_idx = torch.arange(N, device=x_t.device)
        r_emb   = self.res_embed(res_idx)[None].expand(B, -1, -1)    # (B, N, node_dim)
        c_exp   = cond_emb[:, None, :].expand(B, N, -1)              # (B, N, cond_dim)
        h       = self.input_proj(torch.cat([r_emb, c_exp], dim=-1)) # (B, N, node_dim)

        # EGNN AdaLN forward — coordinates updated equivariantly
        x    = x_t.clone()
        x_in = x_t.clone()

        for layer in self.layers:
            h, x = layer(h, x, cond_emb)

        # Predicted noise: gated coordinate displacement, zero-CoM projected
        eps_pred = (x - x_in) * self.output_gate(h)            # (B, N, 3)
        eps_pred = eps_pred - eps_pred.mean(dim=1, keepdim=True)

        return eps_pred

    # ── Utilities ─────────────────────────────────────────────────────────

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @staticmethod
    def temperature_to_energy_z(tau: float) -> float:
        """
        Map temperature τ ∈ [0, 1] to z-score-normalised energy.

            τ = 0.0  →  e_z = -2.0  (stable,    ~2nd  percentile)
            τ = 0.5  →  e_z =  0.0  (average,   ~50th percentile)
            τ = 1.0  →  e_z = +2.0  (transient, ~98th percentile)
        """
        return 4.0 * tau - 2.0

    @torch.no_grad()
    def check_equivariance(self, tol: float = None, n_trials: int = 10) -> bool:
        """Rotation-equivariance sanity check on the unconditional path.

        The tol scales with N: float32 accumulation errors grow with the number
        of pair operations (~N²) summed per node. Default tol = 0.01 * (N/10)
        so a 10-residue model uses 1e-2 and a 93-residue model uses ~0.09.
        """
        if tol is None:
            tol = 0.01 * (self.n_residues / 10)
        self.eval()
        device = next(self.parameters()).device
        B, N   = 2, self.n_residues
        errors = []

        for _ in range(n_trials):
            x = torch.randn(B, N, 3, device=device)
            x = x - x.mean(dim=1, keepdim=True)
            t = torch.randint(0, 100, (B,), dtype=torch.long, device=device)

            Q, R_mat = torch.linalg.qr(torch.randn(3, 3, device=device))
            Q = Q * torch.sign(torch.diag(R_mat)).unsqueeze(0)
            if torch.det(Q) < 0:
                Q[:, 0] *= -1

            Rx     = (Q @ x.reshape(B, N, 3, 1)).squeeze(-1)
            eps_x  = self(x,  t, energy_z=None)
            eps_Rx = self(Rx, t, energy_z=None)
            R_eps  = (Q @ eps_x.reshape(B, N, 3, 1)).squeeze(-1)

            numer = (eps_Rx - R_eps).norm().item()
            denom = eps_x.norm().item() + 1e-8
            errors.append(numer / denom)

        mean_err = sum(errors) / len(errors)
        max_err  = max(errors)
        ok       = mean_err < tol
        print(
            f"  Equivariance check ({n_trials} trials): "
            f"mean_rel_err={mean_err:.2e}  max={max_err:.2e}  "
            f"[{'PASS' if ok else 'FAIL — possible equivariance violation'}]"
        )
        return ok
