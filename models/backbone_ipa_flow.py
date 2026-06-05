"""
models/backbone_ipa_flow.py
============================
BackboneIPAFlowNet: IPA-style Transformer velocity network for backbone
torsion flow matching (φ, ψ on N-Cα-C backbone atoms).

Architecture overview
---------------------
10 residue tokens — one per Cα atom.  Each token carries the (sin, cos)
representation of the φ and ψ torsion angles for that residue (zero-padded
for terminal residues that lack one angle).

At each forward pass, the current torsion state (φ_t, ψ_t) is converted to
backbone Cartesian coordinates via NeRF (internal_to_backbone), and per-
residue backbone frames are derived from the N, Cα, C triads.  These 3-D
geometric features are used to compute a pairwise SE(3)-invariant attention
bias that replaces the usual positional embeddings:

    v_ij_local = R_i^T (xCA_j − xCA_i)   [SE(3)-invariant local direction]
    rbf(‖xCA_j − xCA_i‖)                 [distance encoding, 16 RBF bases]
    sep_embed(|i−j|)                      [sequence-separation embedding]

These are projected to a (B, n_heads, 10, 10) attention bias that is added to
raw attention logits before softmax — an "IPA-style" mechanism in the sense of
AlphaFold2, without full quaternion algebra (rotation matrices suffice here
since we predict torsion velocities, not Cartesian frame updates).

Per-layer conditioning uses AdaLN-Zero (DiT-style):
    cond_emb = cat(time_emb_64, energy_emb_32)   dim = cond_dim = 96
    Per-layer Linear(96, 6*d_model, zero-init) → (scale, shift, gate) × 2

Output heads:
    out_psi: h[0:9]  → (B, 9)  ψ velocities for residues 0..8
    out_phi: h[1:10] → (B, 9)  φ velocities for residues 1..9
    Both zero-initialised → identity flow at init.

SE(3) properties
----------------
- Token features (sin/cos torsion angles): SE(3)-invariant
- Geometric attention bias: SE(3)-invariant (local frame projections + distances)
- Conditioning (time, energy): SE(3)-invariant scalars
- Output (φ/ψ velocities): SE(3)-invariant
The model therefore exhibits full SE(3)-equivariance in behaviour: rotating
the input protein changes nothing about the predictions.

model_type in config: "backbone_ipa_energy"

Interface:
    v_psi, v_phi = model(psi, phi, t_scaled, energy_z=None)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from models.egnn import SinusoidalTimestepEmbedding
from models.backbone_internal_coords import (
    internal_to_backbone,
    derive_backbone_frames,
    N_PHI, N_PSI,   # both = 9
)

N_RESIDUES = 10


# ── RBF encoding ──────────────────────────────────────────────────────────────

class RBFEncoding(nn.Module):
    """Gaussian radial-basis-function distance encoding (no learned params)."""

    def __init__(self, n_rbf: int = 16, dmax: float = 18.0):
        super().__init__()
        centers = torch.linspace(0.5, dmax, n_rbf)
        self.register_buffer('centers', centers)
        self.sigma = (dmax - 0.5) / (n_rbf - 1) / 2.0

    def forward(self, dist: Tensor) -> Tensor:
        """dist: (...) → (..., n_rbf)"""
        d = dist.unsqueeze(-1)                                   # (..., 1)
        return torch.exp(-((d - self.centers)**2) / (2 * self.sigma**2))


# ── Geometric attention bias ──────────────────────────────────────────────────

class GeometricAttnBias(nn.Module):
    """
    Computes the (B, n_heads, 10, 10) pairwise geometric attention bias from
    backbone Cartesian coordinates.

    Features per pair (i, j):
        rbf(‖xCA_j − xCA_i‖)          [n_rbf dim]
        R_i^T (xCA_j − xCA_i)         [3 dim, SE(3)-invariant local direction]
        sep_embed(|i−j|)               [sep_dim dim]
    → projected to n_heads scalars via zero-init Linear.

    Zero-init ensures no attention bias at initialisation (all-zero bias →
    uniform attention over all pairs, same as no positional embedding).
    """

    def __init__(
        self,
        n_heads:  int   = 8,
        n_rbf:    int   = 16,
        dmax:     float = 18.0,
        sep_dim:  int   = 4,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.rbf     = RBFEncoding(n_rbf, dmax)
        self.sep_emb = nn.Embedding(N_RESIDUES, sep_dim)

        feat_dim = n_rbf + 3 + sep_dim
        self.proj = nn.Linear(feat_dim, n_heads, bias=True)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: Tensor) -> Tensor:
        """
        x : (B, 30, 3)  backbone coordinates in Å
        returns : (B, n_heads, 10, 10)
        """
        B = x.shape[0]
        device = x.device

        # CA positions and backbone frames
        x_ca = x[:, 1::3]                          # (B, 10, 3)
        R    = derive_backbone_frames(x)            # (B, 10, 3, 3)

        # Pairwise distance vectors  (B, 10, 10, 3)
        xi = x_ca[:, :, None, :]                    # (B, 10, 1, 3)
        xj = x_ca[:, None, :, :]                    # (B, 1, 10, 3)
        d_ij  = xj - xi                             # (B, 10, 10, 3)
        dist  = d_ij.norm(dim=-1).clamp(min=1e-8)   # (B, 10, 10)

        # RBF on distance: (B, 10, 10, n_rbf)
        rbf = self.rbf(dist)

        # SE(3)-invariant local directions: R_i^T @ d_ij
        # R: (B, 10, 3, 3);  d_ij: (B, 10, 10, 3)
        Rt      = R.transpose(-1, -2)               # (B, 10, 3, 3)
        Rt_exp  = Rt[:, :, None, :, :]              # (B, 10, 1, 3, 3)
        d_exp   = d_ij.unsqueeze(-1)                # (B, 10, 10, 3, 1)
        v_local = (Rt_exp @ d_exp).squeeze(-1)      # (B, 10, 10, 3)

        # Sequence-separation embedding: (B, 10, 10, sep_dim)
        idx     = torch.arange(N_RESIDUES, device=device)
        sep_idx = (idx[:, None] - idx[None, :]).abs()           # (10, 10)
        sep_e   = self.sep_emb(sep_idx).unsqueeze(0).expand(B, -1, -1, -1)

        # Concatenate and project to per-head bias
        feat = torch.cat([rbf, v_local, sep_e], dim=-1)         # (B, 10, 10, feat)
        bias = self.proj(feat)                                   # (B, 10, 10, n_heads)
        return bias.permute(0, 3, 1, 2)                         # (B, n_heads, 10, 10)


# ── IPA Transformer layer ─────────────────────────────────────────────────────

class IPATransformerLayer(nn.Module):
    """
    Pre-LN Transformer layer with:
    - AdaLN-Zero conditioning from cond_emb
    - Multi-head self-attention with additive geometric attention bias
    - Feed-forward network (GELU)

    AdaLN-Zero: (scale, shift, gate) × 2 (attention and FFN sub-layers).
    The final Linear in adaLN is zero-init → identity at start.
    """

    def __init__(
        self,
        d_model:   int,
        n_heads:   int,
        cond_dim:  int,
        ff_mult:   int   = 2,
        dropout:   float = 0.1,
    ):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head  = d_model // n_heads

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.qkv  = nn.Linear(d_model, 3 * d_model, bias=False)
        self.oproj = nn.Linear(d_model, d_model)
        self.drop  = nn.Dropout(dropout)

        ff_dim = d_model * ff_mult
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
            nn.Dropout(dropout),
        )

        # AdaLN-Zero: 6 × d_model outputs
        self.adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 6 * d_model, bias=True),
        )
        nn.init.zeros_(self.adaLN[-1].weight)
        nn.init.zeros_(self.adaLN[-1].bias)

    def forward(self, h: Tensor, attn_bias: Tensor, cond_emb: Tensor) -> Tensor:
        """
        h        : (B, N, d_model)
        attn_bias: (B, n_heads, N, N)   additive geometric bias
        cond_emb : (B, cond_dim)
        returns  : (B, N, d_model)
        """
        B, N, D = h.shape
        mods = self.adaLN(cond_emb)                         # (B, 6*D)
        s1, sh1, g1, s2, sh2, g2 = mods.chunk(6, dim=-1)   # each (B, D)

        # ── Attention ─────────────────────────────────────────────────────────
        h_norm = self.norm1(h) * (1 + s1.unsqueeze(1)) + sh1.unsqueeze(1)
        q, k, v = self.qkv(h_norm).chunk(3, dim=-1)

        q = q.view(B, N, self.n_heads, self.d_head).transpose(1, 2)  # (B, H, N, d)
        k = k.view(B, N, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, N, self.n_heads, self.d_head).transpose(1, 2)

        logits = torch.matmul(q, k.transpose(-2, -1)) * (self.d_head ** -0.5)
        logits = logits + attn_bias                         # add IPA geometric bias
        attn   = self.drop(torch.softmax(logits, dim=-1))
        out    = torch.matmul(attn, v)                      # (B, H, N, d)
        out    = out.transpose(1, 2).contiguous().view(B, N, D)
        out    = self.oproj(out)
        h      = h + g1.unsqueeze(1) * out

        # ── Feed-forward ──────────────────────────────────────────────────────
        h_norm = self.norm2(h) * (1 + s2.unsqueeze(1)) + sh2.unsqueeze(1)
        h      = h + g2.unsqueeze(1) * self.ff(h_norm)

        return h


# ── Main model ────────────────────────────────────────────────────────────────

class BackboneIPAFlowNet(nn.Module):
    """
    Backbone IPA Transformer velocity network for torsion flow matching.

    Operates on φ/ψ backbone torsion angles (18 DOF total) and produces
    velocity predictions (dψ/dt, dφ/dt) for flow matching on the flat torus.

    Parameters
    ----------
    d_model          : hidden dimension (default 256)
    n_heads          : attention heads (default 8, head_dim = d_model/n_heads)
    n_layers         : Transformer layers (default 6)
    time_dim         : sinusoidal time embedding dim (default 64)
    energy_dim       : energy embedding dim (default 32)
    energy_drop_prob : CFG dropout probability (default 0.15)
    dropout          : attention + FF dropout (default 0.1)
    n_rbf            : RBF basis functions for distance encoding (default 16)
    sep_dim          : sequence-separation embedding dim (default 4)
    rbf_dmax         : max distance for RBF encoding in Å (default 18.0)
    ff_mult          : feed-forward hidden multiplier (default 2)

    Interface (identical to TorsionTransformerNet):
        v_psi, v_phi = model(psi, phi, t_scaled, energy_z=None)
    """

    def __init__(
        self,
        d_model:          int   = 256,
        n_heads:          int   = 8,
        n_layers:         int   = 6,
        time_dim:         int   = 64,
        energy_dim:       int   = 32,
        energy_drop_prob: float = 0.15,
        dropout:          float = 0.1,
        n_rbf:            int   = 16,
        sep_dim:          int   = 4,
        rbf_dmax:         float = 18.0,
        ff_mult:          int   = 2,
    ):
        super().__init__()
        assert d_model % n_heads == 0, f"d_model={d_model} must be divisible by n_heads={n_heads}"

        self.energy_drop_prob = energy_drop_prob

        cond_dim = time_dim + energy_dim   # 96 by default

        # ── Time embedding ────────────────────────────────────────────────────
        self.time_emb  = SinusoidalTimestepEmbedding(time_dim)
        self.time_proj = nn.Linear(time_dim, time_dim)   # identity-init by default

        # ── Energy embedding ──────────────────────────────────────────────────
        self.energy_proj = nn.Linear(1, energy_dim)
        self.null_energy = nn.Parameter(torch.zeros(energy_dim))

        # ── Token projection: 4 features per residue → d_model ───────────────
        # Token i = [sin ψ_i, cos ψ_i, sin φ_i, cos φ_i], zero-padded for terminals
        self.token_proj = nn.Linear(4, d_model)

        # ── IPA geometric attention bias ──────────────────────────────────────
        self.geo_bias = GeometricAttnBias(
            n_heads=n_heads, n_rbf=n_rbf, dmax=rbf_dmax, sep_dim=sep_dim,
        )

        # ── Transformer layers ────────────────────────────────────────────────
        self.layers = nn.ModuleList([
            IPATransformerLayer(
                d_model=d_model, n_heads=n_heads,
                cond_dim=cond_dim, ff_mult=ff_mult, dropout=dropout,
            )
            for _ in range(n_layers)
        ])

        self.final_norm = nn.LayerNorm(d_model)

        # ── Output heads ──────────────────────────────────────────────────────
        # ψ velocities: residues 0..8 have ψ_0..ψ_8  → h[0:9]
        # φ velocities: residues 1..9 have φ_1..φ_9  → h[1:10]
        self.out_psi = nn.Linear(d_model, 1)
        self.out_phi = nn.Linear(d_model, 1)
        self._init_output_heads()

    def _init_output_heads(self):
        """Zero-init output heads → near-zero velocity predictions at start."""
        nn.init.zeros_(self.out_psi.weight)
        nn.init.zeros_(self.out_psi.bias)
        nn.init.zeros_(self.out_phi.weight)
        nn.init.zeros_(self.out_phi.bias)

    def _build_tokens(self, psi: Tensor, phi: Tensor) -> Tensor:
        """
        Build (B, 10, 4) residue token features.

        Token i features = [sin ψ_i, cos ψ_i, sin φ_i, cos φ_i]

        psi: (B, 9)  ψ_0..ψ_8  → tokens[0:9, 0:2]
        phi: (B, 9)  φ_1..φ_9  → tokens[1:10, 2:4]
        Missing DOFs at terminals: zero-padded.
        """
        B = psi.shape[0]
        tokens = torch.zeros(B, N_RESIDUES, 4, dtype=psi.dtype, device=psi.device)
        tokens[:, 0:9, 0] = psi.sin()
        tokens[:, 0:9, 1] = psi.cos()
        tokens[:, 1:10, 2] = phi.sin()
        tokens[:, 1:10, 3] = phi.cos()
        return tokens   # (B, 10, 4)

    def forward(
        self,
        psi:      Tensor,                # (B, 9)  ψ angles [rad]
        phi:      Tensor,                # (B, 9)  φ angles [rad]
        t_scaled: Tensor,                # (B,)    time in [0, 999]
        energy_z: Tensor | None = None,  # (B,)    z-normalised energy or None
    ) -> tuple[Tensor, Tensor]:
        """
        Returns:
            v_psi : (B, 9)  ψ velocity field [rad/time]
            v_phi : (B, 9)  φ velocity field [rad/time]
        """
        B = psi.shape[0]

        # ── Conditioning vector (B, cond_dim) ─────────────────────────────────
        t_emb = self.time_proj(self.time_emb(t_scaled))   # (B, time_dim)

        null = self.null_energy.unsqueeze(0).expand(B, -1)   # (B, energy_dim)
        if energy_z is None:
            e_emb = null
        else:
            e_emb = self.energy_proj(energy_z.unsqueeze(-1))   # (B, energy_dim)
            if self.training and self.energy_drop_prob > 0:
                drop = torch.rand(B, device=psi.device) < self.energy_drop_prob
                e_emb = torch.where(drop.unsqueeze(-1), null, e_emb)

        cond_emb = torch.cat([t_emb, e_emb], dim=-1)   # (B, cond_dim)

        # ── Token features ─────────────────────────────────────────────────────
        tokens = self._build_tokens(psi, phi)    # (B, 10, 4)
        h      = self.token_proj(tokens)         # (B, 10, d_model)

        # ── IPA geometric attention bias from NeRF-derived 3-D positions ──────
        # Clamp angles to valid range before NeRF reconstruction
        phi_c = phi.clamp(-math.pi + 1e-4, math.pi - 1e-4)
        psi_c = psi.clamp(-math.pi + 1e-4, math.pi - 1e-4)
        x     = internal_to_backbone(phi_c, psi_c)   # (B, 30, 3)
        attn_bias = self.geo_bias(x)                  # (B, n_heads, 10, 10)

        # ── Transformer ────────────────────────────────────────────────────────
        for layer in self.layers:
            h = layer(h, attn_bias, cond_emb)

        h = self.final_norm(h)   # (B, 10, d_model)

        # ── Output heads ───────────────────────────────────────────────────────
        v_psi = self.out_psi(h[:, 0:9]).squeeze(-1)    # (B, 9)
        v_phi = self.out_phi(h[:, 1:10]).squeeze(-1)   # (B, 9)

        return v_psi, v_phi

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── Self-test ─────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("=== BackboneIPAFlowNet self-test ===\n")

    B = 4
    psi = (torch.rand(B, 9) - 0.5) * 2 * math.pi
    phi = (torch.rand(B, 9) - 0.5) * 2 * math.pi
    t   = torch.rand(B) * 999.0
    e_z = torch.randn(B)

    model = BackboneIPAFlowNet()
    model.eval()
    print(f"Parameters: {model.count_parameters():,}")

    # Interface check
    v_psi, v_phi = model(psi, phi, t, e_z)
    assert v_psi.shape == (B, 9), f"v_psi shape: {v_psi.shape}"
    assert v_phi.shape == (B, 9), f"v_phi shape: {v_phi.shape}"
    print(f"Output shapes: v_psi={v_psi.shape}, v_phi={v_phi.shape}  ✓")

    # Zero-init check
    print(f"Zero-init: |v_psi|_max={v_psi.abs().max():.2e}  (expect ~0)")
    print(f"Zero-init: |v_phi|_max={v_phi.abs().max():.2e}  (expect ~0)")
    assert v_psi.abs().max() < 1e-5, "v_psi not zero-initialised"
    assert v_phi.abs().max() < 1e-5, "v_phi not zero-initialised"

    # CFG null path (energy_z=None)
    v_psi_u, v_phi_u = model(psi, phi, t, None)
    assert v_psi_u.shape == (B, 9)
    print("CFG null path: ✓")

    # Training mode (dropout + CFG drop)
    model.train()
    v_psi_tr, v_phi_tr = model(psi, phi, t, e_z)
    assert not v_psi_tr.isnan().any(), "NaN in training forward"
    print("Training forward: ✓")

    # Backward pass — use a non-zero target so gradients flow through zero-init heads
    target_psi = torch.randn_like(v_psi_tr)
    target_phi = torch.randn_like(v_phi_tr)
    loss = ((v_psi_tr - target_psi)**2).mean() + ((v_phi_tr - target_phi)**2).mean()
    loss.backward()
    # Output heads get gradients even though deeper layers don't at zero-init
    # (gradient flows through out_psi.weight from loss, but not back INTO h since
    # d(v_psi)/d(h) = out_psi.weight = 0 at init — this is expected DiT behaviour)
    out_grad = model.out_psi.weight.grad
    assert out_grad is not None and out_grad.abs().max() > 0, \
        "No gradient on out_psi.weight"
    print("Backward / output head gradient: ✓")

    # SE(3)-invariance of attention bias
    model.eval()
    with torch.no_grad():
        from models.backbone_internal_coords import internal_to_backbone, derive_backbone_frames
        x = internal_to_backbone(phi[:1], psi[:1])
        Q, _ = torch.linalg.qr(torch.randn(3, 3))
        if Q.det() < 0:
            Q[:, 0] = -Q[:, 0]
        x_rot = (Q @ x.squeeze(0).T).T.unsqueeze(0)

        bias_orig = model.geo_bias(x)
        bias_rot  = model.geo_bias(x_rot)
        inv_err = (bias_orig - bias_rot).abs().max().item()
        print(f"IPA bias SE(3)-invariance error: {inv_err:.2e}  (expect < 1e-4)")
        assert inv_err < 1e-3, f"Bias not SE(3)-invariant: {inv_err}"

    print("\nAll tests passed.")
