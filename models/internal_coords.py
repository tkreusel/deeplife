"""
models/internal_coords.py
=========================
Internal coordinate representation for Chignolin Cα chains (N=10).

Provides conversions between Cartesian coordinates (B, 10, 3) and internal
coordinates: bond angles θ (B, 8) and dihedral angles φ (B, 7), with bond
lengths fixed at 3.832 Å by construction.

Bond angles θᵢ (i = 0..7): true bond angle at atom i+1, i.e. ∠(i, i+1, i+2)
    θᵢ = arccos( −b̂ᵢ · b̂ᵢ₊₁ )   where bᵢ = xᵢ₊₁ − xᵢ
    Range (0, π). Typical Chignolin value ≈ 1.895 rad (≈ 108.6°).

    Note the minus sign: this is the angle AT the central atom, not the angle
    between forward bond vectors (which equals π − θ ≈ 71.4°, i.e. IDEAL_BOND_COS
    = 0.320 in physics.py measures cos(π − θ), not cos(θ)).

Dihedral angles φᵢ (i = 0..6): torsion around bond (i+1)→(i+2) using atoms i..i+3
    φᵢ = atan2( |b₂| · b₁·(b₂×b₃),  (b₁×b₂)·(b₂×b₃) )
    Range (−π, π].

Bond lengths: fixed at BOND_LENGTH = 3.832 Å. Never predicted, always exact.
SE(3) invariance: internal coords are invariant under rigid rotations/translations.
NeRF reconstruction places atom 0 at origin, atom 1 along +x, atom 2 in xy-plane.
"""

import math
import torch
import torch.nn.functional as F
from torch import Tensor

BOND_LENGTH: float = 3.832   # Å — ideal Cα–Cα bond length (data mean)
N_ATOMS:     int   = 10
N_ANGLES:    int   = 8       # bond angles (at atoms 1..8)
N_DIHEDRALS: int   = 7       # dihedral angles (around bonds 1-2 .. 7-8)


# ─────────────────────────────────────────────────────────────────────────────
# angle_wrap
# ─────────────────────────────────────────────────────────────────────────────

def angle_wrap(delta: Tensor) -> Tensor:
    """Wrap angular difference to (−π, π] via atan2."""
    return torch.atan2(torch.sin(delta), torch.cos(delta))


# ─────────────────────────────────────────────────────────────────────────────
# cartesian_to_internal
# ─────────────────────────────────────────────────────────────────────────────

def cartesian_to_internal(x: Tensor) -> tuple[Tensor, Tensor]:
    """
    Convert Cα Cartesian coordinates to internal coordinates.

    x       : (B, 10, 3)  Cα coordinates in any consistent unit (Å or scaled)
    returns : (theta, phi)
        theta : (B, 8)   bond angles [rad] — ∠(i, i+1, i+2) at central atom i+1
        phi   : (B, 7)   dihedral angles [rad] — torsion around bond (i+1)-(i+2)

    The coordinate unit of x does not matter (distances cancel in angles).
    Typical Chignolin values: theta ≈ 1.895 rad, phi spanning all of (−π, π).
    """
    # ── Bond angles ────────────────────────────────────────────────────────
    # True bond angle at atom i+1 (∠ i–(i+1)–(i+2)):
    #   b_back[i] = x[i]   − x[i+1]   direction from i+1 toward i
    #   b_fwd[i]  = x[i+2] − x[i+1]   direction from i+1 toward i+2
    #   cos(θᵢ)   = cosine_similarity(b_back, b_fwd)
    b_back = x[:, :8] - x[:, 1:9]    # (B, 8, 3)
    b_fwd  = x[:, 2:] - x[:, 1:9]    # (B, 8, 3)

    cos_theta = F.cosine_similarity(b_back, b_fwd, dim=-1).clamp(-1 + 1e-7, 1 - 1e-7)
    theta = torch.acos(cos_theta)      # (B, 8), range ≈ (0, π)

    # ── Dihedral angles ────────────────────────────────────────────────────
    # Dihedral i (i=0..6): atoms i, i+1, i+2, i+3
    #   b1 = x[i+1] − x[i],   b2 = x[i+2] − x[i+1],   b3 = x[i+3] − x[i+2]
    #   n1 = b1 × b2  (normal to plane ABC)
    #   n2 = b2 × b3  (normal to plane BCD)
    #   φ  = atan2(|b2| · (b1 · n2),  n1 · n2)
    b1 = x[:, 1:8]  - x[:, 0:7]     # (B, 7, 3)
    b2 = x[:, 2:9]  - x[:, 1:8]     # (B, 7, 3)
    b3 = x[:, 3:10] - x[:, 2:9]     # (B, 7, 3)

    n1 = torch.cross(b1, b2, dim=-1)  # (B, 7, 3)
    n2 = torch.cross(b2, b3, dim=-1)  # (B, 7, 3)

    b2_norm = b2.norm(dim=-1)          # (B, 7)
    atan_y  = b2_norm * (b1 * n2).sum(dim=-1)   # |b₂| · (b₁ · n₂)
    atan_x  = (n1 * n2).sum(dim=-1)              # n₁ · n₂

    phi = torch.atan2(atan_y, atan_x)  # (B, 7), range (−π, π]

    return theta, phi


# ─────────────────────────────────────────────────────────────────────────────
# internal_to_cartesian  (NeRF forward kinematics)
# ─────────────────────────────────────────────────────────────────────────────

def internal_to_cartesian(
    theta: Tensor,
    phi:   Tensor,
    bond_length: float = BOND_LENGTH,
) -> Tensor:
    """
    NeRF reconstruction: internal coordinates → Cartesian.

    All bond lengths are exactly `bond_length` by construction — this is the
    guarantee that makes bond validity 100%.

    Canonical frame:  atom 0 at origin,  atom 1 along +x,  atom 2 in xy-plane.
    This frame is arbitrary (SE(3)-non-unique); subtract CoM after calling this.

    theta : (B, 8)   bond angles [rad]
    phi   : (B, 7)   dihedral angles [rad]
    returns : (B, 10, 3)  Cartesian coordinates in the same unit as bond_length (Å)
    """
    B      = theta.shape[0]
    d      = bond_length
    dtype  = theta.dtype
    device = theta.device

    coords: list[Tensor] = []

    # Atom 0: origin
    coords.append(torch.zeros(B, 3, dtype=dtype, device=device))

    # Atom 1: along +x axis
    a1 = torch.zeros(B, 3, dtype=dtype, device=device)
    a1[:, 0] = d
    coords.append(a1)

    # Atom 2: in xy-plane using theta[0]
    # NeRF formula: x[2] = x[1] + d * (−cos(θ₀) · b̂_curr + sin(θ₀) · ŷ)
    # where b̂_curr = (1,0,0) and the canonical perpendicular is ŷ = (0,1,0)
    t0 = theta[:, 0]                               # (B,)
    a2 = torch.zeros(B, 3, dtype=dtype, device=device)
    a2[:, 0] = d - d * torch.cos(t0)              # d*(1 − cos θ₀)
    a2[:, 1] = d * torch.sin(t0)                  # d*sin θ₀
    # a2[:, 2] = 0 (already)
    coords.append(a2)

    # Atoms 3..9: NeRF formula using (θ, φ)
    for k in range(3, 10):
        th = theta[:, k - 2]   # bond angle at atom k-1: index (k-2) in theta array
        ph = phi[:, k - 3]     # dihedral for atoms k-3,k-2,k-1,k: index (k-3) in phi array

        a = coords[k - 3]      # (B, 3)
        b = coords[k - 2]      # (B, 3)
        c = coords[k - 1]      # (B, 3)

        # Local frame at c
        bc = F.normalize(c - b, dim=-1, eps=1e-8)   # forward bond direction b→c
        ba = F.normalize(b - a, dim=-1, eps=1e-8)   # previous forward direction a→b

        n = F.normalize(torch.cross(ba, bc, dim=-1), dim=-1, eps=1e-8)
        m = torch.cross(n, bc, dim=-1)   # in-plane ⊥ bc; unit since n⊥bc, both unit

        # New bond vector (unit length by construction)
        new_bond = (
            -torch.cos(th).unsqueeze(-1) * bc
            + torch.sin(th).unsqueeze(-1) * (
                torch.cos(ph).unsqueeze(-1) * m
                + torch.sin(ph).unsqueeze(-1) * n
            )
        )   # (B, 3)

        coords.append(c + d * new_bond)

    return torch.stack(coords, dim=1)   # (B, 10, 3)


# ─────────────────────────────────────────────────────────────────────────────
# compute_velocity_scales  (training-set warmup)
# ─────────────────────────────────────────────────────────────────────────────

def compute_velocity_scales(
    all_theta: Tensor,
    all_phi:   Tensor,
    theta_source_std: float = 0.30,
) -> tuple[float, float, float]:
    """
    Compute theta_mean, theta_scale, phi_scale from collected training angles.

    theta_scale and phi_scale are the expected stds of the OT-CFM target
    velocities (Δθ, Δφ) used to normalise the two loss terms to equal scale.

    all_theta : (N, 8)  bond angles from full training set
    all_phi   : (N, 7)  dihedral angles from full training set
    returns   : (theta_mean, theta_scale, phi_scale)
    """
    theta_mean       = all_theta.mean().item()
    sigma_theta_data = all_theta.std().item()

    # Velocity std for bond angles:  Δθ = θ_data − θ_source, both ≈ Gaussian
    theta_scale = math.sqrt(sigma_theta_data ** 2 + theta_source_std ** 2)

    # Velocity std for dihedrals: φ_source is Uniform(−π, π), std = π/√3
    phi_scale = math.pi / math.sqrt(3)

    return theta_mean, theta_scale, phi_scale


# ─────────────────────────────────────────────────────────────────────────────
# compute_phi_source_params  (data-informed source distribution)
# ─────────────────────────────────────────────────────────────────────────────

def compute_phi_source_params(
    all_phi: Tensor,
    phi_source_std_override: float | None = None,
) -> tuple[Tensor, Tensor, float]:
    """
    Compute per-dihedral circular standard deviations and inverse-variance
    weights from the training set for use as an informed source distribution.

    A WrappedNormal source φ₀ ~ WN(0, σᵢ) instead of Uniform(−π, π) creates
    shorter flow paths when the data is concentrated (σᵢ < π/√3 ≈ 1.81).
    The per-position weights down-weight high-entropy terminal dihedrals and
    up-weight the tightly constrained central ones (e.g. the β-hairpin turn).

    Parameters
    ----------
    all_phi                 : (N, 7) dihedral angles from training set [rad]
    phi_source_std_override : if given, use this scalar σ for all positions
                              instead of the per-dihedral data stds.

    Returns
    -------
    phi_source_std : (7,) per-dihedral source std  [rad]   (as buffer tensor)
    phi_weights    : (7,) inverse-variance weights  [1/rad²] (normalised to mean=1)
    phi_scale_new  : float — updated velocity std for loss normalisation.
                     With WrappedNormal source: E[u_φ²] ≈ σ_src² + σ_data²,
                     so phi_scale_new = sqrt(mean(σ_src² + σ_data²)).
    """
    # Circular std per dihedral: std(sin), std(cos) → circ_std via Fisher metric
    # circ_std = sqrt(-2 * log(|mean(exp(i*φ))|)) — well-defined on S¹
    # For well-dispersed angles, circ_std ≈ sample std in the tangent space.
    sin_mean = all_phi.sin().mean(dim=0)   # (7,)
    cos_mean = all_phi.cos().mean(dim=0)   # (7,)
    R        = (sin_mean ** 2 + cos_mean ** 2).sqrt().clamp(max=1.0 - 1e-6)  # mean resultant length
    circ_std = (-2.0 * R.log()).sqrt()     # (7,) circular std [rad]

    # Clamp to a physically meaningful range
    circ_std = circ_std.clamp(0.10, math.pi)

    if phi_source_std_override is not None:
        phi_source_std = torch.full_like(circ_std, phi_source_std_override)
    else:
        phi_source_std = circ_std.clone()

    # Per-position inverse-variance weights (normalised so mean weight = 1)
    inv_var     = 1.0 / (circ_std ** 2)          # (7,) higher weight → tighter DOF
    phi_weights = inv_var / inv_var.mean()         # (7,) normalised

    # New phi_scale: expected velocity std with the new source distribution
    # E[u_φ²] ≈ σ_src² + σ_data² (source and data are approx independent)
    phi_scale_new = float(
        ((phi_source_std ** 2 + circ_std ** 2).mean()).sqrt().item()
    )

    print("  Per-dihedral circular std (data):")
    for i, (cs, ps, w) in enumerate(
        zip(circ_std.tolist(), phi_source_std.tolist(), phi_weights.tolist())
    ):
        print(f"    φ{i}: data_std={cs:.3f} rad  src_std={ps:.3f} rad  weight={w:.3f}")
    print(f"  phi_scale_new = {phi_scale_new:.4f} rad")

    return phi_source_std, phi_weights, phi_scale_new
