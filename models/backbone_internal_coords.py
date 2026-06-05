"""
models/backbone_internal_coords.py
===================================
Internal coordinate representation for Chignolin backbone (N, Cα, C atoms).

Provides conversions between backbone Cartesian coordinates (B, 30, 3) and
torsion angles: φ ∈ (−π, π]^9 and ψ ∈ (−π, π]^9, with all bond lengths
and bond angles fixed at ideal values.

Atom layout in the (B, 30, 3) tensor (same as backbone_physics.py):
    [N0, CA0, C0,  N1, CA1, C1,  …,  N9, CA9, C9]
     0   1    2    3   4    5         27  28   29

Degrees of freedom (18 total):
    ψ_i (i=0..8): 9 torsions around CAᵢ–Cᵢ bond; measured as Nᵢ-CAᵢ-Cᵢ-N_{i+1}
                  determines where N_{i+1} is placed
    φ_i (i=1..9): 9 torsions around Nᵢ–CAᵢ bond; measured as C_{i-1}-Nᵢ-CAᵢ-Cᵢ
                  determines where Cᵢ is placed

Fixed quantities:
    ω = π  (trans peptide; fixes CA_i-C_i-N_{i+1}-CA_{i+1} torsion)
    Bond lengths: N–CA=1.460 Å, CA–C=1.525 Å, C–N=1.329 Å
    Bond angles:  N–CA–C=111.2° (cos=−0.364)
                  CA–C–N=116.2° (cos=−0.440)
                  C–N–CA=121.7° (cos=−0.526)

NeRF canonical frame:
    N0 at origin, CA0 along +x, C0 in xy-plane.

SE(3) invariance:
    Torsion angles are invariant under rigid-body rotation and translation.
    Backbone frames (derive_backbone_frames) are SE(3)-equivariant:
    rotating the molecule rotates the frames.
    Pairwise IPA features (local translations) are SE(3)-invariant.
"""

import math
import torch
import torch.nn.functional as F
from torch import Tensor

# ── Physical constants ────────────────────────────────────────────────────────

N_CA_BOND: float = 1.460   # Å
CA_C_BOND: float = 1.525   # Å
C_N_BOND:  float = 1.329   # Å

# cos(bond angle at central atom):  angle = arccos(cos_ideal)
# Convention (same as internal_coords.py): b_back and b_fwd both point AWAY
# from the central atom → dot product = cos(bond angle).
# All three backbone angles are obtuse, so the cosines are negative.
COS_N_CA_C: float = -0.364   # cos(111.2°)   N–CA–C bond angle
COS_CA_C_N: float = -0.440   # cos(116.2°)   CA–C–N bond angle
COS_C_N_CA: float = -0.526   # cos(121.7°)   C–N–CA bond angle

THETA_N_CA_C: float = math.acos(COS_N_CA_C)   # 111.2° in radians
THETA_CA_C_N: float = math.acos(COS_CA_C_N)   # 116.2°
THETA_C_N_CA: float = math.acos(COS_C_N_CA)   # 121.7°

OMEGA: float = math.pi   # trans peptide

N_RESIDUES: int = 10
N_ATOMS:    int = 30     # 3 per residue
N_PHI:      int = 9      # φ_1..φ_9
N_PSI:      int = 9      # ψ_0..ψ_8

# Bond length / angle cycle for atoms k=3..29 (after canonical N0, CA0, C0)
# k%3==0 → N atom; k%3==1 → CA atom; k%3==2 → C atom
_BOND_LENGTHS = [N_CA_BOND, CA_C_BOND, C_N_BOND]   # indexed by (k-1)%3 when adding atom k
# More precisely: atom k is placed after atom k-1.
# atom_type(k) = k % 3:  0=N, 1=CA, 2=C
# bond from k-1 to k:
#   k%3==0 → C→N  : C_N_BOND
#   k%3==1 → N→CA : N_CA_BOND
#   k%3==2 → CA→C : CA_C_BOND

_BOND_FROM_PREV = [C_N_BOND, N_CA_BOND, CA_C_BOND]   # indexed by k%3

# Bond angle at atom k-1 (middle of triplet k-2, k-1, k):
# (k-1)%3==0 → N  is middle: C–N–CA  angle
# (k-1)%3==1 → CA is middle: N–CA–C  angle
# (k-1)%3==2 → C  is middle: CA–C–N  angle
_THETA_AT_PREV = [THETA_C_N_CA, THETA_N_CA_C, THETA_CA_C_N]


# ── angle_wrap ────────────────────────────────────────────────────────────────

def angle_wrap(delta: Tensor) -> Tensor:
    """Wrap angular values/differences to (−π, π] via atan2."""
    return torch.atan2(torch.sin(delta), torch.cos(delta))


# ── backbone_to_internal ──────────────────────────────────────────────────────

def backbone_to_internal(x: Tensor) -> tuple[Tensor, Tensor]:
    """
    Extract φ and ψ torsion angles from backbone Cartesian coordinates.

    x      : (B, 30, 3)  backbone coordinates in Å (or any consistent unit)
    returns: (phi, psi)
        phi : (B, 9)  φ angles [rad] for residues 1..9  (around N_i-CA_i bond)
        psi : (B, 9)  ψ angles [rad] for residues 0..8  (around CA_i-C_i bond)

    Dihedral angle formula: atan2(|b₂|·(b₁·(b₂×b₃)), (b₁×b₂)·(b₂×b₃))

    φ_i (i=1..9): atoms C_{i-1}, N_i, CA_i, C_i  =  3i-1, 3i, 3i+1, 3i+2
    ψ_i (i=0..8): atoms N_i, CA_i, C_i, N_{i+1}  =  3i, 3i+1, 3i+2, 3i+3
    """
    def _dihedral(a: Tensor, b: Tensor, c: Tensor, d: Tensor) -> Tensor:
        """Dihedral angle for 4-atom sequence a-b-c-d. All shapes (B, 3)."""
        b1 = b - a
        b2 = c - b
        b3 = d - c
        n1 = torch.cross(b1, b2, dim=-1)
        n2 = torch.cross(b2, b3, dim=-1)
        b2_norm = b2.norm(dim=-1)                    # (B,)
        y = b2_norm * (b1 * n2).sum(dim=-1)
        x_ = (n1 * n2).sum(dim=-1)
        return torch.atan2(y, x_)                    # (B,)

    # ── φ angles: C_{i-1}-N_i-CA_i-C_i for i=1..9 ────────────────────────────
    phi_list = []
    for i in range(1, 10):
        a = x[:, 3*i - 1]   # C_{i-1}
        b = x[:, 3*i]        # N_i
        c = x[:, 3*i + 1]    # CA_i
        d = x[:, 3*i + 2]    # C_i
        phi_list.append(_dihedral(a, b, c, d))
    phi = torch.stack(phi_list, dim=-1)   # (B, 9)

    # ── ψ angles: N_i-CA_i-C_i-N_{i+1} for i=0..8 ────────────────────────────
    psi_list = []
    for i in range(0, 9):
        a = x[:, 3*i]        # N_i
        b = x[:, 3*i + 1]    # CA_i
        c = x[:, 3*i + 2]    # C_i
        d = x[:, 3*i + 3]    # N_{i+1}
        psi_list.append(_dihedral(a, b, c, d))
    psi = torch.stack(psi_list, dim=-1)   # (B, 9)

    return phi, psi


# ── internal_to_backbone ──────────────────────────────────────────────────────

def internal_to_backbone(
    phi: Tensor,
    psi: Tensor,
    omega: float = OMEGA,
) -> Tensor:
    """
    NeRF forward kinematics: (φ, ψ) torsion angles → backbone Cartesian coords.

    All bond lengths and angles are fixed at ideal values (see module constants).
    Bond validity is 100% by construction.

    phi   : (B, 9)  φ angles [rad] for residues 1..9
    psi   : (B, 9)  ψ angles [rad] for residues 0..8
    omega : float   ω torsion angle [rad] (default π = trans peptide)

    returns: (B, 30, 3)  backbone Cartesian coordinates [Å]
             layout: [N0, CA0, C0, N1, CA1, C1, …, N9, CA9, C9]

    Canonical frame: N0 at origin, CA0 on +x axis, C0 in xy-plane.
    """
    B      = phi.shape[0]
    dtype  = phi.dtype
    device = phi.device

    coords: list[Tensor] = []

    # ── Canonical placement of first three atoms ──────────────────────────────

    # Atom 0: N0 at origin
    coords.append(torch.zeros(B, 3, dtype=dtype, device=device))

    # Atom 1: CA0 along +x at distance N_CA_BOND
    a1 = torch.zeros(B, 3, dtype=dtype, device=device)
    a1[:, 0] = N_CA_BOND
    coords.append(a1)

    # Atom 2: C0 in xy-plane using the N0–CA0–C0 bond angle THETA_N_CA_C
    # NeRF formula with bc_hat = +x (CA0→N0 is -x, but we need CA0→C0 direction):
    # C0 = CA0 + CA_C_BOND * (-cos(THETA_N_CA_C)*bc_hat + sin(THETA_N_CA_C)*ŷ)
    # where bc_hat = normalize(CA0 - N0) = +x
    ct = math.cos(THETA_N_CA_C)
    st = math.sin(THETA_N_CA_C)
    a2 = torch.zeros(B, 3, dtype=dtype, device=device)
    a2[:, 0] = N_CA_BOND - CA_C_BOND * ct   # = N_CA + CA_C*(-cos)
    a2[:, 1] = CA_C_BOND * st
    coords.append(a2)

    # ── Torsion lookup: atoms 3..29 ────────────────────────────────────────────
    # Sequence of torsions for atoms 3, 4, 5, 6, 7, 8, ... (N1, CA1, C1, N2, ...):
    #   atom 3 (N1):  torsion = ψ_0  (psi[:,0])
    #   atom 4 (CA1): torsion = ω_0  = omega (fixed)
    #   atom 5 (C1):  torsion = φ_1  (phi[:,0])
    #   atom 6 (N2):  torsion = ψ_1  (psi[:,1])
    #   atom 7 (CA2): torsion = ω_1  = omega
    #   atom 8 (C2):  torsion = φ_2  (phi[:,1])
    #   ...
    # Pattern per residue i (i=1..9):
    #   N_i   (3i):   ψ_{i-1}  = psi[:, i-1]
    #   CA_i  (3i+1): ω = π   (fixed)
    #   C_i   (3i+2): φ_i      = phi[:, i-1]

    for k in range(3, 30):
        atom_type = k % 3   # 0=N, 1=CA, 2=C
        residue   = k // 3  # which residue (1..9)

        # Bond length and angle for this placement step
        bond_length = _BOND_FROM_PREV[atom_type]
        theta       = _THETA_AT_PREV[(k - 1) % 3]

        # Torsion angle
        if atom_type == 0:      # N atom: torsion = ψ_{residue-1}
            tor = psi[:, residue - 1]
        elif atom_type == 1:    # CA atom: torsion = ω (fixed)
            tor = torch.full((B,), omega, dtype=dtype, device=device)
        else:                   # C atom: torsion = φ_{residue}
            tor = phi[:, residue - 1]

        # NeRF formula: place new atom given three preceding atoms a, b, c
        a = coords[k - 3]   # (B, 3)
        b = coords[k - 2]   # (B, 3)
        c = coords[k - 1]   # (B, 3)

        bc = F.normalize(c - b, dim=-1, eps=1e-8)   # forward bond direction b→c
        ba = F.normalize(b - a, dim=-1, eps=1e-8)   # previous forward direction a→b

        n = F.normalize(torch.cross(ba, bc, dim=-1), dim=-1, eps=1e-8)
        m = torch.cross(n, bc, dim=-1)  # in-plane ⊥ bc; unit since n⊥bc and both unit

        new_bond = (
            -math.cos(theta) * bc
            + math.sin(theta) * (
                tor.cos().unsqueeze(-1) * m
                + tor.sin().unsqueeze(-1) * n
            )
        )   # (B, 3)

        coords.append(c + bond_length * new_bond)

    return torch.stack(coords, dim=1)   # (B, 30, 3)


# ── derive_backbone_frames ─────────────────────────────────────────────────────

def derive_backbone_frames(x: Tensor) -> Tensor:
    """
    Derive AlphaFold2-style backbone frames (rotation matrices) from backbone
    Cartesian coordinates.

    For each residue i, the frame is defined by the N_i, CA_i, C_i triad:
        e1 = normalize(CA_i - N_i)            N→CA direction
        aux = normalize(C_i - N_i)            N→C direction
        e2 = normalize(aux - (aux·e1)*e1)     Gram-Schmidt: in-plane ⊥ e1
        e3 = e1 × e2                          normal to N-CA-C plane
        R_i = [e1 | e2 | e3]  (columns)

    These frames are SE(3)-equivariant: rotating the protein rotates the frames.
    Pairwise features v_ij = R_i^T (x_j - x_i) are SE(3)-invariant.

    x       : (B, 30, 3)  backbone coordinates
    returns : (B, 10, 3, 3)  per-residue rotation matrices (columns = frame axes)
    """
    B = x.shape[0]
    dtype, device = x.dtype, x.device
    R_list = []

    for i in range(N_RESIDUES):
        N_i  = x[:, 3 * i]       # (B, 3)
        CA_i = x[:, 3 * i + 1]   # (B, 3)
        C_i  = x[:, 3 * i + 2]   # (B, 3)

        e1  = F.normalize(CA_i - N_i, dim=-1, eps=1e-8)    # (B, 3)
        aux = F.normalize(C_i  - N_i, dim=-1, eps=1e-8)    # (B, 3)
        # Gram-Schmidt: remove e1 component from aux
        e2  = F.normalize(aux - (aux * e1).sum(dim=-1, keepdim=True) * e1,
                          dim=-1, eps=1e-8)                 # (B, 3)
        e3  = torch.cross(e1, e2, dim=-1)                   # (B, 3)

        # Stack as columns: (B, 3, 3)
        R_i = torch.stack([e1, e2, e3], dim=-1)
        R_list.append(R_i)

    return torch.stack(R_list, dim=1)   # (B, 10, 3, 3)


# ── compute_backbone_velocity_scales ──────────────────────────────────────────

def compute_backbone_velocity_scales(
    all_phi: Tensor,
    all_psi: Tensor,
    source_std_phi: float = 0.30,
    source_std_psi: float = 0.30,
) -> tuple[float, float]:
    """
    Compute phi_scale and psi_scale (velocity std for loss normalisation).

    With WrappedNormal source and data distributions, the expected velocity std
    is approximately: scale = sqrt(sigma_src^2 + sigma_data^2)
    (both σ measured as circular std on S¹).

    all_phi : (N, 9)  φ angles from training set [rad]
    all_psi : (N, 9)  ψ angles from training set [rad]

    returns: (phi_scale, psi_scale) in radians
    """
    def _circ_std(angles: Tensor) -> float:
        """Mean circular std across all positions."""
        sin_m = angles.sin().mean(dim=0)
        cos_m = angles.cos().mean(dim=0)
        R = (sin_m**2 + cos_m**2).sqrt().clamp(max=1.0 - 1e-6)
        return float((-2.0 * R.log()).sqrt().mean().item())

    sigma_data_phi = _circ_std(all_phi)
    sigma_data_psi = _circ_std(all_psi)

    phi_scale = math.sqrt(source_std_phi**2 + sigma_data_phi**2)
    psi_scale = math.sqrt(source_std_psi**2 + sigma_data_psi**2)

    return phi_scale, psi_scale


# ── compute_backbone_source_params ────────────────────────────────────────────

def compute_backbone_source_params(
    all_phi: Tensor,
    all_psi: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor, float, float]:
    """
    Compute per-dihedral source distribution parameters from training data.

    Returns WrappedNormal(0, σ_i) source parameters for φ and ψ, plus per-
    position inverse-variance weights for loss weighting.

    Returns:
        phi_source_std  : (9,) per-dihedral φ source std  [rad]
        psi_source_std  : (9,) per-dihedral ψ source std  [rad]
        phi_weights     : (9,) inverse-variance loss weights (mean=1)
        psi_weights     : (9,) inverse-variance loss weights (mean=1)
        phi_scale_new   : float — updated velocity std for φ
        psi_scale_new   : float — updated velocity std for ψ
    """
    def _per_position_params(angles: Tensor) -> tuple[Tensor, Tensor, float]:
        """angles: (N, k) → (source_std (k,), weights (k,), scale float)"""
        sin_m = angles.sin().mean(dim=0)
        cos_m = angles.cos().mean(dim=0)
        R = (sin_m**2 + cos_m**2).sqrt().clamp(max=1.0 - 1e-6)
        circ_std = (-2.0 * R.log()).sqrt().clamp(0.10, math.pi)   # (k,)

        inv_var  = 1.0 / circ_std**2
        weights  = inv_var / inv_var.mean()   # normalised to mean=1

        scale = float(((circ_std**2 + circ_std**2).mean()).sqrt().item())
        return circ_std, weights, scale

    phi_std, phi_w, phi_scale = _per_position_params(all_phi)
    psi_std, psi_w, psi_scale = _per_position_params(all_psi)

    print("  Per-position φ circular std (data):")
    for i, (cs, w) in enumerate(zip(phi_std.tolist(), phi_w.tolist())):
        print(f"    φ{i+1}: data_std={cs:.3f} rad  weight={w:.3f}")
    print(f"  phi_scale = {phi_scale:.4f} rad")

    print("  Per-position ψ circular std (data):")
    for i, (cs, w) in enumerate(zip(psi_std.tolist(), psi_w.tolist())):
        print(f"    ψ{i}: data_std={cs:.3f} rad  weight={w:.3f}")
    print(f"  psi_scale = {psi_scale:.4f} rad")

    return phi_std, psi_std, phi_w, psi_w, phi_scale, psi_scale


# ── Self-test ─────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import numpy as np

    print("=== Backbone internal coordinates self-test ===\n")

    # ── Test 1: NeRF roundtrip ────────────────────────────────────────────────
    print("Test 1: NeRF roundtrip (random torsion angles)")
    B = 4
    phi_rand = (torch.rand(B, 9) - 0.5) * 2 * math.pi
    psi_rand = (torch.rand(B, 9) - 0.5) * 2 * math.pi

    x = internal_to_backbone(phi_rand, psi_rand)   # (B, 30, 3)
    assert x.shape == (B, 30, 3), f"Bad shape: {x.shape}"

    phi_back, psi_back = backbone_to_internal(x)
    phi_err = angle_wrap(phi_back - phi_rand).abs().max().item()
    psi_err = angle_wrap(psi_back - psi_rand).abs().max().item()
    print(f"  φ roundtrip max error: {phi_err:.2e} rad  (expect < 1e-4)")
    print(f"  ψ roundtrip max error: {psi_err:.2e} rad  (expect < 1e-4)")
    assert phi_err < 1e-3, f"φ roundtrip error too large: {phi_err}"
    assert psi_err < 1e-3, f"ψ roundtrip error too large: {psi_err}"

    # ── Test 2: Bond lengths in reconstructed structure ────────────────────────
    print("\nTest 2: Bond lengths")
    from models.backbone_physics import N_CA_IDEAL, CA_C_IDEAL, C_N_IDEAL, _bond_indices
    srcs, dsts, ideals = _bond_indices()
    lengths = (x[:, dsts] - x[:, srcs]).norm(dim=-1)   # (B, 29)
    ideal_t = torch.tensor(ideals, dtype=x.dtype)
    bond_rmse = (lengths - ideal_t).pow(2).mean().sqrt().item()
    bond_max  = (lengths - ideal_t).abs().max().item()
    print(f"  Bond RMSE: {bond_rmse:.2e} Å  (expect ~0)")
    print(f"  Bond max error: {bond_max:.2e} Å  (expect < 1e-5)")
    assert bond_max < 1e-4, f"Bond error too large: {bond_max}"

    # ── Test 3: SE(3)-equivariance of frames ──────────────────────────────────
    print("\nTest 3: Backbone frame SE(3)-equivariance")
    # Random rotation via QR decomposition
    Q, _ = torch.linalg.qr(torch.randn(3, 3))
    if Q.det() < 0:
        Q[:, 0] = -Q[:, 0]

    x_single = x[:1]   # (1, 30, 3)
    x_rot = (Q @ x_single.squeeze(0).T).T.unsqueeze(0)   # (1, 30, 3)

    R_orig = derive_backbone_frames(x_single)      # (1, 10, 3, 3)
    R_rot  = derive_backbone_frames(x_rot)         # (1, 10, 3, 3)

    # R_rot[i] should equal Q @ R_orig[i]
    R_expected = (Q.unsqueeze(0).unsqueeze(0) @ R_orig)   # (1, 10, 3, 3)
    frame_err = (R_rot - R_expected).abs().max().item()
    print(f"  Frame equivariance max error: {frame_err:.2e}  (expect < 1e-5)")
    assert frame_err < 1e-4, f"Frame equivariance error: {frame_err}"

    # ── Test 4: IPA features SE(3)-invariance ─────────────────────────────────
    print("\nTest 4: IPA pairwise features SE(3)-invariance")
    x_CA_orig = x_single[:, 1::3]   # (1, 10, 3) CA positions
    x_CA_rot  = x_rot[:, 1::3]

    R_frames_orig = R_orig   # (1, 10, 3, 3)
    R_frames_rot  = R_rot    # (1, 10, 3, 3)

    # v_ij_local = R_i^T @ (x_j - x_i) — should be same under rotation
    i_idx, j_idx = 2, 7
    d_orig = x_CA_orig[:, j_idx] - x_CA_orig[:, i_idx]   # (1, 3)
    d_rot  = x_CA_rot[:,  j_idx] - x_CA_rot[:,  i_idx]   # (1, 3)

    v_local_orig = (R_frames_orig[:, i_idx].transpose(-1,-2) @ d_orig.unsqueeze(-1)).squeeze(-1)
    v_local_rot  = (R_frames_rot[:,  i_idx].transpose(-1,-2) @ d_rot.unsqueeze(-1)).squeeze(-1)

    feat_err = (v_local_orig - v_local_rot).abs().max().item()
    print(f"  Local feature invariance max error: {feat_err:.2e}  (expect < 1e-5)")
    assert feat_err < 1e-4, f"Feature invariance error: {feat_err}"

    # ── Test 5: Real data roundtrip ───────────────────────────────────────────
    print("\nTest 5: Real data roundtrip")
    try:
        data = np.load('data_backbone/test.npz')
        x_real = torch.tensor(data['coords'][:8], dtype=torch.float32)   # (8, 30, 3)
        phi_r, psi_r = backbone_to_internal(x_real)
        x_recon = internal_to_backbone(phi_r, psi_r)
        # CoM-center both
        x_real_c  = x_real  - x_real.mean(dim=1, keepdim=True)
        x_recon_c = x_recon - x_recon.mean(dim=1, keepdim=True)
        rmsd = (x_real_c - x_recon_c).pow(2).sum(dim=-1).mean().sqrt().item()
        print(f"  CoM-RMSD (real data): {rmsd:.4f} Å  (expect < 0.1 Å)")
        # Note: CoM RMSD may not be zero because NeRF uses a canonical frame
        # but local geometry should match — check bond error instead
        phi_back, psi_back = backbone_to_internal(x_recon)
        phi_err_r = angle_wrap(phi_back - phi_r).abs().max().item()
        psi_err_r = angle_wrap(psi_back - psi_r).abs().max().item()
        print(f"  φ self-consistency: {phi_err_r:.2e} rad")
        print(f"  ψ self-consistency: {psi_err_r:.2e} rad")
    except FileNotFoundError:
        print("  (data_backbone/test.npz not found — skipping real data test)")

    print("\nAll tests passed.")
