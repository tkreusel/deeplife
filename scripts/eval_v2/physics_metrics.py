"""
scripts/eval_v2/physics_metrics.py
====================================
Physical and biological plausibility metrics for Chignolin structures.

Covers three atom-resolution levels:
  - Cα-only (10 atoms): bond lengths, virtual bond angles, clashes, approximate Ramachandran
  - Backbone N-Cα-C (30 atoms): all bond lengths/angles, ω dihedral, Ramachandran
  - All-atom (93 heavy atoms): 64 covalent bonds, bond angles, VdW-aware clashes, Ramachandran

All functions accept coords as numpy arrays in Ångströms.
"""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from .constants import (
    CA_IDEAL_BOND, CA_IDEAL_COS, CA_CLASH_CUT,
    N_CA_IDEAL, CA_C_IDEAL, C_N_IDEAL, BB_CLASH_CUT,
    BB_ANGLE_IDEALS_COS, BB_ANGLE_IDEALS_DEG,
    AA_CLASH_CUT, VDW_RADII, VDW_CLASH_SCALE,
    OMEGA_IDEAL_DEG,
    RAMA_ALPHA_PHI, RAMA_ALPHA_PSI, RAMA_BETA_PHI, RAMA_BETA_PSI,
    RAMA_FAVOURED_RADIUS, RAMA_ALLOWED_RADIUS,
)


# ─────────────────────────────────────────────────────────────────────────────
# BASIC GEOMETRIC PRIMITIVES
# ─────────────────────────────────────────────────────────────────────────────

def bond_lengths_ca(coords: np.ndarray) -> np.ndarray:
    """Cα–Cα consecutive distances. (N, 10, 3) → (N, 9)."""
    return np.linalg.norm(np.diff(coords, axis=1), axis=-1)


def radius_of_gyration(coords: np.ndarray) -> np.ndarray:
    """Rg per structure. (N, n, 3) → (N,)."""
    com = coords.mean(axis=1, keepdims=True)
    return np.sqrt(((coords - com) ** 2).sum(axis=-1).mean(axis=-1))


def end_to_end(coords: np.ndarray) -> np.ndarray:
    """End-to-end Cα distance. (N, n, 3) → (N,)."""
    return np.linalg.norm(coords[:, -1] - coords[:, 0], axis=-1)


def mmd_rbf(x: np.ndarray, y: np.ndarray,
            sigmas=(0.5, 1.0, 2.0), max_pts: int = 2000) -> float:
    """Maximum Mean Discrepancy with RBF kernel (lower = more similar)."""
    x = x.reshape(len(x), -1).astype(np.float64)
    y = y.reshape(len(y), -1).astype(np.float64)
    if len(x) > max_pts: x = x[np.random.choice(len(x), max_pts, replace=False)]
    if len(y) > max_pts: y = y[np.random.choice(len(y), max_pts, replace=False)]

    def rbf(a, b, s):
        sq = ((a[:, None] - b[None]) ** 2).sum(-1)
        return np.exp(-sq / (2 * s ** 2))

    vals = []
    for s in sigmas:
        Kxx = rbf(x, x, s); Kyy = rbf(y, y, s); Kxy = rbf(x, y, s)
        n, m = len(x), len(y)
        v = ((Kxx.sum() - np.diag(Kxx).sum()) / (n * (n - 1))
           + (Kyy.sum() - np.diag(Kyy).sum()) / (m * (m - 1))
           - 2 * Kxy.mean())
        vals.append(v)
    return float(np.mean(vals))


def pairwise_diversity(coords: np.ndarray, n_sub: int = 300) -> float:
    """Mean pairwise RMSD (subsampled). (N, n, 3) → float."""
    N = len(coords)
    idx_s = np.random.choice(N, min(n_sub, N), replace=False)
    sub   = coords[idx_s]
    sq    = ((sub[:, None] - sub[None]) ** 2).sum(-1).mean(-1)
    triu  = np.triu_indices(len(sub), k=1)
    return float(np.sqrt(sq[triu]).mean()) if len(triu[0]) > 0 else 0.0


def _dihedral(a: np.ndarray, b: np.ndarray,
              c: np.ndarray, d: np.ndarray) -> np.ndarray:
    """
    Compute dihedral angles for arrays of 4-atom sets.
    a, b, c, d : (N, 3)
    returns : (N,) in degrees, range (-180, 180]
    """
    b1 = b - a
    b2 = c - b
    b3 = d - c

    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)
    b2_norm = np.linalg.norm(b2, axis=-1, keepdims=True).clip(1e-8)
    b2_hat  = b2 / b2_norm

    m1 = np.cross(n1, b2_hat)

    x = (n1 * n2).sum(axis=-1)
    y = (m1 * n2).sum(axis=-1)
    return np.degrees(np.arctan2(y, x))


# ─────────────────────────────────────────────────────────────────────────────
# RAMACHANDRAN HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _rama_region(phi_deg: float, psi_deg: float) -> str:
    """Classify a single (φ, ψ) pair into favoured / allowed / disallowed."""
    def _dist_alpha(p, q):
        dp = ((p - RAMA_ALPHA_PHI + 180) % 360) - 180
        dq = ((q - RAMA_ALPHA_PSI + 180) % 360) - 180
        return np.sqrt(dp ** 2 + dq ** 2)

    def _dist_beta(p, q):
        dp = ((p - RAMA_BETA_PHI + 180) % 360) - 180
        dq = ((q - RAMA_BETA_PSI + 180) % 360) - 180
        return np.sqrt(dp ** 2 + dq ** 2)

    d_alpha = _dist_alpha(phi_deg, psi_deg)
    d_beta  = _dist_beta(phi_deg, psi_deg)
    d_min   = min(d_alpha, d_beta)

    if d_min <= RAMA_FAVOURED_RADIUS:
        return 'favoured'
    elif d_min <= RAMA_ALLOWED_RADIUS:
        return 'allowed'
    else:
        return 'disallowed'


def ramachandran_from_backbone(backbone: np.ndarray) -> dict:
    """
    Compute φ/ψ angles from backbone (N, 30, 3) array.
    Returns {'phi': (N, 9), 'psi': (N, 9), 'region_counts': {...}}.

    Atom layout: [N0,CA0,C0, N1,CA1,C1, ..., N9,CA9,C9]
    φᵢ = dihedral(C_{i-1}, N_i, CA_i, C_i)   (residues 1..9)
    ψᵢ = dihedral(N_i, CA_i, C_i, N_{i+1})   (residues 0..8)
    """
    N = backbone.shape[0]
    phis = []
    psis = []

    # ψ for residues 0..8: dihedral(N_i, CA_i, C_i, N_{i+1})
    for i in range(9):
        ni  = backbone[:, 3 * i,     :]
        cai = backbone[:, 3 * i + 1, :]
        ci  = backbone[:, 3 * i + 2, :]
        nj  = backbone[:, 3 * (i+1), :]
        psis.append(_dihedral(ni, cai, ci, nj))

    # φ for residues 1..9: dihedral(C_{i-1}, N_i, CA_i, C_i)
    for i in range(1, 10):
        c_prev = backbone[:, 3 * (i-1) + 2, :]
        ni     = backbone[:, 3 * i,          :]
        cai    = backbone[:, 3 * i + 1,      :]
        ci     = backbone[:, 3 * i + 2,      :]
        phis.append(_dihedral(c_prev, ni, cai, ci))

    phi_arr = np.stack(phis, axis=1)   # (N, 9)
    psi_arr = np.stack(psis, axis=1)   # (N, 9)

    # Count regions over all residues and structures
    counts = {'favoured': 0, 'allowed': 0, 'disallowed': 0}
    for n in range(N):
        for phi, psi in zip(phi_arr[n], psi_arr[n]):
            counts[_rama_region(phi, psi)] += 1
    total = N * 9
    fracs = {k: counts[k] / total for k in counts}

    return {'phi': phi_arr, 'psi': psi_arr, 'region_counts': counts, 'region_fracs': fracs}


def ramachandran_from_ca_nerf(ca: np.ndarray) -> dict:
    """
    Approximate Ramachandran angles by reconstructing ideal N/C backbone from Cα trace.
    ca : (N, 10, 3) — Cα coordinates in Å

    Uses ideal NeRF geometry to place N and C atoms at each residue.
    Angles are approximate: flagged with 'nerf_reconstructed': True.
    """
    import torch
    from models.internal_coords import internal_to_cartesian, cartesian_to_internal

    N_structs = ca.shape[0]
    ca_t = torch.from_numpy(ca).float()

    # Convert Cα to internal coordinates, then reconstruct (just to get canonical coords)
    theta, phi = cartesian_to_internal(ca_t)
    ca_recon   = internal_to_cartesian(theta, phi)   # (N, 10, 3)

    # Build approximate backbone by inserting N and C at ideal geometry
    # N is placed 1.46 Å from Cα along the bisector of the incoming bond
    # C is placed 1.52 Å from Cα along the outgoing bond direction
    # This is a rough approximation — exact placement requires knowing side-chain contexts
    backbone = np.zeros((N_structs, 30, 3), dtype=np.float32)
    ca_np    = ca_recon.numpy()

    for i in range(10):
        cai = ca_np[:, i]
        backbone[:, 3 * i + 1, :] = cai   # Cα at correct position

        # Approximate N: along direction from previous Cα to current, offset by ~1.46 Å
        if i == 0:
            if ca_np.shape[1] > 1:
                d = ca_np[:, 1] - ca_np[:, 0]
            else:
                d = np.zeros_like(cai)
        else:
            d = cai - ca_np[:, i - 1]
        d_norm = np.linalg.norm(d, axis=-1, keepdims=True).clip(1e-8)
        backbone[:, 3 * i, :] = cai - N_CA_IDEAL * (d / d_norm)

        # Approximate C: along direction from current to next Cα, offset by ~1.52 Å
        if i < 9:
            d_fwd = ca_np[:, i + 1] - cai
        else:
            d_fwd = -d   # last residue: extrapolate
        d_fwd_norm = np.linalg.norm(d_fwd, axis=-1, keepdims=True).clip(1e-8)
        backbone[:, 3 * i + 2, :] = cai + CA_C_IDEAL * (d_fwd / d_fwd_norm)

    result = ramachandran_from_backbone(backbone)
    result['nerf_reconstructed'] = True
    return result


# ─────────────────────────────────────────────────────────────────────────────
# OMEGA DIHEDRAL (PEPTIDE BOND PLANARITY)
# ─────────────────────────────────────────────────────────────────────────────

def omega_dihedrals(backbone: np.ndarray) -> np.ndarray:
    """
    Compute ω dihedral angles for backbone (N, 30, 3).
    ω_i = dihedral(CA_i, C_i, N_{i+1}, CA_{i+1})   for i in 0..8
    Returns (N, 9) in degrees. Ideal ~180° for trans peptide.
    """
    N = backbone.shape[0]
    omegas = []
    for i in range(9):
        cai   = backbone[:, 3 * i + 1, :]
        ci    = backbone[:, 3 * i + 2, :]
        nj    = backbone[:, 3 * (i + 1), :]
        caj   = backbone[:, 3 * (i + 1) + 1, :]
        omegas.append(_dihedral(cai, ci, nj, caj))
    return np.stack(omegas, axis=1)   # (N, 9)


def omega_metrics(backbone: np.ndarray) -> dict:
    """ω angle summary statistics."""
    omega = omega_dihedrals(backbone)
    # deviation from 180°
    dev = np.abs(omega - OMEGA_IDEAL_DEG)
    # wrap: also check deviation from -180 side
    dev = np.minimum(dev, 360 - dev)
    return {
        'omega_mean_deg':    float(omega.mean()),
        'omega_std_deg':     float(omega.std()),
        'omega_dev_mean':    float(dev.mean()),
        'omega_trans_frac':  float((dev < 30.0).mean()),   # within 30° of trans
        'omega_cis_frac':    float((omega.abs() < 30.0).mean() if hasattr(omega, 'abs')
                                   else (np.abs(omega) < 30.0).mean()),
    }


# ─────────────────────────────────────────────────────────────────────────────
# BOND ANGLES
# ─────────────────────────────────────────────────────────────────────────────

def backbone_bond_angles(backbone: np.ndarray) -> dict:
    """
    Compute the three backbone bond angle types for (N, 30, 3) backbone coords.
    Returns per-type mean, std, RMSE vs. ideal.

    Angle triplets (atom index mod 3):
      0-1-2 : N-CA-C   ideal cos = -0.364 (111.2°)
      1-2-3 : CA-C-N   ideal cos = -0.440 (116.2°)
      2-3-4 : C-N-CA   ideal cos = -0.526 (121.7°)
    """
    b_back = backbone[:, :28] - backbone[:, 1:29]   # (N, 28, 3) back-direction
    b_fwd  = backbone[:, 2:]  - backbone[:, 1:29]   # (N, 28, 3) fwd-direction

    norm_back = np.linalg.norm(b_back, axis=-1, keepdims=True).clip(1e-8)
    norm_fwd  = np.linalg.norm(b_fwd,  axis=-1, keepdims=True).clip(1e-8)
    cos_theta = ((b_back / norm_back) * (b_fwd / norm_fwd)).sum(axis=-1)  # (N, 28)
    angles_deg = np.degrees(np.arccos(cos_theta.clip(-1 + 1e-7, 1 - 1e-7)))  # (N, 28)

    # Split by position mod 3
    results = {}
    type_names = ['N_CA_C', 'CA_C_N', 'C_N_CA']
    for offset, name in enumerate(type_names):
        ideal_cos = BB_ANGLE_IDEALS_COS[name]
        ideal_deg = BB_ANGLE_IDEALS_DEG[name]
        mask = np.array([k % 3 == offset for k in range(28)])
        sub_cos = cos_theta[:, mask]
        sub_deg = angles_deg[:, mask]
        rmse_cos = float(np.sqrt(((sub_cos - ideal_cos) ** 2).mean()))
        rmse_deg = float(np.sqrt(((sub_deg - ideal_deg) ** 2).mean()))
        results[name] = {
            'mean_deg':  float(sub_deg.mean()),
            'std_deg':   float(sub_deg.std()),
            'ideal_deg': ideal_deg,
            'rmse_deg':  rmse_deg,
            'rmse_cos':  rmse_cos,
        }

    return results


# ─────────────────────────────────────────────────────────────────────────────
# CΑ-ONLY METRICS
# ─────────────────────────────────────────────────────────────────────────────

def metrics_ca(coords: np.ndarray) -> dict:
    """
    Full physics metric suite for Cα-only (10-atom) structures.
    coords : (N, 10, 3) in Å
    """
    bl = bond_lengths_ca(coords)   # (N, 9)

    # Bond validity at multiple tolerances
    v_005 = float((np.abs(bl - CA_IDEAL_BOND) < 0.05).all(axis=1).mean())
    v_01  = float((np.abs(bl - CA_IDEAL_BOND) < 0.10).all(axis=1).mean())
    v_02  = float((np.abs(bl - CA_IDEAL_BOND) < 0.20).all(axis=1).mean())
    v_05  = float((np.abs(bl - CA_IDEAL_BOND) < 0.50).all(axis=1).mean())

    bond_rmse     = float(np.sqrt(((bl - CA_IDEAL_BOND) ** 2).mean()))
    per_bond_rmse = np.sqrt(((bl - CA_IDEAL_BOND) ** 2).mean(axis=0)).tolist()

    # Virtual bond angle (cos of angle between consecutive bond vectors)
    b1 = coords[:, 1:-1] - coords[:, :-2]
    b2 = coords[:, 2:]   - coords[:, 1:-1]
    nb1 = np.linalg.norm(b1, axis=-1, keepdims=True).clip(1e-8)
    nb2 = np.linalg.norm(b2, axis=-1, keepdims=True).clip(1e-8)
    cos_t = ((b1 / nb1) * (b2 / nb2)).sum(axis=-1)   # (N, 8)
    angle_rmse = float(np.sqrt(((cos_t - CA_IDEAL_COS) ** 2).mean()))

    # Clash rate: Cα-Cα non-bonded pairs < 3.5 Å (seq sep ≥ 2)
    N, n_res = coords.shape[:2]
    diff  = coords[:, :, None, :] - coords[:, None, :, :]
    dist  = np.linalg.norm(diff, axis=-1)
    sep   = np.abs(np.arange(n_res)[:, None] - np.arange(n_res)[None, :])
    nb_mask = (sep >= 2)
    clash_rate  = float((dist[:, nb_mask] < CA_CLASH_CUT).any(axis=1).mean())
    clash_count = float((dist[:, nb_mask] < CA_CLASH_CUT).sum(axis=1).mean())

    rg  = radius_of_gyration(coords)
    ete = end_to_end(coords)
    div = pairwise_diversity(coords)

    return {
        'bond_valid_005':   v_005,
        'bond_valid_01':    v_01,
        'bond_valid_02':    v_02,
        'bond_valid_05':    v_05,
        'bond_rmse':        bond_rmse,
        'per_bond_rmse':    per_bond_rmse,
        'angle_rmse_cos':   angle_rmse,
        'clash_rate':       clash_rate,
        'clash_count_mean': clash_count,
        'rg_mean':          float(rg.mean()),
        'rg_std':           float(rg.std()),
        'ete_mean':         float(ete.mean()),
        'ete_std':          float(ete.std()),
        'diversity':        div,
        'atom_type':        'ca',
    }


# ─────────────────────────────────────────────────────────────────────────────
# BACKBONE METRICS (30 atoms)
# ─────────────────────────────────────────────────────────────────────────────

def metrics_backbone(coords: np.ndarray) -> dict:
    """
    Full physics metrics for backbone N-Cα-C (30-atom) structures.
    coords : (N, 30, 3) in Å
    """
    from models.backbone_physics import _SRC, _DST, _IDEAL

    src   = np.array(_SRC)
    dst   = np.array(_DST)
    ideal = np.array(_IDEAL, dtype=np.float32)

    bl = np.linalg.norm(coords[:, dst] - coords[:, src], axis=-1)   # (N, 29)

    # Bond validity at multiple tolerances
    errs = np.abs(bl - ideal)
    v_005 = float((errs < 0.05).all(axis=1).mean())
    v_01  = float((errs < 0.10).all(axis=1).mean())
    v_02  = float((errs < 0.20).all(axis=1).mean())
    v_05  = float((errs < 0.50).all(axis=1).mean())

    bond_rmse     = float(np.sqrt(((bl - ideal) ** 2).mean()))
    per_bond_rmse = np.sqrt(((bl - ideal) ** 2).mean(axis=0)).tolist()

    # Per-bond-type RMSE
    n_ca_mask = np.array([i % 3 == 0 for i in range(29)])
    ca_c_mask = np.array([i % 3 == 1 for i in range(29)])
    c_n_mask  = np.array([i % 3 == 2 for i in range(29)])

    def _type_rmse(mask, target):
        return float(np.sqrt(((bl[:, mask] - target) ** 2).mean()))

    # Bond angles
    angles = backbone_bond_angles(coords)

    # Omega dihedral
    omega = omega_metrics(coords)

    # Clash rate (seq sep ≥ 3)
    N, n_atoms = coords.shape[:2]
    diff  = coords[:, :, None, :] - coords[:, None, :, :]
    dist  = np.linalg.norm(diff, axis=-1)
    sep   = np.abs(np.arange(n_atoms)[:, None] - np.arange(n_atoms)[None, :])
    nb_mask = (sep >= 3)
    clash_rate  = float((dist[:, nb_mask] < BB_CLASH_CUT).any(axis=1).mean())
    clash_count = float((dist[:, nb_mask] < BB_CLASH_CUT).sum(axis=1).mean())

    # Ramachandran
    rama = ramachandran_from_backbone(coords)

    # Global structure metrics (on Cα subset)
    ca = coords[:, 1::3, :]
    rg  = radius_of_gyration(ca)
    ete = end_to_end(ca)
    div = pairwise_diversity(ca)

    return {
        'bond_valid_005':   v_005,
        'bond_valid_01':    v_01,
        'bond_valid_02':    v_02,
        'bond_valid_05':    v_05,
        'bond_rmse':        bond_rmse,
        'per_bond_rmse':    per_bond_rmse,
        'n_ca_rmse':        _type_rmse(n_ca_mask, N_CA_IDEAL),
        'ca_c_rmse':        _type_rmse(ca_c_mask, CA_C_IDEAL),
        'c_n_rmse':         _type_rmse(c_n_mask,  C_N_IDEAL),
        'angles':           angles,
        'omega':            omega,
        'clash_rate':       clash_rate,
        'clash_count_mean': clash_count,
        'rama_favoured':    rama['region_fracs']['favoured'],
        'rama_allowed':     rama['region_fracs']['allowed'],
        'rama_disallowed':  rama['region_fracs']['disallowed'],
        'rama_phi':         rama['phi'].tolist(),
        'rama_psi':         rama['psi'].tolist(),
        'rg_mean':          float(rg.mean()),
        'rg_std':           float(rg.std()),
        'ete_mean':         float(ete.mean()),
        'ete_std':          float(ete.std()),
        'diversity':        div,
        'atom_type':        'backbone',
    }


# ─────────────────────────────────────────────────────────────────────────────
# ALL-ATOM METRICS (93 heavy atoms)
# ─────────────────────────────────────────────────────────────────────────────

def metrics_all_atom(coords: np.ndarray) -> dict:
    """
    Full physics metrics for all-atom Chignolin (93 heavy atoms).
    coords : (N, 93, 3) in Å
    """
    from models.physics_aa import _BOND_INDICES, _BOND_TARGETS as _BT_LIST
    _BT = np.array(_BT_LIST, dtype=np.float32)

    # Covalent bond lengths (64 bonds)
    all_consec = np.linalg.norm(np.diff(coords, axis=1), axis=-1)  # (N, 92)
    bl = all_consec[:, _BOND_INDICES]                               # (N, 64)

    errs = np.abs(bl - _BT)
    v_005 = float((errs < 0.05).all(axis=1).mean())
    v_01  = float((errs < 0.10).all(axis=1).mean())
    v_02  = float((errs < 0.20).all(axis=1).mean())
    v_05  = float((errs < 0.50).all(axis=1).mean())

    bond_rmse     = float(np.sqrt(((bl - _BT) ** 2).mean()))
    per_bond_rmse = np.sqrt(((bl - _BT) ** 2).mean(axis=0)).tolist()

    # Bond angles: for triplets where both (i, i+1) and (i+1, i+2) are covalent bonds
    bond_idx_set = set(_BOND_INDICES)
    angle_triplets = [(i, i+1, i+2) for i in range(91)
                      if i in bond_idx_set and (i+1) in bond_idx_set]
    angle_deviations = []
    if angle_triplets:
        for (i, j, k) in angle_triplets:
            v1 = coords[:, i] - coords[:, j]
            v2 = coords[:, k] - coords[:, j]
            nv1 = np.linalg.norm(v1, axis=-1, keepdims=True).clip(1e-8)
            nv2 = np.linalg.norm(v2, axis=-1, keepdims=True).clip(1e-8)
            cos_a = ((v1 / nv1) * (v2 / nv2)).sum(axis=-1)
            angle_deviations.append(cos_a)
        cos_arr = np.stack(angle_deviations, axis=1)   # (N, n_triplets)
        angle_std = float(cos_arr.std())
    else:
        angle_std = float('nan')

    # Clash rate: non-bonded heavy atoms (seq sep ≥ 4) closer than 2.5 Å
    N, n_atoms = coords.shape[:2]
    diff  = coords[:, :, None, :] - coords[:, None, :, :]
    dist  = np.linalg.norm(diff, axis=-1)
    sep   = np.abs(np.arange(n_atoms)[:, None] - np.arange(n_atoms)[None, :])
    nb_mask = (sep >= 4)
    clash_rate  = float((dist[:, nb_mask] < AA_CLASH_CUT).any(axis=1).mean())
    clash_count = float((dist[:, nb_mask] < AA_CLASH_CUT).sum(axis=1).mean())

    # Ramachandran from backbone subset (atoms 0..29 = N,CA,C backbone)
    backbone_subset = coords[:, :30, :]
    rama = ramachandran_from_backbone(backbone_subset)

    # Global metrics (Cα = every 3rd atom from index 1, first 10 = backbone Cα)
    ca  = coords[:, 1::3, :][:, :10, :]
    rg  = radius_of_gyration(ca)
    ete = end_to_end(ca)
    div = pairwise_diversity(ca)

    return {
        'bond_valid_005':   v_005,
        'bond_valid_01':    v_01,
        'bond_valid_02':    v_02,
        'bond_valid_05':    v_05,
        'bond_rmse':        bond_rmse,
        'per_bond_rmse':    per_bond_rmse,
        'angle_cos_std':    angle_std,
        'clash_rate':       clash_rate,
        'clash_count_mean': clash_count,
        'rama_favoured':    rama['region_fracs']['favoured'],
        'rama_allowed':     rama['region_fracs']['allowed'],
        'rama_disallowed':  rama['region_fracs']['disallowed'],
        'rama_phi':         rama['phi'].tolist(),
        'rama_psi':         rama['psi'].tolist(),
        'rg_mean':          float(rg.mean()),
        'rg_std':           float(rg.std()),
        'ete_mean':         float(ete.mean()),
        'ete_std':          float(ete.std()),
        'diversity':        div,
        'atom_type':        'all_atom',
    }


# ─────────────────────────────────────────────────────────────────────────────
# UNIFIED DISPATCHER
# ─────────────────────────────────────────────────────────────────────────────

def compute_physics_metrics(coords: np.ndarray) -> dict:
    """
    Dispatch to the correct metric function based on atom count.
    coords : (N, n_atoms, 3) in Å
    returns : flat metric dict with 'atom_type' key
    """
    n = coords.shape[1]
    if n == 10:
        return metrics_ca(coords)
    elif n == 30:
        return metrics_backbone(coords)
    elif n == 93:
        return metrics_all_atom(coords)
    else:
        raise ValueError(f"Unsupported atom count: {n}")


def compute_ramachandran(coords: np.ndarray) -> dict:
    """
    Compute Ramachandran angles for any coordinate set.
    For Cα-only models, uses NeRF reconstruction (approximate, flagged).
    """
    n = coords.shape[1]
    if n == 10:
        return ramachandran_from_ca_nerf(coords)
    elif n == 30:
        return ramachandran_from_backbone(coords)
    elif n == 93:
        return ramachandran_from_backbone(coords[:, :30, :])
    else:
        raise ValueError(f"Unsupported atom count: {n}")


# ─────────────────────────────────────────────────────────────────────────────
# CONSOLE TABLE
# ─────────────────────────────────────────────────────────────────────────────

def print_physics_table(metrics: dict, ref_metrics: dict | None = None, label: str = "Model"):
    """Print a formatted physics metrics comparison table."""
    atype = metrics.get('atom_type', 'unknown')
    w = 34

    print(f"\n{'━' * 72}")
    print(f"  {label}  [{atype}]")
    print(f"{'━' * 72}")

    def _ref(key, fmt='.4f'):
        if ref_metrics and key in ref_metrics:
            return f"{ref_metrics[key]:{fmt}}"
        return "—"

    print(f"  {'Metric':<{w}} {'Generated':>12} {'Reference':>12}")
    print(f"  {'─' * 60}")

    rows = [
        ("Bond valid ±0.05 Å (%)", f"{metrics.get('bond_valid_005', float('nan'))*100:.1f}%",
         _ref('bond_valid_005', '.3f')),
        ("Bond valid ±0.10 Å (%)", f"{metrics.get('bond_valid_01',  float('nan'))*100:.1f}%",
         _ref('bond_valid_01', '.3f')),
        ("Bond valid ±0.20 Å (%)", f"{metrics.get('bond_valid_02',  float('nan'))*100:.1f}%",
         _ref('bond_valid_02', '.3f')),
        ("Bond valid ±0.50 Å (%)", f"{metrics.get('bond_valid_05',  float('nan'))*100:.1f}%",
         _ref('bond_valid_05', '.3f')),
        ("Bond RMSE (Å)",          f"{metrics.get('bond_rmse', float('nan')):.4f}",
         _ref('bond_rmse')),
        ("Clash rate (%)",         f"{metrics.get('clash_rate', float('nan'))*100:.1f}%",
         _ref('clash_rate', '.3f')),
        ("Clash count (mean)",     f"{metrics.get('clash_count_mean', float('nan')):.2f}",
         _ref('clash_count_mean', '.2f')),
        ("Rg mean (Å)",            f"{metrics.get('rg_mean', float('nan')):.3f}",
         _ref('rg_mean', '.3f')),
        ("Rg std (Å)",             f"{metrics.get('rg_std',  float('nan')):.3f}",
         _ref('rg_std',  '.3f')),
        ("ETE mean (Å)",           f"{metrics.get('ete_mean', float('nan')):.3f}",
         _ref('ete_mean', '.3f')),
        ("Diversity RMSD (Å)",     f"{metrics.get('diversity', float('nan')):.3f}",
         _ref('diversity', '.3f')),
    ]

    if 'rama_favoured' in metrics:
        rows += [
            ("Ramachandran favoured (%)", f"{metrics['rama_favoured']*100:.1f}%",
             _ref('rama_favoured', '.3f')),
            ("Ramachandran allowed (%)",  f"{metrics['rama_allowed']*100:.1f}%",
             _ref('rama_allowed', '.3f')),
            ("Ramachandran disallowed (%)",f"{metrics['rama_disallowed']*100:.1f}%",
             _ref('rama_disallowed', '.3f')),
        ]

    if 'omega' in metrics:
        rows.append(("ω trans fraction (%)",
                     f"{metrics['omega']['omega_trans_frac']*100:.1f}%", "—"))

    for name, gen, ref in rows:
        print(f"  {name:<{w}} {gen:>12} {ref:>12}")

    print(f"{'━' * 72}")
