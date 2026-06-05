"""
scripts/eval_v2/novelty.py
============================
Novelty analysis for generated Chignolin structures.

Metrics:
  1. Nearest-neighbour distance (NND): generated→test vs. test→test baseline
  2. Coverage & Precision at multiple thresholds
  3. PCA scatter (conformational space coverage)
  4. RMSD pairwise diversity heatmap
  5. Per-temperature novelty (energy-conditioned models)
  6. Physical validity filter (only plausible novel structures)

Adapted from scripts/evaluate_novelty.py with extensions.
"""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from .model_utils import ca_from_coords
from .physics_metrics import compute_physics_metrics


# ─────────────────────────────────────────────────────────────────────────────
# FINGERPRINT
# ─────────────────────────────────────────────────────────────────────────────

def distance_fingerprint(coords: np.ndarray, max_atoms: int = None) -> np.ndarray:
    """
    Rotation-invariant structural fingerprint: upper triangle of pairwise distance matrix.
    coords    : (N_struct, N_atoms, 3)
    max_atoms : subsample atoms for large structures
    returns   : (N_struct, N_atoms*(N_atoms-1)//2)
    """
    if max_atoms is not None and coords.shape[1] > max_atoms:
        step = coords.shape[1] // max_atoms
        coords = coords[:, ::step, :][:, :max_atoms, :]

    N = coords.shape[1]
    diff  = coords[:, :, None, :] - coords[:, None, :, :]
    dists = np.linalg.norm(diff, axis=-1)
    tri   = np.triu_indices(N, k=1)
    return dists[:, tri[0], tri[1]]


# ─────────────────────────────────────────────────────────────────────────────
# NEAREST-NEIGHBOUR DISTANCES
# ─────────────────────────────────────────────────────────────────────────────

def nearest_neighbour_distances(query: np.ndarray, reference: np.ndarray,
                                chunk: int = 256) -> np.ndarray:
    """
    L2 nearest-neighbour distance from each query to the reference set.
    query, reference : (N, D)
    returns : (N_query,)
    """
    if len(query) == 0 or len(reference) == 0:
        return np.full(len(query), np.inf, dtype=np.float64)

    n_q   = query.shape[0]
    min_d = np.full(n_q, np.inf, dtype=np.float64)

    for i in range(0, n_q, chunk):
        q = query[i:i+chunk]
        qq = (q ** 2).sum(axis=1, keepdims=True)
        rr = (reference ** 2).sum(axis=1, keepdims=True)
        qr = q @ reference.T
        sq = np.maximum(qq + rr.T - 2 * qr, 0.0)
        min_d[i:i+chunk] = np.sqrt(sq.min(axis=1))

    return min_d


# ─────────────────────────────────────────────────────────────────────────────
# COVERAGE & PRECISION
# ─────────────────────────────────────────────────────────────────────────────

def coverage_precision(gen_fp: np.ndarray, ref_fp: np.ndarray,
                       threshold: float) -> tuple:
    """
    Coverage  = fraction of ref structures with a generated structure within threshold.
    Precision = fraction of generated structures with a ref structure within threshold.
    """
    gen_nnd = nearest_neighbour_distances(gen_fp, ref_fp)
    ref_nnd = nearest_neighbour_distances(ref_fp, gen_fp)
    return float((ref_nnd < threshold).mean()), float((gen_nnd < threshold).mean())


# ─────────────────────────────────────────────────────────────────────────────
# PCA
# ─────────────────────────────────────────────────────────────────────────────

def pca_2d(gen_fp: np.ndarray, ref_fp: np.ndarray):
    """Fit PCA on reference, project both. Returns (gen_2d, ref_2d) (N, 2)."""
    mean       = ref_fp.mean(axis=0)
    X          = ref_fp - mean
    _, _, Vt   = np.linalg.svd(X, full_matrices=False)
    components = Vt[:2]
    ref_2d     = (ref_fp - mean) @ components.T
    gen_2d     = (gen_fp - mean) @ components.T
    return gen_2d, ref_2d


# ─────────────────────────────────────────────────────────────────────────────
# PAIRWISE RMSD HEATMAP (Kabsch-aligned on Cα)
# ─────────────────────────────────────────────────────────────────────────────

def _kabsch_rmsd(a: np.ndarray, b: np.ndarray) -> float:
    """Kabsch-aligned RMSD between two (N, 3) structures."""
    a = a - a.mean(axis=0)
    b = b - b.mean(axis=0)
    H   = a.T @ b
    U, S, Vt = np.linalg.svd(H)
    d   = np.linalg.det(Vt.T @ U.T)
    D   = np.diag([1, 1, d])
    R   = Vt.T @ D @ U.T
    a_r = a @ R.T
    return float(np.sqrt(((a_r - b) ** 2).sum(axis=-1).mean()))


def pairwise_rmsd_matrix(coords: np.ndarray, n_sub: int = 50) -> np.ndarray:
    """
    Pairwise Kabsch-RMSD matrix for a subsample of structures.
    coords : (N, n_atoms, 3) — uses Cα projection
    returns : (n_sub, n_sub)
    """
    ca    = ca_from_coords(coords)
    N     = len(ca)
    idx   = np.random.choice(N, min(n_sub, N), replace=False)
    sub   = ca[idx]
    n     = len(sub)
    mat   = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(i + 1, n):
            r = _kabsch_rmsd(sub[i], sub[j])
            mat[i, j] = mat[j, i] = r
    return mat


# ─────────────────────────────────────────────────────────────────────────────
# PHYSICAL VALIDITY FILTER
# ─────────────────────────────────────────────────────────────────────────────

def physical_validity_filter(coords: np.ndarray,
                             bond_valid_threshold: float = 0.8) -> np.ndarray:
    """
    Return boolean mask of structures that pass basic physics filters:
      - At least bond_valid_threshold fraction of bonds within ±0.5 Å
      - No clashes (clash_rate = 0 for that individual structure)

    coords : (N, n_atoms, 3)
    returns : (N,) bool
    """
    n = coords.shape[1]
    metrics = compute_physics_metrics(coords)

    # Per-structure bond validity
    if n == 10:
        from .constants import CA_IDEAL_BOND
        bl   = np.linalg.norm(np.diff(coords, axis=1), axis=-1)
        frac = (np.abs(bl - CA_IDEAL_BOND) < 0.5).mean(axis=1)
    elif n == 30:
        from models.backbone_physics import _SRC, _DST, _IDEAL
        src   = np.array(_SRC); dst = np.array(_DST); ideal = np.array(_IDEAL)
        bl    = np.linalg.norm(coords[:, dst] - coords[:, src], axis=-1)
        frac  = (np.abs(bl - ideal) < 0.5).mean(axis=1)
    elif n == 93:
        from models.physics_aa import _BOND_INDICES, _BOND_TARGETS as _BT_LIST
        _BT   = np.array(_BT_LIST)
        consec = np.linalg.norm(np.diff(coords, axis=1), axis=-1)
        bl     = consec[:, _BOND_INDICES]
        frac   = (np.abs(bl - _BT) < 0.5).mean(axis=1)
    else:
        frac = np.ones(len(coords))

    valid_bond = frac >= bond_valid_threshold

    # Per-structure clash check
    if n == 10:
        from .constants import CA_CLASH_CUT
        diff  = coords[:, :, None, :] - coords[:, None, :, :]
        dist  = np.linalg.norm(diff, axis=-1)
        sep   = np.abs(np.arange(n)[:, None] - np.arange(n)[None, :])
        mask  = sep >= 2
        clash_free = ~(dist[:, mask] < CA_CLASH_CUT).any(axis=1)
    elif n == 30:
        from .constants import BB_CLASH_CUT
        diff  = coords[:, :, None, :] - coords[:, None, :, :]
        dist  = np.linalg.norm(diff, axis=-1)
        sep   = np.abs(np.arange(n)[:, None] - np.arange(n)[None, :])
        mask  = sep >= 3
        clash_free = ~(dist[:, mask] < BB_CLASH_CUT).any(axis=1)
    elif n == 93:
        from .constants import AA_CLASH_CUT
        diff  = coords[:, :, None, :] - coords[:, None, :, :]
        dist  = np.linalg.norm(diff, axis=-1)
        sep   = np.abs(np.arange(n)[:, None] - np.arange(n)[None, :])
        mask  = sep >= 4
        clash_free = ~(dist[:, mask] < AA_CLASH_CUT).any(axis=1)
    else:
        clash_free = np.ones(len(coords), dtype=bool)

    return valid_bond & clash_free


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def run_novelty_analysis(
    gen_coords: np.ndarray,
    ref_coords: np.ndarray,
    label: str = "Model",
    max_ref: int = 2000,
    max_atoms: int = 30,
    apply_physics_filter: bool = True,
) -> dict:
    """Full novelty analysis including physics filter and RMSD heatmap."""
    n_gen = len(gen_coords)

    # Physics filter
    if apply_physics_filter:
        valid_mask = physical_validity_filter(gen_coords)
        n_valid = int(valid_mask.sum())
        print(f"  Physics filter: {n_valid}/{n_gen} structures pass "
              f"({n_valid/n_gen*100:.1f}%)")
        gen_valid = gen_coords[valid_mask]
    else:
        valid_mask = np.ones(n_gen, dtype=bool)
        gen_valid  = gen_coords
        n_valid    = n_gen

    # Early exit: nothing passed the filter (e.g. completely failed model)
    if n_valid == 0:
        print("  No structures passed the physics filter — novelty analysis skipped.")
        return {
            'n_total':        n_gen,
            'n_valid':        0,
            'valid_fraction': 0.0,
            'gen_nnd_mean':   float('nan'),
            'gen_nnd_median': float('nan'),
            'self_nnd_mean':  float('nan'),
            'self_nnd_median':float('nan'),
            'nnd_ratio':      float('nan'),
            'coverage':       float('nan'),
            'precision':      float('nan'),
            'threshold':      float('nan'),
            'rmsd_matrix':    [],
            'skipped':        True,
            'skip_reason':    'all structures failed physics filter',
        }

    # Reference subsample
    n_ref    = min(len(ref_coords), max_ref)
    ref_sub  = ref_coords[np.random.choice(len(ref_coords), n_ref, replace=False)]

    # Fingerprints (on Cα for all models)
    ca_gen = ca_from_coords(gen_valid)
    ca_ref = ca_from_coords(ref_sub)
    gen_fp = distance_fingerprint(ca_gen, max_atoms=min(max_atoms, 10))
    ref_fp = distance_fingerprint(ca_ref, max_atoms=min(max_atoms, 10))

    # Generated → test NND
    gen_nnd = nearest_neighbour_distances(gen_fp, ref_fp)

    # Test → test NND baseline
    n_self  = min(500, n_ref)
    idx_a   = np.random.choice(n_ref, n_self, replace=False)
    idx_b   = np.setdiff1d(np.arange(n_ref), idx_a)[:n_self]
    if len(idx_b) < 10:
        idx_b = idx_a
    self_nnd = nearest_neighbour_distances(ref_fp[idx_a], ref_fp[idx_b])

    threshold = float(np.median(self_nnd))
    coverage, precision = coverage_precision(gen_fp, ref_fp, threshold)

    nnd_ratio = float(gen_nnd.mean() / max(self_nnd.mean(), 1e-8))

    # RMSD pairwise heatmap
    rmsd_mat = pairwise_rmsd_matrix(gen_valid, n_sub=50)

    # Print console summary
    print(f"\n  {'━' * 62}")
    print(f"  Novelty — {label}")
    print(f"  {'━' * 62}")
    print(f"  NND (gen→test): mean={gen_nnd.mean():.3f}  median={np.median(gen_nnd):.3f}")
    print(f"  NND (self):     mean={self_nnd.mean():.3f}  median={np.median(self_nnd):.3f}")
    print(f"  NND ratio (gen/self): {nnd_ratio:.3f}")
    if nnd_ratio < 0.9:
        print("    → Possible memorisation (ratio < 0.9)")
    elif nnd_ratio > 1.1:
        print("    → Novel conformations (ratio > 1.1)")
    else:
        print("    → Matches test distribution (ratio ≈ 1.0)")
    print(f"  Coverage @ {threshold:.2f}: {coverage*100:.1f}%")
    print(f"  Precision @ {threshold:.2f}: {precision*100:.1f}%")

    return {
        'n_total':           n_gen,
        'n_valid':           n_valid,
        'valid_fraction':    float(n_valid / n_gen),
        'gen_nnd_mean':      float(gen_nnd.mean()),
        'gen_nnd_median':    float(np.median(gen_nnd)),
        'self_nnd_mean':     float(self_nnd.mean()),
        'self_nnd_median':   float(np.median(self_nnd)),
        'nnd_ratio':         nnd_ratio,
        'coverage':          coverage,
        'precision':         precision,
        'threshold':         threshold,
        'rmsd_matrix':       rmsd_mat.tolist(),
        # Hidden arrays for plotting (stripped before JSON export)
        '_gen_fp':    gen_fp,
        '_ref_fp':    ref_fp,
        '_gen_nnd':   gen_nnd,
        '_self_nnd':  self_nnd,
        '_gen_valid': gen_valid,
    }


# ─────────────────────────────────────────────────────────────────────────────
# PER-TEMPERATURE NOVELTY
# ─────────────────────────────────────────────────────────────────────────────

def per_temperature_novelty(
    all_samples: dict,     # {tau: (N, n_atoms, 3)}
    ref_coords: np.ndarray,
    max_ref: int = 1000,
    max_atoms: int = 10,
) -> dict:
    """
    Run novelty analysis at each temperature level.
    Returns {tau: {'nnd_ratio', 'coverage', 'precision', 'valid_fraction'}}.
    """
    n_ref   = min(len(ref_coords), max_ref)
    ref_sub = ref_coords[np.random.choice(len(ref_coords), n_ref, replace=False)]
    ca_ref  = ca_from_coords(ref_sub)
    ref_fp  = distance_fingerprint(ca_ref, max_atoms=max_atoms)

    # Self-NND baseline (compute once)
    n_self   = min(300, n_ref)
    idx_a    = np.random.choice(n_ref, n_self, replace=False)
    idx_b    = np.setdiff1d(np.arange(n_ref), idx_a)[:n_self]
    if len(idx_b) < 5: idx_b = idx_a
    self_nnd = nearest_neighbour_distances(ref_fp[idx_a], ref_fp[idx_b])
    self_mean = float(self_nnd.mean())
    threshold = float(np.median(self_nnd))

    results = {}
    for tau, samples in sorted(all_samples.items()):
        mask = physical_validity_filter(samples)
        valid_frac = float(mask.mean())
        valid = samples[mask]
        if len(valid) < 5:
            results[tau] = {'nnd_ratio': float('nan'), 'coverage': 0.0,
                            'precision': 0.0, 'valid_fraction': valid_frac}
            continue

        ca_gen = ca_from_coords(valid)
        gen_fp = distance_fingerprint(ca_gen, max_atoms=max_atoms)
        gen_nnd = nearest_neighbour_distances(gen_fp, ref_fp)
        cov, prec = coverage_precision(gen_fp, ref_fp, threshold)
        ratio = float(gen_nnd.mean() / max(self_mean, 1e-8))

        results[tau] = {
            'nnd_mean':       float(gen_nnd.mean()),
            'nnd_ratio':      ratio,
            'coverage':       cov,
            'precision':      prec,
            'valid_fraction': valid_frac,
        }

    return results
