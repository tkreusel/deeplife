"""
scripts/evaluate_novelty.py
============================
Assesses whether generated structures are genuinely novel — not just
reproducing structures already in the test set.

Key idea: compare nearest-neighbour distances of generated→test  vs.
test→test (within-set). If generated structures are just memorised copies,
their nearest-test distances will be very small. If they explore new
conformational space, the distribution will look different.

Three metrics
-------------
1. Nearest-neighbour distance (NND)
   For each generated structure, find the closest test structure.
   Compare the distribution to the within-test NND baseline.
   Uses a rotation-invariant structural fingerprint based on the pairwise
   distance matrix — no Kabsch alignment needed.

2. Coverage & Precision  (at threshold d)
   Coverage  = fraction of TEST  structures that have a generated structure
               within distance d  (did the model cover the reference space?)
   Precision = fraction of GENERATED structures that have a test structure
               within distance d  (are generated structures plausible?)

   A model that memorises: high precision, high coverage.
   A model that generates novel conformations:
     - If diverse and physically meaningful: lower NND, decent coverage
     - If out-of-distribution: large NND, low precision

3. PCA / UMAP visualisation
   Projects all structures (generated + test) into 2D using PCA on the
   rotation-invariant fingerprint. Visual check for:
     - Overlap: generated structures cover the same region as test set → good
     - Separation: generated structures are in a different region → novel
     - Clustering around test points: memorisation → bad

Rotation-invariant fingerprint
--------------------------------
For each structure (N atoms), compute the upper triangle of the pairwise
distance matrix — a vector of N*(N-1)/2 distances.  This is completely
rotation- and translation-invariant, so no alignment is needed.

Usage
-----
# Cα model (10 atoms, 45-dimensional fingerprint):
    python scripts/evaluate_novelty.py \\
        --ckpt   checkpoints/egnn_adaln/v1/best.pt \\
        --test   data/test.npz \\
        --n 1000 --steps 100 \\
        --save   plots/novelty_ca.png

# All-atom model (93 atoms, 4278-dimensional fingerprint — uses subsampling):
    python scripts/evaluate_novelty.py \\
        --ckpt   checkpoints/egnn_adaln_aa/v3/best.pt \\
        --test   data_all_atom/test.npz \\
        --n 500  --steps 100 \\
        --save   plots/novelty_aa.png
"""

import sys, json, argparse
import numpy as np
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.evaluate import load_model_from_ckpt, generate


# ─────────────────────────────────────────────────────────────────────────────
# FINGERPRINT
# ─────────────────────────────────────────────────────────────────────────────

def distance_fingerprint(coords: np.ndarray, max_atoms: int = None) -> np.ndarray:
    """
    Rotation-invariant structural fingerprint: upper triangle of the
    pairwise distance matrix.

    coords    : (N_struct, N_atoms, 3)
    max_atoms : subsample to this many atoms for large structures (all-atom)
    returns   : (N_struct, N_atoms*(N_atoms-1)//2)
    """
    if max_atoms is not None and coords.shape[1] > max_atoms:
        # Use every k-th atom for a uniform subsample
        step = coords.shape[1] // max_atoms
        coords = coords[:, ::step, :][:, :max_atoms, :]

    N = coords.shape[1]
    # Pairwise distances: (N_struct, N_atoms, N_atoms)
    diff  = coords[:, :, None, :] - coords[:, None, :, :]
    dists = np.linalg.norm(diff, axis=-1)
    # Upper triangle (excluding diagonal)
    tri   = np.triu_indices(N, k=1)
    return dists[:, tri[0], tri[1]]   # (N_struct, N_atoms*(N_atoms-1)//2)


# ─────────────────────────────────────────────────────────────────────────────
# NEAREST-NEIGHBOUR DISTANCES
# ─────────────────────────────────────────────────────────────────────────────

def nearest_neighbour_distances(query: np.ndarray, reference: np.ndarray,
                                  chunk: int = 256) -> np.ndarray:
    """
    For each query fingerprint, find the L2 distance to the nearest
    reference fingerprint.

    query     : (N_q, D)
    reference : (N_r, D)
    returns   : (N_q,) — min distance to any reference structure
    """
    n_q   = query.shape[0]
    min_d = np.full(n_q, np.inf, dtype=np.float64)

    # Chunked to avoid OOM
    for i in range(0, n_q, chunk):
        q_chunk = query[i:i+chunk]                         # (chunk, D)
        # Squared Euclidean distances via expansion
        qq = (q_chunk ** 2).sum(axis=1, keepdims=True)    # (chunk, 1)
        rr = (reference ** 2).sum(axis=1, keepdims=True)  # (N_r, 1)
        qr = q_chunk @ reference.T                         # (chunk, N_r)
        sq_dists = np.maximum(qq + rr.T - 2 * qr, 0.0)   # (chunk, N_r)
        min_d[i:i+chunk] = np.sqrt(sq_dists.min(axis=1))

    return min_d


# ─────────────────────────────────────────────────────────────────────────────
# COVERAGE & PRECISION
# ─────────────────────────────────────────────────────────────────────────────

def coverage_precision(gen_fp: np.ndarray, ref_fp: np.ndarray,
                        threshold: float) -> tuple:
    """
    Coverage = fraction of reference structures with at least one generated
               structure within `threshold` distance
    Precision = fraction of generated structures with at least one reference
                structure within `threshold` distance
    """
    # Generated → Reference: precision
    gen_nnd = nearest_neighbour_distances(gen_fp, ref_fp)
    precision = float((gen_nnd < threshold).mean())

    # Reference → Generated: coverage
    ref_nnd = nearest_neighbour_distances(ref_fp, gen_fp)
    coverage  = float((ref_nnd < threshold).mean())

    return coverage, precision


# ─────────────────────────────────────────────────────────────────────────────
# PCA PROJECTION
# ─────────────────────────────────────────────────────────────────────────────

def pca_2d(gen_fp: np.ndarray, ref_fp: np.ndarray):
    """
    PCA on combined fingerprints, returns (gen_2d, ref_2d) each shape (N, 2).
    Fit on reference, transform both.
    """
    mean  = ref_fp.mean(axis=0)
    X     = ref_fp - mean
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    components = Vt[:2]   # top 2 principal components

    ref_2d = (ref_fp - mean) @ components.T
    gen_2d = (gen_fp - mean) @ components.T
    return gen_2d, ref_2d


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def run_novelty_analysis(gen_coords: np.ndarray, ref_coords: np.ndarray,
                          label: str = "Model",
                          max_ref: int = 2000, max_atoms: int = 30) -> dict:
    """
    Full novelty analysis.  max_atoms controls all-atom fingerprint subsampling.
    Returns dict with all metrics.
    """
    n_gen = len(gen_coords)
    n_ref = min(len(ref_coords), max_ref)
    ref_sub = ref_coords[np.random.choice(len(ref_coords), n_ref, replace=False)]

    print(f"\n  Computing fingerprints ({gen_coords.shape[1]} atoms → {max_atoms} used)…")
    gen_fp  = distance_fingerprint(gen_coords, max_atoms=max_atoms)
    ref_fp  = distance_fingerprint(ref_sub,    max_atoms=max_atoms)

    print(f"  Fingerprint dim: {gen_fp.shape[1]}  |  "
          f"N_gen={n_gen}  N_ref={n_ref}")

    # ── NND: generated → test ─────────────────────────────────────────────────
    print("  Computing generated→test nearest-neighbour distances…")
    gen_nnd = nearest_neighbour_distances(gen_fp, ref_fp)

    # ── NND: test → test (within-set baseline) ────────────────────────────────
    print("  Computing test→test nearest-neighbour distances (baseline)…")
    n_self  = min(500, n_ref)
    idx_a   = np.random.choice(n_ref, n_self, replace=False)
    # Remove self-matches by using a different reference set
    idx_b   = np.setdiff1d(np.arange(n_ref), idx_a)[:n_self]
    if len(idx_b) < 50:
        idx_b = idx_a   # fallback if reference is small
    self_nnd = nearest_neighbour_distances(ref_fp[idx_a], ref_fp[idx_b])

    # ── Coverage / Precision at median within-test NND threshold ─────────────
    threshold = float(np.median(self_nnd))
    coverage, precision = coverage_precision(gen_fp, ref_fp, threshold)
    print(f"  Threshold (median within-test NND): {threshold:.3f}")

    results = {
        'gen_nnd_mean':    float(gen_nnd.mean()),
        'gen_nnd_median':  float(np.median(gen_nnd)),
        'self_nnd_mean':   float(self_nnd.mean()),
        'self_nnd_median': float(np.median(self_nnd)),
        'nnd_ratio':       float(gen_nnd.mean() / max(self_nnd.mean(), 1e-8)),
        'coverage':        coverage,
        'precision':       precision,
        'threshold':       threshold,
    }

    # ── Console report ────────────────────────────────────────────────────────
    print(f"\n  {'━'*62}")
    print(f"  Novelty analysis — {label}")
    print(f"  {'━'*62}")
    print(f"  Nearest-neighbour distance (generated → test):")
    print(f"    mean   = {results['gen_nnd_mean']:.3f}")
    print(f"    median = {results['gen_nnd_median']:.3f}")
    print(f"  Within-test NND (test → test baseline):")
    print(f"    mean   = {results['self_nnd_mean']:.3f}")
    print(f"    median = {results['self_nnd_median']:.3f}")
    print(f"  NND ratio (gen/self) = {results['nnd_ratio']:.3f}")
    print(f"    < 1.0 → generated structures are CLOSER than test-to-test "
          f"(possible memorisation)")
    print(f"    ≈ 1.0 → generated structures match test distribution")
    print(f"    > 1.0 → generated structures are FARTHER (novel conformations)")
    print(f"  At threshold {threshold:.3f}:")
    print(f"    Coverage  = {coverage*100:.1f}%  "
          f"(fraction of test set covered by generated)")
    print(f"    Precision = {precision*100:.1f}%  "
          f"(fraction of generated near test set)")

    # PCA for returning
    results['_gen_fp']  = gen_fp
    results['_ref_fp']  = ref_fp
    results['_gen_nnd'] = gen_nnd
    results['_self_nnd'] = self_nnd

    return results


# ─────────────────────────────────────────────────────────────────────────────
# PLOTTING
# ─────────────────────────────────────────────────────────────────────────────

def plot_novelty(results: dict, label: str = "Model", save_path: str = None):
    try:
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
    except ImportError:
        print("matplotlib not installed — skipping plot"); return

    gen_fp   = results['_gen_fp']
    ref_fp   = results['_ref_fp']
    gen_nnd  = results['_gen_nnd']
    self_nnd = results['_self_nnd']
    thr      = results['threshold']

    gen_2d, ref_2d = pca_2d(gen_fp, ref_fp)

    fig = plt.figure(figsize=(15, 5))
    gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)
    fig.suptitle(f"Conformational Novelty — {label}", fontsize=13)

    # ── Panel 1: NND distribution ─────────────────────────────────────────────
    ax = fig.add_subplot(gs[0])
    bins = np.linspace(0, max(gen_nnd.max(), self_nnd.max()) * 1.1, 50)
    ax.hist(self_nnd, bins=bins, density=True, color='#222222', alpha=0.4,
            label='Test→Test (baseline)')
    ax.hist(gen_nnd,  bins=bins, density=True, color='#C44E52', alpha=0.7,
            label='Generated→Test')
    ax.axvline(thr, color='k', linestyle='--', lw=1.5, alpha=0.6,
               label=f'Threshold ({thr:.2f})')
    ax.set_xlabel("Nearest-neighbour distance", fontsize=9)
    ax.set_ylabel("Density", fontsize=9)
    ax.set_title("NND Distribution\n(left=memorised, right=novel)", fontsize=10)
    ax.legend(fontsize=7); ax.grid(alpha=0.3)

    # ── Panel 2: PCA scatter ──────────────────────────────────────────────────
    ax = fig.add_subplot(gs[1])
    n_show = min(2000, len(ref_2d))
    ax.scatter(ref_2d[:n_show, 0], ref_2d[:n_show, 1],
               s=4, alpha=0.25, color='#222222', label='Test set', rasterized=True)
    ax.scatter(gen_2d[:, 0], gen_2d[:, 1],
               s=6, alpha=0.5, color='#C44E52', label='Generated', rasterized=True)
    ax.set_xlabel("PC1", fontsize=9); ax.set_ylabel("PC2", fontsize=9)
    ax.set_title("PCA of Conformational Space\n(overlap = covers test distribution)",
                 fontsize=10)
    ax.legend(fontsize=7, markerscale=3); ax.grid(alpha=0.2)

    # ── Panel 3: Coverage / Precision vs threshold ────────────────────────────
    ax = fig.add_subplot(gs[2])
    thresholds = np.linspace(thr * 0.1, thr * 3.0, 30)
    covs, precs = [], []
    for t in thresholds:
        c, p = coverage_precision(gen_fp, ref_fp, t)
        covs.append(c); precs.append(p)
    ax.plot(thresholds, [c*100 for c in covs],  color='#4C72B0', lw=2, label='Coverage')
    ax.plot(thresholds, [p*100 for p in precs], color='#55A868', lw=2, label='Precision')
    ax.axvline(thr, color='k', linestyle='--', lw=1, alpha=0.5, label='Reference threshold')
    ax.set_xlabel("Distance threshold", fontsize=9)
    ax.set_ylabel("%", fontsize=9)
    ax.set_title("Coverage & Precision\nvs. Distance Threshold", fontsize=10)
    ax.set_ylim(0, 105); ax.legend(fontsize=7); ax.grid(alpha=0.3)

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved → {save_path}")
    else:
        plt.show()
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Test whether generated structures are novel (not in test set)."
    )
    p.add_argument('--ckpt',           required=True)
    p.add_argument('--test',           required=True, help='Path to test .npz')
    p.add_argument('--n',              type=int, default=1000, help='Structures to generate')
    p.add_argument('--steps',          type=int, default=100,  help='DDIM steps')
    p.add_argument('--batch',          type=int, default=256,  help='Generation batch size')
    p.add_argument('--max_ref',        type=int, default=2000, help='Max test structures to compare')
    p.add_argument('--max_atoms',      type=int, default=None)
    p.add_argument('--physics_filter', action='store_true',
                   help='Keep only physically valid structures (bond validity ±0.5 Å + no clash) '
                        'before computing NND. Generates --n total; uses up to --n_valid valid ones.')
    p.add_argument('--n_valid',        type=int, default=1000,
                   help='Target number of valid structures for NND (used with --physics_filter)')
    p.add_argument('--save',           default=None, help='Path to save novelty plot')
    p.add_argument('--out_json',       default=None)
    p.add_argument('--seed',           type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # ── Load test set ──────────────────────────────────────────────────────────
    d         = np.load(args.test)
    coords    = d['coords'].astype(np.float32)
    centroids = d['centroids']
    if centroids.ndim == 2:
        centroids = centroids[:, None, :]
    reference = (coords - centroids).astype(np.float32)
    n_atoms   = reference.shape[1]
    print(f"Test set: {len(reference):,} structures  ({n_atoms} atoms/structure)")

    # ── Load model + generate ─────────────────────────────────────────────────
    model, diffusion, config, coord_scale = load_model_from_ckpt(args.ckpt, device)
    print(f"Generating {args.n} samples…")
    samples = generate(model, diffusion, args.n, n_atoms,
                       coord_scale, args.steps, device, args.batch)
    print(f"Generated: {samples.shape}")

    # ── Optional: keep only physically valid structures ───────────────────────
    n_generated_total = len(samples)
    valid_fraction    = 1.0
    if args.physics_filter and n_atoms <= 20:   # Cα-only physics filter
        _IDEAL_BOND   = 3.832
        _CLASH_CUTOFF = 3.5

        # Bond validity: all 9 consecutive Cα-Cα bonds within ±0.5 Å
        diffs    = np.diff(samples, axis=1)
        lengths  = np.linalg.norm(diffs, axis=-1)           # (N, n_res-1)
        bond_ok  = (np.abs(lengths - _IDEAL_BOND) < 0.5).all(axis=1)

        # Clash: no non-bonded pair (sep ≥ 2) closer than 3.5 Å
        idx  = np.arange(n_atoms)
        sep  = np.abs(idx[:, None] - idx[None, :])
        mask = sep >= 2
        no_clash = np.ones(len(samples), dtype=bool)
        for i in range(len(samples)):
            diff2 = samples[i, :, None, :] - samples[i, None, :, :]
            dist2 = np.linalg.norm(diff2, axis=-1)
            no_clash[i] = not (dist2[mask] < _CLASH_CUTOFF).any()

        valid_mask    = bond_ok & no_clash
        valid_fraction = valid_mask.mean()
        n_valid_found  = valid_mask.sum()
        samples        = samples[valid_mask][:args.n_valid]
        print(f"Physics filter: {n_valid_found}/{n_generated_total} valid "
              f"({valid_fraction*100:.1f}%) — using {len(samples)} for NND")
        if len(samples) == 0:
            print("  WARNING: no physically valid structures — skipping NND analysis")
            import sys; sys.exit(1)

    # ── Auto-select max_atoms ─────────────────────────────────────────────────
    max_atoms = args.max_atoms
    if max_atoms is None:
        max_atoms = n_atoms if n_atoms <= 20 else 30
        print(f"max_atoms={max_atoms} (auto: use --max_atoms to override)")

    # ── Run analysis ──────────────────────────────────────────────────────────
    mt    = config['model_type']
    label = f"{mt} ({n_atoms} atoms)"
    results = run_novelty_analysis(
        samples, reference, label=label,
        max_ref=args.max_ref, max_atoms=max_atoms,
    )

    # ── Save results ──────────────────────────────────────────────────────────
    out = {k: v for k, v in results.items() if not k.startswith('_')}
    out['n_generated_total']  = n_generated_total
    out['valid_fraction']     = valid_fraction
    out['n_used_for_nnd']     = len(samples)
    out['physics_filter_used'] = args.physics_filter
    out_json = args.out_json or str(Path(args.ckpt).parent / 'novelty_metrics.json')
    with open(out_json, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"Metrics → {out_json}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    plot_novelty(results, label=label, save_path=args.save)


if __name__ == '__main__':
    main()
