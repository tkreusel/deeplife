"""
scripts/evaluate_aa.py
=======================
Preliminary evaluation for all-atom Chignolin structure generation.

Uses metrics appropriate for all-atom data (93 atoms) rather than the
Cα-specific metrics in evaluate.py (which assumes bonds at 3.832 Å).

Metrics
-------
  Bond validity   — fraction of structures where ALL 64 identified covalent
                    bonds are within ±tol Å of their data-derived target lengths
  Bond RMSE       — RMS deviation from per-bond target lengths
  Clash rate      — fraction of structures with any heavy-atom pair < 2.5 Å
                    (sequence separation ≥ 4)
  Rg / ete        — radius of gyration and end-to-end distance (still meaningful
                    for global shape; ete = dist between atom 0 and atom 92)
  MMD-RBF         — distribution distance to test set
  Diversity       — mean pairwise RMSD among generated structures

Usage
-----
    python scripts/evaluate_aa.py \\
        --ckpt  checkpoints/egnn_adaln_aa/v3/best.pt \\
        --test  data_all_atom/test.npz \\
        --n     500  --steps 100 \\
        --save  plots/eval_aa.png
"""

import sys, json, argparse
import numpy as np
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.evaluate import load_model_from_ckpt, generate, mmd_rbf
from models.physics_aa import _BOND_INDICES, _BOND_TARGETS, CLASH_CUTOFF, MIN_SEP


# ── Per-bond target array (Å) ─────────────────────────────────────────────────
_TARGETS_NP = np.array(_BOND_TARGETS, dtype=np.float64)
_IDX        = np.array(_BOND_INDICES, dtype=np.int32)


# ─────────────────────────────────────────────────────────────────────────────
# ALL-ATOM METRICS
# ─────────────────────────────────────────────────────────────────────────────

def bond_lengths_aa(coords: np.ndarray) -> np.ndarray:
    """
    Consecutive distances for all 92 pairs.  (N, 93, 3) → (N, 92)
    """
    return np.linalg.norm(np.diff(coords, axis=1), axis=-1)


def bond_validity_aa(coords: np.ndarray, tol: float = 0.1) -> float:
    """
    Fraction of structures where every identified covalent bond is within
    ±tol Å of its data-derived target.  Default tol=0.1 Å (tight).
    """
    bl    = bond_lengths_aa(coords)[:, _IDX]   # (N, 64) — bonded pairs only
    valid = (np.abs(bl - _TARGETS_NP) < tol).all(axis=1)
    return float(valid.mean())


def bond_rmse_aa(coords: np.ndarray) -> float:
    """RMS deviation of the 64 bonded pairs from their targets (Å)."""
    bl = bond_lengths_aa(coords)[:, _IDX]      # (N, 64)
    return float(np.sqrt(((bl - _TARGETS_NP) ** 2).mean()))


def per_bond_rmse_aa(coords: np.ndarray) -> np.ndarray:
    """Per-bond RMSE (Å), length 64 — reveals which bonds are hardest."""
    bl = bond_lengths_aa(coords)[:, _IDX]      # (N, 64)
    return np.sqrt(((bl - _TARGETS_NP) ** 2).mean(axis=0))


def clash_rate_aa(coords: np.ndarray, cutoff: float = CLASH_CUTOFF,
                  min_sep: int = MIN_SEP) -> float:
    """Fraction of structures with any heavy-atom pair < cutoff Å (sep ≥ min_sep)."""
    N_struct, N_atom, _ = coords.shape
    idx  = np.arange(N_atom)
    sep  = np.abs(idx[:, None] - idx[None, :])
    mask = sep >= min_sep

    batch = 128
    clashed = 0
    for i in range(0, N_struct, batch):
        c    = coords[i:i+batch]                       # (b, 93, 3)
        diff = c[:, :, None, :] - c[:, None, :, :]    # (b, 93, 93, 3)
        dist = np.linalg.norm(diff, axis=-1)            # (b, 93, 93)
        clashed += (dist[:, mask] < cutoff).any(axis=1).sum()
    return clashed / N_struct


def radius_of_gyration(coords: np.ndarray) -> np.ndarray:
    com = coords.mean(axis=1, keepdims=True)
    return np.sqrt(((coords - com) ** 2).sum(axis=-1).mean(axis=-1))


def diversity(coords: np.ndarray, n_sub: int = 300) -> float:
    """Mean pairwise RMSD (Å) among a random subset."""
    n   = min(n_sub, len(coords))
    idx = np.random.choice(len(coords), n, replace=False)
    sub = coords[idx]
    sq  = ((sub[:, None] - sub[None]) ** 2).sum(-1).mean(-1)
    tri = np.triu_indices(n, k=1)
    return float(np.sqrt(sq[tri]).mean()) if len(tri[0]) > 0 else 0.0


def compute_all_metrics_aa(samples: np.ndarray, reference: np.ndarray) -> dict:
    rg_s  = radius_of_gyration(samples);   rg_r  = radius_of_gyration(reference)
    ete_s = np.linalg.norm(samples[:, -1]  - samples[:, 0],   axis=-1)
    ete_r = np.linalg.norm(reference[:, -1] - reference[:, 0], axis=-1)

    return {
        # ── Bond geometry ──────────────────────────────────────────────
        'bond_valid_01':   bond_validity_aa(samples, tol=0.1),
        'bond_valid_02':   bond_validity_aa(samples, tol=0.2),
        'bond_rmse':       bond_rmse_aa(samples),
        'bond_rmse_ref':   bond_rmse_aa(reference[:min(2000, len(reference))]),
        'per_bond_rmse':   per_bond_rmse_aa(samples).tolist(),
        # ── Clash ──────────────────────────────────────────────────────
        'clash_rate':      clash_rate_aa(samples),
        'clash_rate_ref':  clash_rate_aa(reference[:500]),
        # ── Global shape ───────────────────────────────────────────────
        'rg_mean':         float(rg_s.mean()),
        'rg_std':          float(rg_s.std()),
        'rg_mean_ref':     float(rg_r.mean()),
        'rg_std_ref':      float(rg_r.std()),
        'ete_mean':        float(ete_s.mean()),
        'ete_std':         float(ete_s.std()),
        'ete_mean_ref':    float(ete_r.mean()),
        # ── Distribution ───────────────────────────────────────────────
        'mmd':             mmd_rbf(samples, reference),
        'diversity':       diversity(samples),
        'diversity_ref':   diversity(reference[:500]),
    }


# ─────────────────────────────────────────────────────────────────────────────
# CONSOLE TABLE
# ─────────────────────────────────────────────────────────────────────────────

def print_table_aa(m: dict, label: str = "Model"):
    w = 34
    print(f"\n{'━'*70}")
    print(f"  {label}  [all-atom, 93 atoms]")
    print(f"{'━'*70}")

    print(f"\n  {'Bond geometry':<{w}} {'Generated':>12} {'Reference':>12}")
    print(f"  {'─'*60}")
    rows = [
        ("Bond validity ±0.1 Å (%)",  f"{m['bond_valid_01']*100:.1f}%",  "—"),
        ("Bond validity ±0.2 Å (%)",  f"{m['bond_valid_02']*100:.1f}%",  "—"),
        ("Bond RMSE (Å)",             f"{m['bond_rmse']:.4f}",           f"{m['bond_rmse_ref']:.4f}"),
        ("Clash rate (%)",            f"{m['clash_rate']*100:.1f}%",     f"{m['clash_rate_ref']*100:.1f}%"),
    ]
    for name, gen, ref in rows:
        print(f"  {name:<{w}} {gen:>12} {ref:>12}")

    print(f"\n  {'Global shape':<{w}} {'Generated':>12} {'Reference':>12}")
    print(f"  {'─'*60}")
    rows2 = [
        ("Rg mean (Å)",               f"{m['rg_mean']:.3f}±{m['rg_std']:.3f}",  f"{m['rg_mean_ref']:.3f}±{m['rg_std_ref']:.3f}"),
        ("End-to-end mean (Å)",       f"{m['ete_mean']:.3f}",                    f"{m['ete_mean_ref']:.3f}"),
        ("MMD-RBF (↓ better)",        f"{m['mmd']:.5f}",                         "—"),
        ("Diversity — RMSD (Å)",      f"{m['diversity']:.3f}",                   f"{m['diversity_ref']:.3f}"),
    ]
    for name, gen, ref in rows2:
        print(f"  {name:<{w}} {gen:>12} {ref:>12}")
    print(f"{'━'*70}")


# ─────────────────────────────────────────────────────────────────────────────
# PLOTTING
# ─────────────────────────────────────────────────────────────────────────────

def plot_aa(samples: np.ndarray, reference: np.ndarray,
            metrics: dict, save_path: str = None):
    try:
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
    except ImportError:
        print("matplotlib not installed — skipping plot")
        return

    fig = plt.figure(figsize=(14, 10))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)
    axes = [fig.add_subplot(gs[r, c]) for r in range(2) for c in range(3)]
    fig.suptitle("All-Atom Generation Quality (93 atoms)", fontsize=13, y=0.98)

    color_gen = '#C44E52'
    color_ref = '#222222'

    # Panel 0: Bond length distribution (all 92 consecutive pairs)
    ax = axes[0]
    bl_gen = bond_lengths_aa(samples).flatten()
    bl_ref = bond_lengths_aa(reference[:2000]).flatten()
    bins = np.linspace(0.8, 5.5, 60)
    ax.hist(bl_ref, bins=bins, density=True, color=color_ref, alpha=0.35, label='Reference')
    ax.hist(bl_gen, bins=bins, density=True, color=color_gen, alpha=0.6,  label='Generated')
    ax.axvline(2.0, color='gray', linestyle='--', lw=1, alpha=0.5, label='Bond/non-bond split')
    ax.set_xlabel("Consecutive atom distance (Å)", fontsize=9)
    ax.set_title("All Consecutive Distances", fontsize=10)
    ax.legend(fontsize=7); ax.grid(alpha=0.3)

    # Panel 1: Bonded-only distribution (64 pairs)
    ax = axes[1]
    bl_gen_b = bond_lengths_aa(samples)[:, _IDX].flatten()
    bl_ref_b = bond_lengths_aa(reference[:2000])[:, _IDX].flatten()
    bins2 = np.linspace(1.0, 2.0, 50)
    ax.hist(bl_ref_b, bins=bins2, density=True, color=color_ref, alpha=0.35, label='Reference')
    ax.hist(bl_gen_b, bins=bins2, density=True, color=color_gen, alpha=0.6,  label='Generated')
    ax.set_xlabel("Covalent bond length (Å)", fontsize=9)
    ax.set_title("Covalent Bond Distribution (64 bonds)", fontsize=10)
    ax.legend(fontsize=7); ax.grid(alpha=0.3)

    # Panel 2: Per-bond RMSE
    ax = axes[2]
    pb_gen = np.array(metrics['per_bond_rmse'])
    pb_ref = per_bond_rmse_aa(reference[:2000])
    ax.plot(range(len(pb_gen)), pb_gen, 'o-', color=color_gen, alpha=0.8, lw=1.2,
            markersize=3, label='Generated')
    ax.plot(range(len(pb_ref)), pb_ref, 'o--', color=color_ref, alpha=0.5, lw=1.2,
            markersize=3, label='Reference')
    ax.set_xlabel("Bond index (of 64 covalent bonds)", fontsize=9)
    ax.set_ylabel("RMSE (Å)", fontsize=9)
    ax.set_title("Per-Bond RMSE from Ideal", fontsize=10)
    ax.legend(fontsize=7); ax.grid(alpha=0.3)

    # Panel 3: Radius of gyration
    ax = axes[3]
    rg_gen = radius_of_gyration(samples)
    rg_ref = radius_of_gyration(reference)
    ax.hist(rg_ref, bins=40, density=True, color=color_ref, alpha=0.35, label='Reference')
    ax.hist(rg_gen, bins=40, density=True, color=color_gen, alpha=0.6,  label='Generated')
    ax.set_xlabel("Rg (Å)", fontsize=9)
    ax.set_title("Radius of Gyration", fontsize=10)
    ax.legend(fontsize=7); ax.grid(alpha=0.3)

    # Panel 4: End-to-end distance
    ax = axes[4]
    ete_gen = np.linalg.norm(samples[:, -1] - samples[:, 0], axis=-1)
    ete_ref = np.linalg.norm(reference[:, -1] - reference[:, 0], axis=-1)
    ax.hist(ete_ref, bins=40, density=True, color=color_ref, alpha=0.35, label='Reference')
    ax.hist(ete_gen, bins=40, density=True, color=color_gen, alpha=0.6,  label='Generated')
    ax.set_xlabel("End-to-end distance (Å)", fontsize=9)
    ax.set_title("End-to-End Distance", fontsize=10)
    ax.legend(fontsize=7); ax.grid(alpha=0.3)

    # Panel 5: Bond validity at multiple tolerances (bar chart)
    ax = axes[5]
    tols = [0.05, 0.10, 0.15, 0.20]
    vals_gen = [bond_validity_aa(samples,    t)*100 for t in tols]
    vals_ref = [bond_validity_aa(reference[:2000], t)*100 for t in tols]
    x_pos = np.arange(len(tols))
    ax.bar(x_pos - 0.2, vals_gen, 0.38, color=color_gen, alpha=0.8, label='Generated')
    ax.bar(x_pos + 0.2, vals_ref, 0.38, color=color_ref, alpha=0.5, label='Reference')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'±{t} Å' for t in tols], fontsize=8)
    ax.set_ylabel("Bond validity (%)", fontsize=9)
    ax.set_title("Bond Validity at Multiple Tolerances", fontsize=10)
    ax.set_ylim(0, 105)
    ax.legend(fontsize=7); ax.grid(alpha=0.3, axis='y')

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
        description="Evaluate all-atom generation quality with appropriate metrics."
    )
    p.add_argument('--ckpt',   required=True, help='Checkpoint path')
    p.add_argument('--test',   required=True, help='Path to all-atom test.npz')
    p.add_argument('--n',      type=int, default=500, help='Structures to generate')
    p.add_argument('--steps',  type=int, default=100, help='DDIM steps')
    p.add_argument('--batch',  type=int, default=64,  help='Generation batch size')
    p.add_argument('--save',   default=None,           help='Path to save plot')
    p.add_argument('--out_json', default=None)
    p.add_argument('--seed',   type=int, default=0)
    args = p.parse_args()

    import torch, numpy as np
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
    print(f"Test set: {len(reference):,} all-atom structures  ({reference.shape[1]} atoms)")

    # ── Load model + generate ─────────────────────────────────────────────────
    model, diffusion, config, coord_scale = load_model_from_ckpt(args.ckpt, device)
    n_atoms = config['data']['n_residues']
    print(f"Generating {args.n} samples ({n_atoms} atoms, {args.steps} DDIM steps)…")

    samples = generate(model, diffusion, args.n, n_atoms,
                       coord_scale, args.steps, device, args.batch)

    # ── Compute metrics ────────────────────────────────────────────────────────
    print("Computing all-atom metrics…")
    metrics = compute_all_metrics_aa(samples, reference)
    print_table_aa(metrics, label=f"egnn_adaln_aa  [epoch={config.get('epoch','?')}]")

    # ── Save JSON ──────────────────────────────────────────────────────────────
    out_json = args.out_json or str(Path(args.ckpt).parent / 'eval_aa_metrics.json')
    with open(out_json, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics → {out_json}")

    # ── Plot ───────────────────────────────────────────────────────────────────
    plot_aa(samples, reference, metrics, save_path=args.save)


if __name__ == '__main__':
    main()
