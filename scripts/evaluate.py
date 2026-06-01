"""
scripts/evaluate.py
====================
Comprehensive evaluation: generate samples from one or two checkpoints
and compare them against the test set with structural metrics.

Handles BOTH baseline (mlp/transformer) and EGNN checkpoints automatically
by reading model_type from the saved config.

Usage
-----
# Evaluate EGNN only
python scripts/evaluate.py \\
    --ckpt checkpoints/egnn/v1/best.pt \\
    --test data/test.npz \\
    --n 1000

# Compare baseline vs EGNN side-by-side
python scripts/evaluate.py \\
    --ckpt     checkpoints/baseline/v1/best.pt \\
    --ckpt_ref checkpoints/egnn/v1/best.pt \\
    --test     data/test.npz \\
    --n 1000

# Save PDB files + plot + metrics JSON
python scripts/evaluate.py \\
    --ckpt checkpoints/egnn/v1/best.pt \\
    --test data/test.npz \\
    --n 500 --save_pdb outputs/egnn_samples --save plots/eval.png

Output
------
  - Console table: bond validity, Rg, MMD, end-to-end distance
  - plots/eval.png: 4-panel comparison plot
  - metrics.json: all numbers for further analysis
  - (optional) PDB files for PyMOL visualisation
"""

import sys
import json
import argparse
import numpy as np
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.diffusion_zerocom import ZeroCoMGaussianDiffusion
from models.diffusion          import GaussianDiffusion


# ─────────────────────────────────────────────────────────────────────────────
# MODEL LOADING  (auto-detects model_type from checkpoint)
# ─────────────────────────────────────────────────────────────────────────────

def load_model_from_ckpt(ckpt_path: str, device: str):
    """
    Load model + diffusion from any checkpoint in this repo.
    Reads model_type from the saved config to pick the right class.

    Returns (model, diffusion, config, coord_scale)
    """
    ckpt   = torch.load(ckpt_path, map_location=device)
    config = ckpt['config']
    mt     = config['model_type']

    # ── Build the model ────────────────────────────────────────────────────
    if mt == 'egnn':
        from models.egnn import EGNNScoreNetwork
        mc    = config['model']
        model = EGNNScoreNetwork(
            n_residues = config['data']['n_residues'],
            node_dim   = mc['hidden_dim'],
            edge_dim   = mc.get('edge_dim', 64),
            time_dim   = mc['time_dim'],
            n_layers   = mc['n_layers'],
        )
        DiffClass = ZeroCoMGaussianDiffusion
    elif mt in ('mlp', 'transformer'):
        from scripts.train import build_model
        model     = build_model(config)
        DiffClass = GaussianDiffusion
    else:
        raise ValueError(f"Unknown model_type: {mt!r}")

    # Load EMA weights (best quality)
    model.load_state_dict(ckpt['ema_shadow'])
    model = model.to(device)
    model.eval()

    dc        = config['diffusion']
    diffusion = DiffClass(T=dc['T'], schedule=dc['schedule']).to(device)

    coord_scale = config['data'].get('coord_scale', 16.32)
    epoch       = ckpt.get('epoch', '?')
    val_loss    = ckpt.get('best_val_loss', float('nan'))

    print(f"  Loaded {mt} checkpoint — epoch {epoch}, val_loss={val_loss:.4f}")
    return model, diffusion, config, coord_scale


# ─────────────────────────────────────────────────────────────────────────────
# GENERATION
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def generate(model, diffusion, n: int, n_residues: int,
             coord_scale: float, ddim_steps: int, device: str,
             batch_size: int = 256) -> np.ndarray:
    """
    Generate n structures, rescale to Ångströms, return (n, N, 3).
    """
    model.eval()
    all_samples = []
    n_done = 0

    while n_done < n:
        bs    = min(batch_size, n - n_done)
        shape = (bs, n_residues, 3)
        x     = diffusion.ddim_sample(model, shape, device=device,
                                       ddim_steps=ddim_steps)
        # center + rescale to Ångströms
        x = x - x.mean(dim=1, keepdim=True)
        x = x * coord_scale
        all_samples.append(x.cpu().numpy())
        n_done += bs

    return np.concatenate(all_samples, axis=0)[:n].astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────────────────────────────────────

def bond_lengths(coords: np.ndarray) -> np.ndarray:
    """Consecutive Cα–Cα distances. coords: (N, n_res, 3) → (N, n_res-1)."""
    return np.linalg.norm(np.diff(coords, axis=1), axis=-1)


def radius_of_gyration(coords: np.ndarray) -> np.ndarray:
    """Rg per structure. coords already zero-CoM → (N,)."""
    com = coords.mean(axis=1, keepdims=True)
    return np.sqrt(((coords - com) ** 2).sum(axis=-1).mean(axis=-1))


def validity(coords: np.ndarray, tol: float = 0.5, ideal: float = 3.8) -> float:
    """Fraction of structures where ALL bonds are within ±tol Å of ideal."""
    bl    = bond_lengths(coords)
    valid = (np.abs(bl - ideal) < tol).all(axis=1)
    return float(valid.mean())


def mmd_rbf(x: np.ndarray, y: np.ndarray,
            sigmas=(0.5, 1.0, 2.0)) -> float:
    """
    Maximum Mean Discrepancy with RBF kernel, averaged over several bandwidths.
    x, y : (N, n_res, 3)  → flattened to (N, n_res*3)
    Lower = more similar distributions.
    """
    x = x.reshape(len(x), -1).astype(np.float64)
    y = y.reshape(len(y), -1).astype(np.float64)

    # sub-sample for speed
    max_pts = 2000
    if len(x) > max_pts: x = x[np.random.choice(len(x), max_pts, replace=False)]
    if len(y) > max_pts: y = y[np.random.choice(len(y), max_pts, replace=False)]

    def rbf(a, b, sig):
        sq = ((a[:, None] - b[None]) ** 2).sum(-1)
        return np.exp(-sq / (2 * sig ** 2))

    vals = []
    for s in sigmas:
        Kxx = rbf(x, x, s); Kyy = rbf(y, y, s); Kxy = rbf(x, y, s)
        n, m = len(x), len(y)
        v = ((Kxx.sum() - np.diag(Kxx).sum()) / (n * (n-1))
           + (Kyy.sum() - np.diag(Kyy).sum()) / (m * (m-1))
           - 2 * Kxy.mean())
        vals.append(v)
    return float(np.mean(vals))


def compute_all_metrics(samples: np.ndarray, reference: np.ndarray) -> dict:
    bl_s = bond_lengths(samples);   bl_r = bond_lengths(reference)
    rg_s = radius_of_gyration(samples); rg_r = radius_of_gyration(reference)
    ete_s = np.linalg.norm(samples[:, -1] - samples[:, 0], axis=-1)
    ete_r = np.linalg.norm(reference[:, -1] - reference[:, 0], axis=-1)
    return {
        'validity_gen':    validity(samples),
        'validity_ref':    validity(reference),
        'bond_mean_gen':   float(bl_s.mean()),
        'bond_mean_ref':   float(bl_r.mean()),
        'bond_std_gen':    float(bl_s.std()),
        'bond_std_ref':    float(bl_r.std()),
        'rg_mean_gen':     float(rg_s.mean()),
        'rg_mean_ref':     float(rg_r.mean()),
        'rg_std_gen':      float(rg_s.std()),
        'rg_std_ref':      float(rg_r.std()),
        'ete_mean_gen':    float(ete_s.mean()),
        'ete_mean_ref':    float(ete_r.mean()),
        'mmd':             mmd_rbf(samples, reference),
    }


def print_table(metrics: dict, label: str = "Model"):
    w = 30
    print(f"\n{'─'*60}")
    print(f"  {label}")
    print(f"{'─'*60}")
    print(f"  {'Metric':<{w}} {'Generated':>12} {'Reference':>12}")
    print(f"  {'─'*56}")
    rows = [
        ("Bond validity (%)",    f"{metrics['validity_gen']*100:.1f}%",  f"{metrics['validity_ref']*100:.1f}%"),
        ("Bond length mean (Å)", f"{metrics['bond_mean_gen']:.3f}",       f"{metrics['bond_mean_ref']:.3f}"),
        ("Bond length std  (Å)", f"{metrics['bond_std_gen']:.3f}",        f"{metrics['bond_std_ref']:.3f}"),
        ("Rg mean (Å)",          f"{metrics['rg_mean_gen']:.3f}",         f"{metrics['rg_mean_ref']:.3f}"),
        ("Rg std  (Å)",          f"{metrics['rg_std_gen']:.3f}",          f"{metrics['rg_std_ref']:.3f}"),
        ("End-to-end mean (Å)",  f"{metrics['ete_mean_gen']:.3f}",        f"{metrics['ete_mean_ref']:.3f}"),
        ("MMD (↓ better)",       f"{metrics['mmd']:.5f}",                 "—"),
    ]
    for name, gen, ref in rows:
        print(f"  {name:<{w}} {gen:>12} {ref:>12}")
    print(f"{'─'*60}")


# ─────────────────────────────────────────────────────────────────────────────
# PLOTTING
# ─────────────────────────────────────────────────────────────────────────────

def plot_comparison(samples_dict: dict, reference: np.ndarray, save_path: str = None):
    """
    samples_dict : {'ModelA': array, 'ModelB': array, ...}
    reference    : test set array
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — skipping plot")
        return

    colors = ['#C44E52', '#4C72B0', '#55A868', '#8172B3']
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Generated vs Reference Structures", fontsize=13)

    def hist(ax, data_dict, ref, fn, xlabel, title, bins=40):
        ref_vals = fn(ref)
        ax.hist(ref_vals.flatten(), bins=bins, density=True,
                color='#222222', alpha=0.3, label='Reference (test set)')
        for (label, samples), color in zip(data_dict.items(), colors):
            vals = fn(samples)
            ax.hist(vals.flatten(), bins=bins, density=True,
                    color=color, alpha=0.6, label=label)
        ax.set_xlabel(xlabel); ax.set_title(title)
        ax.legend(fontsize=8)

    hist(axes[0,0], samples_dict, reference,
         lambda x: bond_lengths(x).flatten(),
         "Cα–Cα distance (Å)", "Bond Length Distribution")
    axes[0,0].axvline(3.3, color='k', linestyle='--', lw=1, alpha=0.5)
    axes[0,0].axvline(4.3, color='k', linestyle='--', lw=1, alpha=0.5)

    hist(axes[0,1], samples_dict, reference,
         radius_of_gyration,
         "Radius of gyration (Å)", "Radius of Gyration")

    hist(axes[1,0], samples_dict, reference,
         lambda x: np.linalg.norm(x[:, -1] - x[:, 0], axis=-1),
         "End-to-end distance (Å)", "End-to-End Distance")

    # Per-residue flexibility
    ax = axes[1,1]
    ref_var = reference.var(axis=0).sum(axis=-1)
    ax.plot(range(len(ref_var)), ref_var, 'o-', color='#222222',
            alpha=0.5, label='Reference', lw=2)
    for (label, samples), color in zip(samples_dict.items(), colors):
        var = samples.var(axis=0).sum(axis=-1)
        ax.plot(range(len(var)), var, 'o-', color=color, alpha=0.8, label=label)
    ax.set_xticks(range(len(ref_var)))
    ax.set_xticklabels([f"R{i+1}" for i in range(len(ref_var))], rotation=45)
    ax.set_ylabel("Positional variance (Å²)")
    ax.set_title("Per-Residue Flexibility")
    ax.legend(fontsize=8)

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved → {save_path}")
    else:
        plt.show()
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# PDB EXPORT  (identical to quick_sample.py but shared here)
# ─────────────────────────────────────────────────────────────────────────────

def save_pdbs(samples: np.ndarray, out_dir: str, n_save: int = 10):
    out      = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    sequence = "YYDPETGTWG"
    aa3      = {'Y':'TYR','D':'ASP','P':'PRO','E':'GLU','T':'THR',
                'G':'GLY','W':'TRP','A':'ALA','K':'LYS','R':'ARG'}

    for i, coords in enumerate(samples[:n_save]):
        lines = [f"REMARK  EGNN generated sample {i+1}\n"]
        for j, (res, xyz) in enumerate(zip(sequence, coords)):
            x, y, z = xyz
            rn = aa3.get(res, 'GLY')
            lines.append(
                f"ATOM  {j+1:5d}  CA  {rn} A{j+1:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C\n"
            )
        lines.append("END\n")
        with open(out / f"sample_{i+1:04d}.pdb", 'w') as f:
            f.writelines(lines)

    print(f"Saved {min(n_save, len(samples))} PDB files → {out_dir}")
    print(f"Visualise: pymol {out_dir}/*.pdb")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--ckpt',     required=True,
                   help='Primary checkpoint (EGNN or baseline)')
    p.add_argument('--ckpt_ref', default=None,
                   help='Optional second checkpoint for side-by-side comparison')
    p.add_argument('--test',     required=True,
                   help='Path to test.npz')
    p.add_argument('--n',        type=int, default=500,
                   help='Number of structures to generate')
    p.add_argument('--steps',    type=int, default=100,
                   help='DDIM steps (more = slower, better quality)')
    p.add_argument('--save_pdb', default=None,
                   help='Directory to write PDB files')
    p.add_argument('--save',     default=None,
                   help='Path to save comparison plot (e.g. plots/eval.png)')
    p.add_argument('--out_json', default=None,
                   help='Path to save metrics JSON')
    p.add_argument('--batch',    type=int, default=256)
    p.add_argument('--seed',     type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # ── Load test set ─────────────────────────────────────────────────────────
    test_data  = np.load(args.test)
    reference  = test_data['coords'].astype(np.float32)
    centroids  = test_data['centroids']
    if centroids.ndim == 2:
        centroids = centroids[:, None, :]
    reference  = reference - centroids          # center with saved centroids
    print(f"Test set: {len(reference):,} structures")

    # ── Generate from primary checkpoint ─────────────────────────────────────
    print(f"\nLoading primary checkpoint: {args.ckpt}")
    model, diffusion, config, scale = load_model_from_ckpt(args.ckpt, device)
    label = config['model_type'].upper()

    print(f"Generating {args.n} samples ({label}) …")
    samples = generate(model, diffusion, args.n,
                       config['data']['n_residues'], scale,
                       args.steps, device, args.batch)

    samples_dict = {label: samples}
    metrics_all  = {label: compute_all_metrics(samples, reference)}
    print_table(metrics_all[label], label=label)

    # ── Optional: second checkpoint ───────────────────────────────────────────
    if args.ckpt_ref:
        print(f"\nLoading reference checkpoint: {args.ckpt_ref}")
        m2, d2, cfg2, scale2 = load_model_from_ckpt(args.ckpt_ref, device)
        label2 = cfg2['model_type'].upper() + "_ref"

        print(f"Generating {args.n} samples ({label2}) …")
        samples2 = generate(m2, d2, args.n,
                             cfg2['data']['n_residues'], scale2,
                             args.steps, device, args.batch)

        samples_dict[label2] = samples2
        metrics_all[label2]  = compute_all_metrics(samples2, reference)
        print_table(metrics_all[label2], label=label2)

    # ── Metrics JSON ──────────────────────────────────────────────────────────
    out_json = args.out_json or str(Path(args.ckpt).parent / 'eval_metrics.json')
    with open(out_json, 'w') as f:
        json.dump(metrics_all, f, indent=2)
    print(f"\nMetrics → {out_json}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    plot_comparison(samples_dict, reference, save_path=args.save)

    # ── PDB files ─────────────────────────────────────────────────────────────
    if args.save_pdb:
        save_pdbs(samples, args.save_pdb, n_save=20)


if __name__ == '__main__':
    main()
