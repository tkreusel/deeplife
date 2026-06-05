"""
scripts/sample_novel.py
========================
For a given energy-conditioned model, generate structures at multiple
temperature levels (τ), rank each by novelty (NND to test set), report
physical validity, and save the most novel structures as PDB files.

Answers three questions per temperature:
  1. How novel are the structures?  (NND ratio vs within-test baseline)
  2. Are the novel structures physically valid?  (bond validity + clash)
  3. Which specific structures are most novel?  (saved as PDB files)

Usage
-----
    python scripts/sample_novel.py \\
        --ckpt  checkpoints/egnn_adaln/v1/best.pt \\
        --test  data/test.npz \\
        --temperatures 0.0 0.5 1.0 \\
        --n 500 --steps 100 --guidance_scale 2.0 \\
        --top_k 20 \\
        --save_pdb outputs/novel_ca \\
        --save    plots/novelty_by_tau.png
"""

import sys, json, argparse
import numpy as np
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.analyze_energy_conditioning import (
    load_energy_model, generate_at_temperature,
)
from scripts.evaluate_novelty import (
    distance_fingerprint, nearest_neighbour_distances,
)

# ── Physical validity metrics (Cα-specific) ──────────────────────────────────
_IDEAL_BOND   = 3.832
_CLASH_CUTOFF = 3.5


def bond_validity(coords: np.ndarray, tol: float = 0.5) -> np.ndarray:
    """Per-structure bond validity flag (True = ALL bonds within ±tol Å)."""
    diffs   = np.diff(coords, axis=1)
    lengths = np.linalg.norm(diffs, axis=-1)        # (N, n_res-1)
    return (np.abs(lengths - _IDEAL_BOND) < tol).all(axis=1)  # (N,)


def bond_rmse(coords: np.ndarray) -> np.ndarray:
    """Per-structure Cα–Cα bond RMSE (Å)."""
    diffs   = np.diff(coords, axis=1)
    lengths = np.linalg.norm(diffs, axis=-1)
    return np.sqrt(((lengths - _IDEAL_BOND) ** 2).mean(axis=1))  # (N,)


def has_clash(coords: np.ndarray) -> np.ndarray:
    """Per-structure clash flag (True = at least one non-bonded pair < 3.5 Å)."""
    N_struct, N_res, _ = coords.shape
    idx  = np.arange(N_res)
    sep  = np.abs(idx[:, None] - idx[None, :])
    mask = sep >= 2                                   # non-bonded pairs only
    result = np.zeros(N_struct, dtype=bool)
    for i in range(N_struct):
        diff = coords[i, :, None, :] - coords[i, None, :, :]
        dist = np.linalg.norm(diff, axis=-1)
        result[i] = (dist[mask] < _CLASH_CUTOFF).any()
    return result


# ── PDB export (Cα-only, 10 residues) ────────────────────────────────────────

def save_pdb(coords: np.ndarray, path: str, label: str = "novel"):
    """Save a single Cα-only chignolin structure as a PDB file."""
    sequence = "YYDPETGTWG"
    aa3 = {'Y':'TYR','D':'ASP','P':'PRO','E':'GLU','T':'THR',
           'G':'GLY','W':'TRP','A':'ALA','K':'LYS','R':'ARG'}
    lines = [f"REMARK  {label}\n"]
    for j, (res, xyz) in enumerate(zip(sequence, coords)):
        x, y, z = xyz
        rn = aa3.get(res, 'GLY')
        lines.append(
            f"ATOM  {j+1:5d}  CA  {rn} A{j+1:4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C\n"
        )
    lines.append("END\n")
    with open(path, 'w') as f:
        f.writelines(lines)


# ── Main analysis per temperature ─────────────────────────────────────────────

def analyse_temperature(
    model, diffusion, tau: float,
    n: int, n_residues: int, coord_scale: float, ddim_steps: int,
    guidance_scale: float, device: str,
    reference: np.ndarray, max_ref: int, max_atoms: int,
    batch_size: int = 128,
) -> dict:
    """Generate at temperature τ, compute NND + physical validity."""

    # ── Generate ──────────────────────────────────────────────────────────────
    print(f"\n  τ={tau:.2f} → generating {n} structures …")
    samples = generate_at_temperature(
        model, diffusion, tau=tau, n=n,
        n_residues=n_residues, coord_scale=coord_scale,
        ddim_steps=ddim_steps, guidance_scale=guidance_scale,
        device=device, batch_size=batch_size,
        is_x1pred=getattr(diffusion, '_sampler_tag', False),
    )

    # ── Rotation-invariant fingerprints ───────────────────────────────────────
    ref_sub = reference[np.random.choice(len(reference),
                        min(max_ref, len(reference)), replace=False)]
    gen_fp  = distance_fingerprint(samples,  max_atoms=max_atoms)
    ref_fp  = distance_fingerprint(ref_sub,  max_atoms=max_atoms)

    # ── NND: generated → test ─────────────────────────────────────────────────
    gen_nnd  = nearest_neighbour_distances(gen_fp, ref_fp)   # (N,)

    # ── NND: test → test (baseline) ──────────────────────────────────────────
    n_self  = min(500, len(ref_sub))
    idx_a   = np.random.choice(len(ref_sub), n_self, replace=False)
    mask    = np.ones(len(ref_sub), dtype=bool); mask[idx_a] = False
    ref_b   = ref_sub[mask][:n_self]
    self_nnd = nearest_neighbour_distances(ref_fp[idx_a], distance_fingerprint(ref_b, max_atoms=max_atoms))

    threshold = float(np.median(self_nnd))
    nnd_ratio = float(gen_nnd.mean() / max(self_nnd.mean(), 1e-8))
    coverage  = float((nearest_neighbour_distances(ref_fp, gen_fp) < threshold).mean())
    precision = float((gen_nnd < threshold).mean())

    # ── Physical validity for each structure ──────────────────────────────────
    valid_flags = bond_validity(samples)     # (N,) bool
    clash_flags = has_clash(samples)         # (N,) bool
    rmse_vals   = bond_rmse(samples)         # (N,) float

    # ── Sort by NND descending (most novel first) ─────────────────────────────
    order = np.argsort(gen_nnd)[::-1]        # most novel first

    return {
        'tau':           tau,
        'samples':       samples,
        'gen_nnd':       gen_nnd,
        'self_nnd':      self_nnd,
        'nnd_ratio':     nnd_ratio,
        'coverage':      coverage,
        'precision':     precision,
        'threshold':     threshold,
        'valid_flags':   valid_flags,
        'clash_flags':   clash_flags,
        'rmse_vals':     rmse_vals,
        'sorted_idx':    order,          # most novel → least novel
    }


# ── Console report ────────────────────────────────────────────────────────────

def print_report(r: dict):
    tau = r['tau']
    print(f"\n  {'━'*62}")
    print(f"  τ={tau:.2f}  NND ratio={r['nnd_ratio']:.3f}  "
          f"Coverage={r['coverage']*100:.1f}%  Precision={r['precision']*100:.1f}%")
    print(f"  {'━'*62}")

    # Top-10 most novel
    top = r['sorted_idx'][:10]
    print(f"  Top-10 most novel structures (highest NND from test set):")
    print(f"  {'Rank':>4}  {'NND':>8}  {'BondRMSE':>9}  {'Valid':>6}  {'Clash':>6}")
    print(f"  {'─'*44}")
    for rank, idx in enumerate(top, 1):
        valid = "✓" if r['valid_flags'][idx] else "✗"
        clash = "✗" if r['clash_flags'][idx] else "✓"
        print(f"  {rank:>4}  {r['gen_nnd'][idx]:>8.3f}  "
              f"{r['rmse_vals'][idx]:>9.4f}  {valid:>6}  {clash:>6}")

    # Summary: among top-K novel, how many are physically valid?
    k = min(50, len(r['sorted_idx']))
    top_k = r['sorted_idx'][:k]
    n_valid     = r['valid_flags'][top_k].sum()
    n_no_clash  = (~r['clash_flags'][top_k]).sum()
    n_both      = (r['valid_flags'][top_k] & ~r['clash_flags'][top_k]).sum()
    print(f"\n  Among top-{k} most novel structures:")
    print(f"    Bond valid  : {n_valid}/{k}  ({n_valid/k*100:.0f}%)")
    print(f"    No clash    : {n_no_clash}/{k}  ({n_no_clash/k*100:.0f}%)")
    print(f"    Both valid  : {n_both}/{k}  ({n_both/k*100:.0f}%)")


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_results(results: list, reference: np.ndarray, save_path: str = None):
    try:
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
    except ImportError:
        print("matplotlib not installed — skipping plot"); return

    n_tau  = len(results)
    colors = ['#4C72B0', '#55A868', '#C44E52', '#8172B3']

    fig = plt.figure(figsize=(15, 10))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)
    fig.suptitle("Novelty vs Physical Validity by Temperature", fontsize=13)

    # Panel 0: NND distributions per τ
    ax = fig.add_subplot(gs[0, 0])
    for i, r in enumerate(results):
        ax.hist(r['gen_nnd'], bins=40, density=True, alpha=0.6,
                color=colors[i % len(colors)], label=f"τ={r['tau']:.2f}")
    # within-test baseline (same for all)
    ax.hist(results[0]['self_nnd'], bins=40, density=True, alpha=0.3,
            color='#222222', label='Test→Test')
    ax.set_xlabel("NND to test set", fontsize=9)
    ax.set_title("Novelty Distribution by τ\n(right = more novel)", fontsize=10)
    ax.legend(fontsize=7); ax.grid(alpha=0.3)

    # Panel 1: NND ratio bar chart
    ax = fig.add_subplot(gs[0, 1])
    taus   = [r['tau'] for r in results]
    ratios = [r['nnd_ratio'] for r in results]
    ax.bar(range(n_tau), ratios,
           color=[colors[i % len(colors)] for i in range(n_tau)], alpha=0.8)
    ax.axhline(1.0, color='k', linestyle='--', lw=1.5, alpha=0.5, label='= test distribution')
    ax.set_xticks(range(n_tau)); ax.set_xticklabels([f'τ={t}' for t in taus])
    ax.set_ylabel("NND ratio (gen / self)", fontsize=9)
    ax.set_title("Novelty Ratio by Temperature\n(>1 = novel, <1 = memorised)", fontsize=10)
    ax.legend(fontsize=7); ax.grid(alpha=0.3, axis='y')

    # Panel 2: Physical validity among top-50 novel structures
    ax = fig.add_subplot(gs[0, 2])
    k = 50
    valid_pct = []
    noclash_pct = []
    both_pct  = []
    for r in results:
        top_k = r['sorted_idx'][:k]
        valid_pct.append(r['valid_flags'][top_k].mean() * 100)
        noclash_pct.append((~r['clash_flags'][top_k]).mean() * 100)
        both_pct.append((r['valid_flags'][top_k] & ~r['clash_flags'][top_k]).mean() * 100)
    x = np.arange(n_tau)
    w = 0.25
    ax.bar(x - w,   valid_pct,   w, label='Bond valid',  alpha=0.8, color='#4C72B0')
    ax.bar(x,       noclash_pct, w, label='No clash',    alpha=0.8, color='#55A868')
    ax.bar(x + w,   both_pct,    w, label='Both valid',  alpha=0.8, color='#C44E52')
    ax.set_xticks(x); ax.set_xticklabels([f'τ={t}' for t in taus])
    ax.set_ylabel("% valid (top-50 novel)", fontsize=9)
    ax.set_title(f"Physical Validity of Top-{k} Novel Structures", fontsize=10)
    ax.set_ylim(0, 105); ax.legend(fontsize=7); ax.grid(alpha=0.3, axis='y')

    # Panel 3: NND vs bond RMSE scatter (most novel τ)
    ax = fig.add_subplot(gs[1, 0])
    for i, r in enumerate(results):
        sc = ax.scatter(r['gen_nnd'], r['rmse_vals'], s=8, alpha=0.4,
                       color=colors[i % len(colors)], label=f"τ={r['tau']:.2f}")
    ax.set_xlabel("NND (novelty)", fontsize=9)
    ax.set_ylabel("Bond RMSE (Å)", fontsize=9)
    ax.set_title("Novel ↔ Valid Trade-off\n(bottom-right = novel AND valid)", fontsize=10)
    ax.legend(fontsize=7); ax.grid(alpha=0.3)

    # Panel 4: Bond RMSE distribution for top-50 novel per τ
    ax = fig.add_subplot(gs[1, 1])
    for i, r in enumerate(results):
        top_k = r['sorted_idx'][:50]
        ax.hist(r['rmse_vals'][top_k], bins=20, density=True, alpha=0.6,
                color=colors[i % len(colors)], label=f"τ={r['tau']:.2f}")
    ax.axvline(0.5, color='k', linestyle='--', lw=1, alpha=0.5, label='±0.5 Å limit')
    ax.set_xlabel("Bond RMSE (Å)", fontsize=9)
    ax.set_title("Bond RMSE of Top-50 Novel Structures", fontsize=10)
    ax.legend(fontsize=7); ax.grid(alpha=0.3)

    # Panel 5: Coverage and Precision vs τ
    ax = fig.add_subplot(gs[1, 2])
    covs  = [r['coverage']*100  for r in results]
    precs = [r['precision']*100 for r in results]
    ax.plot(taus, covs,  'o-', color='#4C72B0', lw=2, markersize=8, label='Coverage')
    ax.plot(taus, precs, 's-', color='#C44E52', lw=2, markersize=8, label='Precision')
    ax.set_xlabel("Temperature τ", fontsize=9)
    ax.set_ylabel("%", fontsize=9)
    ax.set_title("Coverage & Precision vs Temperature", fontsize=10)
    ax.set_ylim(0, 105); ax.legend(fontsize=7); ax.grid(alpha=0.3)

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nPlot saved → {save_path}")
    else:
        plt.show()
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Novelty + physical validity analysis across energy levels."
    )
    p.add_argument('--ckpt',           required=True)
    p.add_argument('--test',           required=True)
    p.add_argument('--temperatures',   nargs='+', type=float, default=[0.0, 0.5, 1.0])
    p.add_argument('--n',              type=int,   default=500)
    p.add_argument('--steps',          type=int,   default=100)
    p.add_argument('--guidance_scale', type=float, default=2.0)
    p.add_argument('--top_k',          type=int,   default=20,
                   help='How many most-novel PDB files to save per temperature')
    p.add_argument('--max_ref',        type=int,   default=2000)
    p.add_argument('--max_atoms',      type=int,   default=None)
    p.add_argument('--batch',          type=int,   default=128)
    p.add_argument('--save_pdb',       default=None,
                   help='Directory prefix for PDB output (τ appended automatically)')
    p.add_argument('--save',           default=None, help='Path for summary plot')
    p.add_argument('--out_json',       default=None)
    p.add_argument('--seed',           type=int,   default=0)
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
    if centroids.ndim == 2: centroids = centroids[:, None, :]
    reference = (coords - centroids).astype(np.float32)
    n_atoms   = reference.shape[1]
    print(f"Test set: {len(reference):,} structures  ({n_atoms} atoms)")

    # ── Load energy-conditioned model ─────────────────────────────────────────
    model, diffusion, config, coord_scale, energy_mean, energy_std, sampler_tag = \
        load_energy_model(args.ckpt, device)
    # store sampler_tag on diffusion so analyse_temperature can access it
    diffusion._sampler_tag = sampler_tag

    max_atoms = args.max_atoms or (n_atoms if n_atoms <= 20 else 30)

    # ── Run analysis per temperature ──────────────────────────────────────────
    results = []
    for tau in args.temperatures:
        r = analyse_temperature(
            model, diffusion, tau=tau,
            n=args.n, n_residues=n_atoms, coord_scale=coord_scale,
            ddim_steps=args.steps, guidance_scale=args.guidance_scale,
            device=device, reference=reference,
            max_ref=args.max_ref, max_atoms=max_atoms,
            batch_size=args.batch,
        )
        print_report(r)
        results.append(r)

        # ── Save most novel structures as PDB files ────────────────────────────
        if args.save_pdb:
            tau_str  = f"{tau:.2f}".replace('.', 'p')
            pdb_dir  = Path(f"{args.save_pdb}_tau{tau_str}")
            pdb_dir.mkdir(parents=True, exist_ok=True)

            top_idx  = r['sorted_idx'][:args.top_k]
            for rank, idx in enumerate(top_idx, 1):
                valid_str = "valid" if r['valid_flags'][idx] else "invalid"
                clash_str = "noclash" if not r['clash_flags'][idx] else "clash"
                fname     = (f"rank{rank:03d}_nnd{r['gen_nnd'][idx]:.2f}"
                             f"_rmse{r['rmse_vals'][idx]:.3f}"
                             f"_{valid_str}_{clash_str}.pdb")
                save_pdb(
                    r['samples'][idx],
                    str(pdb_dir / fname),
                    label=(f"tau={tau:.2f} rank={rank} "
                           f"nnd={r['gen_nnd'][idx]:.2f} rmse={r['rmse_vals'][idx]:.3f}"),
                )
            print(f"  Saved {args.top_k} PDB files → {pdb_dir}/")
            print(f"  Visualise: pymol {pdb_dir}/*.pdb")

    # ── Save JSON ──────────────────────────────────────────────────────────────
    out = []
    for r in results:
        out.append({
            'tau':        r['tau'],
            'nnd_ratio':  r['nnd_ratio'],
            'coverage':   r['coverage'],
            'precision':  r['precision'],
            'gen_nnd_mean':   float(r['gen_nnd'].mean()),
            'self_nnd_mean':  float(r['self_nnd'].mean()),
            'bond_validity_all':  float(r['valid_flags'].mean()),
            'no_clash_all':       float((~r['clash_flags']).mean()),
            'bond_validity_top50': float(r['valid_flags'][r['sorted_idx'][:50]].mean()),
            'no_clash_top50':      float((~r['clash_flags'][r['sorted_idx'][:50]]).mean()),
        })
    out_json = args.out_json or str(Path(args.ckpt).parent / 'novelty_by_tau.json')
    with open(out_json, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nMetrics → {out_json}")

    # ── Plot ───────────────────────────────────────────────────────────────────
    plot_results(results, reference, save_path=args.save)


if __name__ == '__main__':
    main()
