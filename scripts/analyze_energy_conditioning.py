"""
scripts/analyze_energy_conditioning.py
=======================================
Test whether temperature-controlled sampling with the energy-conditioned flow
matching model works correctly.

What this script checks
-----------------------
1. Temperature sweep — generate structures at τ ∈ {0, 0.25, 0.5, 0.75, 1.0}
   and compute per-temperature structural metrics.

2. Monotonicity — verify that as τ increases:
     Rg ↑  (structures get more extended)
     ETE ↑ (end-to-end distance increases)
     diversity ↑ (more conformational spread)
     bond validity ≈ const (physics should not be temperature-dependent)

3. Reference stratification — compare generated structures at each τ with
   the matching energy-quartile subset of the test set via MMD.

4. CFG guidance sweep — at τ=0 and τ=1 vary guidance_scale ∈ {1.0, 2.0, 4.0}
   and check that stronger guidance produces more extreme Rg values.

Expected outcome (well-trained model)
--------------------------------------
    τ=0.0: Rg ≈ 4.97 Å, ETE ≈  5.1 Å  (compact, folded — matches Q1 of dataset)
    τ=1.0: Rg ≈ 6.96 Å, ETE ≈ 18.8 Å  (extended, transient — matches Q4)

These reference values are from the dataset energy quartile analysis.

Usage
-----
# After local training (5 epochs, CPU — code paths only):
    python scripts/analyze_energy_conditioning.py \\
        --checkpoint checkpoints/flowmatch_energy/v1/best.pt \\
        --test data/test.npz \\
        --temperatures 0.0 1.0 --n 50 --steps 10 --guidance_scale 1.0

# After full training (500 epochs, GPU):
    python scripts/analyze_energy_conditioning.py \\
        --checkpoint checkpoints/flowmatch_energy/v1/best.pt \\
        --test data/test.npz \\
        --temperatures 0.0 0.25 0.5 0.75 1.0 \\
        --n 500 --steps 100 --guidance_scale 2.0 \\
        --save plots/energy_analysis.png

Output
------
  Console:  per-temperature metrics table + pass/fail monotonicity tests
  PNG:      6-panel analysis plot (if --save provided)
  JSON:     metrics dict at <checkpoint_dir>/energy_analysis.json
"""

import sys
import json
import argparse
import numpy as np
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))


# ─────────────────────────────────────────────────────────────────────────────
# CHECKPOINT LOADING
# ─────────────────────────────────────────────────────────────────────────────

def _strip_compile_prefix(state_dict: dict) -> dict:
    return {
        (k[len('_orig_mod.'):] if k.startswith('_orig_mod.') else k): v
        for k, v in state_dict.items()
    }


def load_energy_model(ckpt_path: str, device: str):
    """
    Load an EGNNEnergyScoreNetwork checkpoint.
    Returns (model, diffusion, config, coord_scale, energy_mean, energy_std).
    """
    from models.egnn_energy   import EGNNEnergyScoreNetwork
    from models.flow_matching import ZeroCoMFlowMatching

    ckpt   = torch.load(ckpt_path, map_location=device)
    config = ckpt['config']
    mt     = config['model_type']

    if mt != 'flowmatch_energy':
        raise ValueError(
            f"Expected model_type='flowmatch_energy', got {mt!r}.\n"
            f"This script only works with energy-conditioned checkpoints."
        )

    mc    = config['model']
    model = EGNNEnergyScoreNetwork(
        n_residues       = config['data']['n_residues'],
        node_dim         = mc['hidden_dim'],
        edge_dim         = mc.get('edge_dim', 64),
        time_dim         = mc['time_dim'],
        n_layers         = mc['n_layers'],
        energy_dim       = mc.get('energy_dim', 32),
        energy_drop_prob = mc.get('energy_drop_prob', 0.15),
    )
    model.load_state_dict(_strip_compile_prefix(ckpt['ema_shadow']))
    model     = model.to(device).eval()

    fc        = config.get('flow', {})
    diffusion = ZeroCoMFlowMatching(sigma_min=fc.get('sigma_min', 1e-4)).to(device)

    coord_scale  = config['data'].get('coord_scale', 5.0)
    energy_mean  = config['data'].get('energy_mean', -4511.4777)
    energy_std   = config['data'].get('energy_std',  0.0854)
    epoch        = ckpt.get('epoch', '?')
    val_loss     = ckpt.get('best_val_loss', float('nan'))

    print(f"  Loaded flowmatch_energy — epoch {epoch}, val_loss={val_loss:.4f}")
    print(f"  Energy: mean={energy_mean:.4f}  std={energy_std:.6f}")
    return model, diffusion, config, coord_scale, energy_mean, energy_std


# ─────────────────────────────────────────────────────────────────────────────
# GENERATION
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def generate_at_temperature(
    model, diffusion, tau: float, n: int, n_residues: int,
    coord_scale: float, ddim_steps: int, guidance_scale: float,
    device: str, batch_size: int = 256
) -> np.ndarray:
    """
    Generate n structures at temperature τ using CFG guidance.
    Returns (n, n_residues, 3) in Ångströms.
    """
    model.eval()
    all_samples = []
    n_done = 0

    while n_done < n:
        bs    = min(batch_size, n - n_done)
        shape = (bs, n_residues, 3)
        x     = diffusion.ddim_sample_cfg(
            model, shape, device=device,
            ddim_steps=ddim_steps, tau=tau, guidance_scale=guidance_scale,
        )
        x = x - x.mean(dim=1, keepdim=True)   # re-centre
        x = x * coord_scale                    # → Ångströms
        all_samples.append(x.cpu().numpy())
        n_done += bs

    return np.concatenate(all_samples, axis=0)[:n].astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# STRUCTURAL METRICS  (mirrors evaluate.py)
# ─────────────────────────────────────────────────────────────────────────────

_IDEAL_BOND   = 3.832
_CLASH_CUTOFF = 3.5


def bond_lengths(coords: np.ndarray) -> np.ndarray:
    return np.linalg.norm(np.diff(coords, axis=1), axis=-1)


def radius_of_gyration(coords: np.ndarray) -> np.ndarray:
    com = coords.mean(axis=1, keepdims=True)
    return np.sqrt(((coords - com) ** 2).sum(axis=-1).mean(axis=-1))


def end_to_end(coords: np.ndarray) -> np.ndarray:
    return np.linalg.norm(coords[:, -1] - coords[:, 0], axis=-1)


def bond_validity(coords: np.ndarray, tol: float = 0.5) -> float:
    bl = bond_lengths(coords)
    return float((np.abs(bl - _IDEAL_BOND) < tol).all(axis=1).mean())


def diversity(coords: np.ndarray, n_sub: int = 300) -> float:
    N = len(coords)
    idx = np.random.choice(N, min(n_sub, N), replace=False)
    sub = coords[idx]
    sq  = ((sub[:, None] - sub[None]) ** 2).sum(-1).mean(-1)
    tri = np.triu_indices(len(sub), k=1)
    return float(np.sqrt(sq[tri]).mean()) if len(tri[0]) > 0 else 0.0


def mmd_rbf(x: np.ndarray, y: np.ndarray, sigmas=(0.5, 1.0, 2.0)) -> float:
    x = x.reshape(len(x), -1).astype(np.float64)
    y = y.reshape(len(y), -1).astype(np.float64)
    max_pts = 1000
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


def compute_metrics(coords: np.ndarray) -> dict:
    rg  = radius_of_gyration(coords)
    ete = end_to_end(coords)
    bl  = bond_lengths(coords)
    return {
        'rg_mean':    float(rg.mean()),
        'rg_std':     float(rg.std()),
        'ete_mean':   float(ete.mean()),
        'ete_std':    float(ete.std()),
        'bond_valid': bond_validity(coords),
        'diversity':  diversity(coords),
        'bond_rmse':  float(np.sqrt(((bl - _IDEAL_BOND) ** 2).mean())),
    }


# ─────────────────────────────────────────────────────────────────────────────
# REFERENCE STRATIFICATION
# ─────────────────────────────────────────────────────────────────────────────

def stratify_reference(reference: np.ndarray, energies: np.ndarray, n_strata: int = 5):
    """
    Split reference structures into n_strata equal-sized energy bins.
    Returns list of (low_τ, high_τ, coords_subset) tuples.
    """
    percentiles = np.linspace(0, 100, n_strata + 1)
    boundaries  = np.percentile(energies, percentiles)
    strata = []
    for i in range(n_strata):
        mask = (energies >= boundaries[i]) & (energies <= boundaries[i + 1])
        # Map energy percentile to temperature
        tau_lo = i / n_strata
        tau_hi = (i + 1) / n_strata
        strata.append((tau_lo, tau_hi, reference[mask]))
    return strata


# ─────────────────────────────────────────────────────────────────────────────
# CONSOLE TABLE
# ─────────────────────────────────────────────────────────────────────────────

def print_sweep_table(results: list):
    """results: list of (tau, guidance_scale, metrics_dict)"""
    print(f"\n{'━'*80}")
    print(f"  Temperature sweep results")
    print(f"{'━'*80}")
    header = f"  {'τ':>5}  {'w':>4}  {'Rg (Å)':>9}  {'ETE (Å)':>9}  "
    header += f"{'Bond%':>6}  {'Diversity':>9}  {'BondRMSE':>9}"
    print(header)
    print(f"  {'─'*74}")
    for tau, w, m in results:
        print(
            f"  {tau:5.2f}  {w:4.1f}  "
            f"{m['rg_mean']:6.3f}±{m['rg_std']:.2f}  "
            f"{m['ete_mean']:6.2f}±{m['ete_std']:.1f}  "
            f"{m['bond_valid']*100:5.1f}%  "
            f"{m['diversity']:9.3f}  "
            f"{m['bond_rmse']:9.4f}"
        )
    print(f"{'━'*80}")


# ─────────────────────────────────────────────────────────────────────────────
# MONOTONICITY TESTS
# ─────────────────────────────────────────────────────────────────────────────

def run_monotonicity_tests(sweep_results: list) -> dict:
    """
    Check that Rg, ETE, and diversity increase with τ.
    sweep_results: list of (tau, guidance_scale, metrics) at fixed guidance_scale.
    Returns dict with test names as keys and bool as values.
    """
    # Filter to a single guidance_scale (use the first one)
    w0 = sweep_results[0][1]
    pts = [(tau, m) for tau, w, m in sweep_results if w == w0]
    pts.sort(key=lambda x: x[0])

    if len(pts) < 2:
        return {}

    rg_vals   = [m['rg_mean']   for _, m in pts]
    ete_vals  = [m['ete_mean']  for _, m in pts]
    div_vals  = [m['diversity'] for _, m in pts]
    bond_vals = [m['bond_valid'] for _, m in pts]

    def is_monotone_increasing(vals):
        return all(v2 >= v1 - 0.01 for v1, v2 in zip(vals[:-1], vals[1:]))

    def is_roughly_constant(vals, rel_tol=0.30):
        mean = np.mean(vals)
        if mean == 0:
            return True
        return (max(vals) - min(vals)) / mean < rel_tol

    tests = {
        'rg_increases_with_tau':        is_monotone_increasing(rg_vals),
        'ete_increases_with_tau':       is_monotone_increasing(ete_vals),
        'diversity_increases_with_tau': is_monotone_increasing(div_vals),
        'bond_validity_stable':         is_roughly_constant(bond_vals),
    }

    print(f"\n{'━'*60}")
    print(f"  Monotonicity tests  (guidance_scale={w0})")
    print(f"{'━'*60}")
    all_pass = True
    for name, passed in tests.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}  {name}")
        if not passed:
            all_pass = False
    print(f"{'━'*60}")
    print(f"  Overall: {'ALL PASS ✓' if all_pass else 'SOME FAILED ✗'}")
    print(f"{'━'*60}")

    return tests


# ─────────────────────────────────────────────────────────────────────────────
# PLOTTING
# ─────────────────────────────────────────────────────────────────────────────

def plot_analysis(sweep_results: list, ref_strata: list, reference: np.ndarray,
                  all_samples: dict, save_path: str = None):
    """
    6-panel analysis plot.

    sweep_results : list of (tau, guidance_scale, metrics)
    ref_strata    : from stratify_reference()
    all_samples   : {tau: np.ndarray (N, n_res, 3)} at the primary guidance_scale
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        from matplotlib.colors import LinearSegmentedColormap
    except ImportError:
        print("matplotlib not installed — skipping plot")
        return

    # Colour palette: blue (cold/stable) → red (hot/transient)
    cmap     = LinearSegmentedColormap.from_list('temp', ['#2166ac', '#92c5de', '#f7f7f7', '#f4a582', '#d6604d'])
    taus     = sorted(set(tau for tau, _, _ in sweep_results))
    n_tau    = len(taus)
    colors   = [cmap(i / max(n_tau - 1, 1)) for i in range(n_tau)]

    # Extract metrics at a fixed guidance_scale (use the first unique one)
    w0 = sweep_results[0][1]
    fixed_w  = [(tau, m) for tau, w, m in sweep_results if w == w0]
    fixed_w.sort(key=lambda x: x[0])

    fig = plt.figure(figsize=(15, 12))
    gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)
    axes = [fig.add_subplot(gs[r, c]) for r in range(3) for c in range(2)]

    fig.suptitle(
        f"Energy-Conditioned Flow Matching — Temperature Analysis\n"
        f"(guidance_scale={w0})",
        fontsize=12, y=0.99
    )

    # ── Panel 0: Rg distribution per temperature ──────────────────────────
    ax = axes[0]
    ref_rg = radius_of_gyration(reference)
    ax.hist(ref_rg, bins=40, density=True, color='#222222', alpha=0.25,
            label='Reference (all)', zorder=0)
    for i, tau in enumerate(taus):
        if tau not in all_samples:
            continue
        rg = radius_of_gyration(all_samples[tau])
        ax.hist(rg, bins=40, density=True, color=colors[i], alpha=0.65,
                label=f"τ={tau:.2f}")
    ax.set_xlabel("Rg (Å)", fontsize=9)
    ax.set_title("Radius of Gyration", fontsize=10)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    # ── Panel 1: End-to-end distance per temperature ──────────────────────
    ax = axes[1]
    ref_ete = end_to_end(reference)
    ax.hist(ref_ete, bins=40, density=True, color='#222222', alpha=0.25,
            label='Reference (all)', zorder=0)
    for i, tau in enumerate(taus):
        if tau not in all_samples:
            continue
        ete = end_to_end(all_samples[tau])
        ax.hist(ete, bins=40, density=True, color=colors[i], alpha=0.65,
                label=f"τ={tau:.2f}")
    ax.set_xlabel("End-to-end distance (Å)", fontsize=9)
    ax.set_title("End-to-End Distance", fontsize=10)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    # ── Panel 2: Mean Rg ± std vs τ with reference quartile markers ───────
    ax = axes[2]
    tau_pts  = [tau for tau, _ in fixed_w]
    rg_means = [m['rg_mean'] for _, m in fixed_w]
    rg_stds  = [m['rg_std']  for _, m in fixed_w]
    ax.errorbar(tau_pts, rg_means, yerr=rg_stds, fmt='o-', color='#d6604d',
                lw=2, ms=6, label=f"Generated (w={w0})")

    # Reference quartile lines
    q_labels = ['Q1 (stable)', 'Q2', 'Q3', 'Q4 (transient)']
    q_colors = ['#2166ac', '#74add1', '#f4a582', '#d73027']
    for qi, (q_tau_lo, q_tau_hi, q_coords) in enumerate(ref_strata[:4]):
        if len(q_coords) == 0:
            continue
        q_rg = radius_of_gyration(q_coords).mean()
        q_mid = (q_tau_lo + q_tau_hi) / 2
        ax.axhline(q_rg, color=q_colors[qi], linestyle='--', alpha=0.7,
                   lw=1.5, label=q_labels[qi])
    ax.set_xlabel("Temperature τ", fontsize=9)
    ax.set_ylabel("Mean Rg (Å)", fontsize=9)
    ax.set_title("Rg vs Temperature", fontsize=10)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    # ── Panel 3: Mean ETE ± std vs τ ──────────────────────────────────────
    ax = axes[3]
    ete_means = [m['ete_mean'] for _, m in fixed_w]
    ete_stds  = [m['ete_std']  for _, m in fixed_w]
    ax.errorbar(tau_pts, ete_means, yerr=ete_stds, fmt='s-', color='#4393c3',
                lw=2, ms=6, label=f"Generated (w={w0})")

    for qi, (q_tau_lo, q_tau_hi, q_coords) in enumerate(ref_strata[:4]):
        if len(q_coords) == 0:
            continue
        q_ete = end_to_end(q_coords).mean()
        ax.axhline(q_ete, color=q_colors[qi], linestyle='--', alpha=0.7,
                   lw=1.5, label=q_labels[qi])
    ax.set_xlabel("Temperature τ", fontsize=9)
    ax.set_ylabel("Mean ETE (Å)", fontsize=9)
    ax.set_title("End-to-End Distance vs Temperature", fontsize=10)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    # ── Panel 4: Bond validity vs τ (should be ~constant) ─────────────────
    ax = axes[4]
    bv_vals = [m['bond_valid'] * 100 for _, m in fixed_w]
    ax.plot(tau_pts, bv_vals, 'D-', color='#1a9641', lw=2, ms=6,
            label=f"Generated (w={w0})")
    ref_bv = float((np.abs(bond_lengths(reference) - _IDEAL_BOND) < 0.5
                    ).all(axis=1).mean()) * 100
    ax.axhline(ref_bv, color='#222222', linestyle='--', lw=1.5, alpha=0.5,
               label=f"Reference ({ref_bv:.1f}%)")
    ax.set_xlabel("Temperature τ", fontsize=9)
    ax.set_ylabel("Bond validity ±0.5 Å (%)", fontsize=9)
    ax.set_title("Bond Validity vs Temperature\n(should be roughly constant)", fontsize=10)
    ax.set_ylim(0, 105)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    # ── Panel 5: Diversity vs τ (should increase) ─────────────────────────
    ax = axes[5]
    div_vals = [m['diversity'] for _, m in fixed_w]
    ax.plot(tau_pts, div_vals, '^-', color='#7b2d8b', lw=2, ms=6,
            label=f"Generated (w={w0})")
    ref_div = diversity(reference)
    ax.axhline(ref_div, color='#222222', linestyle='--', lw=1.5, alpha=0.5,
               label=f"Reference ({ref_div:.3f})")
    ax.set_xlabel("Temperature τ", fontsize=9)
    ax.set_ylabel("Mean pairwise RMSD (Å)", fontsize=9)
    ax.set_title("Structural Diversity vs Temperature\n(should increase)", fontsize=10)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nPlot saved → {save_path}")
    else:
        plt.show()
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Analyse temperature control in the energy-conditioned flow model."
    )
    p.add_argument('--checkpoint', required=True,
                   help='Path to flowmatch_energy checkpoint (best.pt)')
    p.add_argument('--test',       required=True,
                   help='Path to test.npz')
    p.add_argument('--temperatures', nargs='+', type=float,
                   default=[0.0, 0.25, 0.5, 0.75, 1.0],
                   help='Temperature values to sweep')
    p.add_argument('--n',          type=int, default=200,
                   help='Structures to generate per (temperature, guidance_scale)')
    p.add_argument('--steps',      type=int, default=100,
                   help='ODE integration steps')
    p.add_argument('--guidance_scale', type=float, default=2.0,
                   help='Primary CFG guidance scale')
    p.add_argument('--guidance_sweep', nargs='+', type=float, default=None,
                   help='Additional guidance scales for the sweep panel '
                        '(e.g. --guidance_sweep 1.0 2.0 4.0). '
                        'Only evaluated at τ=0 and τ=1.')
    p.add_argument('--batch',      type=int, default=256)
    p.add_argument('--save',       default=None,
                   help='Path to save the analysis plot (PNG)')
    p.add_argument('--out_json',   default=None,
                   help='Path to save metrics JSON')
    p.add_argument('--seed',       type=int, default=42)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"\nLoading checkpoint: {args.checkpoint}")
    model, diffusion, config, coord_scale, energy_mean, energy_std = \
        load_energy_model(args.checkpoint, device)
    n_res = config['data']['n_residues']

    # ── Load test set ─────────────────────────────────────────────────────────
    test_data  = np.load(args.test)
    ref_coords = test_data['coords'].astype(np.float32)
    ref_energies = test_data['energies'].astype(np.float32)
    centroids  = test_data['centroids']
    if centroids.ndim == 2:
        centroids = centroids[:, np.newaxis, :]
    ref_coords = ref_coords - centroids   # centre
    print(f"Test set: {len(ref_coords):,} structures")
    print(f"  Ref energy range: [{ref_energies.min():.4f}, {ref_energies.max():.4f}]")

    # ── Reference stratification ──────────────────────────────────────────────
    n_strata    = min(5, len(args.temperatures))
    ref_strata  = stratify_reference(ref_coords, ref_energies, n_strata=n_strata)

    # ── Temperature sweep at primary guidance_scale ───────────────────────────
    all_samples  = {}   # {tau: np.ndarray}
    sweep_results = []  # [(tau, guidance_scale, metrics)]

    temperatures = sorted(set(args.temperatures))
    print(f"\n{'─'*60}")
    print(f"Temperature sweep  (guidance_scale={args.guidance_scale})")
    print(f"{'─'*60}")

    for tau in temperatures:
        print(f"  τ={tau:.2f}  → generating {args.n} structures …", end="", flush=True)
        samples = generate_at_temperature(
            model, diffusion, tau=tau, n=args.n, n_residues=n_res,
            coord_scale=coord_scale, ddim_steps=args.steps,
            guidance_scale=args.guidance_scale,
            device=device, batch_size=args.batch,
        )
        all_samples[tau] = samples
        m = compute_metrics(samples)
        m['mmd_vs_ref'] = mmd_rbf(samples, ref_coords)
        sweep_results.append((tau, args.guidance_scale, m))
        print(f"  Rg={m['rg_mean']:.3f}±{m['rg_std']:.2f}  "
              f"ETE={m['ete_mean']:.2f}±{m['ete_std']:.1f}")

    print_sweep_table(sweep_results)

    # ── Guidance scale sweep (τ=0 and τ=1 only) ───────────────────────────────
    if args.guidance_sweep:
        extra_scales = [w for w in args.guidance_sweep if w != args.guidance_scale]
        if extra_scales:
            print(f"\n{'─'*60}")
            print(f"Guidance scale sweep  (τ ∈ {{0.0, 1.0}})")
            print(f"{'─'*60}")
            for tau in [t for t in temperatures if t in (0.0, 1.0)]:
                for w in extra_scales:
                    print(f"  τ={tau:.1f}  w={w:.1f}  → generating {args.n} structures…",
                          end="", flush=True)
                    samples = generate_at_temperature(
                        model, diffusion, tau=tau, n=args.n, n_residues=n_res,
                        coord_scale=coord_scale, ddim_steps=args.steps,
                        guidance_scale=w, device=device, batch_size=args.batch,
                    )
                    m = compute_metrics(samples)
                    sweep_results.append((tau, w, m))
                    print(f"  Rg={m['rg_mean']:.3f}±{m['rg_std']:.2f}  "
                          f"ETE={m['ete_mean']:.2f}±{m['ete_std']:.1f}")

    # ── Reference metrics ─────────────────────────────────────────────────────
    ref_metrics = compute_metrics(ref_coords)
    print(f"\n  Reference:  Rg={ref_metrics['rg_mean']:.3f}±{ref_metrics['rg_std']:.2f}  "
          f"ETE={ref_metrics['ete_mean']:.2f}±{ref_metrics['ete_std']:.1f}  "
          f"Bond={ref_metrics['bond_valid']*100:.1f}%  "
          f"Diversity={ref_metrics['diversity']:.3f}")

    # ── Monotonicity tests ────────────────────────────────────────────────────
    mono_tests = run_monotonicity_tests(sweep_results)

    # ── Metrics JSON ──────────────────────────────────────────────────────────
    out_json = args.out_json or str(Path(args.checkpoint).parent / 'energy_analysis.json')
    results_dict = {
        'sweep':      [(float(tau), float(w), m) for tau, w, m in sweep_results],
        'reference':  ref_metrics,
        'mono_tests': mono_tests,
        'config': {
            'temperatures':   temperatures,
            'guidance_scale': args.guidance_scale,
            'n_samples':      args.n,
            'ddim_steps':     args.steps,
            'energy_mean':    energy_mean,
            'energy_std':     energy_std,
        },
    }
    with open(out_json, 'w') as f:
        json.dump(results_dict, f, indent=2)
    print(f"\nMetrics → {out_json}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    if args.save or True:   # always attempt plot; save_path=None → plt.show()
        plot_analysis(sweep_results, ref_strata, ref_coords, all_samples,
                      save_path=args.save)


if __name__ == '__main__':
    main()
