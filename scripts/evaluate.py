"""
scripts/evaluate.py
====================
Comprehensive evaluation: generate samples from one or more checkpoints
and compare them against the test set with structural and physics metrics.

Handles all model types automatically by reading model_type from the saved config:
  'egnn'        — EGNN + DDPM
  'transformer' — Transformer + DDPM
  'mlp'         — MLP + DDPM
  'flowmatch'   — EGNN + OT-CFM (with or without physics constraints)

Usage
-----
# Single model
python scripts/evaluate.py \\
    --ckpt checkpoints/flowmatch_physics/v3/best.pt \\
    --test data/test.npz --n 1000 --save plots/physics_eval.png

# Compare N models side-by-side (physics-constrained vs baseline)
python scripts/evaluate.py \\
    --ckpt     checkpoints/flowmatch_physics/v3/best.pt \\
    --ckpt_ref checkpoints/flowmatch/v2/best.pt \\
               checkpoints/egnn/v1/best.pt \\
    --labels   FlowMatch+Physics FlowMatch EGNN-DDPM \\
    --test     data/test.npz --n 1000 --save plots/comparison.png

# Save PDB files (20 structures from the primary model)
python scripts/evaluate.py \\
    --ckpt checkpoints/flowmatch_physics/v3/best.pt \\
    --test data/test.npz --n 500 --save_pdb outputs/physics_samples

Output
------
  Console:      two-section table — structural metrics + physics quality metrics
  plots/*.png:  6-panel plot: bond dist, Rg, end-to-end, per-residue flexibility,
                              bond validity thresholds, per-bond RMSE from ideal
  metrics.json: all numbers for downstream analysis
  PDB files:    Cα-only PDB files for PyMOL visualisation
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
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _strip_compile_prefix(state_dict: dict) -> dict:
    """
    Remove the '_orig_mod.' prefix that torch.compile adds to parameter names.
    Training scripts create EMA after torch.compile, so the EMA shadow may
    contain '_orig_mod.*' keys that don't match an uncompiled model at inference.
    """
    return {
        (k[len('_orig_mod.'):] if k.startswith('_orig_mod.') else k): v
        for k, v in state_dict.items()
    }


# ─────────────────────────────────────────────────────────────────────────────
# MODEL LOADING  (auto-detects model_type from checkpoint)
# ─────────────────────────────────────────────────────────────────────────────

def load_model_from_ckpt(ckpt_path: str, device: str):
    """
    Load model + diffusion from any checkpoint in this repo.
    Reads model_type from the saved config to pick the right class.

    Supported model_type values:
      'egnn'        — EGNNScoreNetwork + ZeroCoMGaussianDiffusion (DDPM)
      'mlp'         — MLPScoreNetwork  + GaussianDiffusion (DDPM)
      'transformer' — TransformerScoreNetwork + GaussianDiffusion (DDPM)
      'flowmatch'   — EGNNScoreNetwork + ZeroCoMFlowMatching (OT-CFM)

    Returns (model, diffusion, config, coord_scale)
    """
    ckpt   = torch.load(ckpt_path, map_location=device)
    config = ckpt['config']
    mt     = config['model_type']

    # ── Build model + diffusion (one branch per model_type) ───────────────
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
        dc        = config['diffusion']
        diffusion = ZeroCoMGaussianDiffusion(T=dc['T'], schedule=dc['schedule'])

    elif mt in ('mlp', 'transformer'):
        from scripts.train import build_model
        model     = build_model(config)
        dc        = config['diffusion']
        diffusion = GaussianDiffusion(T=dc['T'], schedule=dc['schedule'])

    elif mt == 'flowmatch':
        from models.egnn          import EGNNScoreNetwork
        from models.flow_matching import ZeroCoMFlowMatching
        mc    = config['model']
        model = EGNNScoreNetwork(
            n_residues = config['data']['n_residues'],
            node_dim   = mc['hidden_dim'],
            edge_dim   = mc.get('edge_dim', 64),
            time_dim   = mc['time_dim'],
            n_layers   = mc['n_layers'],
        )
        fc        = config.get('flow', {})
        diffusion = ZeroCoMFlowMatching(sigma_min=fc.get('sigma_min', 1e-4))

    elif mt == 'flowmatch_energy':
        from models.egnn_energy   import EGNNEnergyScoreNetwork
        from models.flow_matching import ZeroCoMFlowMatching
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
        fc        = config.get('flow', {})
        diffusion = ZeroCoMFlowMatching(sigma_min=fc.get('sigma_min', 1e-4))
        # Note: generate() calls diffusion.ddim_sample(model, ...) which calls
        # model(x, t) — energy_z defaults to None (unconditional/average).
        # Use analyze_energy_conditioning.py for temperature-conditioned generation.

    else:
        raise ValueError(f"Unknown model_type: {mt!r}")

    # ── Shared: load EMA weights, move to device ───────────────────────────
    model.load_state_dict(_strip_compile_prefix(ckpt['ema_shadow']))
    model     = model.to(device).eval()
    diffusion = diffusion.to(device)

    coord_scale = config['data'].get('coord_scale', 16.32)
    epoch       = ckpt.get('epoch', '?')
    val_loss    = ckpt.get('best_val_loss', float('nan'))
    physics_on  = config.get('training', {}).get('physics_weight', 0.0) > 0

    print(f"  Loaded {mt} checkpoint — epoch {epoch}, val_loss={val_loss:.4f}"
          + ("  [physics-constrained]" if physics_on else ""))
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
# STRUCTURAL METRICS
# ─────────────────────────────────────────────────────────────────────────────

# Reference values from training set (79,632 Chignolin structures)
_IDEAL_BOND   = 3.832   # Å — mean Cα–Cα bond length
_IDEAL_COS    = 0.320   # cos(71.4°) — mean consecutive bond-vector angle
_CLASH_CUTOFF = 3.5     # Å — non-bonded pairs below this = clash


def bond_lengths(coords: np.ndarray) -> np.ndarray:
    """Consecutive Cα–Cα distances. (N, n_res, 3) → (N, n_res-1)."""
    return np.linalg.norm(np.diff(coords, axis=1), axis=-1)


def radius_of_gyration(coords: np.ndarray) -> np.ndarray:
    """Rg per structure. (N, n_res, 3) → (N,)."""
    com = coords.mean(axis=1, keepdims=True)
    return np.sqrt(((coords - com) ** 2).sum(axis=-1).mean(axis=-1))


def validity(coords: np.ndarray, tol: float = 0.5, ideal: float = _IDEAL_BOND) -> float:
    """Fraction of structures where ALL bonds are within ±tol Å of ideal."""
    bl    = bond_lengths(coords)
    valid = (np.abs(bl - ideal) < tol).all(axis=1)
    return float(valid.mean())


def mmd_rbf(x: np.ndarray, y: np.ndarray,
            sigmas=(0.5, 1.0, 2.0)) -> float:
    """
    Maximum Mean Discrepancy with RBF kernel, averaged over several bandwidths.
    x, y : (N, n_res, 3) → flattened to (N, n_res*3).  Lower = more similar.
    """
    x = x.reshape(len(x), -1).astype(np.float64)
    y = y.reshape(len(y), -1).astype(np.float64)

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


# ─────────────────────────────────────────────────────────────────────────────
# PHYSICS QUALITY METRICS
# ─────────────────────────────────────────────────────────────────────────────

def compute_physics_metrics(coords: np.ndarray) -> dict:
    """
    Physics-based structural quality metrics for Cα coordinates in Ångströms.

    These complement the distribution-level metrics (MMD, Rg) with per-structure
    constraint satisfaction — the key signal for evaluating physics-constrained models.

    coords : (N, n_res, 3)  in Ångströms
    returns : dict with keys:
        phys_valid_02      fraction of structures with ALL bonds within ±0.2 Å
        phys_valid_03      fraction of structures with ALL bonds within ±0.3 Å
        phys_valid_05      fraction of structures with ALL bonds within ±0.5 Å
        phys_bond_rmse     RMS bond-length deviation from ideal (3.832 Å)
        phys_per_bond_rmse per-bond RMSE (list, length n_res-1) — reveals problem positions
        phys_clash_rate    fraction of structures with any non-bonded clash < 3.5 Å
        phys_angle_rmse    RMS deviation of cos(bond angle) from 0.320 (71.4°)
        phys_diversity     mean pairwise RMSD between generated structures (Å)
    """
    N, n_res, _ = coords.shape
    bl = bond_lengths(coords)                        # (N, n_res-1)

    # ── Bond validity at multiple strictness thresholds ───────────────────
    phys_valid_02 = float((np.abs(bl - _IDEAL_BOND) < 0.2).all(axis=1).mean())
    phys_valid_03 = float((np.abs(bl - _IDEAL_BOND) < 0.3).all(axis=1).mean())
    phys_valid_05 = float((np.abs(bl - _IDEAL_BOND) < 0.5).all(axis=1).mean())

    # ── Bond RMSE (global and per-bond) ──────────────────────────────────
    phys_bond_rmse     = float(np.sqrt(((bl - _IDEAL_BOND) ** 2).mean()))
    phys_per_bond_rmse = np.sqrt(((bl - _IDEAL_BOND) ** 2).mean(axis=0)).tolist()

    # ── Clash rate — fraction of structures with any non-bonded pair < 3.5 Å
    diff = coords[:, :, None, :] - coords[:, None, :, :]   # (N, n_res, n_res, 3)
    dist = np.linalg.norm(diff, axis=-1)                    # (N, n_res, n_res)
    idx  = np.arange(n_res)
    sep  = np.abs(idx[:, None] - idx[None, :])             # (n_res, n_res) separation
    mask = (sep >= 2)                                        # non-bonded pairs only
    phys_clash_rate = float((dist[:, mask] < _CLASH_CUTOFF).any(axis=1).mean())

    # ── Virtual bond angle: cos(angle between consecutive bond vectors) ───
    b1 = coords[:, 1:-1] - coords[:, :-2]                  # (N, n_res-2, 3)
    b2 = coords[:, 2:]   - coords[:, 1:-1]                 # (N, n_res-2, 3)
    norm1 = np.linalg.norm(b1, axis=-1, keepdims=True).clip(1e-8)
    norm2 = np.linalg.norm(b2, axis=-1, keepdims=True).clip(1e-8)
    cos_t = (b1 / norm1 * b2 / norm2).sum(axis=-1)         # (N, n_res-2)
    phys_angle_rmse = float(np.sqrt(((cos_t - _IDEAL_COS) ** 2).mean()))

    # ── Structural diversity: mean pairwise RMSD (subsampled) ────────────
    n_sub  = min(300, N)
    idx_s  = np.random.choice(N, n_sub, replace=False) if N > n_sub else np.arange(N)
    sub    = coords[idx_s]                                  # (n_sub, n_res, 3)
    # Pairwise squared displacement, averaged over residues
    sq     = ((sub[:, None] - sub[None]) ** 2).sum(-1).mean(-1)  # (n_sub, n_sub)
    triu   = np.triu_indices(n_sub, k=1)
    phys_diversity = float(np.sqrt(sq[triu]).mean()) if len(triu[0]) > 0 else 0.0

    return {
        'phys_valid_02':      phys_valid_02,
        'phys_valid_03':      phys_valid_03,
        'phys_valid_05':      phys_valid_05,
        'phys_bond_rmse':     phys_bond_rmse,
        'phys_per_bond_rmse': phys_per_bond_rmse,
        'phys_clash_rate':    phys_clash_rate,
        'phys_angle_rmse':    phys_angle_rmse,
        'phys_diversity':     phys_diversity,
    }


def compute_all_metrics(samples: np.ndarray, reference: np.ndarray) -> dict:
    """
    All structural + physics metrics for a set of generated structures.
    Returns a flat dict suitable for JSON serialisation.
    """
    bl_s = bond_lengths(samples);   bl_r = bond_lengths(reference)
    rg_s = radius_of_gyration(samples); rg_r = radius_of_gyration(reference)
    ete_s = np.linalg.norm(samples[:, -1] - samples[:, 0], axis=-1)
    ete_r = np.linalg.norm(reference[:, -1] - reference[:, 0], axis=-1)

    metrics = {
        # ── Structural distribution metrics ──
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

    # ── Physics quality metrics (generated structures only) ──
    metrics.update(compute_physics_metrics(samples))

    # Reference physics quality (for comparison — these are the floor values)
    ref_phys = compute_physics_metrics(reference[:min(2000, len(reference))])
    metrics['ref_phys_bond_rmse']  = ref_phys['phys_bond_rmse']
    metrics['ref_phys_valid_02']   = ref_phys['phys_valid_02']
    metrics['ref_phys_valid_05']   = ref_phys['phys_valid_05']
    metrics['ref_phys_clash_rate'] = ref_phys['phys_clash_rate']
    metrics['ref_phys_diversity']  = ref_phys['phys_diversity']

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# CONSOLE TABLE
# ─────────────────────────────────────────────────────────────────────────────

def print_table(metrics: dict, label: str = "Model"):
    w = 32
    print(f"\n{'━'*66}")
    print(f"  {label}")
    print(f"{'━'*66}")

    # ── Section 1: Distribution-level metrics ────────────────────────────
    print(f"\n  {'Structural distribution':<{w}} {'Generated':>12} {'Reference':>12}")
    print(f"  {'─'*58}")
    rows1 = [
        ("Bond validity ±0.5 Å (%)",  f"{metrics['validity_gen']*100:.1f}%",    f"{metrics['validity_ref']*100:.1f}%"),
        ("Bond length mean (Å)",       f"{metrics['bond_mean_gen']:.3f}",         f"{metrics['bond_mean_ref']:.3f}"),
        ("Bond length std  (Å)",       f"{metrics['bond_std_gen']:.3f}",          f"{metrics['bond_std_ref']:.3f}"),
        ("Rg mean (Å)",                f"{metrics['rg_mean_gen']:.3f}",           f"{metrics['rg_mean_ref']:.3f}"),
        ("Rg std  (Å)",                f"{metrics['rg_std_gen']:.3f}",            f"{metrics['rg_std_ref']:.3f}"),
        ("End-to-end mean (Å)",        f"{metrics['ete_mean_gen']:.3f}",          f"{metrics['ete_mean_ref']:.3f}"),
        ("MMD-RBF (↓ better)",         f"{metrics['mmd']:.5f}",                  "—"),
    ]
    for name, gen, ref in rows1:
        print(f"  {name:<{w}} {gen:>12} {ref:>12}")

    # ── Section 2: Physics quality metrics ───────────────────────────────
    print(f"\n  {'Physics constraint quality':<{w}} {'Generated':>12} {'Reference':>12}")
    print(f"  {'─'*58}")
    rows2 = [
        ("Bond validity ±0.2 Å (%)",  f"{metrics['phys_valid_02']*100:.1f}%",  f"{metrics.get('ref_phys_valid_02', float('nan'))*100:.1f}%"),
        ("Bond validity ±0.3 Å (%)",  f"{metrics['phys_valid_03']*100:.1f}%",  "—"),
        ("Bond RMSE (Å)",             f"{metrics['phys_bond_rmse']:.4f}",       f"{metrics.get('ref_phys_bond_rmse', float('nan')):.4f}"),
        ("Clash rate (%)",            f"{metrics['phys_clash_rate']*100:.1f}%", f"{metrics.get('ref_phys_clash_rate', 0)*100:.1f}%"),
        ("Angle cos RMSE",            f"{metrics['phys_angle_rmse']:.4f}",      "—"),
        ("Diversity — RMSD (Å)",      f"{metrics['phys_diversity']:.3f}",       f"{metrics.get('ref_phys_diversity', float('nan')):.3f}"),
    ]
    for name, gen, ref in rows2:
        print(f"  {name:<{w}} {gen:>12} {ref:>12}")

    print(f"{'━'*66}")


# ─────────────────────────────────────────────────────────────────────────────
# PLOTTING
# ─────────────────────────────────────────────────────────────────────────────

def plot_comparison(samples_dict: dict, reference: np.ndarray,
                    metrics_all: dict = None, save_path: str = None):
    """
    6-panel comparison plot.

    Panels 1–4: distribution comparisons (bond lengths, Rg, end-to-end, per-residue flexibility)
    Panel 5:    bond validity at multiple strictness thresholds (bar chart)
    Panel 6:    per-bond RMSE from ideal 3.832 Å (shows which chain positions are worst)

    samples_dict : {'ModelA': (N, n_res, 3), ...}
    reference    : (N_ref, n_res, 3)
    metrics_all  : pre-computed metrics dict (used for physics panels); if None, recomputed
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
    except ImportError:
        print("matplotlib not installed — skipping plot")
        return

    n_res   = reference.shape[1]
    colors  = ['#C44E52', '#4C72B0', '#55A868', '#8172B3']
    labels  = list(samples_dict.keys())

    fig = plt.figure(figsize=(14, 12))
    gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)
    axes = [fig.add_subplot(gs[r, c]) for r in range(3) for c in range(2)]

    fig.suptitle("Generated vs Reference Structures", fontsize=13, y=0.98)

    # ── Helper for histogram panels ───────────────────────────────────────
    def hist(ax, fn, xlabel, title, bins=40):
        ref_vals = fn(reference)
        ax.hist(ref_vals.flatten(), bins=bins, density=True,
                color='#222222', alpha=0.3, label='Reference')
        for (lbl, samples), color in zip(samples_dict.items(), colors):
            vals = fn(samples)
            ax.hist(vals.flatten(), bins=bins, density=True,
                    color=color, alpha=0.6, label=lbl)
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

    # Panel 0: Bond length distribution
    hist(axes[0], bond_lengths,
         "Cα–Cα distance (Å)", "Bond Length Distribution")
    axes[0].axvline(3.332, color='k', linestyle='--', lw=1, alpha=0.4, label='±0.5 Å bounds')
    axes[0].axvline(4.332, color='k', linestyle='--', lw=1, alpha=0.4)

    # Panel 1: Radius of gyration
    hist(axes[1], radius_of_gyration, "Rg (Å)", "Radius of Gyration")

    # Panel 2: End-to-end distance
    hist(axes[2], lambda x: np.linalg.norm(x[:, -1] - x[:, 0], axis=-1),
         "End-to-end (Å)", "End-to-End Distance")

    # Panel 3: Per-residue positional variance
    ax = axes[3]
    ref_var = reference.var(axis=0).sum(axis=-1)
    ax.plot(range(n_res), ref_var, 'o-', color='#222222', alpha=0.5, label='Reference', lw=2)
    for (lbl, samples), color in zip(samples_dict.items(), colors):
        var = samples.var(axis=0).sum(axis=-1)
        ax.plot(range(n_res), var, 'o-', color=color, alpha=0.8, label=lbl)
    ax.set_xticks(range(n_res))
    ax.set_xticklabels([f"R{i+1}" for i in range(n_res)], rotation=45, fontsize=7)
    ax.set_ylabel("Positional variance (Å²)", fontsize=9)
    ax.set_title("Per-Residue Flexibility", fontsize=10)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    # Panel 4: Bond validity at multiple strictness thresholds (bar chart)
    ax = axes[4]
    thresholds = ['±0.2 Å', '±0.3 Å', '±0.5 Å']
    keys       = ['phys_valid_02', 'phys_valid_03', 'phys_valid_05']
    x_pos      = np.arange(len(thresholds))
    bar_width  = 0.8 / (len(labels) + 1)

    # Reference bar
    ref_vals = [compute_physics_metrics(reference[:500])['phys_valid_02'],
                None,
                validity(reference)]
    # Re-use pre-computed if available
    if metrics_all:
        first_key = next(iter(metrics_all))
        ref_vals  = [metrics_all[first_key].get('ref_phys_valid_02', ref_vals[0]),
                     None,
                     metrics_all[first_key].get('validity_ref', ref_vals[2])]
    ref_vals[1] = None  # ±0.3 not stored for reference; skip

    for k_idx, (thr, key) in enumerate(zip(thresholds, keys)):
        # Generated model bars
        for m_idx, (lbl, color) in enumerate(zip(labels, colors)):
            val = metrics_all[lbl][key] if metrics_all else compute_physics_metrics(
                samples_dict[lbl])['phys_valid_02']
            offset = (m_idx - len(labels) / 2.0 + 0.5) * bar_width
            ax.bar(k_idx + offset, val * 100, bar_width * 0.9,
                   color=color, alpha=0.8, label=lbl if k_idx == 0 else "")

    # Reference dashes
    for k_idx, rv in enumerate(ref_vals):
        if rv is not None:
            ax.hlines(rv * 100, k_idx - 0.4, k_idx + 0.4,
                      colors='#222222', linewidths=2, linestyles='--',
                      label='Reference' if k_idx == 0 else "")

    ax.set_xticks(x_pos)
    ax.set_xticklabels(thresholds, fontsize=9)
    ax.set_ylabel("Bond validity (%)", fontsize=9)
    ax.set_title("Bond Validity at Multiple Thresholds", fontsize=10)
    ax.set_ylim(0, 105)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, axis='y')

    # Panel 5: Per-bond RMSE from ideal 3.832 Å
    ax = axes[5]
    # Reference per-bond RMSE
    bl_ref = bond_lengths(reference)
    ref_per_bond = np.sqrt(((bl_ref - _IDEAL_BOND) ** 2).mean(axis=0))
    ax.plot(range(n_res - 1), ref_per_bond, 'o--', color='#222222',
            alpha=0.5, label='Reference', lw=1.5, markersize=4)

    for (lbl, samples), color in zip(samples_dict.items(), colors):
        bl = bond_lengths(samples)
        per_bond = np.sqrt(((bl - _IDEAL_BOND) ** 2).mean(axis=0))
        ax.plot(range(n_res - 1), per_bond, 'o-', color=color,
                alpha=0.8, label=lbl, lw=1.5, markersize=4)

    ax.set_xticks(range(n_res - 1))
    ax.set_xticklabels([f"{i+1}–{i+2}" for i in range(n_res - 1)], rotation=45, fontsize=7)
    ax.set_ylabel("Bond RMSE (Å)", fontsize=9)
    ax.set_title("Per-Bond RMSE from Ideal (3.832 Å)", fontsize=10)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved → {save_path}")
    else:
        plt.show()
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# PDB EXPORT
# ─────────────────────────────────────────────────────────────────────────────

def save_pdbs(samples: np.ndarray, out_dir: str, n_save: int = 20,
              label: str = "generated"):
    out      = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    sequence = "YYDPETGTWG"
    aa3      = {'Y':'TYR','D':'ASP','P':'PRO','E':'GLU','T':'THR',
                'G':'GLY','W':'TRP','A':'ALA','K':'LYS','R':'ARG'}

    for i, coords in enumerate(samples[:n_save]):
        lines = [f"REMARK  {label} sample {i+1}\n"]
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

def _make_label(config: dict, used: set) -> str:
    """Derive a unique display label from a checkpoint config."""
    mt   = config['model_type'].upper()
    phys = config.get('training', {}).get('physics_weight', 0.0) > 0
    base = f"{mt}+Physics" if phys else mt
    label, suffix = base, 2
    while label in used:
        label = f"{base}_{suffix}"; suffix += 1
    return label


def main():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Evaluate one or more checkpoints against the test set."
    )
    # ── Checkpoints ───────────────────────────────────────────────────────────
    p.add_argument('--ckpt',     required=True,
                   help='Primary checkpoint path')
    p.add_argument('--ckpt_ref', nargs='*', default=[],
                   help='Additional checkpoints for side-by-side comparison '
                        '(accepts multiple paths)')
    p.add_argument('--labels',   nargs='*', default=None,
                   help='Custom labels for each checkpoint (must match total '
                        'count of --ckpt + --ckpt_ref). '
                        'Defaults to model_type, with "+Physics" suffix if trained with physics.')
    # ── Evaluation options ────────────────────────────────────────────────────
    p.add_argument('--test',     required=True,
                   help='Path to test.npz')
    p.add_argument('--n',        type=int, default=500,
                   help='Number of structures to generate per model')
    p.add_argument('--steps',    type=int, default=100,
                   help='Sampling steps. For DDPM: DDIM steps. '
                        'For flow matching: ODE steps (Heun uses 2 NFE per step).')
    p.add_argument('--save_pdb', default=None,
                   help='Directory to save PDB files for the primary model')
    p.add_argument('--save',     default=None,
                   help='Path to save the 6-panel comparison plot')
    p.add_argument('--out_json', default=None,
                   help='Path to save metrics JSON (default: beside primary checkpoint)')
    p.add_argument('--batch',    type=int, default=256,
                   help='Generation batch size')
    p.add_argument('--seed',     type=int, default=0)
    args = p.parse_args()

    # ── Validate ──────────────────────────────────────────────────────────────
    all_ckpts = [args.ckpt] + list(args.ckpt_ref)
    if args.labels is not None and len(args.labels) != len(all_ckpts):
        p.error(f"--labels has {len(args.labels)} entries but "
                f"{len(all_ckpts)} checkpoints were given.")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # ── Load test set ─────────────────────────────────────────────────────────
    test_data = np.load(args.test)
    reference = test_data['coords'].astype(np.float32)
    centroids = test_data['centroids']
    if centroids.ndim == 2:
        centroids = centroids[:, None, :]
    reference = reference - centroids
    print(f"Test set: {len(reference):,} structures")

    # ── Generate + evaluate each checkpoint ───────────────────────────────────
    samples_dict  = {}
    metrics_all   = {}
    primary_samples = None
    used_labels   = set()

    for i, ckpt_path in enumerate(all_ckpts):
        print(f"\n{'─'*60}")
        print(f"Checkpoint {i+1}/{len(all_ckpts)}: {ckpt_path}")
        model, diffusion, config, scale = load_model_from_ckpt(ckpt_path, device)

        label = args.labels[i] if args.labels else _make_label(config, used_labels)
        used_labels.add(label)

        print(f"Generating {args.n} samples ({label}) …")
        samples = generate(model, diffusion, args.n,
                           config['data']['n_residues'], scale,
                           args.steps, device, args.batch)

        if primary_samples is None:
            primary_samples = samples

        samples_dict[label] = samples
        print(f"Computing metrics …")
        metrics_all[label]  = compute_all_metrics(samples, reference)
        print_table(metrics_all[label], label=label)

    # ── Metrics JSON ──────────────────────────────────────────────────────────
    out_json = args.out_json or str(Path(args.ckpt).parent / 'eval_metrics.json')
    with open(out_json, 'w') as f:
        json.dump(metrics_all, f, indent=2)
    print(f"\nMetrics → {out_json}")

    # ── 6-panel plot ──────────────────────────────────────────────────────────
    plot_comparison(samples_dict, reference,
                    metrics_all=metrics_all, save_path=args.save)

    # ── PDB files from primary checkpoint only ────────────────────────────────
    if args.save_pdb:
        primary_label = list(samples_dict.keys())[0]
        save_pdbs(primary_samples, args.save_pdb, n_save=20, label=primary_label)


if __name__ == '__main__':
    main()
