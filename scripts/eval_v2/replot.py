"""
scripts/eval_v2/replot.py
==========================
Regenerate all evaluation figures from a saved metrics.json without re-running
model inference. All distribution data needed to reproduce figures is stored
during evaluation and read back here.

Usage
-----
# Replot everything in an existing evaluation directory:
python scripts/eval_v2/replot.py --dir plots/eval_overnight/ca_only

# Replot only specific models (subset):
python scripts/eval_v2/replot.py \
    --dir plots/eval_overnight/ca_only \
    --models "AdaLN-Transformer-Energy-Physics-v1" "TorsionTransformer-v1"

# Replot only specific sections:
python scripts/eval_v2/replot.py \
    --dir plots/eval_overnight/backbone \
    --sections physics equivariance

# Replot all groups (ca_only, backbone, all_atom) in one call:
python scripts/eval_v2/replot.py \
    --dir plots/eval_overnight \
    --all_groups

# Save to a different output directory:
python scripts/eval_v2/replot.py \
    --dir plots/eval_overnight/ca_only \
    --out_dir plots/eval_overnight/ca_only/replot/
"""

import sys
import json
import argparse
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# ─────────────────────────────────────────────────────────────────────────────
# HISTOGRAM UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def _from_hist(hist_data, n_samples: int = 1000) -> np.ndarray:
    """
    Reconstruct a sample array from a stored histogram dict {edges, counts}.
    Used to pass into plotting functions that expect raw arrays.
    Returns a 1D float array with values uniformly sampled within each bin,
    weighted by bin counts.

    For 'exact' entries (zero-variance, e.g. NeRF bonds) returns a constant array.
    """
    if hist_data is None:
        return np.array([])
    if 'exact' in hist_data:
        return np.full(n_samples, hist_data['exact'])
    edges  = np.array(hist_data['edges'])
    counts = np.array(hist_data['counts'])
    # Bin widths
    widths = np.diff(edges)
    probs  = counts * widths
    total  = probs.sum()
    if total <= 0:
        return np.array([])
    probs /= total
    # How many samples per bin
    ns = np.random.multinomial(n_samples, probs)
    vals = []
    for k, (n, lo, hi) in enumerate(zip(ns, edges[:-1], edges[1:])):
        if n > 0:
            vals.append(np.random.uniform(lo, hi, n))
    return np.concatenate(vals) if vals else np.array([])


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 1: PHYSICAL QUALITY (from stored histograms)
# ─────────────────────────────────────────────────────────────────────────────

def replot_physics(metrics: dict, labels: list, out_path: str):
    """Regenerate figure1_physics.png from stored per_model plot_data."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
    except ImportError:
        print("  [replot] matplotlib not available — skipping physics figure")
        return

    per_model = metrics['per_model']
    ref_pd    = metrics.get('ref_plot_data', {})
    ref_nm    = metrics.get('sections', {}).get('physics', {}).get('ref_metrics', {})

    fig = plt.figure(figsize=(18, 16))
    gs  = gridspec.GridSpec(4, 3, figure=fig, hspace=0.50, wspace=0.35)
    axes = [fig.add_subplot(gs[r, c]) for r in range(4) for c in range(3)]
    fig.suptitle("Physical / Biological Plausibility", fontsize=14, y=0.99)

    COLORS = ['#C44E52','#4C72B0','#55A868','#DD8452','#8172B2',
              '#937860','#DA8BC3','#8C8C8C','#CCB974','#64B5CD']
    def _col(i): return COLORS[i % len(COLORS)]

    n_models = len(labels)

    # ── Panel 0: Cα bond length histogram ────────────────────────────────────
    ax = axes[0]
    ax.set_title("Cα–Cα Bond Length (Å)", fontsize=10)
    ref_bl = _from_hist(ref_pd.get('ca_bond_hist'))
    if len(ref_bl):
        ax.hist(ref_bl, bins=50, density=True, color='#333333', alpha=0.3, label='Reference')
    for i, lbl in enumerate(labels):
        pd = per_model.get(lbl, {}).get('plot_data', {})
        vals = _from_hist(pd.get('ca_bond_hist'))
        if pd.get('ca_bond_hist', {}) and 'exact' in pd.get('ca_bond_hist', {}):
            ax.axvline(pd['ca_bond_hist']['exact'], color=_col(i), lw=2.5, label=f"{lbl} (exact)")
        elif len(vals):
            ax.hist(vals, bins=50, density=True, color=_col(i), alpha=0.6, label=lbl)
    ax.axvline(3.832, color='k', lw=1.5, linestyle='--', alpha=0.6, label='Ideal 3.832 Å')
    ax.set_xlabel("Cα–Cα distance (Å)"); ax.legend(fontsize=7); ax.grid(alpha=0.3)

    # ── Panel 1: Native bond length histogram ─────────────────────────────────
    ax = axes[1]
    ax.set_title("Native Bond Length Distribution", fontsize=10)
    for i, lbl in enumerate(labels):
        pd  = per_model.get(lbl, {}).get('plot_data', {})
        nah = pd.get('native_bond_hist')
        if nah is None:
            continue
        if 'exact' in nah:
            ax.axvline(nah['exact'], color=_col(i), lw=2.5, label=f"{lbl} (exact)")
        else:
            vals = _from_hist(nah)
            if len(vals):
                ax.hist(vals, bins=50, density=True, color=_col(i), alpha=0.6, label=lbl)
    ax.set_xlabel("Bond length (Å)"); ax.legend(fontsize=7); ax.grid(alpha=0.3)

    # ── Panel 2: Bond validity bar chart ─────────────────────────────────────
    ax = axes[2]
    ax.set_title("Bond Validity at Multiple Tolerances", fontsize=10)
    keys     = ['bond_valid_005', 'bond_valid_01', 'bond_valid_02', 'bond_valid_05']
    x_labels = ['±0.05 Å', '±0.10 Å', '±0.20 Å', '±0.50 Å']
    x_pos    = np.arange(4)
    bar_w    = 0.8 / (n_models + 1)
    for i, lbl in enumerate(labels):
        nm   = per_model.get(lbl, {}).get('native_metrics', {})
        vals = [nm.get(k, float('nan')) * 100 for k in keys]
        offs = x_pos + (i - n_models / 2.0 + 0.5) * bar_w
        ax.bar(offs, vals, bar_w * 0.9, color=_col(i), alpha=0.8, label=lbl)
    ref_vals = [ref_nm.get(k, float('nan')) * 100 for k in keys]
    for xi, rv in zip(x_pos, ref_vals):
        if rv == rv:
            ax.hlines(rv, xi - 0.35, xi + 0.35, colors='k', linewidths=1.5, linestyles='--',
                      label='Reference' if xi == 0 else '')
    ax.set_xticks(x_pos); ax.set_xticklabels(x_labels, fontsize=9)
    ax.set_ylabel("% structures valid"); ax.set_ylim(0, 105)
    ax.legend(fontsize=7); ax.grid(alpha=0.3, axis='y')

    # ── Panel 3: Per-bond RMSE ────────────────────────────────────────────────
    ax = axes[3]
    ax.set_title("Per-Bond RMSE (Cα projected)", fontsize=10)
    for i, lbl in enumerate(labels):
        cm  = per_model.get(lbl, {}).get('ca_metrics', {})
        per = cm.get('per_bond_rmse', [])
        if per and len(per) == 9:
            ax.plot(range(9), per, 'o-', color=_col(i), alpha=0.8, label=lbl, lw=1.5, ms=4)
    ax.set_xticks(range(9))
    ax.set_xticklabels([f"{i+1}–{i+2}" for i in range(9)], fontsize=8)
    ax.set_ylabel("Bond RMSE (Å)"); ax.legend(fontsize=7); ax.grid(alpha=0.3)

    # ── Panel 4: Angle RMSE bar ───────────────────────────────────────────────
    ax = axes[4]
    ax.set_title("Bond Angle Metrics", fontsize=10)
    angle_labels = ['Virtual cos RMSE', 'N-CA-C°', 'CA-C-N°', 'C-N-CA°']
    x_pos = np.arange(4); bar_w = 0.8 / (n_models + 1)
    ax2 = ax.twinx()
    for i, lbl in enumerate(labels):
        nm   = per_model.get(lbl, {}).get('native_metrics', {})
        vals = [nm.get('angle_rmse_cos', float('nan'))] + [
            nm.get('angles', {}).get(k, {}).get('rmse_deg', float('nan'))
            for k in ['N_CA_C', 'CA_C_N', 'C_N_CA']
        ]
        offs = x_pos + (i - n_models / 2.0 + 0.5) * bar_w
        v0 = vals[0]
        if v0 == v0:
            ax.bar(offs[0], v0, bar_w * 0.9, color=_col(i), alpha=0.8)
        for j in range(1, 4):
            if vals[j] == vals[j]:
                ax2.bar(offs[j], vals[j], bar_w * 0.9, color=_col(i), alpha=0.6,
                        hatch='//', label=lbl if j == 1 else '')
    ax.set_xticks(x_pos); ax.set_xticklabels(angle_labels, rotation=15, fontsize=7)
    ax.set_ylabel("cos RMSE (left)"); ax2.set_ylabel("Angle RMSE ° (right)")
    h, ll = ax2.get_legend_handles_labels()
    if h: ax2.legend(fontsize=7)
    ax.grid(alpha=0.3, axis='y')

    # ── Panel 5: ω dihedral histogram ────────────────────────────────────────
    ax = axes[5]
    ax.set_title("ω Dihedral (Peptide Planarity)", fontsize=10)
    has_omega = False
    ref_oh = ref_pd.get('omega_hist')
    if ref_oh:
        vals = _from_hist(ref_oh)
        if len(vals):
            ax.hist(vals, bins=50, density=True, color='#333333', alpha=0.3, label='Reference')
            has_omega = True
    for i, lbl in enumerate(labels):
        oh = per_model.get(lbl, {}).get('plot_data', {}).get('omega_hist')
        if oh:
            vals = _from_hist(oh)
            if len(vals):
                ax.hist(vals, bins=50, density=True, color=_col(i), alpha=0.6, label=lbl)
                has_omega = True
    if has_omega:
        ax.axvline(180, color='k', lw=1.5, linestyle='--', label='Trans (180°)')
        ax.set_xlabel("ω (°)"); ax.legend(fontsize=7)
    else:
        ax.text(0.5, 0.5, 'Not available\n(Cα-only models)', transform=ax.transAxes,
                ha='center', va='center', fontsize=11, color='gray')
    ax.grid(alpha=0.3)

    # ── Panel 6: Clash rate bar ───────────────────────────────────────────────
    ax = axes[6]
    ax.set_title("Clash Rate per Model", fontsize=10)
    clash_vals = [per_model.get(lbl, {}).get('native_metrics', {}).get('clash_rate', float('nan')) * 100
                  for lbl in labels]
    ref_clash  = ref_nm.get('clash_rate', float('nan')) * 100
    ax.bar(labels, clash_vals, color=[_col(i) for i in range(n_models)], alpha=0.8)
    if ref_clash == ref_clash:
        ax.axhline(ref_clash, color='k', lw=1.5, linestyle='--', label=f'Ref ({ref_clash:.1f}%)')
    ax.set_ylabel("Clash rate (%)"); ax.legend(fontsize=8); ax.grid(alpha=0.3, axis='y')
    ax.tick_params(axis='x', rotation=20, labelsize=8)

    # ── Panel 7: Rg histogram ─────────────────────────────────────────────────
    ax = axes[7]
    ax.set_title("Radius of Gyration (Cα)", fontsize=10)
    ref_rg = _from_hist(ref_pd.get('rg_hist'), n_samples=2000)
    if len(ref_rg):
        ax.hist(ref_rg, bins=40, density=True, color='#333333', alpha=0.3, label='Reference')
    for i, lbl in enumerate(labels):
        vals = _from_hist(per_model.get(lbl, {}).get('plot_data', {}).get('rg_hist'))
        if len(vals):
            ax.hist(vals, bins=40, density=True, color=_col(i), alpha=0.6, label=lbl)
    ax.set_xlabel("Rg (Å)"); ax.legend(fontsize=7); ax.grid(alpha=0.3)

    # ── Panel 8: ETE histogram ────────────────────────────────────────────────
    ax = axes[8]
    ax.set_title("End-to-End Distance (Cα)", fontsize=10)
    ref_ete = _from_hist(ref_pd.get('ete_hist'), n_samples=2000)
    if len(ref_ete):
        ax.hist(ref_ete, bins=40, density=True, color='#333333', alpha=0.3, label='Reference')
    for i, lbl in enumerate(labels):
        vals = _from_hist(per_model.get(lbl, {}).get('plot_data', {}).get('ete_hist'))
        if len(vals):
            ax.hist(vals, bins=40, density=True, color=_col(i), alpha=0.6, label=lbl)
    ax.set_xlabel("End-to-end (Å)"); ax.legend(fontsize=7); ax.grid(alpha=0.3)

    # ── Panel 9: Per-residue variance ─────────────────────────────────────────
    ax = axes[9]
    ax.set_title("Per-Residue Flexibility (Cα)", fontsize=10)
    ref_prv = ref_pd.get('per_residue_variance')
    if ref_prv:
        ax.plot(range(10), ref_prv, color='#333333', alpha=0.5, lw=2, label='Reference')
    for i, lbl in enumerate(labels):
        prv = per_model.get(lbl, {}).get('plot_data', {}).get('per_residue_variance')
        if prv:
            ax.plot(range(10), prv, color=_col(i), alpha=0.8, lw=1.5, label=lbl)
    ax.set_xticks(range(10))
    ax.set_xticklabels([f"R{i+1}" for i in range(10)], rotation=45, fontsize=8)
    ax.set_ylabel("Positional variance (Å²)"); ax.legend(fontsize=7); ax.grid(alpha=0.3)

    # ── Panel 10: Ramachandran 2D histogram ──────────────────────────────────
    ax = axes[10]
    rama_plotted = False
    for lbl in labels:
        rd = per_model.get(lbl, {}).get('plot_data', {}).get('ramachandran')
        if rd and rd.get('hist2d'):
            h2d = np.array(rd['hist2d'])
            xe  = np.array(rd['phi_edges'])
            ye  = np.array(rd['psi_edges'])
            ax.pcolormesh(xe, ye, h2d.T, cmap='Blues')
            ax.set_xlabel("φ (°)"); ax.set_ylabel("ψ (°)")
            suffix = ' [~NeRF, approx]' if rd.get('nerf_reconstructed') else ''
            ax.set_title(f"Ramachandran — {lbl[:20]}{suffix}", fontsize=9)
            if rd.get('nerf_reconstructed'):
                ax.text(0.02, 0.98, '⚠ NeRF angles\napproximate',
                        transform=ax.transAxes, fontsize=7, color='darkorange', va='top',
                        bbox=dict(facecolor='white', alpha=0.7))
            ax.axhline(-47, color='green', lw=0.8, linestyle=':', alpha=0.5)
            ax.axvline(-57, color='green', lw=0.8, linestyle=':', alpha=0.5)
            rama_plotted = True
            break
    if not rama_plotted:
        ax.text(0.5, 0.5, 'Ramachandran not stored', transform=ax.transAxes,
                ha='center', va='center', fontsize=10, color='gray')

    # ── Panel 11: Summary table ───────────────────────────────────────────────
    ax = axes[11]
    ax.axis('off')
    ax.set_title("Metric Summary", fontsize=10)
    hdr = ['Model', 'BondRMSE', 'Valid±0.2', 'Clash%', 'Rg (Å)', 'ETE (Å)']
    rows = []
    for lbl in labels:
        nm = per_model.get(lbl, {}).get('native_metrics', {})
        cm = per_model.get(lbl, {}).get('ca_metrics', {})
        rows.append([lbl[:14],
                     f"{nm.get('bond_rmse', float('nan')):.3f}",
                     f"{nm.get('bond_valid_02', float('nan'))*100:.1f}%",
                     f"{nm.get('clash_rate',   float('nan'))*100:.1f}%",
                     f"{cm.get('rg_mean',       float('nan')):.2f}",
                     f"{cm.get('ete_mean',      float('nan')):.2f}"])
    rows.append(['Reference',
                 f"{ref_nm.get('bond_rmse', float('nan')):.3f}",
                 f"{ref_nm.get('bond_valid_02', float('nan'))*100:.1f}%",
                 f"{ref_nm.get('clash_rate',   float('nan'))*100:.1f}%",
                 f"{ref_nm.get('rg_mean',       float('nan')):.2f}",
                 f"{ref_nm.get('ete_mean',      float('nan')):.2f}"])
    tbl = ax.table(cellText=rows, colLabels=hdr, cellLoc='center', loc='center',
                   bbox=[0, 0, 1, 1])
    tbl.auto_set_font_size(False); tbl.set_fontsize(8)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0: cell.set_facecolor('#D0D0D0')

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 2: EQUIVARIANCE (direct from stored lists)
# ─────────────────────────────────────────────────────────────────────────────

def replot_equivariance(metrics: dict, labels: list, out_path: str):
    """Regenerate figure2_equivariance.png — all data already stored as lists."""
    from scripts.eval_v2.plotting import plot_equivariance
    eq_section = metrics.get('sections', {}).get('equivariance', {})
    subset = {lbl: eq_section[lbl] for lbl in labels if lbl in eq_section}
    if not subset:
        print("  [replot] No equivariance data found for requested models — skipping")
        return
    plot_equivariance(results_dict=subset, save_path=out_path)
    print(f"  → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 3: ENERGY (from stored sweep_results + tau_hists)
# ─────────────────────────────────────────────────────────────────────────────

def replot_energy(metrics: dict, labels: list, out_dir: str):
    """Regenerate figure3_energy_<label>.png for each model using stored histograms."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [replot] matplotlib not available — skipping energy figure")
        return

    en_section = metrics.get('sections', {}).get('energy', {})

    TAU_CMAP = plt.cm.coolwarm
    taus_all = [0.0, 0.25, 0.5, 0.75, 1.0]

    for lbl in labels:
        res = en_section.get(lbl)
        if res is None or res.get('skipped'):
            continue

        sweep = res.get('sweep_results', [])
        mono  = res.get('monotonicity', {})
        tau_hists = res.get('tau_hists', {})

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f"Energy Conditioning — {lbl}", fontsize=13)
        axes = axes.flatten()

        taus     = [s[0] for s in sweep]
        rg_means = [s[1].get('rg_mean', float('nan')) for s in sweep]
        rg_stds  = [s[1].get('rg_std',  0) for s in sweep]
        ete_means = [s[1].get('ete_mean', float('nan')) for s in sweep]
        ete_stds  = [s[1].get('ete_std',  0) for s in sweep]
        bv_vals   = [s[1].get('bond_valid', float('nan')) * 100 for s in sweep]
        div_vals  = [s[1].get('diversity', float('nan')) for s in sweep]

        # Panel 0: Rg histograms per tau
        ax = axes[0]
        for tau, s in sweep:
            th = tau_hists.get(str(tau), {})
            rg_h = th.get('rg_hist')
            if rg_h:
                color = TAU_CMAP(tau)
                vals = _from_hist(rg_h, n_samples=500)
                if len(vals):
                    ax.hist(vals, bins=25, density=True, color=color,
                            alpha=0.5, label=f"τ={tau:.2f}")
        ax.set_xlabel("Rg (Å)"); ax.set_title("Rg Distribution per τ", fontsize=10)
        ax.legend(fontsize=7); ax.grid(alpha=0.3)

        # Panel 1: ETE histograms per tau
        ax = axes[1]
        for tau, s in sweep:
            th = tau_hists.get(str(tau), {})
            ete_h = th.get('ete_hist')
            if ete_h:
                color = TAU_CMAP(tau)
                vals = _from_hist(ete_h, n_samples=500)
                if len(vals):
                    ax.hist(vals, bins=25, density=True, color=color,
                            alpha=0.5, label=f"τ={tau:.2f}")
        ax.set_xlabel("ETE (Å)"); ax.set_title("ETE Distribution per τ", fontsize=10)
        ax.legend(fontsize=7); ax.grid(alpha=0.3)

        # Panel 2: Rg mean ± std vs tau
        ax = axes[2]
        ax.errorbar(taus, rg_means, yerr=rg_stds, fmt='o-', color='#C44E52',
                    lw=2, ms=7, capsize=4, label='Rg mean ± std')
        rg_r = mono.get('rg_monotone', {}).get('r', float('nan'))
        rg_p = mono.get('rg_monotone', {}).get('p', float('nan'))
        pass_str = "PASS" if abs(rg_r) > 0.8 and rg_p < 0.05 and rg_r > 0 else "FAIL"
        ax.set_xlabel("τ"); ax.set_ylabel("Rg (Å)")
        ax.set_title(f"Rg vs τ  |  Spearman r={rg_r:.2f} ({pass_str})", fontsize=10)
        ax.grid(alpha=0.3); ax.legend(fontsize=8)

        # Panel 3: ETE mean ± std vs tau
        ax = axes[3]
        ax.errorbar(taus, ete_means, yerr=ete_stds, fmt='o-', color='#4C72B0',
                    lw=2, ms=7, capsize=4, label='ETE mean ± std')
        ete_r = mono.get('ete_monotone', {}).get('r', float('nan'))
        ete_p = mono.get('ete_monotone', {}).get('p', float('nan'))
        pass_str = "PASS" if abs(ete_r) > 0.8 and ete_p < 0.05 and ete_r > 0 else "FAIL"
        ax.set_xlabel("τ"); ax.set_ylabel("ETE (Å)")
        ax.set_title(f"ETE vs τ  |  Spearman r={ete_r:.2f} ({pass_str})", fontsize=10)
        ax.grid(alpha=0.3); ax.legend(fontsize=8)

        # Panel 4: Bond validity vs tau
        ax = axes[4]
        ax.plot(taus, bv_vals, 'o-', color='#55A868', lw=2, ms=7)
        ax.set_xlabel("τ"); ax.set_ylabel("Bond validity ±0.5 Å (%)")
        ax.set_title("Bond Validity vs τ\n(should be ~flat)", fontsize=10)
        ax.set_ylim(0, 105); ax.grid(alpha=0.3)

        # Panel 5: Diversity vs tau
        ax = axes[5]
        ax.plot(taus, div_vals, 'o-', color='#DD8452', lw=2, ms=7)
        div_r = mono.get('div_monotone', {}).get('r', float('nan'))
        ax.set_xlabel("τ"); ax.set_ylabel("Pairwise RMSD diversity (Å)")
        ax.set_title(f"Diversity vs τ  |  r={div_r:.2f}", fontsize=10)
        ax.grid(alpha=0.3)

        safe_lbl = lbl.replace('/', '_')
        save_path = str(Path(out_dir) / f'figure3_energy_{safe_lbl}.png')
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  → {save_path}")

    # τ-vs-REU plot (all energy-conditioned models together)
    replot_tau_reu(metrics, labels, out_dir)


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 3b: τ vs ROSETTA ENERGY (from stored tau_rosetta dicts)
# ─────────────────────────────────────────────────────────────────────────────

def replot_tau_reu(metrics: dict, labels: list, out_dir: str):
    """Regenerate figure3b_tau_reu.png from stored tau_rosetta data in the energy section."""
    en_section = metrics.get('sections', {}).get('energy', {})

    tau_reu_all = {
        lbl: en_section[lbl]['tau_rosetta']
        for lbl in labels
        if lbl in en_section
        and not en_section[lbl].get('skipped')
        and en_section[lbl].get('tau_rosetta')
    }
    if not tau_reu_all:
        print("  [replot] No tau_rosetta data found — skipping figure3b_tau_reu")
        return

    from scripts.eval_v2.plotting import plot_tau_reu
    save_path = str(Path(out_dir) / 'figure3b_tau_reu.png')
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plot_tau_reu(tau_reu_dict=tau_reu_all, save_path=save_path)
    print(f"  → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 4: NOVELTY (from stored NND arrays, PCA coords, RMSD matrix)
# ─────────────────────────────────────────────────────────────────────────────

def replot_novelty(metrics: dict, labels: list, out_dir: str):
    """Regenerate figure4_novelty_<label>.png from stored plot_data."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
    except ImportError:
        print("  [replot] matplotlib not available — skipping novelty figure")
        return

    nov_section = metrics.get('sections', {}).get('novelty', {})

    for lbl in labels:
        nov = nov_section.get(lbl)
        if nov is None:
            continue
        if nov.get('skipped'):
            print(f"  [{lbl}] novelty skipped during eval — skip replot")
            continue

        pd        = nov.get('plot_data', {})
        gen_nnd   = np.array(pd['gen_nnd'])  if pd.get('gen_nnd')  else None
        self_nnd  = np.array(pd['self_nnd']) if pd.get('self_nnd') else None
        pca_gen   = np.array(pd['pca_gen_xy']) if pd.get('pca_gen_xy') else None
        pca_ref   = np.array(pd['pca_ref_xy']) if pd.get('pca_ref_xy') else None
        cov_curve = pd.get('coverage_curve')
        rmsd_mat  = np.array(pd['rmsd_matrix']) if pd.get('rmsd_matrix') else None
        per_tau   = nov.get('per_tau', {})

        thr       = nov.get('threshold', 1.0)
        nnd_ratio = nov.get('nnd_ratio', float('nan'))
        coverage  = nov.get('coverage',  float('nan'))
        precision = nov.get('precision', float('nan'))
        n_valid   = nov.get('n_valid_structures', 0)
        n_total   = nov.get('n_generated', n_valid)

        fig = plt.figure(figsize=(18, 10))
        gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)
        axes = [fig.add_subplot(gs[r, c]) for r in range(2) for c in range(3)]
        fig.suptitle(f"Conformational Novelty — {lbl}", fontsize=13)

        # Panel 0: NND distribution
        ax = axes[0]
        if gen_nnd is not None and self_nnd is not None:
            hi = max(gen_nnd.max(), self_nnd.max()) * 1.1
            bins = np.linspace(0, hi, 50)
            ax.hist(self_nnd, bins=bins, density=True, color='#333333', alpha=0.4,
                    label='Test→Test (baseline)')
            ax.hist(gen_nnd,  bins=bins, density=True, color='#C44E52', alpha=0.7,
                    label='Generated→Test (filtered)')
            ax.axvline(thr, color='k', lw=1.5, linestyle='--', label=f'Threshold ({thr:.2f})')
            ax.text(0.02, 0.97, f"NND ratio: {nnd_ratio:.3f}", transform=ax.transAxes,
                    fontsize=9, va='top', bbox=dict(facecolor='white', alpha=0.7))
        else:
            ax.text(0.5, 0.5, 'No NND data', transform=ax.transAxes,
                    ha='center', va='center', fontsize=10, color='gray')
        ax.set_xlabel("NND"); ax.set_title("NND Distribution", fontsize=10)
        ax.legend(fontsize=7); ax.grid(alpha=0.3)

        # Panel 1: PCA scatter
        ax = axes[1]
        if pca_gen is not None and pca_ref is not None:
            ax.scatter(pca_ref[:, 0], pca_ref[:, 1], s=4, alpha=0.2, color='#333333',
                       label='Test set', rasterized=True)
            ax.scatter(pca_gen[:, 0], pca_gen[:, 1], s=6, alpha=0.5, color='#C44E52',
                       label='Generated (valid)', rasterized=True)
            ax.legend(fontsize=7, markerscale=3)
        else:
            ax.text(0.5, 0.5, 'No PCA data\n(no valid structures)',
                    transform=ax.transAxes, ha='center', va='center', fontsize=10, color='gray')
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
        ax.set_title("PCA of Conformational Space", fontsize=10)
        ax.grid(alpha=0.2)

        # Panel 2: Coverage / Precision curve
        ax = axes[2]
        if cov_curve:
            thrs  = [pt[0] for pt in cov_curve]
            covs  = [pt[1] * 100 for pt in cov_curve]
            precs = [pt[2] * 100 for pt in cov_curve]
            ax.plot(thrs, covs,  color='#4C72B0', lw=2, label='Coverage')
            ax.plot(thrs, precs, color='#55A868', lw=2, label='Precision')
            ax.axvline(thr, color='k', lw=1, linestyle='--',
                       label=f'Ref threshold ({thr:.2f})')
            ax.legend(fontsize=7)
        else:
            ax.text(0.5, 0.5, 'No coverage data', transform=ax.transAxes,
                    ha='center', va='center', fontsize=10, color='gray')
        ax.set_xlabel("Distance threshold"); ax.set_ylabel("%")
        ax.set_title("Coverage & Precision", fontsize=10)
        ax.set_ylim(0, 105); ax.grid(alpha=0.3)

        # Panel 3: Per-tau NND ratio
        ax = axes[3]
        if per_tau:
            taus_plot = sorted(float(k) for k in per_tau.keys())
            ratios = [per_tau.get(str(t), per_tau.get(t, {})).get('nnd_ratio', float('nan'))
                      for t in taus_plot]
            valid_frs = [per_tau.get(str(t), per_tau.get(t, {})).get('valid_fraction',
                                                                      float('nan')) * 100
                         for t in taus_plot]
            ax.plot(taus_plot, ratios, 'o-', color='#C44E52', lw=2, ms=7, label='NND ratio')
            ax.axhline(1.0, color='k', lw=1.5, linestyle='--', alpha=0.5)
            ax2 = ax.twinx()
            ax2.plot(taus_plot, valid_frs, 's--', color='#4C72B0', lw=1.5, ms=5,
                     label='Valid %')
            ax2.set_ylabel("Valid structures (%)", fontsize=9)
            ax2.legend(fontsize=7, loc='lower right')
            ax.legend(fontsize=7, loc='upper left')
        else:
            ax.text(0.5, 0.5, 'Not available\n(non-energy model)', transform=ax.transAxes,
                    ha='center', va='center', fontsize=10, color='gray')
        ax.set_xlabel("τ"); ax.set_ylabel("NND ratio")
        ax.set_title("NND Ratio per Temperature", fontsize=10); ax.grid(alpha=0.3)

        # Panel 4: RMSD heatmap
        ax = axes[4]
        if rmsd_mat is not None and len(rmsd_mat) > 0:
            im = ax.imshow(rmsd_mat, cmap='viridis', aspect='auto')
            plt.colorbar(im, ax=ax, label='RMSD (Å)', fraction=0.04)
            mean_rmsd = rmsd_mat[np.triu_indices_from(rmsd_mat, k=1)].mean()
            ax.set_title(f"Pairwise RMSD Heatmap\n({rmsd_mat.shape[0]} structures)", fontsize=10)
            ax.text(0.02, 0.02, f"Mean RMSD: {mean_rmsd:.2f} Å", transform=ax.transAxes,
                    fontsize=8, bbox=dict(facecolor='white', alpha=0.8))
        else:
            ax.text(0.5, 0.5, 'No RMSD data', transform=ax.transAxes,
                    ha='center', va='center', fontsize=10, color='gray')
        ax.set_xlabel("Structure"); ax.set_ylabel("Structure")

        # Panel 5: Summary text
        ax = axes[5]
        ax.axis('off')
        vf = n_valid / max(n_total, 1) * 100
        novel_str = ('Novel (>1.1)' if nnd_ratio > 1.1
                     else ('Memorised (<0.9)' if nnd_ratio < 0.9
                           else 'Matches distribution'))
        txt = (f"Novelty Summary — {lbl}\n\n"
               f"Generated:      {n_total}\n"
               f"Physics-valid:  {n_valid} ({vf:.1f}%)\n\n"
               f"NND ratio:      {nnd_ratio:.3f}\n"
               f"  → {novel_str}\n\n"
               f"Coverage:       {coverage*100:.1f}%\n"
               f"Precision:      {precision*100:.1f}%\n")
        ax.text(0.1, 0.9, txt, transform=ax.transAxes, fontsize=10, va='top',
                family='monospace',
                bbox=dict(facecolor='#f8f8f8', alpha=0.8, edgecolor='#cccccc'))

        safe_lbl = lbl.replace('/', '_')
        save_path = str(Path(out_dir) / f'figure4_novelty_{safe_lbl}.png')
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 5: PYROSETTA (from stored per_structure list)
# ─────────────────────────────────────────────────────────────────────────────

def replot_pyrosetta(metrics: dict, labels: list, out_dir: str):
    """Regenerate figure5_pyrosetta_<label>.png from stored per_structure scores."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [replot] matplotlib not available — skipping pyrosetta figure")
        return

    ros_section = metrics.get('sections', {}).get('pyrosetta', {})

    for lbl in labels:
        res = ros_section.get(lbl)
        if res is None or res.get('skipped'):
            continue

        per_struct = res.get('per_structure', [])
        if not per_struct:
            print(f"  [{lbl}] no per_structure data stored — cannot replot PyRosetta figure")
            print(f"         (re-evaluate with the updated pipeline to store per_structure data)")
            continue

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f"PyRosetta Validation — {lbl}", fontsize=13)
        axes = axes.flatten()

        def _vals(key):
            return [s[key] for s in per_struct
                    if s and s.get(key) is not None and s.get(key) == s.get(key)]

        for ax_idx, (key, title) in enumerate([
            ('total_score', 'Total Rosetta Score (REF2015)'),
            ('fa_rep',      'fa_rep (VdW Clash Term)'),
            ('rama_prepro', 'Ramachandran Score'),
            ('fa_dun',      'Rotamer Score (fa_dun)'),
        ]):
            ax = axes[ax_idx]
            vals = _vals(key)
            if vals:
                ax.hist(vals, bins=20, density=True, color='#C44E52', alpha=0.7,
                        label=f'Generated (n={len(vals)})')
            if key == 'fa_rep':
                ax.axvline(10.0, color='red', lw=1.5, linestyle='--', label='Max (10.0)')
            elif key == 'rama_prepro':
                ax.axvline(2.0, color='red', lw=1.5, linestyle='--', label='Max (2.0)')
            elif key == 'total_score':
                thr = res.get('total_threshold')
                if thr is not None:
                    ax.axvline(thr, color='red', lw=1.5, linestyle='--',
                               label=f'Threshold ({thr:.1f})')
            ax.set_xlabel(key); ax.set_title(title, fontsize=10)
            ax.legend(fontsize=7); ax.grid(alpha=0.3)

        plt.tight_layout()
        safe_lbl = lbl.replace('/', '_')
        save_path = str(Path(out_dir) / f'figure5_pyrosetta_{safe_lbl}.png')
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Regenerate all eval_v2 figures from a saved metrics.json.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--dir', required=True,
                   help='Directory containing metrics.json (e.g. plots/eval_overnight/ca_only)')
    p.add_argument('--models', nargs='*', default=None,
                   help='Subset of model labels to replot. Default: all models in JSON.')
    p.add_argument('--sections', nargs='+',
                   choices=['physics', 'equivariance', 'energy', 'novelty', 'pyrosetta'],
                   default=['physics', 'equivariance', 'energy', 'novelty', 'pyrosetta'],
                   help='Which figures to regenerate.')
    p.add_argument('--out_dir', default=None,
                   help='Output directory. Default: same as --dir (overwrites existing figures).')
    p.add_argument('--all_groups', action='store_true',
                   help='Replot all three atom groups (ca_only, backbone, all_atom) found in --dir.')
    args = p.parse_args()

    base_dir = Path(args.dir)

    if args.all_groups:
        groups = [d for d in ['ca_only', 'backbone', 'all_atom']
                  if (base_dir / d / 'metrics.json').exists()]
        if not groups:
            print(f"No ca_only/backbone/all_atom sub-directories with metrics.json found in {base_dir}")
            return
        for g in groups:
            print(f"\n{'='*60}")
            print(f"  Replotting group: {g}")
            print(f"{'='*60}")
            _replot_dir(base_dir / g, args.models, args.sections,
                        args.out_dir and Path(args.out_dir) / g)
    else:
        _replot_dir(base_dir, args.models, args.sections,
                    Path(args.out_dir) if args.out_dir else None)


def _replot_dir(eval_dir: Path, model_filter, sections, out_dir):
    json_path = eval_dir / 'metrics.json'
    if not json_path.exists():
        print(f"  [replot] No metrics.json found at {json_path}")
        return

    with open(json_path) as f:
        metrics = json.load(f)

    all_labels = list(metrics.get('per_model', {}).keys())
    labels = model_filter if model_filter else all_labels

    # Validate
    missing = [lbl for lbl in labels if lbl not in metrics.get('per_model', {})]
    if missing:
        print(f"  WARNING: labels not found in metrics.json: {missing}")
        labels = [lbl for lbl in labels if lbl in metrics.get('per_model', {})]

    if not labels:
        print("  No valid model labels — nothing to plot.")
        return

    out = out_dir or eval_dir
    out.mkdir(parents=True, exist_ok=True)

    print(f"  Replotting {len(labels)} models: {labels}")
    print(f"  Sections: {sections}")
    print(f"  Output → {out}/")

    if 'physics' in sections:
        print("\n  [Figure 1] Physics quality...")
        replot_physics(metrics, labels, str(out / 'figure1_physics.png'))

    if 'equivariance' in sections:
        print("\n  [Figure 2] Equivariance...")
        replot_equivariance(metrics, labels, str(out / 'figure2_equivariance.png'))

    if 'energy' in sections:
        print("\n  [Figure 3] Energy conditioning...")
        replot_energy(metrics, labels, str(out))

    if 'novelty' in sections:
        print("\n  [Figure 4] Novelty...")
        replot_novelty(metrics, labels, str(out))

    if 'pyrosetta' in sections:
        print("\n  [Figure 5] PyRosetta...")
        replot_pyrosetta(metrics, labels, str(out))

    print("\n  Done.")


if __name__ == '__main__':
    main()
