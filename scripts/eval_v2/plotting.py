"""
scripts/eval_v2/plotting.py
=============================
All figure generation for the eval_v2 pipeline.

Five figures:
  Figure 1 (physics_quality)   — bond metrics, clashes, Rg/ETE, flexibility, Ramachandran
  Figure 2 (equivariance)      — Tests 1–4 bar/line charts
  Figure 3 (energy_analysis)   — temperature sweep panels + PCA
  Figure 4 (novelty)           — NND, PCA, coverage/precision, RMSD heatmap
  Figure 5 (pyrosetta)         — Rosetta score distributions

Each function accepts a `models_data` dict keyed by model label, plus a reference array.
"""

import sys
import numpy as np
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from .constants import MODEL_COLORS, REF_COLOR, REF_ALPHA
from .model_utils import ca_from_coords
from .physics_metrics import (
    bond_lengths_ca, radius_of_gyration, end_to_end,
    omega_dihedrals, compute_ramachandran,
)

# ── Safe imports ──────────────────────────────────────────────────────────────
try:
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.colors import LinearSegmentedColormap
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


def _check_mpl():
    if not HAS_MPL:
        raise ImportError("matplotlib not installed — cannot generate plots")


def _safe_hist(ax, values: np.ndarray, bins: int = 50, density: bool = True,
               color: str = '#333333', alpha: float = 0.6, label: str = '',
               exact_label_suffix: str = ' (exact)'):
    """
    Plot a histogram, gracefully handling zero-variance data (e.g., NeRF-exact bonds).
    If all values are identical, draws a vertical line instead of a histogram.
    """
    vals = np.asarray(values).flatten()
    if len(vals) == 0:
        return
    if vals.std() < 1e-6:
        ax.axvline(float(vals[0]), color=color, lw=2.5,
                   label=f"{label}{exact_label_suffix}")
    else:
        ax.hist(vals, bins=bins, density=density, color=color, alpha=alpha, label=label)


def _save_or_show(fig, save_path: Optional[str], dpi: int = 150):
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"  Plot saved → {save_path}")
    else:
        plt.show()
    plt.close(fig)


def _color(i: int) -> str:
    return MODEL_COLORS[i % len(MODEL_COLORS)]


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 1: PHYSICAL QUALITY
# ─────────────────────────────────────────────────────────────────────────────

def plot_physics(
    models_data: dict,      # {label: (coords_native, metrics_native, metrics_ca)}
    ref_coords: np.ndarray,
    ref_metrics: dict,
    save_path: Optional[str] = None,
):
    """
    4×3 = 12-panel figure.
    models_data[label] = (coords_native, native_metrics, ca_metrics)
    """
    _check_mpl()

    labels = list(models_data.keys())
    n_models = len(labels)

    fig = plt.figure(figsize=(18, 16))
    gs  = gridspec.GridSpec(4, 3, figure=fig, hspace=0.50, wspace=0.35)
    axes = [fig.add_subplot(gs[r, c]) for r in range(4) for c in range(3)]

    fig.suptitle("Physical / Biological Plausibility", fontsize=14, y=0.99)

    ref_ca = ca_from_coords(ref_coords)

    # ── Row 0: Bond lengths ──────────────────────────────────────────────────
    ax = axes[0]
    ax.set_title("Cα–Cα Bond Length (Å)", fontsize=10)
    ref_bl = bond_lengths_ca(ref_ca).flatten()
    _safe_hist(ax, ref_bl, color=REF_COLOR, alpha=REF_ALPHA, label='Reference')
    for i, (lbl, (coords, nm, cm)) in enumerate(models_data.items()):
        ca = ca_from_coords(coords)
        bl = bond_lengths_ca(ca).flatten()
        _safe_hist(ax, bl, color=_color(i), label=lbl)
    ax.axvline(3.832, color='k', lw=1.5, linestyle='--', alpha=0.6, label='Ideal 3.832 Å')
    ax.axvline(3.332, color='gray', lw=1, linestyle=':', alpha=0.4)
    ax.axvline(4.332, color='gray', lw=1, linestyle=':', alpha=0.4, label='±0.5 Å')
    ax.set_xlabel("Cα–Cα distance (Å)"); ax.legend(fontsize=7); ax.grid(alpha=0.3)

    ax = axes[1]
    ax.set_title("Native Bond Length Distribution", fontsize=10)
    for i, (lbl, (coords, nm, cm)) in enumerate(models_data.items()):
        n_atoms = coords.shape[1]
        if n_atoms == 10:
            bl = bond_lengths_ca(coords).flatten()
            lbl_str = f"{lbl} (Cα-Cα)"
        elif n_atoms == 30:
            from models.backbone_physics import _SRC, _DST, _IDEAL
            src = np.array(_SRC); dst = np.array(_DST)
            bl = np.linalg.norm(coords[:, dst] - coords[:, src], axis=-1).flatten()
            lbl_str = f"{lbl} (backbone)"
        elif n_atoms == 93:
            from models.physics_aa import _BOND_INDICES
            consec = np.linalg.norm(np.diff(coords, axis=1), axis=-1)
            bl = consec[:, _BOND_INDICES].flatten()
            lbl_str = f"{lbl} (all-atom)"
        else:
            continue
        if bl.std() < 1e-6:
            # Zero-variance (NeRF-reconstructed torsion/backbone model — exact bonds)
            ax.axvline(float(bl[0]), color=_color(i), lw=2.5, label=f"{lbl_str} (exact)")
        else:
            ax.hist(bl, bins=50, density=True, color=_color(i), alpha=0.6, label=lbl_str)
    ax.set_xlabel("Bond length (Å)"); ax.legend(fontsize=7); ax.grid(alpha=0.3)

    # ── Bond validity bar chart ──────────────────────────────────────────────
    ax = axes[2]
    ax.set_title("Bond Validity at Multiple Tolerances", fontsize=10)
    keys   = ['bond_valid_005', 'bond_valid_01', 'bond_valid_02', 'bond_valid_05']
    labels_x = ['±0.05 Å', '±0.10 Å', '±0.20 Å', '±0.50 Å']
    x_pos  = np.arange(4)
    bar_w  = 0.8 / (n_models + 1)
    for i, (lbl, (coords, nm, cm)) in enumerate(models_data.items()):
        vals = [nm.get(k, float('nan')) * 100 for k in keys]
        offsets = x_pos + (i - n_models / 2.0 + 0.5) * bar_w
        ax.bar(offsets, vals, bar_w * 0.9, color=_color(i), alpha=0.8, label=lbl)
    ref_vals = [ref_metrics.get(k, float('nan')) * 100 for k in keys]
    for xi, rv in zip(x_pos, ref_vals):
        if not np.isnan(rv):
            ax.hlines(rv, xi - 0.35, xi + 0.35, colors='k', linewidths=1.5,
                      linestyles='--', label='Reference' if xi == 0 else '')
    ax.set_xticks(x_pos); ax.set_xticklabels(labels_x, fontsize=9)
    ax.set_ylabel("% structures (all bonds valid)"); ax.set_ylim(0, 105)
    ax.legend(fontsize=7); ax.grid(alpha=0.3, axis='y')

    # ── Row 1: Per-bond RMSE ────────────────────────────────────────────────
    ax = axes[3]
    ax.set_title("Per-Bond RMSE (Cα projected)", fontsize=10)
    from .constants import CA_IDEAL_BOND
    ref_bl_ca = bond_lengths_ca(ref_ca)
    ref_per = np.sqrt(((ref_bl_ca - CA_IDEAL_BOND) ** 2).mean(axis=0))
    ax.plot(range(9), ref_per, 'o--', color=REF_COLOR, alpha=0.6, label='Reference', lw=1.5, ms=4)
    for i, (lbl, (coords, nm, cm)) in enumerate(models_data.items()):
        per = cm.get('per_bond_rmse', [])
        if per and len(per) == 9:
            ax.plot(range(9), per, 'o-', color=_color(i), alpha=0.8, label=lbl, lw=1.5, ms=4)
    ax.set_xticks(range(9))
    ax.set_xticklabels([f"{i+1}–{i+2}" for i in range(9)], fontsize=8)
    ax.set_ylabel("Bond RMSE (Å)"); ax.legend(fontsize=7); ax.grid(alpha=0.3)

    # ── Angle RMSE bar chart ─────────────────────────────────────────────────
    ax = axes[4]
    ax.set_title("Bond Angle Metrics", fontsize=10)
    angle_labels  = ['Virtual cos RMSE (Cα)', 'N-CA-C RMSE°', 'CA-C-N RMSE°', 'C-N-CA RMSE°']
    angle_keys_ca = [('angle_rmse_cos', False)]
    angle_keys_bb = [('angles', 'N_CA_C', 'rmse_deg'), ('angles', 'CA_C_N', 'rmse_deg'),
                     ('angles', 'C_N_CA', 'rmse_deg')]

    x_pos = np.arange(4)
    bar_w = 0.8 / (n_models + 1)
    ax2 = ax.twinx()

    for i, (lbl, (coords, nm, cm)) in enumerate(models_data.items()):
        n_atoms = coords.shape[1]
        vals = [float('nan')] * 4
        vals[0] = nm.get('angle_rmse_cos', cm.get('angle_rmse_cos', float('nan')))
        if 'angles' in nm:
            for j, aname in enumerate(['N_CA_C', 'CA_C_N', 'C_N_CA']):
                vals[j+1] = nm['angles'][aname].get('rmse_deg', float('nan'))
        offsets = x_pos + (i - n_models / 2.0 + 0.5) * bar_w
        # First bar (cos scale) on ax, rest (degree scale) on ax2
        if not np.isnan(vals[0]):
            ax.bar(offsets[0], vals[0], bar_w * 0.9, color=_color(i), alpha=0.8)
        for j in range(1, 4):
            if not np.isnan(vals[j]):
                ax2.bar(offsets[j], vals[j], bar_w * 0.9, color=_color(i), alpha=0.6,
                        hatch='//', label=lbl if j == 1 else '')

    ax.set_xticks(x_pos); ax.set_xticklabels(angle_labels, rotation=15, fontsize=7)
    ax.set_ylabel("cos RMSE (left)"); ax2.set_ylabel("Angle RMSE ° (right)")
    handles, lbls = ax2.get_legend_handles_labels()
    if handles:
        ax2.legend(fontsize=7)
    ax.grid(alpha=0.3, axis='y')

    # ── ω dihedral distribution ──────────────────────────────────────────────
    ax = axes[5]
    ax.set_title("ω Dihedral (Peptide Bond Planarity)", fontsize=10)
    has_omega = False
    for i, (lbl, (coords, nm, cm)) in enumerate(models_data.items()):
        if coords.shape[1] == 30:
            omega = omega_dihedrals(coords).flatten()
            ax.hist(omega, bins=50, density=True, color=_color(i), alpha=0.6, label=lbl)
            has_omega = True
        elif coords.shape[1] == 93:
            omega = omega_dihedrals(coords[:, :30]).flatten()
            ax.hist(omega, bins=50, density=True, color=_color(i), alpha=0.6, label=lbl)
            has_omega = True
    if ref_coords.shape[1] >= 30:
        omega_ref = omega_dihedrals(ref_coords[:, :30] if ref_coords.shape[1] > 30
                                    else ref_coords).flatten()
        ax.hist(omega_ref, bins=50, density=True, color=REF_COLOR, alpha=REF_ALPHA,
                label='Reference')
        has_omega = True
    if not has_omega:
        ax.text(0.5, 0.5, 'Not available\n(Cα-only models)', transform=ax.transAxes,
                ha='center', va='center', fontsize=11, color='gray')
    else:
        ax.axvline(180, color='k', lw=1.5, linestyle='--', label='Trans (180°)')
        ax.set_xlabel("ω dihedral (°)"); ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    # ── Row 2: Clashes, Rg, ETE ─────────────────────────────────────────────
    ax = axes[6]
    ax.set_title("Clash Rate per Model", fontsize=10)
    model_labels = list(models_data.keys())
    clash_vals   = [nm.get('clash_rate', float('nan')) * 100
                    for lbl, (coords, nm, cm) in models_data.items()]
    ref_clash = ref_metrics.get('clash_rate', float('nan')) * 100
    bars = ax.bar(model_labels, clash_vals,
                  color=[_color(i) for i in range(n_models)], alpha=0.8)
    if not np.isnan(ref_clash):
        ax.axhline(ref_clash, color='k', lw=1.5, linestyle='--', label=f'Ref ({ref_clash:.1f}%)')
    ax.set_ylabel("Clash rate (%)"); ax.set_ylim(0, max(100, max(v for v in clash_vals
                                                                    if not np.isnan(v)) * 1.2
                                                          if any(not np.isnan(v) for v in clash_vals)
                                                          else 100))
    ax.legend(fontsize=8); ax.grid(alpha=0.3, axis='y')
    ax.tick_params(axis='x', rotation=20, labelsize=8)

    ax = axes[7]
    ax.set_title("Radius of Gyration (Cα)", fontsize=10)
    ref_rg = radius_of_gyration(ref_ca)
    _safe_hist(ax, ref_rg, bins=40, color=REF_COLOR, alpha=REF_ALPHA, label='Reference')
    for i, (lbl, (coords, nm, cm)) in enumerate(models_data.items()):
        ca = ca_from_coords(coords)
        rg = radius_of_gyration(ca)
        _safe_hist(ax, rg, bins=40, color=_color(i), label=lbl)
    ax.set_xlabel("Rg (Å)"); ax.legend(fontsize=7); ax.grid(alpha=0.3)

    ax = axes[8]
    ax.set_title("End-to-End Distance (Cα)", fontsize=10)
    ref_ete = end_to_end(ref_ca)
    _safe_hist(ax, ref_ete, bins=40, color=REF_COLOR, alpha=REF_ALPHA, label='Reference')
    for i, (lbl, (coords, nm, cm)) in enumerate(models_data.items()):
        ca = ca_from_coords(coords)
        ete = end_to_end(ca)
        _safe_hist(ax, ete, bins=40, color=_color(i), label=lbl)
    ax.set_xlabel("End-to-end (Å)"); ax.legend(fontsize=7); ax.grid(alpha=0.3)

    # ── Row 3: Per-residue flexibility, Ramachandran, summary ───────────────
    ax = axes[9]
    ax.set_title("Per-Residue Flexibility (Cα)", fontsize=10)
    ref_var = ref_ca.var(axis=0).sum(axis=-1)
    ax.plot(range(10), ref_var, color=REF_COLOR, alpha=0.6, lw=2, label='Reference')
    for i, (lbl, (coords, nm, cm)) in enumerate(models_data.items()):
        ca = ca_from_coords(coords)
        var = ca.var(axis=0).sum(axis=-1)
        ax.plot(range(10), var, color=_color(i), alpha=0.8, lw=1.5, label=lbl)
    ax.set_xticks(range(10))
    ax.set_xticklabels([f"R{i+1}" for i in range(10)], rotation=45, fontsize=8)
    ax.set_ylabel("Positional variance (Å²)"); ax.legend(fontsize=7); ax.grid(alpha=0.3)

    # ── Ramachandran heatmap (first model that has it, with note) ───────────
    ax = axes[10]
    ax.set_title("Ramachandran Plot (first model)", fontsize=10)
    rama_plotted = False
    for i, (lbl, (coords, nm, cm)) in enumerate(models_data.items()):
        try:
            rama = compute_ramachandran(coords[:min(500, len(coords))])
            phi_flat = np.array(rama['phi']).flatten()
            psi_flat = np.array(rama['psi']).flatten()
            is_nerf  = rama.get('nerf_reconstructed', False)
            ax.hist2d(phi_flat, psi_flat, bins=60, cmap='Blues',
                      range=[[-180, 180], [-180, 180]])
            ax.set_xlabel("φ (°)"); ax.set_ylabel("ψ (°)")
            suffix = ' [~NeRF, approx]' if is_nerf else ''
            ax.set_title(f"Ramachandran — {lbl}{suffix}", fontsize=9)
            if is_nerf:
                ax.text(0.02, 0.98, '⚠ NeRF-reconstructed\nangles approximate',
                        transform=ax.transAxes, fontsize=7, color='darkorange',
                        va='top', bbox=dict(facecolor='white', alpha=0.7))
            # Draw approximate favoured region boundaries
            ax.axhline(-47, color='green', lw=0.8, linestyle=':', alpha=0.5)
            ax.axvline(-57, color='green', lw=0.8, linestyle=':', alpha=0.5)
            rama_plotted = True
            break
        except Exception as e:
            continue

    if not rama_plotted:
        ax.text(0.5, 0.5, 'Ramachandran not computed', transform=ax.transAxes,
                ha='center', va='center', fontsize=10, color='gray')

    # ── Summary table panel ─────────────────────────────────────────────────
    ax = axes[11]
    ax.axis('off')
    ax.set_title("Metric Summary", fontsize=10)
    rows_hdr = ['Model', 'BondRMSE', 'Valid±0.2', 'Clash%', 'Rg (Å)', 'ETE (Å)', 'Diversity']
    table_data = []
    for lbl, (coords, nm, cm) in models_data.items():
        table_data.append([
            lbl[:12],
            f"{nm.get('bond_rmse', float('nan')):.4f}",
            f"{nm.get('bond_valid_02', float('nan'))*100:.1f}%",
            f"{nm.get('clash_rate', float('nan'))*100:.1f}%",
            f"{cm.get('rg_mean', float('nan')):.2f}",
            f"{cm.get('ete_mean', float('nan')):.2f}",
            f"{cm.get('diversity', float('nan')):.2f}",
        ])
    # Reference row
    table_data.append([
        'Reference',
        f"{ref_metrics.get('bond_rmse', float('nan')):.4f}",
        f"{ref_metrics.get('bond_valid_02', float('nan'))*100:.1f}%",
        f"{ref_metrics.get('clash_rate', float('nan'))*100:.1f}%",
        f"{ref_metrics.get('rg_mean', float('nan')):.2f}",
        f"{ref_metrics.get('ete_mean', float('nan')):.2f}",
        f"{ref_metrics.get('diversity', float('nan')):.2f}",
    ])

    tbl = ax.table(cellText=table_data, colLabels=rows_hdr,
                   cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor('#D0D0D0')

    _save_or_show(fig, save_path)


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 2: SE(3) EQUIVARIANCE
# ─────────────────────────────────────────────────────────────────────────────

def plot_equivariance(
    results_dict: dict,   # {label: equivariance_results_dict}
    save_path: Optional[str] = None,
):
    _check_mpl()

    from .equivariance import SCORE_THRESHOLD, PIPELINE_THRESHOLD, ISOTROPY_THRESHOLD, ANISO_THRESHOLD

    labels   = list(results_dict.keys())
    n_models = len(labels)

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle("SE(3) Equivariance Analysis", fontsize=13)

    x_pos = np.arange(2)
    bar_w = 0.8 / n_models

    # ── Panels 0 & 1: Test 1 and Test 2 error bars ─────────────────────────
    for ax_idx, (test_key, threshold, title) in enumerate([
        ('test1', SCORE_THRESHOLD,    'Test 1: Score-Network\nEquivariance Error'),
        ('test2', PIPELINE_THRESHOLD, 'Test 2: Full-Pipeline\nEquivariance Error'),
    ]):
        ax = axes[ax_idx]
        for m_idx, lbl in enumerate(labels):
            res = results_dict[lbl][test_key]
            if res.get('skipped'):
                ax.text(m_idx * bar_w, 0.5, f'{lbl}\n(N/A)', fontsize=7, ha='center')
                continue
            prop  = np.array(res['proper'])
            refl  = np.array(res['reflection'])
            means = [prop.mean() if len(prop) > 0 else float('nan'),
                     refl.mean() if len(refl) > 0 else float('nan')]
            stds  = [prop.std()  if len(prop) > 0 else 0,
                     refl.std()  if len(refl) > 0 else 0]
            offsets = x_pos + (m_idx - n_models / 2.0 + 0.5) * bar_w
            ax.bar(offsets, means, bar_w * 0.9, yerr=stds,
                   color=_color(m_idx), alpha=0.8, label=lbl, capsize=3)

        ax.set_yscale('log')
        ax.set_xticks(x_pos); ax.set_xticklabels(['Proper\nrotations', 'Reflections'], fontsize=9)
        ax.set_ylabel('Relative error (log scale)', fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.axhline(threshold, color='k', linestyle='--', lw=1.5, alpha=0.5,
                   label=f'Pass threshold ({threshold})')
        ax.legend(fontsize=7); ax.grid(alpha=0.3, axis='y')

    # ── Panel 2: Isotropy ratio ─────────────────────────────────────────────
    ax = axes[2]
    ratios = [results_dict[lbl]['test3']['ratio'] for lbl in labels]
    ax.barh(labels, ratios, color=[_color(i) for i in range(n_models)], alpha=0.8)
    ax.axvline(ISOTROPY_THRESHOLD, color='green', lw=1.5, linestyle='--',
               label=f'Isotropic < {ISOTROPY_THRESHOLD}')
    ax.axvline(ANISO_THRESHOLD, color='red', lw=1.5, linestyle='--',
               label=f'Anisotropic > {ANISO_THRESHOLD}')
    ax.set_xlabel('λ_max / λ_min', fontsize=9)
    ax.set_title('Test 3: Distribution\nIsotropy Ratio', fontsize=10)
    ax.legend(fontsize=8); ax.grid(alpha=0.3, axis='x')

    # ── Panel 3: Ensemble Rg Wasserstein-1 ─────────────────────────────────
    ax = axes[3]
    w1_vals = []
    for lbl in labels:
        t4 = results_dict[lbl].get('test4', {})
        w1 = t4.get('wasserstein1_rg', float('nan'))
        w1_vals.append(w1 if not t4.get('skipped', False) else float('nan'))

    valid_w1 = [v for v in w1_vals if not np.isnan(v)]
    y_max = max(valid_w1) * 1.3 if valid_w1 else 1.0

    ax.bar(labels, w1_vals, color=[_color(i) for i in range(n_models)], alpha=0.8)
    ax.axhline(0.0, color='green', lw=1.5, linestyle='--', label='Perfect equivariance (0)')
    ax.set_ylabel("Wasserstein-1 on Rg (Å)")
    ax.set_title("Test 4: Ensemble\nEquivariance (Rg W1)", fontsize=10)
    ax.set_ylim(0, y_max)
    ax.legend(fontsize=8); ax.grid(alpha=0.3, axis='y')
    ax.tick_params(axis='x', rotation=20, labelsize=8)

    plt.tight_layout()
    _save_or_show(fig, save_path)


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 3: ENERGY / CONFORMATIONAL SAMPLING
# ─────────────────────────────────────────────────────────────────────────────

def plot_energy_analysis(
    energy_results: dict,   # output of energy_analysis.run_energy_analysis()
    ref_coords: np.ndarray,
    model_label: str = "Model",
    save_path: Optional[str] = None,
):
    _check_mpl()

    if energy_results.get('skipped'):
        print("  [plot_energy_analysis] Skipped (non-energy model).")
        return

    sweep   = energy_results['sweep_results']    # [(tau, metrics)]
    samples = energy_results['all_samples']       # {tau: coords}
    strata  = energy_results['ref_strata']
    mono    = energy_results['monotonicity']
    mmd_q   = energy_results.get('mmd_vs_quartiles', {})

    taus    = sorted(set(t for t, m in sweep))
    n_tau   = len(taus)

    cmap    = LinearSegmentedColormap.from_list('temp',
                  ['#2166ac', '#92c5de', '#f7f7f7', '#f4a582', '#d6604d'])
    colors  = [cmap(i / max(n_tau - 1, 1)) for i in range(n_tau)]

    fig = plt.figure(figsize=(16, 14))
    gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.50, wspace=0.35)
    axes = [fig.add_subplot(gs[r, c]) for r in range(3) for c in range(2)]

    fig.suptitle(f"Energy Conditioning Analysis — {model_label}", fontsize=13, y=0.99)

    ref_ca = ca_from_coords(ref_coords)

    # ── Panel 0: Rg distribution per τ ─────────────────────────────────────
    ax = axes[0]
    ref_rg = radius_of_gyration(ref_ca)
    ax.hist(ref_rg, bins=40, density=True, color=REF_COLOR, alpha=REF_ALPHA,
            label='Reference', zorder=0)
    for i, tau in enumerate(taus):
        if tau not in samples: continue
        rg = radius_of_gyration(ca_from_coords(samples[tau]))
        ax.hist(rg, bins=40, density=True, color=colors[i], alpha=0.65, label=f"τ={tau:.2f}")
    ax.set_xlabel("Rg (Å)"); ax.set_title("Radius of Gyration per τ", fontsize=10)
    ax.legend(fontsize=7); ax.grid(alpha=0.3)

    # ── Panel 1: ETE distribution per τ ────────────────────────────────────
    ax = axes[1]
    ref_ete = end_to_end(ref_ca)
    ax.hist(ref_ete, bins=40, density=True, color=REF_COLOR, alpha=REF_ALPHA,
            label='Reference', zorder=0)
    for i, tau in enumerate(taus):
        if tau not in samples: continue
        ete = end_to_end(ca_from_coords(samples[tau]))
        ax.hist(ete, bins=40, density=True, color=colors[i], alpha=0.65, label=f"τ={tau:.2f}")
    ax.set_xlabel("End-to-end (Å)"); ax.set_title("End-to-End Distance per τ", fontsize=10)
    ax.legend(fontsize=7); ax.grid(alpha=0.3)

    # ── Panel 2: Rg mean ± std vs τ + quartile lines ───────────────────────
    ax = axes[2]
    tau_pts  = [t for t, m in sweep]
    rg_means = [m['rg_mean'] for t, m in sweep]
    rg_stds  = [m['rg_std']  for t, m in sweep]
    ax.errorbar(tau_pts, rg_means, yerr=rg_stds, fmt='o-', color='#d6604d',
                lw=2, ms=6, label='Generated')
    q_colors = ['#2166ac', '#74add1', '#f4a582', '#d73027']
    q_labels = ['Q1 (stable)', 'Q2', 'Q3', 'Q4 (transient)']
    for qi, (tau_lo, tau_hi, q_coords) in enumerate(strata[:4]):
        if len(q_coords) == 0: continue
        q_ca = ca_from_coords(q_coords)
        q_rg = radius_of_gyration(q_ca).mean()
        ax.axhline(q_rg, color=q_colors[qi], lw=1.5, linestyle='--',
                   alpha=0.7, label=q_labels[qi])
    rg_pass = mono.get('rg_monotone', {}).get('pass', '?')
    ax.set_xlabel("τ"); ax.set_ylabel("Rg (Å)")
    ax.set_title(f"Rg vs Temperature [{'✓' if rg_pass else '✗'}]", fontsize=10)
    ax.legend(fontsize=7); ax.grid(alpha=0.3)

    # ── Panel 3: ETE vs τ ───────────────────────────────────────────────────
    ax = axes[3]
    ete_means = [m['ete_mean'] for t, m in sweep]
    ete_stds  = [m['ete_std']  for t, m in sweep]
    ax.errorbar(tau_pts, ete_means, yerr=ete_stds, fmt='s-', color='#4393c3',
                lw=2, ms=6, label='Generated')
    for qi, (tau_lo, tau_hi, q_coords) in enumerate(strata[:4]):
        if len(q_coords) == 0: continue
        q_ca = ca_from_coords(q_coords)
        q_ete = end_to_end(q_ca).mean()
        ax.axhline(q_ete, color=q_colors[qi], lw=1.5, linestyle='--', alpha=0.7,
                   label=q_labels[qi])
    ete_pass = mono.get('ete_monotone', {}).get('pass', '?')
    ax.set_xlabel("τ"); ax.set_ylabel("ETE (Å)")
    ax.set_title(f"ETE vs Temperature [{'✓' if ete_pass else '✗'}]", fontsize=10)
    ax.legend(fontsize=7); ax.grid(alpha=0.3)

    # ── Panel 4: Bond validity vs τ (should be flat) + MMD vs quartile ─────
    ax = axes[4]
    bv_vals = [m['bond_valid'] * 100 for t, m in sweep]
    ax.plot(tau_pts, bv_vals, 'D-', color='#1a9641', lw=2, ms=6, label='Bond valid ±0.5 Å')
    ref_bv  = energy_results['ref_metrics']['bond_valid'] * 100
    ax.axhline(ref_bv, color='k', lw=1.5, linestyle='--', alpha=0.6,
               label=f'Reference ({ref_bv:.1f}%)')

    if mmd_q:
        ax2 = ax.twinx()
        mmd_matched  = [mmd_q[t]['mmd_matched']  for t in tau_pts if t in mmd_q]
        mmd_opposite = [mmd_q[t]['mmd_opposite'] for t in tau_pts if t in mmd_q]
        tau_mmd      = [t for t in tau_pts if t in mmd_q]
        ax2.plot(tau_mmd, mmd_matched,  'o--', color='#7b2d8b', lw=1.5, ms=4,
                 label='MMD vs matched quartile')
        ax2.plot(tau_mmd, mmd_opposite, 's--', color='#d95f02', lw=1.5, ms=4,
                 label='MMD vs opposite quartile')
        ax2.set_ylabel('MMD-RBF', fontsize=9, color='purple')
        ax2.legend(fontsize=7, loc='lower right')

    bv_pass = mono.get('bv_stable', {}).get('pass', '?')
    ax.set_xlabel("τ"); ax.set_ylabel("Bond validity (%)")
    ax.set_title(f"Bond Validity & MMD vs τ [{'✓' if bv_pass else '✗'}]", fontsize=10)
    ax.set_ylim(0, 105); ax.legend(fontsize=7, loc='upper left'); ax.grid(alpha=0.3)

    # ── Panel 5: PCA coloured by τ ──────────────────────────────────────────
    ax = axes[5]
    from .novelty import distance_fingerprint, pca_2d

    all_tau_coords = []
    tau_labels_pca = []
    for tau in taus:
        if tau not in samples: continue
        ca = ca_from_coords(samples[tau])
        all_tau_coords.append(ca)
        tau_labels_pca.extend([tau] * len(ca))

    if all_tau_coords:
        all_gen_ca  = np.concatenate(all_tau_coords, axis=0)
        gen_fp      = distance_fingerprint(all_gen_ca, max_atoms=10)
        ref_fp      = distance_fingerprint(ref_ca[:min(500, len(ref_ca))], max_atoms=10)
        gen_2d, ref_2d = pca_2d(gen_fp, ref_fp)

        ax.scatter(ref_2d[:, 0], ref_2d[:, 1], s=4, alpha=0.2, color='gray',
                   label='Reference', rasterized=True)
        tau_arr = np.array(tau_labels_pca)
        sc = ax.scatter(gen_2d[:, 0], gen_2d[:, 1], s=6, alpha=0.5, c=tau_arr,
                        cmap='RdBu_r', rasterized=True)
        plt.colorbar(sc, ax=ax, label='τ', fraction=0.03)
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
        ax.set_title("Conformational PCA by τ\n(blue=τ=0 stable, red=τ=1 transient)", fontsize=10)
        ax.legend(fontsize=7); ax.grid(alpha=0.2)
    else:
        ax.text(0.5, 0.5, 'No samples', transform=ax.transAxes, ha='center', va='center')

    _save_or_show(fig, save_path)


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 3b: τ vs ROSETTA ENERGY (energy conditioning force-field validation)
# ─────────────────────────────────────────────────────────────────────────────

def plot_tau_reu(
    tau_reu_dict: dict,         # {model_label: {str(tau): {mean, std, n, scores}}}
    save_path: Optional[str] = None,
):
    """
    Scatter + line plot: x = τ (generation temperature), y = mean Rosetta REU.

    A positive slope confirms that energy conditioning transfers to the force-field:
    τ=0 generates stable (low-energy) conformations, τ=1 generates transient (high-energy) ones.
    """
    _check_mpl()

    if not tau_reu_dict:
        print("  [plot_tau_reu] No τ-REU data — skipping figure")
        return

    try:
        from scipy import stats as sp_stats
        _has_scipy = True
    except ImportError:
        _has_scipy = False

    n_models = len(tau_reu_dict)
    cmap     = plt.cm.tab10
    colors   = [cmap(i % 10) for i in range(n_models)]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("τ vs Rosetta Energy (REU) — Energy Conditioning Validation", fontsize=13)

    ax_line   = axes[0]
    ax_violin = axes[1]

    for i, (lbl, tau_data) in enumerate(tau_reu_dict.items()):
        if not tau_data:
            continue
        col   = colors[i]
        taus  = sorted(float(t) for t in tau_data.keys())
        means = [tau_data[str(t)]['mean'] for t in taus]
        stds  = [tau_data[str(t)]['std']  for t in taus]

        # Jittered per-structure scatter behind the mean line
        for t, td in tau_data.items():
            jitter = np.random.uniform(-0.02, 0.02, len(td['scores']))
            ax_line.scatter(
                np.full(len(td['scores']), float(t)) + jitter,
                td['scores'],
                s=12, alpha=0.25, color=col, zorder=1,
            )

        ax_line.errorbar(
            taus, means, yerr=stds,
            fmt='o-', color=col, lw=2.5, ms=7, capsize=5,
            label=lbl, zorder=3,
        )

        if _has_scipy and len(taus) >= 3:
            r, p = sp_stats.spearmanr(taus, means)
            pass_str = "PASS" if r > 0.6 and p < 0.1 else "FAIL"
            ax_line.annotate(
                f"{lbl[:18]}: r={r:.2f} ({pass_str})",
                xy=(taus[-1], means[-1]),
                xytext=(5, 0), textcoords='offset points',
                fontsize=7.5, color=col, va='center',
            )

    ax_line.set_xlabel("τ (generation temperature)", fontsize=11)
    ax_line.set_ylabel("Rosetta REU (total_score)", fontsize=11)
    ax_line.set_title(
        "τ vs Mean REU\npositive slope → stable at low τ, transient at high τ",
        fontsize=10,
    )
    ax_line.legend(fontsize=8, loc='upper left')
    ax_line.grid(alpha=0.3)

    # Right panel: violin distributions at τ_min vs τ_max per model
    positions   = []
    tick_labels = []
    for i, (lbl, tau_data) in enumerate(tau_reu_dict.items()):
        if not tau_data:
            continue
        col        = colors[i]
        taus_avail = sorted(float(t) for t in tau_data.keys())
        tau_lo     = min(taus_avail)
        tau_hi     = max(taus_avail)

        for offset, tau_sel, alpha in [(0.0, tau_lo, 0.7), (0.45, tau_hi, 0.4)]:
            scores = tau_data.get(str(tau_sel), {}).get('scores', [])
            if not scores:
                continue
            pos = i * 1.2 + offset
            vp  = ax_violin.violinplot(scores, positions=[pos], widths=0.38, showmedians=True)
            for pc in vp['bodies']:
                pc.set_facecolor(col); pc.set_alpha(alpha); pc.set_edgecolor(col)
            for key in ('cmedians', 'cmins', 'cmaxes', 'cbars'):
                vp[key].set_color(col)
            positions.append(pos)
            tick_labels.append(f"{lbl[:12]}\nτ={tau_sel:.2f}")

    if positions:
        ax_violin.set_xticks(positions)
        ax_violin.set_xticklabels(tick_labels, fontsize=7, rotation=20)
        ax_violin.set_ylabel("Rosetta REU (total_score)", fontsize=11)
        ax_violin.set_title(
            "Score Distribution: τ_min (solid) vs τ_max (faded)\nlower = more stable",
            fontsize=10,
        )
        ax_violin.grid(alpha=0.3, axis='y')
    else:
        ax_violin.text(0.5, 0.5, 'No data', transform=ax_violin.transAxes,
                       ha='center', va='center', fontsize=12, color='gray')

    plt.tight_layout()
    _save_or_show(fig, save_path)


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 4: NOVELTY
# ─────────────────────────────────────────────────────────────────────────────

def plot_novelty(
    novelty_results: dict,    # output of novelty.run_novelty_analysis()
    per_tau_novelty: dict,    # output of novelty.per_temperature_novelty() (or {})
    ref_coords: np.ndarray,
    label: str = "Model",
    save_path: Optional[str] = None,
):
    _check_mpl()

    gen_fp   = novelty_results.get('_gen_fp')
    ref_fp   = novelty_results.get('_ref_fp')
    gen_nnd  = novelty_results.get('_gen_nnd')
    self_nnd = novelty_results.get('_self_nnd')
    gen_valid = novelty_results.get('_gen_valid')

    fig = plt.figure(figsize=(18, 10))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)
    axes = [fig.add_subplot(gs[r, c]) for r in range(2) for c in range(3)]

    fig.suptitle(f"Conformational Novelty — {label}", fontsize=13)
    thr = novelty_results.get('threshold', 1.0)

    # ── Panel 0: NND distribution ────────────────────────────────────────────
    ax = axes[0]
    if gen_nnd is not None and self_nnd is not None:
        hi = max(gen_nnd.max(), self_nnd.max()) * 1.1
        bins = np.linspace(0, hi, 50)
        ax.hist(self_nnd, bins=bins, density=True, color='#333333', alpha=0.4,
                label='Test→Test (baseline)')
        ax.hist(gen_nnd, bins=bins, density=True, color='#C44E52', alpha=0.7,
                label='Generated→Test (filtered)')
        ax.axvline(thr, color='k', lw=1.5, linestyle='--', label=f'Threshold ({thr:.2f})')
        nnd_r = novelty_results.get('nnd_ratio', float('nan'))
        ax.text(0.02, 0.97, f"NND ratio: {nnd_r:.3f}", transform=ax.transAxes,
                fontsize=9, va='top', bbox=dict(facecolor='white', alpha=0.7))
    else:
        ax.text(0.5, 0.5, 'No data', transform=ax.transAxes, ha='center', va='center')
    ax.set_xlabel("Nearest-neighbour distance"); ax.set_ylabel("Density")
    ax.set_title("NND Distribution\n(left=memorised, right=novel)", fontsize=10)
    if ax.get_legend_handles_labels()[0]: ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    # ── Panel 1: PCA scatter ─────────────────────────────────────────────────
    ax = axes[1]
    if gen_fp is not None and ref_fp is not None and len(gen_fp) > 0:
        from .novelty import pca_2d
        gen_2d, ref_2d = pca_2d(gen_fp, ref_fp)
        n_ref_show = min(2000, len(ref_2d))
        ax.scatter(ref_2d[:n_ref_show, 0], ref_2d[:n_ref_show, 1],
                   s=4, alpha=0.2, color='#333333', label='Test set', rasterized=True)
        ax.scatter(gen_2d[:, 0], gen_2d[:, 1],
                   s=6, alpha=0.5, color='#C44E52', label='Generated (valid)', rasterized=True)
    else:
        ax.text(0.5, 0.5, 'No valid structures\n(physics filter: 0 pass)',
                transform=ax.transAxes, ha='center', va='center', fontsize=10, color='gray')
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    ax.set_title("PCA of Conformational Space\n(overlap = covers test distribution)", fontsize=10)
    if ax.get_legend_handles_labels()[0]: ax.legend(fontsize=7, markerscale=3)
    ax.grid(alpha=0.2)

    # ── Panel 2: Coverage / Precision vs threshold ──────────────────────────
    ax = axes[2]
    if gen_fp is not None and ref_fp is not None and len(gen_fp) > 0 and not np.isnan(thr):
        from .novelty import coverage_precision
        thresholds = np.linspace(thr * 0.1, thr * 3.0, 30)
        covs, precs = [], []
        for t in thresholds:
            c, p = coverage_precision(gen_fp, ref_fp, t)
            covs.append(c); precs.append(p)
        ax.plot(thresholds, [c * 100 for c in covs],  color='#4C72B0', lw=2, label='Coverage')
        ax.plot(thresholds, [p * 100 for p in precs], color='#55A868', lw=2, label='Precision')
        ax.axvline(thr, color='k', lw=1, linestyle='--', label=f'Ref threshold ({thr:.2f})')
    else:
        ax.text(0.5, 0.5, 'No valid structures', transform=ax.transAxes,
                ha='center', va='center', fontsize=10, color='gray')
    ax.set_xlabel("Distance threshold"); ax.set_ylabel("%")
    ax.set_title("Coverage & Precision\nvs. Distance Threshold", fontsize=10)
    ax.set_ylim(0, 105)
    if ax.get_legend_handles_labels()[0]: ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    # ── Panel 3: Per-τ NND ratio (energy model only) ─────────────────────────
    ax = axes[3]
    if per_tau_novelty:
        taus_plot = sorted(per_tau_novelty.keys())
        ratios    = [per_tau_novelty[t].get('nnd_ratio', float('nan')) for t in taus_plot]
        valid_frs = [per_tau_novelty[t].get('valid_fraction', float('nan'))*100 for t in taus_plot]

        ax.plot(taus_plot, ratios, 'o-', color='#C44E52', lw=2, ms=7, label='NND ratio')
        ax.axhline(1.0, color='k', lw=1.5, linestyle='--', alpha=0.5, label='Ratio=1 (matches dist.)')
        ax2 = ax.twinx()
        ax2.plot(taus_plot, valid_frs, 's--', color='#4C72B0', lw=1.5, ms=5,
                 label='Valid fraction (%)')
        ax2.set_ylabel("Valid structures (%)", fontsize=9)
        ax2.legend(fontsize=7, loc='lower right')
        ax.set_xlabel("τ"); ax.set_ylabel("NND ratio")
        ax.set_title("NND Ratio per Temperature\n(>1 = more novel)", fontsize=10)
        ax.legend(fontsize=7, loc='upper left'); ax.grid(alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'Per-τ novelty not available\n(non-energy model)',
                transform=ax.transAxes, ha='center', va='center', fontsize=10, color='gray')
        ax.set_title("NND Ratio per Temperature", fontsize=10)

    # ── Panel 4: RMSD pairwise heatmap ──────────────────────────────────────
    ax = axes[4]
    rmsd_mat = novelty_results.get('rmsd_matrix')
    if rmsd_mat is not None and len(rmsd_mat) > 0:
        mat = np.array(rmsd_mat)
        im  = ax.imshow(mat, cmap='viridis', aspect='auto')
        plt.colorbar(im, ax=ax, label='RMSD (Å)', fraction=0.04)
        ax.set_title(f"Pairwise RMSD Heatmap\n({mat.shape[0]} generated structures)", fontsize=10)
        ax.set_xlabel("Structure index"); ax.set_ylabel("Structure index")
        ax.text(0.02, 0.02, f"Mean RMSD: {mat[np.triu_indices_from(mat, k=1)].mean():.2f} Å",
                transform=ax.transAxes, fontsize=8,
                bbox=dict(facecolor='white', alpha=0.8))
    else:
        ax.text(0.5, 0.5, 'RMSD matrix not computed', transform=ax.transAxes,
                ha='center', va='center', fontsize=10, color='gray')

    # ── Panel 5: Physics filter summary bar ─────────────────────────────────
    ax = axes[5]
    ax.axis('off')
    nv  = novelty_results.get('n_valid', 0)
    ntot = novelty_results.get('n_total', nv)
    nnd_ratio = novelty_results.get('nnd_ratio', float('nan'))
    cov = novelty_results.get('coverage', float('nan'))
    prec = novelty_results.get('precision', float('nan'))

    summary_text = (
        f"Novelty Summary — {label}\n\n"
        f"Structures generated:  {ntot}\n"
        f"Passed physics filter: {nv} ({nv/max(ntot,1)*100:.1f}%)\n\n"
        f"NND ratio (gen/self):  {nnd_ratio:.3f}\n"
        f"  {'→ Novel (>1.1)' if nnd_ratio > 1.1 else ('→ Memorised (<0.9)' if nnd_ratio < 0.9 else '→ Matches distribution')}\n\n"
        f"Coverage:   {cov*100:.1f}%\n"
        f"Precision:  {prec*100:.1f}%\n"
    )
    ax.text(0.1, 0.9, summary_text, transform=ax.transAxes,
            fontsize=10, va='top', family='monospace',
            bbox=dict(facecolor='#f8f8f8', alpha=0.8, edgecolor='#cccccc'))

    _save_or_show(fig, save_path)


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 5: PYROSETTA VALIDATION
# ─────────────────────────────────────────────────────────────────────────────

def plot_pyrosetta(
    rosetta_results: dict,        # output of pyrosetta_utils.pyrosetta_validate()
    ref_rosetta_results: dict,    # same but for reference (or None)
    model_label: str = "Model",
    tau: Optional[float] = None,
    save_path: Optional[str] = None,
):
    _check_mpl()

    if rosetta_results.get('skipped'):
        print(f"  [plot_pyrosetta] Skipped: {rosetta_results.get('reason', '')}")
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f"PyRosetta Validation — {model_label}", fontsize=13)
    axes = axes.flatten()

    per_struct    = rosetta_results.get('per_structure', [])
    ref_structs   = (ref_rosetta_results or {}).get('per_structure', [])

    def _vals(structs, key):
        return [s[key] for s in structs if s and not np.isnan(s.get(key, float('nan')))]

    for ax_idx, (key, title) in enumerate([
        ('total_score',  'Total Rosetta Score (REF2015)'),
        ('fa_rep',       'fa_rep (VdW Clash Term)'),
        ('rama_prepro',  'Ramachandran Score'),
        ('fa_dun',       'Rotamer Score (fa_dun)'),
    ]):
        ax = axes[ax_idx]
        gen_vals = _vals(per_struct, key)
        ref_vals = _vals(ref_structs, key)

        if gen_vals:
            ax.hist(gen_vals, bins=20, density=True, color='#C44E52', alpha=0.7,
                    label=f'Generated (n={len(gen_vals)})')
        if ref_vals:
            ax.hist(ref_vals, bins=20, density=True, color=REF_COLOR, alpha=REF_ALPHA,
                    label=f'Reference (n={len(ref_vals)})')

        # Threshold line
        from .constants import ROSETTA_FA_REP_MAX, ROSETTA_RAMA_MAX
        if key == 'fa_rep':
            ax.axvline(ROSETTA_FA_REP_MAX, color='red', lw=1.5, linestyle='--',
                       label=f'Max ({ROSETTA_FA_REP_MAX})')
        elif key == 'rama_prepro':
            ax.axvline(ROSETTA_RAMA_MAX, color='red', lw=1.5, linestyle='--',
                       label=f'Max ({ROSETTA_RAMA_MAX})')
        elif key == 'total_score':
            thr = rosetta_results.get('total_threshold')
            if thr and not np.isnan(thr):
                ax.axvline(thr, color='red', lw=1.5, linestyle='--',
                           label=f'Threshold ({thr:.1f})')

        ax.set_xlabel(key); ax.set_title(title, fontsize=10)
        ax.legend(fontsize=7); ax.grid(alpha=0.3)

    plt.tight_layout()
    _save_or_show(fig, save_path)
