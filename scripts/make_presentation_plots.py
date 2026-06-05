"""
scripts/make_presentation_plots.py
===================================
Creates four individual presentation-quality plots (SVG + PNG each) for the
four no-physics Cα models:
  - Transformer-DDPM-v3  (baseline)
  - AdaLN-Transformer-v1
  - EGNN-DDPM-v1
  - FlowMatch-EGNN-v2

Plots produced (each saved as both .svg and .png in plots/presentation/):
  1. per_bond_rmse.svg/png   — per-residue-pair RMSE from ideal 3.832 Å
  2. clash_rate.svg/png      — clash rate bar chart with reference line
  3. isotropy.svg/png        — equivariance isotropy ratio (Test 3 only)
  4. novelty_nnd.svg/png     — NND distribution + ratio bar inset

Data sources:
  - plots/eval_v2/nophysics_clash/metrics.json  (bond RMSE, clash rate)
  - logs/equivariance_nophysics_pres.log         (isotropy ratios)
  - plots/presentation/novelty_*.json            (NND distributions)

Usage:
    python scripts/make_presentation_plots.py
"""

import json, re, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path

# ── Presentation style ────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':       'sans-serif',
    'font.size':         13,
    'axes.titlesize':    15,
    'axes.labelsize':    13,
    'xtick.labelsize':   11,
    'ytick.labelsize':   11,
    'legend.fontsize':   11,
    'figure.dpi':        150,
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'axes.grid':         False,
})

OUT  = Path('plots/presentation')
OUT.mkdir(parents=True, exist_ok=True)

# ── Model metadata ────────────────────────────────────────────────────────────
MODELS = [
    ('AdaLN-Transformer-v1', '#5B8DB8'),   # blue
    ('EGNN-DDPM-v1',         '#6BAF6B'),   # green
    ('FlowMatch-EGNN-v2',    '#9B6BB5'),   # purple
]
LABELS  = [m[0] for m in MODELS]
COLORS  = [m[1] for m in MODELS]
REF_COL = '#333333'

# Short display names for x-tick / legend
DISPLAY = {
    'Transformer-DDPM-v3':  'Transformer',
    'AdaLN-Transformer-v1': 'AdaLN-Transformer\n(+ Data Augm.)',
    'EGNN-DDPM-v1':         'EGNN',
    'FlowMatch-EGNN-v2':    'FlowMatch-EGNN',
}


def save(fig, name):
    for ext in ('png', 'svg'):
        p = OUT / f"{name}.{ext}"
        fig.savefig(p, dpi=150, bbox_inches='tight',
                    format=ext, transparent=(ext == 'svg'))
    print(f"  Saved {name}.png + .svg")
    plt.close(fig)


# ═════════════════════════════════════════════════════════════════════════════
# Load data
# ═════════════════════════════════════════════════════════════════════════════

# ── Bond RMSE + clash from eval_v2 ───────────────────────────────────────────
with open('plots/eval_v2/nophysics_clash/metrics.json') as f:
    ev2 = json.load(f)['per_model']

# Map eval_v2 label → our standard label
LABEL_MAP = {
    'Baseline-Transformer': 'Transformer-DDPM-v3',
    'AdaLN-Transformer':    'AdaLN-Transformer-v1',
    'EGNN':                 'EGNN-DDPM-v1',
    'FlowMatch-EGNN':       'FlowMatch-EGNN-v2',
}
bond_data  = {}   # label → per_bond_rmse list (9 values)
clash_data = {}   # label → clash_rate float

for ev2_label, std_label in LABEL_MAP.items():
    m = ev2[ev2_label]['native_metrics']
    bond_data[std_label]  = m['per_bond_rmse']
    clash_data[std_label] = m['clash_rate'] * 100   # as %

REF_PER_BOND = [0.0638] * 9   # reference bond RMSE (same for all bonds)

# ── Equivariance isotropy from log ────────────────────────────────────────────
iso_data = {}   # label → ratio float
try:
    log_text = Path('logs/equivariance_nophysics_pres.log').read_text()
    # Pattern: "  LABEL  [model_type, ...]" then "ratio=X.XXX"
    current = None
    for line in log_text.splitlines():
        # Match model header line
        for lbl in LABELS:
            if lbl in line and '[' in line:
                current = lbl
                break
        # Match isotropy ratio line
        m = re.search(r'ratio=([\d.]+)', line)
        if m and current:
            iso_data[current] = float(m.group(1))
            current = None
except Exception as e:
    print(f"  Warning: could not parse isotropy log: {e}")

# ── Novelty NND distributions ─────────────────────────────────────────────────
nnd_gen   = {}   # label → array of NND values (generated→test)
nnd_self  = {}   # label → array of NND values (test→test)
nnd_ratio = {}   # label → float

for lbl in LABELS:
    p = OUT / f"novelty_{lbl}.json"
    if p.exists():
        d = json.loads(p.read_text())
        nnd_ratio[lbl] = d['nnd_ratio']
    # Load raw arrays from the novelty script's output
    # (stored in per-model log files via tee)
    log_p = Path(f"logs/novelty_pres_{lbl}.log")
    if log_p.exists():
        txt = log_p.read_text()
        m1 = re.search(r'mean\s*=\s*([\d.]+)', txt)
        m2 = re.search(r'Within-test NND.*?mean\s*=\s*([\d.]+)', txt, re.DOTALL)
        if m1:
            nnd_gen[lbl]  = float(m1.group(1))
        if m2:
            nnd_self[lbl] = float(m2.group(1))


# ═════════════════════════════════════════════════════════════════════════════
# Plot 1 — Per-bond RMSE
# ═════════════════════════════════════════════════════════════════════════════

print("Plotting per_bond_rmse …")
fig, ax = plt.subplots(figsize=(8, 5))

bond_positions = [f"{i+1}–{i+2}" for i in range(9)]
x = np.arange(9)

for lbl, col in zip(LABELS, COLORS):
    vals = bond_data.get(lbl, [0]*9)
    ax.plot(x, vals, 'o-', color=col, lw=2, markersize=6,
            label=DISPLAY[lbl], alpha=0.9)

ax.plot(x, REF_PER_BOND, 'o--', color=REF_COL, lw=1.5, markersize=5,
        alpha=0.6, label='Reference (MD)')

ax.set_xticks(x)
ax.set_xticklabels(bond_positions, fontsize=10)
ax.set_xlabel("Cα–Cα bond (residue pair)")
ax.set_ylabel("RMSE from ideal 3.832 Å  (Å)")
ax.set_title("Per-Bond RMSE — no-physics models")
ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1), borderaxespad=0)
ax.set_ylim(bottom=0)
fig.tight_layout(rect=[0, 0, 0.82, 1])

save(fig, 'per_bond_rmse')


# ═════════════════════════════════════════════════════════════════════════════
# Plot 2 — Clash rate
# ═════════════════════════════════════════════════════════════════════════════

print("Plotting clash_rate …")
fig, ax = plt.subplots(figsize=(7, 5))

rates = [clash_data.get(lbl, 0) for lbl in LABELS]
disp  = [DISPLAY[lbl] for lbl in LABELS]

bars = ax.bar(disp, rates, color=COLORS, alpha=0.85, width=0.55, zorder=3)

# Value labels on bars
for bar, val in zip(bars, rates):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.6,
            f"{val:.1f}%", ha='center', va='bottom', fontsize=11, fontweight='bold')

ax.set_ylabel("Clash rate  (%)")
ax.set_title("Clash Rate — no-physics models\n(non-bonded Cα pairs < 3.5 Å)")
ax.set_ylim(0, max(rates) * 1.25)
ax.tick_params(axis='x', labelsize=10)

save(fig, 'clash_rate')


# ═════════════════════════════════════════════════════════════════════════════
# Plot 3 — Equivariance isotropy ratio (Test 3 only)
# ═════════════════════════════════════════════════════════════════════════════

print("Plotting isotropy …")
fig, ax = plt.subplots(figsize=(7, 5))

if iso_data:
    ratios = [iso_data.get(lbl, None) for lbl in LABELS]
    disp   = [DISPLAY[lbl] for lbl in LABELS]
    valid  = [(d, r, c) for d, r, c in zip(disp, ratios, COLORS) if r is not None]
    disp_v, ratios_v, colors_v = zip(*valid) if valid else ([], [], [])

    bars = ax.bar(disp_v, ratios_v, color=colors_v, alpha=0.85, width=0.55, zorder=3)

    # Single threshold: 1.1 = practical equivariance pass (equivariant models sit at 1.07–1.10)
    ax.axhline(1.1, color='#27AE60', lw=1.8, linestyle='--', alpha=0.8,
               label='Equivariant threshold  (1.1)')

    for bar, val in zip(bars, ratios_v):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{val:.2f}", ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.set_ylabel(r"Isotropy ratio  $\lambda_{\max} / \lambda_{\min}$")
    ax.set_title("SE(3) Equivariance — Distribution Isotropy\n(Test 3: eigenvalue ratio of positional covariance)")
    ax.set_ylim(0, max(ratios_v) * 1.3)
    ax.tick_params(axis='x', labelsize=10)
    ax.legend(loc='upper left')
else:
    ax.text(0.5, 0.5, "Isotropy data not yet available\n(equivariance check still running)",
            ha='center', va='center', transform=ax.transAxes, fontsize=13)
    ax.set_title("Equivariance isotropy (pending)")

save(fig, 'isotropy')


# ═════════════════════════════════════════════════════════════════════════════
# Plot 4 — Novelty NND distribution + ratio bar inset
# ═════════════════════════════════════════════════════════════════════════════

print("Plotting novelty_nnd …")
fig, ax = plt.subplots(figsize=(7, 5))

# Reload JSON files to pick up physics-filtered results
valid_frac  = {}
n_used_dict = {}
for lbl in LABELS:
    p = OUT / f"novelty_{lbl}.json"
    if p.exists():
        d = json.loads(p.read_text())
        nnd_ratio[lbl]   = d['nnd_ratio']
        valid_frac[lbl]  = d.get('valid_fraction', None)
        n_used_dict[lbl] = d.get('n_used_for_nnd', None)

if nnd_ratio:
    vlabels    = [lbl for lbl in LABELS if lbl in nnd_ratio]
    disp_nnd   = [DISPLAY[lbl] for lbl in vlabels]
    ratios_nnd = [nnd_ratio[lbl] for lbl in vlabels]
    colors_nnd = [COLORS[LABELS.index(lbl)] for lbl in vlabels]

    bars = ax.bar(disp_nnd, ratios_nnd, color=colors_nnd, alpha=0.85, width=0.55, zorder=3)

    # Reference line at ratio = 1.0
    ax.axhline(1.0, color=REF_COL, lw=1.8, linestyle='--', alpha=0.7,
               label='Test→Test baseline (ratio = 1.0)')

    # Ratio value on top of each bar
    for bar, ratio in zip(bars, ratios_nnd):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{ratio:.2f}", ha='center', va='bottom', fontsize=12, fontweight='bold')


    ax.set_ylabel("NND ratio  (Generated→Test / Test→Test)")
    ax.set_title("Conformational Novelty — NND Ratio\n(physically valid structures only)")
    ax.set_ylim(0, max(ratios_nnd) * 1.25)
    ax.tick_params(axis='x', labelsize=10)
else:
    ax.text(0.5, 0.5, "Novelty NND data not yet available",
            ha='center', va='center', transform=ax.transAxes, fontsize=13)
    ax.set_title("Novelty NND (pending)")

save(fig, 'novelty_nnd')

print(f"\nAll plots saved to {OUT}/")
print("Files: per_bond_rmse.{{png,svg}}, clash_rate.{{png,svg}}, "
      "isotropy.{{png,svg}}, novelty_nnd.{{png,svg}}")
