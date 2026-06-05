"""
scripts/plot_rg_vs_tau.py
=========================
Single-panel Rg vs temperature plot for presentation.
Reads a pre-computed energy_analysis.json.

Usage:
    python scripts/plot_rg_vs_tau.py \
        --json plots/energy_analysis_adaln_ep.json \
        --out  plots/rg_vs_tau
    # produces plots/rg_vs_tau.png and plots/rg_vs_tau.svg
"""

import json
import argparse
import numpy as np
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--json', required=True)
    p.add_argument('--out',  required=True, help='Output path without extension')
    p.add_argument('--guidance_scale', type=float, default=None,
                   help='Filter to this guidance scale (default: first found)')
    p.add_argument('--title', default=None)
    args = p.parse_args()

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    with open(args.json) as f:
        data = json.load(f)

    # Pick guidance scale
    w0 = args.guidance_scale or data['sweep'][0][1]
    pts = [(tau, m) for tau, w, m in data['sweep'] if w == w0]
    pts.sort(key=lambda x: x[0])

    taus     = [tau for tau, _ in pts]
    rg_means = [m['rg_mean'] for _, m in pts]
    rg_stds  = [m['rg_std']  for _, m in pts]
    ref_rg   = data['reference']['rg_mean']
    ref_std  = data['reference']['rg_std']

    # Reference quartile targets from dataset
    ref_q1_rg = 4.97   # compact/folded
    ref_q4_rg = 6.96   # extended/transient

    fig, ax = plt.subplots(figsize=(6, 4))

    # Generated sweep
    ax.errorbar(taus, rg_means, yerr=rg_stds,
                fmt='o-', color='#C44E52', lw=2, ms=7, capsize=4,
                label=f'Generated (w={w0})', zorder=3)

    # Reference lines
    ax.axhline(ref_rg, color='#222222', linestyle='--', lw=1.5, alpha=0.6,
               label=f'Test set mean ({ref_rg:.2f} Å)')
    ax.axhline(ref_q1_rg, color='#4C72B0', linestyle=':', lw=1.5, alpha=0.8,
               label=f'Q1 compact ({ref_q1_rg:.2f} Å)')
    ax.axhline(ref_q4_rg, color='#DD8452', linestyle=':', lw=1.5, alpha=0.8,
               label=f'Q4 extended ({ref_q4_rg:.2f} Å)')

    ax.set_xlabel('Temperature τ', fontsize=13)
    ax.set_ylabel('Radius of gyration (Å)', fontsize=13)
    ax.set_xticks(taus)
    ax.set_xlim(-0.05, 1.05)
    ax.tick_params(labelsize=11)
    ax.legend(fontsize=9, framealpha=0.9)
    ax.grid(axis='y', alpha=0.3)

    if args.title:
        ax.set_title(args.title, fontsize=13)

    fig.tight_layout()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f'{out}.png', dpi=150, bbox_inches='tight')
    fig.savefig(f'{out}.svg', bbox_inches='tight')
    print(f'Saved {out}.png and {out}.svg')
    plt.close(fig)


if __name__ == '__main__':
    main()
