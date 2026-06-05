"""
scripts/plot_novelty.py
=======================
Single-panel novelty plot: validity vs NND ratio scatter for selected models,
with NND distribution inset for the primary model.

Usage:
    python scripts/plot_novelty.py \
        --json plots/eval_overnight/ca_only/metrics.json \
        --out  final_plots/novelty
"""

import json
import argparse
import numpy as np
from pathlib import Path


# Models to highlight — (label_in_json, display_label)
HIGHLIGHT = [
    ('AdaLN-Transformer-Energy-Physics-v1', 'AdaLN+E+Physics'),
    ('AdaLN-Transformer-SelfCond-v3',       'AdaLN+SC'),
    ('TorsionFlow-MLP-v1',                  'TorsionFlow'),
    ('TorsionTransformer-v1',               'TorsionTransformer'),
    ('AdaLN-Transformer-v1',               'AdaLN'),
    ('AdaLN-Transformer-Energy-v1',        'AdaLN+Energy'),
    ('FlowMatch-Energy-Physics-v1',        'FlowMatch+E+Physics'),
    ('EGNN-AdaLN-Energy-v1',              'EGNN+AdaLN+Energy'),
]

COLORS = {
    'AdaLN+E+Physics':     '#C44E52',
    'AdaLN+SC':            '#4C72B0',
    'TorsionFlow':         '#55A868',
    'TorsionTransformer':  '#8172B3',
    'AdaLN':               '#937860',
    'AdaLN+Energy':        '#DA8BC3',
    'FlowMatch+E+Physics': '#CCB974',
    'EGNN+AdaLN+Energy':   '#64B5CD',
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--json', default='plots/eval_overnight/ca_only/metrics.json')
    p.add_argument('--out',  required=True)
    p.add_argument('--primary', default='AdaLN-Transformer-Energy-Physics-v1',
                   help='Model whose NND distribution to show in inset')
    args = p.parse_args()

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    with open(args.json) as f:
        data = json.load(f)
    nov = data['sections']['novelty']

    fig, ax = plt.subplots(figsize=(6, 4.5))

    label_map = dict(HIGHLIGHT)

    # ── Background: all models (grey) ────────────────────────────────────────
    for key, v in nov.items():
        if v.get('nnd_ratio') is None:
            continue
        if key not in label_map:
            ax.scatter(v['nnd_ratio'], v['valid_fraction'] * 100,
                       c='#cccccc', s=30, alpha=0.5, linewidths=0, zorder=1)

    # ── Highlighted models ────────────────────────────────────────────────────
    for key, display in HIGHLIGHT:
        if key not in nov:
            continue
        v = nov[key]
        if v.get('nnd_ratio') is None:
            continue
        color = COLORS.get(display, '#333333')
        ax.scatter(v['nnd_ratio'], v['valid_fraction'] * 100,
                   c=color, s=80, alpha=0.9, linewidths=0.5,
                   edgecolors='white', zorder=3, label=display)

    # ── Reference line at NND ratio = 1 ──────────────────────────────────────
    ax.axvline(1.0, color='#333333', linestyle='--', lw=1.2, alpha=0.6,
               label='NND ratio = 1 (matches training dist.)')

    # ── Annotations ──────────────────────────────────────────────────────────
    offsets = {
        'AdaLN+E+Physics':     ( 0.01, -4),
        'AdaLN+SC':            ( 0.01,  1.5),
        'TorsionFlow':         ( 0.01, -4),
        'TorsionTransformer':  ( 0.01,  1.5),
        'FlowMatch+E+Physics': ( 0.01,  1.5),
        'EGNN+AdaLN+Energy':   ( 0.01, -4),
    }
    for key, display in HIGHLIGHT:
        if key not in nov:
            continue
        v = nov[key]
        if v.get('nnd_ratio') is None or display not in offsets:
            continue
        dx, dy = offsets[display]
        ax.annotate(display,
                    xy=(v['nnd_ratio'], v['valid_fraction'] * 100),
                    xytext=(v['nnd_ratio'] + dx, v['valid_fraction'] * 100 + dy),
                    fontsize=7.5, color=COLORS.get(display, '#333333'),
                    arrowprops=dict(arrowstyle='-', color='#aaaaaa', lw=0.8))

    ax.set_xlabel('NND ratio  (generated / test self-similarity)\n'
                  '← memorised   |   novel →', fontsize=11)
    ax.set_ylabel('Bond validity ±0.5 Å (%)', fontsize=11)
    ax.set_xlim(0.75, 1.7)
    ax.set_ylim(0, 105)
    ax.tick_params(labelsize=10)
    ax.legend(fontsize=7.5, framealpha=0.9, loc='upper right',
              markerscale=1.2, ncol=1)
    ax.grid(alpha=0.2)

    fig.tight_layout()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f'{out}.png', dpi=150, bbox_inches='tight')
    fig.savefig(f'{out}.svg', bbox_inches='tight')
    print(f'Saved {out}.png  {out}.svg')
    plt.close(fig)


if __name__ == '__main__':
    main()
