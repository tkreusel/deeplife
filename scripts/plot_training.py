"""
scripts/plot_training.py

Plot training curves from one or more training runs.

Usage:
    # single run
    python scripts/plot_training.py --logs checkpoints/baseline/v1/log.jsonl

    # compare multiple versions
    python scripts/plot_training.py --logs checkpoints/baseline/v1/log.jsonl \
                                           checkpoints/baseline/v2/log.jsonl \
                                    --labels "v1 lr=3e-4" "v2 lr=1e-4"

    # save to file instead of showing
    python scripts/plot_training.py --logs checkpoints/baseline/v1/log.jsonl \
                                    --save plots/v1_curves.png
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


# ── Data loading ──────────────────────────────────────────────────────────────

def load_log(path: str) -> dict:
    """
    Load a log.jsonl file. Returns dict with lists:
        epochs, train_losses, val_epochs, val_losses, lrs
    """
    epochs, train_losses, lrs = [], [], []
    val_epochs, val_losses    = [], []

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            epochs.append(entry['epoch'])
            train_losses.append(entry['train_loss'])
            lrs.append(entry.get('lr', None))
            if entry.get('val_loss') is not None:
                val_epochs.append(entry['epoch'])
                val_losses.append(entry['val_loss'])

    return {
        'epochs':       epochs,
        'train_losses': train_losses,
        'val_epochs':   val_epochs,
        'val_losses':   val_losses,
        'lrs':          lrs,
    }


def print_summary(label: str, data: dict):
    if not data['epochs']:
        print(f"{label}: no data")
        return

    best_train = min(data['train_losses'])
    best_train_ep = data['epochs'][data['train_losses'].index(best_train)]

    print(f"\n{'─'*50}")
    print(f"  {label}")
    print(f"{'─'*50}")
    print(f"  Epochs completed : {data['epochs'][-1]}")
    print(f"  Final train loss : {data['train_losses'][-1]:.4f}")
    print(f"  Best  train loss : {best_train:.4f}  (epoch {best_train_ep})")

    if data['val_losses']:
        best_val = min(data['val_losses'])
        best_val_ep = data['val_epochs'][data['val_losses'].index(best_val)]
        print(f"  Final val loss   : {data['val_losses'][-1]:.4f}")
        print(f"  Best  val loss   : {best_val:.4f}  (epoch {best_val_ep})")
    else:
        print(f"  Val loss         : not yet logged")


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot(log_paths: list, labels: list, save_path: str = None):
    all_data = [load_log(p) for p in log_paths]

    # print summary for each run
    for label, data in zip(labels, all_data):
        print_summary(label, data)

    has_val = any(len(d['val_losses']) > 0 for d in all_data)
    has_lr  = any(d['lrs'] and d['lrs'][0] is not None for d in all_data)

    # number of subplots depends on what data exists
    n_cols = 2 + int(has_lr)   # loss linear | loss log | (lr)
    fig, axes = plt.subplots(1, n_cols, figsize=(6 * n_cols, 4))
    if n_cols == 1:
        axes = [axes]

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    for i, (data, label) in enumerate(zip(all_data, labels)):
        color = colors[i % len(colors)]

        for ax_idx, (ax, log_scale) in enumerate(zip(axes[:2], [False, True])):
            # training loss — thin line, slightly transparent
            ax.plot(
                data['epochs'], data['train_losses'],
                color=color, linewidth=1.0, alpha=0.6,
                label=f"{label} train" if len(all_data) > 1 else "train",
            )
            # validation loss — thicker line with markers
            if data['val_losses']:
                ax.plot(
                    data['val_epochs'], data['val_losses'],
                    color=color, linewidth=2.0, alpha=1.0,
                    marker='o', markersize=3,
                    label=f"{label} val" if len(all_data) > 1 else "val",
                )
            if log_scale:
                ax.set_yscale('log')

        # learning rate
        if has_lr and data['lrs'] and data['lrs'][0] is not None:
            axes[2].plot(
                data['epochs'], data['lrs'],
                color=color, linewidth=1.2,
                label=label if len(all_data) > 1 else None,
            )

    # ── formatting ────────────────────────────────────────────────────────────
    titles    = ['Loss (linear scale)', 'Loss (log scale)', 'Learning rate']
    ylabels   = ['MSE loss', 'MSE loss', 'LR']

    for i, ax in enumerate(axes):
        ax.set_xlabel('Epoch')
        ax.set_ylabel(ylabels[i])
        ax.set_title(titles[i])
        ax.grid(alpha=0.3, which='both')
        ax.legend(fontsize=8)

    # draw a horizontal dashed line at loss=1.0 on both loss plots
    # (this is the starting loss — random noise prediction)
    for ax in axes[:2]:
        ax.axhline(1.0, color='gray', linewidth=0.8, linestyle='--', alpha=0.5,
                   label='random baseline (loss=1.0)')

    fig.suptitle('Training curves', fontsize=13, y=1.01)
    fig.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nSaved to {save_path}")
    else:
        plt.show()

    plt.close(fig)


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--logs', nargs='+', required=True,
        help='One or more paths to log.jsonl files'
    )
    parser.add_argument(
        '--labels', nargs='+', default=None,
        help='Labels for each log file (must match number of --logs). '
             'Defaults to the version directory name, e.g. v1, v2.'
    )
    parser.add_argument(
        '--save', type=str, default=None,
        help='Path to save the plot, e.g. plots/curves.png. '
             'If not given, opens an interactive window.'
    )
    args = parser.parse_args()

    # default labels: use parent directory name (e.g. v1, v2)
    if args.labels is None:
        args.labels = [Path(p).parent.name for p in args.logs]

    if len(args.labels) != len(args.logs):
        raise ValueError(
            f"Number of --labels ({len(args.labels)}) must match "
            f"number of --logs ({len(args.logs)})"
        )

    plot(args.logs, args.labels, save_path=args.save)


if __name__ == '__main__':
    main()