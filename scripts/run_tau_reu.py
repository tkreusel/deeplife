"""
scripts/run_tau_reu.py
=======================
Patch script: generate per-τ Rosetta energy scores (REU) for all
energy-conditioned models and produce figure3b_tau_reu.png.

This script reads from an existing eval_overnight output directory (produced by
run_eval_all.py + eval_v2/main.py), re-generates temperature sweep samples only
for energy-conditioned models, scores them with PyRosetta, appends the results
back into the existing metrics.json, and writes figure3b_tau_reu.png alongside it.

It is safe to re-run: tau_rosetta data already present in the JSON is merged rather
than overwritten, and no other section of metrics.json is modified.

Usage
-----
# Dry-run: show which models would be scored, without running anything
python scripts/run_tau_reu.py --dry_run

# Run on the overnight eval results:
python scripts/run_tau_reu.py \\
    --eval_dir plots/eval_overnight \\
    --n_per_tau 20 \\
    --steps 100

# Only ca_only group:
python scripts/run_tau_reu.py --eval_dir plots/eval_overnight --groups ca_only

# Override temperatures / guidance scale (must match what was used in original run):
python scripts/run_tau_reu.py \\
    --eval_dir plots/eval_overnight \\
    --temperatures 0.0 0.25 0.5 0.75 1.0 \\
    --guidance_scale 2.0 \\
    --n_per_tau 20

Output (per group, e.g. plots/eval_overnight/ca_only/):
  figure3b_tau_reu.png     — τ vs REU scatter + violin plot
  metrics.json             — updated in-place with tau_rosetta sub-section
"""

import sys
import json
import argparse
import numpy as np
from pathlib import Path

import torch

# ── ensure repo root is on path ───────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.eval_v2.model_utils import (
    load_model_from_ckpt,
    generate_at_temperature,
    is_energy_conditioned,
    ca_from_coords,
    load_reference,
)
from scripts.eval_v2.pyrosetta_utils import score_tau_samples
from scripts.eval_v2.plotting import plot_tau_reu

# ── constants that mirror run_eval_all.py ─────────────────────────────────────
TEST_FILES = {
    'ca_only':   'data/test.npz',
    'backbone':  'data_backbone/test.npz',
    'all_atom':  'data_all_atom/test.npz',
}


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _safe_json(obj):
    """Recursively make object JSON-serialisable (strip private _keys, convert numpy)."""
    if isinstance(obj, dict):
        return {
            (str(k) if not isinstance(k, str) else k): _safe_json(v)
            for k, v in obj.items()
            if not (isinstance(k, str) and k.startswith('_'))
        }
    elif isinstance(obj, (list, tuple)):
        return [_safe_json(x) for x in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, float) and obj != obj:   # NaN → None for JSON
        return None
    else:
        return obj


def _load_metrics(metrics_path: Path) -> dict:
    """Load metrics.json, returning {} if missing or corrupt."""
    if not metrics_path.exists():
        print(f"  [WARN] metrics.json not found: {metrics_path}")
        return {}
    try:
        with open(metrics_path) as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        print(f"  [WARN] metrics.json is corrupt ({e}): {metrics_path}")
        return {}


def _write_metrics(metrics_path: Path, data: dict) -> None:
    """Write metrics.json atomically via a temp file."""
    tmp = metrics_path.with_suffix('.json.tmp')
    with open(tmp, 'w') as f:
        json.dump(data, f, indent=2)
    tmp.replace(metrics_path)
    print(f"  ✓ Wrote {metrics_path}")


def _get_checkpoint_for_label(metrics: dict, label: str) -> str | None:
    """
    Try to recover the checkpoint path for a model label from metrics.json.
    metrics.json (from eval_v2/main.py) does not store checkpoint paths
    directly, so we look in per_model[label] for a stored 'checkpoint' key,
    then fall back to the eval_commands.sh in the same directory.
    """
    per_model = metrics.get('per_model', {})
    if label in per_model:
        ckpt = per_model[label].get('checkpoint')
        if ckpt and Path(ckpt).exists():
            return ckpt
    return None


def _parse_checkpoints_from_sh(sh_path: Path) -> dict:
    """
    Parse eval_commands.sh to recover {label: checkpoint_path} mapping.
    Returns a flat dict of all label→ckpt pairs found in the file.
    """
    if not sh_path.exists():
        return {}
    text = sh_path.read_text()

    import re
    # Extract --ckpt and --ckpt_ref values
    ckpts_raw  = re.findall(r"--ckpt(?:_ref)?\s+((?:'[^']+'\s*)+)", text)
    labels_raw = re.findall(r"--labels\s+((?:'[^']+'\s*)+)", text)

    def _split_quoted(s):
        return re.findall(r"'([^']+)'", s)

    all_ckpts  = []
    all_labels = []
    for block in ckpts_raw:
        all_ckpts.extend(_split_quoted(block))
    for block in labels_raw:
        all_labels.extend(_split_quoted(block))

    return dict(zip(all_labels, all_ckpts))


# ─────────────────────────────────────────────────────────────────────────────
# CORE: score one group
# ─────────────────────────────────────────────────────────────────────────────

def process_group(
    group: str,
    group_dir: Path,
    args,
    device: str,
) -> None:
    """
    Load all energy-conditioned models from a group's metrics.json,
    generate per-tau samples, score with PyRosetta, write results back,
    and emit figure3b_tau_reu.png.
    """
    metrics_path = group_dir / 'metrics.json'
    metrics      = _load_metrics(metrics_path)
    if not metrics:
        print(f"  [SKIP] No metrics.json in {group_dir}")
        return

    # Load reference data for this group
    test_file = REPO_ROOT / TEST_FILES.get(group, TEST_FILES['ca_only'])
    if not test_file.exists():
        print(f"  [SKIP] Test file not found: {test_file}")
        return
    ref_coords, _ = load_reference(str(test_file))
    print(f"  Reference: {len(ref_coords)} structs from {test_file.name}")

    # Recover checkpoint paths from eval_commands.sh
    sh_path        = group_dir / 'eval_commands.sh'
    label_to_ckpt  = _parse_checkpoints_from_sh(sh_path)
    if not label_to_ckpt:
        # Also try the parent-level eval_commands.sh
        label_to_ckpt = _parse_checkpoints_from_sh(group_dir.parent / 'eval_commands.sh')
    if not label_to_ckpt:
        print(f"  [WARN] Could not find eval_commands.sh — "
              f"checkpoint paths not recoverable for {group_dir}")
        print(f"         Use --ckpt_override to provide them explicitly.")

    per_model       = metrics.get('per_model', {})
    energy_section  = metrics.get('sections', {}).get('energy', {})

    # Collect energy-conditioned models that don't already have tau_rosetta
    models_to_score = []
    for label in per_model:
        # Skip non-energy models (energy section entry is skipped or absent)
        energy_entry = energy_section.get(label, {})
        if energy_entry.get('skipped', True):
            continue

        # Check if already scored
        if energy_entry.get('tau_rosetta') and not args.force_rescore:
            taus_done = sorted(energy_entry['tau_rosetta'].keys())
            print(f"  [SKIP] {label}: tau_rosetta already present "
                  f"(τ = {taus_done}). Use --force_rescore to redo.")
            continue

        # Recover checkpoint
        ckpt = label_to_ckpt.get(label)
        if not ckpt:
            ckpt = _get_checkpoint_for_label(metrics, label)
        if not ckpt or not Path(ckpt).exists():
            print(f"  [SKIP] {label}: checkpoint not found "
                  f"(label_to_ckpt had: {label_to_ckpt.get(label, 'N/A')})")
            continue

        models_to_score.append((label, ckpt))

    if not models_to_score:
        print(f"  [INFO] No models to score in {group}.")
        # Still regenerate the plot if there is existing tau_rosetta data
        _maybe_replot(group_dir, metrics, energy_section)
        return

    print(f"\n  Models to score: {[l for l, _ in models_to_score]}")

    if args.dry_run:
        for label, ckpt in models_to_score:
            print(f"    [DRY RUN] Would score {label} from {ckpt}")
        return

    # ── Score each model ──────────────────────────────────────────────────────
    tau_reu_all = {}   # {label: {str(tau): {mean, std, n, scores}}}

    # Carry over models that are already scored (so they appear in the combined plot)
    for label, energy_entry in energy_section.items():
        if energy_entry.get('tau_rosetta') and not energy_entry.get('skipped', True):
            tau_reu_all[label] = energy_entry['tau_rosetta']
            print(f"  [CARRY] {label}: using existing tau_rosetta data")

    for label, ckpt in models_to_score:
        print(f"\n  ── {label} ──")
        print(f"     Checkpoint: {ckpt}")

        # Load model
        try:
            model, diffusion, config, scale = load_model_from_ckpt(ckpt, device)
        except Exception as e:
            print(f"  [ERROR] Failed to load {label}: {e}")
            continue

        if not is_energy_conditioned(config):
            print(f"  [SKIP] {label} is not energy-conditioned after loading config.")
            continue

        n_res = config['data']['n_residues']

        # Generate per-τ samples
        print(f"  Generating {args.n_per_tau} structures at each of "
              f"τ = {args.temperatures} (guidance={args.guidance_scale}) …")
        all_samples = {}
        for tau in args.temperatures:
            try:
                samples = generate_at_temperature(
                    model, diffusion,
                    tau=tau,
                    n=args.n_per_tau,
                    n_residues=n_res,
                    coord_scale=scale,
                    ddim_steps=args.steps,
                    guidance_scale=args.guidance_scale,
                    device=device,
                    batch_size=args.batch,
                )
                all_samples[tau] = samples
                print(f"    τ={tau:.2f}: {len(samples)} structs generated")
            except Exception as e:
                print(f"  [ERROR] Generation failed at τ={tau}: {e}")

        if not all_samples:
            print(f"  [SKIP] No samples generated for {label}.")
            continue

        # Score with PyRosetta
        print(f"  Scoring with PyRosetta ({args.n_per_tau} structs/τ) …")
        tau_reu = score_tau_samples(
            tau_samples=all_samples,
            n_per_tau=args.n_per_tau,
        )

        if not tau_reu:
            print(f"  [WARN] score_tau_samples returned empty for {label} "
                  f"(PyRosetta unavailable or all structures failed).")
            continue

        scored_taus = sorted(tau_reu.keys(), key=float)
        for t in scored_taus:
            r = tau_reu[t]
            print(f"    τ={float(t):.2f}: mean={r['mean']:.1f} ± {r['std']:.1f} "
                  f"(n={r['n']})")

        # Merge into metrics.json energy section
        if 'sections' not in metrics:
            metrics['sections'] = {}
        if 'energy' not in metrics['sections']:
            metrics['sections']['energy'] = {}
        if label not in metrics['sections']['energy']:
            metrics['sections']['energy'][label] = {}

        metrics['sections']['energy'][label]['tau_rosetta'] = _safe_json(tau_reu)
        tau_reu_all[label] = tau_reu

        # Write incrementally after each model so crashes don't lose data
        _write_metrics(metrics_path, metrics)

    # ── Plot figure3b ─────────────────────────────────────────────────────────
    if tau_reu_all:
        fig3b_path = str(group_dir / 'figure3b_tau_reu.png')
        print(f"\n  Plotting figure3b → {fig3b_path}")
        try:
            plot_tau_reu(
                tau_reu_dict=tau_reu_all,
                save_path=fig3b_path,
            )
            print(f"  ✓ figure3b_tau_reu.png saved")
        except Exception as e:
            print(f"  [ERROR] plot_tau_reu failed: {e}")
    else:
        print(f"  [INFO] No tau_rosetta data available — figure3b not generated.")


def _maybe_replot(group_dir: Path, metrics: dict, energy_section: dict) -> None:
    """Re-emit figure3b if tau_rosetta data already exists in the JSON."""
    tau_reu_all = {}
    for label, entry in energy_section.items():
        if entry.get('tau_rosetta') and not entry.get('skipped', True):
            tau_reu_all[label] = entry['tau_rosetta']

    if tau_reu_all:
        fig3b_path = str(group_dir / 'figure3b_tau_reu.png')
        print(f"  Re-plotting existing tau_rosetta data → {fig3b_path}")
        try:
            plot_tau_reu(tau_reu_dict=tau_reu_all, save_path=fig3b_path)
            print(f"  ✓ figure3b_tau_reu.png saved")
        except Exception as e:
            print(f"  [ERROR] plot_tau_reu failed: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# ARGUMENT PARSING
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Generate τ-vs-REU Rosetta scatter for all energy-conditioned models."
    )
    p.add_argument('--eval_dir', default='plots/eval_overnight',
                   help='Root output directory that was used for the original eval run '
                        '(must contain ca_only/, backbone/, all_atom/ sub-dirs with metrics.json)')
    p.add_argument('--groups', nargs='+', default=['ca_only', 'backbone', 'all_atom'],
                   choices=['ca_only', 'backbone', 'all_atom'],
                   help='Which groups to process')

    # Energy sweep params — should match the original run
    p.add_argument('--temperatures', nargs='+', type=float,
                   default=[0.0, 0.25, 0.5, 0.75, 1.0])
    p.add_argument('--guidance_scale', type=float, default=2.0)
    p.add_argument('--n_per_tau', type=int, default=20,
                   help='Structures to generate and score per τ '
                        '(PyRosetta is slow — 20 gives a good spread)')
    p.add_argument('--steps', type=int, default=100,
                   help='DDIM / ODE steps for sample generation')
    p.add_argument('--batch', type=int, default=64,
                   help='Batch size for generation')

    p.add_argument('--force_rescore', action='store_true',
                   help='Re-score even if tau_rosetta already present in metrics.json')
    p.add_argument('--dry_run', action='store_true',
                   help='Print which models would be scored, without executing anything')
    p.add_argument('--seed', type=int, default=42)

    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    eval_dir = REPO_ROOT / args.eval_dir

    print(f"\nrun_tau_reu.py")
    print(f"  Eval dir : {eval_dir}")
    print(f"  Device   : {device}")
    print(f"  Groups   : {args.groups}")
    print(f"  τ values : {args.temperatures}  guidance={args.guidance_scale}")
    print(f"  n/τ      : {args.n_per_tau}  steps={args.steps}")
    print(f"  Dry run  : {args.dry_run}")
    print(f"  Force    : {args.force_rescore}")

    for group in args.groups:
        group_dir = eval_dir / group
        if not group_dir.is_dir():
            print(f"\n[SKIP] {group}: directory not found ({group_dir})")
            continue

        print(f"\n{'═'*70}")
        print(f"  GROUP: {group}  ({group_dir})")
        print(f"{'═'*70}")

        process_group(group, group_dir, args, device)

    print(f"\nDone.")


if __name__ == '__main__':
    main()
