"""
scripts/run_eval_all.py
========================
Batch evaluation of all testable models in MODEL_REGISTRY.yaml using eval_v2.

Reads the registry, filters for production-worthy checkpoints, groups by data type
(ca_only, backbone, all_atom), and runs one eval_v2 call per group — so all models
within the same atom-level are compared side-by-side in the same figures.

Usage
-----
# Dry-run: print all commands without executing
python scripts/run_eval_all.py --dry_run

# Full run (all sections, all groups):
python scripts/run_eval_all.py --out_dir plots/eval_all --n 1000 --steps 100

# Fast run (physics only, no PyRosetta):
python scripts/run_eval_all.py --out_dir plots/eval_fast \\
    --n 500 --steps 50 --sections physics equivariance --no_pyrosetta

# Selected groups only:
python scripts/run_eval_all.py --groups ca_only --out_dir plots/eval_ca

# Best-of-each-family only (one representative per model family):
python scripts/run_eval_all.py --best_only --out_dir plots/eval_best --n 1000

Output
------
  plots/eval_all/
    ca_only/           — all Cα models compared side-by-side
      figure1_physics.png
      figure2_equivariance.png
      figure3_energy_<label>.png   (energy-conditioned models only)
      figure4_novelty_<label>.png
      figure5_pyrosetta_<label>.png
      metrics.json
    backbone/          — all backbone models compared
    all_atom/          — all-atom models compared
    summary.json       — merged metrics from all groups
    eval_commands.sh   — saved shell commands for reproducibility
"""

import sys
import os
import json
import argparse
import subprocess
from pathlib import Path

import yaml

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

# Status values considered testable by default (skip failed, smoke_test, partial)
TESTABLE_STATUSES_DEFAULT  = {'production', 'in_progress'}
TESTABLE_STATUSES_WITH_PARTIAL = {'production', 'in_progress', 'partial'}

# Minimum epochs for a model to be considered worth evaluating
MIN_EPOCHS = 100

# Models explicitly known to be bad runs — excluded even if status=production/partial
# (e.g., large-batch runs that didn't converge, or runs explicitly flagged as "do not use")
KNOWN_BAD_IDS = {
    'baseline/v2',           # large-batch run, explicitly flagged as bad
}
# Additionally, notes containing these strings trigger exclusion
BAD_NOTE_KEYWORDS = ['do not use', 'explicitly flagged as bad', 'stagnated', 'catastrophic']

# Data type → test file path (relative to repo root)
TEST_FILES = {
    'ca_only':        'data/test.npz',
    'torsion_ca':     'data/test.npz',
    'all_atom':       'data_all_atom/test.npz',
    'backbone':       'data_backbone/test.npz',
    'backbone_torsion': 'data_backbone/test.npz',
}

# Group names: which data types map to which eval group
DATA_TYPE_TO_GROUP = {
    'ca_only':          'ca_only',
    'torsion_ca':       'ca_only',      # torsion models generate Cα coords
    'all_atom':         'all_atom',
    'backbone':         'backbone',
    'backbone_torsion': 'backbone',     # backbone torsion generates 30-atom backbone
}

# "Best of each family" — one representative per model family
# Edit this dict to change which checkpoint represents each family
BEST_OF_FAMILY = {
    'ca_only': [
        # label → checkpoint path
        ('AdaLN+E+Physics',    'checkpoints/transformer_adaln_energy_physics/v1/best.pt'),
        ('AdaLN-Transformer',  'checkpoints/transformer_adaln/v1/best.pt'),
        ('EGNN+Physics0.10',   'checkpoints/egnn/v4/best.pt'),
        ('EGNN+AdaLN+Energy',  'checkpoints/egnn_adaln/v1/best.pt'),
        ('FlowMatch+E+Physics','checkpoints/flowmatch_energy_physics/v1/best.pt'),
        ('SE3Flow+AdaLN',      'checkpoints/se3flow_adaln_velocity/v1/best.pt'),
        ('TorsionTransformer', 'checkpoints/torsion_transformer/v1/best.pt'),
    ],
    'all_atom': [
        ('SE3Flow-AllAtom-v2', 'checkpoints/se3flow_all_atom_v2/v1/best.pt'),
    ],
    'backbone': [
        ('BackboneTransformer', 'checkpoints/backbone_transformer/v1/best.pt'),
        ('BackboneIPA',         'checkpoints/backbone_ipa/v1/best.pt'),
    ],
}


# ─────────────────────────────────────────────────────────────────────────────
# MODEL SELECTION
# ─────────────────────────────────────────────────────────────────────────────

def load_registry(repo_root: Path) -> list:
    """Load MODEL_REGISTRY.yaml and return the list of model entries."""
    registry_path = repo_root / 'MODEL_REGISTRY.yaml'
    if not registry_path.exists():
        print(f"ERROR: MODEL_REGISTRY.yaml not found at {registry_path}")
        sys.exit(1)
    with open(registry_path) as f:
        data = yaml.safe_load(f)
    return data.get('models', [])


def is_testable(model: dict, repo_root: Path,
                include_partial: bool = False,
                force_ids: set = None) -> tuple:
    """
    Return (True, reason) if model is worth evaluating, else (False, reason).

    force_ids: set of model IDs that bypass all status/notes filters (checkpoint
               must still exist and data_type must be known).
    """
    model_id = model.get('id', '')
    forced   = force_ids and model_id in force_ids

    if not forced:
        if model_id in KNOWN_BAD_IDS:
            return False, f"explicitly excluded (known bad run)"

        notes = str(model.get('notes', '') or '').lower()
        for kw in BAD_NOTE_KEYWORDS:
            if kw in notes:
                return False, f"notes contain '{kw}'"

        allowed = TESTABLE_STATUSES_WITH_PARTIAL if include_partial else TESTABLE_STATUSES_DEFAULT
        status  = model.get('status', 'unknown')
        if status not in allowed:
            return False, f"status={status}"

        epochs = model.get('training', {}).get('epochs_run', 0)
        if epochs < MIN_EPOCHS:
            return False, f"only {epochs} epochs (< {MIN_EPOCHS})"

    # These checks apply even to forced models (checkpoint must exist)
    ckpt_path = repo_root / model.get('checkpoint', '')
    if not ckpt_path.exists():
        return False, f"checkpoint missing: {ckpt_path}"

    data_type = model.get('data', 'unknown')
    if data_type not in DATA_TYPE_TO_GROUP:
        return False, f"unknown data type: {data_type}"

    test_file = repo_root / TEST_FILES.get(data_type, '')
    if test_file and not test_file.exists():
        return False, f"test file missing: {test_file}"

    if forced:
        return True, "force-included"
    return True, "ok"


def select_models(models: list, repo_root: Path,
                  include_partial: bool = False,
                  force_ids: set = None) -> dict:
    """
    Filter models and group them by data group (ca_only, all_atom, backbone).
    Returns {group: [(label, checkpoint_path, model_entry)]}.
    """
    groups    = {'ca_only': [], 'all_atom': [], 'backbone': []}
    force_ids = force_ids or set()

    for m in models:
        ok, reason = is_testable(m, repo_root, include_partial, force_ids)
        data_type  = m.get('data', '')
        group      = DATA_TYPE_TO_GROUP.get(data_type)

        if ok and group:
            label = m.get('name', m.get('id', 'unknown'))
            ckpt  = str(repo_root / m['checkpoint'])
            tag   = " [force-included]" if reason == "force-included" else ""
            groups[group].append((label, ckpt, m))
            if tag:
                print(f"  FORCE {m.get('id','?'):<45}{tag}")
        else:
            name = m.get('id', m.get('name', '?'))
            print(f"  SKIP  {name:<45} — {reason}")

    return groups


def select_best_of_family(repo_root: Path) -> dict:
    """Return {group: [(label, checkpoint_path)]} using BEST_OF_FAMILY."""
    groups = {}
    for group, entries in BEST_OF_FAMILY.items():
        valid = []
        for label, rel_ckpt in entries:
            ckpt = repo_root / rel_ckpt
            if ckpt.exists():
                valid.append((label, str(ckpt), {}))
            else:
                print(f"  SKIP  {label} — checkpoint missing: {ckpt}")
        if valid:
            groups[group] = valid
    return groups


# ─────────────────────────────────────────────────────────────────────────────
# COMMAND BUILDING
# ─────────────────────────────────────────────────────────────────────────────

def build_eval_command(
    group: str,
    models: list,       # [(label, checkpoint, entry)]
    out_dir: str,
    args,
) -> str:
    """
    Build one eval_v2 command for a group of models with the same data type.
    The first model is --ckpt; the rest are --ckpt_ref.
    """
    repo_root = Path(__file__).parent.parent

    test_file_key = {
        'ca_only':   'ca_only',
        'all_atom':  'all_atom',
        'backbone':  'backbone',
    }[group]
    test_file = str(repo_root / TEST_FILES[test_file_key])

    primary_label, primary_ckpt, _ = models[0]
    ref_models = models[1:]

    import sys
    python_bin = sys.executable   # use the same interpreter that runs this script
    parts = [
        f"{python_bin} scripts/eval_v2/main.py",
        f"    --ckpt '{primary_ckpt}'",
    ]

    if ref_models:
        ref_ckpts  = " ".join(f"'{c}'" for _, c, _ in ref_models)
        ref_labels = " ".join(f"'{l}'" for l, _, _ in ref_models)
        parts.append(f"    --ckpt_ref {ref_ckpts}")
        all_labels = f"'{primary_label}' {ref_labels}"
    else:
        all_labels = f"'{primary_label}'"

    parts.append(f"    --labels {all_labels}")
    parts.append(f"    --test '{test_file}'")
    parts.append(f"    --n {args.n}")
    parts.append(f"    --steps {args.steps}")
    parts.append(f"    --batch {args.batch}")
    parts.append(f"    --sections {' '.join(args.sections)}")

    if args.no_pyrosetta:
        parts.append("    --no_pyrosetta")

    parts.append(f"    --temperatures {' '.join(str(t) for t in args.temperatures)}")
    parts.append(f"    --guidance_scale {args.guidance_scale}")
    parts.append(f"    --n_per_tau {args.n_per_tau}")
    parts.append(f"    --n_noise {args.n_noise}")
    parts.append(f"    --n_rotations {args.n_rotations}")
    parts.append(f"    --n_iso {args.n_iso}")
    parts.append(f"    --n_pyrosetta {args.n_pyrosetta}")
    parts.append(f"    --n_novel_pdb {args.n_novel_pdb}")
    parts.append(f"    --out_dir '{out_dir}'")
    parts.append(f"    --seed {args.seed}")

    return " \\\n".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY AGGREGATION
# ─────────────────────────────────────────────────────────────────────────────

def merge_summaries(out_dir: Path) -> dict:
    """Load metrics.json from each group and merge into a flat summary."""
    summary = {}
    for group in ('ca_only', 'backbone', 'all_atom'):
        metrics_file = out_dir / group / 'metrics.json'
        if metrics_file.exists():
            with open(metrics_file) as f:
                data = json.load(f)
            for label, m in data.get('per_model', {}).items():
                summary[f"{group}/{label}"] = m
    return summary


def print_summary_table(summary: dict):
    """Print a compact summary table of all evaluated models."""
    print(f"\n{'═'*100}")
    print(f"  EVALUATION SUMMARY — all models")
    print(f"{'═'*100}")
    header = (f"  {'Model':<40} {'AtomType':<12} {'Bond±0.5':<10} {'Bond±0.2':<10} "
              f"{'BondRMSE':<10} {'Clash%':<8} {'Rg':>7} {'ETE':>8} {'MMD':>8}")
    print(header)
    print(f"  {'─'*96}")

    for key, m in sorted(summary.items()):
        nm  = m.get('native_metrics', {})
        cm  = m.get('ca_metrics', {})
        grp = key.split('/')[0]
        lbl = key.split('/', 1)[1][:38]
        atype = nm.get('atom_type', '?')[:10]

        v05   = nm.get('bond_valid_05',    float('nan'))
        v02   = nm.get('bond_valid_02',    float('nan'))
        brmse = nm.get('bond_rmse',        float('nan'))
        clash = nm.get('clash_rate',       float('nan'))
        rg    = cm.get('rg_mean',          nm.get('rg_mean', float('nan')))
        ete   = cm.get('ete_mean',         nm.get('ete_mean', float('nan')))
        mmd   = cm.get('mmd',              float('nan'))

        def _f(v, fmt):
            return '—' if v != v else format(v, fmt)

        print(
            f"  {lbl:<40} {atype:<12} "
            f"{_f(v05*100,'.1f')+'%':<10} {_f(v02*100,'.1f')+'%':<10} "
            f"{_f(brmse,'.4f'):<10} {_f(clash*100,'.1f')+'%':<8} "
            f"{_f(rg,'.2f'):>7} {_f(ete,'.2f'):>8} {_f(mmd,'.4f'):>8}"
        )

    print(f"{'═'*100}")


# ─────────────────────────────────────────────────────────────────────────────
# ARGUMENT PARSING
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Batch-evaluate all production models from MODEL_REGISTRY.yaml using eval_v2."
    )
    # ── Selection ─────────────────────────────────────────────────────────────
    p.add_argument('--best_only', action='store_true',
                   help='Evaluate only the best-of-each-family models (see BEST_OF_FAMILY dict). '
                        'Recommended for a fast, clean comparison.')
    p.add_argument('--include_partial', action='store_true',
                   help='Include partial-status models (default: only production + in_progress)')
    p.add_argument('--force_include', nargs='+', default=[],
                   metavar='ID',
                   help='Model IDs to force-include regardless of status/notes filters '
                        '(e.g. egnn_adaln_aa/v3). Checkpoint must still exist.')
    p.add_argument('--groups', nargs='+', default=['ca_only', 'backbone', 'all_atom'],
                   choices=['ca_only', 'backbone', 'all_atom'],
                   help='Which data groups to evaluate')

    # ── Output ────────────────────────────────────────────────────────────────
    p.add_argument('--out_dir', default='plots/eval_all',
                   help='Root output directory (one subfolder per group)')
    p.add_argument('--dry_run', action='store_true',
                   help='Print commands without executing them')

    # ── eval_v2 pass-through args ─────────────────────────────────────────────
    p.add_argument('--n',           type=int,   default=500,
                   help='Structures per model')
    p.add_argument('--steps',       type=int,   default=100,
                   help='DDIM / ODE steps')
    p.add_argument('--batch',       type=int,   default=256)
    p.add_argument('--sections',    nargs='+',  default=['all'],
                   choices=['all', 'physics', 'equivariance', 'energy', 'novelty', 'pyrosetta'],
                   help='eval_v2 sections to run')
    p.add_argument('--no_pyrosetta', action='store_true')
    p.add_argument('--temperatures', nargs='+', type=float, default=[0.0, 0.25, 0.5, 0.75, 1.0])
    p.add_argument('--guidance_scale', type=float, default=2.0)
    p.add_argument('--n_per_tau',   type=int,   default=200)
    p.add_argument('--n_noise',     type=int,   default=10)
    p.add_argument('--n_rotations', type=int,   default=30)
    p.add_argument('--n_iso',       type=int,   default=300)
    p.add_argument('--n_pyrosetta', type=int,   default=30)
    p.add_argument('--n_novel_pdb', type=int,   default=20)
    p.add_argument('--seed',        type=int,   default=0)

    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args     = parse_args()
    repo_root = Path(__file__).parent.parent
    out_dir   = Path(args.out_dir)

    print(f"\nBatch evaluation — eval_v2")
    print(f"  Registry: {repo_root / 'MODEL_REGISTRY.yaml'}")
    print(f"  Out dir:  {out_dir}")
    print(f"  Dry run:  {args.dry_run}")
    print(f"  Groups:   {args.groups}")
    print(f"  Best only: {args.best_only}")

    # ── Select models ─────────────────────────────────────────────────────────
    if args.best_only:
        print("\n[Best-of-family selection]")
        groups = select_best_of_family(repo_root)
    else:
        print("\n[Registry scan]")
        models = load_registry(repo_root)
        all_groups = select_models(models, repo_root,
                                   include_partial=args.include_partial,
                                   force_ids=set(args.force_include))
        groups = {g: v for g, v in all_groups.items() if g in args.groups and v}

    total_models = sum(len(ms) for ms in groups.values())
    for g, ms in groups.items():
        print(f"\n  {g}: {len(ms)} models")
        for label, ckpt, _ in ms:
            print(f"    ✓  {label}")

    # Runtime estimate (rough: ~3 s / model / 100 steps on GPU)
    est_min = total_models * args.n * args.steps / (100 * 1000) * 5   # ~5 min per 1000 structs
    print(f"\n  Total: {total_models} models across {len([g for g in groups if groups[g]])} groups")
    print(f"  Estimated runtime: ~{est_min:.0f} min on GPU (n={args.n}, steps={args.steps})")
    print(f"  Tip: use --best_only to run just 7+1+2 representative models (~{est_min*10//total_models:.0f} min)")

    if not any(groups.values()):
        print("\nNo testable models found. Check registry and checkpoint paths.")
        sys.exit(1)

    # ── Build commands ─────────────────────────────────────────────────────────
    commands = {}
    for group in args.groups:
        models = groups.get(group, [])
        if not models:
            continue
        group_out = str(out_dir / group)
        cmd = build_eval_command(group, models, group_out, args)
        commands[group] = (cmd, group_out, models)

    # ── Save commands to shell script ─────────────────────────────────────────
    sh_path = out_dir / 'eval_commands.sh'
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(sh_path, 'w') as f:
        f.write("#!/bin/bash\n")
        f.write(f"# Generated by run_eval_all.py\n")
        f.write(f"# Run from the deeplife/ repo root.\n\n")
        for group, (cmd, group_out, _) in commands.items():
            f.write(f"# ── {group} ──\n")
            f.write(f"mkdir -p {group_out}\n")
            f.write(cmd + "\n\n")
    sh_path.chmod(0o755)
    print(f"\nCommands saved → {sh_path}")

    # ── Print / execute commands ──────────────────────────────────────────────
    summary_all = {}

    for group, (cmd, group_out, models) in commands.items():
        n_models = len(models)
        print(f"\n{'═'*80}")
        print(f"  GROUP: {group}  ({n_models} models)")
        print(f"  OUT:   {group_out}")
        print(f"{'═'*80}")
        print(cmd)

        if args.dry_run:
            continue

        print(f"\n  Running eval_v2 for {group}…")
        result = subprocess.run(
            cmd, shell=True, cwd=str(repo_root),
            executable='/bin/bash',
        )
        if result.returncode != 0:
            print(f"\n  ⚠ eval_v2 exited with code {result.returncode} for group {group}")
            print(f"    Check logs above and re-run manually if needed.")
        else:
            print(f"\n  ✓ {group} complete → {group_out}")

    # ── Merge and print summary ───────────────────────────────────────────────
    if not args.dry_run:
        summary = merge_summaries(out_dir)
        if summary:
            print_summary_table(summary)

            summary_path = out_dir / 'summary.json'
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2)
            print(f"\nSummary JSON → {summary_path}")

    print(f"\nDone. All outputs in {out_dir}/")


if __name__ == '__main__':
    main()
