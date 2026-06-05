"""
scripts/eval_v2/main.py
========================
Entry point for the comprehensive Chignolin evaluation pipeline v2.

Usage
-----
# Full evaluation on one checkpoint:
python scripts/eval_v2/main.py \\
    --ckpt checkpoints/flowmatch_energy/v3/best.pt \\
    --test data/test.npz \\
    --n 500 --steps 100 \\
    --out_dir plots/eval_v2/

# Compare multiple models:
python scripts/eval_v2/main.py \\
    --ckpt     checkpoints/flowmatch_energy/v3/best.pt \\
    --ckpt_ref checkpoints/backbone_ipa/v1/best.pt \\
               checkpoints/se3flow_aa/v2/best.pt \\
    --labels   "FlowMatch+Energy" "BackboneIPA" "SE3FlowAA" \\
    --test     data/test.npz --n 500

# Select specific sections only:
python scripts/eval_v2/main.py \\
    --ckpt checkpoints/se3flow/v2/best.pt \\
    --sections physics equivariance \\
    --test data/test.npz --n 200

# Skip PyRosetta for speed:
python scripts/eval_v2/main.py \\
    --ckpt checkpoints/flowmatch_energy/v3/best.pt \\
    --test data/test.npz --n 500 --no_pyrosetta

Output
------
  out_dir/figure1_physics.png
  out_dir/figure2_equivariance.png
  out_dir/figure3_energy.png       (energy-conditioned models only)
  out_dir/figure4_novelty.png
  out_dir/figure5_pyrosetta.png    (unless --no_pyrosetta)
  out_dir/metrics.json             (all metrics, all models)
  out_dir/novel_structures/        (PDB files of top novel structures)
"""

import sys
import json
import argparse
import numpy as np
from pathlib import Path

import torch

# Package root on path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.eval_v2.model_utils import (
    load_model_from_ckpt, generate, load_reference,
    is_energy_conditioned, save_pdbs, atom_type_str, ca_from_coords,
)
from scripts.eval_v2.physics_metrics import (
    compute_physics_metrics, mmd_rbf, pairwise_diversity,
    radius_of_gyration, end_to_end, print_physics_table,
)
from scripts.eval_v2.equivariance import (
    run_equivariance_tests, print_equivariance_results,
)
from scripts.eval_v2.energy_analysis import run_energy_analysis
from scripts.eval_v2.novelty import (
    run_novelty_analysis, per_temperature_novelty,
)
from scripts.eval_v2.pyrosetta_utils import (
    pyrosetta_validate, print_pyrosetta_results, score_tau_samples,
)
from scripts.eval_v2.plotting import (
    plot_physics, plot_equivariance, plot_energy_analysis,
    plot_novelty, plot_pyrosetta, plot_tau_reu,
)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _hist(values, bins=50):
    """Pre-compute histogram for compact JSON storage (replotting without raw coords)."""
    v = np.asarray(values).flatten()
    if len(v) == 0:
        return None
    if v.std() < 1e-9:
        return {'exact': float(v[0])}   # zero-variance: store single value
    try:
        counts, edges = np.histogram(v, bins=bins, density=True)
        return {'edges': edges.tolist(), 'counts': counts.tolist()}
    except ValueError:
        # Data range too small for requested bins (near-exact values, e.g. NeRF bonds)
        return {'exact': float(v.mean())}


def _compute_model_plot_data(coords: np.ndarray, ref_coords: np.ndarray) -> dict:
    """
    Compute compact histogram / array data needed to reproduce all physics plots
    without storing raw coordinate arrays. Called once per model during eval.
    """
    from scripts.eval_v2.physics_metrics import (
        bond_lengths_ca, radius_of_gyration, end_to_end,
        omega_dihedrals, compute_ramachandran,
    )
    from scripts.eval_v2.model_utils import ca_from_coords

    n_atoms = coords.shape[1]
    ca = ca_from_coords(coords)

    # Cα bond lengths
    ca_bl = bond_lengths_ca(ca).flatten()

    # Native bond lengths (model-specific)
    try:
        if n_atoms == 10:
            native_bl = ca_bl
        elif n_atoms == 30:
            from models.backbone_physics import _SRC, _DST
            src, dst = np.array(_SRC), np.array(_DST)
            native_bl = np.linalg.norm(coords[:, dst] - coords[:, src], axis=-1).flatten()
        elif n_atoms == 93:
            from models.physics_aa import _BOND_INDICES
            consec = np.linalg.norm(np.diff(coords, axis=1), axis=-1)
            native_bl = consec[:, _BOND_INDICES].flatten()
        else:
            native_bl = ca_bl
    except Exception:
        native_bl = ca_bl

    # Rg, ETE per structure
    rg_vals  = radius_of_gyration(ca)
    ete_vals = end_to_end(ca)

    # Per-residue positional variance (10 values)
    per_res_var = ca.var(axis=0).sum(axis=-1).tolist()

    # ω dihedral (backbone/all-atom only)
    omega_hist = None
    if n_atoms >= 30:
        try:
            bb = coords[:, :30] if n_atoms > 30 else coords
            omega_hist = _hist(omega_dihedrals(bb).flatten())
        except Exception:
            pass

    # Ramachandran as 2D histogram (60×60, compact)
    rama_data = None
    try:
        rama = compute_ramachandran(coords[:min(500, len(coords))])
        phi_f = np.array(rama['phi']).flatten()
        psi_f = np.array(rama['psi']).flatten()
        if len(phi_f) > 0:
            h2d, xe, ye = np.histogram2d(
                phi_f, psi_f, bins=60, range=[[-180, 180], [-180, 180]]
            )
            rama_data = {
                'hist2d':           h2d.tolist(),
                'phi_edges':        xe.tolist(),
                'psi_edges':        ye.tolist(),
                'nerf_reconstructed': rama.get('nerf_reconstructed', False),
            }
    except Exception:
        pass

    return {
        'n_atoms':               n_atoms,
        'ca_bond_hist':          _hist(ca_bl),
        'native_bond_hist':      _hist(native_bl),
        'rg_hist':               _hist(rg_vals, bins=40),
        'ete_hist':              _hist(ete_vals, bins=40),
        'per_residue_variance':  per_res_var,
        'omega_hist':            omega_hist,
        'ramachandran':          rama_data,
    }


def _write_json(all_results: dict, out_dir) -> None:
    """Write metrics.json, called after each major stage so crashes don't lose data."""
    json_path = Path(out_dir) / 'metrics.json'
    with open(json_path, 'w') as f:
        json.dump(_safe_json(all_results), f, indent=2)


def _make_label(config: dict, used: set) -> str:
    mt   = config.get('model_type', 'unknown').upper()
    phys = config.get('training', {}).get('physics_weight', 0.0) > 0
    base = f"{mt}+Physics" if phys else mt
    label, suffix = base, 2
    while label in used:
        label = f"{base}_{suffix}"; suffix += 1
    return label


def _safe_json(obj):
    """Recursively make object JSON-serialisable (strip private _keys, convert numpy)."""
    if isinstance(obj, dict):
        # Keys can be floats (e.g. tau values in mmd_vs_quartiles) — only filter string _keys
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
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, float) and (obj != obj):   # NaN
        return None
    else:
        return obj


# ─────────────────────────────────────────────────────────────────────────────
# ARGUMENT PARSING
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="eval_v2: comprehensive Chignolin evaluation pipeline."
    )
    # ── Checkpoints ───────────────────────────────────────────────────────────
    p.add_argument('--ckpt',     required=True,
                   help='Primary checkpoint path')
    p.add_argument('--ckpt_ref', nargs='*', default=[],
                   help='Additional checkpoints for side-by-side comparison')
    p.add_argument('--labels',   nargs='*', default=None,
                   help='Display labels (must match total checkpoint count if given)')

    # ── Data ─────────────────────────────────────────────────────────────────
    p.add_argument('--test', default=None,
                   help='Test .npz path (auto-detected from checkpoint config if omitted)')

    # ── Generation ────────────────────────────────────────────────────────────
    p.add_argument('--n',     type=int, default=500, help='Structures to generate per model')
    p.add_argument('--steps', type=int, default=100, help='Sampling steps (DDIM / ODE)')
    p.add_argument('--batch', type=int, default=256, help='Generation batch size')
    p.add_argument('--seed',  type=int, default=0)

    # ── Sections ─────────────────────────────────────────────────────────────
    p.add_argument('--sections', nargs='+',
                   default=['all'],
                   choices=['all', 'physics', 'equivariance', 'energy', 'novelty', 'pyrosetta'],
                   help='Which evaluation sections to run')
    p.add_argument('--no_pyrosetta', action='store_true',
                   help='Skip PyRosetta section even if "all" is selected')

    # ── Equivariance ─────────────────────────────────────────────────────────
    p.add_argument('--n_noise',     type=int, default=10,
                   help='Noise vectors for equivariance tests 1 & 2')
    p.add_argument('--n_rotations', type=int, default=30,
                   help='Rotations per noise vector for equivariance tests')
    p.add_argument('--n_iso',       type=int, default=300,
                   help='Structures for isotropy test (Test 3)')

    # ── Energy ───────────────────────────────────────────────────────────────
    p.add_argument('--temperatures', nargs='+', type=float,
                   default=[0.0, 0.25, 0.5, 0.75, 1.0])
    p.add_argument('--guidance_scale', type=float, default=2.0)
    p.add_argument('--n_per_tau',      type=int, default=200,
                   help='Structures to generate per temperature')

    # ── PyRosetta ────────────────────────────────────────────────────────────
    p.add_argument('--n_pyrosetta', type=int, default=30,
                   help='Number of structures to score with PyRosetta')
    p.add_argument('--n_ref_pyrosetta', type=int, default=15,
                   help='Reference structures for Rosetta score calibration')
    p.add_argument('--n_tau_rosetta', type=int, default=15,
                   help='Structures to score per τ for the τ-vs-REU plot '
                        '(energy-conditioned models only; requires pyrosetta)')

    # ── Output ───────────────────────────────────────────────────────────────
    p.add_argument('--out_dir',    default='plots/eval_v2',
                   help='Output directory for figures and JSON')
    p.add_argument('--n_novel_pdb', type=int, default=20,
                   help='Number of most novel structures to save as PDB')

    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\neval_v2 | device={device} | seed={args.seed}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Determine which sections to run
    run_all = 'all' in args.sections
    run = {
        'physics':      run_all or 'physics'      in args.sections,
        'equivariance': run_all or 'equivariance' in args.sections,
        'energy':       run_all or 'energy'       in args.sections,
        'novelty':      run_all or 'novelty'      in args.sections,
        'pyrosetta':    (run_all or 'pyrosetta' in args.sections) and not args.no_pyrosetta,
    }
    print(f"Sections: {[k for k, v in run.items() if v]}")

    # ── Validate args ─────────────────────────────────────────────────────────
    all_ckpts = [args.ckpt] + list(args.ckpt_ref)
    if args.labels is not None and len(args.labels) != len(all_ckpts):
        raise ValueError(f"--labels has {len(args.labels)} entries but "
                         f"{len(all_ckpts)} checkpoints given")

    # ── Determine test path ───────────────────────────────────────────────────
    test_path = args.test
    if test_path is None:
        _ckpt = torch.load(args.ckpt, map_location='cpu', weights_only=False)
        test_path = _ckpt['config']['data'].get('test_path')
        if test_path is None:
            raise ValueError("--test not given and checkpoint has no data.test_path")
        repo_root = Path(__file__).parent.parent.parent
        test_path = str(repo_root / test_path)
        print(f"Auto test path: {test_path}")

    # ── Load reference data ───────────────────────────────────────────────────
    print(f"\nLoading reference: {test_path}")
    ref_coords, ref_energies = load_reference(test_path)
    print(f"  {len(ref_coords):,} reference structures  ({ref_coords.shape[1]} atoms)")

    # ── Generate from each checkpoint ─────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"Generating structures ({args.n} per model, {args.steps} steps)")
    print(f"{'─'*60}")

    models_data   = {}   # {label: (coords_native, metrics_native, metrics_ca)}
    raw_configs   = {}   # {label: config}
    raw_diffusions = {}  # {label: diffusion}
    raw_models    = {}   # {label: model}
    used_labels   = set()

    for i, ckpt_path in enumerate(all_ckpts):
        print(f"\n[{i+1}/{len(all_ckpts)}] {ckpt_path}")
        model, diffusion, config, scale = load_model_from_ckpt(ckpt_path, device)
        label = (args.labels[i] if args.labels else _make_label(config, used_labels))
        used_labels.add(label)

        n_res = config['data']['n_residues']
        print(f"  Label: {label}  n_residues={n_res}  coord_scale={scale}")
        print(f"  Generating {args.n} structures …")

        samples = generate(model, diffusion, args.n, n_res, scale,
                           args.steps, device, args.batch)
        print(f"  Generated: {samples.shape}")

        # Compute metrics (native + Cα-projected)
        native_metrics = compute_physics_metrics(samples)

        ca_samples = ca_from_coords(samples)
        ca_ref     = ca_from_coords(ref_coords)
        ca_metrics = {
            'rg_mean':  float(radius_of_gyration(ca_samples).mean()),
            'rg_std':   float(radius_of_gyration(ca_samples).std()),
            'ete_mean': float(end_to_end(ca_samples).mean()),
            'ete_std':  float(end_to_end(ca_samples).std()),
            'mmd':      mmd_rbf(ca_samples, ca_ref),
            'diversity': pairwise_diversity(ca_samples),
            **{k: v for k, v in compute_physics_metrics(ca_samples).items()
               if k not in ('atom_type',)},
        }
        ca_metrics['atom_type'] = 'ca_projected'

        print_physics_table(native_metrics,
                            ref_metrics=compute_physics_metrics(ref_coords[:min(1000, len(ref_coords))]),
                            label=label)

        models_data[label]    = (samples, native_metrics, ca_metrics)
        raw_configs[label]    = config
        raw_diffusions[label] = diffusion
        raw_models[label]     = model

    # ── Reference metrics ─────────────────────────────────────────────────────
    ref_native_metrics = compute_physics_metrics(ref_coords[:min(2000, len(ref_coords))])
    ref_ca_metrics     = compute_physics_metrics(
        ca_from_coords(ref_coords[:min(2000, len(ref_coords))])
    )

    # Load existing metrics.json if present so incremental reruns MERGE rather
    # than overwrite sections produced in a previous run.
    _json_path = out_dir / 'metrics.json'
    _existing  = {}
    if _json_path.exists():
        try:
            with open(_json_path) as _f:
                _existing = json.load(_f)
            print(f"  Merging with existing metrics.json "
                  f"(sections: {list(_existing.get('sections', {}).keys())})")
        except Exception:
            pass

    all_results = {
        'per_model': _existing.get('per_model', {}),
        'sections':  _existing.get('sections',  {}),
        'ref_plot_data': _existing.get('ref_plot_data', None),
    }

    # Store reference plot data once (needed for replotting)
    if all_results['ref_plot_data'] is None:
        print("  Computing reference plot data...")
        all_results['ref_plot_data'] = _safe_json(
            _compute_model_plot_data(ref_coords[:min(2000, len(ref_coords))],
                                     ref_coords[:min(2000, len(ref_coords))])
        )

    for lbl, (coords, nm, cm) in models_data.items():
        print(f"  Computing plot data for {lbl}...")
        all_results['per_model'][lbl] = {
            'native_metrics': nm,
            'ca_metrics':     cm,
            'n_samples':      len(coords),
            'atom_type':      atom_type_str(coords.shape[1]),
            'plot_data':      _safe_json(_compute_model_plot_data(coords, ref_coords)),
        }

    # Checkpoint: save per_model data immediately so section crashes don't lose it
    _write_json(all_results, out_dir)

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 1: PHYSICAL QUALITY
    # ─────────────────────────────────────────────────────────────────────────
    if run['physics']:
        print(f"\n{'='*60}")
        print("SECTION 1: Physical / Biological Plausibility")
        print(f"{'='*60}")
        plot_physics(
            models_data=models_data,
            ref_coords=ref_coords,
            ref_metrics=ref_native_metrics,
            save_path=str(out_dir / 'figure1_physics.png'),
        )
        all_results['sections']['physics'] = {
            'ref_metrics': ref_native_metrics,
        }
        _write_json(all_results, out_dir)

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 2: SE(3) EQUIVARIANCE
    # ─────────────────────────────────────────────────────────────────────────
    if run['equivariance']:
        print(f"\n{'='*60}")
        print("SECTION 2: SE(3) Equivariance")
        print(f"{'='*60}")
        equiv_results = {}
        for lbl, (coords, nm, cm) in models_data.items():
            print(f"\n  [{lbl}]")
            model     = raw_models[lbl]
            diffusion = raw_diffusions[lbl]
            config    = raw_configs[lbl]
            res = run_equivariance_tests(
                model, diffusion, config,
                n_noise=args.n_noise, n_rotations=args.n_rotations,
                n_generate=args.n_iso, ddim_steps=args.steps,
                device=device,
            )
            print_equivariance_results(res, label=lbl)
            equiv_results[lbl] = res

        plot_equivariance(
            results_dict=equiv_results,
            save_path=str(out_dir / 'figure2_equivariance.png'),
        )
        _existing_equiv = all_results['sections'].get('equivariance', {})
        _existing_equiv.update(_safe_json(equiv_results))
        all_results['sections']['equivariance'] = _existing_equiv
        _write_json(all_results, out_dir)

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 3: ENERGY / CONFORMATIONAL SAMPLING
    # ─────────────────────────────────────────────────────────────────────────
    if run['energy']:
        print(f"\n{'='*60}")
        print("SECTION 3: Energy and Conformational Sampling")
        print(f"{'='*60}")
        energy_section = {}

        for lbl, (coords, nm, cm) in models_data.items():
            print(f"\n  [{lbl}]")
            model     = raw_models[lbl]
            diffusion = raw_diffusions[lbl]
            config    = raw_configs[lbl]
            scale     = config['data'].get('coord_scale', 16.32)
            # Torsion types have coord_scale=1.0
            _torsion  = config['model_type'] in (
                'torsion_flow_energy', 'torsion_transformer_energy', 'backbone_ipa_energy')
            if _torsion: scale = 1.0

            res = run_energy_analysis(
                model, diffusion, config, coord_scale=scale,
                ref_coords=ref_coords, ref_energies=ref_energies,
                temperatures=args.temperatures,
                guidance_scale=args.guidance_scale,
                n_per_tau=args.n_per_tau,
                ddim_steps=args.steps,
                device=device, batch_size=args.batch,
            )
            energy_section[lbl] = res

            if not res.get('skipped'):
                plot_energy_analysis(
                    energy_results=res,
                    ref_coords=ref_coords,
                    model_label=lbl,
                    save_path=str(out_dir / f'figure3_energy_{lbl.replace("/", "_")}.png'),
                )

        # Build energy section with compact per-tau histograms for replotting
        energy_for_json = {}
        for lbl, r in energy_section.items():
            entry = {k: v for k, v in r.items() if k not in ('all_samples', 'ref_strata')}
            # Store per-tau Rg/ETE histograms so the distributions can be replotted
            if not r.get('skipped') and 'all_samples' in r:
                tau_hists = {}
                for tau, samp in r['all_samples'].items():
                    if samp is not None and len(samp) > 0:
                        ca_s = ca_from_coords(np.array(samp))
                        tau_hists[str(tau)] = {
                            'rg_hist':  _hist(radius_of_gyration(ca_s), bins=30),
                            'ete_hist': _hist(end_to_end(ca_s), bins=30),
                        }
                entry['tau_hists'] = tau_hists

                # Per-τ Rosetta scoring for τ-vs-REU validation plot
                if run['pyrosetta']:
                    print(f"  [τ-Rosetta] Scoring {args.n_tau_rosetta} structs/τ for {lbl}…")
                    tau_reu = score_tau_samples(
                        r['all_samples'],
                        n_per_tau=args.n_tau_rosetta,
                    )
                    if tau_reu:
                        entry['tau_rosetta'] = _safe_json(tau_reu)
                        print(f"    Scored τ values: {sorted(tau_reu.keys())}")

            energy_for_json[lbl] = entry

        _existing_energy = all_results['sections'].get('energy', {})
        _existing_energy.update(_safe_json(energy_for_json))
        all_results['sections']['energy'] = _existing_energy
        _write_json(all_results, out_dir)

        # Combine all per-τ REU data and emit figure3b
        tau_reu_all = {
            lbl: energy_for_json[lbl].get('tau_rosetta', {})
            for lbl in energy_for_json
            if not energy_section[lbl].get('skipped')
            and energy_for_json[lbl].get('tau_rosetta')
        }
        if tau_reu_all:
            plot_tau_reu(
                tau_reu_dict=tau_reu_all,
                save_path=str(out_dir / 'figure3b_tau_reu.png'),
            )

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 4: NOVELTY
    # ─────────────────────────────────────────────────────────────────────────
    if run['novelty']:
        print(f"\n{'='*60}")
        print("SECTION 4: Conformational Novelty")
        print(f"{'='*60}")
        novelty_section = {}

        for lbl, (coords, nm, cm) in models_data.items():
            print(f"\n  [{lbl}]")

            # Base novelty analysis
            nov = run_novelty_analysis(
                gen_coords=coords,
                ref_coords=ref_coords,
                label=lbl,
                max_ref=2000,
                max_atoms=10,   # always use Cα for fair comparison
                apply_physics_filter=True,
            )

            # Per-tau novelty (if energy model and energy section was run)
            pt_nov = {}
            if (is_energy_conditioned(raw_configs[lbl]) and
                    run['energy'] and
                    lbl in all_results.get('sections', {}).get('energy', {})):
                energy_res = energy_section.get(lbl, {})
                if not energy_res.get('skipped'):
                    pt_nov = per_temperature_novelty(
                        all_samples=energy_res.get('all_samples', {}),
                        ref_coords=ref_coords,
                        max_ref=1000,
                    )

            # Save top-K most novel structures
            gen_valid = nov.get('_gen_valid', coords)
            gen_nnd   = nov.get('_gen_nnd', np.zeros(len(gen_valid)))
            if args.n_novel_pdb > 0 and len(gen_valid) > 0:
                top_idx = np.argsort(-gen_nnd)[:args.n_novel_pdb]
                novel_dir = str(out_dir / 'novel_structures' / lbl.replace('/', '_'))
                save_pdbs(gen_valid[top_idx], out_dir=novel_dir,
                          n_save=args.n_novel_pdb, label=f"{lbl}_novel")

            plot_novelty(
                novelty_results=nov,
                per_tau_novelty=pt_nov,
                ref_coords=ref_coords,
                label=lbl,
                save_path=str(out_dir / f'figure4_novelty_{lbl.replace("/", "_")}.png'),
            )

            # Build compact plot_data for replotting without raw coords
            _pd: dict = {}
            gen_fp    = nov.get('_gen_fp')
            ref_fp    = nov.get('_ref_fp')
            _gen_nnd  = nov.get('_gen_nnd')
            _self_nnd = nov.get('_self_nnd')
            thr       = nov.get('threshold', 1.0)

            if _gen_nnd is not None:
                _pd['gen_nnd']  = _gen_nnd.tolist()
            if _self_nnd is not None:
                _pd['self_nnd'] = _self_nnd.tolist()
            if nov.get('rmsd_matrix') is not None:
                _pd['rmsd_matrix'] = np.array(nov['rmsd_matrix']).tolist()

            if gen_fp is not None and ref_fp is not None and len(gen_fp) > 0:
                from scripts.eval_v2.novelty import pca_2d, coverage_precision
                try:
                    gen_2d, ref_2d = pca_2d(gen_fp, ref_fp)
                    _pd['pca_gen_xy'] = gen_2d.tolist()
                    _pd['pca_ref_xy'] = ref_2d[:2000].tolist()
                    thrs = np.linspace(thr * 0.1, thr * 3.0, 30)
                    _pd['coverage_curve'] = [
                        [float(t)] + list(coverage_precision(gen_fp, ref_fp, t))
                        for t in thrs
                    ]
                except Exception:
                    pass

            novelty_section[lbl] = {k: v for k, v in nov.items() if not k.startswith('_')}
            novelty_section[lbl]['per_tau']   = pt_nov
            novelty_section[lbl]['plot_data'] = _safe_json(_pd)

        _existing_novelty = all_results['sections'].get('novelty', {})
        _existing_novelty.update(_safe_json(novelty_section))
        all_results['sections']['novelty'] = _existing_novelty
        _write_json(all_results, out_dir)

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 5: PYROSETTA VALIDATION
    # ─────────────────────────────────────────────────────────────────────────
    if run['pyrosetta']:
        print(f"\n{'='*60}")
        print("SECTION 5: PyRosetta Validation")
        print(f"{'='*60}")
        rosetta_section = {}

        # Score reference structures once for calibration
        ref_rosetta = pyrosetta_validate(
            ref_coords[:min(args.n_ref_pyrosetta, len(ref_coords))],
            n_max=args.n_ref_pyrosetta,
            constraint_weight=5.0,
        )

        for lbl, (coords, nm, cm) in models_data.items():
            print(f"\n  [{lbl}]")
            res = pyrosetta_validate(
                coords=coords,
                n_max=args.n_pyrosetta,
                constraint_weight=5.0,
                ref_coords=ref_coords,
            )
            print_pyrosetta_results(res, label=lbl)

            plot_pyrosetta(
                rosetta_results=res,
                ref_rosetta_results=ref_rosetta,
                model_label=lbl,
                save_path=str(out_dir / f'figure5_pyrosetta_{lbl.replace("/", "_")}.png'),
            )
            # Keep per_structure for replotting (max n_pyrosetta entries, small)
            rosetta_section[lbl] = _safe_json(res)

        _existing_pyrosetta = all_results['sections'].get('pyrosetta', {})
        _existing_pyrosetta.update(rosetta_section)
        all_results['sections']['pyrosetta'] = _existing_pyrosetta
        _write_json(all_results, out_dir)

    # Final write (no-op if already up-to-date, but ensures consistent state)
    _write_json(all_results, out_dir)
    json_path = out_dir / 'metrics.json'
    print(f"\nMetrics JSON → {json_path}")
    print(f"All outputs  → {out_dir}/")
    print("\neval_v2 complete.")


if __name__ == '__main__':
    main()
