"""
scripts/eval_v2/energy_analysis.py
=====================================
Energy / conformational sampling analysis for energy-conditioned models.

Tests:
  1. Temperature sweep  — generate at τ ∈ {0, 0.25, 0.5, 0.75, 1.0}
  2. Spearman monotonicity — Rg, ETE, diversity must increase with τ
  3. MMD vs. reference quartile — compare each τ to matching energy quartile
  4. Physical plausibility vs. τ — bond validity, clash rate should be ~constant
  5. Per-τ PCA — conformational space coverage across τ values

Adapted from scripts/analyze_energy_conditioning.py.
"""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from .model_utils import generate_at_temperature, ca_from_coords, is_energy_conditioned
from .physics_metrics import (
    bond_lengths_ca, radius_of_gyration, end_to_end,
    pairwise_diversity, mmd_rbf, compute_physics_metrics,
)


# ─────────────────────────────────────────────────────────────────────────────
# PER-TEMPERATURE METRICS
# ─────────────────────────────────────────────────────────────────────────────

def compute_temperature_metrics(coords: np.ndarray) -> dict:
    """Structural + physics metrics for one temperature's generated structures."""
    n = coords.shape[1]
    ca = ca_from_coords(coords)
    rg  = radius_of_gyration(ca)
    ete = end_to_end(ca)
    div = pairwise_diversity(ca)

    phys = compute_physics_metrics(coords)
    return {
        'rg_mean':    float(rg.mean()),
        'rg_std':     float(rg.std()),
        'ete_mean':   float(ete.mean()),
        'ete_std':    float(ete.std()),
        'diversity':  div,
        'bond_valid': phys['bond_valid_05'],
        'bond_rmse':  phys['bond_rmse'],
        'clash_rate': phys['clash_rate'],
    }


# ─────────────────────────────────────────────────────────────────────────────
# REFERENCE STRATIFICATION
# ─────────────────────────────────────────────────────────────────────────────

def stratify_reference(ref_coords: np.ndarray, ref_energies: np.ndarray,
                       n_strata: int = 5) -> list:
    """
    Split reference structures into n_strata energy bins.
    Returns list of (tau_lo, tau_hi, coords_subset) — tau is a proxy for energy percentile.
    """
    percentiles = np.linspace(0, 100, n_strata + 1)
    boundaries  = np.percentile(ref_energies, percentiles)
    strata = []
    for i in range(n_strata):
        lo, hi = boundaries[i], boundaries[i + 1]
        mask = (ref_energies >= lo) & (ref_energies <= hi)
        strata.append((i / n_strata, (i + 1) / n_strata, ref_coords[mask]))
    return strata


# ─────────────────────────────────────────────────────────────────────────────
# MONOTONICITY TESTS (Spearman rank correlation)
# ─────────────────────────────────────────────────────────────────────────────

def run_monotonicity_tests(sweep_results: list) -> dict:
    """
    Spearman rank correlation between τ and structural metrics.
    sweep_results: list of (tau, metrics_dict)
    PASS criterion: r > 0.8, p < 0.05.
    """
    from scipy import stats

    taus     = [t for t, m in sweep_results]
    rg_vals  = [m['rg_mean']   for t, m in sweep_results]
    ete_vals = [m['ete_mean']  for t, m in sweep_results]
    div_vals = [m['diversity'] for t, m in sweep_results]
    bv_vals  = [m['bond_valid'] for t, m in sweep_results]

    def _spearman(x, y):
        if len(set(x)) < 2:
            return 0.0, 1.0
        r, p = stats.spearmanr(x, y)
        return float(r), float(p)

    rg_r,  rg_p  = _spearman(taus, rg_vals)
    ete_r, ete_p = _spearman(taus, ete_vals)
    div_r, div_p = _spearman(taus, div_vals)
    bv_r,  bv_p  = _spearman(taus, bv_vals)
    bv_range      = max(bv_vals) - min(bv_vals)

    def _pass_mono(r, p): return r > 0.8 and p < 0.05

    tests = {
        'rg_monotone':   {'r': rg_r,  'p': rg_p,  'pass': _pass_mono(rg_r,  rg_p)},
        'ete_monotone':  {'r': ete_r, 'p': ete_p, 'pass': _pass_mono(ete_r, ete_p)},
        'div_monotone':  {'r': div_r, 'p': div_p, 'pass': _pass_mono(div_r, div_p)},
        'bv_stable':     {'range': bv_range,       'pass': bv_range < 0.15},
    }

    print(f"\n{'━' * 70}")
    print(f"  Monotonicity tests (Spearman correlation with τ)")
    print(f"{'━' * 70}")
    for name, res in tests.items():
        status = "✓ PASS" if res['pass'] else "✗ FAIL"
        if 'r' in res:
            print(f"  {status}  {name:<20} r={res['r']:.3f}, p={res['p']:.4f}")
        else:
            print(f"  {status}  {name:<20} range={res['range']*100:.1f}pp")
    all_pass = all(v['pass'] for v in tests.values())
    print(f"{'━' * 70}")
    print(f"  Overall: {'ALL PASS ✓' if all_pass else 'SOME FAILED ✗'}")
    print(f"{'━' * 70}")

    return tests


# ─────────────────────────────────────────────────────────────────────────────
# MMD VS. REFERENCE QUARTILE
# ─────────────────────────────────────────────────────────────────────────────

def mmd_vs_quartiles(
    all_samples: dict,     # {tau: (N, n_atoms, 3)}
    ref_strata: list,      # from stratify_reference()
) -> dict:
    """
    For each τ, compute MMD between generated structures and:
      (a) the matching reference energy quartile (should be low if model is calibrated)
      (b) the opposite reference energy quartile (should be higher)

    Returns {tau: {'mmd_matched': float, 'mmd_opposite': float}}.
    """
    taus = sorted(all_samples.keys())
    n_strata = len(ref_strata)
    results = {}

    for i, tau in enumerate(taus):
        samples = all_samples[tau]
        ca_gen  = ca_from_coords(samples)

        # Matching stratum: map τ → stratum index linearly
        matched_idx  = min(int(tau * n_strata), n_strata - 1)
        opposite_idx = (n_strata - 1) - matched_idx

        _, _, matched_coords  = ref_strata[matched_idx]
        _, _, opposite_coords = ref_strata[opposite_idx]

        ca_match = ca_from_coords(matched_coords)  if len(matched_coords)  > 0 else None
        ca_opp   = ca_from_coords(opposite_coords) if len(opposite_coords) > 0 else None

        mmd_match = mmd_rbf(ca_gen, ca_match)   if ca_match  is not None else float('nan')
        mmd_opp   = mmd_rbf(ca_gen, ca_opp)     if ca_opp    is not None else float('nan')

        results[tau] = {
            'mmd_matched':  mmd_match,
            'mmd_opposite': mmd_opp,
            'ratio':        mmd_match / max(mmd_opp, 1e-9),  # <1 = better match, >1 = worse
        }

    return results


# ─────────────────────────────────────────────────────────────────────────────
# CONSOLE TABLE
# ─────────────────────────────────────────────────────────────────────────────

def print_sweep_table(sweep_results: list, label: str = "Temperature sweep"):
    print(f"\n{'━' * 88}")
    print(f"  {label}")
    print(f"{'━' * 88}")
    header = (f"  {'τ':>5}  {'Rg (Å)':>12}  {'ETE (Å)':>12}  "
              f"{'Bond%':>7}  {'Clash%':>7}  {'Diversity':>10}  {'BondRMSE':>10}")
    print(header)
    print(f"  {'─' * 82}")
    for tau, m in sweep_results:
        print(
            f"  {tau:5.2f}  "
            f"{m['rg_mean']:6.3f}±{m['rg_std']:.2f}  "
            f"{m['ete_mean']:6.2f}±{m['ete_std']:.1f}  "
            f"{m['bond_valid']*100:6.1f}%  "
            f"{m['clash_rate']*100:6.1f}%  "
            f"{m['diversity']:10.3f}  "
            f"{m['bond_rmse']:10.4f}"
        )
    print(f"{'━' * 88}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def run_energy_analysis(
    model, diffusion, config: dict, coord_scale: float,
    ref_coords: np.ndarray, ref_energies: np.ndarray | None,
    temperatures: list = None, guidance_scale: float = 2.0,
    n_per_tau: int = 200, ddim_steps: int = 100,
    device: str = 'cpu', batch_size: int = 256,
) -> dict:
    """
    Run the full energy conditioning analysis.

    Returns dict with:
      'sweep_results'    : [(tau, metrics)] for each temperature
      'all_samples'      : {tau: np.ndarray} generated structures
      'monotonicity'     : test results
      'mmd_vs_quartiles' : {tau: {mmd_matched, mmd_opposite, ratio}}
      'ref_strata'       : reference energy strata
      'ref_metrics'      : metrics on the full reference set
    """
    if not is_energy_conditioned(config):
        print("  [energy_analysis] Skipped — model is not energy-conditioned.")
        return {'skipped': True, 'reason': 'non-energy-conditioned model'}

    if temperatures is None:
        temperatures = [0.0, 0.25, 0.5, 0.75, 1.0]

    n_res = config['data']['n_residues']

    # Reference stratification
    n_strata = min(5, len(temperatures))
    if ref_energies is not None:
        ref_strata = stratify_reference(ref_coords, ref_energies, n_strata=n_strata)
    else:
        ref_strata = []
        print("  [energy_analysis] No reference energies — skipping quartile stratification.")

    # Temperature sweep
    all_samples   = {}
    sweep_results = []

    print(f"\n  Temperature sweep (guidance_scale={guidance_scale})")
    for tau in sorted(temperatures):
        print(f"    τ={tau:.2f} — generating {n_per_tau} structures …", end='', flush=True)
        samples = generate_at_temperature(
            model, diffusion, tau=tau, n=n_per_tau, n_residues=n_res,
            coord_scale=coord_scale, ddim_steps=ddim_steps,
            guidance_scale=guidance_scale, device=device, batch_size=batch_size,
        )
        all_samples[tau] = samples
        m = compute_temperature_metrics(samples)

        # MMD vs full reference (using Cα)
        ca_ref = ca_from_coords(ref_coords)
        m['mmd_vs_ref'] = mmd_rbf(ca_from_coords(samples), ca_ref)

        sweep_results.append((tau, m))
        print(f"  Rg={m['rg_mean']:.3f}±{m['rg_std']:.2f}  "
              f"ETE={m['ete_mean']:.2f}  Bond={m['bond_valid']*100:.1f}%")

    print_sweep_table(sweep_results, label=f"Temperature sweep (guidance_scale={guidance_scale})")

    # Reference metrics
    ref_metrics = compute_temperature_metrics(ref_coords)
    print(f"\n  Reference:  Rg={ref_metrics['rg_mean']:.3f}±{ref_metrics['rg_std']:.2f}  "
          f"ETE={ref_metrics['ete_mean']:.2f}  Bond={ref_metrics['bond_valid']*100:.1f}%  "
          f"Diversity={ref_metrics['diversity']:.3f}")

    # Monotonicity tests
    mono_tests = run_monotonicity_tests(sweep_results)

    # MMD vs. quartiles
    mmd_q = {}
    if ref_strata:
        mmd_q = mmd_vs_quartiles(all_samples, ref_strata)

    return {
        'sweep_results':    [(float(t), m) for t, m in sweep_results],
        'all_samples':      all_samples,
        'monotonicity':     mono_tests,
        'mmd_vs_quartiles': mmd_q,
        'ref_strata':       ref_strata,
        'ref_metrics':      ref_metrics,
        'skipped':          False,
    }
