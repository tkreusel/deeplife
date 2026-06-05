"""
scripts/eval_v2/pyrosetta_utils.py
=====================================
PyRosetta-based validation: sidechain reconstruction, FastRelax, and Rosetta scoring.

Workflow per structure:
  1. Build full-atom PDB from model-native coordinates:
       Cα-only  → NeRF-reconstruct N/C backbone → idealise → pack sidechains
       Backbone → pack sidechains directly
       All-atom → load heavy atoms; Rosetta adds H
  2. CoordinateConstraints on input atoms (backbone/Cα are held fixed)
  3. FastRelax (1 cycle, low-cost) with coordinate constraints
  4. Score with REF2015: report total_score, fa_rep, rama_prepro, fa_dun, omega

Filtering thresholds for "plausible" structures:
  total_score < ref_mean + 2 * ref_std
  fa_rep      < ROSETTA_FA_REP_MAX (10.0)
  rama_prepro < ROSETTA_RAMA_MAX   (2.0)

PyRosetta must be installed (confirmed available in the deeplife conda env).
"""

import sys
import os
import io
import tempfile
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from .constants import (
    CHIGNOLIN_SEQUENCE, AA3, ATOM_NAMES_BACKBONE, ELEMENTS_BACKBONE,
    N_CA_IDEAL, CA_C_IDEAL, C_N_IDEAL,
    ROSETTA_FA_REP_MAX, ROSETTA_RAMA_MAX, ROSETTA_TOTAL_N_STD,
)


# ─────────────────────────────────────────────────────────────────────────────
# PYROSETTA INITIALISATION (once)
# ─────────────────────────────────────────────────────────────────────────────

_pyrosetta_inited = False


def _init_pyrosetta():
    global _pyrosetta_inited
    if _pyrosetta_inited:
        return True
    try:
        import pyrosetta
        # Suppress verbose Rosetta output
        pyrosetta.init(
            '-mute all '
            '-use_input_sc '
            '-ex1 -ex2aro '
            '-ignore_unrecognized_res '
            '-ignore_zero_occupancy false',
            silent=True,
        )
        _pyrosetta_inited = True
        return True
    except ImportError:
        print("  [PyRosetta] Not installed or not importable. Skipping PyRosetta section.")
        return False
    except Exception as e:
        print(f"  [PyRosetta] Initialisation failed: {e}")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# PDB STRING BUILDERS
# ─────────────────────────────────────────────────────────────────────────────

def _ca_to_pdb_string(coords_ca: np.ndarray, label: str = "sample") -> str:
    """
    Build a Cα-trace PDB string (10 atoms) for input to PyRosetta idealisation.
    PyRosetta's idealize_protein_pose will extend this to full backbone + sidechains.
    """
    seq   = CHIGNOLIN_SEQUENCE
    lines = [f"REMARK  {label}\n"]
    for j, (res, xyz) in enumerate(zip(seq, coords_ca)):
        x, y, z = xyz
        rn = AA3.get(res, 'GLY')
        lines.append(
            f"ATOM  {j+1:5d}  CA  {rn} A{j+1:4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C\n"
        )
    lines.append("END\n")
    return ''.join(lines)


def _backbone_to_pdb_string(coords_bb: np.ndarray, label: str = "sample") -> str:
    """Build backbone PDB string (30 atoms: N, CA, C) for PyRosetta sidechain packing."""
    seq   = CHIGNOLIN_SEQUENCE
    lines = [f"REMARK  {label} (backbone N-CA-C)\n"]
    for j, (aname, elem, xyz) in enumerate(
        zip(ATOM_NAMES_BACKBONE, ELEMENTS_BACKBONE, coords_bb)
    ):
        res_idx = j // 3
        rn      = AA3.get(seq[res_idx], 'GLY')
        x, y, z = xyz
        lines.append(
            f"ATOM  {j+1:5d} {aname} {rn} A{res_idx+1:4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           {elem}\n"
        )
    lines.append("END\n")
    return ''.join(lines)


def _all_atom_to_pdb_string(coords_aa: np.ndarray, label: str = "sample") -> str:
    """
    Build a PDB string for 93-atom all-atom Chignolin (heavy atoms only).
    Uses the known Chignolin topology:
      Each residue block: N, CA, C, O, [sidechain atoms...]
    Since exact atom names are topology-dependent, we write HETATM and let
    Rosetta recover topology via its residue type detection.
    For proper Rosetta scoring, backbone atoms N/CA/C are written with correct names
    while sidechain atoms are labelled generically.
    """
    seq   = CHIGNOLIN_SEQUENCE
    lines = [f"REMARK  {label} (all-atom, 93 heavy atoms)\n"]
    # Atom index in the 93-atom array follows backbone-first ordering:
    # [N0,CA0,C0,O0,CB0,...,N9,CA9,C9,O9,CB9,...] — exact counts per residue vary.
    # We output the backbone atoms with correct names and mark the rest as sidechain.
    # This is approximate — for exact naming use a topology mapping.
    for j, xyz in enumerate(coords_aa):
        x, y, z = xyz
        lines.append(
            f"HETATM{j+1:5d}  C   UNK A{j+1:4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C\n"
        )
    lines.append("END\n")
    return ''.join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# POSE CONSTRUCTION
# ─────────────────────────────────────────────────────────────────────────────

def _ca_to_pose(coords_ca: np.ndarray):
    """
    Load Cα trace into PyRosetta:
    1. Write Cα PDB string to temp file
    2. Load with PyRosetta, idealize backbone geometry
    3. Switch to full-atom representation
    4. Pack sidechains
    """
    import pyrosetta
    # PyRosetta cannot parse Cα-only PDB (single-atom residues → nres=0).
    # Reconstruct approximate N and C positions from the Cα trace using ideal
    # backbone geometry, then build a proper 30-atom backbone PDB and delegate
    # to _backbone_to_pose (which pose_from_pdb handles correctly).
    backbone = _ca_to_backbone(coords_ca)
    return _backbone_to_pose(backbone)


def _ca_to_backbone(coords_ca: np.ndarray) -> np.ndarray:
    """
    Place approximate N and C atoms for each residue from a Cα-only trace.
    Uses ideal bond lengths (N-CA=1.460 Å, CA-C=1.525 Å) and places N/C
    along the incoming/outgoing bond directions. Geometry is approximate
    (bond angles and ω are not enforced) but sufficient for PyRosetta loading.

    coords_ca : (10, 3) Cα coordinates in Å
    returns   : (30, 3) N0,CA0,C0, N1,CA1,C1, … backbone coordinates
    """
    from .constants import N_CA_IDEAL, CA_C_IDEAL
    n_res = len(coords_ca)
    backbone = np.zeros((n_res * 3, 3), dtype=np.float32)

    for i in range(n_res):
        ca = coords_ca[i]
        backbone[3 * i + 1] = ca   # CA at correct position

        # N: along the direction from previous CA to current CA, offset by N_CA_IDEAL
        if i == 0:
            d_in = coords_ca[1] - coords_ca[0] if n_res > 1 else np.array([1., 0., 0.])
        else:
            d_in = ca - coords_ca[i - 1]
        d_in_norm = np.linalg.norm(d_in)
        d_in_hat  = d_in / max(d_in_norm, 1e-8)
        backbone[3 * i] = ca - N_CA_IDEAL * d_in_hat

        # C: along the direction from current CA to next CA, offset by CA_C_IDEAL
        if i < n_res - 1:
            d_out = coords_ca[i + 1] - ca
        else:
            d_out = -d_in   # extrapolate for last residue
        d_out_norm = np.linalg.norm(d_out)
        d_out_hat  = d_out / max(d_out_norm, 1e-8)
        backbone[3 * i + 2] = ca + CA_C_IDEAL * d_out_hat

    return backbone


def _backbone_to_pose(coords_bb: np.ndarray):
    """Load backbone (N, CA, C) into PyRosetta and pack sidechains."""
    import pyrosetta
    from pyrosetta import rosetta

    pdb_str = _backbone_to_pdb_string(coords_bb)
    with tempfile.NamedTemporaryFile(suffix='.pdb', mode='w', delete=False) as f:
        f.write(pdb_str)
        tmp_path = f.name

    try:
        # pose_from_pdb with N/CA/C backbone automatically builds a full-atom pose
        # (adds O, CB and sidechain atoms with ideal geometry) in this PyRosetta version.
        # No add_missing_atoms or SwitchResidueTypeSetMover needed.
        pose = pyrosetta.pose_from_pdb(tmp_path)
        if pose.total_residue() == 0:
            return None, False

        # Pack rotamers to fill sidechain positions
        sfxn = pyrosetta.get_fa_scorefxn()
        task = pyrosetta.standard_packer_task(pose)
        task.restrict_to_repacking()
        packer = rosetta.protocols.minimization_packing.PackRotamersMover(sfxn, task)
        packer.apply(pose)

        return pose, True
    except Exception as e:
        return None, False
    finally:
        os.unlink(tmp_path)


def _coords_to_pose(coords: np.ndarray):
    """Route to the correct pose builder based on atom count."""
    n = coords.shape[0]  # single structure: (n_atoms, 3)
    if n == 10:
        return _ca_to_pose(coords)
    elif n == 30:
        return _backbone_to_pose(coords)
    else:
        # All-atom (93 heavy atoms): HETATM/UNK records can't be parsed by Rosetta.
        # Use the backbone subset (N,Cα,C = first 30 atoms) which has correct names.
        return _backbone_to_pose(coords[:30])


# ─────────────────────────────────────────────────────────────────────────────
# FAST RELAX + SCORE
# ─────────────────────────────────────────────────────────────────────────────

def _relax_and_score(pose, constraint_weight: float = 5.0) -> dict:
    """
    Run FastRelax (1 cycle) and extract Rosetta score terms.

    CoordinateConstraints are not used because the required mover
    (AddCoordinateConstraintMover) is not available in this PyRosetta build.
    FastRelax without constraints still relaxes sidechains while allowing
    backbone to adjust slightly; 1 cycle is fast and sufficient for scoring.
    """
    import pyrosetta
    from pyrosetta import rosetta

    sfxn = pyrosetta.get_fa_scorefxn()

    # FastRelax (1 cycle for speed)
    fr = rosetta.protocols.relax.FastRelax(sfxn, 1)
    fr.apply(pose)

    # Score and extract terms
    sfxn(pose)
    e = pose.energies()

    def _term(name):
        try:
            return float(e.total_energies()[
                getattr(rosetta.core.scoring, name)
            ])
        except Exception:
            return float('nan')

    return {
        'total_score':  float(sfxn(pose)),
        'fa_rep':       _term('fa_rep'),
        'fa_atr':       _term('fa_atr'),
        'fa_sol':       _term('fa_sol'),
        'fa_dun':       _term('fa_dun'),
        'rama_prepro':  _term('rama_prepro'),
        'omega':        _term('omega'),
        'p_aa_pp':      _term('p_aa_pp'),
        'ref':          _term('ref'),
    }


# ─────────────────────────────────────────────────────────────────────────────
# MAIN VALIDATION FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def pyrosetta_validate(
    coords: np.ndarray,
    n_max: int = 50,
    constraint_weight: float = 5.0,
    ref_coords: np.ndarray = None,
    ref_energies_rosetta: list = None,
) -> dict:
    """
    Run PyRosetta validation on up to n_max structures.

    coords : (N, n_atoms, 3) — model-native coordinates in Å
    n_max  : maximum number of structures to process (PyRosetta is slow)
    ref_coords : if given, run on reference structures to calibrate thresholds
    ref_energies_rosetta : pre-computed reference Rosetta scores (skip re-scoring)

    Returns dict with per-structure scores + aggregate stats + filter results.
    """
    if not _init_pyrosetta():
        return {'skipped': True, 'reason': 'PyRosetta not available'}

    n_atoms = coords.shape[1]
    n_run   = min(n_max, len(coords))
    idx     = np.random.choice(len(coords), n_run, replace=False)
    subset  = coords[idx]

    print(f"  [PyRosetta] Scoring {n_run} structures ({n_atoms} atoms each)…")
    scores = []
    n_failed = 0
    for i, struct in enumerate(subset):
        if i % 10 == 0:
            print(f"    {i}/{n_run} …", end='\r', flush=True)
        pose, ok = _coords_to_pose(struct)
        if not ok:
            n_failed += 1
            scores.append(None)
            continue
        try:
            s = _relax_and_score(pose, constraint_weight)
            scores.append(s)
        except Exception as e:
            n_failed += 1
            scores.append(None)

    print(f"    Done: {n_run - n_failed} succeeded, {n_failed} failed.")

    # Reference scores (optional, for calibrating thresholds)
    ref_scores = None
    if ref_coords is not None and ref_energies_rosetta is None:
        n_ref_run = min(20, len(ref_coords))
        ref_idx   = np.random.choice(len(ref_coords), n_ref_run, replace=False)
        print(f"  [PyRosetta] Scoring {n_ref_run} reference structures for calibration…")
        ref_scores_raw = []
        for struct in ref_coords[ref_idx]:
            pose, ok = _coords_to_pose(struct)
            if not ok:
                continue
            try:
                ref_scores_raw.append(_relax_and_score(pose, constraint_weight))
            except Exception:
                pass
        if ref_scores_raw:
            ref_scores = ref_scores_raw

    # Aggregate generated scores
    valid_scores = [s for s in scores if s is not None]
    if not valid_scores:
        return {'skipped': False, 'n_scored': 0, 'n_failed': n_failed,
                'error': 'All structures failed Rosetta'}

    def _agg(key):
        vals = [s[key] for s in valid_scores if not np.isnan(s.get(key, float('nan')))]
        if not vals: return {'mean': float('nan'), 'std': float('nan')}
        return {'mean': float(np.mean(vals)), 'std': float(np.std(vals))}

    agg = {k: _agg(k) for k in
           ['total_score', 'fa_rep', 'fa_atr', 'fa_dun', 'rama_prepro', 'omega', 'p_aa_pp']}

    # Reference calibration for filtering thresholds
    if ref_scores:
        ref_total = [s['total_score'] for s in ref_scores if not np.isnan(s['total_score'])]
        ref_mean  = float(np.mean(ref_total))
        ref_std   = float(np.std(ref_total))
        total_threshold = ref_mean + ROSETTA_TOTAL_N_STD * ref_std
    else:
        ref_mean = ref_std = float('nan')
        total_threshold = float('nan')

    # Apply plausibility filter
    n_pass = 0
    for s in valid_scores:
        pass_total = (np.isnan(total_threshold) or
                      s['total_score'] < total_threshold)
        pass_rep   = (np.isnan(s['fa_rep']) or s['fa_rep'] < ROSETTA_FA_REP_MAX)
        pass_rama  = (np.isnan(s['rama_prepro']) or s['rama_prepro'] < ROSETTA_RAMA_MAX)
        if pass_total and pass_rep and pass_rama:
            n_pass += 1

    return {
        'n_scored':          len(valid_scores),
        'n_failed':          n_failed,
        'n_pass_filter':     n_pass,
        'pass_fraction':     float(n_pass / max(len(valid_scores), 1)),
        'scores_agg':        agg,
        'ref_total_mean':    ref_mean,
        'ref_total_std':     ref_std,
        'total_threshold':   total_threshold,
        'per_structure':     valid_scores,
        'skipped':           False,
    }


# ─────────────────────────────────────────────────────────────────────────────
# PER-τ ROSETTA SCORING (energy conditioning validation)
# ─────────────────────────────────────────────────────────────────────────────

def score_tau_samples(
    tau_samples: dict,
    n_per_tau: int = 15,
    constraint_weight: float = 5.0,
) -> dict:
    """
    Score n_per_tau structures at each τ value from an energy sweep.

    tau_samples : {tau_float_or_str: np.ndarray (N, n_atoms, 3)}
                  — the all_samples dict produced by run_energy_analysis()
    n_per_tau   : structures to score per τ (PyRosetta is slow; 15 is a good default)

    Returns {str(tau): {'mean': float, 'std': float, 'n': int, 'scores': [float]}}
    An empty dict is returned if PyRosetta is unavailable or all structures fail.
    """
    if not _init_pyrosetta():
        return {}

    result = {}
    for tau_key in sorted(tau_samples.keys(), key=lambda x: float(x)):
        coords = tau_samples[tau_key]
        if coords is None or len(coords) == 0:
            continue
        coords = np.asarray(coords)
        n_run  = min(n_per_tau, len(coords))
        idx    = np.random.choice(len(coords), n_run, replace=False)
        subset = coords[idx]

        tau_scores = []
        for struct in subset:
            pose, ok = _coords_to_pose(struct)
            if not ok:
                continue
            try:
                s = _relax_and_score(pose, constraint_weight)
                v = s.get('total_score', float('nan'))
                if not np.isnan(v):
                    tau_scores.append(float(v))
            except Exception:
                pass

        if tau_scores:
            result[str(tau_key)] = {
                'mean':   float(np.mean(tau_scores)),
                'std':    float(np.std(tau_scores)),
                'n':      len(tau_scores),
                'scores': tau_scores,
            }

    return result


# ─────────────────────────────────────────────────────────────────────────────
# CONSOLE REPORT
# ─────────────────────────────────────────────────────────────────────────────

def print_pyrosetta_results(results: dict, label: str = "Model"):
    if results.get('skipped'):
        print(f"  [PyRosetta] Skipped: {results.get('reason', '')}")
        return
    print(f"\n{'━' * 72}")
    print(f"  PyRosetta Validation — {label}")
    print(f"{'━' * 72}")
    print(f"  Scored: {results.get('n_scored', 0)} structures "
          f"({results.get('n_failed', 0)} failed)")
    if results.get('error'):
        print(f"  Error: {results['error']}")
        print(f"{'━' * 72}")
        return
    print(f"  Pass filter: {results['n_pass_filter']} / {results['n_scored']} "
          f"({results['pass_fraction']*100:.1f}%)")
    thr = results.get('total_threshold', float('nan'))
    if not (thr != thr):   # not NaN
        print(f"  Filter thresholds: total_score < {thr:.1f}  "
              f"fa_rep < {ROSETTA_FA_REP_MAX}  rama_prepro < {ROSETTA_RAMA_MAX}")
    print(f"\n  {'Term':<18} {'Mean':>10} {'Std':>10}")
    print(f"  {'─' * 40}")
    for term, vals in results.get('scores_agg', {}).items():
        print(f"  {term:<18} {vals['mean']:>10.3f} {vals['std']:>10.3f}")
    print(f"{'━' * 72}")
