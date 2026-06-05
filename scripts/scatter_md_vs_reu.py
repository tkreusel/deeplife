"""
scripts/scatter_md_vs_reu.py
=============================
Scatter plot: MD energy label (x) vs Rosetta REU (y) for the all-atom test set.

Workflow per structure:
  1. Extract backbone N, CA, C coordinates from the 93-atom array using known indices.
  2. Build a PyRosetta pose from those backbone coordinates.
     PyRosetta fills in O and sidechain atoms at ideal geometry (no packing, no relaxation).
  3. Score with REF2015 — no FastRelax, no rotamer packing, no coordinate changes.
  4. Scatter MD energy (x) vs Rosetta total_score (y).

Backbone atom indices in the 93-atom all-atom array
(derived empirically from inter-residue gap analysis):
  Residue  0  TYR  N= 0  CA= 1  C=10  O=11
  Residue  1  TYR  N=12  CA=13  C=22  O=23
  Residue  2  ASP  N=24  CA=25  C=30  O=31
  Residue  3  PRO  N=32  CA=33  C=37  O=38
  Residue  4  GLU  N=39  CA=40  C=46  O=47
  Residue  5  THR  N=48  CA=49  C=53  O=54
  Residue  6  GLY  N=55  CA=56  C=57  O=58
  Residue  7  THR  N=59  CA=60  C=64  O=65
  Residue  8  TRP  N=66  CA=67  C=80  O=81
  Residue  9  GLY  N=83  CA=84  C=?   O=?   (estimated from CA-direction)

Usage:
    /vol/workspace/P4T1/miniforge3/envs/deeplife/bin/python \\
        scripts/scatter_md_vs_reu.py \\
        --test data_all_atom/test.npz \\
        --n 500 \\
        --save plots/scatter_md_vs_reu.png
"""

import sys, os, argparse, tempfile, time
import numpy as np
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# ── Backbone atom indices in the 93-atom all-atom array ───────────────────────
N_IDX  = [0,  12, 24, 32, 39, 48, 55, 59, 66, 83]
CA_IDX = [1,  13, 25, 33, 40, 49, 56, 60, 67, 84]
C_IDX  = [10, 22, 30, 37, 46, 53, 57, 64, 80]   # residues 0-8 (TRP8 C=atom80)
O_IDX  = [11, 23, 31, 38, 47, 54, 58, 65, 81]   # residues 0-8

SEQUENCE = "YYDPETGTWG"
AA3 = {
    'Y': 'TYR', 'D': 'ASP', 'P': 'PRO', 'E': 'GLU',
    'T': 'THR', 'G': 'GLY', 'W': 'TRP',
}
ATOM_NAMES_BB = [' N  ', ' CA ', ' C  '] * 10
ELEMENTS_BB   = ['N',    'C',    'C'  ] * 10


# ─────────────────────────────────────────────────────────────────────────────
# PYROSETTA INIT
# ─────────────────────────────────────────────────────────────────────────────

_pr_inited = False

def _init_pyrosetta():
    global _pr_inited
    if _pr_inited:
        return
    import pyrosetta
    pyrosetta.init(
        '-mute all '
        '-use_input_sc '
        '-ignore_unrecognized_res '
        '-ignore_zero_occupancy false',
        silent=True,
    )
    _pr_inited = True


# ─────────────────────────────────────────────────────────────────────────────
# BACKBONE EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

def extract_backbone(coords_93: np.ndarray) -> np.ndarray:
    """
    Extract backbone (N, CA, C) for all 10 residues from a 93-atom structure.

    Residues 0-8: use exact N, CA, C positions from the 93-atom array.
    Residue 9 (GLY): N and CA are known; C is estimated by extending along
                     the CA8→CA9 direction by 1.525 Å.

    Returns (30, 3) backbone coordinates.
    """
    bb = np.zeros((30, 3), dtype=np.float32)
    for j in range(9):
        bb[3*j]   = coords_93[N_IDX[j]]
        bb[3*j+1] = coords_93[CA_IDX[j]]
        bb[3*j+2] = coords_93[C_IDX[j]]

    # Residue 9 (GLY): N and CA exact, C estimated
    bb[27] = coords_93[N_IDX[9]]
    bb[28] = coords_93[CA_IDX[9]]
    ca8 = coords_93[CA_IDX[8]]
    ca9 = coords_93[CA_IDX[9]]
    d = ca9 - ca8
    dn = np.linalg.norm(d)
    bb[29] = ca9 + 1.525 * (d / max(dn, 1e-8))

    return bb


# ─────────────────────────────────────────────────────────────────────────────
# PDB WRITER (backbone N, CA, C — 30 atoms)
# ─────────────────────────────────────────────────────────────────────────────

def backbone_to_pdb_string(bb: np.ndarray, label: str = "") -> str:
    """Write 30-atom backbone PDB string (N, CA, C per residue)."""
    lines = [f"REMARK {label}\n"]
    seq = SEQUENCE
    atom_num = 0
    for j in range(10):
        rn = AA3.get(seq[j], 'GLY')
        for k, (aname, elem) in enumerate(zip(
            [' N  ', ' CA ', ' C  '],
            ['N', 'C', 'C']
        )):
            x, y, z = bb[3*j + k]
            atom_num += 1
            lines.append(
                f"ATOM  {atom_num:5d} {aname} {rn} A{j+1:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           {elem}\n"
            )
    lines.append("END\n")
    return ''.join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# SCORING — no packing, no relaxation
# ─────────────────────────────────────────────────────────────────────────────

def score_structure(coords_93: np.ndarray) -> float | None:
    """
    Build a pose from the backbone of this 93-atom structure and score it.

    No rotamer packing, no FastRelax, no coordinate changes.
    PyRosetta fills in O + sidechains at ideal geometry when loading the
    backbone-only PDB; the score reflects that idealized full-atom structure
    on top of the given backbone.
    """
    import pyrosetta

    bb = extract_backbone(coords_93)
    pdb_str = backbone_to_pdb_string(bb)

    with tempfile.NamedTemporaryFile(suffix='.pdb', mode='w', delete=False) as f:
        f.write(pdb_str)
        tmp = f.name

    try:
        pose = pyrosetta.pose_from_pdb(tmp)
        if pose.total_residue() == 0:
            return None
        sfxn = pyrosetta.get_fa_scorefxn()
        return float(sfxn(pose))
    except Exception:
        return None
    finally:
        os.unlink(tmp)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Scatter: MD energy label vs Rosetta REU (no relax)",
    )
    p.add_argument('--test', default='data_all_atom/test.npz',
                   help='Path to all-atom test.npz')
    p.add_argument('--n',    type=int, default=500,
                   help='Number of structures to score (randomly sampled)')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--save', default='plots/scatter_md_vs_reu.png',
                   help='Output scatter plot path')
    args = p.parse_args()

    np.random.seed(args.seed)
    _init_pyrosetta()

    # ── Load test set ──────────────────────────────────────────────────────
    d = np.load(args.test, allow_pickle=True)
    coords   = d['coords'].astype(np.float32)    # (N, 93, 3)
    energies = d['energies'].astype(np.float64)  # (N,)

    n_total = len(coords)
    n_score = min(args.n, n_total)
    idx     = np.random.choice(n_total, n_score, replace=False)
    print(f"\nAll-atom test set : {n_total:,} structures")
    print(f"Scoring           : {n_score} structures (seed={args.seed})")
    print(f"Output            : {args.save}\n")

    # ── Score ──────────────────────────────────────────────────────────────
    md_energies  = []
    reu_scores   = []
    n_failed     = 0

    t0 = time.time()
    for i, si in enumerate(idx):
        if i % 50 == 0:
            elapsed = time.time() - t0
            eta     = (elapsed / max(i, 1)) * (n_score - i)
            print(f"  {i:4d}/{n_score}  elapsed={elapsed:.0f}s  ETA={eta:.0f}s  "
                  f"failed={n_failed}", end='\r', flush=True)

        reu = score_structure(coords[si])
        if reu is None:
            n_failed += 1
            continue

        md_energies.append(float(energies[si]))
        reu_scores.append(reu)

    print(f"\n  Done: {len(reu_scores)} scored, {n_failed} failed  "
          f"({time.time()-t0:.0f}s total)")

    if len(reu_scores) < 10:
        print("Too few structures scored — aborting.")
        return

    md_e = np.array(md_energies)
    reu  = np.array(reu_scores)

    # ── Pearson correlation ────────────────────────────────────────────────
    corr = float(np.corrcoef(md_e, reu)[0, 1])
    print(f"\nPearson r = {corr:.4f}")
    print(f"MD energy range : {md_e.min():.2f} to {md_e.max():.2f}")
    print(f"REU range       : {reu.min():.1f} to {reu.max():.1f}")

    # ── Plot ───────────────────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from scipy import stats
    except ImportError:
        print("matplotlib/scipy not available — skipping plot.")
        return

    slope, intercept, r_val, p_val, se = stats.linregress(md_e, reu)

    fig, ax = plt.subplots(figsize=(7, 6))

    ax.scatter(md_e, reu,
               s=12, alpha=0.5, color='#4C72B0',
               linewidths=0, zorder=3,
               label=f'n = {len(md_e):,}')

    x_line = np.linspace(md_e.min(), md_e.max(), 200)
    ax.plot(x_line, slope * x_line + intercept,
            color='#C44E52', lw=1.8, zorder=4,
            label=f'fit  r = {corr:.3f}')

    ax.set_xlabel("MD energy label (dataset units)", fontsize=12)
    ax.set_ylabel("Rosetta REU  (REF2015, no relax)", fontsize=12)
    ax.set_title("All-atom test set: MD energy vs Rosetta REU", fontsize=13)

    textstr = (
        f"Pearson r = {corr:.4f}\n"
        f"p = {p_val:.2e}\n"
        f"n = {len(md_e):,}"
    )
    ax.text(0.97, 0.97, textstr,
            transform=ax.transAxes,
            fontsize=10, va='top', ha='right',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                      edgecolor='#aaaaaa', alpha=0.9))

    ax.legend(fontsize=10, loc='upper left')
    ax.grid(alpha=0.25)
    fig.tight_layout()

    Path(args.save).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.save, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved → {args.save}")
    plt.close(fig)


if __name__ == '__main__':
    main()
