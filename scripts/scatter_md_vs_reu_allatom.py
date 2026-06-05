"""
scripts/scatter_md_vs_reu_allatom.py
======================================
Scatter plot: MD energy label vs Rosetta REU using ALL heavy atoms from
the all-atom test set. No relaxation, no packing, no coordinate changes.

Atom-name mapping derived from per-residue consecutive-distance analysis
(bond lengths identify atom identity within each residue block):

  res0 TYR  atoms  0-11 : N CA CB CG CD1 CE1 CZ OH  CD2 CE2 C  O
  res1 TYR  atoms 12-23 : N CA CB CG CD1 CE1 CZ OH  CD2 CE2 C  O
  res2 ASP  atoms 24-31 : N CA CB CG OD1 OD2 C  O
  res3 PRO  atoms 32-38 : N CA CD CG CB  C   O          (ring ordering)
  res4 GLU  atoms 39-47 : N CA CB CG CD  OE1 OE2 C  O
  res5 THR  atoms 48-54 : N CA CB OG1 CG2 C  O
  res6 GLY  atoms 55-58 : N CA C  O
  res7 THR  atoms 59-65 : N CA CB OG1 CG2 C  O
  res8 TRP  atoms 66-81 : N CA CB CG CD1 NE1 CE2 CZ2 CH2 CZ3 CD2 CE3 C O
                           atoms 78-79 are unidentified extras → skipped
  res9 TYR  atoms 83-92 : N CA CB CG CD1 CE1 CZ OH  CD2 CE2
                           (backbone C/O absent from array → estimated)
  atom 82 between res8/res9 is unidentified → skipped

Usage:
    /vol/workspace/P4T1/miniforge3/envs/deeplife/bin/python \\
        scripts/scatter_md_vs_reu_allatom.py \\
        --test data_all_atom/test.npz \\
        --n 9954 \\
        --save plots/scatter_md_vs_reu_allatom.png
"""

import sys, os, argparse, tempfile, time
import numpy as np
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# ── Full per-atom mapping: (res_idx, res_name, pdb_atom_name, array_idx) ─────
# Atoms skipped: 78, 79 (unknown TRP extras), 82 (isolated), 93-94 (missing)
ATOM_MAP = [
    # TYR 0
    (0,'TYR',' N  ', 0), (0,'TYR',' CA ', 1), (0,'TYR',' CB ', 2),
    (0,'TYR',' CG ', 3), (0,'TYR',' CD1', 4), (0,'TYR',' CE1', 5),
    (0,'TYR',' CZ ', 6), (0,'TYR',' OH ', 7), (0,'TYR',' CD2', 8),
    (0,'TYR',' CE2', 9), (0,'TYR',' C  ',10), (0,'TYR',' O  ',11),
    # TYR 1
    (1,'TYR',' N  ',12), (1,'TYR',' CA ',13), (1,'TYR',' CB ',14),
    (1,'TYR',' CG ',15), (1,'TYR',' CD1',16), (1,'TYR',' CE1',17),
    (1,'TYR',' CZ ',18), (1,'TYR',' OH ',19), (1,'TYR',' CD2',20),
    (1,'TYR',' CE2',21), (1,'TYR',' C  ',22), (1,'TYR',' O  ',23),
    # ASP 2
    (2,'ASP',' N  ',24), (2,'ASP',' CA ',25), (2,'ASP',' CB ',26),
    (2,'ASP',' CG ',27), (2,'ASP',' OD1',28), (2,'ASP',' OD2',29),
    (2,'ASP',' C  ',30), (2,'ASP',' O  ',31),
    # PRO 3  (ring ordering: N CA CD CG CB C O)
    (3,'PRO',' N  ',32), (3,'PRO',' CA ',33), (3,'PRO',' CD ',34),
    (3,'PRO',' CG ',35), (3,'PRO',' CB ',36), (3,'PRO',' C  ',37),
    (3,'PRO',' O  ',38),
    # GLU 4
    (4,'GLU',' N  ',39), (4,'GLU',' CA ',40), (4,'GLU',' CB ',41),
    (4,'GLU',' CG ',42), (4,'GLU',' CD ',43), (4,'GLU',' OE1',44),
    (4,'GLU',' OE2',45), (4,'GLU',' C  ',46), (4,'GLU',' O  ',47),
    # THR 5
    (5,'THR',' N  ',48), (5,'THR',' CA ',49), (5,'THR',' CB ',50),
    (5,'THR',' OG1',51), (5,'THR',' CG2',52), (5,'THR',' C  ',53),
    (5,'THR',' O  ',54),
    # GLY 6
    (6,'GLY',' N  ',55), (6,'GLY',' CA ',56), (6,'GLY',' C  ',57),
    (6,'GLY',' O  ',58),
    # THR 7
    (7,'THR',' N  ',59), (7,'THR',' CA ',60), (7,'THR',' CB ',61),
    (7,'THR',' OG1',62), (7,'THR',' CG2',63), (7,'THR',' C  ',64),
    (7,'THR',' O  ',65),
    # TRP 8 — 14 standard atoms; array indices 78,79 are unidentified extras
    (8,'TRP',' N  ',66), (8,'TRP',' CA ',67), (8,'TRP',' CB ',68),
    (8,'TRP',' CG ',69), (8,'TRP',' CD1',70), (8,'TRP',' NE1',71),
    (8,'TRP',' CE2',72), (8,'TRP',' CZ2',73), (8,'TRP',' CH2',74),
    (8,'TRP',' CZ3',75), (8,'TRP',' CD2',76), (8,'TRP',' CE3',77),
    # skip array 78, 79
    (8,'TRP',' C  ',80), (8,'TRP',' O  ',81),
    # skip array 82 (isolated atom between res8 and res9)
    # TYR 9 — 10 sidechain+backbone atoms visible; C and O must be estimated
    (9,'TYR',' N  ',83), (9,'TYR',' CA ',84), (9,'TYR',' CB ',85),
    (9,'TYR',' CG ',86), (9,'TYR',' CD1',87), (9,'TYR',' CE1',88),
    (9,'TYR',' CZ ',89), (9,'TYR',' OH ',90), (9,'TYR',' CD2',91),
    (9,'TYR',' CE2',92),
    # C and O for res9 appended separately (estimated)
]

_PR_INITED = False

def _init_pyrosetta():
    global _PR_INITED
    if _PR_INITED:
        return
    import pyrosetta
    pyrosetta.init(
        '-mute all '
        '-use_input_sc '
        '-ignore_unrecognized_res '
        '-ignore_zero_occupancy false',
        silent=True,
    )
    _PR_INITED = True


def _estimate_c_o(ca9: np.ndarray, ca8: np.ndarray) -> tuple:
    """Place backbone C and O for the C-terminal residue using ideal geometry."""
    d = ca9 - ca8
    dn = np.linalg.norm(d)
    d_hat = d / max(dn, 1e-8)
    c = ca9 + 1.525 * d_hat
    # O: ~120° from the CA-C bond, in a plane estimated by cross product
    perp = np.cross(d_hat, np.array([0., 1., 0.], dtype=np.float32))
    pn = np.linalg.norm(perp)
    if pn < 1e-6:
        perp = np.cross(d_hat, np.array([1., 0., 0.], dtype=np.float32))
        pn = np.linalg.norm(perp)
    perp /= pn
    # C=O direction: ~120° from C-CA, in the plane
    import math
    angle = math.radians(120.0)
    o_dir = -math.cos(math.radians(60.0)) * d_hat + math.sin(math.radians(60.0)) * perp
    o = c + 1.229 * o_dir
    return c.astype(np.float32), o.astype(np.float32)


def build_all_atom_pdb(coords_93: np.ndarray, label: str = "") -> str:
    """Build a full-atom PDB string from a 93-atom all-atom structure."""
    lines = [f"REMARK {label}\n"]
    atom_num = 0

    # Write all mapped atoms
    for res_idx, res_name, atom_name, arr_idx in ATOM_MAP:
        x, y, z = coords_93[arr_idx]
        atom_num += 1
        elem = atom_name.strip()[0]
        lines.append(
            f"ATOM  {atom_num:5d} {atom_name} {res_name} A{res_idx+1:4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           {elem}\n"
        )

    # Estimated backbone C and O for C-terminal TYR (res9)
    ca9 = coords_93[84]
    ca8 = coords_93[67]
    c_est, o_est = _estimate_c_o(ca9, ca8)
    for aname, elem, xyz in [(' C  ', 'C', c_est), (' O  ', 'O', o_est)]:
        atom_num += 1
        x, y, z = xyz
        lines.append(
            f"ATOM  {atom_num:5d} {aname} TYR A{10:4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           {elem}\n"
        )

    lines.append("END\n")
    return ''.join(lines)


def score_all_atom(coords_93: np.ndarray) -> float | None:
    """
    Score a 93-atom structure in PyRosetta with no relaxation or packing.
    Writes a full-atom PDB → loads → scores directly.
    """
    import pyrosetta

    pdb_str = build_all_atom_pdb(coords_93)
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


def main():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Scatter: MD energy vs Rosetta REU, all heavy atoms, no relax",
    )
    p.add_argument('--test', default='data_all_atom/test.npz')
    p.add_argument('--n',   type=int, default=9954,
                   help='Structures to score (use all by default)')
    p.add_argument('--seed',type=int, default=42)
    p.add_argument('--save',default='plots/scatter_md_vs_reu_allatom.png')
    args = p.parse_args()

    np.random.seed(args.seed)
    _init_pyrosetta()

    d       = np.load(args.test, allow_pickle=True)
    coords  = d['coords'].astype(np.float32)
    energies= d['energies'].astype(np.float64)

    n_total = len(coords)
    n_score = min(args.n, n_total)
    idx     = np.random.choice(n_total, n_score, replace=False)

    print(f"\nAll-atom test set : {n_total:,} structures")
    print(f"Scoring           : {n_score} (no relax, all heavy atoms)")
    print(f"Output            : {args.save}\n")

    md_e, reu, n_failed = [], [], 0
    t0 = time.time()

    for i, si in enumerate(idx):
        if i % 100 == 0:
            el  = time.time() - t0
            eta = (el / max(i, 1)) * (n_score - i)
            print(f"  {i:5d}/{n_score}  {el:.0f}s elapsed  ETA {eta:.0f}s  "
                  f"failed={n_failed}", end='\r', flush=True)

        score = score_all_atom(coords[si])
        if score is None:
            n_failed += 1
            continue
        md_e.append(float(energies[si]))
        reu.append(score)

    print(f"\n  Done: {len(reu)} scored, {n_failed} failed  "
          f"({time.time()-t0:.0f}s)")

    if len(reu) < 10:
        print("Too few — aborting."); return

    md_e = np.array(md_e)
    reu  = np.array(reu)

    from scipy import stats
    r, pval = stats.pearsonr(md_e, reu)
    print(f"\nPearson r = {r:.4f}  p = {pval:.2e}")
    print(f"MD energy : {md_e.min():.3f} → {md_e.max():.3f}")
    print(f"REU       : {reu.min():.1f} → {reu.max():.1f}")

    try:
        import matplotlib; matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib missing"); return

    slope, intercept, *_ = stats.linregress(md_e, reu)
    fig, ax = plt.subplots(figsize=(7, 6))

    ax.scatter(md_e, reu, s=8, alpha=0.35, color='#4C72B0',
               linewidths=0, zorder=3, label=f'n = {len(md_e):,}')
    xl = np.linspace(md_e.min(), md_e.max(), 200)
    ax.plot(xl, slope * xl + intercept, color='#C44E52',
            lw=2, zorder=4, label=f'fit  r = {r:.3f}')

    ax.set_xlabel("MD energy label (dataset units)", fontsize=12)
    ax.set_ylabel("Rosetta REU  (REF2015, all heavy atoms, no relax)", fontsize=12)
    ax.set_title("All-atom test set: MD energy vs Rosetta REU\n"
                 "(all heavy-atom positions from MD, no relaxation)", fontsize=12)
    ax.text(0.97, 0.97,
            f"Pearson r = {r:.4f}\np = {pval:.2e}\nn = {len(md_e):,}",
            transform=ax.transAxes, fontsize=10, va='top', ha='right',
            bbox=dict(boxstyle='round,pad=0.4', fc='white', ec='#aaaaaa', alpha=0.9))
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(alpha=0.25)
    fig.tight_layout()

    Path(args.save).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.save, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved → {args.save}")
    plt.close(fig)


if __name__ == '__main__':
    main()
