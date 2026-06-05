"""
scripts/eval_v2/constants.py
=============================
Shared constants for the eval_v2 pipeline.
All physical ideals are derived from training-set statistics or standard PDB geometry.
"""

# ── Chignolin metadata ────────────────────────────────────────────────────────

CHIGNOLIN_SEQUENCE = "YYDPETGTWG"
AA3 = {
    'Y': 'TYR', 'D': 'ASP', 'P': 'PRO', 'E': 'GLU', 'T': 'THR',
    'G': 'GLY', 'W': 'TRP', 'A': 'ALA', 'K': 'LYS', 'R': 'ARG',
}
ATOM_NAMES_BACKBONE = [' N  ', ' CA ', ' C  '] * 10
ELEMENTS_BACKBONE   = ['N', 'C', 'C'] * 10

# ── Cα-only physics (from models/physics.py training-set statistics) ──────────

CA_IDEAL_BOND = 3.832   # Å  — mean Cα–Cα bond length
CA_IDEAL_COS  = 0.320   # cos(71.4°) — mean consecutive bond-vector angle
CA_CLASH_CUT  = 3.5     # Å  — non-bonded Cα–Cα clash cutoff (seq sep ≥ 2)

# ── Backbone physics (from models/backbone_physics.py) ───────────────────────

N_CA_IDEAL   = 1.460    # Å
CA_C_IDEAL   = 1.525    # Å
C_N_IDEAL    = 1.329    # Å  (peptide bond)
BB_CLASH_CUT = 2.0      # Å  (non-bonded, seq sep ≥ 3)

# Backbone bond-angle ideal cosines (cos of angle AT central atom):
#   N–Cα–C  ≈ 111.2°  cos = −0.364
#   Cα–C–N  ≈ 116.2°  cos = −0.440
#   C–N–Cα  ≈ 121.7°  cos = −0.526
BB_ANGLE_IDEALS_COS = {
    'N_CA_C':  -0.364,   # N as 1st atom (idx 0,3,6,…)
    'CA_C_N':  -0.440,   # Cα as 1st atom (idx 1,4,7,…)
    'C_N_CA':  -0.526,   # C as 1st atom (idx 2,5,8,…)
}
# Backbone bond-angle ideals in degrees (for human-readable output)
BB_ANGLE_IDEALS_DEG = {
    'N_CA_C':  111.2,
    'CA_C_N':  116.2,
    'C_N_CA':  121.7,
}

# ω dihedral (peptide bond planarity): Cα_i–C_i–N_{i+1}–Cα_{i+1}
OMEGA_IDEAL_DEG = 180.0   # trans (dominant); cis ~0° for rare cis-Pro

# ── All-atom (from models/physics_aa.py) ─────────────────────────────────────

AA_CLASH_CUT = 2.5   # Å  (non-bonded heavy-atom, seq sep ≥ 4)
N_ALL_ATOMS  = 93    # heavy atoms in Chignolin (no H)

# ── VdW radii for improved all-atom clash detection ───────────────────────────
# Bondi radii (Å); keyed by element symbol (first character of PDB atom name)
VDW_RADII = {'C': 1.70, 'N': 1.55, 'O': 1.52, 'S': 1.80}
VDW_CLASH_SCALE = 0.80   # clash when dist < (r_i + r_j) * VDW_CLASH_SCALE

# ── Plot colour palette (seaborn-style) ──────────────────────────────────────

MODEL_COLORS = [
    '#C44E52',   # red
    '#4C72B0',   # blue
    '#55A868',   # green
    '#8172B3',   # purple
    '#CCB974',   # yellow
    '#64B5CD',   # cyan
    '#DD8452',   # orange
    '#937860',   # brown
]
REF_COLOR    = '#333333'
REF_ALPHA    = 0.30

# ── Ramachandran region boundaries (simplified) ───────────────────────────────
# Favoured: within 40° of α-helix ideal (φ=−57°, ψ=−47°)
#           OR within 40° of β-sheet ideal (φ=−119°, ψ=+113°)
# Disallowed: outside 60° of BOTH regions
RAMA_ALPHA_PHI = -57.0
RAMA_ALPHA_PSI = -47.0
RAMA_BETA_PHI  = -119.0
RAMA_BETA_PSI  = 113.0
RAMA_FAVOURED_RADIUS  = 40.0   # degrees
RAMA_ALLOWED_RADIUS   = 60.0   # degrees

# ── PyRosetta score thresholds ────────────────────────────────────────────────
# "Plausible" = generated structure is within these Rosetta score bounds
# Calibrated from test-set FastRelax scores; can be tuned per dataset
ROSETTA_FA_REP_MAX    = 10.0   # fa_rep clash term
ROSETTA_RAMA_MAX      = 2.0    # rama_prepro term
# total_score threshold is set dynamically: ref_mean + N_REF_STD * ref_std
ROSETTA_TOTAL_N_STD   = 2.0
