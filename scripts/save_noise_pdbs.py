"""
scripts/save_noise_pdbs.py
===========================
Saves PDB files for presentation:
  1. One clean Cα structure from the training set.
  2. The same structure with 10 levels of DDPM noise applied,
     matching the cosine schedule used during training (T=500).

Output: outputs/noise_t000.pdb ... outputs/noise_t499.pdb
        outputs/clean.pdb

All coordinates are in Ångströms. Noise is applied in normalised
model space (coord_scale=5.0) and converted back for the PDB.
"""

import sys
import math
import numpy as np
import torch
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

OUT_DIR    = ROOT / "outputs"
TRAIN_PATH = ROOT / "data" / "train.npz"
COORD_SCALE = 5.0          # matches transformer_adaln_energy configs
T           = 500
SEQUENCE    = "YYDPETGTWG"
SEED        = 42

# ── Cosine noise schedule (identical to models/diffusion.py) ─────────────────

def cosine_alphas_cumprod(T: int, offset: float = 0.008) -> torch.Tensor:
    t = torch.linspace(0, T, T + 1)
    f = torch.cos((t / T + offset) / (1 + offset) * math.pi / 2) ** 2
    alpha_t = f / f[0]
    betas = (1 - alpha_t[1:] / alpha_t[:-1]).clamp(0, 0.999)
    alphas = 1.0 - betas
    return torch.cumprod(alphas, dim=0)   # (T,)


def q_sample(x0: torch.Tensor, t: int,
             sqrt_acp: torch.Tensor, sqrt_1macp: torch.Tensor,
             noise: torch.Tensor) -> torch.Tensor:
    """x_t = sqrt(ᾱ_t)·x_0 + sqrt(1−ᾱ_t)·ε"""
    return sqrt_acp[t] * x0 + sqrt_1macp[t] * noise


# ── PDB writer (Cα only) ──────────────────────────────────────────────────────

def write_ca_pdb(path: Path, coords_ang: np.ndarray, sequence: str):
    """
    coords_ang : (10, 3) in Ångströms
    sequence   : one-letter amino acid sequence (len 10)
    """
    AA1_TO_3 = {
        'A':'ALA','C':'CYS','D':'ASP','E':'GLU','F':'PHE',
        'G':'GLY','H':'HIS','I':'ILE','K':'LYS','L':'LEU',
        'M':'MET','N':'ASN','P':'PRO','Q':'GLN','R':'ARG',
        'S':'SER','T':'THR','V':'VAL','W':'TRP','Y':'TYR',
    }
    lines = []
    for i, (aa, xyz) in enumerate(zip(sequence, coords_ang)):
        res3 = AA1_TO_3.get(aa, 'UNK')
        x, y, z = xyz
        lines.append(
            f"ATOM  {i+1:5d}  CA  {res3} A{i+1:4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C"
        )
    lines.append("END")
    path.write_text("\n".join(lines) + "\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    OUT_DIR.mkdir(exist_ok=True)

    # Load one training structure and centre it
    data      = np.load(TRAIN_PATH)
    coords_raw = data['coords'][0]           # (10, 3) Å
    centroid   = coords_raw.mean(axis=0)
    coords_ang = coords_raw - centroid       # centred Å

    # Save the clean structure
    write_ca_pdb(OUT_DIR / "clean.pdb", coords_ang, SEQUENCE)
    print(f"Saved clean.pdb")

    # Normalise to model space
    x0 = torch.tensor(coords_ang / COORD_SCALE, dtype=torch.float32)   # (10, 3)

    # Precompute schedule
    acp      = cosine_alphas_cumprod(T)
    sqrt_acp  = acp.sqrt()
    sqrt_1macp = (1.0 - acp).sqrt()

    # Fix one noise sample so all timesteps corrupt the same structure the same way
    noise = torch.randn_like(x0)

    # 10 evenly spaced timesteps: t=0 (almost clean) → t=499 (pure noise)
    timesteps = [int(round(i * (T - 1) / 9)) for i in range(10)]

    for t_val in timesteps:
        x_t     = q_sample(x0, t_val, sqrt_acp, sqrt_1macp, noise)
        coords_t = (x_t.numpy() * COORD_SCALE)   # back to Å

        fname = OUT_DIR / f"noise_t{t_val:03d}.pdb"
        write_ca_pdb(fname, coords_t, SEQUENCE)

        signal_frac = acp[t_val].item()
        print(f"  t={t_val:3d}  ᾱ_t={signal_frac:.4f}  → {fname.name}")

    print(f"\nAll files written to {OUT_DIR}/")


if __name__ == "__main__":
    main()
