"""
scripts/quick_sample.py

Usage:
    python scripts/quick_sample.py --checkpoint checkpoints/baseline/v2/best.pt
    python scripts/quick_sample.py --checkpoint checkpoints/baseline/v2/best.pt --n 50 --steps 100
    python scripts/quick_sample.py --checkpoint checkpoints/baseline/v2/best.pt --n 20 --save_pdb outputs/v2_samples
    python scripts/quick_sample.py --checkpoint checkpoints/baseline/v2/best.pt --n 200 --compare data/test.npz
"""

import sys
import argparse
import numpy as np
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.diffusion import GaussianDiffusion
from scripts.train import build_model


# ── Sample ────────────────────────────────────────────────────────────────────

def sample(checkpoint_path: str, n: int = 10, ddim_steps: int = 100):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    ckpt   = torch.load(checkpoint_path, map_location=device)
    config = ckpt['config']

    model = build_model(config).to(device)
    model.load_state_dict(ckpt['ema_shadow'])
    model.eval()
    print(f"Loaded checkpoint — epoch {ckpt['epoch']+1}, "
          f"best val loss: {ckpt['best_val_loss']:.4f}")

    dc        = config['diffusion']
    diffusion = GaussianDiffusion(T=dc['T'], schedule=dc['schedule']).to(device)

    coord_scale = config['data'].get('coord_scale', 16.32)
    print(f"Coord scale: {coord_scale}")

    print(f"\nGenerating {n} structures with {ddim_steps} DDIM steps...")
    shape   = (n, config['data']['n_residues'], 3)
    samples = diffusion.ddim_sample(model, shape, device=device, ddim_steps=ddim_steps)

    # center
    samples = samples - samples.mean(dim=1, keepdim=True)

    # rescale back to Ångströms
    samples = samples * coord_scale

    samples = samples.cpu().numpy()   # (n, 10, 3)
    return samples, config


# ── Analysis ──────────────────────────────────────────────────────────────────

def analyse(samples: np.ndarray):
    n, n_res, _ = samples.shape
    print(f"\n{'='*55}")
    print(f"  Generated {n} structures, {n_res} residues each")
    print(f"{'='*55}")

    print(f"\n── Coordinate statistics ──")
    print(f"  Mean : {samples.mean():.3f} Å  (should be ~0)")
    print(f"  Std  : {samples.std():.3f} Å   (Chignolin ~16 Å)")
    print(f"  Min  : {samples.min():.3f} Å")
    print(f"  Max  : {samples.max():.3f} Å")

    # bond lengths
    diffs = np.diff(samples, axis=1)
    bl    = np.linalg.norm(diffs, axis=-1)    # (n, n_res-1)
    valid = np.abs(bl - 3.8) < 0.5

    print(f"\n── Bond lengths (Cα–Cα, ideal = 3.8 Å) ──")
    print(f"  Mean  : {bl.mean():.3f} Å")
    print(f"  Std   : {bl.std():.3f} Å")
    print(f"  Min   : {bl.min():.3f} Å")
    print(f"  Max   : {bl.max():.3f} Å")
    print(f"  Valid : {valid.mean()*100:.1f}%  (within ±0.5 Å of 3.8 Å)")

    print(f"\n  Per-bond mean lengths:")
    for i in range(bl.shape[1]):
        bar = '█' * max(1, int(bl[:, i].mean() / 0.3))
        print(f"    bond {i+1:2d}–{i+2:2d} : {bl[:, i].mean():.3f} ± {bl[:, i].std():.3f} Å  {bar}")

    # radius of gyration
    com = samples.mean(axis=1, keepdims=True)
    rg  = np.sqrt(((samples - com)**2).sum(axis=-1).mean(axis=-1))

    print(f"\n── Radius of gyration (Chignolin reference: ~5–6 Å) ──")
    print(f"  Mean : {rg.mean():.3f} Å")
    print(f"  Std  : {rg.std():.3f} Å")
    print(f"  Min  : {rg.min():.3f} Å")
    print(f"  Max  : {rg.max():.3f} Å")

    # pairwise distance matrix (first sample)
    print(f"\n── Pairwise Cα distance matrix (first sample, Å) ──")
    s0   = samples[0]
    diff = s0[:, None, :] - s0[None, :, :]
    dm   = np.linalg.norm(diff, axis=-1)
    header = '     ' + ''.join(f'{i+1:6d}' for i in range(n_res))
    print(f"  {header}")
    for i in range(n_res):
        row = ''.join(f'{dm[i,j]:6.1f}' for j in range(n_res))
        print(f"  {i+1:3d}  {row}")

    # verdict
    print(f"\n── Verdict ──")
    issues = []
    if bl.mean() < 2.5 or bl.mean() > 5.5:
        issues.append(f"bond length mean {bl.mean():.2f} Å — expected ~3.8")
    if valid.mean() < 0.5:
        issues.append(f"only {valid.mean()*100:.0f}% valid bonds — want >80%")
    if rg.mean() < 3.0:
        issues.append(f"structures too compact (Rg {rg.mean():.2f} Å)")
    if rg.mean() > 15.0:
        issues.append(f"structures too extended (Rg {rg.mean():.2f} Å)")
    if abs(samples.mean()) > 2.0:
        issues.append(f"not centered (mean {samples.mean():.2f} Å)")

    if issues:
        print(f"  Issues:")
        for issue in issues:
            print(f"    - {issue}")
    else:
        print(f"  Looks good.")

    return samples


# ── Compare to real data ──────────────────────────────────────────────────────

def compare_to_real(samples: np.ndarray, real_path: str):
    data      = np.load(real_path)
    real      = data['coords'].astype(np.float32)
    centroids = data['centroids']
    if centroids.ndim == 2:
        centroids = centroids[:, None, :]
    real = real - centroids

    real_bl = np.linalg.norm(np.diff(real,    axis=1), axis=-1)
    gen_bl  = np.linalg.norm(np.diff(samples, axis=1), axis=-1)

    real_rg = np.sqrt(((real    - real.mean(   axis=1, keepdims=True))**2).sum(-1).mean(-1))
    gen_rg  = np.sqrt(((samples - samples.mean(axis=1, keepdims=True))**2).sum(-1).mean(-1))

    real_valid = np.abs(real_bl - 3.8) < 0.5
    gen_valid  = np.abs(gen_bl  - 3.8) < 0.5

    print(f"\n── Comparison: generated vs real ──")
    print(f"  {'Metric':<28} {'Generated':>12} {'Real':>12}")
    print(f"  {'─'*54}")
    print(f"  {'Bond length mean (Å)':<28} {gen_bl.mean():>12.3f} {real_bl.mean():>12.3f}")
    print(f"  {'Bond length std (Å)':<28} {gen_bl.std():>12.3f}  {real_bl.std():>12.3f}")
    print(f"  {'Bond validity (%)':<28} {gen_valid.mean()*100:>11.1f}% {real_valid.mean()*100:>11.1f}%")
    print(f"  {'Rg mean (Å)':<28} {gen_rg.mean():>12.3f} {real_rg.mean():>12.3f}")
    print(f"  {'Rg std (Å)':<28} {gen_rg.std():>12.3f}  {real_rg.std():>12.3f}")


# ── Save PDBs ─────────────────────────────────────────────────────────────────

def save_pdbs(samples: np.ndarray, out_dir: str, n_save: int = 5):
    out      = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    sequence = "YYDPETGTWG"
    aa3      = {
        'Y':'TYR','D':'ASP','P':'PRO','E':'GLU','T':'THR',
        'G':'GLY','W':'TRP','A':'ALA','K':'LYS','R':'ARG',
    }

    for i, coords in enumerate(samples[:n_save]):
        lines = [f"REMARK  Generated sample {i+1}\n"]
        for j, (res, xyz) in enumerate(zip(sequence, coords)):
            x, y, z = xyz
            resname  = aa3.get(res, 'GLY')
            lines.append(
                f"ATOM  {j+1:5d}  CA  {resname} A{j+1:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C\n"
            )
        lines.append("END\n")
        with open(out / f"sample_{i+1:04d}.pdb", 'w') as f:
            f.writelines(lines)

    print(f"\nSaved {min(n_save, len(samples))} PDB files to {out_dir}")
    print(f"Open in PyMOL: pymol {out_dir}/*.pdb")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--n',        type=int, default=10,
                        help='Number of structures to generate')
    parser.add_argument('--steps',    type=int, default=100,
                        help='DDIM steps — more = slower but better quality')
    parser.add_argument('--save_pdb', type=str, default=None,
                        help='Directory to save PDB files')
    parser.add_argument('--compare',  type=str, default=None,
                        help='Path to real .npz for comparison, e.g. data/test.npz')
    args = parser.parse_args()

    samples, config = sample(args.checkpoint, n=args.n, ddim_steps=args.steps)
    analyse(samples)

    if args.compare:
        compare_to_real(samples, args.compare)

    if args.save_pdb:
        save_pdbs(samples, args.save_pdb)


if __name__ == '__main__':
    main()