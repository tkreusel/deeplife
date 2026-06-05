"""
scripts/plot_pca_by_tau.py
==========================
Single-panel conformational PCA coloured by temperature τ for presentation.

Generates structures at each τ, computes pairwise-distance fingerprints
(rotation/translation invariant), runs PCA, and plots PC1 vs PC2.

Usage:
    python scripts/plot_pca_by_tau.py \
        --checkpoint checkpoints/transformer_adaln_energy_physics/v1/best.pt \
        --test data/test.npz \
        --out  final_plots/pca_by_tau_adaln_ep \
        --n_per_tau 300 --guidance_scale 2.0 --steps 100
"""

import sys
import json
import argparse
import numpy as np
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))


def pdist_fingerprint(coords: np.ndarray) -> np.ndarray:
    """Upper-triangle pairwise Cα distances — rotation/translation invariant."""
    N = coords.shape[1]
    idx = np.triu_indices(N, k=1)
    diff = coords[:, idx[0]] - coords[:, idx[1]]        # (B, pairs, 3)
    return np.linalg.norm(diff, axis=-1)                 # (B, pairs)


def generate_at_tau(model, diffusion, tau, n, n_residues, coord_scale,
                    ddim_steps, guidance_scale, device, batch_size=256,
                    energy_mean=0.0, energy_std=1.0):
    from models.transformer_adaln_energy import AdaLNEnergyTransformerScoreNetwork
    model.eval()
    e_z_val = 4.0 * tau - 2.0
    T         = diffusion.T
    step_sz   = T // ddim_steps
    timesteps = list(range(0, T, step_sz))[::-1]

    all_x = []
    n_done = 0
    while n_done < n:
        bs  = min(batch_size, n - n_done)
        x   = torch.randn(bs, n_residues, 3, device=device)
        e_z = torch.full((bs,), e_z_val, device=device)

        with torch.no_grad():
            for i, t in enumerate(timesteps):
                t_b    = torch.full((bs,), t, device=device, dtype=torch.long)
                t_prev = timesteps[i + 1] if i + 1 < len(timesteps) else 0
                alpha_t    = diffusion.alphas_cumprod[t]
                alpha_prev = diffusion.alphas_cumprod[t_prev]

                eps_cond = model(x, t_b, energy_z=e_z)
                if guidance_scale != 1.0:
                    eps_unc  = model(x, t_b, energy_z=None)
                    eps = eps_unc + guidance_scale * (eps_cond - eps_unc)
                else:
                    eps = eps_cond

                x0_pred = (x - (1 - alpha_t).sqrt() * eps) / alpha_t.sqrt()
                x0_pred = x0_pred.clamp(-5, 5)
                x = alpha_prev.sqrt() * x0_pred + (1 - alpha_prev).sqrt() * eps

        x = (x - x.mean(dim=1, keepdim=True)) * coord_scale
        all_x.append(x.cpu().numpy())
        n_done += bs

    return np.concatenate(all_x)[:n]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint',    required=True)
    p.add_argument('--test',          default='data/test.npz')
    p.add_argument('--out',           required=True)
    p.add_argument('--temperatures',  nargs='+', type=float,
                   default=[0.0, 0.25, 0.5, 0.75, 1.0])
    p.add_argument('--n_per_tau',     type=int,   default=300)
    p.add_argument('--guidance_scale',type=float, default=2.0)
    p.add_argument('--steps',         type=int,   default=100)
    p.add_argument('--n_ref',         type=int,   default=500,
                   help='Reference test-set structures to include in PCA')
    p.add_argument('--seed',          type=int,   default=0)
    args = p.parse_args()

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')

    # ── Load model ────────────────────────────────────────────────────────────
    ckpt   = torch.load(args.checkpoint, map_location=device)
    config = ckpt['config']
    mc     = config['model']

    from models.transformer_adaln_energy import AdaLNEnergyTransformerScoreNetwork
    from models.diffusion import GaussianDiffusion

    model = AdaLNEnergyTransformerScoreNetwork(
        n_residues       = config['data']['n_residues'],
        hidden_dim       = mc['hidden_dim'],
        n_heads          = mc['n_heads'],
        n_layers         = mc['n_layers'],
        time_dim         = mc['time_dim'],
        dropout          = mc['dropout'],
        energy_drop_prob = mc.get('energy_drop_prob', 0.15),
    )
    shadow = {(k[len('_orig_mod.'):] if k.startswith('_orig_mod.') else k): v
              for k, v in ckpt['ema_shadow'].items()}
    model.load_state_dict(shadow)
    model = model.to(device).eval()

    dc        = config['diffusion']
    diffusion = GaussianDiffusion(T=dc['T'], schedule=dc['schedule']).to(device)

    coord_scale  = config['data'].get('coord_scale', 5.0)
    energy_mean  = ckpt.get('energy_mean', config['data'].get('energy_mean', 0.0))
    energy_std   = ckpt.get('energy_std',  config['data'].get('energy_std',  1.0))
    n_residues   = config['data']['n_residues']
    print(f'Loaded checkpoint  epoch={ckpt.get("epoch","?")}  '
          f'val_loss={ckpt.get("best_val_loss", float("nan")):.4f}')

    # ── Load reference ────────────────────────────────────────────────────────
    test_data  = np.load(args.test)
    ref_coords = test_data['coords'].astype(np.float32)
    centroids  = test_data['centroids']
    if centroids.ndim == 2:
        centroids = centroids[:, None, :]
    ref_coords = ref_coords - centroids
    idx_ref = np.random.choice(len(ref_coords), min(args.n_ref, len(ref_coords)),
                               replace=False)
    ref_coords = ref_coords[idx_ref]
    print(f'Reference: {len(ref_coords)} structures')

    # ── Generate at each τ ────────────────────────────────────────────────────
    tau_coords = {}
    for tau in args.temperatures:
        print(f'  τ={tau:.2f}  generating {args.n_per_tau} …', end='', flush=True)
        samples = generate_at_tau(
            model, diffusion, tau=tau, n=args.n_per_tau,
            n_residues=n_residues, coord_scale=coord_scale,
            ddim_steps=args.steps, guidance_scale=args.guidance_scale,
            device=device, energy_mean=energy_mean, energy_std=energy_std,
        )
        tau_coords[tau] = samples
        print(f'  Rg={np.sqrt(((samples - samples.mean(1, keepdims=True))**2).sum(-1).mean(-1)).mean():.2f} Å')

    # ── Pairwise-distance fingerprints + PCA ──────────────────────────────────
    ref_fp   = pdist_fingerprint(ref_coords)
    tau_fps  = {tau: pdist_fingerprint(c) for tau, c in tau_coords.items()}
    all_fp   = np.concatenate([ref_fp] + list(tau_fps.values()), axis=0)

    pca = PCA(n_components=2, random_state=args.seed)
    pca.fit(all_fp)

    ref_pcs  = pca.transform(ref_fp)
    tau_pcs  = {tau: pca.transform(fp) for tau, fp in tau_fps.items()}

    var = pca.explained_variance_ratio_
    print(f'PCA variance explained: PC1={var[0]*100:.1f}%  PC2={var[1]*100:.1f}%')

    # ── Plot ──────────────────────────────────────────────────────────────────
    # Colour palette: blue (cold) → red (hot)
    cmap   = plt.cm.coolwarm
    taus   = sorted(args.temperatures)
    colors = {tau: cmap(i / max(len(taus) - 1, 1)) for i, tau in enumerate(taus)}

    fig, ax = plt.subplots(figsize=(6, 4.5))

    # Reference: black filled dots
    ref_sc = ax.scatter(ref_pcs[:, 0], ref_pcs[:, 1],
                        marker='o', c='black', s=14, alpha=0.4, linewidths=0,
                        zorder=1)

    # Generated per τ — scatter coloured by continuous tau value
    all_gen_pcs = np.concatenate([tau_pcs[tau] for tau in taus])
    all_gen_tau = np.concatenate([[tau] * len(tau_pcs[tau]) for tau in taus])
    sc = ax.scatter(all_gen_pcs[:, 0], all_gen_pcs[:, 1],
                    c=all_gen_tau, cmap='coolwarm', vmin=0.0, vmax=1.0,
                    s=14, alpha=0.65, linewidths=0, zorder=2)

    # Colorbar for τ
    cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Temperature τ', fontsize=10)
    cbar.set_ticks(taus)
    cbar.ax.tick_params(labelsize=9)

    # Manual legend entry for reference only
    from matplotlib.lines import Line2D
    legend_handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor='black',
                             markersize=6, alpha=0.6, label='Reference (test set)')]
    ax.legend(handles=legend_handles, fontsize=9, framealpha=0.9, loc='upper left')

    ax.set_xlabel(f'PC 1 ({var[0]*100:.1f}% var)', fontsize=12)
    ax.set_ylabel(f'PC 2 ({var[1]*100:.1f}% var)', fontsize=12)
    ax.tick_params(labelsize=10)
    ax.grid(alpha=0.2)

    fig.tight_layout()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f'{out}.png', dpi=150, bbox_inches='tight')
    fig.savefig(f'{out}.svg', bbox_inches='tight')
    print(f'Saved {out}.png  {out}.svg')
    plt.close(fig)


if __name__ == '__main__':
    main()
