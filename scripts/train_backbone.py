"""
scripts/train_backbone.py
==========================
Training script for BackboneTransformerScoreNetwork (DDPM + energy CFG).

Operates on data_backbone/{train,val,test}.npz — 30-atom backbone coordinates
(N, Cα, C × 10 residues) extracted by scripts/prepare_backbone_data.py.

Usage:
    # prepare data first (one-time):
    python scripts/prepare_backbone_data.py

    # CPU sanity-check (~2 min):
    python scripts/train_backbone.py --config configs/backbone_transformer_local.yaml

    # Full GPU run:
    python scripts/train_backbone.py --config configs/backbone_transformer.yaml

    # Resume:
    python scripts/train_backbone.py --config configs/backbone_transformer.yaml \
        --resume checkpoints/backbone_transformer/v1/epoch_0500.pt

model_type must be 'backbone_transformer'.
"""

import os
import sys
import json
import copy
import argparse
import yaml
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.dataset       import get_dataloaders
from models.diffusion   import GaussianDiffusion
from models.backbone_transformer import BackboneTransformerScoreNetwork
from models.backbone_physics     import BackbonePhysics


# ── EMA ───────────────────────────────────────────────────────────────────────

class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.model  = model
        self.decay  = decay
        self.shadow = copy.deepcopy(model.state_dict())

    @torch.no_grad()
    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = (
                    self.decay * self.shadow[name] + (1.0 - self.decay) * param.data
                )

    def apply_shadow(self):
        orig = copy.deepcopy(self.model.state_dict())
        self.model.load_state_dict(self.shadow)
        return orig

    def restore(self, orig):
        self.model.load_state_dict(orig)


# ── Versioning ────────────────────────────────────────────────────────────────

def get_version_dir(base: Path) -> Path:
    existing = sorted(
        [d for d in base.iterdir() if d.is_dir() and d.name.startswith('v')]
    ) if base.exists() else []
    nxt = int(existing[-1].name[1:]) + 1 if existing else 1
    d = base / f'v{nxt}'
    d.mkdir(parents=True, exist_ok=True)
    return d


# ── Energy statistics ─────────────────────────────────────────────────────────

def compute_energy_stats(train_loader):
    print("Computing energy statistics…")
    all_e = torch.cat([b['energies'] for b in train_loader]).float()
    mean, std = all_e.mean().item(), all_e.std().item()
    print(f"  mean={mean:.4f}  std={std:.6f}  N={len(all_e):,}")
    return mean, std


# ── Training ──────────────────────────────────────────────────────────────────

def train(config: dict, resume_path: str = None):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    assert config['model_type'] == 'backbone_transformer', \
        f"This script only supports model_type='backbone_transformer', got {config['model_type']!r}"

    # ── Versioning ────────────────────────────────────────────────────────────
    base = Path(config['paths']['checkpoint_dir'])
    if resume_path:
        ckpt_dir = Path(resume_path).parent
        print(f"Resuming: {ckpt_dir}")
    else:
        ckpt_dir = get_version_dir(base)
        print(f"New run:  {ckpt_dir}")

    # ── Data ──────────────────────────────────────────────────────────────────
    train_loader, val_loader, _ = get_dataloaders(config)
    print(f"Train batches: {len(train_loader)}  Val: {len(val_loader)}")
    print(f"Coords shape: {next(iter(train_loader))['coords'].shape}")  # (B, 30, 3)

    # ── Energy stats ──────────────────────────────────────────────────────────
    energy_mean, energy_std = compute_energy_stats(train_loader)
    config['data']['energy_mean'] = energy_mean
    config['data']['energy_std']  = energy_std
    with open(ckpt_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f)

    # ── Model ─────────────────────────────────────────────────────────────────
    mc    = config['model']
    model = BackboneTransformerScoreNetwork(
        n_residues       = config['data']['n_residues'],
        hidden_dim       = mc['hidden_dim'],
        n_heads          = mc['n_heads'],
        n_layers         = mc['n_layers'],
        time_dim         = mc['time_dim'],
        dropout          = mc['dropout'],
        energy_drop_prob = mc.get('energy_drop_prob', 0.15),
    ).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # ── Diffusion ─────────────────────────────────────────────────────────────
    dc        = config['diffusion']
    diffusion = GaussianDiffusion(T=dc['T'], schedule=dc['schedule']).to(device)

    # ── Physics ───────────────────────────────────────────────────────────────
    tc          = config['training']
    phys_weight = tc.get('physics_weight', 0.0)
    physics     = None
    if phys_weight > 0.0:
        pc      = config.get('physics', {})
        physics = BackbonePhysics(
            bond_weight  = pc.get('bond_weight',  2.0),
            clash_weight = pc.get('clash_weight', 0.1),
            angle_weight = pc.get('angle_weight', 0.5),
            coord_scale  = config['data'].get('coord_scale', 5.0),
        )
        print(f"Physics: {physics}  λ={phys_weight}")
    else:
        print("Physics: disabled")

    # ── Optimiser ─────────────────────────────────────────────────────────────
    optimizer    = AdamW(model.parameters(), lr=tc['lr'], weight_decay=1e-4)
    total_steps  = tc['n_epochs'] * len(train_loader)
    warmup_steps = tc.get('warmup_steps', 1000)
    warmup    = LinearLR(optimizer, start_factor=0.01, end_factor=1.0,
                         total_iters=warmup_steps)
    cosine    = CosineAnnealingLR(optimizer, T_max=max(total_steps - warmup_steps, 1))
    scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine],
                             milestones=[warmup_steps])

    ema = EMA(model, decay=tc.get('ema_decay', 0.9999))

    # ── Resume ────────────────────────────────────────────────────────────────
    start_epoch, best_val_loss, global_step = 0, float('inf'), 0
    if resume_path:
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        ema.shadow    = ckpt['ema_shadow']
        start_epoch   = ckpt['epoch'] + 1
        global_step   = ckpt.get('global_step', start_epoch * len(train_loader))
        best_val_loss = ckpt.get('best_val_loss', float('inf'))
        if 'energy_mean' in ckpt:
            energy_mean = ckpt['energy_mean']
            energy_std  = ckpt['energy_std']
        print(f"Resumed at epoch {start_epoch}")

    log_path = ckpt_dir / 'log.jsonl'
    def log(entry):
        with open(log_path, 'a') as f:
            f.write(json.dumps(entry) + '\n')

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(start_epoch, tc['n_epochs']):
        model.train()
        epoch_loss, n_batches = 0.0, 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{tc['n_epochs']}")
        for batch in pbar:
            x0    = batch['coords'].to(device)          # (B, 30, 3)
            e_raw = batch['energies'].to(device)
            e_z   = (e_raw - energy_mean) / energy_std

            B = x0.shape[0]
            t          = torch.randint(0, diffusion.T, (B,), device=device)
            x_t, noise = diffusion.q_sample(x0, t)
            noise_pred = model(x_t, t, energy_z=e_z)
            loss       = (noise - noise_pred).pow(2).mean()

            if phys_weight > 0.0 and physics is not None:
                x0_pred = diffusion._predict_x0_from_noise(x_t, t, noise_pred)
                snr_w   = diffusion.alphas_cumprod[t].to(device)
                loss    = loss + phys_weight * (snr_w * physics(x0_pred)).mean()

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), tc.get('grad_clip', 1.0))
            optimizer.step()
            scheduler.step()
            ema.update()

            epoch_loss  += loss.item()
            n_batches   += 1
            global_step += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}",
                             lr=f"{scheduler.get_last_lr()[0]:.2e}")

        mean_train = epoch_loss / n_batches
        mean_val   = None

        # ── Validation ────────────────────────────────────────────────────────
        if (epoch + 1) % tc.get('val_every', 10) == 0:
            orig = ema.apply_shadow()
            model.eval()
            val_loss, n_val = 0.0, 0
            with torch.no_grad():
                for batch in val_loader:
                    x0    = batch['coords'].to(device)
                    e_raw = batch['energies'].to(device)
                    e_z   = (e_raw - energy_mean) / energy_std
                    B     = x0.shape[0]
                    t          = torch.randint(0, diffusion.T, (B,), device=device)
                    x_t, noise = diffusion.q_sample(x0, t)
                    noise_pred = model(x_t, t, energy_z=e_z)
                    l          = (noise - noise_pred).pow(2).mean()
                    if phys_weight > 0.0 and physics is not None:
                        x0_pred = diffusion._predict_x0_from_noise(x_t, t, noise_pred)
                        snr_w   = diffusion.alphas_cumprod[t].to(device)
                        l = l + phys_weight * (snr_w * physics(x0_pred)).mean()
                    val_loss += l.item()
                    n_val    += 1
            mean_val = val_loss / n_val
            ema.restore(orig)
            print(f"\nEpoch {epoch+1:4d} | train: {mean_train:.4f} | val: {mean_val:.4f}")

            if mean_val < best_val_loss:
                best_val_loss = mean_val
                torch.save({
                    'epoch': epoch, 'global_step': global_step,
                    'model': model.state_dict(), 'ema_shadow': ema.shadow,
                    'optimizer': optimizer.state_dict(),
                    'best_val_loss': best_val_loss,
                    'energy_mean': energy_mean, 'energy_std': energy_std,
                    'config': config,
                }, ckpt_dir / 'best.pt')
                print(f"  -> best (val: {best_val_loss:.4f})")

        log({'epoch': epoch + 1, 'global_step': global_step,
             'train_loss': mean_train, 'val_loss': mean_val,
             'lr': scheduler.get_last_lr()[0]})

        if (epoch + 1) % tc.get('save_every', 50) == 0:
            torch.save({
                'epoch': epoch, 'global_step': global_step,
                'model': model.state_dict(), 'ema_shadow': ema.shadow,
                'optimizer': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'energy_mean': energy_mean, 'energy_std': energy_std,
                'config': config,
            }, ckpt_dir / f'epoch_{epoch+1:04d}.pt')

    print(f"\nDone. Best val: {best_val_loss:.4f}  |  {ckpt_dir}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--resume', default=None)
    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)
    print(f"Experiment: {config['experiment_name']}")
    train(config, resume_path=args.resume)


if __name__ == '__main__':
    main()
