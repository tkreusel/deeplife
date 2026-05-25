"""
scripts/train.py

Usage:
    python scripts/train.py --config configs/baseline.yaml
    python scripts/train.py --config configs/baseline.yaml --resume checkpoints/baseline/v1/epoch_0050.pt
"""

import os
import sys
import json
import argparse
import yaml
import copy
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.dataset import get_dataloaders
from models.diffusion import GaussianDiffusion
from models.baseline import MLPScoreNetwork, TransformerScoreNetwork


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
                    self.decay * self.shadow[name]
                    + (1.0 - self.decay) * param.data
                )

    def apply_shadow(self):
        original = copy.deepcopy(self.model.state_dict())
        self.model.load_state_dict(self.shadow)
        return original

    def restore(self, original):
        self.model.load_state_dict(original)


# ── Versioning ────────────────────────────────────────────────────────────────

def get_version_dir(base_ckpt_dir: Path) -> Path:
    """
    Finds the next available version directory under base_ckpt_dir.
    e.g. checkpoints/baseline/v1, v2, v3 ...
    If resuming, pass the checkpoint path directly — this is only
    called for fresh runs.
    """
    existing = sorted(
        [d for d in base_ckpt_dir.iterdir() if d.is_dir() and d.name.startswith('v')]
    ) if base_ckpt_dir.exists() else []

    if not existing:
        next_version = 1
    else:
        # parse the highest version number and increment
        last = existing[-1].name  # e.g. 'v3'
        try:
            next_version = int(last[1:]) + 1
        except ValueError:
            next_version = len(existing) + 1

    version_dir = base_ckpt_dir / f'v{next_version}'
    version_dir.mkdir(parents=True, exist_ok=True)
    return version_dir


# ── Model factory ─────────────────────────────────────────────────────────────

def build_model(config: dict) -> nn.Module:
    mc         = config['model']
    model_type = config['model_type']
    n_res      = config['data']['n_residues']

    if model_type == 'mlp':
        return MLPScoreNetwork(
            n_residues = n_res,
            hidden_dim = mc['hidden_dim'],
            n_layers   = mc['n_layers'],
            time_dim   = mc['time_dim'],
            dropout    = mc['dropout'],
        )
    elif model_type == 'transformer':
        return TransformerScoreNetwork(
            n_residues = n_res,
            hidden_dim = mc['hidden_dim'],
            n_heads    = mc['n_heads'],
            n_layers   = mc['n_layers'],
            time_dim   = mc['time_dim'],
            dropout    = mc['dropout'],
        )
    else:
        raise ValueError(f"Unknown model_type '{model_type}'. Choose 'mlp' or 'transformer'.")


# ── Training ──────────────────────────────────────────────────────────────────

def train(config: dict, resume_path: str = None):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # ── versioning ────────────────────────────────────────────────────────────
    base_ckpt_dir = Path(config['paths']['checkpoint_dir'])

    if resume_path is not None:
        # resuming — use the same version directory the checkpoint lives in
        ckpt_dir = Path(resume_path).parent
        print(f"Resuming into existing version: {ckpt_dir}")
    else:
        # fresh run — create next version
        ckpt_dir = get_version_dir(base_ckpt_dir)
        print(f"Starting new version: {ckpt_dir}")

    # save config so you always know what produced this version
    with open(ckpt_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f)

    # ── data ──────────────────────────────────────────────────────────────────
    train_loader, val_loader, _ = get_dataloaders(config)
    print(f"Batches per epoch — train: {len(train_loader)}, val: {len(val_loader)}")

    # ── model ─────────────────────────────────────────────────────────────────
    model = build_model(config).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {config['model_type']}  |  Parameters: {n_params:,}")

    # ── diffusion ─────────────────────────────────────────────────────────────
    dc        = config['diffusion']
    diffusion = GaussianDiffusion(T=dc['T'], schedule=dc['schedule']).to(device)

    # ── optimiser ─────────────────────────────────────────────────────────────
    tc        = config['training']
    optimizer = AdamW(model.parameters(), lr=tc['lr'], weight_decay=1e-4)

    total_steps  = tc['n_epochs'] * len(train_loader)
    warmup_steps = tc.get('warmup_steps', 500)

    warmup    = LinearLR(optimizer, start_factor=0.01, end_factor=1.0,
                         total_iters=warmup_steps)
    cosine    = CosineAnnealingLR(optimizer, T_max=max(total_steps - warmup_steps, 1))
    scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine],
                             milestones=[warmup_steps])

    # ── EMA ───────────────────────────────────────────────────────────────────
    ema = EMA(model, decay=tc.get('ema_decay', 0.9999))

    # ── resume ────────────────────────────────────────────────────────────────
    start_epoch   = 0
    best_val_loss = float('inf')
    global_step   = 0

    if resume_path is not None:
        print(f"Loading checkpoint: {resume_path}")
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        ema.shadow    = ckpt['ema_shadow']
        start_epoch   = ckpt['epoch'] + 1
        global_step   = ckpt.get('global_step', start_epoch * len(train_loader))
        best_val_loss = ckpt.get('best_val_loss', float('inf'))
        print(f"Resumed at epoch {start_epoch}, global step {global_step}")

    # ── log file ──────────────────────────────────────────────────────────────
    log_path = ckpt_dir / 'log.jsonl'

    def log(entry: dict):
        with open(log_path, 'a') as f:
            f.write(json.dumps(entry) + '\n')

    # ── training loop ─────────────────────────────────────────────────────────
    for epoch in range(start_epoch, tc['n_epochs']):
        model.train()
        epoch_loss = 0.0
        n_batches  = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{tc['n_epochs']}")

        for batch in pbar:
            x0 = batch['coords'].to(device)     # (B, 10, 3)

            loss = diffusion.training_loss(model, x0)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), tc.get('grad_clip', 1.0))
            optimizer.step()
            scheduler.step()
            ema.update()

            epoch_loss  += loss.item()
            n_batches   += 1
            global_step += 1

            pbar.set_postfix(
                loss = f"{loss.item():.4f}",
                lr   = f"{scheduler.get_last_lr()[0]:.2e}",
            )

        mean_train_loss = epoch_loss / n_batches

        # ── validation ────────────────────────────────────────────────────────
        mean_val_loss = None

        if (epoch + 1) % tc.get('val_every', 10) == 0:
            original = ema.apply_shadow()
            model.eval()

            val_loss = 0.0
            n_val    = 0

            with torch.no_grad():
                for batch in val_loader:
                    x0        = batch['coords'].to(device)
                    val_loss += diffusion.training_loss(model, x0).item()
                    n_val    += 1

            mean_val_loss = val_loss / n_val
            ema.restore(original)

            print(
                f"\nEpoch {epoch+1:4d} | "
                f"train: {mean_train_loss:.4f} | "
                f"val: {mean_val_loss:.4f} | "
                f"step: {global_step}"
            )

            if mean_val_loss < best_val_loss:
                best_val_loss = mean_val_loss
                torch.save({
                    'epoch':         epoch,
                    'global_step':   global_step,
                    'model':         model.state_dict(),
                    'ema_shadow':    ema.shadow,
                    'optimizer':     optimizer.state_dict(),
                    'best_val_loss': best_val_loss,
                    'config':        config,
                }, ckpt_dir / 'best.pt')
                print(f"  -> best checkpoint (val: {best_val_loss:.4f})")

        # ── log ───────────────────────────────────────────────────────────────
        log({
            'epoch':       epoch + 1,
            'global_step': global_step,
            'train_loss':  mean_train_loss,
            'val_loss':    mean_val_loss,
            'lr':          scheduler.get_last_lr()[0],
        })

        # ── periodic checkpoint ───────────────────────────────────────────────
        if (epoch + 1) % tc.get('save_every', 50) == 0:
            torch.save({
                'epoch':         epoch,
                'global_step':   global_step,
                'model':         model.state_dict(),
                'ema_shadow':    ema.shadow,
                'optimizer':     optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'config':        config,
            }, ckpt_dir / f'epoch_{epoch+1:04d}.pt')
            print(f"  -> saved epoch checkpoint: epoch_{epoch+1:04d}.pt")

    print(f"\nTraining complete.")
    print(f"Best val loss: {best_val_loss:.4f}")
    print(f"All files in:  {ckpt_dir}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config yaml')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    print(f"Experiment: {config['experiment_name']}")
    train(config, resume_path=args.resume)


if __name__ == '__main__':
    main()