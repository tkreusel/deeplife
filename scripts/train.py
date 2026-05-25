"""
scripts/train.py

Main training loop for the Chignolin diffusion model.

Usage:
    python scripts/train.py --config configs/baseline.yaml
    python scripts/train.py --config configs/baseline.yaml --resume checkpoints/baseline/epoch_0050.pt
"""

import os
import sys
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
    """
    Keeps an exponential moving average of model weights.
    Use EMA weights for validation and sampling — they are smoother
    and consistently produce better results than the raw trained weights.

    decay=0.9999 means each update the shadow weights move 0.01% toward
    the current weights. High decay = slow-moving, stable average.
    """

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
        """Swap in EMA weights. Returns original weights so you can restore."""
        original = copy.deepcopy(self.model.state_dict())
        self.model.load_state_dict(self.shadow)
        return original

    def restore(self, original):
        """Restore original weights after evaluation."""
        self.model.load_state_dict(original)


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
        raise ValueError(f"Unknown model_type: {model_type}. Choose 'mlp' or 'transformer'.")


# ── Training ──────────────────────────────────────────────────────────────────

def train(config: dict, resume_path: str = None):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # ── output dirs ───────────────────────────────────────────────────────────
    ckpt_dir = Path(config['paths']['checkpoint_dir'])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # save config next to checkpoints so you always know what produced them
    with open(ckpt_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f)

    # ── data ──────────────────────────────────────────────────────────────────
    train_loader, val_loader, _ = get_dataloaders(config)

    # ── model ─────────────────────────────────────────────────────────────────
    model = build_model(config).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {config['model_type']}  |  Parameters: {n_params:,}")

    # ── diffusion ─────────────────────────────────────────────────────────────
    dc       = config['diffusion']
    diffusion = GaussianDiffusion(T=dc['T'], schedule=dc['schedule']).to(device)

    # ── optimiser ─────────────────────────────────────────────────────────────
    tc        = config['training']
    optimizer = AdamW(model.parameters(), lr=tc['lr'], weight_decay=1e-4)

    # linear warmup for warmup_steps steps, then cosine decay for the rest
    total_steps  = tc['n_epochs'] * len(train_loader)
    warmup_steps = tc.get('warmup_steps', 500)

    warmup    = LinearLR(optimizer, start_factor=0.01, end_factor=1.0,
                         total_iters=warmup_steps)
    cosine    = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps)
    scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine],
                             milestones=[warmup_steps])

    # ── EMA ───────────────────────────────────────────────────────────────────
    ema = EMA(model, decay=tc.get('ema_decay', 0.9999))

    # ── resume ────────────────────────────────────────────────────────────────
    start_epoch   = 0
    best_val_loss = float('inf')

    if resume_path is not None:
        print(f"Resuming from {resume_path}")
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        ema.shadow    = ckpt['ema_shadow']
        start_epoch   = ckpt['epoch'] + 1
        best_val_loss = ckpt.get('best_val_loss', float('inf'))
        print(f"Resumed at epoch {start_epoch}")

    # ── training loop ─────────────────────────────────────────────────────────
    train_losses = []
    val_losses   = []

    for epoch in range(start_epoch, tc['n_epochs']):
        model.train()
        epoch_loss  = 0.0
        n_batches   = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{tc['n_epochs']}")

        for batch in pbar:
            # batch is a dict — pull coords, move to device
            x0 = batch['coords'].to(device)     # (B, 10, 3)

            loss = diffusion.training_loss(model, x0)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), tc.get('grad_clip', 1.0))
            optimizer.step()
            scheduler.step()
            ema.update()

            epoch_loss += loss.item()
            n_batches  += 1

            pbar.set_postfix(
                loss = f"{loss.item():.4f}",
                lr   = f"{scheduler.get_last_lr()[0]:.2e}",
            )

        mean_train_loss = epoch_loss / n_batches
        train_losses.append(mean_train_loss)

        # ── validation ────────────────────────────────────────────────────────
        if (epoch + 1) % tc.get('val_every', 10) == 0:

            # swap in EMA weights for validation
            original = ema.apply_shadow()
            model.eval()

            val_loss  = 0.0
            n_val     = 0

            with torch.no_grad():
                for batch in val_loader:
                    x0       = batch['coords'].to(device)
                    loss     = diffusion.training_loss(model, x0)
                    val_loss += loss.item()
                    n_val    += 1

            mean_val_loss = val_loss / n_val
            val_losses.append((epoch + 1, mean_val_loss))

            # restore training weights
            ema.restore(original)

            print(
                f"\nEpoch {epoch+1:4d} | "
                f"train loss: {mean_train_loss:.4f} | "
                f"val loss: {mean_val_loss:.4f}"
            )

            # save best checkpoint
            if mean_val_loss < best_val_loss:
                best_val_loss = mean_val_loss
                torch.save({
                    'epoch':         epoch,
                    'model':         model.state_dict(),
                    'ema_shadow':    ema.shadow,
                    'optimizer':     optimizer.state_dict(),
                    'best_val_loss': best_val_loss,
                    'config':        config,
                }, ckpt_dir / 'best.pt')
                print(f"  -> saved best checkpoint (val loss: {best_val_loss:.4f})")

        # ── periodic checkpoint ───────────────────────────────────────────────
        if (epoch + 1) % tc.get('save_every', 50) == 0:
            torch.save({
                'epoch':         epoch,
                'model':         model.state_dict(),
                'ema_shadow':    ema.shadow,
                'optimizer':     optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'config':        config,
            }, ckpt_dir / f'epoch_{epoch+1:04d}.pt')

    print(f"\nTraining done. Best val loss: {best_val_loss:.4f}")
    print(f"Checkpoints in: {ckpt_dir}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config yaml, e.g. configs/baseline.yaml')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    print(f"Experiment: {config['experiment_name']}")
    train(config, resume_path=args.resume)


if __name__ == '__main__':
    main()