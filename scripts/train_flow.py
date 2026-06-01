"""
scripts/train_flow.py
======================
GPU-optimised training for the SE(3)-equivariant Flow Matching model.

Architecture: EGNNScoreNetwork (SE(3)-equivariant velocity network)
Framework:    ZeroCoMFlowMatching (OT-CFM on the zero-CoM subspace)

Together these form an SE(3)-invariant generative model for Chignolin Cα
coordinates — the same design principle as FoldFlow (Bose et al., 2024).

GPU optimisations (inherited from train_egnn.py):
  - Mixed-precision training (AMP) via torch.amp
  - cuDNN benchmark mode
  - non_blocking tensor transfers + pin_memory DataLoader
  - torch.compile() if PyTorch ≥ 2.0

Usage
-----
# Sanity check (CPU, ~1 min):
    python scripts/train_flow.py --config configs/flowmatch_local.yaml

# Full GPU training:
    python scripts/train_flow.py --config configs/flowmatch.yaml

# Resume from checkpoint:
    python scripts/train_flow.py --config configs/flowmatch.yaml \\
        --resume checkpoints/flowmatch/v1/epoch_0050.pt

Config schema
-------------
Uses a `flow:` section instead of `diffusion:`:
    flow:
      sigma_min: 1.0e-4   # noise floor (default 1e-4)

All other sections (data, model, training, paths) are identical to egnn.yaml.
model_type must be "flowmatch".
"""

import os, sys, json, copy, argparse, yaml, time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.dataset             import get_dataloaders
from scripts.train            import EMA, get_version_dir   # reuse shared utilities
from models.egnn              import EGNNScoreNetwork
from models.flow_matching     import ZeroCoMFlowMatching


# ─────────────────────────────────────────────────────────────────────────────
# GPU SETUP
# ─────────────────────────────────────────────────────────────────────────────

def setup_gpu() -> str:
    if not torch.cuda.is_available():
        print("WARNING: no CUDA device — running on CPU")
        return "cpu"

    torch.backends.cudnn.benchmark    = True
    torch.backends.cudnn.deterministic = False

    gpu_name = torch.cuda.get_device_name(0)
    vram_gb  = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU : {gpu_name}  ({vram_gb:.1f} GB VRAM)")
    print(f"CUDA: {torch.version.cuda}  |  cuDNN: {torch.backends.cudnn.version()}")
    return "cuda"


# ─────────────────────────────────────────────────────────────────────────────
# MODEL FACTORY
# ─────────────────────────────────────────────────────────────────────────────

def build_model(config: dict) -> nn.Module:
    """
    Build the EGNN velocity network from config.
    model_type must be 'flowmatch'; uses the same model.* keys as egnn.yaml.
    """
    mt = config['model_type']
    if mt != 'flowmatch':
        raise ValueError(
            f"train_flow.py only supports model_type='flowmatch', got {mt!r}.\n"
            f"Use train.py for 'mlp'/'transformer' or train_egnn.py for 'egnn'."
        )

    mc    = config['model']
    n_res = config['data']['n_residues']

    model = EGNNScoreNetwork(
        n_residues = n_res,
        node_dim   = mc['hidden_dim'],
        edge_dim   = mc.get('edge_dim', 64),
        time_dim   = mc['time_dim'],
        n_layers   = mc['n_layers'],
    )
    print(f"EGNNScoreNetwork: {model.count_parameters():,} parameters")
    model.check_equivariance()   # verify SE(3) equivariance on init
    return model


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────────────────────────────────────

def train(config: dict, resume_path: str = None):

    # ── GPU ───────────────────────────────────────────────────────────────────
    device = setup_gpu()

    # ── Versioning ────────────────────────────────────────────────────────────
    base_ckpt_dir = Path(config['paths']['checkpoint_dir'])
    if resume_path:
        ckpt_dir = Path(resume_path).parent
        print(f"Resuming: {ckpt_dir}")
    else:
        ckpt_dir = get_version_dir(base_ckpt_dir)
        print(f"New run:  {ckpt_dir}")

    with open(ckpt_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f)

    # ── Data ──────────────────────────────────────────────────────────────────
    config['training'].setdefault('num_workers', 4)
    train_loader, val_loader, _ = get_dataloaders(config)
    print(f"Train batches/epoch: {len(train_loader)}  "
          f"Val batches: {len(val_loader)}")

    # ── Model ─────────────────────────────────────────────────────────────────
    model   = build_model(config).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: flowmatch (EGNN velocity network)  |  {n_params:,} parameters")

    use_compile = config['training'].get('compile', True)
    if use_compile and hasattr(torch, 'compile') and device == 'cuda':
        try:
            model = torch.compile(model)
            print("torch.compile: enabled")
        except Exception as e:
            print(f"torch.compile: skipped ({e})")

    # ── Flow matching framework ───────────────────────────────────────────────
    fc        = config.get('flow', {})
    diffusion = ZeroCoMFlowMatching(
        sigma_min = fc.get('sigma_min', 1e-4)
    ).to(device)
    print(f"ZeroCoMFlowMatching  sigma_min={diffusion.sigma_min}")

    # ── Optimiser + scheduler ─────────────────────────────────────────────────
    tc           = config['training']
    optimizer    = AdamW(model.parameters(), lr=tc['lr'], weight_decay=1e-4)
    total_steps  = tc['n_epochs'] * len(train_loader)
    warmup_steps = tc.get('warmup_steps', 500)

    warmup    = LinearLR(optimizer, start_factor=0.01, end_factor=1.0,
                         total_iters=warmup_steps)
    cosine    = CosineAnnealingLR(optimizer,
                                   T_max=max(total_steps - warmup_steps, 1))
    scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine],
                              milestones=[warmup_steps])

    # ── AMP ───────────────────────────────────────────────────────────────────
    use_amp = device == 'cuda' and tc.get('amp', True)
    scaler  = torch.amp.GradScaler("cuda", enabled=use_amp)
    print(f"AMP: {'enabled' if use_amp else 'disabled'}")

    # ── EMA ───────────────────────────────────────────────────────────────────
    ema = EMA(model, decay=tc.get('ema_decay', 0.9999))

    # ── Resume ────────────────────────────────────────────────────────────────
    start_epoch   = 0
    best_val_loss = float('inf')
    global_step   = 0

    if resume_path:
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        if 'scaler' in ckpt:
            scaler.load_state_dict(ckpt['scaler'])
        ema.shadow    = ckpt['ema_shadow']
        start_epoch   = ckpt['epoch'] + 1
        global_step   = ckpt.get('global_step', start_epoch * len(train_loader))
        best_val_loss = ckpt.get('best_val_loss', float('inf'))
        print(f"Resumed at epoch {start_epoch}, step {global_step}")

    log_path = ckpt_dir / 'log.jsonl'

    def log(entry: dict):
        with open(log_path, 'a') as f:
            f.write(json.dumps(entry) + '\n')

    # ── Training loop ─────────────────────────────────────────────────────────
    print(f"\nStarting: {tc['n_epochs']} epochs, {total_steps:,} steps\n")

    for epoch in range(start_epoch, tc['n_epochs']):
        model.train()
        epoch_loss, n_batches = 0.0, 0
        t0 = time.time()

        pbar = tqdm(train_loader,
                    desc=f"Epoch {epoch+1}/{tc['n_epochs']}",
                    dynamic_ncols=True)

        for batch in pbar:
            x0 = batch['coords'].to(device, non_blocking=True)  # (B, 10, 3)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=use_amp):
                loss = diffusion.training_loss(model, x0)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = nn.utils.clip_grad_norm_(
                model.parameters(), tc.get('grad_clip', 1.0)
            )
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            ema.update()

            epoch_loss  += loss.item()
            n_batches   += 1
            global_step += 1

            pbar.set_postfix(
                loss  = f"{loss.item():.4f}",
                gnorm = f"{grad_norm:.2f}",
                lr    = f"{scheduler.get_last_lr()[0]:.2e}",
            )

        mean_train = epoch_loss / n_batches
        epoch_time = time.time() - t0

        # ── Validation ────────────────────────────────────────────────────────
        mean_val = None
        if (epoch + 1) % tc.get('val_every', 10) == 0:
            orig = ema.apply_shadow()
            model.eval()
            val_loss, n_val = 0.0, 0
            with torch.no_grad(), torch.amp.autocast("cuda", enabled=use_amp):
                for batch in val_loader:
                    x0 = batch['coords'].to(device, non_blocking=True)
                    val_loss += diffusion.training_loss(model, x0).item()
                    n_val    += 1
            mean_val = val_loss / n_val
            ema.restore(orig)

            throughput = n_batches * tc['batch_size'] / epoch_time
            print(
                f"\nEpoch {epoch+1:4d} | "
                f"train={mean_train:.4f}  val={mean_val:.4f}  "
                f"time={epoch_time:.0f}s  ({throughput:,.0f} structs/s)"
            )

            if mean_val < best_val_loss:
                best_val_loss = mean_val
                torch.save({
                    'epoch':         epoch,
                    'global_step':   global_step,
                    'model':         model.state_dict(),
                    'ema_shadow':    ema.shadow,
                    'optimizer':     optimizer.state_dict(),
                    'scaler':        scaler.state_dict(),
                    'best_val_loss': best_val_loss,
                    'config':        config,
                }, ckpt_dir / 'best.pt')
                print(f"  -> best checkpoint saved (val={best_val_loss:.4f})")

        log({
            'epoch':        epoch + 1,
            'global_step':  global_step,
            'train_loss':   mean_train,
            'val_loss':     mean_val,
            'lr':           scheduler.get_last_lr()[0],
            'epoch_time_s': epoch_time,
        })

        if (epoch + 1) % tc.get('save_every', 50) == 0:
            torch.save({
                'epoch':         epoch,
                'global_step':   global_step,
                'model':         model.state_dict(),
                'ema_shadow':    ema.shadow,
                'optimizer':     optimizer.state_dict(),
                'scaler':        scaler.state_dict(),
                'best_val_loss': best_val_loss,
                'config':        config,
            }, ckpt_dir / f'epoch_{epoch+1:04d}.pt')

    print(f"\nDone. Best val loss: {best_val_loss:.4f}")
    print(f"Checkpoints: {ckpt_dir}")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True,
                   help='Path to config yaml (e.g. configs/flowmatch_local.yaml)')
    p.add_argument('--resume', default=None,
                   help='Checkpoint to resume from')
    args = p.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    print(f"Experiment: {config['experiment_name']}")
    train(config, resume_path=args.resume)


if __name__ == '__main__':
    main()
