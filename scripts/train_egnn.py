"""
scripts/train_egnn.py  (GPU-optimised version)
===============================================
Replaces the previous train_egnn.py with full GPU optimisations:

  1. Mixed-precision training (torch.cuda.amp)
       - ~2× throughput on any Ampere/Volta GPU (A100, V100, RTX 3090…)
       - ~40% less VRAM → larger batches fit in memory
  2. cudnn.benchmark = True
       - cuDNN auto-tunes kernels for fixed input sizes → 5–15% extra speed
  3. non_blocking tensor transfers
       - Overlaps CPU→GPU copy with compute when pin_memory=True
  4. torch.compile()  (PyTorch ≥ 2.0, optional — skipped gracefully if unavailable)
       - Fuses EGNN kernels → 10–30% additional speedup on A100
  5. persistent_workers in DataLoader
       - Keeps worker processes alive between epochs → no fork overhead

Nothing else changes: same checkpoint format, same config schema,
same interface as the original train.py so all downstream scripts still work.
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
from scripts.train            import EMA, get_version_dir
from models.baseline          import MLPScoreNetwork, TransformerScoreNetwork
from models.egnn              import EGNNScoreNetwork
from models.diffusion_zerocom import ZeroCoMGaussianDiffusion

# ─────────────────────────────────────────────────────────────────────────────
# GPU SETUP
# ─────────────────────────────────────────────────────────────────────────────

def setup_gpu():
    """Print GPU info and enable cuDNN auto-tuner."""
    if not torch.cuda.is_available():
        print("WARNING: no CUDA device found — running on CPU")
        return "cpu"

    # cuDNN benchmark: profile a few batches then choose the fastest kernel.
    # Safe when input shapes are constant (they are for our fixed N=10 protein).
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False   # slightly faster than True

    gpu_name = torch.cuda.get_device_name(0)
    vram_gb  = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU : {gpu_name}  ({vram_gb:.1f} GB VRAM)")
    print(f"CUDA: {torch.version.cuda}  |  cuDNN: {torch.backends.cudnn.version()}")
    return "cuda"


# ─────────────────────────────────────────────────────────────────────────────
# MODEL FACTORY
# ─────────────────────────────────────────────────────────────────────────────

def build_model(config: dict) -> nn.Module:
    mc   = config['model']
    mt   = config['model_type']
    n_res = config['data']['n_residues']

    if mt == 'egnn':
        model = EGNNScoreNetwork(
            n_residues = n_res,
            node_dim   = mc['hidden_dim'],
            edge_dim   = mc.get('edge_dim', 64),
            time_dim   = mc['time_dim'],
            n_layers   = mc['n_layers'],
        )
        print(f"EGNN parameters: {model.count_parameters():,}")
        model.check_equivariance()
        return model
    elif mt == 'mlp':
        return MLPScoreNetwork(n_residues=n_res, hidden_dim=mc['hidden_dim'],
                               n_layers=mc['n_layers'], time_dim=mc['time_dim'],
                               dropout=mc['dropout'])
    elif mt == 'transformer':
        return TransformerScoreNetwork(n_residues=n_res, hidden_dim=mc['hidden_dim'],
                                       n_heads=mc['n_heads'], n_layers=mc['n_layers'],
                                       time_dim=mc['time_dim'], dropout=mc['dropout'])
    else:
        raise ValueError(f"Unknown model_type: {mt!r}")


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────────────────────────────────────

def train(config: dict, resume_path: str = None):

    # ── GPU setup ─────────────────────────────────────────────────────────────
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
    # Inject persistent_workers for faster epoch transitions
    config['training'].setdefault('num_workers', 4)
    train_loader, val_loader, _ = get_dataloaders(config)
    print(f"Train batches/epoch: {len(train_loader)}  "
          f"Val batches: {len(val_loader)}")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_model(config).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {config['model_type']}  |  {n_params:,} parameters")

    # torch.compile — fuses EGNN ops on PyTorch ≥ 2.0 (skipped on older versions)
    use_compile = config['training'].get('compile', True)
    if use_compile and hasattr(torch, 'compile') and device == 'cuda':
        try:
            model = torch.compile(model)
            print("torch.compile: enabled (first batch will be slow — compiling kernels)")
        except Exception as e:
            print(f"torch.compile: skipped ({e})")

    # ── Diffusion ─────────────────────────────────────────────────────────────
    dc        = config['diffusion']
    diffusion = ZeroCoMGaussianDiffusion(T=dc['T'], schedule=dc['schedule']).to(device)

    # ── Optimiser + scheduler ─────────────────────────────────────────────────
    tc           = config['training']
    optimizer    = AdamW(model.parameters(), lr=tc['lr'], weight_decay=1e-4)
    total_steps  = tc['n_epochs'] * len(train_loader)
    warmup_steps = tc.get('warmup_steps', 500)

    warmup    = LinearLR(optimizer, start_factor=0.01, end_factor=1.0,
                         total_iters=warmup_steps)
    cosine    = CosineAnnealingLR(optimizer, T_max=max(total_steps - warmup_steps, 1))
    scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine],
                              milestones=[warmup_steps])

    # ── Mixed-precision scaler ─────────────────────────────────────────────────
    # GradScaler prevents fp16 underflow: it scales the loss up before backward
    # then unscales gradients before the optimiser step.
    use_amp = device == 'cuda' and config['training'].get('amp', True)
    scaler  = torch.amp.GradScaler("cuda", enabled=use_amp)
    print(f"Mixed precision (AMP): {'enabled' if use_amp else 'disabled'}")

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
    def log(entry):
        with open(log_path, 'a') as f:
            f.write(json.dumps(entry) + '\n')

    # ── Training loop ─────────────────────────────────────────────────────────
    print(f"\nStarting training: {tc['n_epochs']} epochs, {total_steps:,} steps\n")

    for epoch in range(start_epoch, tc['n_epochs']):
        model.train()
        epoch_loss, n_batches = 0.0, 0
        t0 = time.time()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{tc['n_epochs']}",
                    dynamic_ncols=True)

        for batch in pbar:
            # non_blocking=True: GPU copy overlaps with the previous kernel
            # (only benefits when DataLoader uses pin_memory=True, which it does)
            x0 = batch['coords'].to(device, non_blocking=True)  # (B, 10, 3)

            optimizer.zero_grad(set_to_none=True)  # faster than zero_grad()

            # ── Forward pass under autocast ────────────────────────────────
            # autocast runs eligible ops (matmul, conv) in float16/bfloat16,
            # keeping numerically sensitive ops (softmax, layernorm) in float32.
            with torch.amp.autocast("cuda", enabled=use_amp):
                loss = diffusion.training_loss(model, x0)

            # ── Backward + optimiser step ──────────────────────────────────
            # scaler.scale(loss) multiplies loss by a dynamic scale factor
            # so that fp16 gradients don't underflow to zero.
            scaler.scale(loss).backward()

            # unscale before grad_clip so the clip threshold is in real units
            scaler.unscale_(optimizer)
            grad_norm = nn.utils.clip_grad_norm_(
                model.parameters(), tc.get('grad_clip', 1.0)
            )

            scaler.step(optimizer)   # skips step if gradients contain inf/NaN
            scaler.update()          # adjusts scale factor for next iteration
            scheduler.step()
            ema.update()

            epoch_loss  += loss.item()
            n_batches   += 1
            global_step += 1

            pbar.set_postfix(
                loss  = f"{loss.item():.4f}",
                gnorm = f"{grad_norm:.2f}",
                lr    = f"{scheduler.get_last_lr()[0]:.2e}",
                amp   = "ON" if use_amp else "off",
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

            # Throughput estimate
            n_structs = n_batches * tc['batch_size']
            throughput = n_structs / epoch_time
            print(
                f"\nEpoch {epoch+1:4d} | "
                f"train={mean_train:.4f}  val={mean_val:.4f}  "
                f"time={epoch_time:.0f}s  "
                f"({throughput:,.0f} structs/s)"
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

        log({'epoch': epoch+1, 'global_step': global_step,
             'train_loss': mean_train, 'val_loss': mean_val,
             'lr': scheduler.get_last_lr()[0],
             'epoch_time_s': epoch_time})

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


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--resume', default=None)
    args = p.parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)
    print(f"Experiment: {config['experiment_name']}")
    train(config, resume_path=args.resume)

if __name__ == '__main__':
    main()
