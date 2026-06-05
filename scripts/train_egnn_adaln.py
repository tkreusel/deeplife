"""
scripts/train_egnn_adaln.py
============================
Training script for EGNNAdaLNScoreNetwork:
  SE(3)-equivariant DDPM with AdaLN-Zero timestep conditioning,
  energy-based CFG, and SNR-weighted physics constraints.

Key additions over train_egnn.py
---------------------------------
1. Energy z-score normalisation
   Raw energies are normalised at startup using training-set statistics.
   Mean and std are stored in the checkpoint so sampling scripts are
   self-contained (no need to rescan the dataset).

2. Energy CFG training
   Each batch's normalised energy_z is passed to the model.
   The model randomly drops it (p=0.15) to learn both conditional and
   unconditional distributions — enabling guided sampling at inference.

3. Guided sampling helper printed at end
   After training, shows the exact command to run temperature-controlled
   generation using analyze_energy_conditioning.py.

Usage
-----
    python scripts/train_egnn_adaln.py --config configs/egnn_adaln.yaml
    python scripts/train_egnn_adaln.py --config configs/egnn_adaln.yaml \\
        --resume checkpoints/egnn_adaln/v1/latest.pt

Physics constraints
-------------------
Same SNR-weighted bond/clash/angle loss as train_egnn.py v4:
    L = L_diffusion + physics_weight · E[α̅_t · ChignolinPhysics(x0_pred)]
with physics_weight=0.10, clash_weight=0.4 (best settings from ablation).
"""

import os, sys, json, copy, argparse, yaml, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.dataset             import get_dataloaders
from scripts.train            import EMA, get_version_dir
from models.egnn_adaln        import EGNNAdaLNScoreNetwork
from models.egnn_energy       import EGNNEnergyScoreNetwork
from models.diffusion_zerocom import ZeroCoMGaussianDiffusion
from models.physics           import ChignolinPhysics
from models.physics_aa        import AllAtomPhysics


# ─────────────────────────────────────────────────────────────────────────────
# GPU SETUP
# ─────────────────────────────────────────────────────────────────────────────

def setup_gpu():
    if not torch.cuda.is_available():
        print("WARNING: no CUDA device — running on CPU")
        return "cpu"
    torch.backends.cudnn.benchmark     = True
    torch.backends.cudnn.deterministic = False
    gpu  = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU : {gpu}  ({vram:.1f} GB VRAM)")
    print(f"CUDA: {torch.version.cuda}  |  cuDNN: {torch.backends.cudnn.version()}")
    return "cuda"


# ─────────────────────────────────────────────────────────────────────────────
# ENERGY NORMALISATION
# ─────────────────────────────────────────────────────────────────────────────

def compute_energy_stats(train_loader, device: str):
    """
    Compute mean and std of raw energies over the entire training set.
    Returns (mean, std) as Python floats.
    Used to z-score normalise energies before passing to the model.
    """
    all_e = []
    for batch in train_loader:
        all_e.append(batch['energies'])
    all_e = torch.cat(all_e)
    mean  = all_e.mean().item()
    std   = all_e.std().item()
    std   = max(std, 1e-6)   # guard against degenerate datasets
    print(f"Energy stats:  mean={mean:.3f}  std={std:.3f}  "
          f"(N={len(all_e):,} structures)")
    return mean, std


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────────────────────────────────────

def train(config: dict, resume_path: str = None, reset_schedule: bool = False):

    device = setup_gpu()

    # ── Versioning ────────────────────────────────────────────────────────
    base_ckpt_dir = Path(config['paths']['checkpoint_dir'])
    if resume_path:
        ckpt_dir = Path(resume_path).parent
        print(f"Resuming: {ckpt_dir}")
    else:
        ckpt_dir = get_version_dir(base_ckpt_dir)
        print(f"New run:  {ckpt_dir}")

    with open(ckpt_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f)

    # ── Data ──────────────────────────────────────────────────────────────
    config['training'].setdefault('num_workers', 4)
    train_loader, val_loader, _ = get_dataloaders(config)
    print(f"Train batches/epoch: {len(train_loader)}  Val: {len(val_loader)}")

    # ── Energy normalisation ───────────────────────────────────────────────
    print("Computing energy statistics from training set …")
    energy_mean, energy_std = compute_energy_stats(train_loader, device)
    # Store in config so checkpoints are self-contained
    config.setdefault('energy_stats', {})
    config['energy_stats']['mean'] = energy_mean
    config['energy_stats']['std']  = energy_std

    # ── Model — auto-select based on model_type ───────────────────────────
    mc    = config['model']
    n_res = config['data']['n_residues']
    mt    = config['model_type']

    if mt == 'egnn_energy':
        # Plain EGNN + energy conditioning, no AdaLN — simpler, often better
        model = EGNNEnergyScoreNetwork(
            n_residues       = n_res,
            node_dim         = mc['hidden_dim'],
            edge_dim         = mc.get('edge_dim',         64),
            time_dim         = mc['time_dim'],
            n_layers         = mc['n_layers'],
            energy_dim       = mc.get('energy_dim',       32),
            energy_drop_prob = mc.get('energy_drop_prob', 0.15),
        ).to(device)
        print(f"EGNNEnergy parameters: {model.count_parameters():,}")
    else:
        # egnn_adaln — EGNN + AdaLN-Zero + energy
        model = EGNNAdaLNScoreNetwork(
            n_residues       = n_res,
            node_dim         = mc['hidden_dim'],
            edge_dim         = mc.get('edge_dim',         64),
            time_dim         = mc.get('time_dim',         64),
            n_layers         = mc['n_layers'],
            energy_dim       = mc.get('energy_dim',       32),
            energy_drop_prob = mc.get('energy_drop_prob', 0.15),
        ).to(device)
        print(f"EGNNAdaLN parameters: {model.count_parameters():,}")

    model.check_equivariance()

    # torch.compile
    tc = config['training']
    if tc.get('compile', True) and hasattr(torch, 'compile') and device == 'cuda':
        try:
            model = torch.compile(model)
            print("torch.compile: enabled")
        except Exception as e:
            print(f"torch.compile: skipped ({e})")

    # ── Diffusion (ZeroCoM — all noise projected to zero-CoM subspace) ────
    dc        = config['diffusion']
    diffusion = ZeroCoMGaussianDiffusion(T=dc['T'], schedule=dc['schedule']).to(device)

    # ── Optimiser + scheduler ─────────────────────────────────────────────
    optimizer    = AdamW(model.parameters(), lr=tc['lr'], weight_decay=1e-4)
    total_steps  = tc['n_epochs'] * len(train_loader)
    warmup_steps = tc.get('warmup_steps', 1000)

    warmup    = LinearLR(optimizer, start_factor=0.01, end_factor=1.0,
                         total_iters=warmup_steps)
    cosine    = CosineAnnealingLR(optimizer, T_max=max(total_steps - warmup_steps, 1))
    scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine],
                              milestones=[warmup_steps])

    # ── AMP ───────────────────────────────────────────────────────────────
    use_amp = device == 'cuda' and tc.get('amp', True)
    scaler  = torch.amp.GradScaler("cuda", enabled=use_amp)
    print(f"Mixed precision (AMP): {'enabled' if use_amp else 'disabled'}")

    # ── Physics constraints ───────────────────────────────────────────────
    # Auto-selects AllAtomPhysics for n_residues != 10, ChignolinPhysics otherwise.
    physics_weight = tc.get('physics_weight', 0.0)
    n_res          = config['data']['n_residues']
    if physics_weight > 0.0:
        phys_cfg    = config.get('physics', {})
        coord_scale = config['data'].get('coord_scale', 5.0)
        if n_res != 10:
            # All-atom: per-bond targets derived from training set statistics
            physics_fn = AllAtomPhysics(
                bond_weight  = phys_cfg.get('bond_weight',  1.0),
                clash_weight = phys_cfg.get('clash_weight', 0.1),
                coord_scale  = coord_scale,
            )
        else:
            # Cα-only: single ideal bond length (3.832 Å) + angle + clash
            physics_fn = ChignolinPhysics(
                bond_weight  = phys_cfg.get('bond_weight',  1.0),
                clash_weight = phys_cfg.get('clash_weight', 0.4),
                angle_weight = phys_cfg.get('angle_weight', 0.5),
                coord_scale  = coord_scale,
            )
        print(f"Physics: {physics_fn}  weight={physics_weight}")
    else:
        physics_fn = None
        print("Physics: disabled")

    # ── EMA ───────────────────────────────────────────────────────────────
    ema = EMA(model, decay=tc.get('ema_decay', 0.9999))

    # ── Resume ────────────────────────────────────────────────────────────
    start_epoch   = 0
    best_val_loss = float('inf')
    global_step   = 0

    if resume_path:
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        if 'scaler' in ckpt:
            scaler.load_state_dict(ckpt['scaler'])
        if not reset_schedule:
            if 'scheduler' in ckpt:
                scheduler.load_state_dict(ckpt['scheduler'])
            else:
                for _ in range(ckpt.get('global_step', 0)):
                    scheduler.step()
        # else: keep the freshly-created schedule (new warmup+cosine from this point)
        ema.shadow    = ckpt['ema_shadow']
        start_epoch   = ckpt['epoch'] + 1
        global_step   = ckpt.get('global_step', start_epoch * len(train_loader))
        best_val_loss = ckpt.get('best_val_loss', float('inf'))
        if 'energy_stats' in ckpt.get('config', {}):
            energy_mean = ckpt['config']['energy_stats']['mean']
            energy_std  = ckpt['config']['energy_stats']['std']
        print(f"Resumed at epoch {start_epoch}, step {global_step}, "
              f"lr={scheduler.get_last_lr()[0]:.2e}"
              f"{'  [fresh schedule]' if reset_schedule else ''}")

    log_path = ckpt_dir / 'log.jsonl'
    def log(entry):
        with open(log_path, 'a') as f:
            f.write(json.dumps(entry) + '\n')

    # ── Training loop ─────────────────────────────────────────────────────
    print(f"\nStarting training: {tc['n_epochs']} epochs, {total_steps:,} steps\n")

    for epoch in range(start_epoch, tc['n_epochs']):
        model.train()
        epoch_loss, n_batches = 0.0, 0
        t0 = time.time()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{tc['n_epochs']}",
                    dynamic_ncols=True)

        for batch in pbar:
            x0       = batch['coords'].to(device, non_blocking=True)   # (B, N, 3)
            energies = batch['energies'].to(device, non_blocking=True)  # (B,)
            energy_z = (energies - energy_mean) / energy_std            # z-score

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=use_amp):
                loss = diffusion.training_loss(
                    model, x0,
                    physics_weight = physics_weight,
                    physics_fn     = physics_fn,
                    model_kwargs   = {'energy_z': energy_z},
                )

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

        # ── Validation ────────────────────────────────────────────────────
        mean_val = None
        phys_log = {}
        if (epoch + 1) % tc.get('val_every', 10) == 0:
            orig = ema.apply_shadow()
            model.eval()
            val_loss, n_val = 0.0, 0
            with torch.no_grad(), torch.amp.autocast("cuda", enabled=use_amp):
                for batch in val_loader:
                    x0       = batch['coords'].to(device, non_blocking=True)
                    energies = batch['energies'].to(device, non_blocking=True)
                    energy_z = (energies - energy_mean) / energy_std
                    val_loss += diffusion.training_loss(
                        model, x0,
                        physics_weight = physics_weight,
                        physics_fn     = physics_fn,
                        model_kwargs   = {'energy_z': energy_z},
                    ).item()
                    n_val += 1
                    # Physics breakdown on first val batch
                    if physics_fn is not None and n_val == 1:
                        t_d  = torch.zeros(x0.shape[0], dtype=torch.long, device=device)
                        xt,_ = diffusion.q_sample(x0, t_d)
                        np_  = model(xt, t_d, energy_z=energy_z)
                        x0p  = diffusion._predict_x0_from_noise(xt, t_d, np_)
                        phys_log = physics_fn.breakdown(x0p)
            mean_val = val_loss / n_val
            ema.restore(orig)

            n_structs  = n_batches * tc['batch_size']
            throughput = n_structs / epoch_time
            print(
                f"\nEpoch {epoch+1:4d} | "
                f"train={mean_train:.4f}  val={mean_val:.4f}  "
                f"time={epoch_time:.0f}s  ({throughput:,.0f} structs/s)"
            )
            if phys_log:
                print(f"  bond={phys_log['phys_bond']:.3f}  "
                      f"clash={phys_log['phys_clash']:.3f}  "
                      f"angle={phys_log['phys_angle']:.3f}")

            if mean_val < best_val_loss:
                best_val_loss = mean_val
                torch.save({
                    'epoch':         epoch,
                    'global_step':   global_step,
                    'model':         model.state_dict(),
                    'ema_shadow':    ema.shadow,
                    'optimizer':     optimizer.state_dict(),
                    'scheduler':     scheduler.state_dict(),
                    'scaler':        scaler.state_dict(),
                    'best_val_loss': best_val_loss,
                    'config':        config,
                }, ckpt_dir / 'best.pt')
                print(f"  -> best checkpoint saved (val={best_val_loss:.4f})")

        log({'epoch': epoch+1, 'global_step': global_step,
             'train_loss': mean_train, 'val_loss': mean_val,
             'lr': scheduler.get_last_lr()[0],
             'epoch_time_s': epoch_time,
             **phys_log})

        if (epoch + 1) % tc.get('save_every', 50) == 0:
            torch.save({
                'epoch':         epoch,
                'global_step':   global_step,
                'model':         model.state_dict(),
                'ema_shadow':    ema.shadow,
                'optimizer':     optimizer.state_dict(),
                'scheduler':     scheduler.state_dict(),
                'scaler':        scaler.state_dict(),
                'best_val_loss': best_val_loss,
                'config':        config,
            }, ckpt_dir / f'epoch_{epoch+1:04d}.pt')

    print(f"\nDone. Best val loss: {best_val_loss:.4f}")
    print(f"Checkpoints: {ckpt_dir}")
    print(f"\nTemperature-controlled sampling:")
    print(f"  python scripts/analyze_energy_conditioning.py \\")
    print(f"    --checkpoint {ckpt_dir}/best.pt \\")
    print(f"    --test data/test.npz \\")
    print(f"    --temperatures 0.0 0.25 0.5 0.75 1.0 \\")
    print(f"    --n 500 --steps 100 --guidance_scale 2.0 \\")
    print(f"    --save plots/energy_egnn_adaln.png")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--resume', default=None)
    p.add_argument('--reset_schedule', action='store_true',
                   help='Ignore saved scheduler state and start a fresh LR schedule. '
                        'Use when resuming after a completed run (LR=0) to continue training.')
    args = p.parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)
    print(f"Experiment: {config['experiment_name']}")
    train(config, resume_path=args.resume, reset_schedule=args.reset_schedule)


if __name__ == '__main__':
    main()
