"""
scripts/train_torsion.py
========================
Training script for TorsionFlow — Riemannian flow matching in internal coordinate
space for Chignolin Cα conformations.

Supported model_type values:
  'torsion_flow_energy'       — TorsionFlowNet (MLP, ~230K params)
  'torsion_transformer_energy'— TorsionTransformerNet (Transformer, ~850K params)

Framework:     TorsionalFlowMatching (OT-CFM on bond-angle + torsion-angle space)
Coordinates:   Bond angles θ (8) + dihedral angles φ (7) — bond lengths fixed at
               3.832 Å by representation, so bond validity is 100% by construction.
SE(3):         Internal coordinates are invariant to rigid rotations/translations —
               no equivariant network or data augmentation required.

Key differences from train_flow_energy.py
------------------------------------------
- Converts Cartesian data to internal coordinates (per batch) before training.
- No physics loss — bond lengths are guaranteed, not learned.
- No ZeroCoM projection — internal coords have no CoM concept.
- Two-stage stats warmup: energy stats + torsion stats (theta_mean, velocity scales).
- Optional: data-driven WrappedNormal φ source distribution + per-position weights.

New in v2 (torsion_transformer_energy)
----------------------------------------
- Transformer backbone replaces MLP: captures inter-residue correlations.
- phi_source_dist: 'uniform' (default) or 'data' — uses circular stds from training
  data for a shorter-path source distribution (WrappedNormal instead of Uniform).
- theta_source_std: configurable from config (flow.theta_source_std, default 0.403).
- LR schedule: cosine annealing with warm restarts (CosineAnnealingWarmRestarts)
  via flow.cosine_T0 / flow.cosine_T_mult in config, or plain cosine if not set.

Usage
-----
    # CPU sanity-check (~1 min):
    python scripts/train_torsion.py --config configs/torsion_transformer_local.yaml

    # Full GPU training:
    python scripts/train_torsion.py --config configs/torsion_transformer_energy.yaml

    # Resume:
    python scripts/train_torsion.py --config configs/torsion_transformer_energy.yaml \\
        --resume checkpoints/torsion_transformer/v1/epoch_0050.pt

Evaluation after training
--------------------------
    python scripts/evaluate.py \\
        --ckpt checkpoints/torsion_transformer/v1/best.pt \\
        --test data/test.npz --n 1000

    # Bond validity will be 100% by construction. Key metrics: Rg, MMD, diversity.
"""

import json
import math
import sys
import time
import argparse
import yaml
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    LinearLR,
    SequentialLR,
)
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.dataset             import get_dataloaders
from scripts.train            import EMA, get_version_dir
from models.internal_coords   import (
    cartesian_to_internal, compute_velocity_scales,
    compute_phi_source_params, N_ANGLES, N_DIHEDRALS,
)
from models.torsion_flow      import TorsionalFlowMatching

_SUPPORTED = ('torsion_flow_energy', 'torsion_transformer_energy')


# ─────────────────────────────────────────────────────────────────────────────
# GPU SETUP
# ─────────────────────────────────────────────────────────────────────────────

def setup_gpu() -> str:
    if not torch.cuda.is_available():
        print("WARNING: no CUDA device — running on CPU")
        return "cpu"
    torch.backends.cudnn.benchmark     = True
    torch.backends.cudnn.deterministic = False
    gpu_name = torch.cuda.get_device_name(0)
    vram_gb  = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU : {gpu_name}  ({vram_gb:.1f} GB VRAM)")
    print(f"CUDA: {torch.version.cuda}  |  cuDNN: {torch.backends.cudnn.version()}")
    return "cuda"


# ─────────────────────────────────────────────────────────────────────────────
# MODEL CONSTRUCTION
# ─────────────────────────────────────────────────────────────────────────────

def build_model(config: dict) -> nn.Module:
    """Instantiate the velocity network from config['model_type']."""
    mt = config['model_type']
    mc = config['model']

    if mt == 'torsion_flow_energy':
        from models.torsion_net import TorsionFlowNet
        model = TorsionFlowNet(
            hidden_dim       = mc['hidden_dim'],
            n_layers         = mc['n_layers'],
            time_dim         = mc['time_dim'],
            energy_dim       = mc.get('energy_dim',       32),
            energy_drop_prob = mc.get('energy_drop_prob', 0.15),
        )
        print(f"TorsionFlowNet (MLP): {model.count_parameters():,} parameters")

    elif mt == 'torsion_transformer_energy':
        from models.torsion_transformer import TorsionTransformerNet
        model = TorsionTransformerNet(
            d_model          = mc.get('d_model',          256),
            n_heads          = mc.get('n_heads',          4),
            n_layers         = mc.get('n_layers',         6),
            time_dim         = mc.get('time_dim',         64),
            energy_dim       = mc.get('energy_dim',       32),
            energy_drop_prob = mc.get('energy_drop_prob', 0.15),
            dropout          = mc.get('dropout',          0.1),
        )
        print(f"TorsionTransformerNet: {model.count_parameters():,} parameters")

    else:
        raise ValueError(
            f"Unknown model_type={mt!r}. Supported: {_SUPPORTED}"
        )

    return model


# ─────────────────────────────────────────────────────────────────────────────
# STATISTICS  (one-pass warmup over the training set)
# ─────────────────────────────────────────────────────────────────────────────

def compute_energy_stats(train_loader, device: str) -> tuple[float, float]:
    """Compute energy mean and std from the full training set."""
    print("Computing energy statistics …")
    all_e = []
    for batch in train_loader:
        all_e.append(batch['energies'])
    all_e = torch.cat(all_e).float()
    mean, std = all_e.mean().item(), all_e.std().item()
    print(f"  Energy: mean={mean:.4f}  std={std:.6f}  (N={len(all_e):,} structures)")
    return mean, std


def compute_torsion_stats(
    train_loader, coord_scale: float, device: str,
    theta_source_std: float = 0.30,
    phi_source_dist: str = 'uniform',
) -> dict:
    """
    Collect all bond angles and dihedrals from the training set, then compute:
        theta_mean  : mean bond angle [rad]  (used as Gaussian source centre)
        theta_scale : velocity std for Δθ    (loss normalisation)
        phi_scale   : velocity std for Δφ    (loss normalisation)

    When phi_source_dist='data', also computes:
        phi_source_std : (7,) per-dihedral source std for WrappedNormal source
        phi_weights    : (7,) inverse-variance per-position loss weights
        phi_scale      : updated based on WrappedNormal velocity distribution

    Returns a dict with all computed values + optional phi buffers.
    """
    print("Computing torsion statistics (one pass over training set) …")
    all_theta, all_phi = [], []

    for batch in train_loader:
        x1 = batch['coords'].to(device) * coord_scale   # (B, 10, 3) in Å
        th, ph = cartesian_to_internal(x1)
        all_theta.append(th.cpu())
        all_phi.append(ph.cpu())

    all_theta = torch.cat(all_theta, dim=0)   # (N_train, 8)
    all_phi   = torch.cat(all_phi,   dim=0)   # (N_train, 7)

    theta_mean, theta_scale, phi_scale = compute_velocity_scales(
        all_theta, all_phi, theta_source_std=theta_source_std
    )

    print(f"  theta_source_std={theta_source_std:.4f} rad")
    print(f"  theta_mean ={theta_mean:.4f} rad  ({math.degrees(theta_mean):.1f}°)")
    print(f"  theta_scale={theta_scale:.4f} rad  (loss normalisation)")
    print(f"  phi_scale  ={phi_scale:.4f} rad  (loss normalisation, Uniform source)")

    result = dict(
        theta_mean=theta_mean,
        theta_scale=theta_scale,
        phi_scale=phi_scale,
        phi_source_std=None,
        phi_weights=None,
    )

    if phi_source_dist == 'data':
        print("Computing per-dihedral source stats (phi_source_dist='data') …")
        phi_src_std, phi_weights, phi_scale_new = compute_phi_source_params(all_phi)
        result['phi_source_std'] = phi_src_std
        result['phi_weights']    = phi_weights
        result['phi_scale']      = phi_scale_new   # override with informed scale
        print(f"  phi_scale (updated)={phi_scale_new:.4f} rad")
    else:
        print(f"  phi_source_dist='uniform' (default) — no per-dihedral stats")

    return result


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────────────────────────────────────

def train(config: dict, resume_path: str = None):

    device = setup_gpu()

    # ── Versioning ────────────────────────────────────────────────────────────
    base_ckpt_dir = Path(config['paths']['checkpoint_dir'])
    if resume_path:
        ckpt_dir = Path(resume_path).parent
        print(f"Resuming: {ckpt_dir}")
    else:
        ckpt_dir = get_version_dir(base_ckpt_dir)
        print(f"New run: {ckpt_dir}")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ── Data ──────────────────────────────────────────────────────────────────
    config['training'].setdefault('num_workers', 4)
    train_loader, val_loader, _ = get_dataloaders(config)
    coord_scale = config['data'].get('coord_scale', 5.0)
    print(f"Train batches/epoch: {len(train_loader)}  Val batches: {len(val_loader)}")

    # ── Stats warmup ──────────────────────────────────────────────────────────
    energy_mean, energy_std = compute_energy_stats(train_loader, device)
    config['data']['energy_mean'] = energy_mean
    config['data']['energy_std']  = energy_std

    fc = config.get('flow', {})
    theta_source_std = fc.get('theta_source_std', 0.30)
    phi_source_dist  = fc.get('phi_source_dist',  'uniform')

    stats = compute_torsion_stats(
        train_loader, coord_scale, device,
        theta_source_std=theta_source_std,
        phi_source_dist=phi_source_dist,
    )
    theta_mean   = stats['theta_mean']
    theta_scale  = stats['theta_scale']
    phi_scale    = stats['phi_scale']
    phi_src_std  = stats['phi_source_std']
    phi_weights  = stats['phi_weights']

    config['data']['theta_mean']  = theta_mean
    config['data']['theta_scale'] = theta_scale
    config['data']['phi_scale']   = phi_scale

    with open(ckpt_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_model(config).to(device)

    use_compile = config['training'].get('compile', True)
    if use_compile and hasattr(torch, 'compile') and device == 'cuda':
        try:
            model = torch.compile(model)
            print("torch.compile: enabled")
        except Exception as e:
            print(f"torch.compile: skipped ({e})")

    # ── Flow matching framework ───────────────────────────────────────────────
    flow = TorsionalFlowMatching(
        sigma_min        = fc.get('sigma_min', 1e-4),
        theta_mean       = theta_mean,
        theta_source_std = theta_source_std,
        theta_scale      = theta_scale,
        phi_scale        = phi_scale,
        phi_source_std   = phi_src_std,    # None → Uniform(−π,π)
        phi_weights      = phi_weights,    # None → uniform loss weights
    ).to(device)

    print(
        f"TorsionalFlowMatching  sigma_min={flow.sigma_min}"
        f"  theta_scale={theta_scale:.3f}  phi_scale={phi_scale:.3f}"
        f"  phi_src={'data' if phi_src_std is not None else 'uniform'}"
        f"  phi_weights={'yes' if phi_weights is not None else 'no'}"
    )

    # ── Optimiser + scheduler ─────────────────────────────────────────────────
    tc           = config['training']
    optimizer    = AdamW(model.parameters(), lr=tc['lr'], weight_decay=1e-4)
    total_steps  = tc['n_epochs'] * len(train_loader)
    warmup_steps = tc.get('warmup_steps', 500)

    warmup = LinearLR(optimizer, start_factor=0.01, end_factor=1.0,
                      total_iters=warmup_steps)

    cosine_T0   = fc.get('cosine_T0', None)
    cosine_Tmul = fc.get('cosine_T_mult', 2)

    if cosine_T0 is not None:
        # Cosine annealing with warm restarts for the transformer
        cosine_steps_per_restart = cosine_T0 * len(train_loader)
        after_warmup = CosineAnnealingWarmRestarts(
            optimizer, T_0=max(cosine_steps_per_restart, 1),
            T_mult=int(cosine_Tmul), eta_min=1e-6,
        )
        print(f"LR: warm-up {warmup_steps} steps → "
              f"CosineWarmRestarts T0={cosine_T0} epochs × {cosine_Tmul} mult")
    else:
        after_warmup = CosineAnnealingLR(
            optimizer, T_max=max(total_steps - warmup_steps, 1)
        )
        print(f"LR: warm-up {warmup_steps} steps → CosineAnnealing to 0")

    scheduler = SequentialLR(optimizer, schedulers=[warmup, after_warmup],
                              milestones=[warmup_steps])

    # ── AMP + EMA ─────────────────────────────────────────────────────────────
    use_amp = device == 'cuda' and tc.get('amp', True)
    scaler  = torch.amp.GradScaler("cuda", enabled=use_amp)
    ema     = EMA(model, decay=tc.get('ema_decay', 0.9999))
    print(f"AMP: {'enabled' if use_amp else 'disabled'}")

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
            # Cartesian data → internal coordinates (in Å, then to angles)
            x1    = batch['coords'].to(device, non_blocking=True) * coord_scale
            e_raw = batch['energies'].to(device, non_blocking=True)
            e_z   = (e_raw - energy_mean) / energy_std

            theta1, phi1 = cartesian_to_internal(x1)   # (B, 8), (B, 7)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=use_amp):
                loss = flow.training_loss_energy(model, theta1, phi1, e_z)

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
                    x1    = batch['coords'].to(device) * coord_scale
                    e_raw = batch['energies'].to(device)
                    e_z   = (e_raw - energy_mean) / energy_std
                    theta1, phi1 = cartesian_to_internal(x1)
                    val_loss += flow.training_loss_energy(model, theta1, phi1, e_z).item()
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
                    'flow':          flow.state_dict(),   # buffers: phi_source_std, phi_weights
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
            'energy_mean':  energy_mean,
            'energy_std':   energy_std,
            'theta_mean':   theta_mean,
            'theta_scale':  theta_scale,
            'phi_scale':    phi_scale,
            'lr':           scheduler.get_last_lr()[0],
            'epoch_time_s': epoch_time,
        })

        if (epoch + 1) % tc.get('save_every', 50) == 0:
            torch.save({
                'epoch':         epoch,
                'global_step':   global_step,
                'model':         model.state_dict(),
                'ema_shadow':    ema.shadow,
                'flow':          flow.state_dict(),   # buffers: phi_source_std, phi_weights
                'optimizer':     optimizer.state_dict(),
                'scaler':        scaler.state_dict(),
                'best_val_loss': best_val_loss,
                'config':        config,
            }, ckpt_dir / f'epoch_{epoch+1:04d}.pt')

    print(f"\nDone. Best val loss: {best_val_loss:.4f}")
    print(f"Checkpoints: {ckpt_dir}")
    print(f"\nEvaluate with:")
    print(f"  python scripts/evaluate.py \\")
    print(f"      --ckpt {ckpt_dir}/best.pt \\")
    print(f"      --test data/test.npz --n 1000")
    print(f"\n  Bond validity will be 100% by construction.")
    print(f"  Key metrics to check: Rg distribution, MMD, diversity.")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True,
                   help='Path to config yaml (e.g. configs/torsion_transformer_energy.yaml)')
    p.add_argument('--resume', default=None,
                   help='Checkpoint to resume from')
    args = p.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    assert config['model_type'] in _SUPPORTED, (
        f"This script supports model_type in {_SUPPORTED!r}, "
        f"got {config['model_type']!r}"
    )

    print(f"Experiment: {config['experiment_name']}")
    train(config, resume_path=args.resume)


if __name__ == '__main__':
    main()
