"""
scripts/train_flow_energy.py
=============================
GPU-optimised training for the energy-conditioned SE(3) Flow Matching model.

Architecture: EGNNEnergyScoreNetwork (SE(3)-equivariant velocity network
              conditioned on per-structure energy with CFG dropout)
Framework:    ZeroCoMFlowMatching.training_loss_energy()

Energy conditioning
-------------------
Each structure in the dataset has a scalar potential energy label.  During
training the energy is z-score normalised (zero mean, unit variance over the
training set) and passed to the model.  With probability p_drop=0.15 the
energy embedding is replaced by the learned null embedding — this is the
Classifier-Free Guidance (CFG) dropout that enables guidance-scale control
at inference.

Energy statistics (mean, std) are computed once from the training loader
before the main loop starts, then stored in the checkpoint config so that
sampling scripts can reconstruct the normalisation without re-loading data.

Temperature-controlled sampling
--------------------------------
After training, use scripts/analyze_energy_conditioning.py to sweep
τ ∈ {0, 0.25, 0.5, 0.75, 1.0} and verify that:
    τ=0  →  compact, folded structures  (Rg ≈ 5.0 Å)
    τ=1  →  extended, transient structures (Rg ≈ 7.0 Å)

Or sample directly:
    python scripts/analyze_energy_conditioning.py \\
        --checkpoint checkpoints/flowmatch_energy/v1/best.pt \\
        --test data/test.npz \\
        --temperatures 0.0 0.25 0.5 0.75 1.0 \\
        --n 500 --steps 100 --guidance_scale 2.0 \\
        --save plots/energy_analysis.png

Usage
-----
# CPU smoke test (~2 min):
    python scripts/train_flow_energy.py --config configs/flowmatch_energy_local.yaml

# Full GPU training:
    python scripts/train_flow_energy.py --config configs/flowmatch_energy.yaml

# Resume:
    python scripts/train_flow_energy.py --config configs/flowmatch_energy.yaml \\
        --resume checkpoints/flowmatch_energy/v1/epoch_0050.pt

Config schema
-------------
Identical to flowmatch.yaml with two extra model keys:
    model.energy_dim        (default 32)
    model.energy_drop_prob  (default 0.15)
model_type must be "flowmatch_energy".
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
from models.egnn_energy       import EGNNEnergyScoreNetwork
from models.flow_matching     import ZeroCoMFlowMatching


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
# MODEL FACTORY
# ─────────────────────────────────────────────────────────────────────────────

def build_model(config: dict) -> nn.Module:
    mt = config['model_type']
    if mt != 'flowmatch_energy':
        raise ValueError(
            f"train_flow_energy.py requires model_type='flowmatch_energy', got {mt!r}.\n"
            f"Use train_flow.py for 'flowmatch'."
        )

    mc    = config['model']
    n_res = config['data']['n_residues']

    model = EGNNEnergyScoreNetwork(
        n_residues       = n_res,
        node_dim         = mc['hidden_dim'],
        edge_dim         = mc.get('edge_dim', 64),
        time_dim         = mc['time_dim'],
        n_layers         = mc['n_layers'],
        energy_dim       = mc.get('energy_dim', 32),
        energy_drop_prob = mc.get('energy_drop_prob', 0.15),
    )
    print(f"EGNNEnergyScoreNetwork: {model.count_parameters():,} parameters")
    print(f"  energy_dim={mc.get('energy_dim', 32)}  "
          f"p_drop={mc.get('energy_drop_prob', 0.15)}")
    model.check_equivariance()
    return model


# ─────────────────────────────────────────────────────────────────────────────
# ENERGY STATISTICS
# ─────────────────────────────────────────────────────────────────────────────

def compute_energy_stats(train_loader, device: str):
    """
    Compute energy mean and std from the full training set (one pass).
    Returns (mean: float, std: float).
    """
    print("Computing energy statistics from training set…")
    all_e = []
    for batch in train_loader:
        all_e.append(batch['energies'])
    all_e  = torch.cat(all_e, dim=0).float()
    mean   = all_e.mean().item()
    std    = all_e.std().item()
    print(f"  Energy: mean={mean:.4f}  std={std:.6f}  "
          f"(N={len(all_e):,} structures)")
    return mean, std


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
        print(f"New run:  {ckpt_dir}")

    # ── Data ──────────────────────────────────────────────────────────────────
    config['training'].setdefault('num_workers', 4)
    train_loader, val_loader, _ = get_dataloaders(config)
    print(f"Train batches/epoch: {len(train_loader)}  "
          f"Val batches: {len(val_loader)}")

    # ── Energy normalisation stats ────────────────────────────────────────────
    energy_mean, energy_std = compute_energy_stats(train_loader, device)
    # Persist in config so checkpoints are self-contained
    config['data']['energy_mean'] = energy_mean
    config['data']['energy_std']  = energy_std

    # Save config (includes energy stats)
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
    fc        = config.get('flow', {})
    diffusion = ZeroCoMFlowMatching(
        sigma_min = fc.get('sigma_min', 1e-4)
    ).to(device)
    print(f"ZeroCoMFlowMatching  sigma_min={diffusion.sigma_min}")

    # ── Physics constraints (optional) ────────────────────────────────────────
    tc          = config['training']
    physics     = None
    phys_weight = tc.get('physics_weight', 0.0)
    pc          = config.get('physics', {})

    if phys_weight > 0.0 and pc:
        from models.physics import ChignolinPhysics
        physics = ChignolinPhysics(
            bond_weight  = pc.get('bond_weight',  1.0),
            clash_weight = pc.get('clash_weight', 0.1),
            angle_weight = pc.get('angle_weight', 0.5),
            coord_scale  = config['data'].get('coord_scale', 5.0),
        )
        print(f"Physics: {physics}  λ={phys_weight}")
    else:
        print("Physics: disabled")

    # ── Optimiser + scheduler ─────────────────────────────────────────────────
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
            x0    = batch['coords'].to(device, non_blocking=True)    # (B, 10, 3)
            e_raw = batch['energies'].to(device, non_blocking=True)  # (B,)
            e_z   = (e_raw - energy_mean) / energy_std               # z-normalised

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=use_amp):
                loss = diffusion.training_loss_energy(
                    model, x0, e_z,
                    physics_weight = phys_weight,
                    physics_fn     = physics,
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

        # ── Validation ────────────────────────────────────────────────────────
        mean_val = None
        phys_log = {}

        if (epoch + 1) % tc.get('val_every', 10) == 0:
            orig = ema.apply_shadow()
            model.eval()
            val_loss, n_val = 0.0, 0
            phys_accum: dict = {}
            n_phys = 0

            with torch.no_grad(), torch.amp.autocast("cuda", enabled=use_amp):
                for batch in val_loader:
                    x0    = batch['coords'].to(device, non_blocking=True)
                    e_raw = batch['energies'].to(device, non_blocking=True)
                    e_z   = (e_raw - energy_mean) / energy_std
                    val_loss += diffusion.training_loss_energy(
                        model, x0, e_z,
                        physics_weight = phys_weight,
                        physics_fn     = physics,
                    ).item()
                    n_val += 1

                    if physics is not None:
                        bd = physics.breakdown(x0)
                        for k, v in bd.items():
                            phys_accum[k] = phys_accum.get(k, 0.0) + v
                        n_phys += 1

            mean_val = val_loss / n_val
            ema.restore(orig)

            if physics is not None and n_phys > 0:
                phys_log = {k: v / n_phys for k, v in phys_accum.items()}

            throughput = n_batches * tc['batch_size'] / epoch_time
            phys_str   = (f"  bond={phys_log.get('phys_bond', 0):.4f}"
                          f"  clash={phys_log.get('phys_clash', 0):.4f}"
                          f"  angle={phys_log.get('phys_angle', 0):.4f}"
                          if phys_log else "")
            print(
                f"\nEpoch {epoch+1:4d} | "
                f"train={mean_train:.4f}  val={mean_val:.4f}  "
                f"time={epoch_time:.0f}s  ({throughput:,.0f} structs/s)"
                + phys_str
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
            'energy_mean':  energy_mean,
            'energy_std':   energy_std,
            **phys_log,
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
    print(f"\nTo analyse temperature control:")
    print(f"  python scripts/analyze_energy_conditioning.py \\")
    print(f"      --checkpoint {ckpt_dir}/best.pt \\")
    print(f"      --test data/test.npz \\")
    print(f"      --temperatures 0.0 0.25 0.5 0.75 1.0 \\")
    print(f"      --n 500 --steps 100 --guidance_scale 2.0 \\")
    print(f"      --save plots/energy_analysis.png")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True,
                   help='Path to config yaml (e.g. configs/flowmatch_energy_local.yaml)')
    p.add_argument('--resume', default=None,
                   help='Checkpoint to resume from')
    args = p.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    print(f"Experiment: {config['experiment_name']}")
    train(config, resume_path=args.resume)


if __name__ == '__main__':
    main()
