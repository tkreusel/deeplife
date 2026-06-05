"""
scripts/train_backbone_ipa.py
==============================
Training script for BackboneIPAFlow — IPA-style Transformer + Riemannian OT-CFM
on backbone torsion angles (φ, ψ) for Chignolin N-Cα-C backbone.

model_type: "backbone_ipa_energy"

Key features
------------
- Backbone representation: N, Cα, C atoms (30 total, 18 DOF = 9 φ + 9 ψ)
- NeRF reconstruction guarantees 100% bond validity for all 29 backbone bonds
- SE(3)-invariant torsion-angle input → SE(3)-invariant velocity output
- IPA-style geometric attention bias from backbone frames (AlphaFold2 convention)
- AdaLN-Zero per-layer conditioning from (time, energy)
- Auxiliary Cartesian loss (t²-weighted ETE + Rg penalty) via cart_weight
- Energy conditioning (CFG): τ ∈ [0,1] → e_z = 4τ − 2 at inference
- Optional self-distillation (flow.self_distill: true) when resuming
  from a converged base model

Usage
-----
    # CPU smoke test (~2 min):
    python scripts/train_backbone_ipa.py --config configs/backbone_ipa_local.yaml

    # Full GPU training:
    python scripts/train_backbone_ipa.py --config configs/backbone_ipa_energy.yaml

    # Resume:
    python scripts/train_backbone_ipa.py --config configs/backbone_ipa_energy.yaml \\
        --resume checkpoints/backbone_ipa/v1/best.pt

    # Self-distillation fine-tune (only after base model converged):
    #   Edit config: flow.self_distill: true
    python scripts/train_backbone_ipa.py --config configs/backbone_ipa_energy.yaml \\
        --resume checkpoints/backbone_ipa/v1/best.pt

Evaluation
----------
    python scripts/evaluate.py \\
        --ckpt checkpoints/backbone_ipa/v1/best.pt \\
        --test data_backbone/test.npz --n 1000
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

from data.dataset                      import get_dataloaders
from scripts.train                     import EMA, get_version_dir
from models.backbone_internal_coords  import (
    backbone_to_internal,
    compute_backbone_velocity_scales,
    compute_backbone_source_params,
)
from models.backbone_torsion_flow     import BackboneTorsionalFlowMatching
from models.backbone_ipa_flow         import BackboneIPAFlowNet

_SUPPORTED = ('backbone_ipa_energy',)


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
# STATISTICS  (one-pass warmup)
# ─────────────────────────────────────────────────────────────────────────────

def compute_energy_stats(train_loader, device: str) -> tuple[float, float]:
    print("Computing energy statistics …")
    all_e = []
    for batch in train_loader:
        all_e.append(batch['energies'])
    all_e = torch.cat(all_e).float()
    mean, std = all_e.mean().item(), all_e.std().item()
    print(f"  Energy: mean={mean:.4f}  std={std:.6f}  (N={len(all_e):,} structures)")
    return mean, std


def compute_backbone_torsion_stats(
    train_loader, coord_scale: float, device: str,
    phi_source_dist: str = 'uniform',
    psi_source_dist: str = 'uniform',
) -> dict:
    """
    Collect all (φ, ψ) torsion angles from the training set and compute:
        phi_scale : velocity std for Δφ (loss normalisation)
        psi_scale : velocity std for Δψ (loss normalisation)

    When source_dist='data', also computes per-dihedral WrappedNormal source
    parameters and per-position inverse-variance loss weights.
    """
    print("Computing backbone torsion statistics (one pass over training set) …")
    all_phi, all_psi = [], []

    for batch in train_loader:
        x1 = batch['coords'].to(device) * coord_scale   # (B, 30, 3) in Å
        ph, ps = backbone_to_internal(x1)               # (B, 9), (B, 9)
        all_phi.append(ph.cpu())
        all_psi.append(ps.cpu())

    all_phi = torch.cat(all_phi, dim=0)   # (N_train, 9)
    all_psi = torch.cat(all_psi, dim=0)   # (N_train, 9)

    phi_scale, psi_scale = compute_backbone_velocity_scales(all_phi, all_psi)
    print(f"  phi_scale={phi_scale:.4f} rad  psi_scale={psi_scale:.4f} rad")

    result = dict(
        phi_scale=phi_scale,
        psi_scale=psi_scale,
        phi_source_std=None,
        psi_source_std=None,
        phi_weights=None,
        psi_weights=None,
    )

    if phi_source_dist == 'data' or psi_source_dist == 'data':
        print("Computing per-dihedral source params …")
        phi_std, psi_std, phi_w, psi_w, phi_s2, psi_s2 = \
            compute_backbone_source_params(all_phi, all_psi)

        if phi_source_dist == 'data':
            result['phi_source_std'] = phi_std
            result['phi_weights']    = phi_w
            result['phi_scale']      = phi_s2
        if psi_source_dist == 'data':
            result['psi_source_std'] = psi_std
            result['psi_weights']    = psi_w
            result['psi_scale']      = psi_s2

    return result


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────────────────────────────────────

def train(config: dict, resume_path: str = None, weights_path: str = None):

    device = setup_gpu()

    # ── Versioning ────────────────────────────────────────────────────────────
    base_ckpt_dir = Path(config['paths']['checkpoint_dir'])
    if resume_path:
        ckpt_dir = Path(resume_path).parent
        print(f"Resuming: {ckpt_dir}")
    else:
        # --weights always creates a fresh version dir (fine-tune or distillation)
        ckpt_dir = get_version_dir(base_ckpt_dir)
        print(f"New run: {ckpt_dir}")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ── Data ──────────────────────────────────────────────────────────────────
    config['training'].setdefault('num_workers', 4)
    train_loader, val_loader, _ = get_dataloaders(config)
    coord_scale = config['data'].get('coord_scale', 3.42)
    print(f"coord_scale={coord_scale} Å  "
          f"Train batches/epoch: {len(train_loader)}  Val: {len(val_loader)}")

    # ── Stats warmup ──────────────────────────────────────────────────────────
    energy_mean, energy_std = compute_energy_stats(train_loader, device)
    config['data']['energy_mean'] = energy_mean
    config['data']['energy_std']  = energy_std

    fc = config.get('flow', {})
    stats = compute_backbone_torsion_stats(
        train_loader, coord_scale, device,
        phi_source_dist=fc.get('phi_source_dist', 'uniform'),
        psi_source_dist=fc.get('psi_source_dist', 'uniform'),
    )

    phi_scale     = stats['phi_scale']
    psi_scale     = stats['psi_scale']
    phi_source_std = stats['phi_source_std']
    psi_source_std = stats['psi_source_std']
    phi_weights   = stats['phi_weights']
    psi_weights   = stats['psi_weights']

    config['data']['phi_scale'] = phi_scale
    config['data']['psi_scale'] = psi_scale

    with open(ckpt_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f)

    # ── Model ─────────────────────────────────────────────────────────────────
    mc = config['model']
    model = BackboneIPAFlowNet(
        d_model          = mc.get('d_model',          256),
        n_heads          = mc.get('n_heads',          8),
        n_layers         = mc.get('n_layers',         6),
        time_dim         = mc.get('time_dim',         64),
        energy_dim       = mc.get('energy_dim',       32),
        energy_drop_prob = mc.get('energy_drop_prob', 0.15),
        dropout          = mc.get('dropout',          0.1),
        n_rbf            = mc.get('n_rbf',            16),
        sep_dim          = mc.get('sep_dim',          4),
        rbf_dmax         = mc.get('rbf_dmax',         18.0),
        ff_mult          = mc.get('ff_mult',          2),
    ).to(device)
    print(f"BackboneIPAFlowNet: {model.count_parameters():,} parameters")

    use_compile = config['training'].get('compile', True)
    if use_compile and hasattr(torch, 'compile') and device == 'cuda':
        try:
            model = torch.compile(model)
            print("torch.compile: enabled")
        except Exception as e:
            print(f"torch.compile: skipped ({e})")

    # ── Flow matching framework ───────────────────────────────────────────────
    flow = BackboneTorsionalFlowMatching(
        sigma_min      = fc.get('sigma_min', 1e-4),
        phi_scale      = phi_scale,
        psi_scale      = psi_scale,
        phi_source_std = phi_source_std,
        psi_source_std = psi_source_std,
        phi_weights    = phi_weights,
        psi_weights    = psi_weights,
    ).to(device)

    cart_weight    = fc.get('cart_weight', 0.1)
    self_distill   = fc.get('self_distill', False)
    distill_weight = fc.get('self_distill_weight', 0.5)
    distill_steps  = fc.get('self_distill_steps', 5)

    print(
        f"BackboneTorsionalFlowMatching  "
        f"phi_scale={phi_scale:.3f}  psi_scale={psi_scale:.3f}  "
        f"phi_src={'data' if phi_source_std is not None else 'uniform'}  "
        f"psi_src={'data' if psi_source_std is not None else 'uniform'}\n"
        f"  cart_weight={cart_weight}  "
        f"self_distill={'ON' if self_distill else 'off'}"
    )

    # ── Optimiser + scheduler ─────────────────────────────────────────────────
    tc           = config['training']
    optimizer    = AdamW(model.parameters(), lr=tc['lr'], weight_decay=1e-4)
    total_steps  = tc['n_epochs'] * len(train_loader)
    warmup_steps = tc.get('warmup_steps', 1000)

    warmup = LinearLR(optimizer, start_factor=0.01, end_factor=1.0,
                      total_iters=warmup_steps)

    cosine_T0   = fc.get('cosine_T0', None)
    cosine_Tmul = fc.get('cosine_T_mult', 2)

    if cosine_T0 is not None:
        cosine_steps_per_restart = cosine_T0 * len(train_loader)
        after_warmup = CosineAnnealingWarmRestarts(
            optimizer, T_0=max(cosine_steps_per_restart, 1),
            T_mult=int(cosine_Tmul), eta_min=1e-6,
        )
        print(f"LR: warm-up {warmup_steps} steps → "
              f"CosineWarmRestarts T0={cosine_T0} epochs × {cosine_Tmul}")
    else:
        after_warmup = CosineAnnealingLR(
            optimizer, T_max=max(total_steps - warmup_steps, 1),
        )
        print(f"LR: warm-up {warmup_steps} steps → CosineAnnealing")

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

    def _load_ema_weights(ckpt_path: str):
        """Load EMA shadow weights, normalising _orig_mod. prefix for compiled models."""
        ckpt = torch.load(ckpt_path, map_location=device)
        sd   = ckpt['ema_shadow']
        sd   = {k.replace('_orig_mod.', ''): v for k, v in sd.items()}
        if any(k.startswith('_orig_mod.') for k in model.state_dict()):
            sd = {'_orig_mod.' + k: v for k, v in sd.items()}
        model.load_state_dict(sd, strict=True)
        # Re-register EMA shadow using the model's actual parameter names (which may
        # include _orig_mod. after torch.compile) so ema.update() can find them.
        ema.shadow = {name: param.data.clone()
                      for name, param in model.named_parameters()
                      if param.requires_grad}
        return ckpt

    if resume_path:
        ckpt = _load_ema_weights(resume_path)
        optimizer.load_state_dict(ckpt['optimizer'])
        if 'scheduler' in ckpt:
            scheduler.load_state_dict(ckpt['scheduler'])
        if 'scaler' in ckpt:
            scaler.load_state_dict(ckpt['scaler'])
        start_epoch   = ckpt['epoch'] + 1
        global_step   = ckpt.get('global_step', start_epoch * len(train_loader))
        best_val_loss = ckpt.get('best_val_loss', float('inf'))
        print(f"Resumed at epoch {start_epoch}, step {global_step}")
        if self_distill:
            print("Self-distillation mode ACTIVE — teacher = EMA model")

    elif weights_path:
        # Fine-tune / distillation: load only EMA weights, reset everything else.
        # Optimizer, scheduler, scaler all start fresh — avoids LR-spike regression.
        _load_ema_weights(weights_path)
        print(f"Loaded EMA weights from {weights_path} (optimizer + scheduler reset)")
        if self_distill:
            print("Self-distillation mode ACTIVE — teacher = EMA model")

    # ── EMA model for self-distillation teacher ───────────────────────────────
    # We access the EMA shadow weights through a wrapper for teacher forward pass
    ema_model_wrapper = None
    if self_distill:
        # Build a CPU/GPU copy of the model seeded from EMA shadow
        ema_model_wrapper = BackboneIPAFlowNet(
            d_model          = mc.get('d_model',          256),
            n_heads          = mc.get('n_heads',          8),
            n_layers         = mc.get('n_layers',         6),
            time_dim         = mc.get('time_dim',         64),
            energy_dim       = mc.get('energy_dim',       32),
            energy_drop_prob = mc.get('energy_drop_prob', 0.15),
            dropout          = mc.get('dropout',          0.1),
            n_rbf            = mc.get('n_rbf',            16),
            sep_dim          = mc.get('sep_dim',          4),
            rbf_dmax         = mc.get('rbf_dmax',         18.0),
            ff_mult          = mc.get('ff_mult',          2),
        ).to(device)
        ema_model_wrapper.eval()

    log_path = ckpt_dir / 'log.jsonl'

    def log_entry(entry: dict):
        with open(log_path, 'a') as f:
            f.write(json.dumps(entry) + '\n')

    # ── Training loop ─────────────────────────────────────────────────────────
    print(f"\nStarting: {tc['n_epochs']} epochs, {total_steps:,} steps\n")

    for epoch in range(start_epoch, tc['n_epochs']):
        model.train()

        # Update EMA teacher weights for self-distillation
        if self_distill and ema_model_wrapper is not None:
            # Load current EMA shadow into teacher wrapper
            shadow_sd = {k.replace('_orig_mod.', ''): v
                         for k, v in ema.shadow.items()}
            # strict=False: ema.shadow has only parameters, not buffers (e.g.
            # geo_bias.rbf.centers). Buffers are fixed at init so missing is fine.
            ema_model_wrapper.load_state_dict(shadow_sd, strict=False)
            ema_model_wrapper.eval()

        epoch_loss, fm_loss_sum, n_batches = 0.0, 0.0, 0
        t0 = time.time()

        pbar = tqdm(train_loader,
                    desc=f"Epoch {epoch+1}/{tc['n_epochs']}",
                    dynamic_ncols=True)

        for batch in pbar:
            x1    = batch['coords'].to(device, non_blocking=True) * coord_scale
            e_raw = batch['energies'].to(device, non_blocking=True)
            e_z   = (e_raw - energy_mean) / energy_std

            # Backbone Cartesian → (φ, ψ) torsion angles
            phi1, psi1 = backbone_to_internal(x1)   # (B, 9), (B, 9)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=use_amp):
                if self_distill and ema_model_wrapper is not None:
                    loss, fm_loss = flow.training_loss_with_distillation(
                        model, ema_model_wrapper, phi1, psi1, e_z,
                        cart_weight=cart_weight,
                        distill_weight=distill_weight,
                        distill_steps=distill_steps,
                    )
                elif cart_weight > 0.0:
                    loss, fm_loss = flow.training_loss_with_cartesian(
                        model, phi1, psi1, e_z, cart_weight=cart_weight,
                    )
                else:
                    loss    = flow.training_loss_energy(model, phi1, psi1, e_z)
                    fm_loss = loss

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
            fm_loss_sum += fm_loss.item() if hasattr(fm_loss, 'item') else float(fm_loss)
            n_batches   += 1
            global_step += 1

            pbar.set_postfix(
                loss  = f"{loss.item():.4f}",
                fm    = f"{fm_loss.item() if hasattr(fm_loss,'item') else fm_loss:.4f}",
                gnorm = f"{grad_norm:.2f}",
                lr    = f"{scheduler.get_last_lr()[0]:.2e}",
            )

        mean_train_total = epoch_loss  / n_batches
        mean_train_fm    = fm_loss_sum / n_batches
        epoch_time       = time.time() - t0

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
                    phi1, psi1 = backbone_to_internal(x1)
                    # Always use the base FM loss for validation (comparable across modes)
                    val_loss += flow.training_loss_energy(model, phi1, psi1, e_z).item()
                    n_val    += 1

            mean_val = val_loss / n_val
            ema.restore(orig)

            throughput = n_batches * tc['batch_size'] / epoch_time
            print(
                f"\nEpoch {epoch+1:4d} | "
                f"train={mean_train_total:.4f} (fm={mean_train_fm:.4f})  "
                f"val={mean_val:.4f}  "
                f"time={epoch_time:.0f}s  ({throughput:,.0f} structs/s)"
            )

            if mean_val < best_val_loss:
                best_val_loss = mean_val
                torch.save({
                    'epoch':         epoch,
                    'global_step':   global_step,
                    'model':         model.state_dict(),
                    'ema_shadow':    ema.shadow,
                    'flow':          flow.state_dict(),
                    'optimizer':     optimizer.state_dict(),
                    'scheduler':     scheduler.state_dict(),
                    'scaler':        scaler.state_dict(),
                    'best_val_loss': best_val_loss,
                    'config':        config,
                }, ckpt_dir / 'best.pt')
                print(f"  -> best checkpoint saved (val={best_val_loss:.4f})")

        log_entry({
            'epoch':           epoch + 1,
            'global_step':     global_step,
            'train_loss':      mean_train_total,
            'train_fm_loss':   mean_train_fm,
            'val_loss':        mean_val,
            'energy_mean':     energy_mean,
            'energy_std':      energy_std,
            'phi_scale':       phi_scale,
            'psi_scale':       psi_scale,
            'cart_weight':     cart_weight,
            'self_distill':    self_distill,
            'lr':              scheduler.get_last_lr()[0],
            'epoch_time_s':    epoch_time,
        })

        if (epoch + 1) % tc.get('save_every', 50) == 0:
            torch.save({
                'epoch':         epoch,
                'global_step':   global_step,
                'model':         model.state_dict(),
                'ema_shadow':    ema.shadow,
                'flow':          flow.state_dict(),
                'optimizer':     optimizer.state_dict(),
                'scheduler':     scheduler.state_dict(),
                'scaler':        scaler.state_dict(),
                'best_val_loss': best_val_loss,
                'config':        config,
            }, ckpt_dir / f'epoch_{epoch+1:04d}.pt')

    print(f"\nDone. Best val loss: {best_val_loss:.4f}")
    print(f"Checkpoints: {ckpt_dir}")
    print(f"\nEvaluate with:")
    print(f"  python scripts/evaluate.py \\")
    print(f"      --ckpt {ckpt_dir}/best.pt \\")
    print(f"      --test data_backbone/test.npz --n 1000")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True,
                   help='Path to config yaml')
    p.add_argument('--resume', default=None,
                   help='Checkpoint to resume from (full state: model + optimizer + scheduler)')
    p.add_argument('--weights', default=None,
                   help='Load only EMA model weights from checkpoint, reset everything else '
                        '(creates a new version dir). Use for fine-tuning or self-distillation.')
    args = p.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    assert config['model_type'] in _SUPPORTED, (
        f"This script supports {_SUPPORTED!r}, got {config['model_type']!r}"
    )

    print(f"Experiment: {config['experiment_name']}")
    train(config, resume_path=args.resume, weights_path=args.weights)


if __name__ == '__main__':
    main()
