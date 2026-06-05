"""
scripts/train_adaln_sc.py
==========================
Training for AdaLN Transformer + self-conditioning + energy CFG + physics.

Self-conditioning procedure
----------------------------
At each denoising step during *inference* the model receives the x₀ prediction
from the *previous* DDIM step as x0_self_cond.  That previous step operated at
a noise level of t + Δ, where Δ = T // ddim_steps (e.g. 5 for 100-step DDIM).

To match this exactly during *training*:
  • 50 % of batches: corrupt x₀ to noise level t_sc = t + Δ, run a no-grad
    preliminary forward pass to get x0_sc, then train the main pass at t with
    that x0_sc.  This is the signal the model will actually see at inference.
  • 50 % of batches: x0_self_cond = zeros (teaches the "first step" code path).

Using t_sc > t (rather than t itself) eliminates the training/inference
mismatch that caused exponential noise divergence at low timesteps in earlier
versions.

Diffusion process
-----------------
Uses plain GaussianDiffusion (same as the successful AdaLN+Energy+Physics
model) rather than ZeroCoMGaussianDiffusion.  The zero-CoM constraint forced
projection of noise predictions during inference, which introduced systematic
errors that compounded through the SC feedback loop.

Usage:
    python scripts/train_adaln_sc.py --config configs/transformer_adaln_sc.yaml
    python scripts/train_adaln_sc.py --config configs/transformer_adaln_sc.yaml \\
        --resume checkpoints/transformer_adaln_sc/v3/epoch_0200.pt
"""

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

from data.dataset import get_dataloaders
from models.diffusion import GaussianDiffusion
from models.transformer_adaln_sc import AdaLNSCScoreNetwork


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

def get_version_dir(base: Path) -> Path:
    existing = sorted(
        [d for d in base.iterdir() if d.is_dir() and d.name.startswith('v')]
    ) if base.exists() else []
    nxt = int(existing[-1].name[1:]) + 1 if existing else 1
    d   = base / f'v{nxt}'
    d.mkdir(parents=True, exist_ok=True)
    return d


# ── Energy stats ──────────────────────────────────────────────────────────────

def compute_energy_stats(train_loader) -> tuple[float, float]:
    print("Computing energy statistics…")
    all_e = torch.cat([b['energies'] for b in train_loader]).float()
    mean, std = all_e.mean().item(), all_e.std().item()
    print(f"  mean={mean:.4f}  std={std:.6f}  N={len(all_e):,}")
    return mean, std


# ── Model factory ─────────────────────────────────────────────────────────────

def build_model(config: dict) -> nn.Module:
    mc = config['model']
    assert config['model_type'] == 'transformer_adaln_sc', \
        f"Expected model_type='transformer_adaln_sc', got {config['model_type']!r}"
    return AdaLNSCScoreNetwork(
        n_residues       = config['data']['n_residues'],
        hidden_dim       = mc['hidden_dim'],
        n_heads          = mc['n_heads'],
        n_layers         = mc['n_layers'],
        time_dim         = mc['time_dim'],
        dropout          = mc['dropout'],
        energy_drop_prob = mc.get('energy_drop_prob', 0.15),
    )


# ── Training ──────────────────────────────────────────────────────────────────

def train(config: dict, resume_path: str = None):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    base_ckpt_dir = Path(config['paths']['checkpoint_dir'])
    if resume_path:
        ckpt_dir = Path(resume_path).parent
        print(f"Resuming: {ckpt_dir}")
    else:
        ckpt_dir = get_version_dir(base_ckpt_dir)
        print(f"New run:  {ckpt_dir}")

    train_loader, val_loader, _ = get_dataloaders(config)
    print(f"Train batches/epoch: {len(train_loader)}  Val: {len(val_loader)}")

    energy_mean, energy_std = compute_energy_stats(train_loader)
    config['data']['energy_mean'] = energy_mean
    config['data']['energy_std']  = energy_std

    with open(ckpt_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f)

    model    = build_model(config).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_params:,}")

    dc        = config['diffusion']
    diffusion = GaussianDiffusion(T=dc['T'], schedule=dc['schedule']).to(device)

    # SC step Δ: x0_sc comes from noise level t+Δ during training, matching the
    # DDIM inference loop where x0_sc always came from one step earlier (higher noise).
    # Default matches 100-step DDIM: Δ = T // 100 = 5 for T=500.
    sc_ddim_steps = tc_sc = config.get('training', {}).get('sc_ddim_steps', 100)
    sc_delta      = max(1, dc['T'] // sc_ddim_steps)
    print(f"SC Δ = {sc_delta}  (T={dc['T']}, sc_ddim_steps={sc_ddim_steps})")

    tc          = config['training']
    phys_weight = tc.get('physics_weight', 0.0)
    physics     = None
    if phys_weight > 0.0:
        from models.physics import ChignolinPhysics
        pc      = config.get('physics', {})
        physics = ChignolinPhysics(
            bond_weight  = pc.get('bond_weight',  1.0),
            clash_weight = pc.get('clash_weight', 0.1),
            angle_weight = pc.get('angle_weight', 0.5),
            coord_scale  = config['data'].get('coord_scale', 5.0),
        )
        print(f"Physics: {physics}  λ={phys_weight}")
    else:
        print("Physics: disabled")

    optimizer    = AdamW(model.parameters(), lr=tc['lr'], weight_decay=1e-4)
    total_steps  = tc['n_epochs'] * len(train_loader)
    warmup_steps = tc.get('warmup_steps', 1000)
    warmup    = LinearLR(optimizer, start_factor=0.01, end_factor=1.0,
                         total_iters=warmup_steps)
    cosine    = CosineAnnealingLR(optimizer, T_max=max(total_steps - warmup_steps, 1))
    scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine],
                             milestones=[warmup_steps])

    ema = EMA(model, decay=tc.get('ema_decay', 0.9999))

    start_epoch   = 0
    best_val_loss = float('inf')
    global_step   = 0

    if resume_path:
        ckpt          = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        ema.shadow    = ckpt['ema_shadow']
        start_epoch   = ckpt['epoch'] + 1
        global_step   = ckpt.get('global_step', start_epoch * len(train_loader))
        best_val_loss = ckpt.get('best_val_loss', float('inf'))
        if 'energy_mean' in ckpt:
            energy_mean = ckpt['energy_mean']
            energy_std  = ckpt['energy_std']
        print(f"Resumed at epoch {start_epoch}, step {global_step}")

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
            x0    = batch['coords'].to(device)
            e_raw = batch['energies'].to(device)
            e_z   = (e_raw - energy_mean) / energy_std

            B = x0.shape[0]
            t = torch.randint(0, diffusion.T, (B,), device=device)
            x_t, noise = diffusion.q_sample(x0, t)

            # ── Self-conditioning ─────────────────────────────────────────────
            # 50 % of batches: compute x0_sc from noise level t_sc = t + Δ.
            #
            # At inference step t, x0_sc came from the DDIM step at t+Δ (one
            # step earlier = slightly higher noise).  Training with t_sc = t+Δ
            # instead of t eliminates the training/inference mismatch that caused
            # the model to diverge at low t in the previous ZeroCoM implementation.
            #
            # 50 % of batches: x0_sc = zeros — teaches the "first step" path where
            # no prior x0 estimate is available.
            if torch.rand(1).item() < 0.5:
                with torch.no_grad():
                    t_sc    = (t + sc_delta).clamp(max=diffusion.T - 1)
                    x_t_sc, _ = diffusion.q_sample(x0, t_sc)
                    noise_prelim = model(x_t_sc, t_sc, energy_z=e_z,
                                        x0_self_cond=None)
                    x0_sc = diffusion._predict_x0_from_noise(
                        x_t_sc, t_sc, noise_prelim
                    ).clamp(-5, 5)
            else:
                x0_sc = torch.zeros_like(x0)

            # ── Main forward + loss ───────────────────────────────────────────
            noise_pred = model(x_t, t, energy_z=e_z, x0_self_cond=x0_sc)
            loss = (noise - noise_pred).pow(2).mean()

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

            pbar.set_postfix(
                loss = f"{loss.item():.4f}",
                lr   = f"{scheduler.get_last_lr()[0]:.2e}",
            )

        mean_train = epoch_loss / n_batches

        # ── Validation ────────────────────────────────────────────────────────
        mean_val = None

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
                    t     = torch.randint(0, diffusion.T, (B,), device=device)
                    x_t, noise = diffusion.q_sample(x0, t)
                    # Validate on the "no SC" path (x0_sc=None → zeros internally).
                    # This is the same distribution as the first inference step and
                    # gives a clean, reproducible signal for model selection.
                    noise_pred = model(x_t, t, energy_z=e_z, x0_self_cond=None)
                    l = (noise - noise_pred).pow(2).mean()
                    if phys_weight > 0.0 and physics is not None:
                        x0_pred = diffusion._predict_x0_from_noise(x_t, t, noise_pred)
                        snr_w   = diffusion.alphas_cumprod[t].to(device)
                        l = l + phys_weight * (snr_w * physics(x0_pred)).mean()
                    val_loss += l.item()
                    n_val    += 1

            mean_val = val_loss / n_val
            ema.restore(orig)

            print(
                f"\nEpoch {epoch+1:4d} | "
                f"train: {mean_train:.4f} | val: {mean_val:.4f} | step: {global_step}"
            )

            if mean_val < best_val_loss:
                best_val_loss = mean_val
                torch.save({
                    'epoch':         epoch,
                    'global_step':   global_step,
                    'model':         model.state_dict(),
                    'ema_shadow':    ema.shadow,
                    'optimizer':     optimizer.state_dict(),
                    'best_val_loss': best_val_loss,
                    'energy_mean':   energy_mean,
                    'energy_std':    energy_std,
                    'config':        config,
                }, ckpt_dir / 'best.pt')
                print(f"  -> best (val: {best_val_loss:.4f})")

        log({
            'epoch':       epoch + 1,
            'global_step': global_step,
            'train_loss':  mean_train,
            'val_loss':    mean_val,
            'lr':          scheduler.get_last_lr()[0],
        })

        if (epoch + 1) % tc.get('save_every', 50) == 0:
            torch.save({
                'epoch':         epoch,
                'global_step':   global_step,
                'model':         model.state_dict(),
                'ema_shadow':    ema.shadow,
                'optimizer':     optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'energy_mean':   energy_mean,
                'energy_std':    energy_std,
                'config':        config,
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
