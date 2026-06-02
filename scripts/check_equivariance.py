"""
scripts/check_equivariance.py
==============================
Empirical SE(3)-equivariance tests for Chignolin diffusion models.

Three tests per checkpoint
--------------------------
1. Score-network equivariance — single forward pass, exact, no sampling:
       measures  ‖model(R·x, t) − R·model(x, t)‖ / ‖model(x, t)‖
       for random (x, t, R) triples.  Tests proper rotations (det=+1)
       and reflections (det=−1) separately.

2. Full-pipeline equivariance — deterministic sampling from fixed noise:
       measures  ‖f(R·x₀) − R·f(x₀)‖ / ‖f(x₀)‖
       where f is either DDIM (eta=0) or Heun's ODE.  No averaging needed
       because both samplers are fully deterministic given the same x₀.

3. Distribution isotropy — practical consequence of equivariance:
       an equivariant model with isotropic Gaussian noise generates an
       isotropic distribution of structures (all orientations equally likely).
       Metric: eigenvalue ratio λ_max/λ_min of the 3×3 positional covariance.
       Equivariant ≈ 1.0,  non-equivariant potentially >> 1.

Expected results
----------------
  EGNN / FlowMatch (equivariant) : Test 1 error ~1e-4,  Test 2 ~1e-3,  ratio ~1.0
  Transformer / MLP (non-equivariant): Test 1 error ~1–5, Test 2 ~1–5, ratio >> 1

Usage
-----
# Single model
python scripts/check_equivariance.py \\
    --ckpt checkpoints/flowmatch_physics/v3/best.pt

# Compare equivariant vs non-equivariant (+ optional plot)
python scripts/check_equivariance.py \\
    --ckpt     checkpoints/flowmatch_physics/v3/best.pt \\
    --ckpt_ref checkpoints/egnn/v1/best.pt \\
               checkpoints/baseline/v3/best.pt \\
    --labels   FlowMatch+Physics EGNN-DDPM Transformer \\
    --save     plots/equivariance_comparison.png

# Quick smoke-test on CPU (~1 min)
python scripts/check_equivariance.py \\
    --ckpt checkpoints/flowmatch/v2/best.pt \\
    --n_noise 3 --n_rotations 5 --n_generate 30 --steps 10
"""

import sys
import argparse
import numpy as np
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.flow_matching     import ContinuousFlowMatching, ZeroCoMFlowMatching
from models.diffusion_zerocom import ZeroCoMGaussianDiffusion


# ── Pass / fail thresholds ────────────────────────────────────────────────────
SCORE_THRESHOLD    = 1e-3   # mean relative error  (Test 1 — score network)
PIPELINE_THRESHOLD = 1e-2   # mean relative error  (Test 2 — full pipeline)
ISOTROPY_THRESHOLD = 1.1    # λ_max/λ_min < this → ISOTROPIC   (Test 3)
ANISO_THRESHOLD    = 2.0    # λ_max/λ_min > this → ANISOTROPIC (Test 3)


# ── Rotation helpers ──────────────────────────────────────────────────────────

def _random_rotation(device, proper: bool = True) -> torch.Tensor:
    """
    Haar-uniform rotation matrix.
    proper=True  → det = +1  (SO(3))
    proper=False → det = −1  (O(3) reflection)

    Same QR idiom as data/transforms.py:74–90.
    """
    Q, R = torch.linalg.qr(torch.randn(3, 3, device=device))
    Q = Q * torch.sign(torch.diag(R)).unsqueeze(0)   # fix column signs
    if not proper:
        Q[:, 0] *= -1   # flip determinant to -1
    return Q   # (3, 3)


def _apply_rotation(R: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Apply (3, 3) orthogonal matrix R to coords of shape (B, N, 3)."""
    return (R @ x.reshape(-1, 3, 1)).reshape(x.shape)


# ── Noise helpers ─────────────────────────────────────────────────────────────

def _make_noise(diffusion, shape: tuple, device) -> torch.Tensor:
    """
    Zero-CoM Gaussian noise for ZeroCoM models, plain Gaussian otherwise.
    Matches each diffusion class's own noise-sampling convention.
    """
    x = torch.randn(*shape, device=device)
    if isinstance(diffusion, (ZeroCoMFlowMatching, ZeroCoMGaussianDiffusion)):
        x = x - x.mean(dim=1, keepdim=True)
    return x


def _sample_timestep(diffusion, B: int, device) -> torch.Tensor:
    """
    Random timestep of shape (B,) appropriate for the diffusion type:
      Flow matching: float in [0, 999]  (continuous, scaled from [0, 1])
      DDPM:          int   in [0, T-1]  (discrete)
    """
    if isinstance(diffusion, ContinuousFlowMatching):
        t_val = torch.rand(1).item()
        return torch.full((B,), diffusion._scale_t(t_val), device=device)
    else:
        return torch.randint(0, diffusion.T, (B,), device=device, dtype=torch.long)


# ── Deterministic sampling from fixed initial noise ───────────────────────────

@torch.no_grad()
def _sample_from_noise(
    model,
    diffusion,
    x0:         torch.Tensor,   # (B, N, 3) — fixed initial noise
    ddim_steps: int = 50,
) -> torch.Tensor:
    """
    Deterministic sampling from a supplied initial noise tensor.

    Replicates the DDIM (eta=0) loop from GaussianDiffusion / ZeroCoMGaussianDiffusion
    and the Heun ODE from ZeroCoMFlowMatching — without generating new noise internally.
    This lets the caller control the starting point for equivariance testing.

    The two samplers:
      Flow matching → Heun predictor-corrector ODE (mirrors flow_matching.py:326–348)
      DDPM          → DDIM with eta=0; sigma=0 makes all stochastic terms vanish
                      (mirrors diffusion_zerocom.py:189–224)
    """
    x      = x0.clone()
    device = x0.device

    if isinstance(diffusion, ContinuousFlowMatching):
        is_zcom = isinstance(diffusion, ZeroCoMFlowMatching)
        dt      = 1.0 / ddim_steps

        def _com(v: torch.Tensor) -> torch.Tensor:
            return diffusion._remove_com(v) if is_zcom else v

        for i in range(ddim_steps):
            t_val  = i / ddim_steps
            t      = torch.full((x.shape[0],), diffusion._scale_t(t_val), device=device)
            v1     = _com(model(x, t))
            x_pred = x + dt * v1

            if i < ddim_steps - 1:   # Heun corrector (skip on last step)
                t_next = torch.full(
                    (x.shape[0],), diffusion._scale_t(t_val + dt), device=device
                )
                v2 = _com(model(x_pred, t_next))
                x  = x + dt * (v1 + v2) * 0.5
            else:
                x = x_pred

    else:
        # DDIM, eta=0  →  sigma=0  →  no stochastic terms
        is_zcom   = isinstance(diffusion, ZeroCoMGaussianDiffusion)
        step_size = diffusion.T // ddim_steps
        timesteps = list(range(0, diffusion.T, step_size))[::-1]

        for i, t_int in enumerate(timesteps):
            t_tensor   = torch.full((x.shape[0],), t_int, device=device, dtype=torch.long)
            t_prev     = timesteps[i + 1] if i + 1 < len(timesteps) else 0
            alpha_t    = diffusion.alphas_cumprod[t_int]
            alpha_prev = diffusion.alphas_cumprod[t_prev]

            noise_pred = model(x, t_tensor)
            if is_zcom:
                noise_pred = diffusion._remove_com(noise_pred)

            x0_pred = (x - (1.0 - alpha_t).sqrt() * noise_pred) / alpha_t.sqrt()
            if is_zcom:
                x0_pred = diffusion._remove_com(x0_pred.clamp(-5, 5))
            else:
                x0_pred = x0_pred.clamp(-5, 5)

            # eta=0 → sigma=0 → dir_xt simplification from the general formula
            dir_xt = (1.0 - alpha_prev).sqrt() * noise_pred
            x      = alpha_prev.sqrt() * x0_pred + dir_xt
            # (no noise term: sigma * randn == 0)

    return x


# ── Label utilities ───────────────────────────────────────────────────────────

def _derive_label(config: dict, used_labels: set) -> str:
    """Auto-derive a unique display label from checkpoint config."""
    name_map = {
        'egnn':        'EGNN',
        'flowmatch':   'FlowMatch',
        'mlp':         'MLP',
        'transformer': 'Transformer',
    }
    mt   = config['model_type']
    phys = config.get('training', {}).get('physics_weight', 0.0) > 0
    base = name_map.get(mt, mt.upper())
    if phys:
        base += '+Physics'
    label, suffix = base, 2
    while label in used_labels:
        label = f"{base}_{suffix}"; suffix += 1
    return label


def _is_equivariant(config: dict) -> bool:
    return config['model_type'] in ('egnn', 'flowmatch')


# ── Test 1: score-network equivariance ────────────────────────────────────────

@torch.no_grad()
def test1_score_network(
    model,
    diffusion,
    config:      dict,
    n_noise:     int = 20,
    n_rotations: int = 50,
    device:      str = 'cpu',
) -> dict:
    """
    Score-network equivariance: ||model(Rx,t) − R·model(x,t)|| / ||model(x,t)||

    Vectorised: all n_rotations rotated variants batched into one forward pass
    per noise vector.  This gives O(n_noise) model calls rather than
    O(n_noise × n_rotations).
    """
    model.eval()
    n_res  = config['data']['n_residues']
    errors: dict = {'proper': [], 'reflection': []}

    for proper in (True, False):
        key = 'proper' if proper else 'reflection'

        for _ in range(n_noise):
            x = _make_noise(diffusion, (1, n_res, 3), device)   # (1, N, 3)
            t = _sample_timestep(diffusion, 1, device)           # (1,)

            eps_x = model(x, t)   # (1, N, 3)

            # Batch all rotations together for a single forward pass
            rotations = [_random_rotation(device, proper=proper) for _ in range(n_rotations)]
            Rx_batch  = torch.cat([_apply_rotation(R, x) for R in rotations], dim=0)  # (n_rot, N, 3)
            t_batch   = t.expand(n_rotations)                                          # (n_rot,)

            eps_Rx  = model(Rx_batch, t_batch)                                         # (n_rot, N, 3)
            R_eps_x = torch.stack([_apply_rotation(R, eps_x) for R in rotations])     # (n_rot, 1, N, 3)
            R_eps_x = R_eps_x.squeeze(1)                                               # (n_rot, N, 3)

            numer = (eps_Rx - R_eps_x).reshape(n_rotations, -1).norm(dim=-1)          # (n_rot,)
            denom = eps_x.reshape(1, -1).norm().clamp(min=1e-8)
            errors[key].extend((numer / denom).cpu().tolist())

    return errors


# ── Test 2: full-pipeline equivariance ────────────────────────────────────────

@torch.no_grad()
def test2_pipeline(
    model,
    diffusion,
    config:      dict,
    n_noise:     int = 20,
    n_rotations: int = 50,
    ddim_steps:  int = 50,
    device:      str = 'cpu',
) -> dict:
    """
    Full-pipeline equivariance: ||f(R·x₀) − R·f(x₀)|| / ||f(x₀)||

    For each noise vector x₀:
      1. Run deterministic sampling from x₀           → y₀     (batch size 1)
      2. Batch all rotated noises R_i·x₀ into one run → y_R    (batch size n_rotations)
      3. Compare y_R[i] vs R_i·y₀ for each i

    This is the user's core test, with the key insight that NO averaging is
    needed because both DDIM (eta=0) and Heun's ODE are fully deterministic.
    """
    model.eval()
    n_res  = config['data']['n_residues']
    errors: dict = {'proper': [], 'reflection': []}

    for proper in (True, False):
        key = 'proper' if proper else 'reflection'

        for _ in range(n_noise):
            x0 = _make_noise(diffusion, (1, n_res, 3), device)             # (1, N, 3)
            y0 = _sample_from_noise(model, diffusion, x0, ddim_steps)     # (1, N, 3)

            rotations = [_random_rotation(device, proper=proper) for _ in range(n_rotations)]

            # Stack all rotated noises and run the full pipeline once
            Rx0_batch = torch.cat(
                [_apply_rotation(R, x0) for R in rotations], dim=0
            )                                                               # (n_rot, N, 3)
            y_R_batch = _sample_from_noise(
                model, diffusion, Rx0_batch, ddim_steps
            )                                                               # (n_rot, N, 3)

            denom = y0.reshape(1, -1).norm().clamp(min=1e-8)

            for j, R in enumerate(rotations):
                y_R   = y_R_batch[j : j + 1]           # (1, N, 3)
                Ry0   = _apply_rotation(R, y0)          # (1, N, 3)
                numer = (y_R - Ry0).norm()
                errors[key].append((numer / denom).item())

    return errors


# ── Test 3: distribution isotropy ─────────────────────────────────────────────

@torch.no_grad()
def test3_isotropy(
    model,
    diffusion,
    config:     dict,
    n_generate: int = 500,
    ddim_steps: int = 50,
    device:     str = 'cpu',
    batch_size: int = 64,
) -> dict:
    """
    Distribution isotropy: eigenvalue ratio of the 3×3 positional covariance.

    An equivariant model with isotropic Gaussian noise must generate an
    isotropic distribution of structures — because f(Rx₀) = Rf(x₀) and
    Gaussian noise is rotation-invariant, so rotating the noise gives the
    same distribution as rotating the output.

    Metric: λ_max / λ_min of the 3×3 covariance of all generated Cα positions.
      ~1.0 → isotropic (equivariant)
      >> 1 → preferred orientation (non-equivariant)
    """
    model.eval()
    n_res       = config['data']['n_residues']
    all_samples = []
    n_done      = 0

    while n_done < n_generate:
        bs = min(batch_size, n_generate - n_done)
        x0 = _make_noise(diffusion, (bs, n_res, 3), device)
        y  = _sample_from_noise(model, diffusion, x0, ddim_steps)
        all_samples.append(y.cpu())
        n_done += bs

    samples = torch.cat(all_samples, dim=0)   # (n_generate, N, 3)

    # Pool all Cα positions: (n_generate * N, 3)
    pts = samples.reshape(-1, 3)
    pts = pts - pts.mean(dim=0, keepdim=True)

    cov   = (pts.T @ pts) / len(pts)              # (3, 3)  symmetric positive-semidefinite
    evals = torch.linalg.eigvalsh(cov)            # ascending: [λ_min, λ_mid, λ_max]
    ratio = (evals[2] / evals[0].clamp(min=1e-8)).item()

    return {
        'evals': evals.tolist(),   # [λ_min, λ_mid, λ_max]
        'ratio': ratio,
    }


# ── Console output ────────────────────────────────────────────────────────────

def _fmt_errs(arr) -> str:
    a = np.array(arr)
    return f"mean={a.mean():.7f}  std={a.std():.6f}  max={a.max():.6f}"


def _verdict1(arr) -> str:
    return 'PASS' if np.array(arr).mean() < SCORE_THRESHOLD    else 'FAIL'

def _verdict2(arr) -> str:
    return 'PASS' if np.array(arr).mean() < PIPELINE_THRESHOLD else 'FAIL'


def print_model_header(label: str, config: dict):
    mt   = config['model_type']
    phys = config.get('training', {}).get('physics_weight', 0.0) > 0
    tags = [mt, 'equivariant' if _is_equivariant(config) else 'non-equivariant']
    if phys:
        tags.append('physics')
    print(f"\n{'━' * 70}")
    print(f"  {label}  [{', '.join(tags)}]")
    print(f"{'━' * 70}")


def print_test1(result: dict):
    print(f"  Test 1: Score-network equivariance")
    print(f"    Proper rotations:  {_fmt_errs(result['proper'])}  "
          f"{_verdict1(result['proper'])}")
    print(f"    Reflections:       {_fmt_errs(result['reflection'])}  "
          f"{_verdict1(result['reflection'])}")


def print_test2(result: dict):
    print(f"  Test 2: Full pipeline equivariance")
    print(f"    Proper rotations:  {_fmt_errs(result['proper'])}  "
          f"{_verdict2(result['proper'])}")
    print(f"    Reflections:       {_fmt_errs(result['reflection'])}  "
          f"{_verdict2(result['reflection'])}")


def print_test3(result: dict):
    ev      = result['evals']
    r       = result['ratio']
    verdict = ('ISOTROPIC'   if r < ISOTROPY_THRESHOLD else
               'ANISOTROPIC' if r > ANISO_THRESHOLD    else 'BORDERLINE')
    print(f"  Test 3: Distribution isotropy")
    print(f"    Eigenvalues:       {ev[0]:.4f} / {ev[1]:.4f} / {ev[2]:.4f}  "
          f"ratio={r:.3f}  {verdict}")


# ── Optional 3-panel plot ─────────────────────────────────────────────────────

def plot_summary(results_dict: dict, save_path: str):
    """3-panel bar chart comparing equivariance errors and isotropy across models."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — skipping plot")
        return

    labels   = list(results_dict.keys())
    n_models = len(labels)
    colors   = ['#C44E52', '#4C72B0', '#55A868', '#8172B3', '#CCB974']

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle("SE(3) Equivariance Analysis", fontsize=13, y=1.01)

    x_pos = np.arange(2)       # 0 = proper rotations, 1 = reflections
    bar_w = 0.8 / n_models

    for ax_idx, (test_key, threshold, title) in enumerate([
        ('test1', SCORE_THRESHOLD,    'Test 1: Score-Network\nEquivariance Error'),
        ('test2', PIPELINE_THRESHOLD, 'Test 2: Full-Pipeline\nEquivariance Error'),
    ]):
        ax = axes[ax_idx]
        for m_idx, lbl in enumerate(labels):
            res   = results_dict[lbl]
            prop  = np.array(res[f'{test_key}_proper'])
            refl  = np.array(res[f'{test_key}_refl'])
            means = [prop.mean(), refl.mean()]
            stds  = [prop.std(),  refl.std()]
            offsets = x_pos + (m_idx - n_models / 2.0 + 0.5) * bar_w
            ax.bar(offsets, means, bar_w * 0.9,
                   yerr=stds, color=colors[m_idx % len(colors)],
                   alpha=0.8, label=lbl, capsize=3, error_kw={'lw': 1.2})

        ax.set_yscale('log')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(['Proper\nrotations', 'Reflections'], fontsize=9)
        ax.set_ylabel('Relative equivariance error (log scale)', fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.axhline(threshold, color='k', linestyle='--', lw=1.2,
                   alpha=0.5, label='Pass threshold')
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3, axis='y')

    # Panel 3: isotropy ratio (horizontal bar)
    ax     = axes[2]
    ratios = [results_dict[lbl]['test3']['ratio'] for lbl in labels]
    ax.barh(labels, ratios,
            color=[colors[i % len(colors)] for i in range(n_models)], alpha=0.8)
    ax.axvline(ISOTROPY_THRESHOLD, color='green', linestyle='--', lw=1.5,
               label=f'Isotropic  (< {ISOTROPY_THRESHOLD})')
    ax.axvline(ANISO_THRESHOLD,    color='red',   linestyle='--', lw=1.5,
               label=f'Anisotropic (> {ANISO_THRESHOLD})')
    ax.set_xlabel('Isotropy ratio  λ_max / λ_min', fontsize=9)
    ax.set_title('Test 3: Distribution\nIsotropy', fontsize=10)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, axis='x')

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved → {save_path}")
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=(
            "Empirically test SE(3)-equivariance of Chignolin diffusion models.\n"
            "Runs three tests per checkpoint:\n"
            "  1. Score-network equivariance  (single forward pass)\n"
            "  2. Full-pipeline equivariance  (deterministic DDIM/Heun ODE)\n"
            "  3. Distribution isotropy       (eigenvalue ratio of ensemble covariance)\n"
        ),
    )
    p.add_argument('--ckpt',         required=True,
                   help='Primary checkpoint path')
    p.add_argument('--ckpt_ref',     nargs='*', default=[],
                   help='Additional checkpoints for side-by-side comparison')
    p.add_argument('--labels',       nargs='*', default=None,
                   help='Display labels — must match total number of checkpoints')
    p.add_argument('--n_noise',      type=int, default=20,
                   help='Random noise vectors for Tests 1 and 2')
    p.add_argument('--n_rotations',  type=int, default=50,
                   help='Rotations per noise vector (Tests 1 and 2)')
    p.add_argument('--n_generate',   type=int, default=500,
                   help='Structures generated for isotropy test (Test 3)')
    p.add_argument('--steps',        type=int, default=50,
                   help='Deterministic sampling steps (DDIM eta=0 or Heun ODE)')
    p.add_argument('--seed',         type=int, default=0,
                   help='Random seed for reproducibility')
    p.add_argument('--save',         default=None,
                   help='Path to save the 3-panel comparison plot')
    args = p.parse_args()

    all_ckpts = [args.ckpt] + list(args.ckpt_ref)
    if args.labels is not None and len(args.labels) != len(all_ckpts):
        p.error(f"--labels has {len(args.labels)} entries but "
                f"{len(all_ckpts)} checkpoints provided")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"Device:       {device}")
    print(f"n_noise={args.n_noise}  n_rotations={args.n_rotations}  "
          f"n_generate={args.n_generate}  steps={args.steps}")
    print(f"Checkpoints:  {len(all_ckpts)}")

    from scripts.evaluate import load_model_from_ckpt

    results_dict = {}
    used_labels  = set()

    for i, ckpt_path in enumerate(all_ckpts):
        print(f"\n{'─' * 70}")
        print(f"Checkpoint {i + 1}/{len(all_ckpts)}: {ckpt_path}")

        model, diffusion, config, _coord_scale = load_model_from_ckpt(ckpt_path, device)
        label = args.labels[i] if args.labels else _derive_label(config, used_labels)
        used_labels.add(label)

        print_model_header(label, config)

        print(f"  Running Test 1 (score network, {args.n_noise}×{args.n_rotations} samples)…")
        t1 = test1_score_network(
            model, diffusion, config, args.n_noise, args.n_rotations, device
        )
        print_test1(t1)

        print(f"  Running Test 2 (full pipeline, {args.n_noise}×{args.n_rotations} samples, "
              f"{args.steps} steps)…")
        t2 = test2_pipeline(
            model, diffusion, config, args.n_noise, args.n_rotations, args.steps, device
        )
        print_test2(t2)

        print(f"  Running Test 3 (isotropy, {args.n_generate} samples, {args.steps} steps)…")
        t3 = test3_isotropy(
            model, diffusion, config, args.n_generate, args.steps, device
        )
        print_test3(t3)

        print(f"{'━' * 70}")

        results_dict[label] = {
            'test1_proper': t1['proper'],
            'test1_refl':   t1['reflection'],
            'test2_proper': t2['proper'],
            'test2_refl':   t2['reflection'],
            'test3':        t3,
        }

    if args.save:
        plot_summary(results_dict, args.save)


if __name__ == '__main__':
    main()
