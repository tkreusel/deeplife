"""
scripts/eval_v2/equivariance.py
=================================
SE(3) equivariance tests for all Chignolin model types.

Tests (adapted from scripts/check_equivariance.py):
  Test 1: Score-network equivariance — ‖model(Rx,t) − R·model(x,t)‖ / ‖model(x,t)‖
  Test 2: Full-pipeline equivariance — ‖f(Rx₀) − R·f(x₀)‖ / ‖f(x₀)‖
  Test 3: Distribution isotropy      — λ_max / λ_min of 3×3 positional covariance
  Test 4: Ensemble equivariance      — Wasserstein-1 distance between Rg distributions
           with and without rotating all input noises by a fixed rotation R

Expected results:
  EGNN / FlowMatch  : Test 1 ~1e-4, Test 2 ~1e-3, Test 3 ~1.0, Test 4 ~0
  Transformer / MLP : Test 1 ~1–5,  Test 2 ~1–5,  Test 3 >> 1, Test 4 >> 0
"""

import sys
import numpy as np
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from models.flow_matching     import ContinuousFlowMatching, ZeroCoMFlowMatching
from models.diffusion_zerocom import ZeroCoMGaussianDiffusion

# ── Pass/fail thresholds ──────────────────────────────────────────────────────
SCORE_THRESHOLD    = 1e-3
PIPELINE_THRESHOLD = 1e-2
ISOTROPY_THRESHOLD = 1.1    # λ_max/λ_min < this → ISOTROPIC
ANISO_THRESHOLD    = 2.0    # λ_max/λ_min > this → ANISOTROPIC


# ── Rotation helpers ──────────────────────────────────────────────────────────

def _random_rotation(device, proper: bool = True) -> torch.Tensor:
    """Haar-uniform rotation matrix. proper=True → det=+1 (SO(3))."""
    Q, R = torch.linalg.qr(torch.randn(3, 3, device=device))
    Q = Q * torch.sign(torch.diag(R)).unsqueeze(0)
    if not proper:
        Q[:, 0] *= -1
    return Q


def _apply_rotation(R: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Apply (3,3) rotation matrix R to coords (B, N, 3)."""
    return (R @ x.reshape(-1, 3, 1)).reshape(x.shape)


def _make_noise(diffusion, shape: tuple, device) -> torch.Tensor:
    """Zero-CoM noise for ZeroCoM models, plain Gaussian otherwise."""
    x = torch.randn(*shape, device=device)
    if isinstance(diffusion, (ZeroCoMFlowMatching, ZeroCoMGaussianDiffusion)):
        x = x - x.mean(dim=1, keepdim=True)
    return x


def _sample_timestep(diffusion, B: int, device) -> torch.Tensor:
    if isinstance(diffusion, ContinuousFlowMatching):
        t_val = torch.rand(1).item()
        return torch.full((B,), diffusion._scale_t(t_val), device=device)
    else:
        return torch.randint(0, diffusion.T, (B,), device=device, dtype=torch.long)


# ── Deterministic sampling from fixed initial noise ───────────────────────────

@torch.no_grad()
def _sample_from_noise(model, diffusion, x0: torch.Tensor,
                       ddim_steps: int = 50) -> torch.Tensor:
    """Run deterministic DDIM (eta=0) or Heun ODE from fixed initial noise x0."""
    x      = x0.clone()
    device = x0.device

    if isinstance(diffusion, ContinuousFlowMatching):
        is_zcom = isinstance(diffusion, ZeroCoMFlowMatching)
        dt      = 1.0 / ddim_steps

        def _com(v):
            return diffusion._remove_com(v) if is_zcom else v

        for i in range(ddim_steps):
            t_val = i / ddim_steps
            t     = torch.full((x.shape[0],), diffusion._scale_t(t_val), device=device)
            v1    = _com(model(x, t))
            x_pred = x + dt * v1
            if i < ddim_steps - 1:
                t_next = torch.full(
                    (x.shape[0],), diffusion._scale_t(t_val + dt), device=device
                )
                v2 = _com(model(x_pred, t_next))
                x  = x + dt * (v1 + v2) * 0.5
            else:
                x = x_pred
    else:
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

            dir_xt = (1.0 - alpha_prev).sqrt() * noise_pred
            x      = alpha_prev.sqrt() * x0_pred + dir_xt

    return x


# ── Test 1: Score-network equivariance ────────────────────────────────────────

@torch.no_grad()
def test1_score_network(model, diffusion, config: dict,
                        n_noise: int = 20, n_rotations: int = 50,
                        device: str = 'cpu') -> dict:
    """
    Measures ‖model(Rx,t) − R·model(x,t)‖ / ‖model(x,t)‖.

    Skipped for torsional / backbone IPA models (internal-coordinate parameterisation)
    and for energy-conditioned models where model() requires an energy_z argument.
    Returns {'proper': [...], 'reflection': [...], 'skipped': bool}.
    """
    mt = config.get('model_type', '')
    skip_types = {'torsion_flow_energy', 'torsion_transformer_energy', 'backbone_ipa_energy'}
    if mt in skip_types:
        return {'proper': [], 'reflection': [], 'skipped': True,
                'skip_reason': f"{mt} uses internal coordinates, not Cartesian score network"}

    model.eval()
    n_res        = config['data']['n_residues']
    use_backbone = getattr(diffusion, '_is_backbone', False)
    n_atoms      = n_res * 3 if use_backbone else n_res
    errors  = {'proper': [], 'reflection': []}

    for proper in (True, False):
        key = 'proper' if proper else 'reflection'
        for _ in range(n_noise):
            x = _make_noise(diffusion, (1, n_atoms, 3), device)
            t = _sample_timestep(diffusion, 1, device)

            # Forward pass — energy-conditioned models run unconditionally (energy_z=None)
            try:
                eps_x = model(x, t)
            except TypeError:
                eps_x = model(x, t, energy_z=None)

            rotations = [_random_rotation(device, proper=proper) for _ in range(n_rotations)]
            Rx_batch  = torch.cat([_apply_rotation(R, x) for R in rotations], dim=0)
            t_batch   = t.expand(n_rotations)

            try:
                eps_Rx = model(Rx_batch, t_batch)
            except TypeError:
                e_z = torch.zeros(n_rotations, device=device)
                eps_Rx = model(Rx_batch, t_batch, energy_z=None)

            R_eps_x = torch.stack([_apply_rotation(R, eps_x) for R in rotations]).squeeze(1)

            numer = (eps_Rx - R_eps_x).reshape(n_rotations, -1).norm(dim=-1)
            denom = eps_x.reshape(1, -1).norm().clamp(min=1e-8)
            errors[key].extend((numer / denom).cpu().tolist())

    errors['skipped'] = False
    return errors


# ── Test 2: Full-pipeline equivariance ────────────────────────────────────────

@torch.no_grad()
def test2_pipeline(model, diffusion, config: dict,
                   n_noise: int = 20, n_rotations: int = 50,
                   ddim_steps: int = 50, device: str = 'cpu') -> dict:
    """
    Measures ‖f(Rx₀) − Rf(x₀)‖ / ‖f(x₀)‖.
    Works for all model types that have a deterministic DDIM/Heun ODE.
    Skipped for torsional models (non-Cartesian pipeline).
    """
    mt = config.get('model_type', '')
    skip_types = {'torsion_flow_energy', 'torsion_transformer_energy', 'backbone_ipa_energy'}
    if mt in skip_types:
        return {'proper': [], 'reflection': [], 'skipped': True,
                'skip_reason': f"{mt} uses internal coordinates"}

    model.eval()
    n_res        = config['data']['n_residues']
    use_backbone = getattr(diffusion, '_is_backbone', False)
    n_atoms      = n_res * 3 if use_backbone else n_res
    errors  = {'proper': [], 'reflection': []}

    for proper in (True, False):
        key = 'proper' if proper else 'reflection'
        for _ in range(n_noise):
            x0 = _make_noise(diffusion, (1, n_atoms, 3), device)
            y0 = _sample_from_noise(model, diffusion, x0, ddim_steps)

            rotations  = [_random_rotation(device, proper=proper) for _ in range(n_rotations)]
            Rx0_batch  = torch.cat([_apply_rotation(R, x0) for R in rotations], dim=0)
            y_R_batch  = _sample_from_noise(model, diffusion, Rx0_batch, ddim_steps)

            denom = y0.reshape(1, -1).norm().clamp(min=1e-8)
            for j, R in enumerate(rotations):
                y_R   = y_R_batch[j : j + 1]
                Ry0   = _apply_rotation(R, y0)
                numer = (y_R - Ry0).norm()
                errors[key].append((numer / denom).item())

    errors['skipped'] = False
    return errors


# ── Test 3: Distribution isotropy ─────────────────────────────────────────────

@torch.no_grad()
def test3_isotropy(model, diffusion, config: dict,
                   n_generate: int = 500, ddim_steps: int = 50,
                   device: str = 'cpu', batch_size: int = 64) -> dict:
    """
    λ_max / λ_min of 3×3 positional covariance of generated ensemble.
    Equivariant model → ~1.0. Non-equivariant → >> 1.
    """
    from .model_utils import generate

    model.eval()
    n_res = config['data']['n_residues']
    coords = generate(model, diffusion, n_generate, n_res,
                      coord_scale=1.0, ddim_steps=ddim_steps,
                      device=device, batch_size=batch_size)
    # Use Cα atoms for isotropy measurement
    from .model_utils import ca_from_coords
    ca = ca_from_coords(coords)

    pts = torch.from_numpy(ca.reshape(-1, 3)).float()
    pts = pts - pts.mean(dim=0, keepdim=True)
    cov = (pts.T @ pts) / len(pts)
    evals = torch.linalg.eigvalsh(cov)
    ratio = (evals[2] / evals[0].clamp(min=1e-8)).item()

    return {
        'evals': evals.tolist(),
        'ratio': ratio,
        'verdict': ('ISOTROPIC'   if ratio < ISOTROPY_THRESHOLD else
                    'ANISOTROPIC' if ratio > ANISO_THRESHOLD    else 'BORDERLINE'),
    }


# ── Test 4: Ensemble-level equivariance ───────────────────────────────────────

@torch.no_grad()
def test4_ensemble(model, diffusion, config: dict,
                   n_generate: int = 200, ddim_steps: int = 50,
                   device: str = 'cpu', batch_size: int = 64) -> dict:
    """
    Generate two ensembles: one from plain noise, one from the same noise rotated
    by a fixed rotation R. Compare their Rg distributions via Wasserstein-1 distance.

    For equivariant models: Rg is rotation-invariant, so W1 ≈ 0.
    For non-equivariant models: different noise orientations → different ensembles → W1 > 0.
    """
    from .model_utils import generate, ca_from_coords
    from .physics_metrics import radius_of_gyration

    mt = config.get('model_type', '')
    skip_types = {'torsion_flow_energy', 'torsion_transformer_energy', 'backbone_ipa_energy'}
    if mt in skip_types:
        return {'wasserstein1_rg': float('nan'), 'skipped': True,
                'skip_reason': f"{mt} uses internal coordinates"}

    model.eval()
    n_res        = config['data']['n_residues']
    use_backbone = getattr(diffusion, '_is_backbone', False)
    n_atoms      = n_res * 3 if use_backbone else n_res

    # Generate fixed noise batch
    all_x0 = []
    n_done = 0
    while n_done < n_generate:
        bs = min(batch_size, n_generate - n_done)
        x0 = _make_noise(diffusion, (bs, n_atoms, 3), device)
        all_x0.append(x0)
        n_done += bs
    x0_all = torch.cat(all_x0, dim=0)[:n_generate]

    # Fixed rotation
    R = _random_rotation(device, proper=True)

    # Ensemble A: from x0
    samples_a, samples_b = [], []
    for i in range(0, n_generate, batch_size):
        chunk   = x0_all[i:i+batch_size]
        rot_chunk = _apply_rotation(R, chunk)
        ya = _sample_from_noise(model, diffusion, chunk,     ddim_steps).cpu().numpy()
        yb = _sample_from_noise(model, diffusion, rot_chunk, ddim_steps).cpu().numpy()
        samples_a.append(ya)
        samples_b.append(yb)

    a = np.concatenate(samples_a, axis=0)
    b = np.concatenate(samples_b, axis=0)

    rg_a = radius_of_gyration(ca_from_coords(a))
    rg_b = radius_of_gyration(ca_from_coords(b))

    # Wasserstein-1 (sorted arrays trick for 1D)
    w1 = float(np.abs(np.sort(rg_a) - np.sort(rg_b)).mean())

    return {
        'wasserstein1_rg': w1,
        'rg_mean_a':       float(rg_a.mean()),
        'rg_mean_b':       float(rg_b.mean()),
        'skipped':         False,
    }


# ─────────────────────────────────────────────────────────────────────────────
# RUN ALL TESTS FOR ONE MODEL
# ─────────────────────────────────────────────────────────────────────────────

def run_equivariance_tests(model, diffusion, config: dict,
                           n_noise: int = 20, n_rotations: int = 50,
                           n_generate: int = 500, ddim_steps: int = 50,
                           device: str = 'cpu') -> dict:
    """Run all 4 equivariance tests and return a combined results dict."""
    print("    Test 1: score-network equivariance …")
    t1 = test1_score_network(model, diffusion, config,
                              n_noise, n_rotations, device)

    print("    Test 2: full-pipeline equivariance …")
    t2 = test2_pipeline(model, diffusion, config,
                         n_noise, n_rotations, ddim_steps, device)

    print("    Test 3: distribution isotropy …")
    t3 = test3_isotropy(model, diffusion, config,
                         n_generate, ddim_steps, device)

    print("    Test 4: ensemble equivariance (Wasserstein-1) …")
    t4 = test4_ensemble(model, diffusion, config,
                         min(n_generate, 200), ddim_steps, device)

    return {
        'test1': t1,
        'test2': t2,
        'test3': t3,
        'test4': t4,
        'model_type': config.get('model_type', 'unknown'),
    }


# ─────────────────────────────────────────────────────────────────────────────
# CONSOLE REPORT
# ─────────────────────────────────────────────────────────────────────────────

def _fmt(arr) -> str:
    a = np.array(arr)
    if len(a) == 0: return "N/A"
    return f"mean={a.mean():.5f}  std={a.std():.5f}  max={a.max():.5f}"


def print_equivariance_results(results: dict, label: str = "Model"):
    print(f"\n{'━' * 72}")
    print(f"  SE(3) Equivariance — {label}  [{results['model_type']}]")
    print(f"{'━' * 72}")

    t1 = results['test1']
    if t1['skipped']:
        print(f"  Test 1: SKIPPED — {t1.get('skip_reason', '')}")
    else:
        prop_pass = "PASS" if (np.array(t1['proper']).mean() < SCORE_THRESHOLD) else "FAIL"
        refl_pass = "PASS" if (np.array(t1['reflection']).mean() < SCORE_THRESHOLD) else "FAIL"
        print(f"  Test 1 (score network):")
        print(f"    Proper:      {_fmt(t1['proper'])}  [{prop_pass}]")
        print(f"    Reflections: {_fmt(t1['reflection'])}  [{refl_pass}]")

    t2 = results['test2']
    if t2['skipped']:
        print(f"  Test 2: SKIPPED — {t2.get('skip_reason', '')}")
    else:
        prop_pass = "PASS" if (np.array(t2['proper']).mean() < PIPELINE_THRESHOLD) else "FAIL"
        refl_pass = "PASS" if (np.array(t2['reflection']).mean() < PIPELINE_THRESHOLD) else "FAIL"
        print(f"  Test 2 (full pipeline):")
        print(f"    Proper:      {_fmt(t2['proper'])}  [{prop_pass}]")
        print(f"    Reflections: {_fmt(t2['reflection'])}  [{refl_pass}]")

    t3 = results['test3']
    ev = t3['evals']
    print(f"  Test 3 (isotropy):")
    print(f"    λ_min/λ_mid/λ_max = {ev[0]:.4f} / {ev[1]:.4f} / {ev[2]:.4f}")
    print(f"    ratio = {t3['ratio']:.3f}  [{t3['verdict']}]")

    t4 = results['test4']
    if t4['skipped']:
        print(f"  Test 4: SKIPPED — {t4.get('skip_reason', '')}")
    else:
        print(f"  Test 4 (ensemble Rg W1 = {t4['wasserstein1_rg']:.4f})")

    print(f"{'━' * 72}")
