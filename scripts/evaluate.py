"""
scripts/evaluate.py
====================
Comprehensive evaluation: generate samples from one or more checkpoints
and compare them against the test set with structural and physics metrics.

Handles all model types automatically by reading model_type from the saved config:
  'egnn'        — EGNN + DDPM
  'transformer' — Transformer + DDPM
  'mlp'         — MLP + DDPM
  'flowmatch'   — EGNN + OT-CFM (with or without physics constraints)

Usage
-----
# Single model
python scripts/evaluate.py \\
    --ckpt checkpoints/flowmatch_physics/v3/best.pt \\
    --test data/test.npz --n 1000 --save plots/physics_eval.png

# Compare N models side-by-side (physics-constrained vs baseline)
python scripts/evaluate.py \\
    --ckpt     checkpoints/flowmatch_physics/v3/best.pt \\
    --ckpt_ref checkpoints/flowmatch/v2/best.pt \\
               checkpoints/egnn/v1/best.pt \\
    --labels   FlowMatch+Physics FlowMatch EGNN-DDPM \\
    --test     data/test.npz --n 1000 --save plots/comparison.png

# Save PDB files (20 structures from the primary model)
python scripts/evaluate.py \\
    --ckpt checkpoints/flowmatch_physics/v3/best.pt \\
    --test data/test.npz --n 500 --save_pdb outputs/physics_samples

Output
------
  Console:      two-section table — structural metrics + physics quality metrics
  plots/*.png:  6-panel plot: bond dist, Rg, end-to-end, per-residue flexibility,
                              bond validity thresholds, per-bond RMSE from ideal
  metrics.json: all numbers for downstream analysis
  PDB files:    Cα-only PDB files for PyMOL visualisation
"""

import sys
import json
import argparse
import numpy as np
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.diffusion_zerocom import ZeroCoMGaussianDiffusion
from models.diffusion          import GaussianDiffusion


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _strip_compile_prefix(state_dict: dict) -> dict:
    """
    Remove the '_orig_mod.' prefix that torch.compile adds to parameter names.
    Training scripts create EMA after torch.compile, so the EMA shadow may
    contain '_orig_mod.*' keys that don't match an uncompiled model at inference.
    """
    return {
        (k[len('_orig_mod.'):] if k.startswith('_orig_mod.') else k): v
        for k, v in state_dict.items()
    }


# ─────────────────────────────────────────────────────────────────────────────
# MODEL LOADING  (auto-detects model_type from checkpoint)
# ─────────────────────────────────────────────────────────────────────────────

def load_model_from_ckpt(ckpt_path: str, device: str):
    """
    Load model + diffusion from any checkpoint in this repo.
    Reads model_type from the saved config to pick the right class.

    Supported model_type values:
      'egnn'        — EGNNScoreNetwork + ZeroCoMGaussianDiffusion (DDPM)
      'mlp'         — MLPScoreNetwork  + GaussianDiffusion (DDPM)
      'transformer'       — TransformerScoreNetwork + GaussianDiffusion (DDPM)
      'transformer_adaln' — AdaLNTransformerScoreNetwork + GaussianDiffusion (DDPM)
      'flowmatch'   — EGNNScoreNetwork + ZeroCoMFlowMatching (OT-CFM)

    Returns (model, diffusion, config, coord_scale)
    """
    from models.diffusion_zerocom import ZeroCoMGaussianDiffusion
    from models.diffusion        import GaussianDiffusion

    ckpt   = torch.load(ckpt_path, map_location=device)
    config = ckpt['config']
    mt     = config['model_type']

    # ── Build model + diffusion (one branch per model_type) ───────────────
    if mt == 'egnn_adaln':
        from models.egnn_adaln import EGNNAdaLNScoreNetwork
        mc    = config['model']
        model = EGNNAdaLNScoreNetwork(
            n_residues       = config['data']['n_residues'],
            node_dim         = mc['hidden_dim'],
            edge_dim         = mc.get('edge_dim',         64),
            time_dim         = mc.get('time_dim',         64),
            n_layers         = mc['n_layers'],
            energy_dim       = mc.get('energy_dim',       32),
            energy_drop_prob = mc.get('energy_drop_prob', 0.15),
        )
        dc        = config['diffusion']
        diffusion = ZeroCoMGaussianDiffusion(T=dc['T'], schedule=dc['schedule'])
        # energy_z=None → unconditional via learned null embedding

    elif mt == 'egnn':
        from models.egnn import EGNNScoreNetwork
        mc    = config['model']
        model = EGNNScoreNetwork(
            n_residues = config['data']['n_residues'],
            node_dim   = mc['hidden_dim'],
            edge_dim   = mc.get('edge_dim', 64),
            time_dim   = mc['time_dim'],
            n_layers   = mc['n_layers'],
        )
        dc        = config['diffusion']
        diffusion = ZeroCoMGaussianDiffusion(T=dc['T'], schedule=dc['schedule'])

    elif mt in ('mlp', 'transformer', 'transformer_adaln'):
        from scripts.train import build_model
        model     = build_model(config)
        dc        = config['diffusion']
        diffusion = GaussianDiffusion(T=dc['T'], schedule=dc['schedule'])

    elif mt == 'transformer_adaln_energy':
        from models.transformer_adaln_energy import AdaLNEnergyTransformerScoreNetwork
        mc    = config['model']
        model = AdaLNEnergyTransformerScoreNetwork(
            n_residues       = config['data']['n_residues'],
            hidden_dim       = mc['hidden_dim'],
            n_heads          = mc['n_heads'],
            n_layers         = mc['n_layers'],
            time_dim         = mc['time_dim'],
            dropout          = mc['dropout'],
            energy_drop_prob = mc.get('energy_drop_prob', 0.15),
        )
        dc        = config['diffusion']
        diffusion = GaussianDiffusion(T=dc['T'], schedule=dc['schedule'])
        # generates unconditionally (energy_z=None); for CFG sweeps use
        # analyze_energy_conditioning.py

    elif mt == 'backbone_transformer':
        from models.backbone_transformer import BackboneTransformerScoreNetwork
        mc    = config['model']
        model = BackboneTransformerScoreNetwork(
            n_residues       = config['data']['n_residues'],
            hidden_dim       = mc['hidden_dim'],
            n_heads          = mc['n_heads'],
            n_layers         = mc['n_layers'],
            time_dim         = mc['time_dim'],
            dropout          = mc['dropout'],
            energy_drop_prob = mc.get('energy_drop_prob', 0.15),
        )
        dc        = config['diffusion']
        diffusion = GaussianDiffusion(T=dc['T'], schedule=dc['schedule'])
        diffusion._is_backbone = True   # flag for generate() routing

    elif mt == 'transformer_adaln_sc':
        from models.transformer_adaln_sc import AdaLNSCScoreNetwork
        mc    = config['model']
        model = AdaLNSCScoreNetwork(
            n_residues       = config['data']['n_residues'],
            hidden_dim       = mc['hidden_dim'],
            n_heads          = mc['n_heads'],
            n_layers         = mc['n_layers'],
            time_dim         = mc['time_dim'],
            dropout          = mc['dropout'],
            energy_drop_prob = mc.get('energy_drop_prob', 0.15),
        )
        dc        = config['diffusion']
        # Plain GaussianDiffusion — same as AdaLN+Energy+Physics.
        # ZeroCoMGaussianDiffusion forced zero-CoM projection of noise_pred at
        # every inference step, introducing systematic errors that compounded
        # through the SC feedback loop and caused bond-length divergence.
        diffusion = GaussianDiffusion(T=dc['T'], schedule=dc['schedule'])
        diffusion._is_self_cond = True   # route generate() → ddim_sample_sc

    elif mt == 'flowmatch':
        from models.egnn          import EGNNScoreNetwork
        from models.flow_matching import ZeroCoMFlowMatching
        mc    = config['model']
        model = EGNNScoreNetwork(
            n_residues = config['data']['n_residues'],
            node_dim   = mc['hidden_dim'],
            edge_dim   = mc.get('edge_dim', 64),
            time_dim   = mc['time_dim'],
            n_layers   = mc['n_layers'],
        )
        fc        = config.get('flow', {})
        diffusion = ZeroCoMFlowMatching(sigma_min=fc.get('sigma_min', 1e-4))

    elif mt == 'egnn_energy':
        # Plain EGNN + energy conditioning (no AdaLN) + ZeroCoM DDPM
        from models.egnn_energy       import EGNNEnergyScoreNetwork
        from models.diffusion_zerocom import ZeroCoMGaussianDiffusion
        mc    = config['model']
        model = EGNNEnergyScoreNetwork(
            n_residues       = config['data']['n_residues'],
            node_dim         = mc['hidden_dim'],
            edge_dim         = mc.get('edge_dim', 64),
            time_dim         = mc['time_dim'],
            n_layers         = mc['n_layers'],
            energy_dim       = mc.get('energy_dim', 32),
            energy_drop_prob = mc.get('energy_drop_prob', 0.15),
        )
        dc        = config['diffusion']
        diffusion = ZeroCoMGaussianDiffusion(T=dc['T'], schedule=dc['schedule'])

    elif mt == 'flowmatch_energy':
        from models.egnn_energy   import EGNNEnergyScoreNetwork
        from models.flow_matching import ZeroCoMFlowMatching
        mc    = config['model']
        model = EGNNEnergyScoreNetwork(
            n_residues       = config['data']['n_residues'],
            node_dim         = mc['hidden_dim'],
            edge_dim         = mc.get('edge_dim', 64),
            time_dim         = mc['time_dim'],
            n_layers         = mc['n_layers'],
            energy_dim       = mc.get('energy_dim', 32),
            energy_drop_prob = mc.get('energy_drop_prob', 0.15),
        )
        fc        = config.get('flow', {})
        diffusion = ZeroCoMFlowMatching(sigma_min=fc.get('sigma_min', 1e-4))
        diffusion._is_energy_cond = True

    elif mt in ('flowmatch_v2_energy', 'se3flow_energy'):
        from models.se3flow_energy import SE3FlowEnergyNet
        from models.flow_matching  import ZeroCoMFlowMatching
        mc    = config['model']
        n_res = config['data']['n_residues']
        # Auto-detect whether this checkpoint included cond_emb in phi_e by
        # inspecting the saved weight shape before building the model.
        # Checkpoints trained before cond_emb was restored to phi_e have
        # phi_e.0.weight of shape [edge_dim, node_dim*2 + n_rbf + sep_dim].
        # Keys may have '_orig_mod.' prefix when saved from torch.compile.
        _sd       = ckpt.get('ema_shadow', ckpt.get('model', {}))
        _node_dim = mc['hidden_dim']
        _n_rbf    = mc.get('n_rbf', 16)
        _sep_dim  = mc.get('sep_dim', 4)
        _no_cond_width = _node_dim * 2 + _n_rbf + _sep_dim
        _cond_in_phi_e = True
        for _sfx in ('layers.0.phi_e.0.weight', '_orig_mod.layers.0.phi_e.0.weight'):
            if _sfx in _sd and _sd[_sfx].shape[1] == _no_cond_width:
                _cond_in_phi_e = False
                break
        model = SE3FlowEnergyNet(
            n_residues       = n_res,
            node_dim         = _node_dim,
            edge_dim         = mc.get('edge_dim', 96),
            time_dim         = mc['time_dim'],
            n_layers         = mc['n_layers'],
            energy_dim       = mc.get('energy_dim', 32),
            energy_drop_prob = mc.get('energy_drop_prob', 0.15),
            n_rbf            = _n_rbf,
            sep_dim          = _sep_dim,
            x1_pred          = mc.get('x1_pred', False),
            self_cond        = mc.get('self_cond', False),
            cond_in_phi_e    = _cond_in_phi_e,
        )
        fc = config.get('flow', {})
        # Wire harmonic prior when trained with it (all-atom only, n_residues=93)
        prior_fn = None
        if fc.get('harmonic_prior', False) and n_res == 93:
            from models.harmonic_prior import sample_all_atom_chain_batched
            from functools import partial
            _cs = config['data'].get('coord_scale', 16.32)
            prior_fn = partial(sample_all_atom_chain_batched, coord_scale=_cs)
        diffusion = ZeroCoMFlowMatching(sigma_min=fc.get('sigma_min', 1e-4),
                                        prior_fn=prior_fn)
        diffusion._x1pred         = mc.get('x1_pred', False)
        diffusion._is_energy_cond = True   # use ddim_sample_cfg in generate()

    elif mt == 'torsion_flow_energy':
        from models.torsion_net  import TorsionFlowNet
        from models.torsion_flow import TorsionalFlowMatching
        mc    = config['model']
        dc    = config.get('data', {})
        model = TorsionFlowNet(
            hidden_dim       = mc['hidden_dim'],
            n_layers         = mc['n_layers'],
            time_dim         = mc['time_dim'],
            energy_dim       = mc.get('energy_dim',       32),
            energy_drop_prob = mc.get('energy_drop_prob', 0.15),
        )
        fc        = config.get('flow', {})
        diffusion = TorsionalFlowMatching(
            sigma_min        = fc.get('sigma_min', 1e-4),
            theta_mean       = dc.get('theta_mean',       1.895),
            theta_source_std = fc.get('theta_source_std', 0.30),
            theta_scale      = dc.get('theta_scale',      0.40),
            phi_scale        = dc.get('phi_scale',        1.81),
        )
        diffusion._is_torsion = True   # routing flag for generate()

    elif mt == 'torsion_transformer_energy':
        from models.torsion_transformer import TorsionTransformerNet
        from models.torsion_flow        import TorsionalFlowMatching
        mc    = config['model']
        dc    = config.get('data', {})
        model = TorsionTransformerNet(
            d_model          = mc.get('d_model',          256),
            n_heads          = mc.get('n_heads',          4),
            n_layers         = mc.get('n_layers',         6),
            time_dim         = mc.get('time_dim',         64),
            energy_dim       = mc.get('energy_dim',       32),
            energy_drop_prob = mc.get('energy_drop_prob', 0.15),
            dropout          = mc.get('dropout',          0.1),
        )
        fc        = config.get('flow', {})
        diffusion = TorsionalFlowMatching(
            sigma_min        = fc.get('sigma_min', 1e-4),
            theta_mean       = dc.get('theta_mean',       1.895),
            theta_source_std = fc.get('theta_source_std', 0.30),
            theta_scale      = dc.get('theta_scale',      0.40),
            phi_scale        = dc.get('phi_scale',        1.81),
            # phi_source_std and phi_weights are restored from the checkpoint
            # state dict as registered buffers — no need to pass them here.
        )
        diffusion._is_torsion = True   # routing flag for generate()

    elif mt == 'backbone_ipa_energy':
        from models.backbone_ipa_flow      import BackboneIPAFlowNet
        from models.backbone_torsion_flow  import BackboneTorsionalFlowMatching
        mc    = config['model']
        dc    = config.get('data', {})
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
        )
        fc        = config.get('flow', {})
        diffusion = BackboneTorsionalFlowMatching(
            sigma_min  = fc.get('sigma_min', 1e-4),
            phi_scale  = dc.get('phi_scale', 1.81),
            psi_scale  = dc.get('psi_scale', 1.81),
            # phi/psi source_std and weights are restored from checkpoint buffers below.
        )
        diffusion._is_backbone_torsion = True   # routing flag for generate() — torsion IPA model

    else:
        raise ValueError(f"Unknown model_type: {mt!r}")

    # ── Shared: load EMA weights, move to device ───────────────────────────
    shadow = _strip_compile_prefix(ckpt['ema_shadow'])
    missing, unexpected = model.load_state_dict(shadow, strict=False)
    if missing or unexpected:
        print(f"  Note: checkpoint/model architecture mismatch "
              f"({len(missing)} missing keys, {len(unexpected)} unexpected) — "
              f"new keys use their initialised values")
    model     = model.to(device).eval()
    diffusion = diffusion.to(device)

    # Restore flow matching buffers (phi_source_std, phi_weights) when present.
    # Old checkpoints (torsion_flow/v1) have no 'flow' key — silently skip.
    # None-valued buffers are excluded from state_dict, so register manually
    # instead of relying on load_state_dict to handle None→tensor transitions.
    if 'flow' in ckpt:
        flow_sd = ckpt['flow']
        for buf_name in ('phi_source_std', 'phi_weights',
                         'psi_source_std', 'psi_weights'):
            if buf_name in flow_sd:
                diffusion.register_buffer(buf_name, flow_sd[buf_name])

    # Torsion models: NeRF output is already in Å; coord_scale=1.0 is a no-op.
    _torsion_types = ('torsion_flow_energy', 'torsion_transformer_energy',
                      'backbone_ipa_energy')
    coord_scale = 1.0 if mt in _torsion_types else config['data'].get('coord_scale', 16.32)
    epoch       = ckpt.get('epoch', '?')
    val_loss    = ckpt.get('best_val_loss', float('nan'))
    physics_on  = config.get('training', {}).get('physics_weight', 0.0) > 0

    print(f"  Loaded {mt} checkpoint — epoch {epoch}, val_loss={val_loss:.4f}"
          + ("  [physics-constrained]" if physics_on else ""))
    return model, diffusion, config, coord_scale


# ─────────────────────────────────────────────────────────────────────────────
# GENERATION
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def generate(model, diffusion, n: int, n_residues: int,
             coord_scale: float, ddim_steps: int, device: str,
             batch_size: int = 256) -> np.ndarray:
    """
    Generate n structures, rescale to Ångströms, return (n, N, 3).

    Routing:
      - torsion_flow_energy → ddim_sample_cfg in (θ, φ) space + NeRF reconstruction
        (output already in Å; bond lengths exactly 3.832 Å by construction)
      - x1_pred models      → ddim_sample_x1pred_cfg (tau=0.5, no CFG amplification)
      - energy-cond models  → ddim_sample_cfg (tau=0.5, guidance_scale=1.0)
      - plain flow/diffusion → ddim_sample
    """
    model.eval()
    all_samples = []
    n_done = 0

    use_torsion          = getattr(diffusion, '_is_torsion',          False)
    use_x1pred           = getattr(diffusion, '_x1pred',             False)
    use_energy           = getattr(diffusion, '_is_energy_cond',     False)
    use_sc               = getattr(diffusion, '_is_self_cond',       False)
    use_backbone         = getattr(diffusion, '_is_backbone',        False)
    use_backbone_torsion = getattr(diffusion, '_is_backbone_torsion', False)

    while n_done < n:
        bs       = min(batch_size, n - n_done)
        n_atoms  = n_residues * 3 if use_backbone else n_residues
        shape    = (bs, n_atoms, 3)

        if use_backbone_torsion:
            # BackboneIPAFlow: sample (φ, ψ) backbone torsion space, reconstruct via NeRF.
            # Returns Cα positions only (10 atoms) in Å; no coord_scale needed.
            from models.backbone_internal_coords import internal_to_backbone
            phi, psi = diffusion.ddim_sample_cfg(
                model, bs, device=device, ddim_steps=ddim_steps,
                tau=0.5, guidance_scale=1.0,
            )
            x    = internal_to_backbone(phi, psi)         # (bs, 30, 3) in Å
            x    = x - x.mean(dim=1, keepdim=True)        # centre CoM
            x_ca = x[:, 1::3]                             # (bs, 10, 3) CA only
            all_samples.append(x_ca.cpu().numpy())
            n_done += bs
            continue

        if use_backbone:
            # BackboneTransformer: plain DDPM on 30-atom Cartesian backbone coords.
            # shape is already (bs, 30, 3) from the n_atoms calculation above.
            x = diffusion.ddim_sample(model, shape, device=device,
                                      ddim_steps=ddim_steps)
            x = x - x.mean(dim=1, keepdim=True)
            x = x * coord_scale            # → Ångströms
            all_samples.append(x.cpu().numpy())
            n_done += bs
            continue

        if use_torsion:
            # Torsional model: sample in (θ, φ) space, reconstruct via NeRF.
            # Output is already in Ångströms (bond_length = 3.832 Å); subtract CoM.
            from models.internal_coords import internal_to_cartesian
            theta, phi = diffusion.ddim_sample_cfg(
                model, bs, device=device, ddim_steps=ddim_steps,
                tau=0.5, guidance_scale=1.0,
            )
            x = internal_to_cartesian(theta, phi)   # (bs, 10, 3) in Å
            x = x - x.mean(dim=1, keepdim=True)     # centre CoM (NeRF canonical ≠ CoM centred)
            all_samples.append(x.cpu().numpy())
            n_done += bs
            continue

        if use_x1pred:
            # x₁-prediction ODE; tau=0.5 = average energy, guidance_scale=1.0 = no CFG
            x = diffusion.ddim_sample_x1pred_cfg(
                model, shape, device=device, ddim_steps=ddim_steps,
                tau=0.5, guidance_scale=1.0,
            )
        elif use_energy:
            # Energy-conditioned model: use ddim_sample_cfg so the harmonic prior
            # (if wired in via diffusion.prior_fn) is applied at x₀.
            # tau=0.5 → average energy; guidance_scale=1.0 → single forward pass.
            x = diffusion.ddim_sample_cfg(
                model, shape, device=device, ddim_steps=ddim_steps,
                tau=0.5, guidance_scale=1.0,
            )
        elif use_sc:
            # Self-conditioning model: carry x₀ forward at each DDIM step.
            # energy_z=None → unconditional (null embedding); guidance_scale=1.0
            # means a single forward pass per step (no CFG doubling).
            x = diffusion.ddim_sample_sc(
                model, shape, device=device, ddim_steps=ddim_steps,
                energy_z=None, guidance_scale=1.0,
            )
        else:
            x = diffusion.ddim_sample(model, shape, device=device,
                                       ddim_steps=ddim_steps)
        # center + rescale to Ångströms
        x = x - x.mean(dim=1, keepdim=True)
        x = x * coord_scale
        all_samples.append(x.cpu().numpy())
        n_done += bs

    return np.concatenate(all_samples, axis=0)[:n].astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# STRUCTURAL METRICS
# ─────────────────────────────────────────────────────────────────────────────

# Reference values from training set (79,632 Chignolin structures)
_IDEAL_BOND   = 3.832   # Å — mean Cα–Cα bond length
_IDEAL_COS    = 0.320   # cos(71.4°) — mean consecutive bond-vector angle
_CLASH_CUTOFF = 3.5     # Å — non-bonded pairs below this = clash


def bond_lengths(coords: np.ndarray) -> np.ndarray:
    """Consecutive Cα–Cα distances. (N, n_res, 3) → (N, n_res-1)."""
    return np.linalg.norm(np.diff(coords, axis=1), axis=-1)


def radius_of_gyration(coords: np.ndarray) -> np.ndarray:
    """Rg per structure. (N, n_res, 3) → (N,)."""
    com = coords.mean(axis=1, keepdims=True)
    return np.sqrt(((coords - com) ** 2).sum(axis=-1).mean(axis=-1))


def validity(coords: np.ndarray, tol: float = 0.5, ideal: float = _IDEAL_BOND) -> float:
    """Fraction of structures where ALL bonds are within ±tol Å of ideal."""
    bl    = bond_lengths(coords)
    valid = (np.abs(bl - ideal) < tol).all(axis=1)
    return float(valid.mean())


def mmd_rbf(x: np.ndarray, y: np.ndarray,
            sigmas=(0.5, 1.0, 2.0)) -> float:
    """
    Maximum Mean Discrepancy with RBF kernel, averaged over several bandwidths.
    x, y : (N, n_res, 3) → flattened to (N, n_res*3).  Lower = more similar.
    """
    x = x.reshape(len(x), -1).astype(np.float64)
    y = y.reshape(len(y), -1).astype(np.float64)

    max_pts = 2000
    if len(x) > max_pts: x = x[np.random.choice(len(x), max_pts, replace=False)]
    if len(y) > max_pts: y = y[np.random.choice(len(y), max_pts, replace=False)]

    def rbf(a, b, sig):
        sq = ((a[:, None] - b[None]) ** 2).sum(-1)
        return np.exp(-sq / (2 * sig ** 2))

    vals = []
    for s in sigmas:
        Kxx = rbf(x, x, s); Kyy = rbf(y, y, s); Kxy = rbf(x, y, s)
        n, m = len(x), len(y)
        v = ((Kxx.sum() - np.diag(Kxx).sum()) / (n * (n-1))
           + (Kyy.sum() - np.diag(Kyy).sum()) / (m * (m-1))
           - 2 * Kxy.mean())
        vals.append(v)
    return float(np.mean(vals))


# ─────────────────────────────────────────────────────────────────────────────
# PHYSICS QUALITY METRICS
# ─────────────────────────────────────────────────────────────────────────────

def compute_physics_metrics(coords: np.ndarray) -> dict:
    """
    Physics-based structural quality metrics for Cα coordinates in Ångströms.

    These complement the distribution-level metrics (MMD, Rg) with per-structure
    constraint satisfaction — the key signal for evaluating physics-constrained models.

    coords : (N, n_res, 3)  in Ångströms
    returns : dict with keys:
        phys_valid_02      fraction of structures with ALL bonds within ±0.2 Å
        phys_valid_03      fraction of structures with ALL bonds within ±0.3 Å
        phys_valid_05      fraction of structures with ALL bonds within ±0.5 Å
        phys_bond_rmse     RMS bond-length deviation from ideal (3.832 Å)
        phys_per_bond_rmse per-bond RMSE (list, length n_res-1) — reveals problem positions
        phys_clash_rate    fraction of structures with any non-bonded clash < 3.5 Å
        phys_angle_rmse    RMS deviation of cos(bond angle) from 0.320 (71.4°)
        phys_diversity     mean pairwise RMSD between generated structures (Å)
    """
    N, n_res, _ = coords.shape
    bl = bond_lengths(coords)                        # (N, n_res-1)

    # ── Bond validity at multiple strictness thresholds ───────────────────
    phys_valid_02 = float((np.abs(bl - _IDEAL_BOND) < 0.2).all(axis=1).mean())
    phys_valid_03 = float((np.abs(bl - _IDEAL_BOND) < 0.3).all(axis=1).mean())
    phys_valid_05 = float((np.abs(bl - _IDEAL_BOND) < 0.5).all(axis=1).mean())

    # ── Bond RMSE (global and per-bond) ──────────────────────────────────
    phys_bond_rmse     = float(np.sqrt(((bl - _IDEAL_BOND) ** 2).mean()))
    phys_per_bond_rmse = np.sqrt(((bl - _IDEAL_BOND) ** 2).mean(axis=0)).tolist()

    # ── Clash rate — fraction of structures with any non-bonded pair < 3.5 Å
    diff = coords[:, :, None, :] - coords[:, None, :, :]   # (N, n_res, n_res, 3)
    dist = np.linalg.norm(diff, axis=-1)                    # (N, n_res, n_res)
    idx  = np.arange(n_res)
    sep  = np.abs(idx[:, None] - idx[None, :])             # (n_res, n_res) separation
    mask = (sep >= 2)                                        # non-bonded pairs only
    phys_clash_rate = float((dist[:, mask] < _CLASH_CUTOFF).any(axis=1).mean())

    # ── Virtual bond angle: cos(angle between consecutive bond vectors) ───
    b1 = coords[:, 1:-1] - coords[:, :-2]                  # (N, n_res-2, 3)
    b2 = coords[:, 2:]   - coords[:, 1:-1]                 # (N, n_res-2, 3)
    norm1 = np.linalg.norm(b1, axis=-1, keepdims=True).clip(1e-8)
    norm2 = np.linalg.norm(b2, axis=-1, keepdims=True).clip(1e-8)
    cos_t = (b1 / norm1 * b2 / norm2).sum(axis=-1)         # (N, n_res-2)
    phys_angle_rmse = float(np.sqrt(((cos_t - _IDEAL_COS) ** 2).mean()))

    # ── Structural diversity: mean pairwise RMSD (subsampled) ────────────
    n_sub  = min(300, N)
    idx_s  = np.random.choice(N, n_sub, replace=False) if N > n_sub else np.arange(N)
    sub    = coords[idx_s]                                  # (n_sub, n_res, 3)
    # Pairwise squared displacement, averaged over residues
    sq     = ((sub[:, None] - sub[None]) ** 2).sum(-1).mean(-1)  # (n_sub, n_sub)
    triu   = np.triu_indices(n_sub, k=1)
    phys_diversity = float(np.sqrt(sq[triu]).mean()) if len(triu[0]) > 0 else 0.0

    return {
        'phys_valid_02':      phys_valid_02,
        'phys_valid_03':      phys_valid_03,
        'phys_valid_05':      phys_valid_05,
        'phys_bond_rmse':     phys_bond_rmse,
        'phys_per_bond_rmse': phys_per_bond_rmse,
        'phys_clash_rate':    phys_clash_rate,
        'phys_angle_rmse':    phys_angle_rmse,
        'phys_diversity':     phys_diversity,
    }


def compute_physics_metrics_aa(coords: np.ndarray) -> dict:
    """
    Physics-based structural quality metrics for all-atom Chignolin (n=93).

    Uses the 64 data-identified covalent bonds from AllAtomPhysics (physics_aa.py)
    and their per-bond ideal lengths (data-derived mean distances from training set).
    Heavy-atom clash cutoff is 2.5 Å (1st percentile of non-bonded distances).

    coords : (N, 93, 3)  in Ångströms
    """
    from models.physics_aa import _BOND_INDICES, _BOND_TARGETS as _BT_LIST
    _BT = np.array(_BT_LIST, dtype=np.float32)   # (64,) per-bond ideal lengths
    _AA_CLASH = 2.5

    N, n_atoms, _ = coords.shape

    # Extract the 64 covalent bond lengths from consecutive atom pairs
    all_consec = np.linalg.norm(np.diff(coords, axis=1), axis=-1)  # (N, 92)
    bl = all_consec[:, _BOND_INDICES]                               # (N, 64)

    phys_valid_02 = float((np.abs(bl - _BT) < 0.2).all(axis=1).mean())
    phys_valid_03 = float((np.abs(bl - _BT) < 0.3).all(axis=1).mean())
    phys_valid_05 = float((np.abs(bl - _BT) < 0.5).all(axis=1).mean())

    phys_bond_rmse     = float(np.sqrt(((bl - _BT) ** 2).mean()))
    phys_per_bond_rmse = np.sqrt(((bl - _BT) ** 2).mean(axis=0)).tolist()

    # Clash rate: non-bonded heavy-atom pairs (sep ≥ 4) closer than 2.5 Å
    diff_mat = coords[:, :, None, :] - coords[:, None, :, :]  # (N, 93, 93, 3)
    dist_mat = np.linalg.norm(diff_mat, axis=-1)               # (N, 93, 93)
    idx = np.arange(n_atoms)
    sep = np.abs(idx[:, None] - idx[None, :])
    nb_mask = (sep >= 4)
    phys_clash_rate = float((dist_mat[:, nb_mask] < _AA_CLASH).any(axis=1).mean())

    # Structural diversity
    n_sub = min(300, N)
    idx_s = np.random.choice(N, n_sub, replace=False) if N > n_sub else np.arange(N)
    sub   = coords[idx_s]
    sq    = ((sub[:, None] - sub[None]) ** 2).sum(-1).mean(-1)
    triu  = np.triu_indices(n_sub, k=1)
    phys_diversity = float(np.sqrt(sq[triu]).mean()) if len(triu[0]) > 0 else 0.0

    return {
        'phys_valid_02':      phys_valid_02,
        'phys_valid_03':      phys_valid_03,
        'phys_valid_05':      phys_valid_05,
        'phys_bond_rmse':     phys_bond_rmse,
        'phys_per_bond_rmse': phys_per_bond_rmse,
        'phys_clash_rate':    phys_clash_rate,
        'phys_angle_rmse':    float('nan'),   # not meaningful for all-atom
        'phys_diversity':     phys_diversity,
    }


def _aa_bond_lengths(coords: np.ndarray) -> np.ndarray:
    """For all-atom Chignolin (n=93): extract the 64 covalent bond lengths. → (N, 64)"""
    from models.physics_aa import _BOND_INDICES
    all_consec = np.linalg.norm(np.diff(coords, axis=1), axis=-1)  # (N, 92)
    return all_consec[:, _BOND_INDICES]                              # (N, 64)


def compute_backbone_metrics(samples: np.ndarray, reference: np.ndarray) -> dict:
    """
    Metrics for 30-atom backbone (N, Cα, C × 10 residues) structures.

    Bond validity is computed against the three real bond targets:
        N–Cα = 1.460 Å,  Cα–C = 1.525 Å,  C–N = 1.329 Å
    using a ±0.5 Å threshold (strict: ±0.1 Å).

    Rg, ETE and MMD are computed on the Cα subset (atoms 1,4,7,…,28).
    """
    from models.backbone_physics import _SRC, _DST, _IDEAL, N_CA_IDEAL, CA_C_IDEAL, C_N_IDEAL

    src   = np.array(_SRC)
    dst   = np.array(_DST)
    ideal = np.array(_IDEAL, dtype=np.float32)

    def _bond_errs(x):
        # x: (N, 30, 3)  → (N, 29) bond-length errors
        bl = np.linalg.norm(x[:, dst] - x[:, src], axis=-1)   # (N, 29)
        return np.abs(bl - ideal)                               # (N, 29)

    errs_s  = _bond_errs(samples)
    errs_r  = _bond_errs(reference)
    valid_s = float((errs_s < 0.5).all(axis=1).mean())
    valid_r = float((errs_r < 0.5).all(axis=1).mean())

    # Cα subset for global-structure metrics (every 3rd atom from index 1)
    ca_s = samples[:, 1::3, :]    # (N, 10, 3)
    ca_r = reference[:, 1::3, :]

    rg_s, rg_r   = radius_of_gyration(ca_s), radius_of_gyration(ca_r)
    ete_s = np.linalg.norm(ca_s[:, -1] - ca_s[:, 0], axis=-1)
    ete_r = np.linalg.norm(ca_r[:, -1] - ca_r[:, 0], axis=-1)

    bl_s  = np.linalg.norm(samples[:, dst] - samples[:, src], axis=-1)   # (N, 29)
    bl_r  = np.linalg.norm(reference[:, dst] - reference[:, src], axis=-1)

    # Per-bond-type RMSE
    n_ca_mask = np.array([i % 3 == 0 for i in range(29)])   # N-CA bonds
    ca_c_mask = np.array([i % 3 == 1 for i in range(29)])   # CA-C bonds
    c_n_mask  = np.array([i % 3 == 2 for i in range(29)])   # C-N bonds (9 bonds)

    def rmse_bonds(bl, mask, target):
        return float(np.sqrt(((bl[:, mask] - target) ** 2).mean()))

    n_sub   = min(300, len(ca_s))
    idx_s2  = np.random.choice(len(ca_s), n_sub, replace=False) if len(ca_s) > n_sub else np.arange(len(ca_s))
    sub     = ca_s[idx_s2]
    sq      = ((sub[:, None] - sub[None]) ** 2).sum(-1).mean(-1)
    triu    = np.triu_indices(n_sub, k=1)
    diversity_s = float(np.sqrt(sq[triu]).mean()) if len(triu[0]) > 0 else 0.0

    # Overall bond RMSE across all 29 bonds
    overall_bond_rmse = float(np.sqrt(((bl_s - ideal) ** 2).mean()))
    # Provide standard phys_* keys so plot_comparison and print_table don't crash
    phys_valid_02 = float((errs_s < 0.2).all(axis=1).mean())
    phys_valid_03 = float((errs_s < 0.3).all(axis=1).mean())

    # Clash rate: non-bonded backbone atom pairs closer than 2.0 Å
    # (|i-j| >= 3 in the 30-atom chain to exclude bonded/1,3-related pairs)
    diff_s  = samples[:, :, None, :] - samples[:, None, :, :]    # (N,30,30,3)
    dist_s  = np.linalg.norm(diff_s, axis=-1)                     # (N,30,30)
    sep     = np.abs(np.arange(30)[:, None] - np.arange(30)[None, :])
    clash_mask = sep >= 3
    clash_rate = float(((dist_s < 2.0) * clash_mask[None]).any(axis=(-1, -2)).mean())

    return {
        'validity_gen':      valid_s,
        'validity_ref':      valid_r,
        'bond_mean_gen':     float(bl_s.mean()),
        'bond_mean_ref':     float(bl_r.mean()),
        'bond_std_gen':      float(bl_s.std()),
        'bond_std_ref':      float(bl_r.std()),
        'n_ca_rmse_gen':     rmse_bonds(bl_s, n_ca_mask, N_CA_IDEAL),
        'ca_c_rmse_gen':     rmse_bonds(bl_s, ca_c_mask, CA_C_IDEAL),
        'c_n_rmse_gen':      rmse_bonds(bl_s, c_n_mask,  C_N_IDEAL),
        'rg_mean_gen':       float(rg_s.mean()),
        'rg_mean_ref':       float(rg_r.mean()),
        'rg_std_gen':        float(rg_s.std()),
        'rg_std_ref':        float(rg_r.std()),
        'ete_mean_gen':      float(ete_s.mean()),
        'ete_mean_ref':      float(ete_r.mean()),
        'mmd':               mmd_rbf(ca_s, ca_r),
        'diversity':         diversity_s,
        'is_backbone':       True,
        # Standard phys_* keys expected by plot_comparison / print_table
        'phys_valid_02':     phys_valid_02,
        'phys_valid_03':     phys_valid_03,
        'phys_valid_05':     valid_s,
        'phys_bond_rmse':    overall_bond_rmse,
        'phys_per_bond_rmse': [rmse_bonds(bl_s, n_ca_mask, N_CA_IDEAL),
                                rmse_bonds(bl_s, ca_c_mask, CA_C_IDEAL),
                                rmse_bonds(bl_s, c_n_mask,  C_N_IDEAL)],
        'phys_clash_rate':   clash_rate,
        'phys_angle_rmse':   float('nan'),
        'phys_diversity':    diversity_s,
        'ref_phys_bond_rmse': float(np.sqrt(((errs_r) ** 2).mean())),
        'ref_phys_valid_02': float((errs_r < 0.2).all(axis=1).mean()),
        'ref_phys_valid_05': valid_r,
        'ref_phys_clash_rate': 0.0,
        'ref_phys_diversity':  0.0,
    }


def compute_all_metrics(samples: np.ndarray, reference: np.ndarray,
                        is_all_atom: bool = False,
                        is_backbone: bool = False) -> dict:
    """
    All structural + physics metrics for a set of generated structures.
    Returns a flat dict suitable for JSON serialisation.

    is_all_atom : set True for 93-atom all-atom Chignolin models; uses per-bond
                  ideal lengths from AllAtomPhysics instead of the Cα ideal 3.832 Å.
    """
    if is_backbone:
        return compute_backbone_metrics(samples, reference)

    # If reference is backbone (30 atoms) but model generates Cα (10 atoms),
    # extract the Cα subset from the reference so metrics are comparable.
    if reference.shape[1] == 30 and samples.shape[1] == 10:
        reference = reference[:, 1::3, :]   # Cα at indices 1,4,7,...,28

    rg_s  = radius_of_gyration(samples)
    rg_r  = radius_of_gyration(reference)
    ete_s = np.linalg.norm(samples[:, -1]   - samples[:, 0],   axis=-1)
    ete_r = np.linalg.norm(reference[:, -1] - reference[:, 0], axis=-1)

    if is_all_atom:
        from models.physics_aa import _BOND_INDICES, _BOND_TARGETS as _BT_LIST
        _BT = np.array(_BT_LIST, dtype=np.float32)
        bl_s = _aa_bond_lengths(samples)
        bl_r = _aa_bond_lengths(reference)
        valid_gen = float((np.abs(bl_s - _BT) < 0.5).all(axis=1).mean())
        valid_ref = float((np.abs(bl_r - _BT) < 0.5).all(axis=1).mean())
    else:
        bl_s      = bond_lengths(samples)
        bl_r      = bond_lengths(reference)
        valid_gen = validity(samples)
        valid_ref = validity(reference)

    metrics = {
        'validity_gen':    valid_gen,
        'validity_ref':    valid_ref,
        'bond_mean_gen':   float(bl_s.mean()),
        'bond_mean_ref':   float(bl_r.mean()),
        'bond_std_gen':    float(bl_s.std()),
        'bond_std_ref':    float(bl_r.std()),
        'rg_mean_gen':     float(rg_s.mean()),
        'rg_mean_ref':     float(rg_r.mean()),
        'rg_std_gen':      float(rg_s.std()),
        'rg_std_ref':      float(rg_r.std()),
        'ete_mean_gen':    float(ete_s.mean()),
        'ete_mean_ref':    float(ete_r.mean()),
        'mmd':             mmd_rbf(samples, reference),
        'is_all_atom':     is_all_atom,
    }

    # Physics quality metrics
    _phys_fn  = compute_physics_metrics_aa if is_all_atom else compute_physics_metrics
    metrics.update(_phys_fn(samples))
    ref_phys  = _phys_fn(reference[:min(2000, len(reference))])
    metrics['ref_phys_bond_rmse']  = ref_phys['phys_bond_rmse']
    metrics['ref_phys_valid_02']   = ref_phys['phys_valid_02']
    metrics['ref_phys_valid_05']   = ref_phys['phys_valid_05']
    metrics['ref_phys_clash_rate'] = ref_phys['phys_clash_rate']
    metrics['ref_phys_diversity']  = ref_phys['phys_diversity']

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# CONSOLE TABLE
# ─────────────────────────────────────────────────────────────────────────────

def print_table(metrics: dict, label: str = "Model"):
    w = 34
    is_aa       = metrics.get('is_all_atom', False)
    is_backbone = metrics.get('is_backbone', False)

    print(f"\n{'━'*68}")
    print(f"  {label}")
    print(f"{'━'*68}")

    if is_backbone:
        print(f"\n  {'Backbone structural metrics (Cα used for Rg/ETE/MMD)':<{w}}")
        print(f"  {'Structural distribution':<{w}} {'Generated':>12} {'Reference':>12}")
        print(f"  {'─'*60}")
        rows1 = [
            ("Bond valid ±0.5 Å, all 29 bonds",  f"{metrics['validity_gen']*100:.1f}%",
                                                  f"{metrics['validity_ref']*100:.1f}%"),
            ("Bond mean (Å)",                     f"{metrics['bond_mean_gen']:.3f}",
                                                  f"{metrics['bond_mean_ref']:.3f}"),
            ("Bond std  (Å)",                     f"{metrics['bond_std_gen']:.3f}",
                                                  f"{metrics['bond_std_ref']:.3f}"),
            ("N–Cα RMSE (Å, ideal 1.460)",        f"{metrics['n_ca_rmse_gen']:.4f}", "—"),
            ("Cα–C  RMSE (Å, ideal 1.525)",       f"{metrics['ca_c_rmse_gen']:.4f}", "—"),
            ("C–N   RMSE (Å, ideal 1.329)",       f"{metrics['c_n_rmse_gen']:.4f}", "—"),
            ("Rg mean (Å, from Cα)",               f"{metrics['rg_mean_gen']:.3f}",
                                                   f"{metrics['rg_mean_ref']:.3f}"),
            ("Rg std  (Å)",                        f"{metrics['rg_std_gen']:.3f}",
                                                   f"{metrics['rg_std_ref']:.3f}"),
            ("End-to-end mean (Å)",                f"{metrics['ete_mean_gen']:.3f}",
                                                   f"{metrics['ete_mean_ref']:.3f}"),
            ("MMD-RBF on Cα (↓ better)",           f"{metrics['mmd']:.5f}", "—"),
            ("Diversity RMSD (Å)",                 f"{metrics['diversity']:.3f}", "—"),
        ]
        for name, gen, ref in rows1:
            print(f"  {name:<{w}} {gen:>12} {ref:>12}")
        print(f"{'━'*68}")
        return

    bond_label      = "AA-bond valid ±0.5 Å (%)" if is_aa else "Bond validity ±0.5 Å (%)"
    bond02_label    = "AA-bond valid ±0.2 Å (%)" if is_aa else "Bond validity ±0.2 Å (%)"
    bond03_label    = "AA-bond valid ±0.3 Å (%)" if is_aa else "Bond validity ±0.3 Å (%)"
    bond_mean_label = "Cov-bond mean (Å)"        if is_aa else "Bond length mean (Å)"

    print(f"\n  {'Structural distribution':<{w}} {'Generated':>12} {'Reference':>12}")
    print(f"  {'─'*60}")
    rows1 = [
        (bond_label,          f"{metrics['validity_gen']*100:.1f}%",   f"{metrics['validity_ref']*100:.1f}%"),
        (bond_mean_label,     f"{metrics['bond_mean_gen']:.3f}",        f"{metrics['bond_mean_ref']:.3f}"),
        ("Bond/cov std (Å)",  f"{metrics['bond_std_gen']:.3f}",         f"{metrics['bond_std_ref']:.3f}"),
        ("Rg mean (Å)",       f"{metrics['rg_mean_gen']:.3f}",          f"{metrics['rg_mean_ref']:.3f}"),
        ("Rg std  (Å)",       f"{metrics['rg_std_gen']:.3f}",           f"{metrics['rg_std_ref']:.3f}"),
        ("End-to-end mean (Å)", f"{metrics['ete_mean_gen']:.3f}",       f"{metrics['ete_mean_ref']:.3f}"),
        ("MMD-RBF (↓ better)", f"{metrics['mmd']:.5f}",                 "—"),
    ]
    for name, gen, ref in rows1:
        print(f"  {name:<{w}} {gen:>12} {ref:>12}")

    print(f"\n  {'Physics constraint quality':<{w}} {'Generated':>12} {'Reference':>12}")
    print(f"  {'─'*60}")
    _angle = metrics.get('phys_angle_rmse', float('nan'))
    _angle_str = "n/a" if (_angle != _angle) else f"{_angle:.4f}"  # nan check
    rows2 = [
        (bond02_label,        f"{metrics['phys_valid_02']*100:.1f}%",   f"{metrics.get('ref_phys_valid_02', float('nan'))*100:.1f}%"),
        (bond03_label,        f"{metrics['phys_valid_03']*100:.1f}%",   "—"),
        ("Bond RMSE (Å)",     f"{metrics['phys_bond_rmse']:.4f}",        f"{metrics.get('ref_phys_bond_rmse', float('nan')):.4f}"),
        ("Clash rate (%)",    f"{metrics['phys_clash_rate']*100:.1f}%",  f"{metrics.get('ref_phys_clash_rate', 0)*100:.1f}%"),
        ("Angle cos RMSE",    _angle_str,                                 "—"),
        ("Diversity RMSD (Å)", f"{metrics['phys_diversity']:.3f}",       f"{metrics.get('ref_phys_diversity', float('nan')):.3f}"),
    ]
    for name, gen, ref in rows2:
        print(f"  {name:<{w}} {gen:>12} {ref:>12}")

    print(f"{'━'*68}")


# ─────────────────────────────────────────────────────────────────────────────
# PLOTTING
# ─────────────────────────────────────────────────────────────────────────────

def plot_comparison(samples_dict: dict, reference: np.ndarray,
                    metrics_all: dict = None, save_path: str = None,
                    is_all_atom: bool = False):
    """
    6-panel comparison plot.

    Panels 1–4: distribution comparisons (bond lengths, Rg, end-to-end, per-atom/residue flexibility)
    Panel 5:    bond validity at multiple strictness thresholds (bar chart)
    Panel 6:    per-bond RMSE from ideal lengths (shows which bonds are worst)

    samples_dict : {'ModelA': (N, n_res, 3), ...}
    reference    : (N_ref, n_res, 3)
    metrics_all  : pre-computed metrics dict (used for physics panels); if None, recomputed
    is_all_atom  : True for 93-atom all-atom Chignolin models
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
    except ImportError:
        print("matplotlib not installed — skipping plot")
        return

    n_res   = reference.shape[1]
    colors  = ['#C44E52', '#4C72B0', '#55A868', '#8172B3', '#CCB974', '#64B5CD']
    labels  = list(samples_dict.keys())

    fig = plt.figure(figsize=(14, 12))
    gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)
    axes = [fig.add_subplot(gs[r, c]) for r in range(3) for c in range(2)]

    title_suffix = " (all-atom, n=93)" if is_all_atom else ""
    fig.suptitle(f"Generated vs Reference Structures{title_suffix}", fontsize=13, y=0.98)

    # ── Helper for histogram panels ───────────────────────────────────────
    def _to_ca(x):
        """If x has 30 atoms (backbone), extract Cα subset for Cα-based metrics."""
        return x[:, 1::3] if x.shape[1] == 30 else x

    def hist(ax, fn, xlabel, title, bins=40, ca_project=False):
        ref_in = _to_ca(reference) if ca_project else reference
        ref_vals = fn(ref_in)
        ax.hist(ref_vals.flatten(), bins=bins, density=True,
                color='#222222', alpha=0.3, label='Reference')
        for (lbl, samples), color in zip(samples_dict.items(), colors):
            s_in = _to_ca(samples) if ca_project else samples
            vals = fn(s_in)
            if vals.std() < 1e-6:
                # Zero-variance data (e.g., torsion model bond lengths exactly fixed)
                ax.axvline(float(vals.flat[0]), color=color, lw=2.5, label=f"{lbl} (exact)")
            else:
                ax.hist(vals.flatten(), bins=bins, density=True,
                        color=color, alpha=0.6, label=lbl)
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

    # Mixed backbone+Cα comparison: project backbone reference to Cα for shared panels
    _mixed = (n_res == 30)   # backbone reference; Cα models have 10-atom output

    # Panel 0: Bond length distribution
    if is_all_atom:
        hist(axes[0], _aa_bond_lengths, "Covalent bond length (Å)",
             "Covalent Bond Length Distribution (64 bonds)")
    elif _mixed:
        # Backbone model: show per-backbone-bond distances; Cα model: Cα-Cα
        hist(axes[0], lambda x: np.linalg.norm(
                 x[:, 1:] - x[:, :-1], axis=-1) if x.shape[1] > 10 else bond_lengths(x),
             "Bond length (Å)", "Bond Length Distribution (backbone or Cα–Cα)")
    else:
        hist(axes[0], bond_lengths, "Cα–Cα distance (Å)", "Bond Length Distribution")
        axes[0].axvline(3.332, color='k', linestyle='--', lw=1, alpha=0.4, label='±0.5 Å bounds')
        axes[0].axvline(4.332, color='k', linestyle='--', lw=1, alpha=0.4)

    # Panel 1: Radius of gyration (always on Cα)
    hist(axes[1], radius_of_gyration, "Rg (Å)", "Radius of Gyration (Cα)",
         ca_project=_mixed)

    # Panel 2: End-to-end distance (always on Cα)
    hist(axes[2], lambda x: np.linalg.norm(x[:, -1] - x[:, 0], axis=-1),
         "End-to-end (Å)", "End-to-End Distance (Cα)", ca_project=_mixed)

    # Panel 3: Per-atom/residue positional variance
    # For mixed backbone(30) + Cα(10) comparisons, project everything to Cα.
    ax = axes[3]
    ref_plot = reference[:, 1::3] if n_res == 30 else reference   # (N, 10, 3)
    ref_var = ref_plot.var(axis=0).sum(axis=-1)
    n_plot = ref_plot.shape[1]
    ax.plot(range(n_plot), ref_var, color='#222222', alpha=0.5, label='Reference', lw=2)
    for (lbl, samples), color in zip(samples_dict.items(), colors):
        s_plot = samples[:, 1::3] if samples.shape[1] == 30 else samples
        var = s_plot.var(axis=0).sum(axis=-1)
        ax.plot(range(n_plot), var, color=color, alpha=0.8, label=lbl)
    if not is_all_atom:
        ax.set_xticks(range(n_plot))
        ax.set_xticklabels([f"R{i+1}" for i in range(n_plot)], rotation=45, fontsize=7)
    else:
        tick_pos = list(range(0, n_plot, 10))
        ax.set_xticks(tick_pos)
        ax.set_xticklabels([str(i) for i in tick_pos], rotation=45, fontsize=7)
    ax.set_ylabel("Positional variance (Å²)", fontsize=9)
    ax.set_title("Per-Atom Positional Variance" if is_all_atom else "Per-Residue Flexibility",
                 fontsize=10)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    # Panel 4: Bond validity at multiple strictness thresholds (bar chart)
    ax = axes[4]
    thresholds = ['±0.2 Å', '±0.3 Å', '±0.5 Å']
    keys       = ['phys_valid_02', 'phys_valid_03', 'phys_valid_05']
    x_pos      = np.arange(len(thresholds))
    bar_width  = 0.8 / (len(labels) + 1)

    # Reference values — use pre-computed if available
    if metrics_all:
        first_key = next(iter(metrics_all))
        ref_v02 = metrics_all[first_key].get('ref_phys_valid_02', None)
        ref_v05 = metrics_all[first_key].get('validity_ref', None)
    else:
        _pf = compute_physics_metrics_aa if is_all_atom else compute_physics_metrics
        ref_v02 = _pf(reference[:500])['phys_valid_02']
        ref_v05 = validity(reference) if not is_all_atom else None

    for k_idx, (thr, key) in enumerate(zip(thresholds, keys)):
        for m_idx, (lbl, color) in enumerate(zip(labels, colors)):
            val = (metrics_all[lbl][key] if metrics_all
                   else (compute_physics_metrics_aa if is_all_atom else
                         compute_physics_metrics)(samples_dict[lbl])['phys_valid_02'])
            offset = (m_idx - len(labels) / 2.0 + 0.5) * bar_width
            ax.bar(k_idx + offset, val * 100, bar_width * 0.9,
                   color=color, alpha=0.8, label=lbl if k_idx == 0 else "")

    # Reference dashes
    for k_idx, rv in [(0, ref_v02), (2, ref_v05)]:
        if rv is not None:
            ax.hlines(rv * 100, k_idx - 0.4, k_idx + 0.4,
                      colors='#222222', linewidths=2, linestyles='--',
                      label='Reference' if k_idx == 0 else "")

    ax.set_xticks(x_pos)
    ax.set_xticklabels(thresholds, fontsize=9)
    ax.set_ylabel("Bond validity (%)", fontsize=9)
    ax.set_title("Bond Validity at Multiple Thresholds", fontsize=10)
    ax.set_ylim(0, 105)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, axis='y')

    # Panel 5: Per-bond RMSE
    ax = axes[5]
    if is_all_atom:
        from models.physics_aa import _BOND_INDICES, _BOND_TARGETS as _BT_LIST
        _BT = np.array(_BT_LIST, dtype=np.float32)
        bl_ref = _aa_bond_lengths(reference)
        ref_per_bond = np.sqrt(((bl_ref - _BT) ** 2).mean(axis=0))
        bond_positions = list(range(len(_BOND_INDICES)))
        ax.plot(bond_positions, ref_per_bond, 'o--', color='#222222',
                alpha=0.5, label='Reference', lw=1.5, markersize=3)
        for (lbl, samples), color in zip(samples_dict.items(), colors):
            bl = _aa_bond_lengths(samples)
            per_bond = np.sqrt(((bl - _BT) ** 2).mean(axis=0))
            ax.plot(bond_positions, per_bond, 'o-', color=color,
                    alpha=0.8, label=lbl, lw=1.5, markersize=3)
        ax.set_xlabel("Covalent bond index (of 64)", fontsize=9)
        ax.set_title("Per-Bond RMSE (all-atom, per data-derived ideal)", fontsize=10)
    else:
        # For mixed backbone+Cα: project reference and all samples to Cα for a fair comparison
        ref_bl_in = _to_ca(reference) if _mixed else reference
        bl_ref = bond_lengths(ref_bl_in)
        ref_per_bond = np.sqrt(((bl_ref - _IDEAL_BOND) ** 2).mean(axis=0))
        n_bonds_plot = bl_ref.shape[1]
        ax.plot(range(n_bonds_plot), ref_per_bond, 'o--', color='#222222',
                alpha=0.5, label='Reference (Cα)', lw=1.5, markersize=4)
        for (lbl, samples), color in zip(samples_dict.items(), colors):
            s_bl_in = _to_ca(samples) if samples.shape[1] == 30 else samples
            bl = bond_lengths(s_bl_in)
            per_bond = np.sqrt(((bl - _IDEAL_BOND) ** 2).mean(axis=0))
            ax.plot(range(n_bonds_plot), per_bond, 'o-', color=color,
                    alpha=0.8, label=lbl, lw=1.5, markersize=4)
        ax.set_xticks(range(n_bonds_plot))
        ax.set_xticklabels([f"{i+1}–{i+2}" for i in range(n_bonds_plot)], rotation=45, fontsize=7)
        ax.set_title("Per-Cα-Bond RMSE from Ideal (3.832 Å)", fontsize=10)
    ax.set_ylabel("Bond RMSE (Å)", fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved → {save_path}")
    else:
        plt.show()
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# PDB EXPORT
# ─────────────────────────────────────────────────────────────────────────────

def save_pdbs(samples: np.ndarray, out_dir: str, n_save: int = 20,
              label: str = "generated"):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    n_atoms = samples.shape[1]

    if n_atoms == 10:
        # Cα-only Chignolin
        sequence = "YYDPETGTWG"
        aa3      = {'Y':'TYR','D':'ASP','P':'PRO','E':'GLU','T':'THR',
                    'G':'GLY','W':'TRP','A':'ALA','K':'LYS','R':'ARG'}
        for i, coords in enumerate(samples[:n_save]):
            lines = [f"REMARK  {label} sample {i+1}\n"]
            for j, (res, xyz) in enumerate(zip(sequence, coords)):
                x, y, z = xyz
                rn = aa3.get(res, 'GLY')
                lines.append(
                    f"ATOM  {j+1:5d}  CA  {rn} A{j+1:4d}    "
                    f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C\n"
                )
            lines.append("END\n")
            with open(out / f"sample_{i+1:04d}.pdb", 'w') as f:
                f.writelines(lines)
    elif n_atoms == 30:
        # Backbone N-Cα-C model: write proper ATOM records with known atom names
        sequence = "YYDPETGTWG"
        aa3      = {'Y':'TYR','D':'ASP','P':'PRO','E':'GLU','T':'THR',
                    'G':'GLY','W':'TRP','A':'ALA','K':'LYS','R':'ARG'}
        atom_names = [' N  ', ' CA ', ' C  '] * 10
        elements   = ['N', 'C', 'C'] * 10
        for i, coords in enumerate(samples[:n_save]):
            lines = [f"REMARK  {label} sample {i+1} (backbone N-CA-C, 30 atoms)\n"]
            for j, (aname, elem, xyz) in enumerate(zip(atom_names, elements, coords)):
                res_idx = j // 3
                res     = sequence[res_idx]
                rn      = aa3.get(res, 'GLY')
                x, y, z = xyz
                lines.append(
                    f"ATOM  {j+1:5d} {aname} {rn} A{res_idx+1:4d}    "
                    f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           {elem}\n"
                )
            lines.append("END\n")
            with open(out / f"sample_{i+1:04d}.pdb", 'w') as f:
                f.writelines(lines)
    else:
        # Generic all-atom: HETATM records (atom types unknown without full topology)
        for i, coords in enumerate(samples[:n_save]):
            lines = [f"REMARK  {label} sample {i+1} (all-atom, {n_atoms} heavy atoms)\n"]
            for j, xyz in enumerate(coords):
                x, y, z = xyz
                lines.append(
                    f"HETATM{j+1:5d}  C   UNK A{j+1:4d}    "
                    f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C\n"
                )
            lines.append("END\n")
            with open(out / f"sample_{i+1:04d}.pdb", 'w') as f:
                f.writelines(lines)

    print(f"Saved {min(n_save, len(samples))} PDB files → {out_dir}")
    print(f"Visualise: pymol {out_dir}/*.pdb")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def _make_label(config: dict, used: set) -> str:
    """Derive a unique display label from a checkpoint config."""
    mt   = config['model_type'].upper()
    phys = config.get('training', {}).get('physics_weight', 0.0) > 0
    base = f"{mt}+Physics" if phys else mt
    label, suffix = base, 2
    while label in used:
        label = f"{base}_{suffix}"; suffix += 1
    return label


def main():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Evaluate one or more checkpoints against the test set."
    )
    # ── Checkpoints ───────────────────────────────────────────────────────────
    p.add_argument('--ckpt',     required=True,
                   help='Primary checkpoint path')
    p.add_argument('--ckpt_ref', nargs='*', default=[],
                   help='Additional checkpoints for side-by-side comparison '
                        '(accepts multiple paths)')
    p.add_argument('--labels',   nargs='*', default=None,
                   help='Custom labels for each checkpoint (must match total '
                        'count of --ckpt + --ckpt_ref). '
                        'Defaults to model_type, with "+Physics" suffix if trained with physics.')
    # ── Evaluation options ────────────────────────────────────────────────────
    p.add_argument('--test',     default=None,
                   help='Path to test.npz. If omitted, auto-read from checkpoint config '
                        '(data.test_path) relative to the repo root.')
    p.add_argument('--n',        type=int, default=500,
                   help='Number of structures to generate per model')
    p.add_argument('--steps',    type=int, default=100,
                   help='Sampling steps. For DDPM: DDIM steps. '
                        'For flow matching: ODE steps (Heun uses 2 NFE per step).')
    p.add_argument('--save_pdb', default=None,
                   help='Directory to save PDB files for the primary model')
    p.add_argument('--save',     default=None,
                   help='Path to save the 6-panel comparison plot')
    p.add_argument('--out_json', default=None,
                   help='Path to save metrics JSON (default: beside primary checkpoint)')
    p.add_argument('--batch',    type=int, default=256,
                   help='Generation batch size')
    p.add_argument('--seed',     type=int, default=0)
    args = p.parse_args()

    # ── Validate ──────────────────────────────────────────────────────────────
    all_ckpts = [args.ckpt] + list(args.ckpt_ref)
    if args.labels is not None and len(args.labels) != len(all_ckpts):
        p.error(f"--labels has {len(args.labels)} entries but "
                f"{len(all_ckpts)} checkpoints were given.")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # ── Determine test path (auto-detect from first checkpoint if not given) ──
    test_path = args.test
    if test_path is None:
        _ckpt_data = torch.load(args.ckpt, map_location='cpu')
        _cfg_test  = _ckpt_data['config']['data'].get('test_path')
        if _cfg_test is None:
            p.error("--test not provided and checkpoint config has no data.test_path")
        # Resolve relative to repo root (one level above scripts/)
        repo_root = Path(__file__).parent.parent
        test_path = str(repo_root / _cfg_test)
        print(f"Auto test path from config: {test_path}")

    # ── Load test set ─────────────────────────────────────────────────────────
    test_data = np.load(test_path)
    reference = test_data['coords'].astype(np.float32)
    centroids = test_data['centroids']
    if centroids.ndim == 2:
        centroids = centroids[:, None, :]
    reference = reference - centroids
    print(f"Test set: {len(reference):,} structures  "
          f"shape per structure: {reference.shape[1:]} atoms")

    # ── Generate + evaluate each checkpoint ───────────────────────────────────
    samples_dict    = {}
    metrics_all     = {}
    primary_samples = None
    used_labels     = set()
    is_all_atom     = None   # determined from first checkpoint

    for i, ckpt_path in enumerate(all_ckpts):
        print(f"\n{'─'*60}")
        print(f"Checkpoint {i+1}/{len(all_ckpts)}: {ckpt_path}")
        model, diffusion, config, scale = load_model_from_ckpt(ckpt_path, device)

        n_res        = config['data']['n_residues']
        _is_aa       = (n_res == 93)
        _is_backbone = getattr(diffusion, '_is_backbone', False)
        if is_all_atom is None:
            is_all_atom = _is_aa
        elif is_all_atom != _is_aa and not _is_backbone:
            print("  WARNING: mixing all-atom and Cα-only checkpoints — "
                  "metrics may be inaccurate.")

        label = args.labels[i] if args.labels else _make_label(config, used_labels)
        used_labels.add(label)

        print(f"Generating {args.n} samples ({label}) …")
        samples = generate(model, diffusion, args.n, n_res, scale,
                           args.steps, device, args.batch)

        if primary_samples is None:
            primary_samples = samples

        samples_dict[label] = samples
        print(f"Computing metrics …")
        metrics_all[label]  = compute_all_metrics(samples, reference,
                                                   is_all_atom=_is_aa,
                                                   is_backbone=_is_backbone)
        print_table(metrics_all[label], label=label)

    is_all_atom = is_all_atom or False

    # ── Metrics JSON ──────────────────────────────────────────────────────────
    out_json = args.out_json or str(Path(args.ckpt).parent / 'eval_metrics.json')
    with open(out_json, 'w') as f:
        json.dump(metrics_all, f, indent=2)
    print(f"\nMetrics → {out_json}")

    # ── 6-panel plot ──────────────────────────────────────────────────────────
    plot_comparison(samples_dict, reference,
                    metrics_all=metrics_all, save_path=args.save,
                    is_all_atom=is_all_atom)

    # ── PDB files from primary checkpoint only ────────────────────────────────
    if args.save_pdb:
        primary_label = list(samples_dict.keys())[0]
        save_pdbs(primary_samples, args.save_pdb, n_save=20, label=primary_label)


if __name__ == '__main__':
    main()
