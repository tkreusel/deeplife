"""
scripts/eval_v2/model_utils.py
================================
Model loading and structure generation utilities.

Copied from scripts/evaluate.py and scripts/analyze_energy_conditioning.py
with minor extensions (energy-conditioned generation, atom-type helpers).
"""

import sys
import numpy as np
from pathlib import Path

import torch

# Ensure repo root is on the path regardless of how the package is invoked
_REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _strip_compile_prefix(state_dict: dict) -> dict:
    """Remove '_orig_mod.' prefix added by torch.compile from EMA shadow keys."""
    return {
        (k[len('_orig_mod.'):] if k.startswith('_orig_mod.') else k): v
        for k, v in state_dict.items()
    }


def atom_type_str(n_atoms: int) -> str:
    """Return 'ca', 'backbone', or 'all_atom' from atom count."""
    if n_atoms == 10:
        return 'ca'
    elif n_atoms == 30:
        return 'backbone'
    elif n_atoms == 93:
        return 'all_atom'
    else:
        return f'unknown_{n_atoms}'


def ca_from_coords(coords: np.ndarray) -> np.ndarray:
    """
    Extract Cα atoms from any atom-level coordinate array.

    coords : (N, n_atoms, 3)
    returns : (N, 10, 3)  — Cα atoms only
      - n=10  → identity (already Cα-only)
      - n=30  → backbone; Cα at indices 1,4,7,...,28 (every 3rd from 1)
      - n=93  → all-atom; Cα at indices 1,4,7,...,28 in the backbone-ordered subset
                Chignolin heavy-atom ordering has N,CA,C,... pattern: CA at 1,4,7,...,28
    """
    n = coords.shape[1]
    if n == 10:
        return coords
    elif n == 30:
        return coords[:, 1::3, :]     # (N, 10, 3)
    elif n == 93:
        # Chignolin all-atom layout: each residue block starts with N, CA, C.
        # The first 30 atoms are the backbone (N0,CA0,C0,N1,CA1,C1,...) so CA at 1,4,...,28.
        return coords[:, 1::3, :][:, :10, :]   # (N, 10, 3)
    else:
        raise ValueError(f"Unsupported atom count: {n}")


# ─────────────────────────────────────────────────────────────────────────────
# MODEL LOADING  (auto-detects model_type from checkpoint)
# ─────────────────────────────────────────────────────────────────────────────

def load_model_from_ckpt(ckpt_path: str, device: str):
    """
    Load model + diffusion/flow from any checkpoint in this repo.
    Reads model_type from the saved config to pick the right class.

    Returns (model, diffusion, config, coord_scale)
    """
    from models.diffusion_zerocom import ZeroCoMGaussianDiffusion
    from models.diffusion          import GaussianDiffusion

    ckpt   = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = ckpt['config']
    mt     = config['model_type']

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
        diffusion._is_backbone = True

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
        diffusion = GaussianDiffusion(T=dc['T'], schedule=dc['schedule'])
        diffusion._is_self_cond = True

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
        prior_fn = None
        if fc.get('harmonic_prior', False) and n_res == 93:
            from models.harmonic_prior import sample_all_atom_chain_batched
            from functools import partial
            _cs = config['data'].get('coord_scale', 16.32)
            prior_fn = partial(sample_all_atom_chain_batched, coord_scale=_cs)
        diffusion = ZeroCoMFlowMatching(sigma_min=fc.get('sigma_min', 1e-4),
                                        prior_fn=prior_fn)
        diffusion._x1pred         = mc.get('x1_pred', False)
        diffusion._is_energy_cond = True

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
        diffusion._is_torsion = True

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
        )
        diffusion._is_torsion = True

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
        )
        diffusion._is_backbone = True

    else:
        raise ValueError(f"Unknown model_type: {mt!r}")

    # Load EMA weights
    shadow = _strip_compile_prefix(ckpt['ema_shadow'])
    missing, unexpected = model.load_state_dict(shadow, strict=False)
    if missing or unexpected:
        print(f"  Note: {len(missing)} missing keys, {len(unexpected)} unexpected "
              f"— new keys use initialised values")
    model     = model.to(device).eval()
    diffusion = diffusion.to(device)

    # Restore flow matching buffers from checkpoint
    if 'flow' in ckpt:
        flow_sd = ckpt['flow']
        for buf_name in ('phi_source_std', 'phi_weights',
                         'psi_source_std', 'psi_weights'):
            if buf_name in flow_sd:
                diffusion.register_buffer(buf_name, flow_sd[buf_name])

    _torsion_types = ('torsion_flow_energy', 'torsion_transformer_energy',
                      'backbone_ipa_energy')
    coord_scale = 1.0 if mt in _torsion_types else config['data'].get('coord_scale', 16.32)
    epoch       = ckpt.get('epoch', '?')
    val_loss    = ckpt.get('best_val_loss', float('nan'))
    physics_on  = config.get('training', {}).get('physics_weight', 0.0) > 0

    print(f"  Loaded {mt} — epoch {epoch}, val_loss={val_loss:.4f}"
          + ("  [physics]" if physics_on else ""))
    return model, diffusion, config, coord_scale


# ─────────────────────────────────────────────────────────────────────────────
# GENERATION
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def generate(model, diffusion, n: int, n_residues: int,
             coord_scale: float, ddim_steps: int, device: str,
             batch_size: int = 256) -> np.ndarray:
    """
    Generate n structures at default settings (tau=0.5 for energy models).
    Returns (n, n_atoms, 3) in Ångströms.
    """
    model.eval()
    all_samples = []
    n_done = 0

    use_torsion  = getattr(diffusion, '_is_torsion',      False)
    use_x1pred   = getattr(diffusion, '_x1pred',          False)
    use_energy   = getattr(diffusion, '_is_energy_cond',  False)
    use_sc       = getattr(diffusion, '_is_self_cond',    False)
    use_backbone = getattr(diffusion, '_is_backbone',     False)

    while n_done < n:
        bs      = min(batch_size, n - n_done)
        n_atoms = n_residues * 3 if use_backbone else n_residues
        shape   = (bs, n_atoms, 3)

        if use_backbone:
            # backbone_transformer uses plain GaussianDiffusion (ddim_sample exists).
            # backbone_ipa_energy uses BackboneTorsionalFlowMatching (ddim_sample_cfg,
            # returns (phi, psi) torsions → reconstruct to Cartesian via NeRF).
            if hasattr(diffusion, 'ddim_sample'):
                x = diffusion.ddim_sample(model, shape, device=device, ddim_steps=ddim_steps)
            else:
                from models.backbone_internal_coords import internal_to_backbone
                phi, psi = diffusion.ddim_sample_cfg(
                    model, bs, device=device, ddim_steps=ddim_steps,
                    tau=0.5, guidance_scale=1.0,
                )
                x = internal_to_backbone(phi, psi)   # (bs, 30, 3) in Å
            x = x - x.mean(dim=1, keepdim=True)
            x = x * coord_scale
            all_samples.append(x.cpu().numpy())
            n_done += bs
            continue

        if use_torsion:
            from models.internal_coords import internal_to_cartesian
            theta, phi = diffusion.ddim_sample_cfg(
                model, bs, device=device, ddim_steps=ddim_steps,
                tau=0.5, guidance_scale=1.0,
            )
            x = internal_to_cartesian(theta, phi)
            x = x - x.mean(dim=1, keepdim=True)
            all_samples.append(x.cpu().numpy())
            n_done += bs
            continue

        if use_x1pred:
            x = diffusion.ddim_sample_x1pred_cfg(
                model, shape, device=device, ddim_steps=ddim_steps,
                tau=0.5, guidance_scale=1.0,
            )
        elif use_energy:
            x = diffusion.ddim_sample_cfg(
                model, shape, device=device, ddim_steps=ddim_steps,
                tau=0.5, guidance_scale=1.0,
            )
        elif use_sc:
            x = diffusion.ddim_sample_sc(
                model, shape, device=device, ddim_steps=ddim_steps,
                energy_z=None, guidance_scale=1.0,
            )
        else:
            x = diffusion.ddim_sample(model, shape, device=device, ddim_steps=ddim_steps)

        x = x - x.mean(dim=1, keepdim=True)
        x = x * coord_scale
        all_samples.append(x.cpu().numpy())
        n_done += bs

    return np.concatenate(all_samples, axis=0)[:n].astype(np.float32)


@torch.no_grad()
def generate_at_temperature(
    model, diffusion, tau: float, n: int, n_residues: int,
    coord_scale: float, ddim_steps: int, guidance_scale: float,
    device: str, batch_size: int = 256,
) -> np.ndarray:
    """
    Generate n structures at temperature τ. Only works for energy-conditioned models.
    For non-energy models, falls back to generate() with tau ignored.
    """
    use_torsion  = getattr(diffusion, '_is_torsion',     False)
    use_x1pred   = getattr(diffusion, '_x1pred',         False)
    use_energy   = getattr(diffusion, '_is_energy_cond', False)
    use_sc       = getattr(diffusion, '_is_self_cond',   False)
    use_backbone = getattr(diffusion, '_is_backbone',    False)
    use_zcom_ddpm = hasattr(diffusion, 'alphas_cumprod')

    if not use_energy and not use_sc and not use_torsion:
        # Non-energy model — tau has no effect, use default generation
        return generate(model, diffusion, n, n_residues, coord_scale,
                        ddim_steps, device, batch_size)

    model.eval()
    all_samples = []
    n_done = 0

    while n_done < n:
        bs = min(batch_size, n - n_done)

        if use_torsion:
            from models.internal_coords import internal_to_cartesian
            theta, phi = diffusion.ddim_sample_cfg(
                model, bs, device=device,
                ddim_steps=ddim_steps, tau=tau, guidance_scale=guidance_scale,
            )
            x = internal_to_cartesian(theta, phi)
            x = x - x.mean(dim=1, keepdim=True)
            all_samples.append(x.cpu().numpy())

        elif use_sc:
            from models.transformer_adaln_sc import AdaLNSCScoreNetwork
            e_z_val = AdaLNSCScoreNetwork.temperature_to_energy_z(tau)
            e_z = torch.full((bs,), e_z_val, device=device)
            x = diffusion.ddim_sample_sc(
                model, shape=(bs, n_residues, 3), device=device,
                ddim_steps=ddim_steps, energy_z=e_z,
                guidance_scale=guidance_scale,
            )
            x = (x - x.mean(dim=1, keepdim=True)) * coord_scale
            all_samples.append(x.cpu().numpy())

        elif use_zcom_ddpm and use_energy and not use_x1pred:
            # DDPM + CFG path (backbone_transformer, transformer_adaln_energy)
            e_z_val = 4.0 * tau - 2.0
            n_atoms = n_residues * 3 if use_backbone else n_residues
            x   = torch.randn(bs, n_atoms, 3, device=device)
            x   = x - x.mean(dim=1, keepdim=True)
            e_z = torch.full((bs,), e_z_val, device=device)

            T          = diffusion.T
            step_sz    = T // ddim_steps
            timesteps  = list(range(0, T, step_sz))[::-1]

            for i, t in enumerate(timesteps):
                t_batch    = torch.full((bs,), t, device=device, dtype=torch.long)
                t_prev     = timesteps[i + 1] if i + 1 < len(timesteps) else 0
                alpha_t    = diffusion.alphas_cumprod[t]
                alpha_prev = diffusion.alphas_cumprod[t_prev]

                eps_cond = model(x, t_batch, energy_z=e_z)
                eps_cond = eps_cond - eps_cond.mean(dim=1, keepdim=True)
                if guidance_scale != 1.0:
                    eps_uncond = model(x, t_batch, energy_z=None)
                    eps_uncond = eps_uncond - eps_uncond.mean(dim=1, keepdim=True)
                    eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
                else:
                    eps = eps_cond

                x0_pred = (x - (1 - alpha_t).sqrt() * eps) / alpha_t.sqrt()
                x0_pred = (x0_pred - x0_pred.mean(dim=1, keepdim=True)).clamp(-5, 5)
                x = alpha_prev.sqrt() * x0_pred + (1 - alpha_prev).sqrt() * eps

            x = (x - x.mean(dim=1, keepdim=True)) * coord_scale
            all_samples.append(x.cpu().numpy())

        elif use_backbone and not hasattr(diffusion, 'ddim_sample'):
            # BackboneTorsionalFlowMatching: torsion ODE → NeRF reconstruction
            from models.backbone_internal_coords import internal_to_backbone
            phi, psi = diffusion.ddim_sample_cfg(
                model, bs, device=device, ddim_steps=ddim_steps,
                tau=tau, guidance_scale=guidance_scale,
            )
            x = internal_to_backbone(phi, psi)  # (bs, 30, 3) in Å
            x = (x - x.mean(dim=1, keepdim=True)) * coord_scale
            all_samples.append(x.cpu().numpy())

        else:
            # Flow matching + CFG (velocity or x1-pred)
            n_atoms = n_residues * 3 if use_backbone else n_residues
            shape = (bs, n_atoms, 3)
            sampler = (diffusion.ddim_sample_x1pred_cfg if use_x1pred
                       else diffusion.ddim_sample_cfg)
            x = sampler(model, shape, device=device,
                        ddim_steps=ddim_steps, tau=tau, guidance_scale=guidance_scale)
            x = (x - x.mean(dim=1, keepdim=True)) * coord_scale
            all_samples.append(x.cpu().numpy())

        n_done += bs

    return np.concatenate(all_samples, axis=0)[:n].astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# ENERGY / CAPABILITY DETECTION
# ─────────────────────────────────────────────────────────────────────────────

def is_energy_conditioned(config: dict) -> bool:
    """True if the model type supports temperature-conditioned sampling."""
    energy_types = {
        'egnn_adaln', 'egnn_energy', 'flowmatch_energy',
        'flowmatch_v2_energy', 'se3flow_energy',
        'torsion_flow_energy', 'torsion_transformer_energy',
        'transformer_adaln_energy', 'transformer_adaln_sc',
        'backbone_transformer', 'backbone_ipa_energy',
    }
    return config.get('model_type', '') in energy_types


def is_equivariant_model(config: dict) -> bool:
    """True if the model architecture is SE(3)-equivariant by design."""
    equivariant_types = {
        'egnn', 'egnn_adaln', 'egnn_energy',
        'flowmatch', 'flowmatch_energy',
        'flowmatch_v2_energy', 'se3flow_energy',
    }
    return config.get('model_type', '') in equivariant_types


# ─────────────────────────────────────────────────────────────────────────────
# REFERENCE DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_reference(test_path: str) -> tuple:
    """
    Load test set from .npz file.
    Returns (coords, energies) — coords are centroid-subtracted, in Ångströms.
    energies may be None if not present in the file.
    """
    d         = np.load(test_path)
    coords    = d['coords'].astype(np.float32)
    centroids = d['centroids']
    if centroids.ndim == 2:
        centroids = centroids[:, None, :]
    coords = coords - centroids

    energies = None
    if 'energies' in d:
        energies = d['energies'].astype(np.float32)

    return coords, energies


# ─────────────────────────────────────────────────────────────────────────────
# PDB EXPORT
# ─────────────────────────────────────────────────────────────────────────────

def save_pdbs(samples: np.ndarray, out_dir: str, n_save: int = 20, label: str = "generated"):
    """Write Cα, backbone, or all-atom structures as PDB files."""
    from .constants import CHIGNOLIN_SEQUENCE, AA3, ATOM_NAMES_BACKBONE, ELEMENTS_BACKBONE

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    n_atoms = samples.shape[1]

    if n_atoms == 10:
        for i, coords in enumerate(samples[:n_save]):
            lines = [f"REMARK  {label} sample {i+1}\n"]
            for j, (res, xyz) in enumerate(zip(CHIGNOLIN_SEQUENCE, coords)):
                x, y, z = xyz
                rn = AA3.get(res, 'GLY')
                lines.append(
                    f"ATOM  {j+1:5d}  CA  {rn} A{j+1:4d}    "
                    f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C\n"
                )
            lines.append("END\n")
            (out / f"sample_{i+1:04d}.pdb").write_text(''.join(lines))

    elif n_atoms == 30:
        for i, coords in enumerate(samples[:n_save]):
            lines = [f"REMARK  {label} sample {i+1} (backbone N-CA-C)\n"]
            for j, (aname, elem, xyz) in enumerate(
                zip(ATOM_NAMES_BACKBONE, ELEMENTS_BACKBONE, coords)
            ):
                res_idx = j // 3
                rn      = AA3.get(CHIGNOLIN_SEQUENCE[res_idx], 'GLY')
                x, y, z = xyz
                lines.append(
                    f"ATOM  {j+1:5d} {aname} {rn} A{res_idx+1:4d}    "
                    f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           {elem}\n"
                )
            lines.append("END\n")
            (out / f"sample_{i+1:04d}.pdb").write_text(''.join(lines))

    else:
        for i, coords in enumerate(samples[:n_save]):
            lines = [f"REMARK  {label} sample {i+1} (all-atom, {n_atoms} heavy atoms)\n"]
            for j, xyz in enumerate(coords):
                x, y, z = xyz
                lines.append(
                    f"HETATM{j+1:5d}  C   UNK A{j+1:4d}    "
                    f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C\n"
                )
            lines.append("END\n")
            (out / f"sample_{i+1:04d}.pdb").write_text(''.join(lines))

    print(f"Saved {min(n_save, len(samples))} PDB files → {out_dir}")
