"""
models/torsion_flow.py
======================
Torsional OT-CFM (Riemannian flow matching on a flat torus) for Chignolin Cα.

Coordinate space
----------------
Internal coordinates: bond angles θ ∈ (0, π)^8 and dihedral angles φ ∈ (−π, π]^7.
Bond lengths are fixed at 3.832 Å — never predicted, always exactly correct.

Riemannian structure
--------------------
Dihedral angles live on S¹ (a circle), so the joint dihedral space is a flat
torus T^7 — a compact Riemannian manifold. The geodesic distance between two
angles is the wrapped angular difference. Bond angles live in (0, π) ⊂ ℝ;
geodesics there are just straight lines.

Flow matching
-------------
OT-CFM straight-line (geodesic) paths in internal coordinate space:
    θ_t = θ₀ + t · (θ₁ − θ₀)                   (bond angles: linear)
    φ_t = φ₀ + t · angle_wrap(φ₁ − φ₀)          (dihedrals: geodesic on S¹)

Constant conditional velocity fields (the OT property):
    u_θ = θ₁ − θ₀
    u_φ = angle_wrap(φ₁ − φ₀)

Training loss
-------------
Normalised MSE to balance the two DOF types (bond angle velocities ~0.4 rad
vs dihedral velocities ~1.8 rad; without normalisation dihedrals dominate 20×):

    L = MSE(dθ/dt_pred / σ_θ,  u_θ / σ_θ)
      + MSE(dφ/dt_pred / σ_φ,  u_φ / σ_φ)

σ_θ and σ_φ are empirical velocity stds computed by train_torsion.py and
stored in the config/checkpoint. Default fallbacks: 0.40 and 1.81 rad.

Source distribution (x₀)
-------------------------
    θ₀ ~ Gaussian(θ_mean, theta_source_std rad), clamped to (0.05, π − 0.05)
    φ₀ ~ Uniform(−π, π)  by default (maximum-entropy prior on S¹)
       OR WrappedNormal(0, phi_source_std) when phi_source_std is provided.
         phi_source_std can be a scalar or a (7,) tensor (per-dihedral).
         Passing the per-dihedral data std creates much shorter flow paths
         because the source distribution is already concentrated near the
         Ramachandran modes of the target.

Per-position φ loss weighting (optional)
-----------------------------------------
phi_weights: None (uniform, default) or (7,) tensor of inverse-variance weights
    w_i = 1 / σ²_i computed from training data.  Down-weights high-entropy
    terminal dihedrals and up-weights the tightly constrained central ones
    (φ₃–φ₅ around the β-hairpin turn).

Energy conditioning
-------------------
CFG identical to ZeroCoMFlowMatching:
    τ ∈ [0, 1]  →  e_z = 4τ − 2   (±2σ of training energy distribution)
    v_guided = v_uncond + guidance_scale · (v_cond − v_uncond)

ODE integration
---------------
Heun's 2nd-order predictor-corrector; bond angles clamped to (0.05, π − 0.05)
and dihedrals wrapped to (−π, π] after each step.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Union

from models.internal_coords import angle_wrap, N_ANGLES, N_DIHEDRALS


class TorsionalFlowMatching(nn.Module):
    """
    OT-CFM framework in internal coordinate (torsion) space.

    Instantiated by train_torsion.py with σ values computed from training data.
    At inference, loaded from checkpoint config in evaluate.py.

    Parameters
    ----------
    sigma_min        : small noise floor; applied to bond-angle interpolant only
    theta_mean       : mean bond angle [rad] from training data (≈ 1.895 rad)
    theta_source_std : std of Gaussian source for bond angles (default 0.30 rad)
    theta_scale      : velocity std for bond angles (loss normalisation, ≈ 0.40)
    phi_scale        : velocity std for dihedrals (loss normalisation, ≈ 1.81)
    """

    def __init__(
        self,
        sigma_min:        float = 1e-4,
        theta_mean:       float = 1.895,   # arccos(−0.320) ≈ 108.6°
        theta_source_std: float = 0.30,
        theta_scale:      float = 0.40,
        phi_scale:        float = 1.81,
        phi_source_std:   Optional[Union[float, Tensor]] = None,
        phi_weights:      Optional[Tensor] = None,
    ):
        """
        phi_source_std : if given, sample φ₀ ~ WrappedNormal(0, phi_source_std)
                         instead of Uniform(−π,π).  Can be a scalar or a (7,)
                         per-dihedral tensor (stored as a buffer for device moves).
        phi_weights    : if given, a (7,) tensor of per-position loss weights
                         applied to the φ MSE term.  Typically 1/σ²_data_i.
        """
        super().__init__()
        self.sigma_min        = sigma_min
        self.theta_mean       = theta_mean
        self.theta_source_std = theta_source_std
        self.theta_scale      = theta_scale
        self.phi_scale        = phi_scale

        # Register as buffers so they move with .to(device) and are saved in ckpt.
        # Always register (even as None) so load_state_dict can restore them.
        if phi_source_std is not None and not isinstance(phi_source_std, Tensor):
            phi_source_std = torch.tensor(phi_source_std, dtype=torch.float32)
        self.register_buffer('phi_source_std', phi_source_std)

        if phi_weights is not None and not isinstance(phi_weights, Tensor):
            phi_weights = torch.tensor(phi_weights, dtype=torch.float32)
        self.register_buffer('phi_weights', phi_weights)

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _scale_t(self, t: Tensor) -> Tensor:
        """Map t ∈ [0,1] → [0, 999] for SinusoidalTimestepEmbedding."""
        return t * 999.0

    # ── Source distribution ───────────────────────────────────────────────────

    def _sample_source(self, B: int, device) -> tuple[Tensor, Tensor]:
        """
        Sample source (t=0) distribution:
            θ₀ ~ Gaussian(theta_mean, theta_source_std), clamped to (0.05, π−0.05)
            φ₀ ~ Uniform(−π, π)                             [default]
               OR WrappedNormal(0, phi_source_std)           [when phi_source_std is set]
        """
        theta0 = (
            self.theta_mean
            + self.theta_source_std * torch.randn(B, N_ANGLES, device=device)
        ).clamp(0.05, math.pi - 0.05)

        if self.phi_source_std is not None:
            # phi_source_std can be scalar or (7,) per-dihedral tensor
            std = self.phi_source_std.to(device)       # (7,) or scalar
            phi0 = angle_wrap(std * torch.randn(B, N_DIHEDRALS, device=device))
        else:
            phi0 = 2 * math.pi * torch.rand(B, N_DIHEDRALS, device=device) - math.pi

        return theta0, phi0

    # ── Interpolation ─────────────────────────────────────────────────────────

    def _interpolate(
        self,
        theta0: Tensor, theta1: Tensor,
        phi0:   Tensor, phi1:   Tensor,
        t:      Tensor,
    ) -> tuple[Tensor, Tensor]:
        """
        Compute x_t = x₀ + t · (x₁ − x₀) for bond angles (linear),
        and    φ_t = φ₀ + t · Δφ for dihedrals (geodesic on S¹).
        """
        t_ = t.unsqueeze(-1)                       # (B, 1)
        theta_t = theta0 + t_ * (theta1 - theta0)

        delta_phi = angle_wrap(phi1 - phi0)         # (B, 7), shortest arc ∈ (−π, π]
        phi_t     = angle_wrap(phi0 + t_ * delta_phi)

        return theta_t, phi_t

    # ── Target velocities ─────────────────────────────────────────────────────

    def _target_velocity(
        self,
        theta0: Tensor, theta1: Tensor,
        phi0:   Tensor, phi1:   Tensor,
    ) -> tuple[Tensor, Tensor]:
        """
        Constant OT-CFM conditional velocities:
            u_θ = θ₁ − θ₀
            u_φ = angle_wrap(φ₁ − φ₀)   (shortest-arc angular velocity)
        """
        return theta1 - theta0, angle_wrap(phi1 - phi0)

    # ── Training loss ─────────────────────────────────────────────────────────

    def training_loss_energy(
        self,
        model:    nn.Module,
        theta1:   Tensor,    # (B, 8) bond angles from data [rad]
        phi1:     Tensor,    # (B, 7) dihedral angles from data [rad]
        energy_z: Tensor,    # (B,)   z-normalised energy
    ) -> Tensor:
        """
        Torsional OT-CFM loss with energy conditioning and velocity normalisation.

        Samples t ∈ [0,1], draws (θ₀, φ₀) from source, computes interpolant,
        regresses predicted velocities to analytic targets.
        """
        B = theta1.shape[0]
        t = torch.rand(B, device=theta1.device)     # (B,) ∈ [0, 1]

        theta0, phi0 = self._sample_source(B, theta1.device)

        theta_t, phi_t = self._interpolate(theta0, theta1, phi0, phi1, t)
        u_theta, u_phi = self._target_velocity(theta0, theta1, phi0, phi1)

        pred_theta, pred_phi = model(theta_t, phi_t, self._scale_t(t), energy_z)

        # Normalised MSE: divides by velocity scale so both terms have unit variance
        theta_loss = F.mse_loss(pred_theta / self.theta_scale,
                                u_theta    / self.theta_scale)

        if self.phi_weights is not None:
            # Per-position weighting: w_i = 1/σ²_data_i (normalised to mean=1)
            # Scales the squared error per dihedral before averaging.
            phi_err  = (pred_phi - u_phi) / self.phi_scale   # (B, 7)
            w        = self.phi_weights.to(device=phi_err.device, dtype=phi_err.dtype)
            phi_loss = (w * phi_err ** 2).mean()
        else:
            phi_loss = F.mse_loss(pred_phi   / self.phi_scale,
                                  u_phi      / self.phi_scale)

        return theta_loss + phi_loss

    # ── Sampling: Heun ODE ────────────────────────────────────────────────────

    @torch.no_grad()
    def ddim_sample_cfg(
        self,
        model:          nn.Module,
        B:              int,
        device:         str   = 'cuda',
        ddim_steps:     int   = 100,
        tau:            float = 0.5,
        guidance_scale: float = 1.0,
    ) -> tuple[Tensor, Tensor]:
        """
        Generate B structures using Heun's 2nd-order ODE with CFG guidance.

        Temperature mapping (same as all v2 models):
            τ ∈ [0,1] → e_z = 4τ − 2

        guidance_scale = 1.0 : single forward pass (no CFG amplification)
        guidance_scale > 1.0 : amplify energy conditioning toward tau

        Returns (theta, phi) in radians: (B, 8) and (B, 7).
        Call models.internal_coords.internal_to_cartesian(theta, phi)
        to get Cartesian coordinates in Ångströms.
        """
        model.eval()
        e_z = torch.full((B,), 4.0 * tau - 2.0, device=device)

        theta, phi = self._sample_source(B, device)
        dt = 1.0 / ddim_steps

        for i in range(ddim_steps):
            t_cur  = torch.full((B,), self._scale_t(i / ddim_steps),           device=device)
            t_next = torch.full((B,), self._scale_t((i + 1) / ddim_steps),     device=device)

            # ── Velocity at current step (CFG) ────────────────────────────
            v_th_c, v_ph_c = model(theta, phi, t_cur, e_z)

            if guidance_scale != 1.0:
                v_th_u, v_ph_u = model(theta, phi, t_cur, None)
                v_th = v_th_u + guidance_scale * (v_th_c - v_th_u)
                v_ph = v_ph_u + guidance_scale * (v_ph_c - v_ph_u)
            else:
                v_th, v_ph = v_th_c, v_ph_c

            # ── Euler predictor ────────────────────────────────────────────
            theta_mid = (theta + dt * v_th).clamp(0.05, math.pi - 0.05)
            phi_mid   = angle_wrap(phi + dt * v_ph)

            # ── Velocity at predicted next step (CFG) ─────────────────────
            v_th2_c, v_ph2_c = model(theta_mid, phi_mid, t_next, e_z)

            if guidance_scale != 1.0:
                v_th2_u, v_ph2_u = model(theta_mid, phi_mid, t_next, None)
                v_th2 = v_th2_u + guidance_scale * (v_th2_c - v_th2_u)
                v_ph2 = v_ph2_u + guidance_scale * (v_ph2_c - v_ph2_u)
            else:
                v_th2, v_ph2 = v_th2_c, v_ph2_c

            # ── Heun corrector ─────────────────────────────────────────────
            theta = (theta + (dt / 2) * (v_th + v_th2)).clamp(0.05, math.pi - 0.05)
            phi   = angle_wrap(phi + (dt / 2) * (v_ph + v_ph2))

        return theta, phi
