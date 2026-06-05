"""
models/backbone_torsion_flow.py
================================
Backbone torsional OT-CFM (Riemannian flow matching on a flat torus) for
Chignolin backbone (N, Cα, C atoms, 30 atoms total).

State space: φ ∈ (−π, π]^9 × ψ ∈ (−π, π]^9  (18 DOF)
    φ_i (i=1..9): backbone dihedral around N_i–CA_i bond
    ψ_i (i=0..8): backbone dihedral around CA_i–C_i bond

Analogous to TorsionalFlowMatching in torsion_flow.py but for the backbone
representation. The NeRF reconstruction (internal_to_backbone) guarantees 100%
bond validity for all 29 backbone bonds by construction.

Flow matching
-------------
OT-CFM straight-line geodesic paths on the flat torus T^9 × T^9:
    φ_t = φ_0 + t · angle_wrap(φ_1 − φ_0)    (geodesic on S¹)
    ψ_t = ψ_0 + t · angle_wrap(ψ_1 − ψ_0)

Constant conditional velocity (OT property):
    u_φ = angle_wrap(φ_1 − φ_0)
    u_ψ = angle_wrap(ψ_1 − ψ_0)

Loss
----
Normalised MSE balancing φ and ψ terms:
    L = MSE(dφ/dt_pred / φ_scale, u_φ / φ_scale)
      + MSE(dψ/dt_pred / ψ_scale, u_ψ / ψ_scale)

Optional auxiliary Cartesian loss (t²-weighted):
    Penalises ETE and Rg discrepancy between predicted and true endpoint.

Self-distillation (toggleable)
-------------------------------
When enabled: run k Euler steps with the EMA model (teacher) from (φ_t, ψ_t)
to get a teacher-predicted endpoint, then train the student to match it.
Only activate after a well-converged base model exists.

Source distribution
-------------------
    φ_0 ~ WrappedNormal(0, phi_source_std)   per-dihedral or scalar
    ψ_0 ~ WrappedNormal(0, psi_source_std)   per-dihedral or scalar
    OR Uniform(−π, π) if source_std not provided.

Energy conditioning
-------------------
Same CFG convention as all other models:
    τ ∈ [0,1] → e_z = 4τ − 2  (±2σ of training energy distribution)
    v_guided = v_uncond + guidance_scale · (v_cond − v_uncond)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Union

from models.backbone_internal_coords import (
    angle_wrap, internal_to_backbone, N_PHI, N_PSI,
)


class BackboneTorsionalFlowMatching(nn.Module):
    """
    OT-CFM framework in backbone torsion space (φ, ψ ∈ T^9 × T^9).

    Instantiated by train_backbone_ipa.py with σ values computed from training
    data. At inference, loaded from checkpoint config in evaluate.py.

    Parameters
    ----------
    sigma_min       : small noise floor (applied to bond-angle interpolant only)
    phi_scale       : velocity std for φ [rad] (loss normalisation)
    psi_scale       : velocity std for ψ [rad] (loss normalisation)
    phi_source_std  : std for WrappedNormal φ source; None → Uniform(−π,π)
    psi_source_std  : std for WrappedNormal ψ source; None → Uniform(−π,π)
    phi_weights     : (9,) per-position φ loss weights (1/σ²_data_i, mean=1)
    psi_weights     : (9,) per-position ψ loss weights
    """

    def __init__(
        self,
        sigma_min:      float = 1e-4,
        phi_scale:      float = 1.81,
        psi_scale:      float = 1.81,
        phi_source_std: Optional[Union[float, Tensor]] = None,
        psi_source_std: Optional[Union[float, Tensor]] = None,
        phi_weights:    Optional[Tensor] = None,
        psi_weights:    Optional[Tensor] = None,
    ):
        super().__init__()
        self.sigma_min = sigma_min
        self.phi_scale = phi_scale
        self.psi_scale = psi_scale

        def _to_buf(v):
            if v is not None and not isinstance(v, Tensor):
                return torch.tensor(v, dtype=torch.float32)
            return v

        self.register_buffer('phi_source_std', _to_buf(phi_source_std))
        self.register_buffer('psi_source_std', _to_buf(psi_source_std))
        self.register_buffer('phi_weights',    _to_buf(phi_weights))
        self.register_buffer('psi_weights',    _to_buf(psi_weights))

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _scale_t(self, t: Tensor) -> Tensor:
        """Map t ∈ [0,1] → [0, 999] for SinusoidalTimestepEmbedding."""
        return t * 999.0

    def _sample_source(self, B: int, device) -> tuple[Tensor, Tensor]:
        """Sample source (t=0) distribution for (φ, ψ)."""
        def _sample_one(source_std, n_dof):
            if source_std is not None:
                std = source_std.to(device)
                return angle_wrap(std * torch.randn(B, n_dof, device=device))
            else:
                return 2 * math.pi * torch.rand(B, n_dof, device=device) - math.pi

        phi0 = _sample_one(self.phi_source_std, N_PHI)
        psi0 = _sample_one(self.psi_source_std, N_PSI)
        return phi0, psi0

    # ── Interpolation ─────────────────────────────────────────────────────────

    def _interpolate(
        self,
        phi0: Tensor, phi1: Tensor,
        psi0: Tensor, psi1: Tensor,
        t:    Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Geodesic interpolation on the flat torus."""
        t_ = t.unsqueeze(-1)
        phi_t = angle_wrap(phi0 + t_ * angle_wrap(phi1 - phi0))
        psi_t = angle_wrap(psi0 + t_ * angle_wrap(psi1 - psi0))
        return phi_t, psi_t

    # ── Target velocities ─────────────────────────────────────────────────────

    def _target_velocity(
        self,
        phi0: Tensor, phi1: Tensor,
        psi0: Tensor, psi1: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Constant OT-CFM velocities (shortest arc on S¹)."""
        return angle_wrap(phi1 - phi0), angle_wrap(psi1 - psi0)

    # ── Normalised MSE loss term ───────────────────────────────────────────────

    def _angle_loss(
        self,
        pred:    Tensor,
        target:  Tensor,
        scale:   float,
        weights: Optional[Tensor],
    ) -> Tensor:
        """Normalised MSE with optional per-position weighting."""
        err = (pred - target) / scale   # (B, 9)
        if weights is not None:
            w   = weights.to(device=err.device, dtype=err.dtype)
            return (w * err**2).mean()
        return (err**2).mean()

    # ── Standard training loss ────────────────────────────────────────────────

    def training_loss_energy(
        self,
        model:    nn.Module,
        phi1:     Tensor,    # (B, 9) φ data [rad]
        psi1:     Tensor,    # (B, 9) ψ data [rad]
        energy_z: Tensor,    # (B,)   z-normalised energy
    ) -> Tensor:
        """OT-CFM loss on backbone torsion angles with energy conditioning."""
        B = phi1.shape[0]
        t = torch.rand(B, device=phi1.device)

        phi0, psi0 = self._sample_source(B, phi1.device)
        phi_t, psi_t = self._interpolate(phi0, phi1, psi0, psi1, t)
        u_phi, u_psi = self._target_velocity(phi0, phi1, psi0, psi1)

        v_psi, v_phi = model(psi_t, phi_t, self._scale_t(t), energy_z)

        phi_loss = self._angle_loss(v_phi, u_phi, self.phi_scale, self.phi_weights)
        psi_loss = self._angle_loss(v_psi, u_psi, self.psi_scale, self.psi_weights)
        return phi_loss + psi_loss

    # ── Training loss with auxiliary Cartesian penalty ─────────────────────────

    def training_loss_with_cartesian(
        self,
        model:       nn.Module,
        phi1:        Tensor,
        psi1:        Tensor,
        energy_z:    Tensor,
        cart_weight: float = 0.1,
    ) -> tuple[Tensor, Tensor]:
        """
        OT-CFM loss + t²-weighted Cartesian structural loss (ETE + Rg).

        Returns (total_loss, fm_loss) so the training script can log separately.
        """
        B = phi1.shape[0]
        t = torch.rand(B, device=phi1.device)

        phi0, psi0 = self._sample_source(B, phi1.device)
        phi_t, psi_t = self._interpolate(phi0, phi1, psi0, psi1, t)
        u_phi, u_psi = self._target_velocity(phi0, phi1, psi0, psi1)

        v_psi, v_phi = model(psi_t, phi_t, self._scale_t(t), energy_z)

        phi_loss = self._angle_loss(v_phi, u_phi, self.phi_scale, self.phi_weights)
        psi_loss = self._angle_loss(v_psi, u_psi, self.psi_scale, self.psi_weights)
        fm_loss  = phi_loss + psi_loss

        if cart_weight <= 0.0:
            return fm_loss, fm_loss

        # ── Predicted endpoint ─────────────────────────────────────────────────
        t_ = t.unsqueeze(-1)
        phi1_pred = angle_wrap(phi_t + (1.0 - t_) * v_phi)
        psi1_pred = angle_wrap(psi_t + (1.0 - t_) * v_psi)

        # NeRF reconstruction + CoM-centering
        x1_pred = internal_to_backbone(phi1_pred, psi1_pred)           # (B, 30, 3)
        x1_true = internal_to_backbone(phi1, psi1)

        x1_pred = x1_pred - x1_pred.mean(dim=1, keepdim=True)
        x1_true = x1_true - x1_true.mean(dim=1, keepdim=True)

        # CA positions only (index 1, 4, 7, …)
        ca_pred = x1_pred[:, 1::3]   # (B, 10, 3)
        ca_true = x1_true[:, 1::3]

        ete_pred = (ca_pred[:, -1] - ca_pred[:, 0]).norm(dim=-1)   # (B,)
        ete_true = (ca_true[:, -1] - ca_true[:, 0]).norm(dim=-1)

        rg_pred = ((ca_pred - ca_pred.mean(dim=1, keepdim=True))**2
                   ).mean(dim=-1).mean(dim=-1).sqrt()
        rg_true = ((ca_true - ca_true.mean(dim=1, keepdim=True))**2
                   ).mean(dim=-1).mean(dim=-1).sqrt()

        ete_loss = F.mse_loss(ete_pred, ete_true)
        rg_loss  = F.mse_loss(rg_pred,  rg_true)

        # Cα clash penalty: soft repulsion for non-bonded pairs (|i-j|≥2) below 3.5 Å
        diff = ca_pred.unsqueeze(2) - ca_pred.unsqueeze(1)   # (B, 10, 10, 3)
        dist = diff.norm(dim=-1)                              # (B, 10, 10)
        idx  = torch.arange(10, device=phi1.device)
        sep  = (idx.unsqueeze(1) - idx.unsqueeze(0)).abs()   # (10, 10)
        mask = (sep >= 2).float()
        clash_loss = (F.relu(3.5 - dist) ** 2 * mask).mean()

        # t²-weighting: zero near t=0, strong near t=1
        t_sq   = (t**2).mean()
        cart_l = t_sq * (ete_loss + rg_loss + 0.5 * clash_loss)

        total = fm_loss + cart_weight * cart_l
        return total, fm_loss

    # ── Training loss with self-distillation ──────────────────────────────────

    def training_loss_with_distillation(
        self,
        model:           nn.Module,
        ema_model:       nn.Module,   # teacher (EMA shadow, eval mode)
        phi1:            Tensor,
        psi1:            Tensor,
        energy_z:        Tensor,
        cart_weight:     float = 0.1,
        distill_weight:  float = 0.5,
        distill_steps:   int   = 5,
    ) -> tuple[Tensor, Tensor]:
        """
        OT-CFM loss + Cartesian loss + self-distillation loss.

        Teacher: run distill_steps Euler steps from (φ_t, ψ_t) using EMA model.
        Student: predict endpoint directly from (φ_t, ψ_t, t).
        Distillation loss: angular MSE between student and teacher endpoints.

        Returns (total_loss, fm_loss).
        """
        total, fm_loss = self.training_loss_with_cartesian(
            model, phi1, psi1, energy_z, cart_weight=cart_weight,
        )

        if distill_weight <= 0.0 or distill_steps <= 0:
            return total, fm_loss

        # ── Compute teacher endpoint from (φ_t, ψ_t) ─────────────────────────
        B = phi1.shape[0]
        t_train = torch.rand(B, device=phi1.device)
        phi0, psi0 = self._sample_source(B, phi1.device)
        phi_t, psi_t = self._interpolate(phi0, phi1, psi0, psi1, t_train)

        dt = 1.0 / 100   # small Euler step (same as sampling)
        phi_k = phi_t.clone()
        psi_k = psi_t.clone()
        t_k   = t_train.clone()

        with torch.no_grad():
            for _ in range(distill_steps):
                v_psi_k, v_phi_k = ema_model(psi_k, phi_k, self._scale_t(t_k), energy_z)
                phi_k = angle_wrap(phi_k + dt * v_phi_k)
                psi_k = angle_wrap(psi_k + dt * v_psi_k)
                t_k   = (t_k + dt).clamp(max=1.0 - 1e-6)

            # Teacher's one-step endpoint prediction from t_k
            v_psi_t, v_phi_t = ema_model(psi_k, phi_k, self._scale_t(t_k), energy_z)
            t_k_ = t_k.unsqueeze(-1)
            phi1_teacher = angle_wrap(phi_k + (1.0 - t_k_) * v_phi_t)
            psi1_teacher = angle_wrap(psi_k + (1.0 - t_k_) * v_psi_t)

        # ── Student's one-step endpoint prediction from (φ_t, ψ_t) ──────────
        v_psi_s, v_phi_s = model(psi_t, phi_t, self._scale_t(t_train), energy_z)
        t_ = t_train.unsqueeze(-1)
        phi1_student = angle_wrap(phi_t + (1.0 - t_) * v_phi_s)
        psi1_student = angle_wrap(psi_t + (1.0 - t_) * v_psi_s)

        # Distillation loss: angular MSE between endpoints
        phi_distill = self._angle_loss(phi1_student, phi1_teacher,
                                       self.phi_scale, None)
        psi_distill = self._angle_loss(psi1_student, psi1_teacher,
                                       self.psi_scale, None)
        distill_l = phi_distill + psi_distill

        return total + distill_weight * distill_l, fm_loss

    # ── Heun ODE sampler (CFG) ────────────────────────────────────────────────

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

        Temperature mapping: τ ∈ [0,1] → e_z = 4τ − 2
        guidance_scale = 1.0: single forward pass (no CFG amplification)
        guidance_scale > 1.0: amplify energy conditioning toward tau

        Returns (phi, psi) in radians: (B, 9) and (B, 9).
        Call internal_to_backbone(phi, psi) to get Cartesian coordinates.
        """
        model.eval()
        e_z = torch.full((B,), 4.0 * tau - 2.0, device=device)

        phi, psi = self._sample_source(B, device)
        dt = 1.0 / ddim_steps

        for i in range(ddim_steps):
            t_cur  = torch.full((B,), self._scale_t(i / ddim_steps),         device=device)
            t_next = torch.full((B,), self._scale_t((i + 1) / ddim_steps),   device=device)

            # ── Velocity at current step (with optional CFG) ────────────────
            v_psi_c, v_phi_c = model(psi, phi, t_cur, e_z)

            if guidance_scale != 1.0:
                v_psi_u, v_phi_u = model(psi, phi, t_cur, None)
                v_psi = v_psi_u + guidance_scale * (v_psi_c - v_psi_u)
                v_phi = v_phi_u + guidance_scale * (v_phi_c - v_phi_u)
            else:
                v_psi, v_phi = v_psi_c, v_phi_c

            # ── Euler predictor ──────────────────────────────────────────────
            phi_mid = angle_wrap(phi + dt * v_phi)
            psi_mid = angle_wrap(psi + dt * v_psi)

            # ── Velocity at predicted next step ─────────────────────────────
            v_psi2_c, v_phi2_c = model(psi_mid, phi_mid, t_next, e_z)

            if guidance_scale != 1.0:
                v_psi2_u, v_phi2_u = model(psi_mid, phi_mid, t_next, None)
                v_psi2 = v_psi2_u + guidance_scale * (v_psi2_c - v_psi2_u)
                v_phi2 = v_phi2_u + guidance_scale * (v_phi2_c - v_phi2_u)
            else:
                v_psi2, v_phi2 = v_psi2_c, v_phi2_c

            # ── Heun corrector ──────────────────────────────────────────────
            phi = angle_wrap(phi + (dt / 2) * (v_phi + v_phi2))
            psi = angle_wrap(psi + (dt / 2) * (v_psi + v_psi2))

        return phi, psi
