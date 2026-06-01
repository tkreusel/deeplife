"""
models/diffusion_zerocom.py
============================
Zero-center-of-mass (zero-CoM) drop-in for GaussianDiffusion.

Why this matters
----------------
The existing GaussianDiffusion.q_sample adds plain Gaussian noise:
    ε ~ N(0, I)
But Cα point clouds live in the ZERO-CoM subspace (Σᵢ xᵢ = 0).
Plain Gaussian noise breaks this: x_t drifts away from zero-CoM.

This class fixes that by projecting all noise onto the zero-CoM subspace:
    ε_raw ~ N(0, I)
    ε     = ε_raw - mean(ε_raw, dim=atoms)   ← zero-CoM projection

Results:
  • x_t stays zero-CoM at every step
  • The model only needs to learn zero-CoM corrections
  • Generated samples are inherently centered (no post-hoc centering needed)

Usage
-----
Import this instead of GaussianDiffusion in train_egnn.py:
    from models.diffusion_zerocom import ZeroCoMGaussianDiffusion as GaussianDiffusion

It subclasses GaussianDiffusion and overrides only the noise-sampling methods,
so all math (schedules, _extract, _predict_x0_from_noise, etc.) is inherited.
"""

import torch
from torch import Tensor
from typing import Optional

# Import the original — we extend it, not replace it
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.diffusion import GaussianDiffusion


class ZeroCoMGaussianDiffusion(GaussianDiffusion):
    """
    GaussianDiffusion with all noise projected to the zero-CoM subspace.

    Inherits everything from GaussianDiffusion; only overrides the methods
    that sample noise:
        q_sample  — adds zero-CoM noise in the forward process
        p_sample  — projects predicted noise before the reverse step
        sample    — starts from zero-CoM Gaussian noise
        ddim_sample — same
    """

    # ── Zero-CoM helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _remove_com(x: Tensor) -> Tensor:
        """
        Subtract center of mass from a (B, N, 3) point cloud.
        After this call: x.mean(dim=1) == 0 for every item in the batch.
        """
        return x - x.mean(dim=1, keepdim=True)

    @staticmethod
    def _zero_com_noise(shape: tuple, device) -> Tensor:
        """
        Sample Gaussian noise projected onto the zero-CoM subspace.

        shape : (B, N, 3)

        Mathematically:
            ε ~ N(0,  (I - 11ᵀ/N) ⊗ I₃ )
        Equivalently:
            ε_raw ~ N(0, I),  then ε = ε_raw - mean(ε_raw)
        """
        eps = torch.randn(shape, device=device)
        return eps - eps.mean(dim=1, keepdim=True)

    # ── Override: forward process ─────────────────────────────────────────────

    def q_sample(
        self,
        x0:    Tensor,
        t:     Tensor,
        noise: Optional[Tensor] = None,
    ):
        """
        Forward process with zero-CoM noise.
        x_t = √ᾱ_t · x₀  +  √(1-ᾱ_t) · ε,    ε ∈ zero-CoM subspace

        Identical to parent except noise is zero-CoM projected.
        """
        if noise is None:
            noise = self._zero_com_noise(x0.shape, x0.device)

        # Inherit the mixing coefficients from the parent
        sqrt_a  = self._extract(self.sqrt_alphas_cumprod,           t, x0.shape)
        sqrt_1a = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x0.shape)

        x_t = sqrt_a * x0 + sqrt_1a * noise
        return x_t, noise

    # ── Override: one reverse step ────────────────────────────────────────────

    @torch.no_grad()
    def p_sample(
        self,
        model,
        x_t:            Tensor,
        t:              int,
        t_tensor:       Tensor,
        guidance_fn     = None,
        guidance_scale: float = 1.0,
    ) -> Tensor:
        """
        One reverse diffusion step with zero-CoM projected noise prediction.
        Identical to parent except:
          1. noise_pred is projected to zero-CoM
          2. stochastic noise added at each step is zero-CoM
        """
        noise_pred = model(x_t, t_tensor)
        noise_pred = self._remove_com(noise_pred)  # ← zero-CoM projection

        if guidance_fn is not None:
            sqrt_1a = self._extract(
                self.sqrt_one_minus_alphas_cumprod, t_tensor, x_t.shape
            )
            force      = guidance_fn(x_t)
            noise_pred = noise_pred - guidance_scale * sqrt_1a * force
            noise_pred = self._remove_com(noise_pred)  # project again after guidance

        betas_t   = self._extract(self.betas,                         t_tensor, x_t.shape)
        sqrt_1a_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t_tensor, x_t.shape)
        recip_a_t = self._extract(self.sqrt_recip_alphas,             t_tensor, x_t.shape)

        mean = recip_a_t * (x_t - betas_t / sqrt_1a_t * noise_pred)
        mean = self._remove_com(mean)  # keep mean zero-CoM

        if t == 0:
            return mean

        log_var = self._extract(
            self.posterior_log_variance_clipped, t_tensor, x_t.shape
        )
        noise = self._zero_com_noise(x_t.shape, x_t.device)  # ← zero-CoM stochastic noise
        return mean + (0.5 * log_var).exp() * noise

    # ── Override: full reverse sampling (DDPM) ────────────────────────────────

    @torch.no_grad()
    def sample(
        self,
        model,
        shape:          tuple,
        device:         str = 'cuda',
        guidance_fn     = None,
        guidance_scale: float = 1.0,
    ) -> Tensor:
        """
        DDPM sampling starting from zero-CoM Gaussian noise.
        Identical to parent except starting noise is zero-CoM projected.
        """
        x = self._zero_com_noise(shape, device)   # ← zero-CoM start

        for t in reversed(range(self.T)):
            t_tensor = torch.full((shape[0],), t, device=device, dtype=torch.long)
            x = self.p_sample(model, x, t, t_tensor, guidance_fn, guidance_scale)

        return x

    # ── Override: DDIM sampling ───────────────────────────────────────────────

    @torch.no_grad()
    def ddim_sample(
        self,
        model,
        shape:          tuple,
        device:         str = 'cuda',
        ddim_steps:     int = 50,
        eta:            float = 0.0,
        guidance_fn     = None,
        guidance_scale: float = 1.0,
    ) -> Tensor:
        """
        DDIM sampling with zero-CoM noise throughout.
        Uses fewer steps (ddim_steps ≪ T) for faster generation.
        """
        step_size = self.T // ddim_steps
        timesteps = list(range(0, self.T, step_size))[::-1]

        x = self._zero_com_noise(shape, device)   # ← zero-CoM start

        for i, t in enumerate(timesteps):
            t_tensor   = torch.full((shape[0],), t, device=device, dtype=torch.long)
            t_prev     = timesteps[i + 1] if i + 1 < len(timesteps) else 0

            alpha_t    = self.alphas_cumprod[t]
            alpha_prev = self.alphas_cumprod[t_prev]

            noise_pred = model(x, t_tensor)
            noise_pred = self._remove_com(noise_pred)  # ← zero-CoM

            if guidance_fn is not None:
                sqrt_1a    = (1 - alpha_t).sqrt()
                force      = guidance_fn(x)
                noise_pred = noise_pred - guidance_scale * sqrt_1a * force
                noise_pred = self._remove_com(noise_pred)

            x0_pred = (x - (1 - alpha_t).sqrt() * noise_pred) / alpha_t.sqrt()
            x0_pred = self._remove_com(x0_pred.clamp(-5, 5))  # ← zero-CoM + clamp

            sigma   = eta * ((1 - alpha_prev) / (1 - alpha_t)
                             * (1 - alpha_t / alpha_prev)).sqrt()
            dir_xt  = (1 - alpha_prev - sigma ** 2).sqrt() * noise_pred

            if t_prev > 0:
                noise = self._zero_com_noise(x.shape, device)  # ← zero-CoM
            else:
                noise = 0

            x = alpha_prev.sqrt() * x0_pred + dir_xt + sigma * noise

        return x
