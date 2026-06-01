"""
models/flow_matching.py
========================
Optimal Transport Conditional Flow Matching (OT-CFM) for protein Cα coordinates.

Convention
----------
  t = 0 : source — Gaussian noise  x_0 ~ N(0, I)
  t = 1 : target — protein data    x_1 ~ p_data

Straight-line interpolant (Lipman et al. 2022 / Liu et al. 2022):
    x_t = (1 - (1 - σ_min) · t) · x_0  +  t · x_1

Conditional velocity (analytic target, no learned schedule needed):
    u_t(x_t | x_0, x_1) = x_1 - (1 - σ_min) · x_0

Training objective:
    L(θ) = E_{t,x_0,x_1} [ || v_θ(x_t, t) - u_t ||² ]

Sampling: integrate  dx/dt = v_θ(x_t, t)  from t=0 to t=1.
  - sample()      : Euler (1st order, fast)
  - ddim_sample() : Heun's predictor-corrector (2nd order, better quality per NFE)
                    — drop-in replacement for GaussianDiffusion.ddim_sample()

Time encoding
-------------
The existing SinusoidalTimestepEmbedding is designed for integer timesteps in
[0, T-1] (typically T=500).  Here t ∈ [0,1], so we scale to [0, 999] before
passing to the model — matching the embedding's useful frequency range.

SE(3) invariance
----------------
Pair this class with EGNNScoreNetwork as the velocity network.  EGNN is
SE(3)-equivariant: v(Rx + b, t) = R · v(x, t).  Combined with zero-CoM noise
sampling (ZeroCoMFlowMatching), the full pipeline satisfies:
    p(generated) is invariant under rotations and translations.
This mirrors the FoldFlow framework (Bose et al., 2024).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContinuousFlowMatching(nn.Module):
    """
    OT-CFM base class for R^(N×3) coordinate spaces.

    Use with any velocity network v_θ(x_t, t) → (B, N, 3).
    For SE(3)-equivariant generation, use ZeroCoMFlowMatching + EGNNScoreNetwork.
    """

    def __init__(self, sigma_min: float = 1e-4):
        super().__init__()
        self.sigma_min = sigma_min

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _sample_noise(self, shape: tuple, device) -> torch.Tensor:
        """Source distribution: standard Gaussian."""
        return torch.randn(shape, device=device)

    def _scale_t(self, t: torch.Tensor) -> torch.Tensor:
        """Map t ∈ [0,1] → [0,999] for the sinusoidal timestep embedding."""
        return t * 999.0

    def _interpolate(self, x0: torch.Tensor, x1: torch.Tensor,
                     t: torch.Tensor) -> torch.Tensor:
        """x_t = (1-(1-σ_min)·t)·x0 + t·x1"""
        t_ = t[:, None, None]                        # (B,1,1) for broadcast
        return (1.0 - (1.0 - self.sigma_min) * t_) * x0 + t_ * x1

    def _target_velocity(self, x0: torch.Tensor,
                         x1: torch.Tensor) -> torch.Tensor:
        """Analytic conditional velocity: d(x_t)/dt = x1 - (1-σ_min)·x0"""
        return x1 - (1.0 - self.sigma_min) * x0

    # ── Training ──────────────────────────────────────────────────────────────

    def training_loss(
        self,
        model:          nn.Module,
        x1:             torch.Tensor,   # (B, N, 3)  clean structures
        physics_weight: float = 0.0,
        physics_fn      = None,
    ) -> torch.Tensor:
        """
        Flow matching loss for one batch.

        Samples a random t ∈ [0,1] per structure, interpolates to x_t,
        and regresses the predicted velocity toward the analytic target.
        """
        B = x1.shape[0]

        t  = torch.rand(B, device=x1.device)                     # (B,)
        x0 = self._sample_noise(x1.shape, x1.device)             # (B, N, 3)

        x_t    = self._interpolate(x0, x1, t)                    # (B, N, 3)
        target = self._target_velocity(x0, x1)                    # (B, N, 3)

        pred = model(x_t, self._scale_t(t))                      # (B, N, 3)

        return F.mse_loss(pred, target)

    # ── Sampling ──────────────────────────────────────────────────────────────

    @torch.no_grad()
    def sample(
        self,
        model:          nn.Module,
        shape:          tuple,
        device          = 'cuda',
        n_steps:        int = 100,
        guidance_fn     = None,
        guidance_scale: float = 1.0,
    ) -> torch.Tensor:
        """
        Euler ODE integration from t=0 (noise) to t=1 (data).
        Simple but effective; use ddim_sample for better quality at the same NFE.
        """
        x  = self._sample_noise(shape, device)
        dt = 1.0 / n_steps

        for i in range(n_steps):
            t_val    = i / n_steps
            t_tensor = torch.full((shape[0],), self._scale_t(t_val), device=device)
            v = model(x, t_tensor)

            if guidance_fn is not None:
                v = v + guidance_scale * guidance_fn(x)

            x = x + dt * v

        return x

    @torch.no_grad()
    def ddim_sample(
        self,
        model:          nn.Module,
        shape:          tuple,
        device          = 'cuda',
        ddim_steps:     int = 100,
        eta:            float = 0.0,    # accepted for API compat, ignored (ODE is deterministic)
        guidance_fn     = None,
        guidance_scale: float = 1.0,
    ) -> torch.Tensor:
        """
        Heun's method (2nd-order predictor-corrector ODE integration).

        Uses the same number of model evaluations as Euler with half the steps
        for the same wall-clock time, but achieves O(dt²) vs O(dt) error.
        The `ddim_steps` and `eta` signature matches GaussianDiffusion.ddim_sample()
        so evaluate.py and quick_sample.py work unchanged.
        """
        x  = self._sample_noise(shape, device)
        dt = 1.0 / ddim_steps

        for i in range(ddim_steps):
            t_val = i / ddim_steps

            # ── Euler predictor ────────────────────────────────────────────
            t = torch.full((shape[0],), self._scale_t(t_val), device=device)
            v1 = model(x, t)
            if guidance_fn is not None:
                v1 = v1 + guidance_scale * guidance_fn(x)
            x_pred = x + dt * v1

            # ── Heun's corrector (skip on the very last step) ──────────────
            if i < ddim_steps - 1:
                t_next = torch.full((shape[0],),
                                    self._scale_t(t_val + dt), device=device)
                v2 = model(x_pred, t_next)
                if guidance_fn is not None:
                    v2 = v2 + guidance_scale * guidance_fn(x_pred)
                x = x + dt * (v1 + v2) * 0.5
            else:
                x = x_pred

        return x


# ─────────────────────────────────────────────────────────────────────────────

class ZeroCoMFlowMatching(ContinuousFlowMatching):
    """
    Flow matching restricted to the zero-center-of-mass subspace.

    Why: Chignolin Cα coordinates are always zero-CoM (centered per-structure).
    Plain Gaussian noise breaks this; projecting noise and velocities onto the
    zero-CoM subspace keeps every x_t zero-CoM throughout the ODE trajectory.

    Since the straight-line path between two zero-CoM points is zero-CoM by
    linearity, and EGNNScoreNetwork already projects its output to zero-CoM,
    the generated distribution is inherently centered.  The explicit projection
    here ensures numerical correctness even if the model output drifts slightly.

    Use with EGNNScoreNetwork for full SE(3)-equivariant generation:
        - EGNN is equivariant: v(Rx+b, t) = R·v(x, t)
        - Zero-CoM noise is translation-invariant
        → generated distribution p(x) is SE(3)-invariant
    """

    # ── Zero-CoM helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _remove_com(x: torch.Tensor) -> torch.Tensor:
        """Subtract center of mass. (B, N, 3) → zero-CoM (B, N, 3)."""
        return x - x.mean(dim=1, keepdim=True)

    def _sample_noise(self, shape: tuple, device) -> torch.Tensor:
        """Zero-CoM Gaussian noise: ε ~ N(0, (I - 11ᵀ/N) ⊗ I₃)"""
        eps = torch.randn(shape, device=device)
        return self._remove_com(eps)

    # ── Training ──────────────────────────────────────────────────────────────

    def training_loss(
        self,
        model:          nn.Module,
        x1:             torch.Tensor,
        physics_weight: float = 0.0,
        physics_fn      = None,
    ) -> torch.Tensor:
        B = x1.shape[0]

        t  = torch.rand(B, device=x1.device)
        x0 = self._sample_noise(x1.shape, x1.device)

        x_t    = self._remove_com(self._interpolate(x0, x1, t))
        target = self._remove_com(self._target_velocity(x0, x1))

        pred = self._remove_com(model(x_t, self._scale_t(t)))

        return F.mse_loss(pred, target)

    # ── Sampling ──────────────────────────────────────────────────────────────

    @torch.no_grad()
    def sample(
        self,
        model:          nn.Module,
        shape:          tuple,
        device          = 'cuda',
        n_steps:        int = 100,
        guidance_fn     = None,
        guidance_scale: float = 1.0,
    ) -> torch.Tensor:
        x  = self._sample_noise(shape, device)
        dt = 1.0 / n_steps

        for i in range(n_steps):
            t_tensor = torch.full((shape[0],),
                                  self._scale_t(i / n_steps), device=device)
            v = self._remove_com(model(x, t_tensor))

            if guidance_fn is not None:
                v = v + guidance_scale * self._remove_com(guidance_fn(x))

            x = x + dt * v  # zero-CoM + zero-CoM·dt stays zero-CoM

        return x

    @torch.no_grad()
    def ddim_sample(
        self,
        model:          nn.Module,
        shape:          tuple,
        device          = 'cuda',
        ddim_steps:     int = 100,
        eta:            float = 0.0,
        guidance_fn     = None,
        guidance_scale: float = 1.0,
    ) -> torch.Tensor:
        """Heun's method on the zero-CoM subspace."""
        x  = self._sample_noise(shape, device)
        dt = 1.0 / ddim_steps

        for i in range(ddim_steps):
            t_val = i / ddim_steps

            t  = torch.full((shape[0],), self._scale_t(t_val), device=device)
            v1 = self._remove_com(model(x, t))
            if guidance_fn is not None:
                v1 = v1 + guidance_scale * self._remove_com(guidance_fn(x))
            x_pred = x + dt * v1  # stays zero-CoM

            if i < ddim_steps - 1:
                t_next = torch.full((shape[0],),
                                    self._scale_t(t_val + dt), device=device)
                v2 = self._remove_com(model(x_pred, t_next))
                if guidance_fn is not None:
                    v2 = v2 + guidance_scale * self._remove_com(guidance_fn(x_pred))
                x = x + dt * (v1 + v2) * 0.5
            else:
                x = x_pred

        return x
