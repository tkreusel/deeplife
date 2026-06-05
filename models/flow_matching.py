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

    def _reconstruct_x1(
        self, pred: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """
        Reconstruct x₁ from a velocity prediction and noisy x_t.

        From the OT-CFM interpolant  x_t = x₀ + t·v  (see derivation in
        flow_matching.py module docstring), solving for x₁ gives:

            x₁ = v · α  +  (1 − σ_min) · x_t
            where  α = 1 − (1 − σ_min) · t

        At t=0: x₁ ≈ pred + x₀  (unreliable; pred is a raw noise prediction)
        At t=1: x₁ ≈ (1−σ) · x_t ≈ x_t  (reliable; x_t ≈ x₁)

        This estimate is used to apply physics constraints during training.
        It is clamped to ±5 (normalised units, i.e. ±25 Å) to prevent NaNs
        from exploding gradients at early training steps.
        """
        t_  = t[:, None, None]
        alpha = 1.0 - (1.0 - self.sigma_min) * t_
        return (pred * alpha + (1.0 - self.sigma_min) * x_t).clamp(-5.0, 5.0)

    def training_loss(
        self,
        model:          nn.Module,
        x1:             torch.Tensor,   # (B, N, 3)  clean structures
        physics_weight: float = 0.0,
        physics_fn      = None,         # callable: (B,N,3) normalised → (B,) loss
    ) -> torch.Tensor:
        """
        Flow matching loss for one batch, with optional physics regularisation.

        Samples a random t ∈ [0,1] per structure, interpolates to x_t,
        and regresses the predicted velocity toward the analytic target.

        Physics regularisation (when physics_weight > 0 and physics_fn given):
            Reconstructs x₁_pred from the velocity prediction and applies
            physics_fn to it. The physics loss is weighted by t² so that it
            is near-zero at t≈0 (x₁_pred unreliable) and full at t≈1 (x₁_pred
            converges to the real x₁). Gradient flows back through x₁_pred
            to the model velocity prediction.

            physics_fn must accept (B, N, 3) normalised coords and return (B,).
            Use models.physics.ChignolinPhysics for the standard constraints.
        """
        B = x1.shape[0]

        t  = torch.rand(B, device=x1.device)                     # (B,)
        x0 = self._sample_noise(x1.shape, x1.device)             # (B, N, 3)

        x_t    = self._interpolate(x0, x1, t)                    # (B, N, 3)
        target = self._target_velocity(x0, x1)                    # (B, N, 3)

        pred = model(x_t, self._scale_t(t))                      # (B, N, 3)

        flow_loss = F.mse_loss(pred, target)

        if physics_weight > 0.0 and physics_fn is not None:
            x1_pred    = self._reconstruct_x1(pred, x_t, t)      # (B, N, 3)
            per_sample = physics_fn(x1_pred)                      # (B,)
            # t²-weight: physics signal only meaningful when x₁_pred is reliable
            phys_loss  = (t ** 2 * per_sample).mean()
            flow_loss  = flow_loss + physics_weight * phys_loss

        return flow_loss

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

    Energy-conditioned extensions (for EGNNEnergyScoreNetwork):
        training_loss_energy()  — pass z-normalised energy per structure
        ddim_sample_cfg()       — Heun ODE with CFG guidance at given temperature

    Bond-graph harmonic prior (optional):
        Pass prior_fn=sample_all_atom_chain (or sample_ca_chain) to replace
        Gaussian x₀ with a geometry-preserving sampler where all covalent bond
        lengths are correct by construction at t=0.  When prior_fn=None (default)
        the standard zero-CoM Gaussian is used (backward-compatible).
    """

    def __init__(self, sigma_min: float = 1e-4, prior_fn=None):
        """
        sigma_min : OT-CFM lower bound on the source distribution std
        prior_fn  : optional callable(B, N, device) → (B, N, 3) in model units,
                    zero-CoM.  Replaces Gaussian x₀ with a geometry-preserving
                    sampler.  Set via configs/se3flow_all_atom_v2.yaml
                    flow.harmonic_prior: true and wired in train_flow_energy.py.
        """
        super().__init__(sigma_min)
        self.prior_fn = prior_fn

    # ── Zero-CoM helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _remove_com(x: torch.Tensor) -> torch.Tensor:
        """Subtract center of mass. (B, N, 3) → zero-CoM (B, N, 3)."""
        return x - x.mean(dim=1, keepdim=True)

    def _sample_noise(self, shape: tuple, device) -> torch.Tensor:
        """
        Sample x₀ from the source distribution.

        If prior_fn is set: use the geometry-preserving bond-graph sampler
        (all covalent bonds at ideal lengths at t=0).
        Otherwise: standard zero-CoM Gaussian.
        """
        if self.prior_fn is not None:
            B, N, _ = shape
            x0 = self.prior_fn(B, N, device)
            return self._remove_com(x0)
        eps = torch.randn(shape, device=device)
        return self._remove_com(eps)

    # ── Training ──────────────────────────────────────────────────────────────

    def training_loss(
        self,
        model:          nn.Module,
        x1:             torch.Tensor,
        physics_weight: float = 0.0,
        physics_fn      = None,         # callable: (B,N,3) normalised → (B,) loss
    ) -> torch.Tensor:
        B = x1.shape[0]

        t  = torch.rand(B, device=x1.device)
        x0 = self._sample_noise(x1.shape, x1.device)

        x_t    = self._remove_com(self._interpolate(x0, x1, t))
        target = self._remove_com(self._target_velocity(x0, x1))

        pred = self._remove_com(model(x_t, self._scale_t(t)))

        flow_loss = F.mse_loss(pred, target)

        if physics_weight > 0.0 and physics_fn is not None:
            # Reconstruct x₁_pred and project to zero-CoM subspace
            x1_pred    = self._remove_com(self._reconstruct_x1(pred, x_t, t))
            per_sample = physics_fn(x1_pred)                      # (B,)
            phys_loss  = (t ** 2 * per_sample).mean()
            flow_loss  = flow_loss + physics_weight * phys_loss

        return flow_loss

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

    # ── SHAKE bond-length projection ─────────────────────────────────────────

    @staticmethod
    def _project_bonds(
        x:          torch.Tensor,
        ideal_norm: float,
        n_iter:     int = 3,
    ) -> torch.Tensor:
        """
        SHAKE-like iterative bond-length projection for a linear chain.

        Projects consecutive atom pairs toward `ideal_norm` (in model units)
        by moving each atom halfway along the correction vector.  SE(3)-
        equivariant: depends only on difference vectors.  Zero-CoM preserving:
        each correction cancels between the two bonded atoms.

        ideal_norm : ideal bond length in model-space units (Å / coord_scale)
        n_iter     : number of SHAKE iterations (2–3 converges for a chain)
        """
        for _ in range(n_iter):
            bonds = x[:, 1:] - x[:, :-1]                           # (B, N-1, 3)
            lens  = bonds.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            corr  = 0.5 * (lens - ideal_norm) * (bonds / lens)     # equivariant
            x     = x.clone()
            x[:, 1:]  -= corr
            x[:, :-1] += corr
        return x - x.mean(dim=1, keepdim=True)                      # re-zero CoM

    # ── Energy-conditioned extensions ─────────────────────────────────────────

    def training_loss_energy(
        self,
        model:          nn.Module,
        x1:             torch.Tensor,   # (B, N, 3)  clean structures
        energy_z:       torch.Tensor,   # (B,)       z-score normalised energy
        physics_weight: float = 0.0,
        physics_fn      = None,
    ) -> torch.Tensor:
        """
        Flow matching loss with energy conditioning.

        Identical to training_loss() except the model receives energy_z
        per structure.  CFG dropout is handled inside the model's forward()
        via its energy_drop_prob parameter — no extra logic needed here.

        energy_z : (B,) z-score normalised energy values
                   Normalisation: e_z = (E_raw - E_mean) / E_std
                   Compute E_mean / E_std from the training set once and
                   pass pre-normalised values at every step.
        """
        B = x1.shape[0]

        t  = torch.rand(B, device=x1.device)
        x0 = self._sample_noise(x1.shape, x1.device)

        x_t    = self._remove_com(self._interpolate(x0, x1, t))
        target = self._remove_com(self._target_velocity(x0, x1))

        # Self-conditioning: 50% of batches use a stop-gradient pre-pass to
        # generate x̂₁, which is then fed back as the sc_x1 argument.
        # The other 50% pass sc_x1=None (teaches both code paths equally).
        sc_x1 = None
        if getattr(model, 'self_cond', False) and torch.rand(1).item() < 0.5:
            with torch.no_grad():
                pred_sc = self._remove_com(
                    model(x_t, self._scale_t(t), energy_z=energy_z, sc_x1=None)
                )
                sc_x1 = self._remove_com(
                    self._reconstruct_x1(pred_sc, x_t, t)
                ).detach()

        pred = self._remove_com(
            model(x_t, self._scale_t(t), energy_z=energy_z, sc_x1=sc_x1)
        )

        flow_loss = F.mse_loss(pred, target)

        if physics_weight > 0.0 and physics_fn is not None:
            x1_pred   = self._remove_com(self._reconstruct_x1(pred, x_t, t))
            phys_loss = (t ** 2 * physics_fn(x1_pred)).mean()
            flow_loss = flow_loss + physics_weight * phys_loss

        return flow_loss

    @torch.no_grad()
    def ddim_sample_cfg(
        self,
        model:           nn.Module,
        shape:           tuple,
        device           = 'cuda',
        ddim_steps:      int   = 100,
        tau:             float = 0.5,
        guidance_scale:  float = 2.0,
        shake_bonds:     bool  = False,
        shake_ideal:     float = None,    # ideal bond length in model units; inferred if None
        shake_frac:      float = 0.2,     # apply SHAKE in the last shake_frac of steps
        shake_iter:      int   = 3,
    ) -> torch.Tensor:
        """
        Heun's ODE with temperature-controlled CFG guidance.

        tau            : float ∈ [0, 1]
                         0 = stable / folded  (low energy,  compact)
                         1 = transient / extended (high energy, unfolded)
        guidance_scale : CFG weight w
                         1.0  → pure conditional (no amplification, single pass)
                         >1.0 → amplified conditioning (two passes per step)
                         0.0  → unconditional

        shake_bonds    : apply SHAKE bond-length projection at late ODE steps
        shake_ideal    : ideal Cα–Cα bond in model units (3.832 / coord_scale)
        shake_frac     : fraction of ODE trajectory where SHAKE is active (default last 20%)
        shake_iter     : SHAKE iterations per step (default 3)
        """
        B   = shape[0]
        e_z = 4.0 * tau - 2.0
        e_t = torch.full((B,), e_z, device=device)

        x  = self._sample_noise(shape, device)
        dt = 1.0 / ddim_steps
        use_cfg        = guidance_scale != 1.0
        shake_start    = int(ddim_steps * (1.0 - shake_frac))
        use_sc         = getattr(model, 'self_cond', False)

        sc_x1 = None  # self-conditioning state: x̂₁ from previous step

        def _call(net, coords, timestep, energy_z):
            """Call model with sc_x1 only when the model supports self-conditioning."""
            if use_sc:
                return self._remove_com(net(coords, timestep, energy_z=energy_z, sc_x1=sc_x1))
            return self._remove_com(net(coords, timestep, energy_z=energy_z))

        for i in range(ddim_steps):
            t_val = i / ddim_steps
            t     = torch.full((B,), self._scale_t(t_val), device=device)

            # ── Euler predictor ──────────────────────────────────────────
            v1_cond = _call(model, x, t, e_t)
            if use_cfg:
                v1_unc = _call(model, x, t, None)
                v1 = v1_unc + guidance_scale * (v1_cond - v1_unc)
            else:
                v1 = v1_cond

            x_pred = x + dt * v1

            # ── Heun's corrector (skip on last step) ─────────────────────
            if i < ddim_steps - 1:
                t_next = torch.full((B,), self._scale_t(t_val + dt), device=device)
                # Update sc_x1 from current v1 before corrector step
                if use_sc:
                    t_sc = torch.full((B,), t_val, device=device)
                    sc_x1 = self._remove_com(
                        self._reconstruct_x1(v1_cond, x, t_sc)
                    ).detach()
                v2_cond = _call(model, x_pred, t_next, e_t)
                if use_cfg:
                    v2_unc = _call(model, x_pred, t_next, None)
                    v2 = v2_unc + guidance_scale * (v2_cond - v2_unc)
                else:
                    v2 = v2_cond
                x = x + dt * (v1 + v2) * 0.5
            else:
                x = x_pred

            # ── SHAKE bond-length projection (late steps only) ────────────
            if shake_bonds and i >= shake_start and shake_ideal is not None:
                x = self._project_bonds(x, shake_ideal, n_iter=shake_iter)

        return x

    # ── x₁-prediction extensions (for EGNNv2EnergyScoreNetwork) ─────────────

    def training_loss_x1pred_energy(
        self,
        model:          nn.Module,
        x1:             torch.Tensor,   # (B, N, 3)  clean structures
        energy_z:       torch.Tensor,   # (B,)       z-score normalised energy
        physics_weight: float = 0.0,
        physics_fn      = None,
    ) -> torch.Tensor:
        """
        Flow matching loss where the model predicts x̂₁ directly.

        Primary loss : MSE(f_θ(x_t, t, e_z), x₁)

        Physics loss : linear-t weighted — (t · phys(f_θ)).mean()
          Effective mean weight = 0.075 vs 0.017 for legacy t²×0.05 (4.5× more).
          Still suppresses noisy gradients at t≈0 (pure-noise input).
        """
        B  = x1.shape[0]
        t  = torch.rand(B, device=x1.device)
        x0 = self._sample_noise(x1.shape, x1.device)
        x_t = self._remove_com(self._interpolate(x0, x1, t))

        x1_pred = self._remove_com(model(x_t, self._scale_t(t), energy_z=energy_z))
        flow_loss = F.mse_loss(x1_pred, x1)

        if physics_weight > 0.0 and physics_fn is not None:
            per_sample = physics_fn(x1_pred)              # (B,)
            phys_loss  = (t * per_sample).mean()          # linear-t weight (not t²)
            flow_loss  = flow_loss + physics_weight * phys_loss

        return flow_loss

    @torch.no_grad()
    def ddim_sample_x1pred_cfg(
        self,
        model:           nn.Module,
        shape:           tuple,
        device           = 'cuda',
        ddim_steps:      int   = 100,
        tau:             float = 0.5,
        guidance_scale:  float = 2.0,
    ) -> torch.Tensor:
        """
        Heun's ODE for x₁-prediction models with temperature CFG.

        The model predicts x̂₁ = f_θ(x_t, t, e_z).  The ODE velocity is:
            v(x_t, t) = (x̂₁ − x_t) / max(1 − t, 1e-3)

        Verified: rel_err vs exact OT-CFM formula < 1e-2 at t=0.99.
        tau: 0=stable/folded, 1=transient/extended
        guidance_scale: 1.0=conditional only, >1.0=CFG-amplified
        """
        B      = shape[0]
        e_z    = 4.0 * tau - 2.0
        e_t    = torch.full((B,), e_z, device=device)
        eps    = 1e-3
        use_cfg = guidance_scale != 1.0

        x  = self._sample_noise(shape, device)
        dt = 1.0 / ddim_steps

        def _velocity(x_curr: torch.Tensor, t_val: float) -> torch.Tensor:
            t_sc  = torch.full((B,), self._scale_t(t_val), device=device)
            x1_c  = self._remove_com(model(x_curr, t_sc, energy_z=e_t))
            if use_cfg:
                x1_u  = self._remove_com(model(x_curr, t_sc, energy_z=None))
                x1_h  = x1_u + guidance_scale * (x1_c - x1_u)
            else:
                x1_h = x1_c
            return (x1_h - x_curr) / max(1.0 - t_val, eps)

        for i in range(ddim_steps):
            t_val  = i / ddim_steps
            v1     = _velocity(x, t_val)
            x_pred = x + dt * v1

            if i < ddim_steps - 1:
                v2 = _velocity(x_pred, (i + 1) / ddim_steps)
                x  = x + dt * (v1 + v2) * 0.5
            else:
                x  = x_pred

        return x
