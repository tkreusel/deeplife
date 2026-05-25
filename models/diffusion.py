import numpy as np 
import torch 
from torch import nn
from typing import Optional

def cosine_beta_schedule(timestep:int, offset:float = 0.008) -> torch.tensor:
    """ 
    Cosine schedule. Returns tensor of shape (timestep,) with noise at each step t.
    """
    t = torch.linspace(0, timestep, timestep + 1) # [0, 1, 2, ..., timestep]
    f = torch.cos((t / timestep + offset) / (1 + offset) * torch.pi / 2) ** 2
    alpha_t = f / f[0]
    betas = 1 - alpha_t[1:] / alpha_t[:-1]
    return betas.clamp(0, 0.999)

def linear_beta_schedule(T: int, beta_start: float = 1e-4, beta_end: float = 0.02) -> torch.Tensor:
    return torch.linspace(beta_start, beta_end, T)

class GaussianDiffusion(nn.Module):
    def __init__(self, T=500, schedule = 'cosine'):
        super().__init__()
        self.T = T
        if schedule == 'cosine':
            betas = cosine_beta_schedule(T)
        elif schedule == 'linear':
            betas = linear_beta_schedule(T)
        else:
            raise ValueError(f'Unknown schedule: {schedule}')
        
        alphas = 1.0 - betas  # (T,)
        alphas_cumprod = torch.cumprod(alphas, dim=0) # (T,)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])  # (T,)
        
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        self.register_buffer('sqrt_alphas_cumprod', alphas_cumprod.sqrt())
        self.register_buffer('sqrt_one_minus_alphas_cumprod', (1.0 - alphas_cumprod).sqrt())
        
        self.register_buffer('sqrt_recip_alphas', alphas.rsqrt())
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)
        self.register_buffer('posterior_log_variance_clipped', posterior_variance.clamp(min=1e-20).log())
    
        self.register_buffer('sqrt_recip_alphas_cumprod', (1.0 / alphas_cumprod).sqrt())
        self.register_buffer('sqrt_recipm1_alphas_cumprod', ((1.0 / alphas_cumprod) - 1).sqrt())
        
    def _extract(self, arr: torch.Tensor, t: torch.Tensor, shape: tuple) -> torch.Tensor:
        """
        Index into a (T,) schedule array using per-sample timesteps t,
        then reshape for broadcasting over (B, N, 3).

        arr:   (T,)   — one value per timestep
        t:     (B,)   — one timestep index per sample in the batch
        shape: tuple  — the shape of the tensor you want to broadcast onto,
                        typically (B, N, 3)

        returns: (B, 1, 1)  — broadcasts cleanly over (B, N, 3)
        """
        B = t.shape[0]
        out = arr[t]                              # (B,)
        n_dims = len(shape) - 1                   # number of dims to pad
        return out.view(B, *([1] * n_dims))        # (B, 1, 1)
    
    def q_sample(
        self,
        x0:    torch.Tensor,                # (B, N, 3)  clean structures
        t:     torch.Tensor,                # (B,)       integer timesteps
        noise: Optional[torch.Tensor] = None
    ):
        """
        Corrupt clean structures x0 to noise level t in one step.
        Returns the noisy structure x_t and the noise that was added.
        """
        if noise is None:
            noise = torch.randn_like(x0)    # (B, N, 3)

        sqrt_a  = self._extract(self.sqrt_alphas_cumprod,           t, x0.shape)
        sqrt_1a = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x0.shape)

        x_t = sqrt_a * x0 + sqrt_1a * noise

        return x_t, noise
    
    def _predict_x0_from_noise(
        self,
        x_t:        torch.Tensor,   # (B, N, 3)
        t:          torch.Tensor,   # (B,)
        noise_pred: torch.Tensor,   # (B, N, 3)
    ) -> torch.Tensor:
        """
        Given noisy x_t and a predicted noise, recover the predicted x_0.
        Rearranging:  x_t = √ᾱ_t · x_0 + √(1-ᾱ_t) · ε
        Gives:        x_0 = (x_t - √(1-ᾱ_t) · ε) / √ᾱ_t
                          = (1/√ᾱ_t) · x_t  -  √(1/ᾱ_t - 1) · ε
        """
        s1 = self._extract(self.sqrt_recip_alphas_cumprod,  t, x_t.shape)
        s2 = self._extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        return s1 * x_t - s2 * noise_pred
    
    def training_loss(
        self,
        model:          nn.Module,
        x0:             torch.Tensor,       # (B, N, 3)  clean structures
        physics_weight: float = 0.0,
        physics_fn      = None,             # callable(x_pred) → scalar
    ) -> torch.Tensor:
        """
        Compute the diffusion training loss for one batch.
        """
        B = x0.shape[0]
        device = x0.device

        # 1. sample a random timestep for each structure in the batch
        t = torch.randint(0, self.T, (B,), device=device)

        # 2. corrupt x0 → x_t
        x_t, noise = self.q_sample(x0, t)

        # 3. ask the network to predict the noise
        noise_pred = model(x_t, t)          # (B, N, 3)

        # 4. simple MSE between true and predicted noise
        loss = (noise - noise_pred).pow(2).mean()

        # 5. optional physics regularisation
        if physics_weight > 0.0 and physics_fn is not None:
            x0_pred = self._predict_x0_from_noise(x_t, t, noise_pred)
            loss = loss + physics_weight * physics_fn(x0_pred)

        return loss
    
    @torch.no_grad()
    def p_sample(
        self,
        model:          nn.Module,
        x_t:            torch.Tensor,       # (B, N, 3)
        t:              int,                # scalar integer
        t_tensor:       torch.Tensor,       # (B,)  same t for all items
        guidance_fn     = None,
        guidance_scale: float = 1.0,
    ) -> torch.Tensor:
        """
        One reverse diffusion step: x_t → x_{t-1}.
        """
        # predict noise
        noise_pred = model(x_t, t_tensor)   # (B, N, 3)

        # optional physics guidance: nudge noise prediction toward
        # lower structural energy
        if guidance_fn is not None:
            sqrt_1a = self._extract(
                self.sqrt_one_minus_alphas_cumprod, t_tensor, x_t.shape
            )
            force = guidance_fn(x_t)        # (B, N, 3)
            noise_pred = noise_pred - guidance_scale * sqrt_1a * force

        # DDPM reverse mean:
        # μ_t = (1/√α_t) · (x_t  -  β_t/√(1-ᾱ_t) · ε̂)
        betas_t   = self._extract(self.betas,                          t_tensor, x_t.shape)
        sqrt_1a_t = self._extract(self.sqrt_one_minus_alphas_cumprod,  t_tensor, x_t.shape)
        recip_a_t = self._extract(self.sqrt_recip_alphas,              t_tensor, x_t.shape)

        mean = recip_a_t * (x_t - betas_t / sqrt_1a_t * noise_pred)

        # at t=0 return the mean directly — no noise on the final step
        if t == 0:
            return mean

        # otherwise add noise scaled by the posterior variance
        log_var = self._extract(
            self.posterior_log_variance_clipped, t_tensor, x_t.shape
        )
        noise = torch.randn_like(x_t)
        return mean + (0.5 * log_var).exp() * noise
    
    @torch.no_grad()
    def sample(
        self,
        model:          nn.Module,
        shape:          tuple,              # (B, N, 3)
        device:         str = 'cuda',
        guidance_fn     = None,
        guidance_scale: float = 1.0,
    ) -> torch.Tensor:
        """
        Generate new structures by running the full reverse process.
        Starts from pure Gaussian noise, denoises for T steps.
        """
        x = torch.randn(*shape, device=device)   # x_T ~ N(0, I)

        for t in reversed(range(self.T)):         # T-1, T-2, ..., 1, 0
            t_tensor = torch.full(
                (shape[0],), t, device=device, dtype=torch.long
            )
            x = self.p_sample(model, x, t, t_tensor, guidance_fn, guidance_scale)

        return x    # (B, N, 3) — generated structures
    
    @torch.no_grad()
    def ddim_sample(
        self,
        model:          nn.Module,
        shape:          tuple,
        device:         str = 'cuda',
        ddim_steps:     int = 50,
        eta:            float = 0.0,
        guidance_fn     = None,
        guidance_scale: float = 1.0,
    ) -> torch.Tensor:
        """
        Faster sampling using DDIM — skips most timesteps.
        eta=0.0 is fully deterministic. eta=1.0 recovers DDPM.
        """
        # pick evenly spaced subset of timesteps
        step_size = self.T // ddim_steps
        timesteps = list(range(0, self.T, step_size))[::-1]  # descending

        x = torch.randn(*shape, device=device)

        for i, t in enumerate(timesteps):
            t_tensor   = torch.full((shape[0],), t, device=device, dtype=torch.long)
            t_prev     = timesteps[i + 1] if i + 1 < len(timesteps) else 0

            alpha_t    = self.alphas_cumprod[t]
            alpha_prev = self.alphas_cumprod[t_prev]

            noise_pred = model(x, t_tensor)

            # optional guidance
            if guidance_fn is not None:
                sqrt_1a = (1 - alpha_t).sqrt()
                force   = guidance_fn(x)
                noise_pred = noise_pred - guidance_scale * sqrt_1a * force

            # predict x_0
            x0_pred = (x - (1 - alpha_t).sqrt() * noise_pred) / alpha_t.sqrt()
            x0_pred = x0_pred.clamp(-5, 5)      # prevent outliers

            # DDIM step
            sigma   = eta * ((1 - alpha_prev) / (1 - alpha_t)
                             * (1 - alpha_t / alpha_prev)).sqrt()
            dir_xt  = (1 - alpha_prev - sigma ** 2).sqrt() * noise_pred
            noise   = torch.randn_like(x) if t_prev > 0 else 0

            x = alpha_prev.sqrt() * x0_pred + dir_xt + sigma * noise

        return x