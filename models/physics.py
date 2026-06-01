"""
models/physics.py
==================
Physics-based structural constraints for Chignolin Cα coordinate generation.

All constraint functions operate on (B, N, 3) tensors in **Ångström space**.
The ChignolinPhysics class handles the coord_scale conversion so it can be
passed directly as `physics_fn` to training_loss().

Reference geometry (from training set, 79 632 structures):
    Cα–Cα bond length : mean = 3.832 Å,  std = 0.062 Å  (nearly rigid)
    Cα–Cα–Cα angle   : mean = 71.4°,    std = 15.5°    (moderate variance)
    Non-bonded min    : 3.73 Å  (all non-bonded pairs in real data > 3.5 Å)

Design
------
At training time, ZeroCoMFlowMatching reconstructs x₁_pred from the velocity
prediction and applies physics losses weighted by t² (reliability weight):

    L_physics(θ) = E_{t,x₀,x₁} [ t² · Σᵢ wᵢ · Lᵢ(x₁_pred) ]

The t² factor suppresses physics loss at t≈0 (x₁_pred is noise-dominated)
and gives it full weight at t≈1 (x₁_pred converges to real x₁).

Usage
-----
    from models.physics import ChignolinPhysics

    physics = ChignolinPhysics(
        bond_weight=1.0, clash_weight=0.1, angle_weight=0.5, coord_scale=5.0
    )

    # In training_loss:
    loss = diffusion.training_loss(
        model, x0, physics_weight=0.05, physics_fn=physics
    )

    # During validation, get a per-constraint breakdown for logging:
    breakdown = physics.breakdown(x_normalised)
    # → {'phys_bond': 0.12, 'phys_clash': 0.00, 'phys_angle': 0.08, 'phys_total': 0.16}
"""

import torch
import torch.nn.functional as F
from torch import Tensor

# ── Physical constants ────────────────────────────────────────────────────────

IDEAL_BOND_LENGTH = 3.832   # Å  — data mean (real Cα–Cα bond)
IDEAL_BOND_COS    = 0.320   # cos(71.4°) — data mean for consecutive bond vector angle
CLASH_CUTOFF      = 3.5     # Å  — minimum allowed non-bonded Cα distance


# ─────────────────────────────────────────────────────────────────────────────
# Individual constraint functions
# Each returns a (B,) per-structure loss tensor.
# Input x is always (B, N, 3) in Ångströms.
# ─────────────────────────────────────────────────────────────────────────────

def bond_length_loss(x: Tensor) -> Tensor:
    """
    MSE between consecutive Cα–Cα bond lengths and the ideal 3.832 Å.

    Bond lengths in real Chignolin have std = 0.062 Å, so this is a very tight
    constraint. Any deviation > ~0.1 Å is physically implausible.

    x       : (B, N, 3)  Ångströms
    returns : (B,)       per-structure MSE
    """
    diffs   = x[:, 1:] - x[:, :-1]                 # (B, N-1, 3)
    lengths = diffs.norm(dim=-1)                     # (B, N-1)
    return (lengths - IDEAL_BOND_LENGTH).pow(2).mean(dim=-1)


def clash_loss(x: Tensor, cutoff: float = CLASH_CUTOFF) -> Tensor:
    """
    Soft quadratic repulsion for non-bonded Cα pairs closer than `cutoff` Å.

    Only pairs with sequence separation |i−j| ≥ 2 are considered (bonded
    neighbours are expected to be at 3.8 Å). Real structures have no
    non-bonded pairs closer than 3.73 Å, so cutoff = 3.5 Å is safe.

    x       : (B, N, 3)  Ångströms
    returns : (B,)       per-structure mean repulsion loss
    """
    B, N, _ = x.shape

    # Pairwise squared distances — memory-efficient for N=10
    diff = x.unsqueeze(2) - x.unsqueeze(1)          # (B, N, N, 3)
    dist = diff.norm(dim=-1)                          # (B, N, N)

    # Sequence-separation mask: keep |i-j| >= 2 only
    idx = torch.arange(N, device=x.device)
    sep = (idx.unsqueeze(1) - idx.unsqueeze(0)).abs()   # (N, N)
    mask = (sep >= 2).float()                            # (B,)-broadcastable

    # Quadratic penalty for pairs below cutoff; zero otherwise
    repulsion = F.relu(cutoff - dist).pow(2)             # (B, N, N)

    n_pairs = mask.sum().clamp(min=1.0)
    return (repulsion * mask).sum(dim=(-2, -1)) / n_pairs


def bond_angle_loss(x: Tensor) -> Tensor:
    """
    Huber-style penalty on Cα–Cα–Cα virtual bond angles.

    The angle is measured as the angle between consecutive bond vectors
    (b_i = x_{i+1} - x_i), not the supplement. In real Chignolin:
        cos(angle) mean = 0.320 (≈ 71.4°),  std ≈ 0.26

    A Smooth-L1 (Huber) loss with δ = 0.5 tolerates the natural backbone
    flexibility (±2σ ≈ ±0.52) without over-constraining turns and bends.

    x       : (B, N, 3)  Ångströms
    returns : (B,)       per-structure mean angle loss
    """
    b1 = x[:, 1:-1] - x[:, :-2]                    # (B, N-2, 3)
    b2 = x[:, 2:]   - x[:, 1:-1]                   # (B, N-2, 3)

    cos_theta = F.cosine_similarity(b1, b2, dim=-1) # (B, N-2)
    deviation = cos_theta - IDEAL_BOND_COS           # (B, N-2)

    # Huber: quadratic within δ=0.5, linear outside (robust to outliers)
    return F.huber_loss(cos_theta,
                        torch.full_like(cos_theta, IDEAL_BOND_COS),
                        delta=0.5,
                        reduction='none').mean(dim=-1)


# ─────────────────────────────────────────────────────────────────────────────
# Combined class
# ─────────────────────────────────────────────────────────────────────────────

class ChignolinPhysics:
    """
    Configurable physics loss for Chignolin Cα structures.

    Accepts coordinates in **normalised (model) space** and rescales to Å
    internally using `coord_scale`. This makes it drop-in compatible with
    the `physics_fn` argument of training_loss(), which receives x₁_pred
    in model space.

    Parameters
    ----------
    bond_weight   : weight for Cα–Cα bond-length MSE  (highest priority)
    clash_weight  : weight for non-bonded clash penalty
    angle_weight  : weight for virtual Cα–Cα–Cα bond-angle Huber loss
    coord_scale   : inverse of the normalisation factor used in data/dataset.py
                    (must match data.coord_scale in config)

    Returning per-sample losses
    ---------------------------
    __call__ returns a (B,) tensor so that training_loss() can weight each
    sample by its t² value before averaging.

    breakdown() returns a plain dict of scalar floats for logging.
    """

    def __init__(
        self,
        bond_weight:  float = 1.0,
        clash_weight: float = 0.1,
        angle_weight: float = 0.5,
        coord_scale:  float = 5.0,
    ):
        self.bond_weight  = bond_weight
        self.clash_weight = clash_weight
        self.angle_weight = angle_weight
        self.coord_scale  = coord_scale

    def __call__(self, x_norm: Tensor) -> Tensor:
        """
        x_norm  : (B, N, 3)  in normalised (model) space
        returns : (B,)       weighted total per-structure physics loss
        """
        x    = x_norm * self.coord_scale              # → Ångströms
        loss = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)

        if self.bond_weight > 0:
            loss = loss + self.bond_weight  * bond_length_loss(x)
        if self.clash_weight > 0:
            loss = loss + self.clash_weight * clash_loss(x)
        if self.angle_weight > 0:
            loss = loss + self.angle_weight * bond_angle_loss(x)

        return loss   # (B,)

    def breakdown(self, x_norm: Tensor) -> dict:
        """
        Per-constraint losses as Python floats — use for JSONL logging.

        x_norm  : (B, N, 3) in normalised space
        returns : dict with keys phys_bond, phys_clash, phys_angle, phys_total
        """
        x = (x_norm * self.coord_scale).detach()
        result: dict = {}

        b = bond_length_loss(x).mean().item() if self.bond_weight  > 0 else 0.0
        c = clash_loss(x).mean().item()       if self.clash_weight > 0 else 0.0
        a = bond_angle_loss(x).mean().item()  if self.angle_weight > 0 else 0.0

        result['phys_bond']  = b
        result['phys_clash'] = c
        result['phys_angle'] = a
        result['phys_total'] = (self.bond_weight  * b
                                + self.clash_weight * c
                                + self.angle_weight * a)
        return result

    def __repr__(self) -> str:
        return (f"ChignolinPhysics(bond×{self.bond_weight}, "
                f"clash×{self.clash_weight}, angle×{self.angle_weight}, "
                f"coord_scale={self.coord_scale})")
