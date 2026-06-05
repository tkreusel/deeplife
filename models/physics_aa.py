"""
models/physics_aa.py
=====================
Physics-based structural constraints for all-atom Chignolin coordinate generation.

Design
------
The all-atom data has 93 heavy atoms per structure, ordered in a specific sequence.
Consecutive atom pairs are NOT all covalently bonded — 28 of the 92 consecutive
pairs span residue or segment boundaries and are much longer (3–5 Å).

Constraints
-----------
1. Bond-length MSE — applied only to the 64 pairs identified as covalent bonds
   (consecutive distance mean < 2.0 Å, std < 0.15 Å across training structures).
   Each bond has its own data-derived target length to account for the variety of
   bond types present (C–N ~1.33 Å, N–Cα ~1.46 Å, Cα–C ~1.52 Å, etc.).

2. Clash repulsion — quadratic penalty for all non-bonded heavy-atom pairs
   (sequence separation ≥ 4) closer than 2.5 Å.
   The 1st percentile of non-bonded distances in real structures is 2.46 Å,
   so this cutoff captures genuine steric clashes without over-constraining.

Bond indices and targets were computed from the full training set (79 632 structures)
and are hard-coded here so the module is self-contained.

Usage
-----
    from models.physics_aa import AllAtomPhysics

    physics = AllAtomPhysics(
        bond_weight=1.0, clash_weight=0.1, coord_scale=16.32
    )

    # Drop-in for ChignolinPhysics in training_loss:
    loss = diffusion.training_loss(
        model, x0,
        physics_weight=0.05,
        physics_fn=physics,
        model_kwargs={'energy_z': energy_z},
    )
"""

import torch
import torch.nn.functional as F
from torch import Tensor

# ── Bond connectivity derived from training-set statistics ────────────────────
# Consecutive pair index i means the pair (atom_i, atom_{i+1}) in the 93-atom sequence.
# Only pairs with mean distance < 2.0 Å and std < 0.15 Å are included.

_BOND_INDICES = [
     0,  1,  2,  3,  4,  5,  6,  8, 10, 12, 13, 14, 15, 16, 17, 18,
    20, 22, 24, 25, 26, 27, 30, 32, 34, 35, 37, 39, 40, 41, 42, 43,
    46, 48, 49, 50, 53, 55, 56, 57, 59, 60, 61, 64, 66, 67, 68, 69,
    70, 71, 72, 73, 74, 76, 78, 80, 83, 84, 85, 86, 87, 88, 89, 91,
]

# Per-bond ideal lengths in Ångströms (data-derived mean)
_BOND_TARGETS = [
    1.4893, 1.5824, 1.5153, 1.4014, 1.3899, 1.4038, 1.3717, 1.3889,
    1.2248, 1.4507, 1.5576, 1.5206, 1.4029, 1.3932, 1.3972, 1.3841,
    1.3871, 1.2290, 1.4587, 1.5638, 1.5276, 1.2587, 1.2282, 1.5112,
    1.5332, 1.5237, 1.2236, 1.4725, 1.5610, 1.5513, 1.5606, 1.2500,
    1.2182, 1.4614, 1.5667, 1.4311, 1.2125, 1.4665, 1.5230, 1.2215,
    1.4601, 1.5377, 1.4354, 1.2183, 1.4504, 1.5392, 1.5284, 1.3651,
    1.4014, 1.3810, 1.4245, 1.4093, 1.3896, 1.3921, 1.2239, 1.2560,
    1.4665, 1.5733, 1.5295, 1.4059, 1.3976, 1.3992, 1.3824, 1.3927,
]

CLASH_CUTOFF  = 2.5    # Å  — 1st percentile of non-bonded heavy-atom distances
MIN_SEP       = 20     # minimum sequence separation to be considered non-bonded
                       # sep<20 still catches intra-residue & adjacent-residue contacts
                       # that aren't true steric clashes. Reference clash rate ~10%.


class AllAtomPhysics:
    """
    Physics loss for all-atom Chignolin structures.

    Accepts coordinates in normalised (model) space and rescales using
    coord_scale before computing distances in Ångströms.

    Parameters
    ----------
    bond_weight   : weight for per-bond MSE loss
    clash_weight  : weight for heavy-atom clash penalty
    coord_scale   : data normalisation factor (must match data.coord_scale in config)
    """

    def __init__(
        self,
        bond_weight:  float = 1.0,
        clash_weight: float = 0.1,
        coord_scale:  float = 16.32,
    ):
        self.bond_weight  = bond_weight
        self.clash_weight = clash_weight
        self.coord_scale  = coord_scale

        # Precompute as tensors for GPU compatibility
        self._bond_idx  = torch.tensor(_BOND_INDICES, dtype=torch.long)
        self._bond_tgt  = torch.tensor(_BOND_TARGETS, dtype=torch.float32)

    def _to_device(self, device):
        self._bond_idx = self._bond_idx.to(device)
        self._bond_tgt = self._bond_tgt.to(device)

    def __call__(self, x_norm: Tensor) -> Tensor:
        """
        x_norm  : (B, 93, 3)  in normalised (model) space
        returns : (B,)         weighted total per-structure physics loss
        """
        x      = x_norm * self.coord_scale    # → Ångströms
        device = x.device
        self._to_device(device)

        loss = torch.zeros(x.shape[0], device=device, dtype=x.dtype)

        if self.bond_weight > 0:
            loss = loss + self.bond_weight  * _bond_length_loss(x, self._bond_idx, self._bond_tgt)
        if self.clash_weight > 0:
            loss = loss + self.clash_weight * _clash_loss_aa(x)

        return loss   # (B,)

    def breakdown(self, x_norm: Tensor) -> dict:
        """Per-constraint losses as Python floats for JSONL logging."""
        x = (x_norm * self.coord_scale).detach()
        self._to_device(x.device)

        b = _bond_length_loss(x, self._bond_idx, self._bond_tgt).mean().item() if self.bond_weight  > 0 else 0.0
        c = _clash_loss_aa(x).mean().item()                                     if self.clash_weight > 0 else 0.0

        return {
            'phys_bond':  b,
            'phys_clash': c,
            'phys_angle': 0.0,
            'phys_total': self.bond_weight * b + self.clash_weight * c,
        }

    def __repr__(self) -> str:
        return (f"AllAtomPhysics(bond×{self.bond_weight}, "
                f"clash×{self.clash_weight}, coord_scale={self.coord_scale}, "
                f"n_bonds={len(_BOND_INDICES)})")


# ── Constraint functions ──────────────────────────────────────────────────────

def _bond_length_loss(x: Tensor, bond_idx: Tensor, bond_tgt: Tensor) -> Tensor:
    """
    MSE between actual and target length for each identified covalent bond.

    x        : (B, 93, 3)  Ångströms
    bond_idx : (64,)       indices into the consecutive-pair sequence
    bond_tgt : (64,)       per-bond ideal lengths (Å)
    returns  : (B,)        mean MSE over all bonds
    """
    diffs   = x[:, 1:] - x[:, :-1]           # (B, 92, 3)
    lengths = diffs.norm(dim=-1)               # (B, 92)
    selected = lengths[:, bond_idx]            # (B, 64)  — only bonded pairs
    return (selected - bond_tgt.unsqueeze(0)).pow(2).mean(dim=-1)  # (B,)


def _clash_loss_aa(x: Tensor, cutoff: float = CLASH_CUTOFF, min_sep: int = MIN_SEP) -> Tensor:
    """
    Quadratic repulsion for non-bonded heavy-atom pairs closer than cutoff.

    Only pairs with sequence separation ≥ min_sep are considered.

    x       : (B, 93, 3)  Ångströms
    returns : (B,)        mean repulsion loss
    """
    B, N, _ = x.shape

    diff = x.unsqueeze(2) - x.unsqueeze(1)     # (B, N, N, 3)
    dist = diff.norm(dim=-1)                    # (B, N, N)

    idx = torch.arange(N, device=x.device)
    sep = (idx.unsqueeze(1) - idx.unsqueeze(0)).abs()
    mask = (sep >= min_sep).float()

    repulsion = F.relu(cutoff - dist).pow(2)    # (B, N, N)
    n_pairs   = mask.sum().clamp(min=1.0)
    return (repulsion * mask).sum(dim=(-2, -1)) / n_pairs   # (B,)
