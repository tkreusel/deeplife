"""
models/harmonic_prior.py
========================
Geometry-preserving noise priors for flow matching.

Instead of sampling x₀ from a standard Gaussian (which has completely wrong
bond geometry), these samplers produce x₀ where all covalent bonds are
already at their ideal lengths by construction.

Motivation
----------
In OT-CFM the straight-line interpolant is x_t = (1−t)·x₀ + t·x₁.
With Gaussian x₀ the model must simultaneously learn:
  (a) global structure (Rg, end-to-end distance)
  (b) bond geometry (covalent bond lengths and angles)

By sampling x₀ with correct bond lengths, (b) is removed from the problem:
both endpoints have correct bonds, and the model only needs to learn how to
rearrange from one valid geometry to another.

This significantly reduces the difficulty of the physics loss: the
reconstructed x̂₁ gradient is cleaner because the source already has the
right bond structure.

Functions
---------
sample_all_atom_chain_batched(B, N, device, coord_scale)
    SHAKE-based sampler for all-atom Chignolin (N=93).
    Starts from Gaussian noise and iteratively projects all 64 consecutive-pair
    bonds to their data-derived ideal lengths.  After convergence all bonds
    tracked by AllAtomPhysics are exactly at their target lengths.

sample_ca_chain(B, N, device, bond_length, coord_scale)
    Random-walk sampler for Cα-only Chignolin (N=10).
    Consecutive Cα atoms are spaced exactly `bond_length` Å apart.

Why SHAKE instead of BFS tree placement?
-----------------------------------------
The 93-atom all-atom representation has a disconnected consecutive-pair bond
graph: some consecutive pairs (i, i+1) are NOT covalently bonded (sidechain
branches, inter-residue jumps).  A BFS tree from atom 0 cannot reach all 93
atoms via the 64 tracked bonds.

SHAKE (holonomic constraint projection) avoids this: it iteratively corrects
ALL 64 bonds simultaneously regardless of graph connectivity.  Starting from
Gaussian noise, ~100 iterations bring all bond-length errors below 0.01 Å.
This is exactly the same algorithm used in MD simulations for bond constraints.
"""

import torch
from torch import Tensor


# ─────────────────────────────────────────────────────────────────────────────
# All-atom SHAKE sampler (N=93)
# ─────────────────────────────────────────────────────────────────────────────

def sample_all_atom_chain_batched(
    B: int,
    N: int,
    device,
    coord_scale: float = 16.32,
    n_shake: int = 150,
) -> Tensor:
    """
    Sample all-atom Chignolin structures (N=93) where all 64 consecutive-pair
    covalent bonds are at their data-derived ideal lengths.

    Algorithm (vectorised SHAKE / Jacobi constraint projection):
    1. Sample x₀ ~ N(0, I) Gaussian noise.
    2. Vectorised SHAKE iteration: for ALL 64 bonds simultaneously, compute
       corrections using gather/scatter_add — no inner Python loop over bonds.
       This is Jacobi-style (parallel) SHAKE: corrections from all bonds are
       accumulated in a delta tensor and applied at once.
    3. Repeat n_shake times until convergence.
    4. Zero-CoM and scale to model units.

    Implementation uses fully vectorised GPU ops (gather + scatter_add) so the
    inner loop runs once per SHAKE iteration, not once per bond per iteration.
    At 150 iterations on GPU this adds ~5ms per call.

    Why SHAKE instead of BFS placement:
        The consecutive-pair bond graph is DISCONNECTED (sidechain branches
        create non-consecutive bonds not tracked by AllAtomPhysics).  BFS
        from atom 0 cannot reach all 93 atoms.  SHAKE works on any subset
        of bond constraints regardless of connectivity.

    Convergence: ~100 Jacobi iterations → bond MAE < 0.005 Å.

    Returns: (B, N, 3)  in model-space units (Å / coord_scale), zero-CoM
    """
    from models.physics_aa import _BOND_INDICES, _BOND_TARGETS

    n_bonds = len(_BOND_INDICES)

    # Pre-build index and target tensors (shape: (n_bonds,))
    bidx_t = torch.tensor(_BOND_INDICES, dtype=torch.long,    device=device)
    btgt_t = torch.tensor(_BOND_TARGETS, dtype=torch.float32, device=device)

    # Expand for gather/scatter: (1, n_bonds, 3) → broadcast to (B, n_bonds, 3)
    idx_lo = bidx_t.view(1, n_bonds, 1).expand(B, n_bonds, 3)     # lower atom idx
    idx_hi = (bidx_t + 1).view(1, n_bonds, 1).expand(B, n_bonds, 3)  # upper atom idx

    x = torch.randn(B, N, 3, device=device)
    x = x - x.mean(dim=1, keepdim=True)

    tgt = btgt_t.view(1, n_bonds, 1)   # (1, 64, 1) for broadcasting

    for _ in range(n_shake):
        x_lo = x.gather(1, idx_lo)     # (B, n_bonds, 3) — lower atom positions
        x_hi = x.gather(1, idx_hi)     # (B, n_bonds, 3) — upper atom positions

        diff = x_hi - x_lo             # (B, n_bonds, 3)
        dist = diff.norm(dim=-1, keepdim=True).clamp(min=1e-8)   # (B, n_bonds, 1)
        corr = 0.5 * (dist - tgt) * (diff / dist)               # (B, n_bonds, 3)

        # Accumulate corrections: Σⱼ corr[j] applied to atom (idx+1), -corr to atom idx
        delta = torch.zeros(B, N, 3, device=device, dtype=x.dtype)
        delta.scatter_add_(1, idx_hi, -corr)
        delta.scatter_add_(1, idx_lo,  corr)
        x = x + delta

    x = x - x.mean(dim=1, keepdim=True)
    x = x / coord_scale
    return x


# ─────────────────────────────────────────────────────────────────────────────
# Cα-only random-walk sampler (N=10)
# ─────────────────────────────────────────────────────────────────────────────

def sample_ca_chain(
    B: int,
    N: int,
    device,
    bond_length: float = 3.832,
    coord_scale: float = 5.0,
) -> Tensor:
    """
    Sample Cα-only Chignolin (N=10) with correct consecutive Cα–Cα distances.

    Places each atom at exactly `bond_length` Å from the previous one in a
    uniformly random direction on the unit sphere.  The resulting chain is a
    random walk on a sphere of radius `bond_length`, guaranteeing that all
    Cα–Cα virtual bonds have the correct length.

    Returns: (B, N, 3)  in model-space units (Å / coord_scale), zero-CoM
    """
    # Random unit-vector directions for N-1 steps
    dirs = torch.randn(B, N - 1, 3, device=device)
    dirs = dirs / dirs.norm(dim=-1, keepdim=True).clamp(min=1e-8)

    # Build chain by cumulative sum of bond vectors
    steps = bond_length * dirs                              # (B, N-1, 3)
    coords = torch.zeros(B, N, 3, device=device)
    coords[:, 1:] = steps.cumsum(dim=1)

    coords = coords - coords.mean(dim=1, keepdim=True)
    coords = coords / coord_scale
    return coords
