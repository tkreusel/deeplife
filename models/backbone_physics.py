"""
models/backbone_physics.py
===========================
Physics loss for backbone N–Cα–C coordinates (30 atoms, 10 residues).

Backbone bond inventory (29 covalent bonds):
    N–Cα   ×10  ideal = 1.460 Å  (std ≈ 0.005 Å — very rigid)
    Cα–C   ×10  ideal = 1.525 Å  (std ≈ 0.005 Å)
    C–N    × 9  ideal = 1.329 Å  (std ≈ 0.004 Å — peptide resonance)

Atom layout in the (B, 30, 3) tensor:
    [N0, CA0, C0,  N1, CA1, C1,  …,  N9, CA9, C9]
     0   1    2    3   4    5         27  28   29

Bond indices (consecutive triplets):
    N_i  → CA_i  : positions 3i,   3i+1  (i=0..9)
    CA_i → C_i   : positions 3i+1, 3i+2
    C_i  → N_{i+1}: positions 3i+2, 3i+3  (i=0..8)

Clash threshold:
    Minimum non-bonded backbone heavy-atom distance in real structures ≈ 2.4 Å.
    We use 2.0 Å as a conservative soft repulsion cutoff (backbone atoms can
    legitimately approach ~2.4 Å in close turns; 2.0 Å catches only clashes).
"""

import torch
import torch.nn.functional as F
from torch import Tensor

# ── Physical constants ────────────────────────────────────────────────────────

N_CA_IDEAL  = 1.460   # Å
CA_C_IDEAL  = 1.525   # Å
C_N_IDEAL   = 1.329   # Å
CLASH_CUTOFF = 2.0    # Å   soft repulsion cutoff for non-bonded backbone pairs


# ── Bond index arrays (built once, moved to device on first call) ─────────────

def _bond_indices():
    """
    Returns three lists of (src, dst, ideal_length) for the 29 backbone bonds.

    N–CA bonds  (×10): (0→1), (3→4), (6→7), …, (27→28)
    CA–C bonds  (×10): (1→2), (4→5), (7→8), …, (28→29)
    C–N bonds   (×9) : (2→3), (5→6), (8→9), …, (26→27)
    """
    src, dst, ideal = [], [], []
    for i in range(10):
        base = 3 * i
        src.append(base);     dst.append(base + 1); ideal.append(N_CA_IDEAL)   # N→CA
        src.append(base + 1); dst.append(base + 2); ideal.append(CA_C_IDEAL)   # CA→C
    for i in range(9):
        base = 3 * i
        src.append(base + 2); dst.append(base + 3); ideal.append(C_N_IDEAL)    # C→N
    return src, dst, ideal

_SRC, _DST, _IDEAL = _bond_indices()


# ── Individual loss functions ─────────────────────────────────────────────────

def backbone_bond_loss(x: Tensor) -> Tensor:
    """
    MSE between all 29 backbone bond lengths and their ideal values.

    x : (B, 30, 3)  backbone coordinates in Å
    returns : (B,)  per-structure mean bond-length MSE
    """
    device = x.device
    src   = torch.tensor(_SRC,   device=device)
    dst   = torch.tensor(_DST,   device=device)
    ideal = torch.tensor(_IDEAL, device=device, dtype=x.dtype)

    lengths = (x[:, dst] - x[:, src]).norm(dim=-1)   # (B, 29)
    return (lengths - ideal).pow(2).mean(dim=-1)       # (B,)


def backbone_clash_loss(x: Tensor, cutoff: float = CLASH_CUTOFF) -> Tensor:
    """
    Soft quadratic repulsion for non-bonded backbone atom pairs.

    Pairs excluded: |i−j| < 3 (bonded or 1,3-related within the chain).
    Real backbone structures have no non-bonded pair below ~2.4 Å; the
    2.0 Å cutoff only fires on genuine clashes.

    x : (B, 30, 3)  backbone coordinates in Å
    returns : (B,)  per-structure mean repulsion
    """
    B, N, _ = x.shape
    diff = x.unsqueeze(2) - x.unsqueeze(1)   # (B, N, N, 3)
    dist = diff.norm(dim=-1)                   # (B, N, N)

    idx = torch.arange(N, device=x.device)
    sep = (idx.unsqueeze(1) - idx.unsqueeze(0)).abs()
    mask = (sep >= 3).float()

    repulsion = F.relu(cutoff - dist).pow(2)   # (B, N, N)
    n_pairs = mask.sum().clamp(min=1.0)
    return (repulsion * mask).sum(dim=(-2, -1)) / n_pairs   # (B,)


def backbone_angle_loss(x: Tensor) -> Tensor:
    """
    Huber penalty on backbone bond angles.

    Three angle types with distinct ideals (all nearly fixed in proteins):
        N–Cα–C  ≈ 111.2°  cos = −0.364
        Cα–C–N  ≈ 116.2°  cos = −0.440
        C–N–Cα  ≈ 121.7°  cos = −0.526

    Angles are formed by three consecutive backbone atoms. Each triplet
    (i, i+1, i+2) in the 30-atom chain corresponds to one of the three types.

    x : (B, 30, 3)  backbone in Å
    returns : (B,)  mean Huber angle loss
    """
    # Ideal cosines (supplementary of the angle as measured between bond vectors)
    # cos(angle at central atom) where angle = arccos(-b1·b2 / |b1||b2|)
    # Equivalently cos_theta = dot(b_back, b_fwd) where b_back = prev-central,
    # b_fwd = next-central — same sign convention as internal_coords.py
    IDEALS = {0: -0.364, 1: -0.440, 2: -0.526}   # N-CA-C, CA-C-N, C-N-CA

    b_back = x[:, :28] - x[:, 1:29]    # (B, 28, 3)  toward previous atom
    b_fwd  = x[:, 2:]  - x[:, 1:29]   # (B, 28, 3)  toward next atom
    cos_theta = F.cosine_similarity(b_back, b_fwd, dim=-1)   # (B, 28)

    # Build per-position ideal cosine tensor
    # position k in [0..27]: angle type = k % 3
    ideal_cos = torch.tensor(
        [IDEALS[k % 3] for k in range(28)],
        dtype=x.dtype, device=x.device
    )   # (28,)

    return F.huber_loss(
        cos_theta,
        ideal_cos.unsqueeze(0).expand_as(cos_theta),
        delta=0.3,
        reduction='none',
    ).mean(dim=-1)   # (B,)


# ── Combined class ─────────────────────────────────────────────────────────────

class BackbonePhysics:
    """
    Physics constraint loss for backbone N–Cα–C structures.

    Drop-in replacement for ChignolinPhysics — accepts normalised model-space
    coordinates and converts to Å internally via coord_scale.

    Parameters
    ----------
    bond_weight   : weight for the 29-bond MSE (primary constraint)
    clash_weight  : weight for non-bonded repulsion
    angle_weight  : weight for bond-angle Huber loss
    coord_scale   : inverse normalisation factor (must match data.coord_scale)
    """

    def __init__(
        self,
        bond_weight:  float = 2.0,
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
        x_norm  : (B, 30, 3)  normalised backbone coordinates
        returns : (B,)        weighted total physics loss per structure
        """
        x    = x_norm * self.coord_scale
        loss = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
        if self.bond_weight  > 0: loss = loss + self.bond_weight  * backbone_bond_loss(x)
        if self.clash_weight > 0: loss = loss + self.clash_weight * backbone_clash_loss(x)
        if self.angle_weight > 0: loss = loss + self.angle_weight * backbone_angle_loss(x)
        return loss

    def breakdown(self, x_norm: Tensor) -> dict:
        x = (x_norm * self.coord_scale).detach()
        b = backbone_bond_loss(x).mean().item()  if self.bond_weight  > 0 else 0.0
        c = backbone_clash_loss(x).mean().item() if self.clash_weight > 0 else 0.0
        a = backbone_angle_loss(x).mean().item() if self.angle_weight > 0 else 0.0
        return {
            'phys_bond':  b,
            'phys_clash': c,
            'phys_angle': a,
            'phys_total': self.bond_weight * b + self.clash_weight * c + self.angle_weight * a,
        }

    def __repr__(self):
        return (f"BackbonePhysics(bond×{self.bond_weight}, "
                f"clash×{self.clash_weight}, angle×{self.angle_weight}, "
                f"coord_scale={self.coord_scale})")
