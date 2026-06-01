"""
data/transforms.py
==================
SE(3) data augmentation for Chignolin Cα training.

Referenced in data/dataset.py (line was commented out):
    # from utils.transforms import RandomSE3Transform

Set  augment_se3: true  in your config to use this.

Why augment with SE(3)?
-----------------------
The non-equivariant baseline (MLP, Transformer) has NO built-in awareness
that rotating a structure gives the same molecule. Data augmentation forces
the model to learn this by presenting the same structure at many orientations.

For the EGNN (models/egnn.py), SE(3) augmentation is OPTIONAL — the
architecture is already equivariant, so it does not need to learn rotational
invariance from data. It can still help regularise training slightly.

Usage in dataset.py (already supported via the transform= argument):
    from data.transforms import RandomSE3Transform
    transform = RandomSE3Transform()
    ds = ChignolinDataset(path, transform=transform)
"""

import torch
import torch.nn as nn
from torch import Tensor


class RandomSE3Transform:
    """
    Apply a random rotation (and optionally translation) to a Cα point cloud.

    The random rotation is drawn uniformly from SO(3) using the QR decomposition
    of a random Gaussian matrix — this gives a Haar-uniform rotation.

    Called on a single structure (10, 3) from ChignolinDataset.__getitem__.

    Note: structures are already centered (zero-CoM), so translation is omitted
    by default to preserve the zero-CoM property needed for diffusion.

    Parameters
    ----------
    apply_rotation    : always True — the main augmentation
    apply_translation : False by default — would break zero-CoM centering
    """

    def __init__(self,
                 apply_rotation:    bool = True,
                 apply_translation: bool = False,
                 translation_std:   float = 0.1):
        self.apply_rotation    = apply_rotation
        self.apply_translation = apply_translation
        self.translation_std   = translation_std

    def __call__(self, coords: Tensor) -> Tensor:
        """
        coords : (N, 3)  centered + scaled Cα coordinates (from __getitem__)
        Returns: (N, 3)  rotated (and optionally translated) coordinates
        """
        if self.apply_rotation:
            coords = self._random_rotation(coords)

        if self.apply_translation:
            t = torch.randn(1, 3, device=coords.device) * self.translation_std
            coords = coords + t

        return coords

    @staticmethod
    def _random_rotation(x: Tensor) -> Tensor:
        """
        Apply a Haar-uniform random rotation to x ∈ ℝ^{N×3}.

        Method: sample A ~ N(0,I)_{3×3}, compute QR decomposition,
                fix sign so det(Q)=+1 (proper rotation, not improper).
        """
        A = torch.randn(3, 3, device=x.device, dtype=x.dtype)
        Q, R = torch.linalg.qr(A)

        # Ensure det(Q) = +1 (proper rotation)
        # QR decomposition is unique up to sign of columns;
        # sign(diag(R)) tells us how to fix it.
        sign = torch.sign(torch.diag(R))
        Q    = Q * sign.unsqueeze(0)   # flip columns where needed

        # Q: (3, 3),  x: (N, 3)  →  (x @ Qᵀ): (N, 3)
        return x @ Q.T


class IdentityTransform:
    """No-op transform — returned when augment_se3: false."""

    def __call__(self, coords: Tensor) -> Tensor:
        return coords


def get_transform(config: dict):
    """
    Factory function: return the right transform based on config.

    config['data']['augment_se3'] : bool
    """
    if config.get('data', {}).get('augment_se3', False):
        return RandomSE3Transform()
    return None
