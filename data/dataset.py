"""
data/dataset.py

PyTorch Dataset for Chignolin Cα coordinates.
Loads from train.npz / valid.npz / test.npz, each with keys:
    coords:    (N, 10, 3)  Cα coordinates in Ångströms
    energies:  (N,)        scalar potential energy per structure
    centroids: (N, 1, 3)   centre of mass per structure
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class ChignolinDataset(Dataset):
    """
    Dataset of Chignolin Cα coordinates loaded from a .npz file.

    Coordinates are centered at the origin by subtracting the centroid.
    Each call to __getitem__ returns a dict with keys:
        coords:    (10, 3)  centered Cα coordinates
        energies:  ()       scalar energy
        centroids: (1, 3)   original centroid (before centering)

    Args:
        path:      path to .npz file, e.g. 'data/train.npz'
        transform: optional callable applied to coords after centering,
                   e.g. RandomSE3Transform for data augmentation
    """

    def __init__(self, path: str, transform=None):
        data = np.load(path)

        self.coords    = torch.tensor(data['coords'],    dtype=torch.float32)
        self.energies  = torch.tensor(data['energies'],  dtype=torch.float32)
        self.centroids = torch.tensor(data['centroids'], dtype=torch.float32)
        self.transform = transform

        # centroids may be shape (N, 3) or (N, 1, 3) — normalise to (N, 1, 3)
        if self.centroids.ndim == 2:
            self.centroids = self.centroids.unsqueeze(1)

        assert self.coords.ndim == 3, \
            f"Expected coords shape (N, 10, 3), got {self.coords.shape}"
        assert self.coords.shape[1:] == (10, 3), \
            f"Expected 10 residues with 3 coords each, got {self.coords.shape[1:]}"
        assert len(self.coords) == len(self.energies) == len(self.centroids), \
            "coords, energies and centroids must have the same length"

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        coords   = self.coords[idx].clone()    # (10, 3)
        centroid = self.centroids[idx]         # (1, 3)

        # center to origin — diffusion assumes zero-mean coordinates
        coords = coords - centroid

        if self.transform is not None:
            coords = self.transform(coords)

        return {
            'coords':    coords,                  # (10, 3)
            'energies':  self.energies[idx],      # scalar
            'centroids': centroid,                # (1, 3)  kept for reference
        }


def get_dataloaders(config: dict):
    """
    Build train/val/test DataLoaders from config dict.

    Expected config keys:
        data.train_path, data.val_path, data.test_path
        data.augment_se3
        training.batch_size, training.num_workers
    """
    # from utils.transforms import RandomSE3Transform

    # transform = RandomSE3Transform() if config['data'].get('augment_se3', False) else None
    transform = None

    train_ds = ChignolinDataset(config['data']['train_path'], transform=transform)
    val_ds   = ChignolinDataset(config['data']['val_path'])
    test_ds  = ChignolinDataset(config['data']['test_path'])

    print(f"Dataset sizes — train: {len(train_ds)}, val: {len(val_ds)}, test: {len(test_ds)}")

    train_loader = DataLoader(
        train_ds,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training'].get('num_workers', 4),
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training'].get('num_workers', 4),
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=config['training']['batch_size'],
        shuffle=False,
    )

    return train_loader, val_loader, test_loader