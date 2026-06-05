"""
scripts/prepare_backbone_data.py
=================================
Extract N, Cα, C backbone atoms from data_all_atom/ and save to data_backbone/.

All 93 heavy atoms share the same topology across all 79 632 structures.
We extract the 30 backbone indices once from structure 0 (bond-graph walk),
then apply the same index slice to every structure.

Output shapes match data/train.npz exactly (coords, energies, centroids)
except coords is (N, 30, 3) instead of (N, 10, 3).

Usage:
    python scripts/prepare_backbone_data.py
"""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

SPLITS = ['train', 'val', 'test']
SRC_DIR = Path('data_all_atom')
DST_DIR = Path('data_backbone')


# ── Backbone index extraction ─────────────────────────────────────────────────

def find_backbone_indices(xyz: np.ndarray) -> list[int]:
    """
    Walk the backbone chain N→CA→C using covalent bond geometry.

    Works for all standard amino acids including PRO (where N bonds both to
    CA and to CD in the ring; CA is identified by having more bonds than CD).

    xyz : (93, 3)  one all-atom structure in Å
    returns : list of 30 atom indices [N0, CA0, C0, N1, CA1, C1, ...]
    """
    dist   = np.linalg.norm(xyz[:, None] - xyz[None, :], axis=-1)
    bonded = (dist < 1.65) & (dist > 0.1)
    n_bonds = bonded.sum(axis=1)

    def has_co(j: int) -> bool:
        return bool(np.any(dist[j, bonded[j]] < 1.28))

    # N-terminus: the unique atom with exactly 1 heavy-atom bond at N-CA length
    n_term = next(
        i for i in range(len(xyz))
        if (n_bonds[i] == 1
            and 1.39 <= float(dist[i, np.where(bonded[i])[0][0]]) <= 1.52)
    )

    bb, prev, curr = [], -1, n_term
    for res in range(10):
        n_i = int(curr)
        bb.append(n_i)

        # CA: N-neighbor at N-CA distance, pick atom with MOST bonds when ambiguous.
        # This correctly handles PRO where N bonds to both CA (3 bonds) and CD (2 bonds).
        nbrs = [int(j) for j in np.where(bonded[n_i])[0]
                if j != prev and 1.39 <= float(dist[n_i, j]) <= 1.55]
        ca_i = max(nbrs, key=lambda j: int(n_bonds[j]))
        bb.append(ca_i)

        # C: CA-neighbor (not N) that has a C=O double-bond neighbor (dist < 1.28Å)
        nbrs_ca = [int(j) for j in np.where(bonded[ca_i])[0] if j != n_i]
        c_i = next((j for j in nbrs_ca if has_co(j)), None)
        # Fallback: one hop further (catches edge cases where O is misdetected)
        if c_i is None:
            for mid in nbrs_ca:
                for k in [int(j) for j in np.where(bonded[mid])[0] if j != ca_i]:
                    if has_co(k) and 1.48 <= float(dist[ca_i, k]) <= 1.60:
                        c_i = k
                        break
                if c_i is not None:
                    break
        bb.append(int(c_i))

        if res < 9:
            # N_next: C-neighbor at peptide bond length (1.28–1.42Å), not O (< 1.26Å)
            cnbrs = [int(j) for j in np.where(bonded[c_i])[0]
                     if j != ca_i and float(dist[c_i, j]) > 1.26]
            n_next = next((j for j in cnbrs
                           if 1.28 <= float(dist[c_i, j]) <= 1.42), cnbrs[0])
            prev, curr = c_i, n_next

    return bb


def verify_backbone(indices: list[int], xyz: np.ndarray, tol: float = 1.65) -> bool:
    dist = np.linalg.norm(xyz[:, None] - xyz[None, :], axis=-1)
    return all(float(dist[indices[i], indices[i + 1]]) < tol for i in range(29))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    DST_DIR.mkdir(exist_ok=True)

    # Determine backbone indices from train split
    print("Finding backbone indices from train split, structure 0…")
    ref = np.load(SRC_DIR / 'train.npz')
    bb_idx = find_backbone_indices(ref['coords'][0])
    print(f"  Indices: {bb_idx}")

    # Verify on a handful of structures
    print("Verifying across 5 random structures…")
    for i in [0, 1000, 10000, 40000, 79000]:
        ok = verify_backbone(bb_idx, ref['coords'][i], tol=1.65)
        print(f"  struct {i:5d}: {'OK' if ok else 'FAIL'}")

    # Print bond lengths from structure 0 for the record
    dist0 = np.linalg.norm(ref['coords'][0][:, None] - ref['coords'][0][None, :], axis=-1)
    labels = ['N', 'CA', 'C'] * 10
    print("\nBond lengths (structure 0):")
    for i in range(29):
        d = dist0[bb_idx[i], bb_idx[i + 1]]
        print(f"  {labels[i]}{i // 3}→{labels[(i + 1) % 3]}{(i + 1) // 3}"
              f"  [{bb_idx[i]:2d}→{bb_idx[i + 1]:2d}]  {d:.4f} Å")

    idx = np.array(bb_idx)

    for split in SPLITS:
        src = SRC_DIR / f'{split}.npz'
        dst = DST_DIR / f'{split}.npz'
        if not src.exists():
            print(f"  Skipping {src} (not found)")
            continue

        data = np.load(src)
        coords_aa = data['coords']                     # (N, 93, 3)
        coords_bb = coords_aa[:, idx, :]              # (N, 30, 3)

        # Recompute centroid from backbone atoms (not all 93)
        centroids_bb = coords_bb.mean(axis=1, keepdims=True)  # (N, 1, 3)

        np.savez(
            dst,
            coords    = coords_bb,
            energies  = data['energies'],
            centroids = centroids_bb,
        )
        print(f"  {split}: {coords_bb.shape} → {dst}")

    print("\nDone.")


if __name__ == '__main__':
    main()
