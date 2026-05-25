import numpy as np
data      = np.load('data/train.npz')
coords    = data['coords']
centroids = data['centroids']
if centroids.ndim == 2:
    centroids = centroids[:, None, :]
centered  = coords - centroids
print(centered.std())