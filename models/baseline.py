import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalTimestepEmbedding(nn.Module):
    """
    Encodes a scalar timestep t into a continuous vector of dimension `dim`.
    Uses sine and cosine at exponentially spaced frequencies.

    Input:  t  (B,)   integer timesteps
    Output:    (B, dim)
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half   = self.dim // 2

        # exponentially spaced frequencies from 1 to 10000
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=device) / (half - 1)
        )                                           # (half,)

        # outer product: each timestep × each frequency
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)   # (B, half)

        # concatenate sin and cos
        embedding = torch.cat([args.sin(), args.cos()], dim=-1)  # (B, dim)

        # if dim is odd, pad by one zero
        if self.dim % 2 == 1:
            embedding = F.pad(embedding, (0, 1))

        return embedding
    
class MLPScoreNetwork(nn.Module):
    """
    Non-equivariant MLP baseline.

    Flattens (B, N, 3) → (B, N*3), concatenates time embedding,
    runs through MLP, reshapes back to (B, N, 3).

    Input:  x_t (B, N, 3),  t (B,)
    Output:     (B, N, 3)   predicted noise
    """

    def __init__(
        self,
        n_residues: int = 10,
        hidden_dim: int = 256,
        n_layers:   int = 4,
        time_dim:   int = 64,
        dropout:    float = 0.1,
    ):
        super().__init__()
        self.n_residues = n_residues
        input_dim = n_residues * 3      # 30 for Chignolin

        # timestep embedding: scalar t → (B, time_dim)
        self.time_mlp = nn.Sequential(
            SinusoidalTimestepEmbedding(time_dim),
            nn.Linear(time_dim, time_dim * 4),
            nn.SiLU(),
            nn.Linear(time_dim * 4, time_dim),
        )

        # main network: coords + time → coords
        layers = [nn.Linear(input_dim + time_dim, hidden_dim), nn.SiLU()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Dropout(dropout)]
        layers.append(nn.Linear(hidden_dim, input_dim))

        self.net = nn.Sequential(*layers)

        # initialise final layer to zero — standard for diffusion models,
        # means the network starts as a no-op and learns from there
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        B, N, _ = x_t.shape

        x_flat = x_t.reshape(B, -1)                      # (B, 30)
        t_emb  = self.time_mlp(t)                        # (B, time_dim)
        h      = torch.cat([x_flat, t_emb], dim=-1)      # (B, 30 + time_dim)
        out    = self.net(h)                              # (B, 30)

        return out.reshape(B, N, 3)
    
    
class TransformerBlock(nn.Module):
    """
    Standard pre-norm Transformer encoder block.
    Pre-norm (LayerNorm before attention) is more stable than post-norm
    for small datasets.
    """

    def __init__(self, dim: int, n_heads: int, ff_mult: int = 4, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.attn  = nn.MultiheadAttention(
            dim, n_heads, dropout=dropout, batch_first=True
        )
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * ff_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * ff_mult, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # self-attention with residual
        h, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x    = x + h
        # feed-forward with residual
        x    = x + self.ff(self.norm2(x))
        return x
    
class TransformerScoreNetwork(nn.Module):
    """
    Transformer-based score network.
    Treats each residue as a token — self-attention models all pairwise
    residue interactions.

    Input:  x_t (B, N, 3),  t (B,)
    Output:     (B, N, 3)   predicted noise
    """

    def __init__(
        self,
        n_residues: int = 10,
        hidden_dim: int = 128,
        n_heads:    int = 4,
        n_layers:   int = 4,
        time_dim:   int = 64,
        dropout:    float = 0.1,
    ):
        super().__init__()
        self.n_residues = n_residues

        # timestep embedding
        self.time_mlp = nn.Sequential(
            SinusoidalTimestepEmbedding(time_dim),
            nn.Linear(time_dim, time_dim * 4),
            nn.SiLU(),
            nn.Linear(time_dim * 4, hidden_dim),
        )

        # project 3D coords to hidden_dim
        self.input_proj = nn.Linear(3, hidden_dim)

        # learnable residue position embeddings
        # these encode which position in the chain each residue is at
        self.pos_embed = nn.Embedding(n_residues, hidden_dim)

        # transformer layers
        self.blocks = nn.ModuleList([
            TransformerBlock(hidden_dim, n_heads, dropout=dropout)
            for _ in range(n_layers)
        ])

        self.norm_out = nn.LayerNorm(hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, 3)

        # zero-init output projection
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        B, N, _ = x_t.shape
        device  = x_t.device

        # project coordinates to token embeddings
        h = self.input_proj(x_t)                              # (B, N, D)

        # add residue positional embeddings
        pos = torch.arange(N, device=device)                  # (N,)
        h   = h + self.pos_embed(pos).unsqueeze(0)            # (B, N, D)

        # add timestep embedding — same for all residues in a structure
        t_emb = self.time_mlp(t).unsqueeze(1)                 # (B, 1, D)
        h     = h + t_emb                                     # (B, N, D)

        # transformer layers
        for block in self.blocks:
            h = block(h)                                      # (B, N, D)

        h   = self.norm_out(h)                                # (B, N, D)
        out = self.out_proj(h)                                # (B, N, 3)

        return out