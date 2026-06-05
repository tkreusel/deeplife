"""
models/backbone_transformer.py
================================
AdaLN Transformer score network for backbone N–Cα–C frames.

Operates on 30-atom backbone coordinates (N, Cα, C per residue × 10 residues)
with the same AdaLN-Zero + energy CFG architecture that achieved 90% bond
validity on the 10-atom Cα model.

Token design
------------
Each of the 30 backbone atoms is one token.  Three embeddings are ADDED:

    h_i = coord_proj(x_i)            # 3-D position → D
          + atom_type_embed(type_i)   # N=0, Cα=1, C=2  →  D
          + residue_embed(res_i)      # residue index 0–9  →  D

atom_type_embed captures the chemical identity of each backbone atom.
residue_embed captures which residue each atom belongs to.
Together they let the model distinguish, e.g., the N of residue 3 from the
Cα of residue 3 or the N of residue 4 — without relying on position order alone.

The attention mechanism learns both intra-residue constraints (N–CA–C rigid
triangle) and inter-residue correlations (peptide bond geometry, φ/ψ couplings,
the β-hairpin fold topology).

Energy conditioning / CFG
--------------------------
Same convention as AdaLNEnergyTransformerScoreNetwork:
    τ=0 → compact/folded,  τ=1 → extended/transient
    c = time_emb + energy_emb, with CFG dropout during training.

model_type: backbone_transformer   (in config yaml)
"""

import torch
import torch.nn as nn

from models.baseline    import SinusoidalTimestepEmbedding
from models.transformer_adaln import AdaLNBlock

# Atom type indices used throughout the pipeline
ATOM_TYPES = {'N': 0, 'CA': 1, 'C': 2}
N_ATOM_TYPES = 3   # N, Cα, C


class BackboneTransformerScoreNetwork(nn.Module):
    """
    AdaLN Transformer + energy CFG for 30-atom backbone.

    Input:  x_t      (B, 30, 3)  noisy backbone coordinates
            t        (B,)        diffusion timestep
            energy_z (B,) | None z-score energy; None → unconditional
    Output:          (B, 30, 3)  predicted noise
    """

    def __init__(
        self,
        n_residues:       int   = 10,
        hidden_dim:       int   = 256,
        n_heads:          int   = 8,
        n_layers:         int   = 6,
        time_dim:         int   = 64,
        dropout:          float = 0.1,
        energy_drop_prob: float = 0.15,
    ):
        super().__init__()
        self.n_residues       = n_residues
        self.n_atoms          = n_residues * 3   # 30
        self.energy_drop_prob = energy_drop_prob

        # ── Conditioning: time + energy (same as AdaLNEnergyTransformerScoreNetwork) ──
        self.time_mlp = nn.Sequential(
            SinusoidalTimestepEmbedding(time_dim),
            nn.Linear(time_dim, time_dim * 4),
            nn.SiLU(),
            nn.Linear(time_dim * 4, hidden_dim),
        )
        self.energy_mlp = nn.Sequential(
            nn.Linear(1, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.null_energy_emb = nn.Parameter(torch.zeros(hidden_dim))

        # ── Token embeddings ──────────────────────────────────────────────────
        self.coord_proj      = nn.Linear(3, hidden_dim)
        # atom type: N / Cα / C  (3 types, repeated 10× in the sequence)
        self.atom_type_embed = nn.Embedding(N_ATOM_TYPES, hidden_dim)
        # residue index: which residue (0–9) each atom belongs to
        self.residue_embed   = nn.Embedding(n_residues, hidden_dim)

        # Pre-compute fixed token metadata as buffers (no grad)
        # atom_types : [0,1,2, 0,1,2, …]  (30,)
        # res_indices : [0,0,0, 1,1,1, …]  (30,)
        atom_types  = torch.tensor([t for _ in range(n_residues)
                                    for t in range(N_ATOM_TYPES)], dtype=torch.long)
        res_indices = torch.tensor([r for r in range(n_residues)
                                    for _ in range(N_ATOM_TYPES)], dtype=torch.long)
        self.register_buffer('atom_types',  atom_types)
        self.register_buffer('res_indices', res_indices)

        # ── Transformer blocks ────────────────────────────────────────────────
        self.blocks = nn.ModuleList([
            AdaLNBlock(hidden_dim, n_heads, dropout=dropout)
            for _ in range(n_layers)
        ])

        # ── Final AdaLN norm + output projection ──────────────────────────────
        self.norm_out = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.final_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 2 * hidden_dim, bias=True),
        )
        nn.init.zeros_(self.final_modulation[-1].weight)
        nn.init.zeros_(self.final_modulation[-1].bias)

        self.out_proj = nn.Linear(hidden_dim, 3)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    @staticmethod
    def temperature_to_energy_z(tau: float) -> float:
        return 4.0 * tau - 2.0

    def forward(
        self,
        x_t:      torch.Tensor,
        t:        torch.Tensor,
        energy_z: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        x_t      : (B, 30, 3)
        t        : (B,)
        energy_z : (B,) | None
        returns  : (B, 30, 3)
        """
        B = x_t.shape[0]
        device = x_t.device

        # ── Conditioning vector c ─────────────────────────────────────────────
        c = self.time_mlp(t)                                     # (B, D)

        if energy_z is not None:
            e_emb = self.energy_mlp(energy_z.unsqueeze(-1))     # (B, D)
            if self.training and self.energy_drop_prob > 0.0:
                drop = torch.rand(B, device=device) < self.energy_drop_prob
                null = self.null_energy_emb.unsqueeze(0).expand(B, -1)
                e_emb = torch.where(drop.unsqueeze(-1), null, e_emb)
        else:
            e_emb = self.null_energy_emb.unsqueeze(0).expand(B, -1)

        c = c + e_emb                                            # (B, D)

        # ── Token sequence ────────────────────────────────────────────────────
        h  = self.coord_proj(x_t)                                # (B, 30, D)
        h  = h + self.atom_type_embed(self.atom_types)           # (B, 30, D)
        h  = h + self.residue_embed(self.res_indices)            # (B, 30, D)

        for block in self.blocks:
            h = block(h, c)                                      # (B, 30, D)

        # ── Final modulated norm + projection ─────────────────────────────────
        shift, scale = self.final_modulation(c).chunk(2, dim=-1)
        h = self.norm_out(h)
        h = h * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

        return self.out_proj(h)                                  # (B, 30, 3)
