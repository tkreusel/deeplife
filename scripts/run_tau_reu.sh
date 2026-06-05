#!/bin/bash
# Run τ-vs-REU Rosetta scoring for all energy-conditioned models.
#
# This appends `tau_rosetta` to the existing metrics.json for each group
# without touching physics, equivariance, novelty, or pyrosetta sections.
#
# Output: plots/eval_overnight/{ca_only,backbone}/
#   • metrics.json  ← sections.energy.<label>.tau_rosetta added
#   • figure3b_tau_reu.png  ← new scatter/violin plot
#   • figure3_energy_<label>.png  ← regenerated (same data, no change)
#
# Runtime estimate: ~45 min for ca_only (12 energy models × 5τ × 15 structs)
#                   ~5 min for backbone (2 models)
#
# Run from the deeplife/ repo root:
#   bash scripts/run_tau_reu.sh

set -e
cd "$(dirname "$0")/.."

PYTHON=/vol/workspace/P4T1/miniforge3/envs/deeplife/bin/python

# ── ca_only ──────────────────────────────────────────────────────────────────
echo "=== ca_only group ==="
$PYTHON scripts/eval_v2/main.py \
    --ckpt '/vol/workspace/P4T1/deeplife/checkpoints/baseline/v3/best.pt' \
    --ckpt_ref \
        '/vol/workspace/P4T1/deeplife/checkpoints/baseline_transformed/v1/best.pt' \
        '/vol/workspace/P4T1/deeplife/checkpoints/transformer_adaln/v1/best.pt' \
        '/vol/workspace/P4T1/deeplife/checkpoints/transformer_adaln_energy/v1/best.pt' \
        '/vol/workspace/P4T1/deeplife/checkpoints/transformer_adaln_energy_physics/v1/best.pt' \
        '/vol/workspace/P4T1/deeplife/checkpoints/transformer_adaln_sc/v1/best.pt' \
        '/vol/workspace/P4T1/deeplife/checkpoints/transformer_adaln_sc/v2/best.pt' \
        '/vol/workspace/P4T1/deeplife/checkpoints/transformer_adaln_sc/v3/best.pt' \
        '/vol/workspace/P4T1/deeplife/checkpoints/egnn/v1/best.pt' \
        '/vol/workspace/P4T1/deeplife/checkpoints/egnn/v3/best.pt' \
        '/vol/workspace/P4T1/deeplife/checkpoints/egnn/v4/best.pt' \
        '/vol/workspace/P4T1/deeplife/checkpoints/egnn_adaln/v1/best.pt' \
        '/vol/workspace/P4T1/deeplife/checkpoints/flowmatch/v2/best.pt' \
        '/vol/workspace/P4T1/deeplife/checkpoints/flowmatch_physics/v3/best.pt' \
        '/vol/workspace/P4T1/deeplife/checkpoints/flowmatch_energy/v2/best.pt' \
        '/vol/workspace/P4T1/deeplife/checkpoints/flowmatch_energy_physics/v1/best.pt' \
        '/vol/workspace/P4T1/deeplife/checkpoints/se3flow_energy/v1/best.pt' \
        '/vol/workspace/P4T1/deeplife/checkpoints/se3flow_adaln_velocity/v1/best.pt' \
        '/vol/workspace/P4T1/deeplife/checkpoints/torsion_flow/v1/best.pt' \
        '/vol/workspace/P4T1/deeplife/checkpoints/torsion_transformer/v1/best.pt' \
    --labels \
        'Transformer-DDPM-v3' \
        'Transformer-DDPM-SE3aug-v1' \
        'AdaLN-Transformer-v1' \
        'AdaLN-Transformer-Energy-v1' \
        'AdaLN-Transformer-Energy-Physics-v1' \
        'AdaLN-Transformer-SelfCond-v1' \
        'AdaLN-Transformer-SelfCond-v2' \
        'AdaLN-Transformer-SelfCond-v3' \
        'EGNN-DDPM-v1' \
        'EGNN-DDPM-Physics0.05-v3' \
        'EGNN-DDPM-Physics0.10-v4' \
        'EGNN-AdaLN-Energy-v1' \
        'FlowMatch-EGNN-v2' \
        'FlowMatch-Physics-v3' \
        'FlowMatch-Energy-v2' \
        'FlowMatch-Energy-Physics-v1' \
        'SE3Flow-Energy-v1' \
        'SE3Flow-AdaLN-Velocity-v1' \
        'TorsionFlow-MLP-v1' \
        'TorsionTransformer-v1' \
    --test '/vol/workspace/P4T1/deeplife/data/test.npz' \
    --n 1000 \
    --steps 100 \
    --batch 256 \
    --sections energy \
    --temperatures 0.0 0.25 0.5 0.75 1.0 \
    --guidance_scale 2.0 \
    --n_per_tau 200 \
    --n_tau_rosetta 15 \
    --out_dir 'plots/eval_overnight/ca_only' \
    --seed 0

# ── backbone ──────────────────────────────────────────────────────────────────
echo "=== backbone group ==="
$PYTHON scripts/eval_v2/main.py \
    --ckpt '/vol/workspace/P4T1/deeplife/checkpoints/backbone_transformer/v1/best.pt' \
    --ckpt_ref '/vol/workspace/P4T1/deeplife/checkpoints/backbone_ipa/v1/best.pt' \
    --labels 'BackboneTransformer-v1' 'BackboneIPA-v1' \
    --test '/vol/workspace/P4T1/deeplife/data_backbone/test.npz' \
    --n 1000 \
    --steps 100 \
    --batch 256 \
    --sections energy \
    --temperatures 0.0 0.25 0.5 0.75 1.0 \
    --guidance_scale 2.0 \
    --n_per_tau 200 \
    --n_tau_rosetta 15 \
    --out_dir 'plots/eval_overnight/backbone' \
    --seed 0

echo ""
echo "Done. New outputs:"
echo "  plots/eval_overnight/ca_only/figure3b_tau_reu.png"
echo "  plots/eval_overnight/backbone/figure3b_tau_reu.png"
echo "  (tau_rosetta added to metrics.json in both dirs)"
