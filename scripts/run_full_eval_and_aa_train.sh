#!/bin/bash
# run_full_eval_and_aa_train.sh
# ==============================
# Runs after egnn_adaln/v1 training completes:
#   1. Structural + physics evaluation (7 models)
#   2. SE(3) equivariance check        (7 models)
#   3. Energy conditioning sweep       (2 energy-conditioned models)
#   4. All-atom smoke-test             (local config, 3 epochs)
#   5. All-atom production training    (500 epochs, GPU 1)
#
# Usage:
#   CUDA_VISIBLE_DEVICES=1 bash scripts/run_full_eval_and_aa_train.sh

set -e
cd /vol/workspace/P4T1/deeplife
PY=/vol/workspace/P4T1/miniforge3/envs/deeplife/bin/python

CKPT_EGNN_ADALN=$(ls -t checkpoints/egnn_adaln/v*/best.pt 2>/dev/null | head -1)
echo "Primary checkpoint: $CKPT_EGNN_ADALN"

# ── 1. Structural + physics metrics ──────────────────────────────────────────
echo ""
echo "=== [1/5] Structural + physics evaluation ==="
$PY scripts/evaluate.py \
  --ckpt     "$CKPT_EGNN_ADALN" \
  --ckpt_ref checkpoints/egnn/v4/best.pt \
             checkpoints/transformer_adaln/v1/best.pt \
             checkpoints/transformer_adaln_energy_physics/v1/best.pt \
             checkpoints/flowmatch_physics/v3/best.pt \
             checkpoints/flowmatch_energy/v2/best.pt \
             checkpoints/flowmatch_energy_physics/v1/best.pt \
  --labels   "EGNN+AdaLN+Physics" "EGNN+Physics" "AdaLN" "AdaLN+Energy+Physics" \
             "FlowMatch+Physics" "FlowMatch+Energy" "FlowMatch+Energy+Physics" \
  --test     data/test.npz \
  --n 1000 --steps 100 \
  --save     plots/comparison_adaln_final.png \
  --out_json plots/comparison_adaln_final.json \
  2>&1 | tee logs/eval_adaln_final.log

# ── 2. SE(3) equivariance check ───────────────────────────────────────────────
echo ""
echo "=== [2/5] Equivariance check ==="
$PY scripts/check_equivariance.py \
  --ckpt     "$CKPT_EGNN_ADALN" \
  --ckpt_ref checkpoints/egnn/v4/best.pt \
             checkpoints/transformer_adaln/v1/best.pt \
             checkpoints/transformer_adaln_energy_physics/v1/best.pt \
             checkpoints/flowmatch_physics/v3/best.pt \
             checkpoints/flowmatch_energy/v2/best.pt \
             checkpoints/flowmatch_energy_physics/v1/best.pt \
  --labels   "EGNN+AdaLN+Physics" "EGNN+Physics" "AdaLN" "AdaLN+Energy+Physics" \
             "FlowMatch+Physics" "FlowMatch+Energy" "FlowMatch+Energy+Physics" \
  --n_noise 20 --n_rotations 50 --n_generate 500 --steps 50 \
  --save     plots/equivariance_adaln_final.png \
  2>&1 | tee logs/equivariance_adaln_final.log

# ── 3. Energy conditioning sweep ─────────────────────────────────────────────
echo ""
echo "=== [3/5] Energy conditioning sweep — EGNN+AdaLN ==="
$PY scripts/analyze_energy_conditioning.py \
  --checkpoint "$CKPT_EGNN_ADALN" \
  --test       data/test.npz \
  --temperatures 0.0 0.25 0.5 0.75 1.0 \
  --n 500 --steps 100 --guidance_scale 2.0 \
  --save plots/energy_egnn_adaln.png \
  2>&1 | tee logs/energy_egnn_adaln.log

echo ""
echo "=== [3b/5] Energy conditioning sweep — FlowMatch+Energy ==="
$PY scripts/analyze_energy_conditioning.py \
  --checkpoint checkpoints/flowmatch_energy/v2/best.pt \
  --test       data/test.npz \
  --temperatures 0.0 0.25 0.5 0.75 1.0 \
  --n 500 --steps 100 --guidance_scale 2.0 \
  --save plots/energy_flowmatch.png \
  2>&1 | tee logs/energy_flowmatch.log

# ── 4. All-atom smoke-test ────────────────────────────────────────────────────
echo ""
echo "=== [4/5] All-atom smoke-test (local config, 3 epochs) ==="
$PY scripts/train_egnn_adaln.py --config configs/egnn_adaln_aa_local.yaml \
  2>&1 | tee logs/egnn_adaln_aa_local.log
echo "Smoke-test passed — launching production training"

# ── 5. All-atom production training ──────────────────────────────────────────
echo ""
echo "=== [5/5] All-atom production training (500 epochs) ==="
$PY scripts/train_egnn_adaln.py --config configs/egnn_adaln_aa.yaml \
  2>&1 | tee logs/egnn_adaln_aa_v1.log

echo ""
echo "All done. Run evaluate.py on checkpoints/egnn_adaln_aa/v1/best.pt when ready."
