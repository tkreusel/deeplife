# Shared TODO

Tasks for the full team. When you pick up a task, add your name + date next to it. When done, mark it and add results to [STATUS.md](STATUS.md).

**Format:** `- [ ] Task — *Owner (date)*`  
**Done:** `- [x] Task — *Owner (date)* — result summary`

---

## Immediate / Blockers

- [ ] Distribute data files to all team members — source: `/g/korbel2/shahp/deeplife/data/` on cluster

---

## In Progress

*(nothing active right now)*

---

## Up Next

### Infrastructure
- [ ] Fix `local_baseline.yaml` + `egnn_local.yaml` to use `data/val.npz` (currently reference `data/valid.npz`)
- [ ] Delete `scripts/tmp.py` — one-off debugging script, pollutes repo

### Training: improve FlowMatch bond validity
Bond validity is only 5% after 500 epochs (see STATUS.md). Suggested experiments:
- [ ] Try higher LR (`lr: 3e-4` instead of `1e-4`) — flow matching is less sensitive than EGNN+DDPM
- [ ] Try `coord_scale: 16.32` (same as DDPM default) — the current 5.0 may produce velocity targets with the wrong scale
- [ ] Try more epochs (e.g. 1000) with the current config
- [ ] Try `augment_se3: true` — data augmentation may help the equivariant model generalize

### Evaluation
- [ ] Generate 50+ PDB files from best model and visualize in PyMOL:
  ```bash
  python scripts/evaluate.py \
    --ckpt checkpoints/egnn/v1/best.pt \
    --test data/test.npz --n 100 --save_pdb outputs/egnn_pdbs
  ```
- [ ] Run `plot_training.py` to compare loss curves from all runs and save for the paper:
  ```bash
  python scripts/plot_training.py \
    --logs checkpoints/flowmatch/v2/log.jsonl \
           checkpoints/baseline/v3/log.jsonl \
           checkpoints/egnn/v1/log.jsonl \
    --labels "FlowMatch-EGNN" "Transformer-DDPM" "EGNN-DDPM" \
    --save plots/loss_all.png
  ```
- [ ] ODE/DDIM step count sweep: compare quality at 10, 25, 50, 100, 200 steps

---

## Ideas / Stretch Goals

- [ ] Try MLP model (`model_type: mlp`) — no full-training config yet, 5 min to write
- [ ] Explore physics regularization: `training_loss()` in `diffusion.py` has `physics_reg_weight` — wire it up and test effect on bond validity
- [ ] Add WandB logging to training scripts (`wandb/` already gitignored, clean to add)
- [ ] Benchmark EGNN equivariance: run `model.check_equivariance()` on `egnn/v1` checkpoint to confirm it holds after training
- [ ] Try FlowMatch with DDPM-style noise (`ContinuousFlowMatching` base class) instead of zero-CoM variant — check if it converges differently
- [ ] Investigate whether `augment_se3: true` helps Transformer-DDPM convergence (currently unused)

---

## Completed

- [x] Implement MLP score network — *initial commit*
- [x] Implement Transformer score network — *initial commit*
- [x] Implement DDPM diffusion (`models/diffusion.py`) — *initial commit*
- [x] Add noise scaling by std (bug fix) — *commit 5d2ae54*
- [x] Add `requirements.txt` — *commit 6df1d89*
- [x] Implement EGNN score network (`models/egnn.py`) — *commit 5b23581*
- [x] Add GPU-optimized EGNN training script (`train_egnn.py`) — *commit 5b23581*
- [x] Add ZeroCoM diffusion variant (`diffusion_zerocom.py`) — *commit 5b23581*
- [x] Add SE(3) data augmentation (`data/transforms.py`) — *commit 5b23581*
- [x] Add dual-checkpoint evaluation mode (`evaluate.py`) — *commit 5b23581*
- [x] Create collaborative workspace (COLLAB/, CLAUDE.md, environment.yml) — *marik, 2026-06-01*
- [x] Fix `configs/egnn.yaml` hardcoded cluster paths → relative paths — *tkreusel, 2026-06-01*
- [x] Fix `configs/baseline.yaml` val_path `valid.npz` → `val.npz` — *tkreusel, 2026-06-01*
- [x] Add `configs/*personal*` to `.gitignore` — *tkreusel, 2026-06-01*
- [x] Run full EGNN training (500 epochs) — *pshah, 2026-06-01* — best val 0.1779, ckpt: `checkpoints/egnn/v1/best.pt`
- [x] Run full Transformer-DDPM training (500 epochs, batch=512) — *tkreusel, 2026-06-01* — best val 0.0891, ckpt: `checkpoints/baseline/v3/best.pt`
- [x] Implement SE(3)-equivariant flow matching (`models/flow_matching.py`, `scripts/train_flow.py`) — *marik, 2026-06-01*
- [x] Run full FlowMatch training (500 epochs) — *marik, 2026-06-01* — best val 0.6351 (different scale from DDPM), ckpt: `checkpoints/flowmatch/v2/best.pt`
- [x] Expand `evaluate.py` to N-model comparison — *marik, 2026-06-01*
- [x] Three-way evaluation (FlowMatch vs Transformer-DDPM vs EGNN-DDPM) — *marik, 2026-06-01* — results in STATUS.md; FlowMatch has best MMD+Rg, worst bond validity
- [x] Reorganise EGNN checkpoint `checkpoints/v2/` → `checkpoints/egnn/v1/` — *2026-06-01*
