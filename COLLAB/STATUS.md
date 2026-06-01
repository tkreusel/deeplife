# Project Status

Current state of the codebase, known issues, and run history. **Update this after every significant training run or fix.**

Last updated: 2026-06-01 (final session update)

---

## What works (verified in code)

| Component | Status | Notes |
|-----------|--------|-------|
| MLPScoreNetwork | Code complete | No full training run yet |
| TransformerScoreNetwork | Code complete | Full run — best val 0.0891 (`baseline/v3`) |
| EGNNScoreNetwork | Code complete | SE(3)-equivariant; `check_equivariance()` verified |
| GaussianDiffusion (DDPM) | Code complete | Cosine + linear schedules |
| ZeroCoMGaussianDiffusion | Code complete | Used by `train_egnn.py` |
| ContinuousFlowMatching | Code complete | OT-CFM; Euler ODE sampling |
| ZeroCoMFlowMatching | Code complete | Zero-CoM OT-CFM; Heun's 2nd-order ODE; used by `train_flow.py` |
| DDIM sampling | Code complete | Configurable eta, fewer steps |
| SE(3) data augmentation | Code complete | `RandomSE3Transform` via QR decomp |
| Local test configs | Verified | `local_baseline.yaml`, `egnn_local.yaml`, `flowmatch_local.yaml` |
| EMA weight tracking | Code complete | All three training scripts |
| Epoch-versioned checkpoints | Code complete | Auto v1/v2/v3 dirs |
| `quick_sample.py` | Code complete | Handles mlp/transformer/egnn/flowmatch |
| `evaluate.py` | Code complete | N-model comparison; mlp/transformer/egnn/flowmatch; MMD, bond %, Rg |
| `plot_training.py` | Code complete | log.jsonl → multi-run overlay plots (linear + log + LR) |
| AMP training (EGNN + FlowMatch) | Code complete | GradScaler, scaler saved in ckpt |
| `torch.compile` (EGNN + FlowMatch) | Code complete | Optional, PyTorch ≥2.0; strips `_orig_mod.` prefix on load |

---

## Known issues

### `val.npz` vs `valid.npz` in local configs — open
`configs/egnn_local.yaml` and `configs/local_baseline.yaml` still reference `data/valid.npz`.  
**Fix:** `cp data/val.npz data/valid.npz` before running local sanity checks, or edit both configs.

---

### `scripts/tmp.py` — debugging artifact
Loads `data/train.npz` and prints coordinate std. Safe to delete and commit.

---

### FlowMatch bond validity is low (5% after 500 epochs)
The Rg and MMD match the reference well, but bond lengths are too spread (std 0.83 vs 0.06 Å).  
Likely causes: flow matching may need a higher LR, more epochs, or the `coord_scale` (currently 5.0) may be suboptimal relative to what DDPM was tuned for.  
**Next steps:** see stretch goals in TODO.md.

---

## Not yet run at scale

- MLP full training run
- Physics regularization hook in `training_loss()` (parameter exists, not wired up)
- DDIM/ODE step count speed-quality experiments
- FlowMatch with improved hyperparameters (see TODO.md)

---

## Run history

> Add entries here after every training run. Include: model, config, hardware, epochs, best val loss, key metrics.

### Template
```
**Date:** YYYY-MM-DD  
**Model:** transformer | egnn | mlp | flowmatch  
**Config:** configs/...yaml  
**Hardware:** GPU model  
**Epochs run:** N / target N  
**Best val loss:** X.XXXX (epoch N)  
**Checkpoint:** checkpoints/.../best.pt  
**Key metrics:** bond validity X%, Rg mean=X.XX Å, MMD=X.XXXX  
**Notes:** anything notable
```

---

**Date:** 2026-06-01  
**Model:** egnn (DDPM)  
**Config:** configs/egnn.yaml (pshah — cluster paths → relative, n_epochs 200→500)  
**Hardware:** GPU (workspace)  
**Epochs run:** 500 / 500  
**Best val loss:** 0.1779 (epoch 460)  
**Checkpoint:** checkpoints/egnn/v1/best.pt  
**Key metrics:** bond validity 17.8%, Rg mean=5.79 Å, MMD=0.0327  
**Notes:** Originally landed in `checkpoints/v2/`; reorganised to `checkpoints/egnn/v1/`. Owner: pshah.

---

**Date:** 2026-06-01  
**Model:** transformer (DDPM) — BAD RUN  
**Config:** configs/baseline.yaml (batch=2048)  
**Hardware:** GPU (workspace)  
**Epochs run:** 500 / 500  
**Best val loss:** 0.7495 (epoch ~230)  
**Checkpoint:** checkpoints/baseline/v2/best.pt  
**Notes:** batch_size=2048 caused poor convergence. Do not use. Owner: tkreusel.

---

**Date:** 2026-06-01  
**Model:** transformer (DDPM)  
**Config:** configs/baseline.yaml (batch=512)  
**Hardware:** GPU (workspace)  
**Epochs run:** 500 / 500  
**Best val loss:** 0.0891 (epoch 500)  
**Checkpoint:** checkpoints/baseline/v3/best.pt  
**Key metrics:** bond validity 47.2%, Rg mean=4.93 Å (low), MMD=0.0618  
**Notes:** batch_size=512. Converged cleanly but Rg and end-to-end distance are below reference — structures are too compact. Owner: tkreusel.

---

**Date:** 2026-06-01  
**Model:** flowmatch (ZeroCoMFlowMatching + EGNN, SE(3)-equivariant)  
**Config:** configs/flowmatch.yaml (batch=512, sigma_min=1e-4, EGNN hidden_dim=128, n_layers=5)  
**Hardware:** GPU (L40S, workspace)  
**Epochs run:** 500 / 500  
**Best val loss:** 0.6351 (epoch 480)  
**Checkpoint:** checkpoints/flowmatch/v2/best.pt  
**Key metrics:** bond validity 5.0%, Rg mean=5.96 Å ✓, end-to-end=12.4 Å ✓, MMD=0.0323 ✓  
**Notes:** Loss scale not comparable to DDPM (velocity vs noise target). Rg/MMD are the best of all three models — global structure is right. Bond geometry needs work (see known issues). Owner: marik.

---

## Three-way comparison (500 samples, 100 steps, 2026-06-01)

```
python scripts/evaluate.py \
    --ckpt checkpoints/flowmatch/v2/best.pt \
    --ckpt_ref checkpoints/baseline/v3/best.pt checkpoints/egnn/v1/best.pt \
    --labels FlowMatch-EGNN Transformer-DDPM EGNN-DDPM \
    --test data/test.npz --n 500 --save plots/comparison_all.png
```

| Metric | FlowMatch-EGNN | Transformer-DDPM | EGNN-DDPM | Reference |
|--------|---------------|-----------------|-----------|-----------|
| Bond validity | 5.0% | 47.2% | 17.8% | 100% |
| Bond length mean (Å) | 4.24 | 3.78 | 3.59 | 3.83 |
| Bond length std (Å) | 0.83 | 0.28 | 0.44 | 0.06 |
| Rg mean (Å) | **5.96** | 4.93 | 5.79 | 5.92 |
| Rg std (Å) | **1.09** | 0.32 | 1.13 | 1.11 |
| End-to-end mean (Å) | **12.4** | 6.9 | 11.5 | 12.0 |
| MMD ↓ | **0.0323** | 0.0618 | 0.0327 | — |

FlowMatch-EGNN has the best global structure (Rg, end-to-end, MMD) despite poor bond validity. Transformer-DDPM has best bond validity but worst Rg/MMD. EGNN-DDPM is the best balanced model so far.

---

## Git history

| Commit | Message | What changed |
|--------|---------|-------------|
| (this commit) | Add SE(3)-equivariant flow matching + collab workspace + evaluation improvements | See below |
| 5f43700 | nothing | transforms.py gitignore fix |
| 5b23581 | Added EGNN model | `egnn.py`, `train_egnn.py`, `diffusion_zerocom.py`, `transforms.py`, `egnn*.yaml`, `evaluate.py` dual-ckpt |
| 6df1d89 | add requirements.txt | `requirements.txt` |
| 5d2ae54 | fixed stuff, added noise scaling by std | noise scaling bug fix in training |
| 6e91e9d | initial commit | all base files |

**This commit includes:**
- `models/flow_matching.py` — OT-CFM: `ContinuousFlowMatching` + `ZeroCoMFlowMatching` (Heun's ODE)
- `scripts/train_flow.py` — GPU-optimized training (AMP, torch.compile) for flow matching
- `configs/flowmatch_local.yaml` + `configs/flowmatch.yaml` — CPU test and full GPU configs
- `scripts/evaluate.py` — expanded to N-model comparison (`--ckpt_ref` accepts multiple paths), `flowmatch` branch, `_orig_mod.` prefix fix for torch.compile
- `scripts/quick_sample.py` — `flowmatch` branch + `_orig_mod.` prefix fix
- `COLLAB/` — full collaborative workspace (README, SETUP, TODO, STATUS, WORKFLOW)
- `CLAUDE.md` — AI agent context with all commands
- `environment.yml` — reproducible conda environment
- `configs/egnn.yaml` — cluster paths → relative, n_epochs 200→500
- `configs/baseline.yaml` — val_path fixed (`valid.npz` → `val.npz`)
- `.gitignore` — `configs/*personal*` pattern added
