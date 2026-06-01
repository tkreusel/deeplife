# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Diffusion model framework for generating Chignolin protein Cα conformations. Three architectures (MLP, Transformer, EGNN) trained with DDPM/DDIM. Team of 3 — see [COLLAB/README.md](COLLAB/README.md) for human onboarding docs.

**Repo:** https://github.com/tkreusel/deeplife

---

## Commands

### Sanity check (CPU, ~1 min)
```bash
python scripts/train.py --config configs/local_baseline.yaml
python scripts/train_egnn.py --config configs/egnn_local.yaml
python scripts/train_flow.py --config configs/flowmatch_local.yaml
python scripts/train_flow.py --config configs/flowmatch_physics_local.yaml   # with physics constraints
```

### Full training (GPU)
```bash
python scripts/train.py --config configs/baseline.yaml
python scripts/train_egnn.py --config configs/egnn_personal.yaml  # personal config with local paths
python scripts/train.py --config configs/baseline.yaml --resume checkpoints/baseline/v1/latest.pt
python scripts/train_flow.py --config configs/flowmatch.yaml             # SE(3)-equivariant flow matching
python scripts/train_flow.py --config configs/flowmatch_physics.yaml     # + bond/angle/clash physics loss
python scripts/train_flow.py --config configs/flowmatch.yaml --resume checkpoints/flowmatch/v1/epoch_0050.pt
```

### Evaluation & sampling
```bash
python scripts/quick_sample.py --checkpoint checkpoints/egnn_local/v1/best.pt --n 50 --steps 100
python scripts/evaluate.py --ckpt checkpoints/egnn/v1/best.pt --test data/test.npz --n 1000 --save plots/eval.png
# Compare two models side-by-side:
python scripts/evaluate.py --ckpt checkpoints/egnn/v1/best.pt --ckpt_ref checkpoints/baseline/v1/best.pt --test data/test.npz --n 500
python scripts/plot_training.py --logs checkpoints/egnn/v1/log.jsonl --save plots/curves.png
# Flow matching — same sample/eval scripts, auto-detected via model_type in checkpoint:
python scripts/quick_sample.py --checkpoint checkpoints/flowmatch/v1/best.pt --n 50 --steps 100
python scripts/evaluate.py --ckpt checkpoints/flowmatch/v1/best.pt --test data/test.npz --n 1000
```

---

## Architecture

The diffusion pipeline is: `ChignolinDataset` → `GaussianDiffusion` wrapping a score network → training script.

**Score networks** (`models/`) all share the same interface — they predict noise ε: `(B, N, 3)` from noisy coords `x_t: (B, N, 3)` and timestep `t: (B,)`:
- `baseline.py` — `MLPScoreNetwork` (flattens coords, 4-layer MLP) and `TransformerScoreNetwork` (residues as tokens, self-attention)
- `egnn.py` — `EGNNScoreNetwork`: SE(3)-equivariant via coordinate updates as weighted sums of difference vectors `(xᵢ - xⱼ)`; has `equivariance_check()` method

**Diffusion** (`models/`):
- `diffusion.py` — `GaussianDiffusion`: standard DDPM. Core methods: `q_sample()` (forward), `training_loss()`, `p_sample()` (one reverse step), `sample()` (full DDPM), `ddim_sample()` (faster, fewer steps)
- `diffusion_zerocom.py` — `ZeroCoMGaussianDiffusion`: subclasses the above; projects all noise onto zero-center-of-mass subspace so generated structures are inherently centered. Used by `train_egnn.py`.

**Configs** map directly to these: `model_type: mlp|transformer|egnn` selects the score network; `diffusion.T` and `diffusion.schedule` configure the process.

**Checkpoints** store `config`, `model`, `ema_shadow`, `optimizer`, `scheduler`, `epoch`, `best_val_loss`. Scripts auto-detect `model_type` from the saved config. Training auto-versions checkpoint dirs (`v1/`, `v2/`, …) and writes `log.jsonl` (one JSON line per epoch).

---

## Critical gotchas

**`configs/egnn.yaml` has hardcoded EMBL cluster paths.** All `data.*_path` and `paths.checkpoint_dir` fields point to `/g/korbel2/shahp/deeplife/...`. Use `egnn_local.yaml` for local work; for full-scale runs create `configs/egnn_personal.yaml` with relative paths (gitignored via `configs/*personal*`).

**Data files are not in the repo.** `*.npz` is gitignored. Source on cluster: `/g/korbel2/shahp/deeplife/data/`. The actual filename on disk is `val.npz` but local configs reference `valid.npz` — run `cp data/val.npz data/valid.npz` to fix.

**Always use `best.pt` for generation** (EMA weights, best val loss), not `latest.pt` (last epoch, raw weights).

**`scripts/tmp.py`** is a one-off debugging script (prints coordinate std). Ignore it.

**Gitignored — do not commit:** `checkpoints/`, `*.pt`, `*.npz`, `*.npy`, `plots/`, `samples/`, `*.pdb`.

---

## Before starting work

```bash
git pull
# Then read:
# COLLAB/STATUS.md  — known issues and run history
# COLLAB/TODO.md    — open tasks (avoid duplicating effort)
```

After finishing: update `COLLAB/STATUS.md` with run results and `COLLAB/TODO.md` with task status.
