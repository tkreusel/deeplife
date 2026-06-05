# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Diffusion model framework for generating Chignolin protein Cα conformations. Three architectures (MLP, Transformer, EGNN) trained with DDPM/DDIM, plus SE(3)-equivariant flow matching. Team of 3 — see [COLLAB/README.md](COLLAB/README.md) for human onboarding docs.

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

### SE(3) equivariance analysis
```bash
# Quick smoke-test on a single model (CPU, ~1 min):
python scripts/check_equivariance.py --ckpt checkpoints/flowmatch/v2/best.pt \
    --n_noise 3 --n_rotations 5 --n_generate 30 --steps 10

# Full comparison — equivariant vs non-equivariant models (+ 3-panel plot):
python scripts/check_equivariance.py \
    --ckpt     checkpoints/flowmatch_physics/v3/best.pt \
    --ckpt_ref checkpoints/egnn/v1/best.pt checkpoints/baseline/v3/best.pt \
    --labels   FlowMatch+Physics EGNN-DDPM Transformer \
    --n_noise 20 --n_rotations 50 --n_generate 500 --steps 50 \
    --save plots/equivariance_comparison.png
```

---

## Architecture

The diffusion pipeline is: `ChignolinDataset` → `GaussianDiffusion` wrapping a score network → training script.

**Score networks** (`models/`) all share the same interface — they predict noise ε: `(B, N, 3)` from noisy coords `x_t: (B, N, 3)` and timestep `t: (B,)`:
- `baseline.py` — `MLPScoreNetwork` (flattens coords, 4-layer MLP) and `TransformerScoreNetwork` (residues as tokens, self-attention)
- `egnn.py` — `EGNNScoreNetwork`: SE(3)-equivariant via coordinate updates as weighted sums of difference vectors `(xᵢ - xⱼ)`; has `check_equivariance()` method

**Diffusion** (`models/`):
- `diffusion.py` — `GaussianDiffusion`: standard DDPM. Core methods: `q_sample()` (forward), `training_loss()`, `p_sample()` (one reverse step), `sample()` (full DDPM), `ddim_sample()` (faster, fewer steps)
- `diffusion_zerocom.py` — `ZeroCoMGaussianDiffusion`: subclasses the above; projects all noise onto zero-center-of-mass subspace so generated structures are inherently centered. Used by `train_egnn.py`.
- `flow_matching.py` — `ZeroCoMFlowMatching`: OT-CFM on the zero-CoM subspace; Heun's ODE sampler. Used by `train_flow.py`.
- `physics.py` — `ChignolinPhysics`: bond-length MSE, clash repulsion, virtual bond-angle Huber loss; wired into `train_flow.py` via `physics_weight`.

**Configs** map directly to these: `model_type: mlp|transformer|egnn` selects the score network; `diffusion.T` and `diffusion.schedule` configure the process.

**Checkpoints** store `config`, `model`, `ema_shadow`, `optimizer`, `scheduler`, `epoch`, `best_val_loss`. Scripts auto-detect `model_type` from the saved config. Training auto-versions checkpoint dirs (`v1/`, `v2/`, …) and writes `log.jsonl` (one JSON line per epoch).

---

## Critical gotchas

**`configs/egnn.yaml` has hardcoded EMBL cluster paths.** All `data.*_path` and `paths.checkpoint_dir` fields point to `/g/korbel2/shahp/deeplife/...`. Use `egnn_local.yaml` for local work; for full-scale runs create `configs/egnn_personal.yaml` with relative paths (gitignored via `configs/*personal*`).

**Data files are not in the repo.** `*.npz` is gitignored. Source on cluster: `/g/korbel2/shahp/deeplife/data/`. The actual filename on disk is `val.npz` but local configs reference `valid.npz` — run `cp data/val.npz data/valid.npz` to fix.

**Always use `best.pt` for generation** (EMA weights, best val loss), not `latest.pt` (last epoch, raw weights).

**Gitignored — do not commit:** `checkpoints/`, `*.pt`, `*.npz`, `*.npy`, `plots/`, `samples/`, `*.pdb`.

---

## Session protocol (automatic)

**At the start of every session**, `COLLAB/STATUS.md` and `COLLAB/TODO.md` are automatically injected into your context via a `UserPromptSubmit` hook. You do not need to read them manually — act on their contents immediately (check for open blockers, ongoing work, known issues).

**After every significant action** (training run, bug fix, new script, evaluation), you MUST update these files before the conversation ends:
- `COLLAB/STATUS.md` — add a run-history entry or update the known-issues section
- `COLLAB/TODO.md` — mark completed tasks `[x]`, claim in-progress tasks with your name + date, add new tasks
- `MODEL_REGISTRY.yaml` + `MODEL_REGISTRY.md` — add or update the model entry whenever a training run completes or is meaningfully advanced (≥100 epochs); fill in `eval_metrics` after running `evaluate.py`

Do not wait to be asked. These files are how the team stays in sync across sessions and across teammates.

### Model registry update rules

**Add a new entry** when:
- A new training run starts (status: `in_progress`, fill in architecture + target_epochs)
- A run completes or is cancelled after ≥100 epochs (update epochs_run, best_val_loss, status)
- Evaluation metrics are computed (fill in eval_metrics block)

**Template** (copy to bottom of `MODEL_REGISTRY.yaml` models list):
```yaml
  - id: <family>/<version>
    name: <ShortDescriptiveName>
    family: <checkpoint_dir_name>
    version: <v1|v2|...>
    model_type: <from config>
    framework: <ddpm|flow_match|torsion_flow|backbone_ipa>
    data: <ca_only|all_atom|backbone|backbone_torsion|torsion_ca>
    description: >
      One sentence describing what is new/different about this run.
    architecture:
      hidden_dim: <int>
      n_layers: <int>
    training:
      epochs_run: <int>
      target_epochs: <int>
      best_val_loss: <float>
    checkpoint: checkpoints/<family>/<version>/best.pt
    eval_metrics: null   # fill in after evaluate.py
    status: <production|partial|smoke_test|failed|in_progress>
    notes: ""
```

Add a matching row to `MODEL_REGISTRY.md` using the row template at the bottom of that file.
