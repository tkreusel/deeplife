# deeplife — Team Quick Start

Generative diffusion model for Chignolin protein conformation sampling. We compare three architectures (MLP, Transformer, EGNN) trained with DDPM on a dataset of Cα coordinates from the 10-residue Chignolin miniprotein.

**GitHub:** https://github.com/tkreusel/deeplife  
**Team:** Marik, tkreusel + one more  

---

## Prerequisites

- [Miniforge / Conda](https://github.com/conda-forge/miniforge) installed
- GPU with CUDA 12.4 recommended for full training (CPU fine for local tests)
- Access to the data files — see [SETUP.md](SETUP.md)

---

## Setup (4 steps)

```bash
# 1. Clone
git clone https://github.com/tkreusel/deeplife.git
cd deeplife

# 2. Create conda environment
conda env create -f environment.yml
conda activate deeplife

# 3. GPU PyTorch (skip if CPU-only)
pip install torch --index-url https://download.pytorch.org/whl/cu124

# 4. Get data files (NOT in the repo — see SETUP.md for source)
#    Place train.npz, valid.npz, test.npz inside data/
```

Full details and troubleshooting in [SETUP.md](SETUP.md).

---

## Verify your install (< 1 minute, CPU)

```bash
# Test Transformer baseline
python scripts/train.py --config configs/local_baseline.yaml

# Test EGNN
python scripts/train_egnn.py --config configs/egnn_local.yaml

# Quick sample (needs a checkpoint from the above runs)
python scripts/quick_sample.py \
  --checkpoint checkpoints/egnn_local/v1/best.pt --n 10
```

If both training runs complete without errors, you're good to go.

---

## Important: things that trip people up

| Issue | Fix |
|-------|-----|
| `configs/egnn.yaml` fails with file not found | That config has hardcoded cluster paths. Use `egnn_local.yaml` locally. |
| `data/train.npz` not found | Data is gitignored. Copy from cluster or ask a teammate. See [SETUP.md](SETUP.md). |
| `val.npz` vs `valid.npz` mismatch | `baseline.yaml` uses `val.npz`; local configs use `valid.npz`. Rename your file to match whichever config you're using. |
| Wrong model loaded | Always use `best.pt` (EMA weights) for generation — not `latest.pt`. |
| Committed a `.pt` file | `checkpoints/` and `*.pt` are gitignored on purpose — remove with `git rm --cached`. |

---

## Key docs

| File | Purpose |
|------|---------|
| [SETUP.md](SETUP.md) | Full environment setup, data, personal configs |
| [WORKFLOW.md](WORKFLOW.md) | How we work: configs, commits, experiment logging |
| [TODO.md](TODO.md) | Shared task list — check here before starting new work |
| [STATUS.md](STATUS.md) | What works, known issues, run history |
| [../CLAUDE.md](../CLAUDE.md) | AI agent context (auto-read by Claude Code) |

---

## Quick command reference

```bash
# Train (local test)
python scripts/train.py --config configs/local_baseline.yaml
python scripts/train_egnn.py --config configs/egnn_local.yaml

# Train (production — GPU cluster)
python scripts/train.py --config configs/baseline.yaml
python scripts/train_egnn.py --config configs/egnn_personal.yaml  # your copy of egnn.yaml

# Resume a run
python scripts/train.py --config configs/baseline.yaml \
  --resume checkpoints/baseline/v1/latest.pt

# Sample & analyze
python scripts/quick_sample.py \
  --checkpoint checkpoints/egnn/v1/best.pt --n 50 --steps 100

# Full evaluation
python scripts/evaluate.py \
  --ckpt checkpoints/egnn/v1/best.pt \
  --test data/test.npz --n 1000 --save plots/eval.png

# Compare two models
python scripts/evaluate.py \
  --ckpt checkpoints/egnn/v1/best.pt \
  --ckpt_ref checkpoints/baseline/v1/best.pt \
  --test data/test.npz --n 500 --save plots/comparison.png

# Plot training curves
python scripts/plot_training.py \
  --logs checkpoints/egnn/v1/log.jsonl --save plots/egnn_curves.png
```
