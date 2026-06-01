# Environment Setup

Full setup guide for new team members. After completing this, you should be able to run a full local test in under 2 minutes.

---

## 1. Clone the repo

```bash
git clone https://github.com/tkreusel/deeplife.git
cd deeplife
```

---

## 2. Create the conda environment

```bash
conda env create -f environment.yml
conda activate deeplife
```

This installs Python 3.11 + all required packages (torch, numpy, matplotlib, PyYAML, tqdm).

> If you don't have conda/miniforge yet: https://github.com/conda-forge/miniforge

---

## 3. GPU PyTorch (skip for CPU-only work)

The `environment.yml` installs a CPU build of PyTorch by default. For GPU training on the cluster (CUDA 12.4):

```bash
conda activate deeplife
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

Check CUDA availability:
```bash
python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
```

Expected output on A100: `True 12.4`

---

## 4. Get the data files

Data is **not in the repo** (`.npz` files are gitignored). You need three files:

```
data/
  train.npz
  valid.npz
  test.npz
```

**Source on EMBL cluster:**
```
/g/korbel2/shahp/deeplife/data/train.npz
/g/korbel2/shahp/deeplife/data/val.npz     ← note: cluster uses "val", local configs use "valid"
/g/korbel2/shahp/deeplife/data/test.npz
```

Copy to your machine:
```bash
mkdir -p data
scp username@cluster.embl.de:/g/korbel2/shahp/deeplife/data/*.npz data/
# Rename if needed:
mv data/val.npz data/valid.npz
```

Or ask a teammate to share via SFTP / shared network drive.

**Filename note:** `configs/baseline.yaml` references `data/val.npz`; all other configs use `data/valid.npz`. Easiest fix — keep both:
```bash
cp data/valid.npz data/val.npz
```

---

## 5. Verify the install

Run the local (CPU, fast) configs to confirm everything works:

```bash
# Transformer baseline — should complete in ~30 seconds
python scripts/train.py --config configs/local_baseline.yaml

# EGNN — should complete in ~30 seconds
python scripts/train_egnn.py --config configs/egnn_local.yaml

# Sample from the EGNN checkpoint
python scripts/quick_sample.py \
  --checkpoint checkpoints/egnn_local/v1/best.pt \
  --n 10
```

Look for output like:
```
Bond length validity: X/10 structures valid
Radius of gyration:   mean=X.XX Å
```

If you see these, setup is complete.

---

## 6. Set up a personal production config

`configs/egnn.yaml` has hardcoded absolute paths from the original developer's cluster environment. **Do not edit that file** — it would break for everyone else.

Instead, create your own copy:

```bash
cp configs/egnn.yaml configs/egnn_personal.yaml
```

Edit `configs/egnn_personal.yaml` to update the paths:
```yaml
data:
  train_path: "data/train.npz"     # relative from deeplife/
  val_path:   "data/valid.npz"
  test_path:  "data/test.npz"

paths:
  checkpoint_dir: "checkpoints/egnn"   # relative, will be created automatically
```

Personal configs are gitignored via the `*personal*` pattern — you need to add this to `.gitignore` if it isn't there:
```bash
echo "configs/*personal*" >> .gitignore
```

Then train with:
```bash
python scripts/train_egnn.py --config configs/egnn_personal.yaml
```

---

## 7. IDE setup

`.vscode/`, `.idea/`, and similar IDE directories are gitignored — you can configure your editor however you like without polluting the repo.

Recommended VSCode extensions: Python (ms-python), Pylance, GitLens.

---

## What's gitignored (do not try to commit)

| Pattern | What it covers |
|---------|---------------|
| `*.npz`, `*.npy` | Data files |
| `checkpoints/`, `*.pt` | Model checkpoints |
| `plots/`, `samples/`, `*.pdb` | Generated outputs |
| `wandb/`, `mlruns/` | Experiment tracking |
| `.vscode/`, `.idea/` | IDE configs |

Full list in [../.gitignore](../.gitignore).
