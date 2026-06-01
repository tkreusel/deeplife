# Workflow & Conventions

How the team works together. Follow these to avoid conflicts and duplicated work.

---

## Before starting any session

```bash
git pull                   # always get latest from teammates first
cat COLLAB/STATUS.md       # see current state + known issues
cat COLLAB/TODO.md         # pick a task or check nothing is already in progress
```

---

## Config file convention

| Pattern | Purpose | Paths | Safe to commit? |
|---------|---------|-------|----------------|
| `*_local.yaml` | CPU debug run (small batch, few epochs) | Relative | Yes |
| `baseline.yaml` | Production Transformer | Relative | Yes |
| `egnn.yaml` | Production EGNN (EMBL cluster original) | **Absolute cluster paths** | Yes (don't edit) |
| `*_personal.yaml` | Your machine's production config | Your paths | **No — gitignored** |

**Rule: Never commit absolute paths into shared configs.** Create `configs/egnn_personal.yaml` for your own machine (gitignored). `egnn.yaml` stays as the original cluster config so the original developer can use it unchanged.

Make sure `configs/*personal*` is in `.gitignore`:
```bash
grep "personal" .gitignore || echo "configs/*personal*" >> .gitignore
```

---

## Starting a new training run

1. Choose or create a config file.
2. Run training — checkpoints auto-version to `v1/`, `v2/`, etc:
   ```bash
   python scripts/train_egnn.py --config configs/egnn_personal.yaml
   ```
3. Note the checkpoint path when it finishes.
4. Evaluate and add the result to [STATUS.md](STATUS.md) under "Run history".

**Never rename checkpoint directories manually** — the versioning is automatic and the scripts track the latest version.

---

## Evaluation after a run

Always evaluate with the EMA checkpoint (`best.pt`):

```bash
python scripts/evaluate.py \
  --ckpt checkpoints/egnn/v1/best.pt \
  --test data/test.npz \
  --n 1000 \
  --save plots/egnn_v1_eval.png
```

Record the output metrics in [STATUS.md](STATUS.md) (bond validity %, Rg mean, MMD).

---

## Git workflow

**Branch strategy:** We work on a single `main` branch (3 people is small enough). No feature branches unless a change is large and experimental.

**Commit messages:** Be descriptive. Not "stuff" — say what you actually changed:
```
# Good
Add WandB logging to train_egnn.py
Fix val/valid filename inconsistency in baseline.yaml
Add mlp_baseline config

# Bad
stuff
fix
update
```

**What to commit:**
- Code changes (`models/`, `scripts/`, `data/`)
- Config changes (`configs/`) — but never absolute personal paths
- This `COLLAB/` folder — keep docs up to date
- `environment.yml`, `requirements.txt`, `.gitignore`

**What NOT to commit** (all gitignored):
- `checkpoints/`, `*.pt`, `*.pth` — model weights
- `*.npz`, `*.npy` — data files
- `plots/`, `samples/`, `*.pdb` — generated outputs
- `wandb/`, `mlruns/` — experiment tracking
- `configs/*personal*` — personal machine configs
- `.vscode/`, `.idea/` — IDE files

**Before pushing:**
```bash
git status        # verify no unwanted files staged
git diff --cached # review what you're committing
git push
```

---

## AI agent workflow (Claude Code)

When starting a Claude Code session in this repo:

1. Claude Code auto-reads `CLAUDE.md` at startup — it has the full project context.
2. Ask Claude to read `COLLAB/STATUS.md` and `COLLAB/TODO.md` before starting new work, to avoid redoing what's already done.
3. After Claude finishes a task, update the COLLAB docs manually (or ask Claude to do it).

Claude Code should **not** commit or push automatically — always review diffs before committing.

---

## Experiment naming convention

Checkpoint directories are auto-named by the training scripts using `experiment_name` from the config + auto-incrementing version:
```
checkpoints/
  chignolin_baseline/v1/    ← experiment_name="chignolin_baseline", first run
  chignolin_baseline/v2/    ← same config, second run (e.g. after a resume)
  chignolin_egnn/v1/
```

Each versioned directory contains:
```
v1/
  best.pt       ← lowest val loss checkpoint (EMA weights) — USE THIS for generation
  latest.pt     ← most recent epoch checkpoint — USE THIS to resume
  log.jsonl     ← one JSON line per epoch: train_loss, val_loss, lr, epoch
```

---

## Updating these docs

**After a training run:** Add a row to the "Run history" table in [STATUS.md](STATUS.md).  
**After fixing a bug:** Add a note to the relevant Known Issue in [STATUS.md](STATUS.md) or remove it.  
**After completing a task:** Check it off in [TODO.md](TODO.md) and add a note.  
**When discovering a new issue:** Add it to [STATUS.md](STATUS.md) under Known Issues.

Treat `COLLAB/` like a shared lab notebook — brief but accurate.
