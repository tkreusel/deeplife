# Project Status

Current state of the codebase, known issues, and run history. **Update this after every significant training run or fix.**

Last updated: 2026-06-04 (backbone_ipa/v8 fully evaluated)

---

## backbone_ipa/v8 — Evaluated (2026-06-04)

`checkpoints/backbone_ipa/v8/best.pt` — epoch 3039, val_loss **0.5231** (vs v1's 0.5506). Still training.
Results appended to `plots/eval_overnight/backbone/metrics.json`.

### Results vs backbone group

| Metric | BackboneTransformer-v1 | BackboneIPA-v1 | **BackboneIPA-v8** | Reference |
|---|---|---|---|---|
| Bond RMSE | 0.069 Å | **0.000 Å** | **0.000 Å** | 0.063 Å |
| Bond valid ±0.2 | 85% | 100% | **100%** | ~98% |
| Clash rate | **0.0%** | 2.8% | **0.8%** | 0% |
| Rg mean | 5.90 Å | 6.11 Å | 6.16 Å | 5.92 Å |
| ETE mean | **11.7 Å** | 14.2 Å | 14.4 Å | 12.0 Å |
| MMD ↓ | 0.0356 | 0.0351 | **0.0345** | — |
| Rosetta pass | 30% | 90% | 75% | — |
| Rosetta score | +37.9 REU | −1.8 REU | **−5.8 REU** | — |
| NND ratio | 0.835 | 1.251 | 1.164 | — |
| Valid fraction | 100% | 96.8% | **99.2%** | — |
| Novelty precision | 0.657 | 0.269 | **0.402** | — |

### Key findings

- **Best Rosetta energy of any backbone model** (−5.8 REU mean). Extended training produced more stable force-field conformations than v1 (−1.8 REU).
- **Clash rate halved** vs v1 (0.8% vs 2.8%).
- **Best MMD** in the backbone group (0.0345).
- **ETE bias stuck at 14.4 Å** — unchanged from v1 (14.2 Å). This is a NeRF compounding ceiling shared across all IPA models; more training does not help.
- BackboneTransformer-v1 remains the best for ETE (11.7 Å ≈ ref 12.0 Å) but scores badly on Rosetta (30% pass, +37.9 REU).
- **Energy conditioning (τ→geometry): FAIL** — Rg/ETE do not respond monotonically to τ (r≈0.2). However τ-REU Rosetta response is present (Δ=21.4 REU from τ=0→1), meaning the temperature signal reaches the force-field even without visible geometry change.
- **Novelty precision improved**: 0.402 vs v1's 0.269 — the novel structures v8 generates are more plausible.

---

## eval_v2: τ-vs-REU Rosetta Validation (2026-06-04)

New metric added to Section 3 (Energy): **τ vs Rosetta Energy Units scatter plot** (`figure3b_tau_reu.png`).

For each energy-conditioned model, `n_tau_rosetta` (default 15) structures are scored with
PyRosetta at each τ value and mean REU is plotted against τ. A positive Spearman r confirms
the learned temperature signal transfers to actual force-field stability — τ=0 generates
genuinely more stable structures (lower REU), τ=1 generates genuinely more transient ones.

**New code:**
- `pyrosetta_utils.score_tau_samples()` — scores n_per_tau structures at each τ, returns `{tau: {mean, std, n, scores}}`
- `plotting.plot_tau_reu()` — 2-panel figure (mean ± std line + violin at τ_min/τ_max)
- `replot.replot_tau_reu()` — reads stored `tau_rosetta` from JSON, regenerates figure
- `scripts/run_tau_reu.sh` — standalone script to populate τ-rosetta data using `--sections energy` only (no other sections re-run)

**Run to populate data:**
```bash
bash scripts/run_tau_reu.sh    # ~50 min, all energy-conditioned models
```

Results will go into `plots/eval_overnight/{ca_only,backbone}/figure3b_tau_reu.png`.
`tau_rosetta` is stored as `sections.energy.<label>.tau_rosetta` in metrics.json.

---

## eval_v2 Overnight Run — All 24 Models (2026-06-04)

Full evaluation complete via `scripts/run_eval_all.py` + `scripts/eval_v2/main.py`.
Outputs: `plots/eval_overnight/{ca_only,backbone,all_atom}/` — all 5 sections, all figures.

### Key findings

**Physical quality tiers:**
- Exact by construction (internal coords): TorsionFlow-MLP, TorsionTransformer, BackboneIPA — 100% bond valid, RMSE=0
- Best Cartesian: AdaLN-Transformer-Energy-Physics-v1 (90.2%, RMSE=0.172Å, 2.8% clash), AdaLN-Transformer-SelfCond-v3 (90%, RMSE=0.178Å)
- Broken: SE3Flow-Energy-v1 (0% valid, RMSE=7.19Å — coordinate scale bug); all-atom models (100% clash)

**SE(3) equivariance confirmed:**
- EGNN (all): T1~3e-7 PASS, T2~0 PASS — machine-precision equivariance
- SE3Flow, FlowMatch-EGNN: T1~5–8e-7 PASS
- Transformers: T1=0.04–1.16 FAIL as expected; SE3aug helps significantly (1.16→0.09)
- Torsion models: isotropy ratio ~2.8 ANISOTROPIC (deterministic output, no orientation randomisation)

**Energy conditioning:**
- PASS (monotone Rg with τ): FlowMatch-Energy-Physics (r=1.0, 19.8Å range), FlowMatch-Energy-v2, SE3Flow-AdaLN-Velocity, AdaLN-SelfCond-v1/v2/v3, TorsionFlow, TorsionTransformer
- FAIL: AdaLN-Transformer-Energy-v1 (no response), EGNN-AdaLN-Energy-v1 (no response)
- INVERTED: AdaLN-Transformer-Energy-Physics-v1 (r=−0.90, higher τ → smaller Rg — wrong sign)

**PyRosetta biological validation:**
- BackboneIPA-v1: **60% structures pass Rosetta filter** (score=−0.3 REU, fa_rep=6.9) — most biologically plausible model
- BackboneTransformer-v1: 35% pass (score=33.9, fa_rep=28.3)
- All Cα models: 0% pass (fa_rep 600–1000 REU; NeRF reconstruction artifact, not model failure per se)
- All-atom models: 0% pass (omega 80–115 REU; backbone geometry catastrophically wrong)

**Novelty (physics-filtered NND ratio):**
- Most novel valid structures: EGNN-AdaLN-Energy-v1 (ratio=1.59), FlowMatch-Energy-Physics-v1 (1.55), AdaLN-SelfCond-v1 (1.51)
- Best validity × novelty: TorsionTransformer (98% valid, ratio=1.17), TorsionFlow-MLP (97%, 1.25)
- Conservative (stays near training data): BackboneTransformer-v1 (ratio=0.84)

### Open issues from this run
| Issue | Priority |
|---|---|
| SE3Flow-Energy-v1: BondRMSE=7.19Å (should be ~0.5Å) — coordinate scaling bug | High |
| SE3Flow-AllAtom-v2-v1: T4 Wasserstein=27.8 (anomalous) | Medium |
| AdaLN-Transformer-Energy-Physics-v1: temperature response inverted (r=−0.90) | Medium |
| All-atom models: 100% clash, catastrophic ω score — architecture needs rethink | Low |

---

## What works (verified in code)

| Component | Status | Notes |
|-----------|--------|-------|
| MLPScoreNetwork | Code complete | No full training run yet |
| TransformerScoreNetwork | Code complete | Full run — best val 0.0891 (`baseline/v3`) |
| EGNNScoreNetwork | Code complete | SE(3)-equivariant; `check_equivariance()` verified |
| GaussianDiffusion (DDPM) | Code complete | Cosine + linear schedules |
| ZeroCoMGaussianDiffusion | Code complete | Used by `train_egnn.py` |
| ContinuousFlowMatching | Code complete | OT-CFM; Euler ODE sampling; physics hook implemented |
| ZeroCoMFlowMatching | Code complete | Zero-CoM OT-CFM; Heun's ODE; physics t²-weighted; used by `train_flow.py` |
| ChignolinPhysics (`models/physics.py`) | Code complete | Bond MSE + clash repulsion + angle Huber; coord_scale-aware |
| DDIM sampling | Code complete | Configurable eta, fewer steps |
| SE(3) data augmentation | Code complete | `RandomSE3Transform` via QR decomp |
| Local test configs | Verified | `local_baseline.yaml`, `egnn_local.yaml`, `flowmatch_local.yaml` |
| EMA weight tracking | Code complete | All three training scripts |
| Epoch-versioned checkpoints | Code complete | Auto v1/v2/v3 dirs |
| `quick_sample.py` | Code complete | Handles mlp/transformer/egnn/flowmatch |
| `evaluate.py` | Code complete | N-model comparison; all model types incl. torsion_flow_energy, backbone_transformer; MMD, bond %, Rg |
| `plot_training.py` | Code complete | log.jsonl → multi-run overlay plots (linear + log + LR) |
| `check_equivariance.py` | Code complete | 3-test SE(3) equivariance analysis; all model types; optional plot |
| AMP training (EGNN + FlowMatch) | Code complete | GradScaler, scaler saved in ckpt |
| `torch.compile` (EGNN + FlowMatch) | Code complete | Optional, PyTorch ≥2.0; strips `_orig_mod.` prefix on load |
| AdaLNEnergyTransformerScoreNetwork | Code complete + trained | Best Cartesian model — 90% bond validity (`transformer_adaln_energy_physics/v1`) |
| AdaLNSCScoreNetwork | Code complete + trained | Self-conditioning Transformer; good bond validity but broken global structure |
| EGNNAdaLNScoreNetwork | Code complete + trained | EGNN + AdaLN + energy; 43.7% bond validity, 39.1% clash rate (no physics) |
| BackboneTransformerScoreNetwork | Code complete, smoke-tested | 30-token AdaLN Transformer on N-Cα-C backbone; smoke test converges |
| BackbonePhysics (`models/backbone_physics.py`) | Code complete | 29 real backbone bonds: N-Cα/Cα-C/C-N with distinct ideal lengths; clash + angle |
| `prepare_backbone_data.py` | Code complete | Bond-graph walk extracts N/Cα/C indices; handles PRO ring; saves `data_backbone/` |
| BackboneIPAFlowNet | Code complete, smoke-tested | IPA-style Transformer + torsion flow matching on (φ,ψ) backbone; 4M params; SE(3)-invariant |
| BackboneTorsionalFlowMatching | Code complete | OT-CFM on backbone (φ,ψ); Heun ODE; aux Cartesian loss; self-distillation |
| `backbone_internal_coords.py` | Code complete, all tests pass | Backbone NeRF (φ,ψ→30-atom); AlphaFold2 frames; SE(3) equivariance verified <1e-6 |

---

## Phase 5b: BackboneIPAFlow — IPA Transformer + backbone torsion flow (2026-06-03) — READY TO TRAIN

### Motivation
Higher ODE steps make TorsionTransformer WORSE (ETE 14.51 Å at 200 steps vs 14.14 at 100 steps). The ETE offset is NeRF error compounding + no 3D spatial context in attention. The IPA approach adds geometric features from backbone frames (AlphaFold2-style N-CA-C frames) to fix both.

### New files

| File | Description |
|------|-------------|
| `models/backbone_internal_coords.py` | Backbone NeRF (φ,ψ → 30-atom), AlphaFold2 frames, velocity scales, source params. All tests pass: roundtrip <1e-7 rad, bond error <1e-7 Å, SE(3) equivariance <1e-6 |
| `models/backbone_ipa_flow.py` | `BackboneIPAFlowNet`: 10-residue tokens, IPA geometric attention bias (RBF + local frame dir + seq-sep), AdaLN-Zero per layer, 4.06M params |
| `models/backbone_torsion_flow.py` | `BackboneTorsionalFlowMatching`: OT-CFM on (φ,ψ) torus, Heun ODE, aux Cartesian loss (ETE+Rg penalty), self-distillation |
| `scripts/train_backbone_ipa.py` | Training script: backbone data loading, torsion stats warmup, cart_weight + self_distill modes |
| `configs/backbone_ipa_energy.yaml` | GPU config: d_model=256, 8 heads, 6 layers, 2000 epochs, cart_weight=0.1, data-driven source |
| `configs/backbone_ipa_local.yaml` | CPU smoke test: d_model=64, 2 layers, 5 epochs |

### Key design decisions
- **SE(3)-invariant**: φ/ψ are invariant; IPA features are SE(3)-invariant; output velocities are SE(3)-invariant
- **Backbone frames** (N-CA-C): AlphaFold2 convention (e1=CA-N direction, e2=Gram-Schmidt, e3=cross)
- **IPA geometric bias**: `v_ij_local = R_i^T(x_CA_j - x_CA_i)` + RBF(dist) + seq-sep → per-head attention bias
- **AdaLN-Zero** per layer (cond_dim=96): replaces broadcast-add conditioning
- **Auxiliary Cartesian loss**: t²-weighted ETE + Rg penalty on predicted endpoint (cart_weight=0.1)
- **Self-distillation** (off by default): k-step EMA teacher rollout → student endpoint matching

### Smoke test verified (2026-06-03)
- 5 epochs, CPU, batch_size=32, loss 5.67→4.11 (train), 5.52→4.09 (val)
- Checkpoint: `checkpoints/backbone_ipa_local/v1/best.pt`
- SE(3) invariance errors: roundtrip <1e-7 rad, frames <1e-6, IPA features 0.00

### Run commands
```bash
# CPU smoke test (~8 min):
python scripts/train_backbone_ipa.py --config configs/backbone_ipa_local.yaml

# Full GPU training:
python scripts/train_backbone_ipa.py --config configs/backbone_ipa_energy.yaml

# Evaluate:
python scripts/evaluate.py \
    --ckpt checkpoints/backbone_ipa/v1/best.pt \
    --test data/test.npz --n 1000 --steps 100 \
    --save plots/backbone_ipa_eval.png

# Compare to TorsionTransformer:
python scripts/evaluate.py \
    --ckpt     checkpoints/backbone_ipa/v1/best.pt \
    --ckpt_ref checkpoints/torsion_transformer/v1/best.pt \
    --labels   BackboneIPA TorsionTransformer-v1 \
    --test data/test.npz --n 1000 --steps 100 \
    --save plots/backbone_ipa_vs_transformer.png

# Self-distillation fine-tune (only after base model converged):
# Set flow.self_distill: true in config, then:
python scripts/train_backbone_ipa.py --config configs/backbone_ipa_energy.yaml \
    --resume checkpoints/backbone_ipa/v1/best.pt
```

### Expected improvements vs TorsionTransformer
| Metric | TorsionTransformer | BackboneIPA (target) |
|--------|-------------------|---------------------|
| ETE mean | 14.14 Å | 12.0–13.0 Å |
| Rg mean | 6.18 Å | 5.75–6.05 Å |
| val_loss | 0.7159 | < 0.700 |
| Bond validity | 100% | 100% |

---

## Phase 5: BackboneTransformer — N–Cα–C rigid frame model (2026-06-03) — TRAINING

### Motivation
Every Cartesian model has a diversity ceiling (~2× ref) and transformer models are systematically compact (ETE 10–11 Å vs ref 12 Å). TorsionTransformer has 100% bond validity but ETE +20% too high from virtual Cα-angle entanglement. Backbone frames (N, Cα, C per residue = 30 atoms) directly encode real (φ, ψ) angles — the true backbone DOF — without introducing the entanglement of virtual Cα-Cα bonds.

### What was built

| File | Description |
|------|-------------|
| `scripts/prepare_backbone_data.py` | One-time extraction: bond-graph walk finds N, Cα, C per residue from `data_all_atom/`, handles PRO ring (N bonds both CA and CD; CA selected by having most bonds), saves `data_backbone/{train,val,test}.npz` (30-atom, (N,30,3)) |
| `models/backbone_physics.py` | `BackbonePhysics`: 29-bond MSE with 3 distinct targets: N–Cα=1.460 Å (×10), Cα–C=1.525 Å (×10), C–N=1.329 Å (×9); clash repulsion; bond-angle Huber; coord_scale-aware |
| `models/backbone_transformer.py` | `BackboneTransformerScoreNetwork`: 30-token AdaLN Transformer; each token = coord_proj + atom_type_embed(N/CA/C) + residue_embed(0–9); same AdaLN-Zero + energy CFG as winning Cα model |
| `scripts/train_backbone.py` | Training script, mirrors `train_adaln_energy.py` |
| `configs/backbone_transformer.yaml` | Full GPU: hidden=256, 6 layers, 8 heads, 500 epochs, physics_weight=0.05, coord_scale=3.42, bond_weight=2.0 |
| `configs/backbone_transformer_local.yaml` | CPU/smoke-test: hidden=64, 2 layers, 5 epochs |
| `scripts/evaluate.py` | Added `backbone_transformer` branch in loader; `compute_backbone_metrics()`; backbone routing in `generate()` and `compute_all_metrics()`; `print_table()` backbone display |

### Data
`data_backbone/train.npz`: (79632, 30, 3) — coord_scale=3.42 Å (std of centred backbone coords). Backbone indices: `[0,1,10, 12,13,22, 24,25,30, 32,34,37, 39,40,46, 48,49,53, 55,56,57, 59,60,64, 66,67,78, 83,84,80]`.

### Smoke test result (5 epochs, tiny model, GPU)
- Loss: 0.179 → 0.088 (train), 0.173 → 0.088 (val) — clean convergence ✓
- Physics: BackbonePhysics wired, no errors ✓
- Checkpoint: `checkpoints/backbone_transformer_local/v1/best.pt`

### Key differences vs Cα model
- 30 tokens instead of 10; atom_type_embed distinguishes N/CA/C within each residue
- Three bond targets instead of one (N–Cα/Cα–C/C–N vs just 3.832 Å)
- bond_weight=2.0 (real backbone bonds are 7× more constrained than virtual Cα–Cα)
- coord_scale=3.42 Å (tighter than Cα's 5.0 Å — same molecular extent, more atoms)

### Training command
```bash
python scripts/train_backbone.py --config configs/backbone_transformer.yaml
# Checkpoint: checkpoints/backbone_transformer/v1/best.pt
```

### Evaluation (after training)
```bash
python scripts/evaluate.py \
    --ckpt checkpoints/backbone_transformer/v1/best.pt \
    --ckpt_ref checkpoints/transformer_adaln_energy_physics/v1/best.pt \
    --labels "BackboneTransformer" "AdaLN+E+Physics-Cα" \
    --test data_backbone/test.npz \
    --n 1000 --steps 100 --save plots/backbone_vs_ca_transformer.png
```

### Expected improvements
- Bond validity: 100% by construction for all 29 bonds (N–Cα, Cα–C, C–N all fixed)
- ETE bias: should reduce — model operates directly in (φ,ψ) space, no virtual-angle entanglement
- Diversity: unknown; may be better calibrated via Ramachandran basin structure

---

## Grand Comparison + Ablation Study (2026-06-03)

### evaluate.py bug fix: se3flow_adaln_velocity checkpoint loading
`se3flow_adaln_velocity/v1` was trained when `cond_emb` was removed from `phi_e` (the AdaLN campaign), causing a weight-shape mismatch with the current code (phi_e input 404 vs 500). Fixed by:
- `SE3FlowLayer.__init__`: new `cond_in_phi_e: bool = True` param; conditionally omits cond_emb from edge concat
- `SE3FlowEnergyNet.__init__`: passes `cond_in_phi_e` to all layers
- `evaluate.py` loader: auto-detects from checkpoint `ema_shadow['(_orig_mod.)layers.0.phi_e.0.weight'].shape[1]` — if equal to `node_dim*2 + n_rbf + sep_dim`, sets `cond_in_phi_e=False`

### Grand comparison (1000 samples, 100 DDIM steps)

| Model | Bond±0.5% | Bond±0.2% | Bond RMSE | Rg | Rg std | ETE | MMD | Diversity |
|-------|-----------|-----------|-----------|-----|--------|-----|-----|-----------|
| **Reference** | 100% | 98% | 0.063 | 5.92 | 1.11 | 12.0 | — | 3.79 |
| AdaLN+E+Physics | **90.0%** | 32.6% | 0.186 | 5.73 | 1.02 | 10.82 | **0.0323** | 7.91 |
| TorsionTransformer | **100%** | **100%** | 0.000 | 6.23 | 1.03 | 14.36 | 0.0350 | 6.72 |
| AdaLN+SC-v3 | 89.7% | 18.1% | 0.180 | 6.52 | 1.31 | **14.69** | 0.0358 | **9.10** |
| SE3flow-AdaLN | 35.2% | 7.1% | 0.478 | **5.93** | 1.07 | **11.90** | **0.0323** | 8.27 |
| EGNN (baseline) | 15.7% | 1.8% | 0.495 | 5.84 | 1.14 | 11.97 | 0.0358 | 8.17 |

AdaLN+SC-v3 — good bond validity but badly broken global structure (ETE=14.7, diversity=9.1). Do not use as reference model.

### Transformer AdaLN ablation — what drives the +37% bond validity gain?

| Model | Bond±0.5% | Bond RMSE | Rg | ETE | MMD |
|-------|-----------|-----------|-----|-----|-----|
| Baseline (no AdaLN) | ~38% | ~0.30 | 4.95 | 7.0 | 0.062 |
| **AdaLN-only** (transformer_adaln/v1) | **75.4%** | 0.272 | 5.65 | 10.69 | 0.0323 |
| AdaLN + Energy | 74.8% | 0.253 | 5.62 | 10.43 | 0.0342 |
| **AdaLN + E + Physics** | **90.4%** | 0.197 | 5.74 | 10.87 | 0.0358 |

**AdaLN alone provides +37% bond validity.** Energy conditioning adds ~0% to bond validity (74.8% vs 75.4%). Physics loss adds +15.6%. Energy helps for temperature-controlled sampling at inference, not bond geometry.

### EGNN physics weight ablation

| Model | Bond±0.5% | Bond RMSE | Rg | ETE | MMD |
|-------|-----------|-----------|-----|-----|-----|
| EGNN baseline (v1) | 15.7% | 0.495 | 5.84 | 11.97 | 0.0358 |
| EGNN + P(0.05) (v3) | 44.2% | 0.331 | 5.87 | 12.16 | **0.0323** |
| EGNN + P(0.10) (v4) | **58.8%** | 0.314 | 6.05 | 12.79 | 0.0342 |

Doubling physics weight: +14.6pp validity. Rg/ETE creep up at 0.10 — model avoids clashes at cost of slightly extended structures.

### EGNN+AdaLN+Energy evaluation (egnn_adaln/v1)

| Model | Bond±0.5% | Bond RMSE | Rg | ETE | MMD | Clash% |
|-------|-----------|-----------|-----|-----|-----|--------|
| EGNN baseline | 15.2% | 0.498 | 5.79 | 11.62 | 0.034 | 25.8% |
| **EGNN+AdaLN+E** | **43.7%** | 0.377 | 5.73 | **12.04** | **0.032** | **39.1%** |

ETE is near-perfect (12.04 ≈ ref 12.0) and MMD is excellent. But clash rate jumped to 39.1% — model improved bond lengths without the physics loss to prevent clashes. **Next: add physics to egnn_adaln.**

### All-atom EGNN (egnn_adaln_aa/v3) — FAILED
- 0% bond validity, bond RMSE=1.78 Å (catastrophic). Per-bond RMSEs 0.7–3.2 Å.
- Root cause: EGNN + 93 atoms is far too hard to optimise. The se3flow_all_atom (24.6%) is still the best all-atom result.

### Cross-cutting findings
1. **Universal diversity problem**: every model generates ~8 Å pairwise RMSD vs ref 3.79 Å. All 2× too diverse. Energy conditioning at τ<0.5 or higher guidance_scale needed to focus on the folded basin.
2. **Transformer family is systematically compact**: Rg 5.6–5.7, ETE 10.4–10.9 vs ref 5.92/12.0. Self-attention averages toward the mean conformation.
3. **TorsionTransformer ETE bias is structural**: ETE=14.4 (+20%) is unchanged vs MLP v1. Root cause: virtual Cα angles entangle φ and ψ — BackboneTransformer is the direct fix.
4. **Bond RMSE at ±0.2 Å**: best Cartesian model (90% at ±0.5) only achieves 32.6% at ±0.2. Only TorsionFlow/Transformer reach 100% at ±0.2.

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
**Physics constraints implemented** (`models/physics.py`, `configs/flowmatch_physics.yaml`):
- t²-weighted bond length MSE, clash repulsion, virtual bond angle Huber loss
- Applied to reconstructed x₁_pred during training; zero impact when `physics_weight=0`
- phys_* columns logged per val epoch in log.jsonl (oracle reference on clean val data)
- Reference floor: `phys_bond=0.0039`, `phys_clash=0.0`, `phys_angle=0.0298`

**Next step:** run `python scripts/train_flow.py --config configs/flowmatch_physics.yaml`

---

## Energy-conditioned flow matching (2026-06-02)

New model family: `flowmatch_energy` and `flowmatch_v2_energy`. Energy label from dataset used to condition generation; temperature τ ∈ [0,1] controls stable↔transient conformations at inference. Implemented: CFG with null embedding, `ddim_sample_cfg`, analysis script.

### Best results so far (flowmatch_energy_physics, 500 epochs)

| Metric | FM+E+P | FM+E | FM+Physics | FlowMatch | EGNN-DDPM | Transformer | Reference |
|--------|--------|------|-----------|-----------|-----------|-------------|-----------|
| Bond±0.5% | 17.2% | 16.0% | 10.8% | 6.0% | 17.1% | **39.8%** | 100% |
| Bond RMSE | 0.581 | 0.666 | 0.793 | 0.938 | 0.500 | **0.295** | 0.063 |
| Rg mean | **5.89** | 5.90 | 5.90 | 5.91 | 5.75 | 4.94 | 5.92 |
| Rg std | **1.04** | 1.03 | 1.07 | 1.06 | 1.10 | 0.30 | 1.11 |
| MMD | **0.032** | 0.034 | 0.036 | 0.033 | 0.036 | 0.063 | — |

Transformer has best bond validity but is mode-collapsed (Rg std=0.30 vs 1.11 ref, ETE=6.9 vs 12.0). Flow models capture full ensemble but have poor bond geometry.

### Temperature control (flowmatch_energy/v2, guidance_scale=2.0)
τ=0 → Rg≈4.9Å, ETE≈4.6Å (compact/folded) | τ=1 → Rg≈8.2Å, ETE≈24Å (extended/transient). All monotonicity tests pass.

## SE3flow next-gen architecture (2026-06-02)

### Phase 1 — already implemented (no retraining needed)

**Inference-time:**
- `models/flow_matching.py`: `_project_bonds()` static method (SHAKE algorithm, equivariant, SE3-preserving)
- `ZeroCoMFlowMatching.ddim_sample_cfg()` accepts `shake_bonds=True, shake_ideal, shake_frac=0.2, shake_iter=3`

**Architecture:**
- `models/se3flow_energy.py`: `cond_emb` restored in `SE3FlowLayer.phi_e` (additive with AdaLN — regression fix)
- `scripts/train_flow_energy.py`: AllAtomPhysics auto-selected when `n_residues != 10`
- `configs/se3flow_all_atom.yaml`: all-atom SE3flow config (n_residues=93, 5 layers, AllAtomPhysics)

---

## se3flow_all_atom_v2/v1 evaluation fixes (2026-06-03) — DONE

Existing `eval_metrics.json` was garbage (bond RMSE = 598 641 Å, validity = 0%) due to 5 bugs
in the evaluation pipeline. All fixed in `scripts/evaluate.py` and `scripts/analyze_energy_conditioning.py`.

**Root causes fixed:**

| Bug | Effect | Fix |
|-----|--------|-----|
| `SE3FlowEnergyNet` built without `self_cond=True` | RuntimeError / silent wrong weights | Added `self_cond=mc.get('self_cond', False)` to both scripts |
| Harmonic prior not wired at eval time | Model saw OOD x₀ (Gaussian vs SHAKE prior) | Wire `sample_all_atom_chain_batched` via `partial` when `flow.harmonic_prior: true` and `n_residues=93` |
| `generate()` called `ddim_sample()` not `ddim_sample_cfg()` | Energy conditioning bypassed | Added `_is_energy_cond` flag; generate() routes to `ddim_sample_cfg(tau=0.5, guidance_scale=1.0)` |
| Cα bond ideal (3.832 Å) applied to 93-atom all-atom output | Nonsense metrics | Added `compute_physics_metrics_aa()` using AllAtomPhysics bond indices & per-bond targets |
| `--test data/test.npz` (10 Cα atoms) vs model output (93 all-atom) | Wrong reference | `--test` now optional; auto-read from `config['data']['test_path']` |

**New in evaluate.py:**
- `compute_physics_metrics_aa()` — 64-bond covalent RMSE, per-bond targets, 2.5 Å heavy-atom clash
- `_aa_bond_lengths()` — extracts only the 64 bonded pairs from 92 consecutive distances
- `compute_all_metrics(…, is_all_atom=False)` — dispatches to the right metric function
- `print_table()` — updated labels for all-atom (AA-bond valid, Cov-bond mean, etc.)
- `plot_comparison(…, is_all_atom=False)` — per-atom flexibility panel, per-bond RMSE for 64 bonds
- `save_pdbs()` — HETATM fallback for non-Cα structures
- `main()` — `--test` is now optional (auto from config), detects `is_all_atom` from n_residues

**New in analyze_energy_conditioning.py:**
- Same SE3FlowEnergyNet + harmonic prior loading fix
- `_aa_bond_validity()`, `_aa_bond_rmse()`, `_aa_bond_lengths()` functions
- `compute_metrics(…, is_all_atom=False)` dispatches to all-atom bond metrics
- `--test` optional (auto from config)
- `plot_analysis(…, is_all_atom=False)` uses correct reference bond validity line

**Smoke-test result (10 samples, 10 ODE steps, CPU):**
- Bond RMSE: 0.25 Å (vs 598 641 Å before); Bond valid ±0.5 Å: 40%; Rg: 5.94 Å ✓

**New eval commands for se3flow_all_atom_v2:**
```bash
# Full evaluation (auto-detects data_all_atom/test.npz from config):
python scripts/evaluate.py \
    --ckpt checkpoints/se3flow_all_atom_v2/v1/best.pt \
    --n 1000 --steps 100 --save plots/se3flow_aa_v2_eval.png

# Temperature sweep:
python scripts/analyze_energy_conditioning.py \
    --checkpoint checkpoints/se3flow_all_atom_v2/v1/best.pt \
    --temperatures 0.0 0.25 0.5 0.75 1.0 \
    --n 500 --steps 100 --guidance_scale 2.0 \
    --save plots/se3flow_aa_v2_energy.png
```

---

## Phase 2: Bond-graph harmonic prior + self-conditioning (2026-06-02) — READY TO TRAIN

The primary new model targets the fundamental bond-validity ceiling (35.2%) by addressing
its root cause: Gaussian x₀ has completely wrong bond geometry, splitting gradient capacity.

### Changes implemented

**New file: `models/harmonic_prior.py`**
- `sample_all_atom_chain_batched(B, N, device, coord_scale, n_shake=150)`:
  Vectorised Jacobi-SHAKE sampler. Starts from Gaussian noise, applies 150 iterations of
  parallel bond constraint projection across all 64 covalent bonds simultaneously via
  `gather` + `scatter_add_` (no inner Python loop). After convergence: bond MAE = 0.00000 Å.
  Speed: ~12ms/call (B=64, N=93) on L40S.
- `sample_ca_chain(B, N, device, bond_length, coord_scale)`:
  Cα random-walk prior — perfect bond lengths for Cα-only model.

**Modified: `models/flow_matching.py`**
- `ZeroCoMFlowMatching(prior_fn=None)`: optional prior callable `(B, N, device) → (B, N, 3)`
- `_sample_noise()` delegates to `prior_fn` when set (Gaussian fallback otherwise)
- `training_loss_energy()`: 50% self-conditioning pre-pass (stop-gradient)
- `ddim_sample_cfg()`: SC state `sc_x1` tracked between ODE steps

**Modified: `models/se3flow_energy.py`**
- `SE3FlowEnergyNet(self_cond=False)`: new parameter
- `self_cond_proj = Linear(n_rbf, node_dim)`, zero-init → identity at start
- `forward(x_t, t, energy_z, sc_x1=None)`: SE(3)-invariant RBF(x̂₁) per-node features
  injected into `h` before message-passing layers
- Equivariance preserved: distances are rotation-invariant; `h` is invariant, `x` equivariant

**Modified: `scripts/train_flow_energy.py`**
- Reads `flow.harmonic_prior: true` → wires `prior_fn` to `ZeroCoMFlowMatching`
- Reads `model.self_cond: true` → passes to `SE3FlowEnergyNet`

**New configs:**
- `configs/se3flow_all_atom_v2.yaml`: all-atom (n=93) + harmonic prior + self-cond + AllAtomPhysics
- `configs/se3flow_all_atom_v2_local.yaml`: CPU smoke test (2 layers, 5 epochs, batch=4)
- `configs/se3flow_sc_local.yaml`: Cα smoke test of Phase 2 features

### Verified (2026-06-02/03)
- SHAKE convergence: bond MAE = 0.00000 Å (150 Jacobi iterations) ✓
- Self-conditioning equivariance: max_err=1.25e-07 ✓
- Zero-init: v(sc_x1=None) ≡ v(sc_x1=rand) at init — diff = 2.87e-07 ✓
- Smoke test (5 epochs, Cα data, harmonic prior + self-cond + physics):
  - Equivariance max_err=7.77e-08 ✓
  - Training converges: loss 0.727→0.676
  - Physics at reference floor: bond=0.0039, clash=0.000, angle=0.0298 ✓
  - ~25s/epoch on L40S ✓
- `scripts/evaluate.py`: fixed missing `x1_pred` + `self_cond` in SE3FlowEnergyNet constructor (caused RuntimeError on checkpoint load) ✓
- `configs/se3flow_all_atom_v2.yaml`: fixed `val.npz` → `valid.npz` to match actual data_all_atom/ filenames ✓

### Run commands

```bash
# Smoke test (Cα data, all Phase 2 features, ~2 min):
python scripts/train_flow_energy.py --config configs/se3flow_sc_local.yaml

# Full all-atom training (needs data_all_atom/ from cluster):
python scripts/train_flow_energy.py --config configs/se3flow_all_atom_v2.yaml
# Checkpoints: checkpoints/se3flow_all_atom_v2/v1/

# Evaluate:
python scripts/evaluate.py \
    --ckpt checkpoints/se3flow_all_atom_v2/v1/best.pt \
    --test data_all_atom/test.npz --n 1000 --save plots/se3flow_aa_v2.png
```

### Expected improvements
- Harmonic prior alone: +10–20% bond validity (model only learns rearrangement, not creation)
- Self-conditioning alone: +5–15% bond validity (iterative refinement per ODE step)
- Combined target: ≥50% bond validity (vs 35.2% SE3flow AdaLN baseline, 60.3% EGNN+Physics)

### TorsionFlow implemented — see Phase 3 section below.

---

## SE3flow AdaLN campaign (2026-06-02) — COMPLETE

### Goal
Improve bond validity of SE3flow beyond 39.8% (original v1) by adding AdaLN per-layer conditioning and x1_pred training mode.

### Architecture changes made to `models/se3flow_energy.py`
- `SE3FlowLayer`: added `adaLN_modulation` (zero-init, SiLU + Linear → 3×node_dim scale/shift/gate); removed `cond_emb` from `phi_e` edge input (time now reaches edges indirectly via h)
- `SE3FlowEnergyNet`: added `x1_pred: bool` param; added dedicated zero-init equivariant x1 head (`x1_phi_e`, `x1_phi_x`, `x1_sep_embed`); head always present for fine-tuning but only active when `x1_pred=True`
- `scripts/train_flow_energy.py`: added `--finetune` flag (strict=False loading, reset optimizer/epoch, create new version dir); fixed versioning bug where `--finetune --resume` wrote into the source checkpoint directory
- `scripts/evaluate.py`: fixed `UnboundLocalError` for `ZeroCoMGaussianDiffusion` (now imported at top of `load_model_from_ckpt`)
- New configs: `configs/se3flow_adaln_velocity.yaml` (clean foundation dir), `configs/se3flow_energy_finetune.yaml`

### Runs attempted (se3flow_energy/)

| Run | Config | Key changes | Epochs | Final loss | Outcome |
|-----|--------|-------------|--------|-----------|---------|
| v1 | original | baseline SE3flow | 500 | 0.628 | ✓ converged — 39.8% bond validity |
| v2 | x1_pred=True, phys=0.20 | new arch + x1_pred | 201 | 10.62 | ✗ stagnated |
| v3 | x1_pred=True, phys=0.05 | lower physics | 146 | 10.61 | ✗ stagnated |
| v4 | x1_pred=False, phys=0.15 | velocity + AdaLN | 74 | 0.74 | interrupted |
| v5 | x1_pred=True (bug) | wrong config | 155 | 10.65 | ✗ stagnated; logs overwritten by finetune bug |

### Run in new directory (se3flow_adaln_velocity/)

| Run | Config | Epochs | val_loss | bond±0.5% | bond RMSE | Rg mean | MMD |
|-----|--------|--------|---------|-----------|-----------|---------|-----|
| v1 | se3flow_adaln_velocity.yaml (x1_pred=False) | 500 | 0.6266 | **35.2%** | 0.478 Å | 5.927 | 0.032 |

### x1_pred fine-tune attempts (se3flow_energy_finetune/)
All fine-tune attempts from any base produced loss ≈ 10.6, consistent with stagnation. Root cause not fully diagnosed — suspected interaction between the accumulated coordinate update architecture and the x1_pred MSE objective. The x1_head zero-init is verified correct (δ=0.000000 at init), and initial loss should be ~1.5, but observed stagnation from epoch 1 suggests a deeper incompatibility.

### Conclusion
**AdaLN slightly hurt bond validity (35.2% vs 39.8% original)**. Removing `cond_emb` from edge messages likely weakened the model's ability to modulate bond constraints by timestep. The AdaLN node-feature modulation did not compensate. x1_pred mode is consistently incompatible with this architecture — every attempt across 5 runs produced loss ≈ 10.6. Global structure metrics (Rg, MMD, diversity) are unchanged — SE3flow is still better than diffusion models on ensemble quality.

### Known ceiling: bond validity ~35–40% for SE3flow velocity mode
The root cause is the flow matching framework: straight-line interpolant paths create unphysical intermediate structures (x_t is not a valid protein), and the physics loss on a reconstructed x1_pred carries noisy gradients. Diffusion (DDPM) achieves higher bond validity (EGNN+Physics: 60.3%) because x0_pred at each denoising step is trained to look like real data with correct geometry. See TODO.md for architectural improvement options.

## Phase 4: TorsionFlow v2 — Transformer + data-driven φ source (2026-06-03)

### Motivation
TorsionFlow v1 (MLP, 44K params) achieved 100% bond validity by construction, but global
structure metrics were off: ETE 14.24 Å vs 11.99 Å ref (+18.8%), Rg 6.20 vs 5.92 Å, diversity
6.75 vs 3.79 Å. Root causes: (1) global MLP cannot model inter-residue dihedral correlations,
(2) Uniform(−π,π) source creates very long flow paths (expected |Δφ| ≈ π/2), (3) underfitting
(loss plateaued at ~0.93 by epoch 350 with LR decayed to near zero).

### Changes implemented

| File | Change |
|------|--------|
| `models/torsion_transformer.py` | NEW — 15-token Transformer (θ₀…θ₇, φ₀…φ₆) with self-attention; captures inter-residue dihedral correlations |
| `models/torsion_flow.py` | Added `phi_source_std` + `phi_weights` parameters; data-driven WrappedNormal(0,σᵢ) source; per-position inverse-variance φ loss weighting; both stored as buffers in checkpoint |
| `models/internal_coords.py` | Added `compute_phi_source_params()` — circular std per dihedral + normalised inverse-variance weights + updated phi_scale |
| `scripts/train_torsion.py` | Supports both `torsion_flow_energy` and `torsion_transformer_energy`; configurable `theta_source_std` and `phi_source_dist`; cosine annealing with warm restarts; `flow.state_dict()` saved in checkpoint |
| `scripts/evaluate.py` | Added `torsion_transformer_energy` branch; restores `phi_source_std`/`phi_weights` buffers from `ckpt['flow']` |
| `configs/torsion_transformer_energy.yaml` | NEW — full GPU config (d_model=256, 6 layers, 4 heads, 1000 epochs, cosine restarts T0=100) |
| `configs/torsion_transformer_local.yaml` | NEW — CPU sanity-check (d_model=64, 2 layers, 5 epochs, uniform source) |

### Key design decisions
- **Transformer tokens**: each of the 15 DOFs (θ₀…θ₇, φ₀…φ₆) is a token with (sin,cos) features + learnable positional embedding. Time and energy are broadcast-added before attention.
- **Data-driven φ source**: `phi_source_dist: 'data'` computes per-dihedral circular std from training set, replaces Uniform(−π,π) with WrappedNormal(0, σᵢ). Shorter paths → easier to learn.
- **theta_source_std: 0.403** matches training-set θ std so source ≈ target width (symmetric flow).
- **Cosine warm restarts** (T0=100 epochs, T_mult=2) prevent the LR-decay plateau that stalled v1 at epoch ~350.

### Verified (2026-06-03)
- Zero-init output heads confirmed: `pred_theta.abs().max < 1e-6` ✓
- Training loss finite and backward pass OK ✓
- phi_source_std and phi_weights state_dict roundtrip ✓
- Heun ODE sampling OK ✓

### Run commands
```bash
# CPU sanity-check (~2 min):
python scripts/train_torsion.py --config configs/torsion_transformer_local.yaml

# Full GPU training:
python scripts/train_torsion.py --config configs/torsion_transformer_energy.yaml

# Evaluate:
python scripts/evaluate.py \
    --ckpt checkpoints/torsion_transformer/v1/best.pt \
    --test data/test.npz --n 1000 --steps 100 \
    --save plots/torsion_transformer_eval.png

# Compare v2 vs v1:
python scripts/evaluate.py \
    --ckpt checkpoints/torsion_transformer/v1/best.pt \
    --ckpt_ref checkpoints/torsion_flow/v1/best.pt \
    --labels TorsionTransformer-v2 TorsionMLP-v1 \
    --test data/test.npz --n 1000 --steps 100 \
    --save plots/torsion_v2_vs_v1.png
```

### Full evaluation result (2000 epochs — 2026-06-03)

Best val loss: **0.7159** (epoch 1999) vs 0.9298 for MLP v1 — **−23%**

| Metric | Transformer v2 | MLP v1 | Reference | Δ |
|--------|---------------|--------|-----------|---|
| val_loss | **0.7159** | 0.9298 | — | −23% ✅ |
| MMD ↓ | **0.0333** | 0.0349 | — | better ✅ |
| Clash rate | **1.3%** | 3.6% | 0% | −64% ✅ |
| Rg mean (Å) | 6.18 | 6.15 | 5.92 | ~same ⚠️ |
| ETE mean (Å) | 14.14 | 13.97 | 11.99 | ~same ⚠️ |
| Angle cos RMSE | 0.245 | 0.244 | ~0 | ~same ⚠️ |
| Diversity RMSD (Å) | 6.70 | 6.71 | 3.79 | ~same ⚠️ |
| Bond validity | **100%** | **100%** | 100% | — ✅ |

Plot: `plots/torsion_v2_vs_v1.png`

### Interpretation

**What improved:** Velocity field accuracy (lower loss), distribution quality (MMD), steric clashes. These are real gains from the Transformer + data-driven source.

**What didn't move:** ETE, Rg, angle RMSE, diversity are essentially identical between architectures. Root cause: these errors are **not** architectural. The phi_scale only fell 1.81 → 1.61, meaning Chignolin's dihedral distribution is genuinely broad in the MD ensemble. The ~2 Å ETE bias is likely a systematic NeRF artifact — small per-angle errors compound along the chain and tilt reconstructed structures toward more extended conformations.

### Pass criteria for iteration 2

| Criterion | Result |
|-----------|--------|
| val_loss < 0.930 | ✅ PASS (0.7159) |
| MMD improved | ✅ PASS (0.033 vs 0.035) |
| Clash rate reduced | ✅ PASS (1.3% vs 3.6%) |
| ETE mean 11–13 Å | ❌ FAIL (14.14 Å) |
| Rg mean 5.7–6.1 Å | ❌ FAIL (6.18 Å) |
| Angle RMSE < 0.15 rad | ❌ FAIL (0.245 rad) |
| Diversity < 5.0 Å | ❌ FAIL (6.70 Å) |

### Known ceiling: ETE/Rg/diversity stuck — different root cause needed

The ETE/Rg/diversity ceiling is shared by both MLP and Transformer — the problem is **not** model capacity. Most likely causes:
1. MD ensemble genuinely contains many extended conformations; the reference statistics may reflect the full thermodynamic ensemble rather than only the folded state.
2. NeRF reconstruction amplifies angular errors: a bias of ~5° per bond angle across 8 atoms compounds into ~2 Å ETE offset.
3. Flow paths still too long: phi_scale=1.61 still large (data dihedrals broadly distributed).

**Candidates for iteration 3:**
- Sample at lower temperature (τ=0.4) to bias toward compact/folded basin
- More ODE steps (200–500) for more accurate integration
- Per-residue angle RMSE analysis to find where the bias concentrates

---

## Phase 3: TorsionFlow — Riemannian flow matching in internal coordinate space (2026-06-03)

### Motivation
All-atom v2 did not improve bond validity. Root cause: Cartesian coordinates force the model
to simultaneously learn global structure AND covalent geometry — nearly orthogonal objectives.

### Solution: internal coordinates
Represent the Cα chain as bond angles θ (8 DOF) + dihedral angles φ (7 DOF) with bond
lengths fixed at 3.832 Å. Bond validity is 100% by construction — the lengths are a
constant of the representation, not a learned quantity.

Dihedral angles live on S¹ (a circle), making the joint space a flat torus T^7 — a compact
Riemannian manifold. Flow matching on this manifold uses geodesic angular interpolation
(wrapped shortest-arc paths) rather than straight Cartesian lines.

### New files

| File | Description |
|------|-------------|
| `models/internal_coords.py` | `cartesian_to_internal`, `internal_to_cartesian` (NeRF), `angle_wrap`, `compute_velocity_scales` |
| `models/torsion_net.py` | `TorsionFlowNet` — 4-layer MLP, 118-dim input, energy CFG dropout, 44K params |
| `models/torsion_flow.py` | `TorsionalFlowMatching` — OT-CFM on (θ,φ) space; Heun ODE with torus-aware wrapping |
| `configs/torsion_flow_energy.yaml` | Full GPU training config (256 hidden, 4 layers, 500 epochs) |
| `configs/torsion_flow_local.yaml` | CPU sanity-check (5 epochs, 128 hidden, 2 layers) |
| `scripts/train_torsion.py` | Training script — converts Cartesian batches to internal coords on the fly |

### Modified files
- `scripts/evaluate.py`: added `torsion_flow_energy` model type; generates in (θ,φ) space then NeRF reconstructs to Cartesian; handles zero-variance bond-length histogram

### Key design decisions
- **Loss normalisation**: θ velocities (~0.4 rad) vs φ velocities (~1.8 rad) differ by 4×. Both
  normalised by their empirical velocity stds (σ_θ=0.40, σ_φ=1.81) so both loss terms have
  unit variance. Computed on first pass over training data and stored in checkpoint.
- **Source distribution**: θ₀ ~ Gaussian(data mean 108.6°, std 0.30 rad); φ₀ ~ Uniform(−π, π)
- **NeRF canonical frame**: atom 0 at origin, atom 1 along +x, atom 2 in xy-plane. CoM
  subtracted after reconstruction in generate().
- **No physics loss, no equivariant network, no ZeroCoM projection** — all three
  become unnecessary when operating in internal coordinate space.

### Sanity check result (5 epochs, CPU/GPU, 200 samples, 20 ODE steps)
```
Bond validity ±0.5 Å :  100.0%  (vs 39.8% best Cartesian model)
Bond validity ±0.2 Å :  100.0%  (vs ~10% best Cartesian model)
Bond RMSE            :  0.0000 Å (exact by construction)
MMD-RBF              :  0.032  (already matches FlowMatch v1 after 5 epochs!)
Rg mean              :  6.10 Å  (ref 5.92 — will improve with full training)
```

Bond validity is **100% at all thresholds** and is guaranteed regardless of training quality.

### Run commands
```bash
# CPU sanity-check (~1 min):
python scripts/train_torsion.py --config configs/torsion_flow_local.yaml

# Full GPU training (500 epochs):
python scripts/train_torsion.py --config configs/torsion_flow_energy.yaml

# Evaluate:
python scripts/evaluate.py \
    --ckpt checkpoints/torsion_flow/v1/best.pt \
    --test data/test.npz --n 1000 --steps 100 \
    --save plots/torsion_flow_eval.png
```

---

## Not yet run at scale

- MLP full training run
- DDIM/ODE step count speed-quality experiments

---

## Run history

---

**Date:** 2026-06-03
**Model:** torsion_transformer_energy (TorsionTransformerNet — 15-token Transformer)
**Config:** configs/torsion_transformer_energy.yaml (d_model=256, 6 layers, 4 heads, phi_source_dist=data, cosine restarts T0=100×2)
**Hardware:** GPU (L40S, workspace)
**Epochs run:** 2000 / 2000
**Best val loss:** 0.7159 (epoch 1999)
**Checkpoint:** checkpoints/torsion_transformer/v1/best.pt
**Key metrics:** MMD=0.033, clash=1.3%, bond validity=100%; ETE=14.14 Å (ref 11.99), Rg=6.18 Å (ref 5.92), diversity=6.70 Å (ref 3.79) — structural topology unchanged vs MLP v1
**Notes:** Transformer significantly improved velocity field fit (−23% val_loss) and clashes (−64%). ETE/Rg/diversity ceiling shared with MLP v1 — not an architectural problem. Owner: marik.

---

**Date:** 2026-06-03
**Model:** transformer_adaln (AdaLNTransformerScoreNetwork — no energy, no physics)
**Config:** configs/transformer_adaln_energy.yaml without energy (inferred from checkpoint: transformer_adaln/v1)
**Hardware:** GPU (workspace)
**Epochs run:** 500 / 500
**Best val loss:** 0.1887
**Checkpoint:** checkpoints/transformer_adaln/v1/best.pt
**Key metrics:** bond validity 75.4%, Rg=5.65, ETE=10.69, MMD=0.0323, diversity=7.80 Å
**Notes:** Ablation control — AdaLN alone (no energy, no physics). Proves AdaLN per-layer conditioning is the primary driver of bond validity improvement (+37pp vs baseline transformer). Energy and physics add on top. Owner: pshah.

---

**Date:** 2026-06-03
**Model:** egnn + physics (physics_weight=0.05 and 0.10)
**Config:** egnn physics ablation (v3 = 0.05, v4 = 0.10)
**Hardware:** GPU (workspace)
**Epochs run:** 500 / 500
**Best val loss:** v3=0.1980, v4=0.2067
**Checkpoint:** checkpoints/egnn/v3/best.pt, checkpoints/egnn/v4/best.pt
**Key metrics:** v3: 44.2% bond validity, MMD=0.032 ✓; v4: 58.8% bond validity, Rg slightly high (6.05)
**Notes:** Physics weight 0.05→0.10 gives +14.6pp bond validity. Diminishing returns expected beyond 0.10. Owner: pshah.

---

**Date:** 2026-06-03
**Model:** egnn_adaln (EGNNAdaLNScoreNetwork — AdaLN + energy + physics_weight=0.10)
**Config:** configs/egnn_adaln.yaml
**Hardware:** GPU (workspace)
**Epochs run:** 500 / 500
**Best val loss:** 0.2076
**Checkpoint:** checkpoints/egnn_adaln/v1/best.pt
**Key metrics:** bond validity 43.7%, Rg=5.73, ETE=12.04 (near-perfect!), MMD=0.032, clash_rate=39.1% ⚠️
**Notes:** Best ETE of any EGNN-family model. High clash rate because no physics loss; must add physics before using. Owner: pshah.

---

**Date:** 2026-06-03
**Model:** backbone_transformer (BackboneTransformerScoreNetwork — N/Cα/C frames, 30 atoms)
**Config:** configs/backbone_transformer_local.yaml (smoke test only)
**Hardware:** GPU (workspace)
**Epochs run:** 5 / 500 (smoke test)
**Best val loss:** 0.0878
**Checkpoint:** checkpoints/backbone_transformer_local/v1/best.pt
**Key metrics:** loss converging 0.179→0.088, physics wired, all systems go ✓
**Notes:** Smoke test to verify implementation. Full 500-epoch run launched with configs/backbone_transformer.yaml. Backbone coords_scale=3.42 Å; 30 tokens (N/CA/C × 10 residues); atom_type + residue embeddings. Owner: tkreusel.

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

---

## SE3flow v2 architecture — ready to train (2026-06-02)

Two targeted improvements to `models/se3flow_energy.py` and `configs/se3flow_energy.yaml`:

### Root causes addressed

**Problem 1 (primary):** In velocity-mode OT-CFM, physics is applied to a *reconstructed* x₁_pred with gradient-killing `clamp(-5,5)`. Despite equal effective physics weight (~0.05) to EGNN+Physics, SE3flow achieves only 39.8% bond validity vs EGNN's 60.3%. The difference is gradient quality, not weight.

**Problem 2 (secondary):** Time conditioning via concat to edge messages competes with RBF geometry features for MLP capacity. The same-architecture AdaLN transformer improves bond validity 40.6% → 75.5% purely by switching to per-layer scale/shift/gate modulation (AdaLN-Zero).

### Changes made

1. **AdaLN-Zero per-layer conditioning** (`SE3FlowLayer`):
   - Removed `cond_emb` from edge messages (phi_e)
   - Added `adaLN_modulation` (zero-init) to each layer: derives `(scale, shift, gate)` from `cond_emb` and applies to node features after LayerNorm
   - Gates start at 0 → identity mapping at init; time still reaches edge messages via node features h

2. **`x1_pred` output mode** (`SE3FlowEnergyNet`):
   - New `x1_pred: bool` constructor parameter; when True, returns accumulated coordinates `x` (zero-CoM) directly as x̂₁
   - Mirrors DDPM's x₀-prediction: physics applied directly to model output, no reconstruction formula, no gradient-killing clamp
   - `ddim_sample_x1pred_cfg` already exists and handles ODE velocity conversion at inference

3. **Config updates** (`se3flow_energy.yaml`):
   - `model.x1_pred: true`
   - `training.physics_weight: 0.20` → effective 0.10 with linear-t (2× EGNN+Physics's 0.05)

4. **Training script fix** (`train_flow_energy.py`): `x1_pred` now passed to model constructor.

### Verification
- SE(3) equivariance: max_err=2.38e-07 (x1_pred=True), 1.74e-07 (velocity mode) — both PASS
- Parameters: 1,593,504 (up from ~1.27M for AdaLN modules)
- Smoke-test training loss: 5.69 (finite, backward OK, AdaLN grads activate during training)
- Expected improvement: **55–65% bond validity** (from 39.8%), potentially exceeding EGNN+Physics (60.3%)

**To train:** `python scripts/train_flow_energy.py --config configs/se3flow_energy.yaml`  
**To evaluate:** `python scripts/evaluate.py --ckpt checkpoints/se3flow_energy/v2/best.pt --test data/test.npz --n 1000`

---

## SE(3) equivariance analysis (2026-06-02)

`scripts/check_equivariance.py` — three empirical tests across all model types.

**Method:**
- Test 1: score-network error `‖model(Rx,t) − R·model(x,t)‖ / ‖model(x,t)‖` (single forward pass, exact)
- Test 2: full-pipeline error `‖f(Rx₀) − R·f(x₀)‖ / ‖f(x₀)‖` (deterministic DDIM eta=0 / Heun ODE — no averaging needed)
- Test 3: distribution isotropy λ_max/λ_min of generated ensemble covariance (~1 = isotropic)

| Model | T1 proper | T1 refl | T2 proper | T2 refl | Isotropy ratio |
|-------|-----------|---------|-----------|---------|----------------|
| FlowMatch (equivariant) | ~1e-7 PASS | ~1e-7 PASS | ~4e-7 PASS | ~4e-7 PASS | ~1.3 (100 samples) |
| Transformer (non-equivariant) | ~1.09 FAIL | ~1.01 FAIL | ~1.38 FAIL | ~1.34 FAIL | 6.15 ANISOTROPIC |

The 6-order-of-magnitude gap between equivariant (~1e-7) and non-equivariant (~1) models confirms SE(3) equivariance is working as intended. Isotropy ratio of 6.15 for the Transformer shows it generates structures with a strong preferred orientation learned from training data.

Note: isotropy ratio for equivariant models becomes more reliable with more samples; ~500 recommended for the full test.

---

## Git history

| Commit | Message | What changed |
|--------|---------|-------------|
| (this commit) | Add SE(3) equivariance analysis script | `scripts/check_equivariance.py`; COLLAB docs updated |
| 21c89c2 | Add physics-constrained flow matching (bond, clash, angle losses) | See below |
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
