# Shared TODO

Tasks for the full team. When you pick up a task, add your name + date next to it. When done, mark it and add results to [STATUS.md](STATUS.md).

**Format:** `- [ ] Task — *Owner (date)*`  
**Done:** `- [x] Task — *Owner (date)* — result summary`

---

## Immediate / Blockers

- [ ] Distribute data files to all team members — source: `/g/korbel2/shahp/deeplife/data/` on cluster

---

## eval_v2 follow-up (from 2026-06-04 overnight run)

- [ ] **Run τ-vs-REU scoring** — `bash scripts/run_eval_all.py` already complete but `tau_rosetta` field not yet populated. Run `bash scripts/run_tau_reu.sh` (~50 min) to add `figure3b_tau_reu.png` and `sections.energy.<label>.tau_rosetta` to both `ca_only` and `backbone` metrics.json. — *tbd*

- [ ] **Investigate SE3Flow-Energy-v1 coordinate scaling bug** — BondRMSE=7.19Å vs expected ~0.5Å. Check `coord_scale` in checkpoint config vs generation code. Likely the generated coords are in a different unit than the bond targets. — *tbd*

- [ ] **Investigate SE3Flow-AllAtom-v2-v1 T4 Wasserstein=27.8** — T1/T2 equivariance PASS at ~7e-7 but ensemble-level Rg distribution shifts under rotation (W1=27.8 vs expected ~0). Anomalous. Check if coord_scale interacts with the Rg computation. — *tbd*

- [ ] **Diagnose AdaLN-Transformer-Energy-Physics-v1 inverted temperature response** — Spearman r=−0.90 means higher τ → more compact structures (wrong sign). Physics regularisation may have suppressed the energy CFG signal. Try evaluating at guidance_scale=0 to see if the base model is OK. — *tbd*

- [ ] **All-atom architecture rethink** — both all-atom models (EGNN-v3 and SE3Flow-v2) produce 100% clash and catastrophic omega scores (80–115 REU). The 93-atom space is too unconstrained. Consider: (1) backbone-IPA approach for all-atom, (2) constrained sampling (SHAKE), (3) post-hoc relaxation without scoring filter. — *tbd*

---

## In Progress

- [ ] Train backbone_ipa/v8 — *marik, 2026-06-04* — ep 3039+, best val=0.5231 (ep3039); still running. Added to MODEL_REGISTRY as `backbone_ipa/v8`, eval command ready below.

- [ ] Train backbone_transformer (full 500-epoch GPU run) — *tkreusel, 2026-06-03*
  ```bash
  python scripts/train_backbone.py --config configs/backbone_transformer.yaml
  # Checkpoint: checkpoints/backbone_transformer/v1/best.pt
  ```

---

## Up Next — Backbone Transformer

- [x] **Evaluate backbone_ipa/v8** — *marik, 2026-06-04* — 100% bond valid, 0.8% clash, MMD=0.0345 (best backbone), Rosetta −5.8 REU (best backbone). ETE still 14.4 Å (NeRF ceiling). eval_metrics filled in MODEL_REGISTRY.yaml.

- [ ] Evaluate backbone_transformer/v1 after training — *tkreusel*
  ```bash
  python scripts/evaluate.py \
      --ckpt checkpoints/backbone_transformer/v1/best.pt \
      --ckpt_ref checkpoints/transformer_adaln_energy_physics/v1/best.pt \
      --labels "BackboneTransformer" "AdaLN+E+Physics-Cα" \
      --test data_backbone/test.npz \
      --n 1000 --steps 100 --save plots/backbone_vs_ca_transformer.png
  ```
  Key questions: (1) does bond validity reach 100% for all 29 real bonds? (2) does ETE bias shrink vs TorsionTransformer (14.4 Å)?

- [ ] Temperature sweep for backbone_transformer — *tkreusel*
  ```bash
  python scripts/analyze_energy_conditioning.py \
      --checkpoint checkpoints/backbone_transformer/v1/best.pt \
      --temperatures 0.0 0.25 0.5 0.75 1.0 \
      --n 500 --steps 100 --guidance_scale 2.0 \
      --save plots/backbone_temperature.png
  ```

---

## Up Next — Quick wins from ablation findings

- [ ] Add physics to EGNN+AdaLN — egnn_adaln/v1 has 39.1% clash rate due to missing physics loss. Expected: bond validity 60%+, clash rate < 5%  — *pshah*
  ```bash
  # Edit configs/egnn_adaln.yaml: add physics_weight: 0.05, then run:
  python scripts/train_egnn.py --config configs/egnn_adaln.yaml
  ```

- [ ] Temperature sweep for AdaLN+E+Physics to reduce diversity (currently 7.9 vs ref 3.8 Å) — *tbd*
  ```bash
  python scripts/analyze_energy_conditioning.py \
      --checkpoint checkpoints/transformer_adaln_energy_physics/v1/best.pt \
      --temperatures 0.0 0.1 0.2 0.3 0.4 0.5 \
      --n 500 --steps 100 --guidance_scale 3.0 \
      --save plots/best_model_temperature.png
  ```

---

## Up Next (iteration 3 — TorsionFlow ETE/Rg/diversity ceiling)

The ETE/Rg/diversity metrics are stuck at the same values for both MLP and Transformer.
The ceiling is not architectural — BackboneTransformer (real φ/ψ angles) is the main fix.
Remaining candidates for the Cα torsion model:

- [ ] Temperature sweep: sample at τ=0.0–0.5 to find the compact-fold basin — *tbd*
  ```bash
  python scripts/analyze_energy_conditioning.py \
      --checkpoint checkpoints/torsion_transformer/v1/best.pt \
      --temperatures 0.0 0.1 0.2 0.3 0.4 0.5 \
      --n 500 --steps 100 --guidance_scale 2.0 \
      --save plots/torsion_v2_temperature.png
  ```
- [ ] ODE step sweep: compare quality at 50/100/200/500 steps — check if integration error is source of ETE bias — *tbd*

## Completed (eval_v2 pipeline + overnight run, 2026-06-04)

- [x] Build `scripts/eval_v2/` package (5 sections: physics, equivariance, energy, novelty, PyRosetta) — *marik, 2026-06-04* — all 5 sections working; PyRosetta integrated; NeRF Cα→backbone reconstruction; 4 SE(3) tests including new ensemble Wasserstein test
- [x] Build `scripts/run_eval_all.py` batch evaluator with MODEL_REGISTRY.yaml — *marik, 2026-06-04*
- [x] Fix equivariance `use_backbone` bug (n_atoms=10 instead of 30 for backbone_transformer) — *marik, 2026-06-04* — was in 3 places (test1, test2, test4), fixed to `getattr(diffusion, '_is_backbone', False)`
- [x] Fix PyRosetta API mismatches (no `add_missing_atoms`, no `AddCoordinateConstraintMover`, Cα PDB unparseable) — *marik, 2026-06-04* — added `_ca_to_backbone()` NeRF reconstruction; removed broken API calls
- [x] Fix JSON section merge (sections were overwritten per run, not merged per-model) — *marik, 2026-06-04* — all 4 section dicts now do per-model update, not full replacement
- [x] Full overnight evaluation: 24 models × 5 sections — *marik, 2026-06-04* — results in STATUS.md; outputs at `plots/eval_overnight/`
- [x] Add compact histogram storage + `replot.py` for figure regeneration without re-inference — *marik, 2026-06-04* — `plot_data`, `tau_hists`, `per_structure`, `novelty.plot_data` all stored; `replot.py` regenerates all 5 figures from JSON
- [x] Add τ-vs-REU Rosetta scatter (figure3b_tau_reu.png) — *marik, 2026-06-04* — `score_tau_samples()` in pyrosetta_utils, `plot_tau_reu()` in plotting, `--n_tau_rosetta` CLI arg, `scripts/run_tau_reu.sh`; run `bash scripts/run_tau_reu.sh` to populate

## Completed (Phase 5: BackboneTransformer — N/Cα/C frames, 2026-06-03)

- [x] Implement `scripts/prepare_backbone_data.py` — bond-graph backbone extraction (handles PRO ring by picking most-bonded N-neighbor for CA) — *tkreusel, 2026-06-03* — `data_backbone/` created: (79632,30,3), coord_scale=3.42 Å
- [x] Implement `models/backbone_physics.py` — `BackbonePhysics`: 29 bonds, 3 targets (1.460/1.525/1.329 Å), bond_weight=2.0 — *tkreusel, 2026-06-03*
- [x] Implement `models/backbone_transformer.py` — 30-token AdaLN Transformer; atom_type_embed(N/CA/C) + residue_embed(0-9) — *tkreusel, 2026-06-03*
- [x] Implement `scripts/train_backbone.py` + `configs/backbone_transformer.yaml` + `_local.yaml` — *tkreusel, 2026-06-03*
- [x] Wire `scripts/evaluate.py`: `backbone_transformer` branch, `compute_backbone_metrics()`, backbone routing — *tkreusel, 2026-06-03*
- [x] Fix evaluate.py cond_in_phi_e bug for se3flow_adaln_velocity checkpoint loading — *tkreusel, 2026-06-03* — auto-detects from phi_e weight shape (404 = no cond_emb, 500 = with cond_emb); `SE3FlowLayer` + `SE3FlowEnergyNet` now accept `cond_in_phi_e=True` param
- [x] Smoke test backbone_transformer (5 epochs): loss 0.179→0.088, all converging ✓ — *tkreusel, 2026-06-03*

## Completed (Grand comparison + ablations, 2026-06-03)

- [x] Grand 5-model comparison (1000 samples, 100 steps): AdaLN+E+Physics, TorsionTransformer, AdaLN+SC-v3, SE3flow-AdaLN, EGNN — *tkreusel, 2026-06-03* — results in STATUS.md; plot: `plots/grand_comparison.png`
- [x] Transformer AdaLN ablation (3 models: AdaLN-only, +Energy, +E+Physics) — *tkreusel, 2026-06-03* — AdaLN alone = +37pp; energy = 0pp; physics = +15pp. Plot: `plots/transformer_adaln_ablation.png`
- [x] EGNN physics weight ablation (baseline, 0.05, 0.10) — *pshah, 2026-06-03* — 15.7% → 44.2% → 58.8%. Plot: `plots/egnn_physics_ablation.png`
- [x] EGNN+AdaLN+Energy evaluation vs baseline — *pshah, 2026-06-03* — 43.7% bond validity, near-perfect ETE, 39.1% clash rate. Plot: `plots/egnn_adaln_comparison.png`
- [x] All-atom EGNN (egnn_adaln_aa/v3) evaluation — *pshah, 2026-06-03* — 0% bond validity, RMSE=1.78 Å, failed. Plot: `plots/egnn_adaln_aa_eval.png`

## Completed (Phase 4: TorsionFlow v2 — Transformer + data-driven φ source, 2026-06-03)

- [x] Run full TorsionTransformer training (2000 epochs) — *marik, 2026-06-03* — **val=0.7159** (vs MLP 0.9298, −23%); MMD=0.033, clash=1.3% (−64%); ETE/Rg/diversity unchanged vs MLP — ceiling is not architectural
- [x] Evaluate TorsionTransformer v1 vs MLP v1 — *marik, 2026-06-03* — results in STATUS.md; plot: `plots/torsion_v2_vs_v1.png`

- [x] Assess TorsionFlow v1 performance: ETE +18.8%, Rg +4.7%, over-diverse — *marik, 2026-06-03* — root causes: no inter-residue interactions, uniform φ source too wide, LR decay plateau
- [x] Implement `models/torsion_transformer.py` — 15-token Transformer, (sin/cos) inputs, pre-LN, 104K params — *marik, 2026-06-03*
- [x] Extend `models/torsion_flow.py` — `phi_source_std`, `phi_weights` buffers; WrappedNormal source; per-position φ loss weighting — *marik, 2026-06-03*
- [x] Extend `models/internal_coords.py` — `compute_phi_source_params()` (circular std, inv-var weights, new phi_scale) — *marik, 2026-06-03*
- [x] Rewrite `scripts/train_torsion.py` — both model types; `phi_source_dist`; cosine warm restarts; `flow.state_dict()` in checkpoint — *marik, 2026-06-03*
- [x] Extend `scripts/evaluate.py` — `torsion_transformer_energy` branch; restore phi buffers from `ckpt['flow']` — *marik, 2026-06-03*
- [x] Create `configs/torsion_transformer_energy.yaml` + `torsion_transformer_local.yaml` — *marik, 2026-06-03*
- [x] Smoke test: zero-init, training loss, state_dict roundtrip, Heun sampling — all PASS — *marik, 2026-06-03*

## Completed (Phase 3: TorsionFlow — Riemannian flow matching, 2026-06-03)

- [x] Implement `models/internal_coords.py` — NeRF kinematics + cartesian↔internal conversion — *marik, 2026-06-03*
- [x] Implement `models/torsion_net.py` — MLP velocity network (4 layers, 256 hidden, CFG dropout) — *marik, 2026-06-03*
- [x] Implement `models/torsion_flow.py` — TorsionalFlowMatching (OT-CFM on torus, Heun ODE, velocity normalisation) — *marik, 2026-06-03*
- [x] Create training configs (`torsion_flow_energy.yaml`, `torsion_flow_local.yaml`) — *marik, 2026-06-03*
- [x] Implement `scripts/train_torsion.py` — training script with torsion-stat warmup — *marik, 2026-06-03*
- [x] Extend `scripts/evaluate.py` for `torsion_flow_energy` model type — *marik, 2026-06-03*
- [x] Sanity check (5 epochs, GPU): **100% bond validity, 0.0000 Å RMSE, MMD=0.032** — *marik, 2026-06-03* — all verified

## Completed (Phase 1: SE3flow next-gen, 2026-06-02)

- [x] Add AdaLN per-layer conditioning to SE3FlowLayer — *marik, 2026-06-02* — zero-init gates, cond_emb removed from edges
- [x] Add proper zero-init x1_pred head to SE3FlowEnergyNet — *marik, 2026-06-02* — always present, δ=0 at init, equivariance verified
- [x] Fix --finetune versioning bug in train_flow_energy.py — *marik, 2026-06-02* — fine-tune now always creates new version dir
- [x] Fix ZeroCoMGaussianDiffusion import bug in evaluate.py — *marik, 2026-06-02* — moved to top of load_model_from_ckpt
- [x] Train SE3flow AdaLN velocity model (500 epochs) — *marik, 2026-06-02* — val=0.6266, **35.2% bond validity** (slightly worse than v1 39.8%)
- [x] Attempt x1_pred fine-tuning (×5 runs) — *marik, 2026-06-02* — all stagnated at loss≈10.6; dropped
- [x] Restore cond_emb in SE3FlowLayer.phi_e (additive with AdaLN) — *marik, 2026-06-02* — fixed regression
- [x] Add AllAtomPhysics auto-selection to train_flow_energy.py — *marik, 2026-06-02* — n_residues≠10 uses AllAtomPhysics
- [x] Create se3flow_all_atom.yaml (baseline all-atom config) — *marik, 2026-06-02*

## Completed (Phase 2: Bond-graph prior + self-conditioning, 2026-06-02)

- [x] Create models/harmonic_prior.py — *marik, 2026-06-02* — vectorised Jacobi-SHAKE (150 iter), bond MAE=0 Å; + sample_ca_chain
- [x] Add prior_fn to ZeroCoMFlowMatching — *marik, 2026-06-02* — backward-compat; Gaussian fallback when None
- [x] Add self_cond to SE3FlowEnergyNet — *marik, 2026-06-02* — zero-init, equivariance 1.25e-07, zero-init diff 2.87e-07
- [x] Add self-conditioning pre-pass to training_loss_energy() — *marik, 2026-06-02* — 50% batches, stop-gradient
- [x] Add sc_x1 tracking to ddim_sample_cfg() — *marik, 2026-06-02* — per-step x̂₁ for next step
- [x] Wire harmonic_prior + self_cond flags in train_flow_energy.py — *marik, 2026-06-02*
- [x] Create configs/se3flow_all_atom_v2.yaml and _local.yaml — *marik, 2026-06-02*
- [x] Smoke test Phase 2 (5 epochs, Cα data) — *marik, 2026-06-02* — PASS; physics at ref floor, equivariance ✓

## x1_pred status — DROPPED
Root cause: incompatibility between 7-layer coordinate accumulation and x1_pred MSE.
Do not retry without fundamentally different output mechanism.

---

## Up Next

### Phase 2 training
- [x] `data_all_atom/` confirmed present (train.npz, valid.npz, test.npz) — *2026-06-03*
- [x] Fixed `evaluate.py` SE3FlowEnergyNet constructor missing `x1_pred` + `self_cond` — *marik, 2026-06-03*
- [x] Fixed `se3flow_all_atom_v2.yaml` val_path `val.npz` → `valid.npz` — *marik, 2026-06-03*
- [x] Fixed all-atom eval pipeline in `evaluate.py` + `analyze_energy_conditioning.py` — *marik, 2026-06-03*
  - Wired harmonic prior at eval time; use `ddim_sample_cfg`; added `compute_physics_metrics_aa`; auto test-path from config; `--test` optional
- [x] Run full evaluation of se3flow_all_atom_v2/v1 — *tkreusel, 2026-06-03* — 24.6% bond validity, phys_bond_rmse=0.264 Å; see STATUS.md
- [ ] Run temperature sweep for se3flow_all_atom_v2 — *tbd*
  ```bash
  python scripts/analyze_energy_conditioning.py --checkpoint checkpoints/se3flow_all_atom_v2/v1/best.pt --n 500 --steps 100 --guidance_scale 2.0 --save plots/se3flow_aa_v2_energy.png
  ```
- [ ] Resume all-atom v2 training to 500 epochs (stopped at 350) — low priority given all-atom approach seems limited
  ```bash
  python scripts/train_flow_energy.py --config configs/se3flow_all_atom_v2.yaml --resume checkpoints/se3flow_all_atom_v2/v1/epoch_0350.pt
  ```

### Infrastructure
- [ ] Fix `local_baseline.yaml` + `egnn_local.yaml` to use `data/val.npz` (currently reference `data/valid.npz`)
- [x] Delete `scripts/tmp.py` — one-off debugging script, pollutes repo

### Training: improve FlowMatch bond validity — physics constraints ready
Physics module implemented. Run the physics-constrained training:
```bash
python scripts/train_flow.py --config configs/flowmatch_physics.yaml
```
The log.jsonl will have `phys_bond`, `phys_clash`, `phys_angle` columns.  
Reference floor (real data): `phys_bond=0.0039`, `phys_clash=0.0`, `phys_angle=0.0298`.

- [ ] **Run full physics-constrained training** (500 epochs) and compare bond validity vs `flowmatch/v2`
- [ ] If bond validity is still low after 200 epochs, try increasing `physics_weight: 0.1`
- [ ] Also try higher LR (`lr: 3e-4`) — flow matching is less LR-sensitive than EGNN+DDPM
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

## Next directions for improving SE3flow bond validity

### Option A — Physics-guided sampling at inference (no retraining, test immediately)
Apply bond-physics gradient as a correction at each ODE step. Equivariant, no new training needed.
```python
# In ZeroCoMFlowMatching.ddim_sample_cfg, after each Heun step:
with torch.enable_grad():
    x_eval = x.detach().requires_grad_(True)
    phys = physics_fn(x_eval).sum()
    grad = torch.autograd.grad(phys, x_eval)[0]
x = x - physics_guidance_scale * grad
x = x - x.mean(dim=1, keepdim=True)  # re-zero CoM
```
Test on `checkpoints/se3flow_adaln_velocity/v1/best.pt`. Likely +5–15% bond validity with no retraining.

### Option B — Equivariant DDPM (switch framework, keep SE3 architecture)
Keep the SE3FlowLayer GNN architecture but train it as a DDPM noise predictor instead of flow matching.
This is exactly what EGNN+Physics does (60.3% bond validity). The x0_pred at each DDPM step
naturally learns correct geometry from the data distribution without needing x1_pred tricks.
New training script needed (`train_egnn_adaln.py`), reuse `models/diffusion_zerocom.py`.
Expected: **50–65% bond validity** — bridges the gap to EGNN+Physics.

### Option C — Restore cond_emb in edges + keep AdaLN (additive, not replacement)
The AdaLN run removed `cond_emb` from phi_e, which may have weakened bond constraint learning.
Try: keep both (`cond_emb` in edges AND AdaLN on nodes). 500-epoch run needed.
Low risk, simple config/code change, may recover the lost 4.6% from v1→AdaLN.

### Option D — Harmonic / bond-preserving noise prior
Replace standard Gaussian source with noise that already has correct bond lengths.
E.g. sample x0 with random orientations but fixed consecutive distances = 3.832 Å.
This removes the hard problem of learning bond geometry from scratch at every timestep.
Medium effort (new sampler), potentially high impact.

### Option E — Internal coordinates (ZMatrix)
Represent Chignolin as (bond lengths × 9, bond angles × 8, dihedrals × 7) instead of Cartesian.
Bond lengths can be fixed to 3.832 Å by construction → 100% bond validity by design.
High effort (new dataset, new model, loses equivariance) but eliminates the problem entirely.

## Ideas / Stretch Goals

- [ ] Try MLP model (`model_type: mlp`) — no full-training config yet, 5 min to write
- [ ] Add WandB logging to training scripts (`wandb/` already gitignored, clean to add)
- [x] Benchmark EGNN equivariance: run `model.check_equivariance()` on `egnn/v1` checkpoint — *marik, 2026-06-02* — superseded by `check_equivariance.py`
- [ ] ODE/DDIM step count sweep: compare quality at 10, 25, 50, 100, 200 steps

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
- [x] Add SE(3) equivariance analysis script (`scripts/check_equivariance.py`) — *marik, 2026-06-02* — 3 tests: score-network, full-pipeline, distribution isotropy; results in STATUS.md
