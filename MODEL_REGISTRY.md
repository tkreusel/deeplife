# Model Registry

Quick-reference for all trained Chignolin models. Full metadata (architecture params, training config, all versions) is in [MODEL_REGISTRY.yaml](MODEL_REGISTRY.yaml).

**When you train a new model**, add an entry to both files — see the template at the bottom of `MODEL_REGISTRY.yaml` and the row template at the bottom of this table.

## How to evaluate any model

```bash
# Standard evaluation (auto-detects model type from checkpoint config):
python scripts/evaluate.py \
    --ckpt checkpoints/<family>/<ver>/best.pt \
    --test data/test.npz --n 1000 --steps 100

# Quick sample:
python scripts/quick_sample.py \
    --checkpoint checkpoints/<family>/<ver>/best.pt --n 50 --steps 100

# All-atom models (se3flow_all_atom_v2, egnn_*_aa): omit --test (auto-reads from config)
python scripts/evaluate.py --ckpt checkpoints/se3flow_all_atom_v2/v1/best.pt --n 1000 --steps 100

# Backbone models:
python scripts/evaluate.py --ckpt checkpoints/backbone_transformer/v1/best.pt \
    --test data_backbone/test.npz --n 1000 --steps 100
```

Reference (MD ensemble): Bond±0.5%=100%, Bond RMSE=0.063 Å, Rg=5.92 Å, ETE=12.0 Å, Diversity=3.79 Å

---

## Production & Notable Runs (≥100 training epochs)

| ID | Name | Framework | Data | Epochs | Best Val | Bond±0.5% | Rg (Å) | ETE (Å) | MMD | Status | Checkpoint |
|----|------|-----------|------|-------:|----------|----------:|-------:|--------:|-----|--------|------------|
| baseline/v3 | Transformer-DDPM | ddpm | Cα | 499 | 0.0891 | 47.2% | 4.93 | 6.9 | 0.0618 | production | `checkpoints/baseline/v3/best.pt` |
| baseline/v2 | Transformer-DDPM-LargeBatch | ddpm | Cα | 129 | 0.7495 | — | — | — | — | partial ⚠️ | `checkpoints/baseline/v2/best.pt` |
| baseline_transformed/v1 | Transformer-DDPM-SE3aug | ddpm | Cα | 469 | 0.1966 | — | — | — | — | production | `checkpoints/baseline_transformed/v1/best.pt` |
| transformer_adaln/v1 | AdaLN-Transformer | ddpm | Cα | 419 | 0.1836 | **75.4%** | 5.65 | 10.69 | 0.0323 | production | `checkpoints/transformer_adaln/v1/best.pt` |
| transformer_adaln_energy/v1 | AdaLN-Transformer+Energy | ddpm | Cα | 499 | 0.1764 | 74.8% | 5.62 | 10.43 | 0.0342 | production | `checkpoints/transformer_adaln_energy/v1/best.pt` |
| **transformer_adaln_energy_physics/v1** | **AdaLN+Energy+Physics** | ddpm | Cα | 409 | 0.1822 | **90.0%** | 5.73 | 10.82 | 0.0323 | **production ★** | `checkpoints/transformer_adaln_energy_physics/v1/best.pt` |
| transformer_adaln_sc/v1 | AdaLN+SelfCond | ddpm | Cα | 409 | 0.1178 | — | — | — | — | production | `checkpoints/transformer_adaln_sc/v1/best.pt` |
| transformer_adaln_sc/v2 | AdaLN+SelfCond | ddpm | Cα | 479 | 0.1169 | — | — | — | — | production | `checkpoints/transformer_adaln_sc/v2/best.pt` |
| transformer_adaln_sc/v3 | AdaLN+SelfCond | ddpm | Cα | 499 | 0.1826 | 89.7% | 6.52 | 14.69 | 0.0358 | production ⚠️ | `checkpoints/transformer_adaln_sc/v3/best.pt` |
| egnn/v1 | EGNN-DDPM | ddpm | Cα | 459 | 0.1779 | 15.7% | 5.84 | 11.97 | 0.0358 | production | `checkpoints/egnn/v1/best.pt` |
| egnn/v3 | EGNN-DDPM+Physics(0.05) | ddpm | Cα | 429 | 0.1968 | 44.2% | 5.87 | 12.16 | 0.0323 | production | `checkpoints/egnn/v3/best.pt` |
| egnn/v4 | EGNN-DDPM+Physics(0.10) | ddpm | Cα | 459 | 0.2035 | **58.8%** | 6.05 | 12.79 | 0.0342 | **production ★** | `checkpoints/egnn/v4/best.pt` |
| egnn_adaln/v1 | EGNN+AdaLN+Energy | ddpm | Cα | 499 | 0.2076 | 43.7% | 5.73 | **12.04** | 0.032 | production ⚠️ | `checkpoints/egnn_adaln/v1/best.pt` |
| egnn_adaln_aa/v3 | EGNN+AdaLN (all-atom) | ddpm | All-atom | 449 | 0.0634 | **0%** | — | — | — | failed ✗ | `checkpoints/egnn_adaln_aa/v3/best.pt` |
| egnn_energy_aa/v2 | EGNN+Energy (all-atom) | ddpm | All-atom | 179 | 0.0786 | — | — | — | — | partial | `checkpoints/egnn_energy_aa/v2/best.pt` |
| flowmatch/v2 | FlowMatch-EGNN | flow_match | Cα | 479 | 0.6351 | 5.0% | 5.96 | 12.4 | 0.0323 | production | `checkpoints/flowmatch/v2/best.pt` |
| flowmatch_physics/v3 | FlowMatch+Physics | flow_match | Cα | 489 | 0.6394 | 10.8% | 5.90 | — | — | production | `checkpoints/flowmatch_physics/v3/best.pt` |
| flowmatch_energy/v2 | FlowMatch+Energy | flow_match | Cα | 449 | 0.5989 | 16.0% | 5.90 | — | — | production | `checkpoints/flowmatch_energy/v2/best.pt` |
| **flowmatch_energy_physics/v1** | **FlowMatch+Energy+Physics** | flow_match | Cα | 449 | 0.6112 | **17.2%** | 5.89 | — | 0.032 | **production ★** | `checkpoints/flowmatch_energy_physics/v1/best.pt` |
| flowmatch_v2_energy/v2 | FlowMatch-v2+Energy (x1_pred) | flow_match | Cα | 449 | 10.607 | — | — | — | — | failed ✗ | `checkpoints/flowmatch_v2_energy/v2/best.pt` |
| se3flow_energy/v1 | SE3Flow+Energy (baseline) | flow_match | Cα | 459 | 0.6201 | 39.8% | 5.91 | — | — | production | `checkpoints/se3flow_energy/v1/best.pt` |
| se3flow_energy/v2 | SE3Flow+Energy (x1_pred, phys=0.20) | flow_match | Cα | 199 | 10.640 | — | — | — | — | failed ✗ | `checkpoints/se3flow_energy/v2/best.pt` |
| se3flow_energy/v3 | SE3Flow+Energy (x1_pred, phys=0.05) | flow_match | Cα | 139 | 10.785 | — | — | — | — | failed ✗ | `checkpoints/se3flow_energy/v3/best.pt` |
| se3flow_energy/v5 | SE3Flow+Energy (x1_pred, bug) | flow_match | Cα | 199 | 11.424 | — | — | — | — | failed ✗ | `checkpoints/se3flow_energy/v5/best.pt` |
| se3flow_energy_finetune/v1 | SE3Flow+Energy (finetune→x1_pred) | flow_match | Cα | 479 | 10.618 | — | — | — | — | failed ✗ | `checkpoints/se3flow_energy_finetune/v1/best.pt` |
| **se3flow_adaln_velocity/v1** | **SE3Flow+AdaLN (velocity)** | flow_match | Cα | 489 | 0.6266 | **35.2%** | 5.93 | 11.90 | 0.0323 | **production ★** | `checkpoints/se3flow_adaln_velocity/v1/best.pt` |
| **se3flow_all_atom_v2/v1** | **SE3Flow AllAtom+SHAKE+SC** | flow_match | All-atom | 339 | 0.0065 | **40%** | 5.94 | — | — | **production ★** | `checkpoints/se3flow_all_atom_v2/v1/best.pt` |
| **torsion_flow/v1** | **TorsionFlow-MLP** | torsion_flow | Torsion | 409 | 0.9298 | **100%** | 6.15 | 13.97 | 0.0349 | **production ★** | `checkpoints/torsion_flow/v1/best.pt` |
| **torsion_transformer/v1** | **TorsionTransformer** | torsion_flow | Torsion | 1999 | 0.7159 | **100%** | 6.18 | 14.14 | 0.0333 | **production ★** | `checkpoints/torsion_transformer/v1/best.pt` |
| backbone_transformer/v1 | BackboneTransformer (N/CA/C) | ddpm | Backbone | 479 | 0.0872 | — | — | — | — | production | `checkpoints/backbone_transformer/v1/best.pt` |
| backbone_ipa/v1 | BackboneIPA (IPA+TorsionFlow) | backbone_ipa | Backbone-φψ | 1176 | 0.5506 | — | — | — | — | in_progress | `checkpoints/backbone_ipa/v1/best.pt` |
| backbone_ipa/v8 | BackboneIPA v8 (3000+ epochs) | backbone_ipa | Backbone-φψ | 3039 | **0.5231** | **100%** | 6.16 | 14.38 | **0.0345** | in_progress | `checkpoints/backbone_ipa/v8/best.pt` |

**★ = recommended checkpoint for that family**
**⚠️ = completed training but known structural issues**
**✗ = training stagnated or catastrophic failure**

---

## Smoke Tests (< 20 epochs; for verification only, not evaluation)

| ID | Description | Checkpoint |
|----|-------------|------------|
| backbone_ipa_local/v1 | BackboneIPA CPU smoke test | `checkpoints/backbone_ipa_local/v1/best.pt` |
| backbone_transformer_local/v1 | BackboneTransformer CPU smoke test | `checkpoints/backbone_transformer_local/v1/best.pt` |
| egnn_adaln_aa/v2 | EGNN+AdaLN all-atom, 9 epochs | `checkpoints/egnn_adaln_aa/v2/best.pt` |
| egnn_energy_aa_local/v2 | EGNN+Energy all-atom CPU smoke test | `checkpoints/egnn_energy_aa_local/v2/best.pt` |
| flowmatch/v1 | FlowMatch CPU smoke test | `checkpoints/flowmatch/v1/best.pt` |
| flowmatch_energy/v1 | FlowMatch+Energy CPU smoke test | `checkpoints/flowmatch_energy/v1/best.pt` |
| flowmatch_physics/v2 | FlowMatch+Physics CPU smoke test | `checkpoints/flowmatch_physics/v2/best.pt` |
| flowmatch_v2_energy/v1 | FlowMatch-v2 CPU smoke test (confirms x1_pred stagnation) | `checkpoints/flowmatch_v2_energy/v1/best.pt` |
| se3flow_energy/v4 | SE3Flow+AdaLN velocity, interrupted at epoch 49 | `checkpoints/se3flow_energy/v4/best.pt` |
| se3flow_sc_local/v2–v4 | SE3Flow+SelfCond CPU smoke tests | `checkpoints/se3flow_sc_local/v{2,3,4}/best.pt` |
| torsion_flow_local/v1 | TorsionFlow CPU smoke test | `checkpoints/torsion_flow_local/v1/best.pt` |
| baseline/v1 | Transformer-DDPM, stopped at epoch 19 | `checkpoints/baseline/v1/best.pt` |

---

## Architecture Summary

| `model_type` | Base | Architecture | Score network | Diffusion/Flow | SE(3) equivariant | Params |
|---|---|---|---|---|---|---:|
| `transformer` | Transformer | 4-layer Transformer, residues as tokens | `TransformerScoreNetwork` | DDPM | No | 0.85M |
| `transformer_adaln` | Transformer | AdaLN-Zero Transformer | `AdaLNTransformerScoreNetwork` | DDPM | No | 7.32M |
| `transformer_adaln_energy` | Transformer | AdaLN + energy CFG | same | DDPM | No | 7.45M |
| `transformer_adaln_sc` | Transformer | AdaLN + self-conditioning | `AdaLNSCScoreNetwork` | DDPM | No | 7.45M |
| `egnn` | EGNN | SE(3)-equivariant GNN | `EGNNScoreNetwork` | ZeroCoM-DDPM | Yes | 0.44M |
| `egnn_adaln` | EGNN | EGNN + AdaLN + energy | `EGNNAdaLNScoreNetwork` | ZeroCoM-DDPM | Yes | 0.64M |
| `egnn_energy` | EGNN | EGNN + energy CFG | `EGNNAdaLNScoreNetwork` | ZeroCoM-DDPM | Yes | 0.64M |
| `flowmatch` | EGNN | EGNN OT-CFM | `EGNN` | ZeroCoM-FlowMatch | Yes | 0.44M |
| `flowmatch_energy` | EGNN | EGNN OT-CFM + energy CFG | — | ZeroCoM-FlowMatch | Yes | 0.64M |
| `flowmatch_v2_energy` | EGNN | EGNNv2 OT-CFM + energy + x1_pred | — | ZeroCoM-FlowMatch | Yes | 1.71M |
| `se3flow_energy` | EGNN | SE3FlowEnergyNet (AdaLN variant) | `SE3FlowEnergyNet` | ZeroCoM-FlowMatch | Yes | 1.71M |
| `torsion_flow_energy` | MLP | MLP on (θ,φ) dihedral torus | `TorsionFlowNet` | TorsionalFlowMatch | Invariant | 0.30M |
| `torsion_transformer_energy` | Transformer | Transformer on (θ,φ) tokens | `TorsionTransformerNet` | TorsionalFlowMatch | Invariant | 4.77M |
| `backbone_transformer` | Transformer | 30-token AdaLN Transformer (N/CA/C) | `BackboneTransformerScoreNetwork` | DDPM | No | 7.45M |
| `backbone_ipa_energy` | Transformer | IPA geometric attention + torsion flow | `BackboneIPAFlowNet` | BackboneTorsionalFlow | Invariant | 4.06M |

---

## Adding a new model

After training, run `evaluate.py` then add a row here and a full entry in `MODEL_REGISTRY.yaml`.

```
| <family>/<ver> | <ShortName> | <framework> | <data> | <epochs> | <val_loss> | <bond%> | <rg> | <ete> | <mmd> | <status> | `checkpoints/<family>/<ver>/best.pt` |
```
