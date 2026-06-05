# deeplife
Project for the master seminar DeepLife - AI Methods for Advanced Molecular Modelling
(https://deeplife4eu.github.io/)

Generative models for sampling Chignolin protein backbone conformations — comparing Cartesian DDPM, SE(3)-equivariant flow matching, and torsion-space representations across 24 trained variants.

---

## Models

| Model | Architecture |
|---|---|
| Transformer-DDPM | Transformer + DDPM, Cα |
| AdaLN-Transformer + Physics | AdaLN Transformer + physics loss, Cα |
| EGNN-DDPM + Physics | SE(3)-equivariant EGNN, Cα |
| FlowMatch + Energy + Physics | OT-CFM EGNN + physics loss, Cα |
| SE3Flow-AdaLN-Velocity | SE(3)-equivariant OT-CFM, Cα |
| TorsionTransformer | Transformer flow on φ/ψ angles, Cα |
| BackboneTransformer | AdaLN Transformer, N-Cα-C |
| BackboneIPA-v8 | IPA flow on backbone φ/ψ, N-Cα-C |

---

## Setup

```bash
git clone https://github.com/tkreusel/deeplife.git && cd deeplife
conda env create -f environment.yml && conda activate deeplife
# Data taken from Wang et al. 2023, Scientific data (https://doi.org/10.1038/s41597-023-02465-9)
```
---

## Repository layout

```
models/        score networks, diffusion, and flow-matching frameworks
scripts/       training, evaluation, and analysis scripts
  eval_v2/     unified eval pipeline (physics · SE3 · energy · novelty · PyRosetta)
configs/       one YAML config per model variant
data/          dataset loader and SE(3) data augmentation
COLLAB/        team docs: SETUP.md · WORKFLOW.md · STATUS.md · TODO.md
plots/
  eval_overnight/  canonical evaluation outputs (24 models × 5 sections)
  training/        loss curves
  equivariance/    SE(3) equivariance tests
  energy/          temperature sweeps and energy conditioning
  novelty/         NND / diversity analysis
  model_evals/     per-model and comparison plots
```

Full model registry: [MODEL_REGISTRY.md]
