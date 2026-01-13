# Hypergraph Point-Cloud Registration

Hypergraph-based correspondence + registration for deforming and articulated 3D shapes.
Datasets: DeformingThings4D, FAUST, SAPIEN/PartNet-Mobility.

## Repo layout
- `src/project/` — core library (datasets, models, training/eval)
- `configs/` — YAML configs for datasets, baselines, models, runs
- `scripts/` — helper scripts (preprocess, sweeps)
- `results/` — experiment outputs (not committed)
- `docs/` — notes and reproducibility docs
- `paper/` — paper source + figures
- `demo/` — demo notebook / assets

## Setup
Create the environment (edit as needed):
- `conda env create -f environment.yml`
- `conda activate hyper-align`
