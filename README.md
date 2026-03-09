# Hypergraph Point-Cloud Registration

Aligning 3D point clouds of deforming or articulated objects is challenging because standard rigid-body methods fail when the underlying shape changes between scans. This project tackles **non-rigid point-cloud registration** by learning higher-order relationships among point groups through hypergraph neural networks.

We represent each point cloud as a hypergraph, where hyperedges capture local geometric structure (clusters of nearby points). A graph neural network (HyperGNN) learns to match hyperedges across source and target clouds, producing a set of correspondence hypotheses that are then refined into a final 6-DoF transformation. We evaluate on two benchmarks:

- **FAUST** — 300 scans of 10 human subjects in varied poses; tests inter- and intra-subject alignment.
- **PartNet-Mobility (SAPIEN)** — articulated objects (chairs, cabinets, etc.) under rigid and affine deformations at multiple noise levels.

A classical ICP baseline and a spectral-matching baseline (HyperGCT) are included for comparison.

---

## Repository layout

```text
src/project/          # Core library
  config/             # YAML configs per model × dataset
  dataset.py          # Dataset loaders (FAUST, PartNet)
  hypergnn_main.py    # HyperGNN train / eval entry point
  hypergct_main.py    # HyperGCT train / eval entry point
  models/             # Model definitions and losses
    hypergnn/         # HyperGNN (Wasserstein aggregation)
    hypergct/         # HyperGCT (spectral matching)
    baselines/        # ICP baseline
  trainers/           # Training loops
  utils/              # SE3, FPFH, correspondence utils, data processing
  eval/               # Evaluation notebook (visualize_results.ipynb)
configs/              # Top-level run configs (sweeps)
scripts/              # Preprocessing + sweep runner scripts
docs/                 # Project website + result tables/plots
```

## Data layout

```text
data/
|-- faust/
|   |-- corres/
|   |   |-- maps/
|   |   `-- pairs/
|   |-- fpfh/
|   |-- point_clouds/
|   |   |-- test/
|   |   `-- train/
|   |-- test/
|   |   |-- challenge_pairs/
|   |   `-- scans/
|   `-- training/
|       |-- ground_truth_vertices/
|       |-- registrations/
|       `-- scans/
`-- partnet/
    |-- affine_1-5/
    |-- affine_2-0/
    |-- fpfh_rigid/
    |-- fpfh_affine_1-5/
    |-- fpfh_affine_2-0/
    |-- point_clouds/
    `-- rigid/
```

---

## Setup

### 1. Create the environment

Two environment files are provided — pick the one matching your CUDA version:

```bash
conda env create -f environment.yml        # CUDA 12.8 (default)
conda env create -f environment_cu126.yml  # CUDA 12.6
conda activate hyperalign
```

### 2. Install the package (editable)

```bash
pip install -e .
```

### 3. Download datasets

**FAUST** — download the training and test scans from [http://faust.is.tue.mpg.de](http://faust.is.tue.mpg.de) and place the raw files under `data/faust/` following the data layout above.

**PartNet-Mobility** — download object point clouds from [https://sapien.ucsd.edu/downloads](https://sapien.ucsd.edu/downloads) and place them under `data/partnet/point_clouds/`.

### 4. Set data path environment variable

```bash
export DATA_PATH=data          # adjust to your absolute path if needed
```

---

## Preprocessing

### FAUST

```bash
bash scripts/process_faust.sh
```

This generates point clouds, ground-truth correspondence maps/pairs, and FPFH features under `data/faust/`.

### PartNet

```bash
bash scripts/process_partnet.sh
```

This augments raw point clouds into rigid / affine-1.5 / affine-2.0 variants and computes FPFH features for each.

---

## Running experiments

### Train HyperGNN

**FAUST:**
```bash
bash scripts/hypergnn_faust.sh
```

**PartNet (rigid):**
```bash
bash scripts/hypergnn_partnet.sh
```

Checkpoints are saved to `snapshot/`. Pass `--use_wandb` to log metrics to Weights & Biases.

### Evaluate HyperGNN

```bash
# FAUST
export MODEL_PATH=snapshot/<run_id>
bash scripts/hypergnn_faust_eval.sh

# PartNet
export MODEL_PATH=snapshot/<run_id>
bash scripts/hypergnn_partnet_eval.sh
```

### Train / evaluate HyperGCT

```bash
bash scripts/hypergct_faust.sh
# or
bash scripts/hypergct_partnet.sh
```

### ICP baseline sweep

```bash
python scripts/icp_sweep.py --out results/icp_sweep.csv --repeats 5
```

---

## Expected outputs

After evaluation the scripts print a per-pair CSV and a summary table. Typical metrics reported:

| Metric | Description |
|---|---|
| **L1 error** | Mean absolute coordinate error after alignment (lower is better) |
| **Rotation error (RE)** | Mean angular error in degrees (threshold: 15°) |
| **Translation error (TE)** | Mean translation error in cm (threshold: 30 cm) |
| **Inlier ratio** | Fraction of predicted correspondences within the inlier threshold |

Pre-computed result tables and Wasserstein metric plots are available in [docs/](docs/index.html).

---

## Future work

- [ ] **Systematic layer-placement study** — the current results show that Wasserstein pooling layer placement matters, but only a limited set of configurations was tested. A more exhaustive grid or search over placement patterns is needed to determine whether the observed trends generalize.
- [ ] **Runtime and memory benchmarking** — Wasserstein-based aggregation is more expensive than mean pooling, but this overhead has not been quantified. Future work should profile wall-clock time and peak memory usage and weigh them against accuracy gains to inform practical adoption.
- [ ] **Broader benchmark evaluation** — experiments are currently limited to FAUST and PartNet transformation regimes (rigid, affine-1.5, affine-2.0). Evaluating on settings with heavy noise, partial overlap, outliers, or large-scale scenes would better characterize the method's real-world applicability.
- [ ] **Mechanistic understanding** — structural diagnostics show how representations change under Wasserstein aggregation, but do not fully explain *why* certain configurations yield better final alignment. Closing this gap would require analysis linking representation geometry to downstream registration accuracy.
