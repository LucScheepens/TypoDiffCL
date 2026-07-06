# TypoDiffCL — Diffusion-Guided Graph Augmentation for AML/Fraud Detection

This directory contains the core thesis codebase: a graph diffusion model that
generates synthetic transaction subgraphs, used to guide SimCLR-style
contrastive pretraining of a graph encoder for anti-money-laundering (AML) /
fraud detection on transaction networks. The method is referred to as
**TypoDiffCL** throughout the code (see `--augment-method diffusion`).

Pipeline, at a glance:

1. **Diffusion model** (`diffusion/`) is trained to denoise/generate small
   transaction subgraphs, conditioned on structural + AML pattern features
   (fan-in/out, stacking, cycles, scatter-gather, bipartite score — computed
   by `patterns/detector.py`).
2. **Guided generation** (`generation/generation.py`,
   `generation/latent_seed_generation.py`) uses the trained diffusion model
   to produce novel synthetic laundering/fraud subgraphs as augmentation
   views.
3. **SimCLR contrastive pretraining** (`simclr/`) trains a graph encoder
   using both classic structural augmentations (subgraph crop, edge/node
   drop, node addition — `simclr/augmentation.py`) and diffusion-generated
   views as positive pairs, with an optional supervised contrastive (SupCon)
   term.
4. **Downstream evaluation** (`generation/evaluate_classifiers.py`) trains
   several classifiers (GIN, GraphTransformer, GraphSAGE, DeepSets, FraudGT,
   ExSTraQt) on top of the pretrained encoder, with/without augmentation, and
   compares against baseline augmentation methods (GAN, GraphSMOTE, DiGa) —
   this script is the shared evaluation engine invoked by nearly every
   experiment driver below.

## Datasets

Three datasets are supported (`--dataset {elliptic,ibm,ethereum}`). Raw data
is **not** stored in this directory — it is expected one level up, at the
repo root (`grad/grad/data/`):

| Dataset | Expected path (repo root) |
|---|---|
| Elliptic Bitcoin | `data/elliptic_bitcoin_dataset/` (`elliptic_txs_classes.csv`, `elliptic_txs_edgelist.csv`, `elliptic_txs_features.csv`) |
| IBM AML | `data/IBM/<HI\|LI>-<Small\|Medium\|Large>_Trans.csv` |
| Ethereum Phishing | `data/Ethereum Phishing Transaction Network/` (`MulDiGraph.pkl`) |

`igraph_version/data/` is a **separate** folder — it only holds
dataset adapters (`elliptic_adapter.py`, `ethereum_adapter.py`) and derived
caches (preprocessed tensors, `gen_cache_*.pkl` generation outputs). It is
not where raw datasets live.

## Setup

Dependencies are declared in the repo-root `requirements.txt`
(`grad/grad/requirements.txt`); install from there:

```bash
pip install -r ../requirements.txt
```

Key third-party packages used in this directory: `torch`, `torch_geometric`,
`igraph`, `networkx`, `scikit-learn`, `xgboost`, `optuna`, `umap-learn`,
`matplotlib`.

## Usage

All commands below are run from inside `igraph_version/`.

### Train the diffusion model / SimCLR encoder

```bash
python train.py --dataset elliptic --phase all       # diffusion + simclr
python train.py --dataset ibm --phase diffusion       # diffusion only
python train.py --dataset ibm --phase simclr --ckpt-dir checkpoints/simclr_ibm
```

### Single-condition ablation training

Used internally by `run_ablation.py`, but runnable standalone to retrain one
encoder variant:

```bash
python elliptic_simclr_train_ablation.py --condition full
python elliptic_simclr_train_ablation.py --condition no_diffusion --p-diffusion 0.0
python ibm_simclr_train_ablation.py --condition diffusion_aug_only --p-diffusion 1.0
```

### Full ablation study

```bash
python run_ablation.py --dataset elliptic --n-gen 40
python run_ablation.py --dataset ibm --n-gen 40 --pattern-only   # IBM AML pattern-feature ablations only
python run_ablation.py --dataset elliptic --n-gen 10 --epochs 10 --low-data 0.2 --skip-training  # quick smoke test
```

Results are written as CSVs under `results/`; summarize/plot with:

```bash
python show_ablation.py --dataset elliptic
python plot_ablation.py --dataset elliptic --metric auc
```

### Sensitivity / sweep experiments

These three drivers are **not** CLI-configurable — edit the constants
(`DEPTHS`, `T0_VALUES`, `NGEN_VALUES`, model/seed lists) near the top of the
file, then run:

```bash
python run_bfs_depth_experiment.py        # Elliptic: BFS ego-subgraph depth sweep (2/3/4/6 hops)
python run_t0_sensitivity_experiment.py   # diffusion reverse-process start step t0 sweep
python run_ngen_sweep_experiment.py       # Ethereum: number of generated augmentations sweep
```

Each shells out to `generation/evaluate_classifiers.py`, which does expose a
full CLI (`--dataset`, `--augment`, `--augment-method {diffusion,gan,graphsmote,diga}`,
`--n-gen`, `--low-data`, `--models`, `--n-runs`, ...) if you want to run
single evaluation conditions directly.

### Visualization

```bash
python visualize_augmentations.py --out figures        # illustrate the 4 SimCLR structural augmentations
python viz_real_generated.py --out figures_real         # real vs. augmented vs. diffusion-generated subgraphs
python _run_pipeline.py                                 # rebuild generation_pipeline.png from cache only
python _run_gallery.py                                  # rebuild generated_gallery.png from cache only
```

