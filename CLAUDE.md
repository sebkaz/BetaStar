# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

BetaStar is a Kedro 0.19.10 pipeline for **community-aware node classification in graphs**. It computes graph-structural features (beta_star, participation, in_mod_deg, etc.) and evaluates how well various embeddings and classifiers recover community structure across 11 datasets (Facebook, Reddit, LastFM, Twitch, Amazon, Grid, and synthetic ABCD+o graphs x3–x7 o1000).

## Commands

### Running Pipelines

```bash
# Run a specific pipeline on a specific dataset
kedro run --pipeline=emb --env=amazon
kedro run --pipeline=t1 --env=reddit
kedro run --pipeline=t2 --env=lastfm
kedro run --pipeline=t3 --env=twitch

# Default env is Facebook (conf/base/)
kedro run --pipeline=t1

# Run all pipelines on all 10 datasets (parallel)
make run-all

# Run a single pipeline type on all datasets
make run-all-t1
make run-all-t2
make run-all-t3
```

### Struc2Vec Embeddings (run before `emb` pipeline)

```bash
python struc2vec/src/main.py \
  --input struc2vec/graph/beta/facebook.csv \
  --output struc2vec/emb/facebook.emb \
  --num-walks 10 --walk-length 50 --window-size 5 --dimensions 16 \
  --OPT1 True --OPT2 True --OPT3 True --worker 1
```

### Linting

```bash
ruff check src/
isort --check-only src/
```

### Jupyter

```bash
kedro jupyter lab
```

## Architecture

### Pipeline Flow

```
01_raw (edges + node features)
  └─► [emb pipeline] preprocess_graph → node2vec_embedding → concat_data → NaN_preprocessing
        └─► 03_embeddings/
              └─► [t1] regression on 13 targets (R², Spearman, Kendall) → 05_reporting/
              └─► [t2] classification (ROC-AUC, AvgPrecision) → 05_reporting/
              └─► [t3] permutation feature importance → 05_reporting/
```

### Pipeline Registry (`src/betastar/pipeline_registry.py`)

| Key | Pipeline | Purpose |
|-----|----------|---------|
| `emb` | embeddings | Preprocessing + node2vec + struc2vec concat |
| `t1` | task1 | Regression on structural targets |
| `t2` | task2 | Binary node classification |
| `t3` | task3 | Permutation feature importance |

### Features Used

- **8 community-free graph features**: `lcc`, `bc`, `cc`, `dc`, `ndc`, `ec`, `eccen`, `core`
- **2 community-aware features** (test features): `participation`, `in_mod_deg`
- **Embeddings**: node2vec (16D, fastnode2vec) + struc2vec (16D, external script)
- **Regression targets** (13): `cada`, `edge_contr`, `in_mod_deg`, `participation`, `beta_star`, `l1`, `l2`, `kl`, `hd`, `l12`, `l22`, `kl2`, `hd2`

### Multi-Environment Configuration

Each dataset has its own Kedro environment in `conf/<dataset>/` that overrides `conf/base/catalog.yml` to point to the correct raw data files. Parameters in `conf/base/parameters.yml` are shared across all environments.

### Struc2Vec

The `struc2vec/` directory contains a modified struc2vec implementation. It must be run **manually** before the `emb` pipeline, as Kedro treats the `.emb` output as a raw input (`03_embeddings/<dataset>_struc2vec.emb`). The pipeline reads it via `_preprocess_struc()` in `embeddings/nodes.py`.

### Julia Code

`julia_codes/compute_stats.jl` computes the ground-truth node statistics (beta_star, participation, in_mod_deg, etc.) from raw edge lists. Uses PyCall to interface with Python/NetworkX. Run this once per dataset to generate features in `01_raw/data/`.

### Data Layout

```
data/
  01_raw/edges/   – edge lists (parquet)
  01_raw/data/    – node features incl. community targets (parquet)
  02_graphs/      – preprocessed graphs
  03_embeddings/  – node2vec and struc2vec embeddings
  04_models/      – train/test splits and trained models (pickle)
  05_reporting/   – result DataFrames (parquet)
```
