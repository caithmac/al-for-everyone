# Active Learning for Drug Discovery

Pick the best compounds from your virtual library for testing.
Choose your model, kernel, and protocol — or use the defaults.

## Quick start

```bash
# 1. Create environment
conda create -n al python=3.11 -y
conda activate al

# 2. Install dependencies (choose one or all)

# Basic (ECFP fingerprints + GP or RF — works for most users):
pip install torch gpytorch rdkit-pypi pandas numpy scikit-learn scipy matplotlib tqdm

# CheMeleon (better accuracy, optional):
pip install "chemprop>=2.2.0"

# 3. Run
python al_for_everyone.py --data my_compounds.csv --model gp
```

## CLI usage

```bash
# Minimal
python al_for_everyone.py --data my_data.csv

# Choose model, kernel, protocol
python al_for_everyone.py --data my_data.csv --model chemeleon \
    --protocol ucb-alternate --lower=True

# Screen a virtual library
python al_for_everyone.py --data measured.csv --library virtual_library.csv \
    --model rf --top_n 500

# Run GP + RF + CheMeleon and rank by consensus
python al_for_everyone.py --data measured.csv --library virtual_library.csv \
    --model consensus --top_n 500

# Use a config file (recommended for repeatability)
python al_for_everyone.py --config config.txt
```

### Config file format (`config.txt`)
```
--data my_compounds.csv
--smiles_col SMILES
--val_col calc_DDG_kcal
--lower=True
--model gp
--kernel tanimoto
--protocol ucb-alternate
--init_size 60
--batch_size 30
--n_rounds 10
--library virtual_library.csv
--library_smi_col smiles
--top_n 1000
--out al_results
```
Usage: `python al_for_everyone.py --config config.txt`

## All options

| Flag | Default | Choices | Description |
|------|---------|---------|-------------|
| `--data` | `my_compounds.csv` | | Your CSV with measured compounds |
| `--smiles_col` | `SMILES` | | Column with SMILES |
| `--val_col` | `calc_DDG_kcal` | | Column with measured activity |
| `--lower` | `True` | `True`/`False` | True = lower is better (DDG, IC50) |
| `--library` | (empty) | | CSV of untested SMILES to screen |
| `--library_smi_col` | `smiles` | | SMILES column in library |
| `--top_n` | `1000` | | Top candidates to save |
| `--model` | `gp` | `gp`, `rf`, `chemeleon`, `consensus` | Predictor model |
| `--kernel` | `tanimoto` | `tanimoto`, `rbf`, `matern` | GP kernel (only for `--model gp`) |
| `--protocol` | `ucb-alternate` | See below | Selection strategy |
| `--ucb_beta` | `2.0` | | Exploration weight |
| `--init_size` | `60` | | Random start size |
| `--batch_size` | `30` | | Picks per round |
| `--n_rounds` | `10` | | Selection rounds |
| `--gp_epochs` | `150` | | GP training epochs |
| `--gp_lr` | `0.01` | | GP learning rate |
| `--gp_lr_decay` | `0.95` | | GP LR decay |
| `--rf_trees` | `500` | | RF trees |
| `--out` | `al_results` | | Output folder |
| `--seed` | `7` | | Random seed |

### Protocols

| Protocol | Strategy |
|----------|----------|
| `random` | Pick randomly |
| `ucb-balanced` | Always UCB (α=1, β=1) |
| `ucb-alternate` | **Default.** Explore ↔ exploit, alternating |
| `ucb-sandwich` | Explore → exploit → explore |
| `ucb-explore-heavy` | More exploration |
| `ucb-exploit-heavy` | More exploitation |
| `ucb-gradual` | Explore → UCB → exploit |

### Models

| Model | Description | Needs |
|-------|-------------|-------|
| `gp` | **Default.** Gaussian Process with Tanimoto kernel on ECFP fingerprints. Best for small datasets (<2000). | `torch gpytorch rdkit-pypi` |
| `rf` | Random Forest on ECFP fingerprints. Fast, no GPU needed. | `rdkit-pypi scikit-learn` |
| `chemeleon` | **Best accuracy.** Pretrained CheMeleon fingerprints + Random Forest. R² up to 0.575 vs 0.438 for GP. | `chemprop>=2.2.0` |
| `consensus` | Runs GP, RF, and CheMeleon, then averages normalized library predictions. | All dependencies above |

## Output

| File | Contents |
|------|----------|
| `al_results/al_summary.csv` | Round-by-round metrics (R², recall, picks) |
| `al_results/honest_test.csv` | Predicted vs true on compounds the model never saw |
| `al_results/top_candidates.csv` | Ranked library compounds, sorted by predicted activity |
| `al_results/full_ranked_library.csv` | Full per-model and consensus ranking (`--model consensus`) |

## Files in this folder

| File | Purpose |
|------|---------|
| `al_for_everyone.py` | The script — everything in one file |
| `chemeleon_repo/` | CheMeleon fingerprint code (needed for `--model chemeleon`) |
| `README.md` | This file |

## CheMeleon setup (optional)

Install Chemprop 2.2 or newer. The included helper downloads the pretrained
CheMeleon checkpoint from Zenodo on first use and caches it in `~/.chemprop`.

```bash
pip install "chemprop>=2.2.0"
```

Then run:
```bash
python al_for_everyone.py --data my.csv --model chemeleon
```
