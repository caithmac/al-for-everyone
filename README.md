# Active Learning for Drug Discovery

A single Python script that uses Gaussian Processes to intelligently pick
which compounds to test next — no ML expertise needed.

## Quick Start

1. **Install dependencies** (once):
   ```
   pip install torch gpytorch rdkit-pypi pandas numpy scikit-learn scipy matplotlib tqdm
   ```

2. **Prepare your data** — a CSV file with at least two columns:
   - SMILES strings
   - Activity values (pIC50, IC50, DDG, %inhibition, etc.)

3. **Edit the config** at the top of `al_for_everyone.py`:
   - `DATA_PATH` — path to your CSV
   - `SMILES_COL` — name of the SMILES column
   - `VALUE_COL` — name of the activity column
   - `LOWER_IS_BETTER` — True if lower values = better (e.g., IC50, DDG)
   - Pick a protocol and kernel (defaults work well)

4. **Run:**
   ```
   python al_for_everyone.py
   ```

5. **Check `al_results/`** for outputs:
   - `cycle_results.csv` — per-cycle metrics
   - `selected_compounds.csv` — which compounds were picked
   - `recall_curves.png` — how many top compounds were found
   - `model_quality.png` — R², Spearman, RMSE over cycles
   - `pool_predictions.csv` — predictions on new molecules (optional)

## What's Inside

Everything is in ONE file — no imports from messy codebases:

| Section | What it does |
|---------|-------------|
| **TanimotoKernel** | Jaccard similarity kernel — the only one that works for fingerprints |
| **GP model** | Gaussian Process with constant mean |
| **Featurization** | SMILES → ECFP4 fingerprints (4096 bits) |
| **Acquisition** | UCB (α·μ + β·σ), Probability of Improvement, Expected Improvement |
| **Active learning** | Iterative: pick → train → predict → repeat |
| **Plots** | Recall curves, R²/Spearman/RMSE over cycles |
| **Pool prediction** | Predict activity for new, unlabeled compounds |

## Protocols

| Name | Strategy |
|------|----------|
| `random` | Random selection (baseline) |
| `ucb-balanced` | UCB with equal explore/exploit weight |
| `ucb-alternate` | Alternating explore/exploit cycles (best on TYK2) |
| `ucb-sandwich` | 2 explore → 6 exploit → 2 explore |
| `ucb-explore-heavy` | 7 explore → 3 exploit |
| `ucb-exploit-heavy` | 3 explore → 7 exploit |
| `ucb-gradual` | explore → UCB → exploit transition |

## Kernels

| Kernel | Best for |
|--------|----------|
| `tanimoto` | ECFP fingerprints (recommended!) |
| `rbf` | Continuous descriptors (not fingerprints) |
| `matern` | Smooth functions (not recommended for fingerprints) |

## Tips

- **Start with the defaults** — `ucb-alternate` + `tanimoto` works great
- **Small dataset?** Reduce `INITIAL_SIZE` to 20-30
- **Large dataset (>2000 compounds)?** Consider fewer cycles or a sparse GP
- **Check the uncertainty column** — high uncertainty = the model wants you to test that compound

## Troubleshooting

**"No module named rdkit"**
→ Install with conda: `conda install -c conda-forge rdkit`
  Or pip: `pip install rdkit-pypi`

**Out of memory**
→ Your dataset is too large. Reduce `N_CYCLES` or use fewer compounds.

**Slow featurization**
→ Normal for >10k compounds. ECFP computation takes time.

---

*Built for Lindsey & the chemistry team. May 2026.*
