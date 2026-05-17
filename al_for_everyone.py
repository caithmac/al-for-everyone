#!/usr/bin/env python
"""
Active Learning for Drug Discovery
===================================
Self-contained script for chemists. No dependencies on explainable_al package.

HOW TO USE:
  1. Edit the CONFIG section below (file paths, column names, settings)
  2. Run: python al_for_everyone.py

WHAT IT DOES:
  - Loads your CSV of compounds (SMILES + activity values)
  - Computes ECFP4 fingerprints
  - Runs active learning simulation (GP model picks compounds intelligently)
  - Shows recall curves, R², Spearman, RMSE
  - Optionally predicts activity for new, unlabeled compounds
  - Saves everything to al_results/

DEPENDENCIES:
  pip install torch gpytorch rdkit-pypi pandas numpy scikit-learn scipy matplotlib tqdm
  (Or use the 'al' conda environment if you have it)

Author: Satya / MJ
Date: May 2026
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import torch
import gpytorch
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import r2_score
from scipy.stats import spearmanr
from torch.distributions import Normal

# ── Quiet mode ──────────────────────────────────────────────────────────────
warnings.filterwarnings("ignore")
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")
from rdkit import Chem
from rdkit.Chem import AllChem

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG — Edit this section for your data
# ═══════════════════════════════════════════════════════════════════════════════

DATA_PATH      = "your_compounds.csv"    # Path to your CSV
SMILES_COL     = "SMILES"                # Column with SMILES strings
VALUE_COL      = "affinity"              # Column with activity values
LOWER_IS_BETTER = False                  # True if lower = better (e.g., DDG, IC50)
                                         # False if higher = better (e.g., pIC50)

# ── Active Learning Settings ─────────────────────────────────────────────────
PROTOCOL       = "ucb-alternate"         # random | ucb-balanced | ucb-alternate |
                                         # ucb-sandwich | ucb-explore-heavy |
                                         # ucb-exploit-heavy | ucb-gradual
KERNEL         = "tanimoto"              # tanimoto | rbf | matern
INITIAL_SIZE   = 60                      # Random compounds to start with
BATCH_SIZE     = 30                      # Compounds to pick each cycle
N_CYCLES       = 10                      # Number of acquisition cycles
EPOCHS         = 150                     # GP training epochs per cycle
LR             = 0.01                    # Learning rate
LR_DECAY       = 0.95                    # LR decay per epoch

# ── Optional: Predict on new molecules ───────────────────────────────────────
PREDICT_PATH       = ""                  # Path to CSV of unlabeled SMILES
PREDICT_SMILES_COL = "smiles"            # Column name for SMILES in that file

# ── Output ───────────────────────────────────────────────────────────────────
OUT_DIR        = "al_results"            # Where to save results

# ═══════════════════════════════════════════════════════════════════════════════
# CORE: Tanimoto Kernel + GP Model
# ═══════════════════════════════════════════════════════════════════════════════

class TanimotoKernel(gpytorch.kernels.Kernel):
    """Tanimoto (Jaccard) kernel for binary fingerprint vectors.

    K(x1, x2) = (x1 · x2) / (||x1||² + ||x2||² - x1 · x2)

    This is the only kernel that makes physical sense for ECFP fingerprints.
    RBF/Matern use Euclidean distance, which is meaningless for binary vectors.
    """

    def forward(self, x1, x2, diag=False, **params):
        if diag:
            return torch.ones_like(x1[:, 0])
        x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
        x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
        x1_dot_x2 = torch.matmul(x1, x2.transpose(-1, -2))
        denominator = x1_norm + x2_norm.transpose(-1, -2) - x1_dot_x2
        return x1_dot_x2 / denominator.clamp(min=1e-9)


class GPRegressionModel(gpytorch.models.ExactGP):
    """GP regression model with constant mean and configurable kernel."""

    def __init__(self, train_x, train_y, likelihood, kernel=None):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        if kernel is None:
            self.covar_module = TanimotoKernel()
        elif hasattr(kernel, 'get_kernel'):
            self.covar_module = kernel.get_kernel()
        else:
            self.covar_module = kernel

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(
            mean_x, covar_x.add_jitter(1e-6)
        )

# ═══════════════════════════════════════════════════════════════════════════════
# TRAINING
# ═══════════════════════════════════════════════════════════════════════════════

def train_gp_model(train_x, train_y, likelihood, model,
                   epochs=50, lr=0.1, lr_decay=0.95):
    """Train a GP model using Exact Marginal Log Likelihood + Adam."""
    device = train_x.device
    model = model.to(device)
    likelihood = likelihood.to(device)

    model.train()
    likelihood.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_decay)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    losses = []
    for i in range(epochs):
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        losses.append(loss.item())
        if (i + 1) % 10 == 0:
            print(f"  Epoch {i+1}/{epochs} | Loss: {loss.item():.3f}")
        loss.backward()
        optimizer.step()
        scheduler.step()

    return model, likelihood, losses

# ═══════════════════════════════════════════════════════════════════════════════
# FEATURIZATION
# ═══════════════════════════════════════════════════════════════════════════════

def smiles_to_ecfp8(smiles_list, radius=4, nBits=4096):
    """Convert a list of SMILES strings to ECFP fingerprints.

    Returns:
        np.ndarray of shape (N, nBits), dtype=int8
    """
    fingerprints = []
    for smiles in tqdm(smiles_list, desc="Computing ECFP fingerprints"):
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            fp = AllChem.GetMorganFingerprintAsBitVect(
                mol, radius=radius, nBits=nBits
            )
            arr = np.zeros((nBits,), dtype=np.int8)
            AllChem.DataStructs.ConvertToNumpyArray(fp, arr)
            fingerprints.append(arr)
        else:
            fingerprints.append(np.zeros((nBits,), dtype=np.int8))
    return np.vstack(fingerprints)

# ═══════════════════════════════════════════════════════════════════════════════
# ACQUISITION FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def ucb_selection(fingerprints, model, likelihood, batch_size,
                  alpha=1.0, beta=1.0, already_selected=None):
    """Upper Confidence Bound — pick compounds with highest α·μ + β·σ."""
    already_selected = already_selected or []
    model.eval()
    likelihood.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        pool_idx = list(set(range(len(fingerprints))) - set(already_selected))
        device = next(model.parameters()).device
        pool_fp = torch.tensor(
            np.array([np.array(fp) for fp in fingerprints])[pool_idx]
        ).float().to(device)
        preds = likelihood(model(pool_fp))
        ucb = alpha * preds.mean + beta * preds.stddev
        best = torch.argsort(ucb, descending=True)[:batch_size]
        return np.array(pool_idx)[best.cpu().numpy()]


def pi_selection(fingerprints, model, likelihood, batch_size,
                 already_selected, current_best_y, xi=0.01):
    """Probability of Improvement."""
    already_selected = already_selected or []
    model.eval()
    likelihood.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        pool_idx = list(set(range(len(fingerprints))) - set(already_selected))
        device = next(model.parameters()).device
        pool_fp = torch.tensor(
            np.array([np.array(fp) for fp in fingerprints])[pool_idx]
        ).float().to(device)
        preds = likelihood(model(pool_fp))
        Z = (preds.mean - current_best_y - xi) / (preds.stddev + 1e-9)
        pi = Normal(torch.tensor(0.0), torch.tensor(1.0)).cdf(Z)
        best = torch.argsort(pi, descending=True)[:batch_size]
        return np.array(pool_idx)[best.cpu().numpy()]


def ei_selection(fingerprints, model, likelihood, batch_size,
                 already_selected, current_best_y, xi=0.01):
    """Expected Improvement."""
    already_selected = already_selected or []
    model.eval()
    likelihood.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        pool_idx = list(set(range(len(fingerprints))) - set(already_selected))
        device = next(model.parameters()).device
        pool_fp = torch.tensor(
            np.array([np.array(fp) for fp in fingerprints])[pool_idx]
        ).float().to(device)
        preds = likelihood(model(pool_fp))
        mean, std = preds.mean, preds.stddev
        Z = (mean - current_best_y - xi) / (std + 1e-9)
        normal = Normal(torch.tensor(0.0), torch.tensor(1.0))
        ei = (mean - current_best_y - xi) * normal.cdf(Z) + std * torch.exp(
            normal.log_prob(Z))
        best = torch.argsort(ei, descending=True)[:batch_size]
        return np.array(pool_idx)[best.cpu().numpy()]

# ═══════════════════════════════════════════════════════════════════════════════
# METRICS
# ═══════════════════════════════════════════════════════════════════════════════

def calculate_metrics(model, likelihood, test_x, test_y):
    """Compute R² and Spearman correlation."""
    model.eval()
    likelihood.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        preds = likelihood(model(test_x)).mean
    r2 = r2_score(test_y.cpu().numpy(), preds.cpu().numpy())
    sp, _ = spearmanr(test_y.cpu().numpy(), preds.cpu().numpy())
    return r2, sp

# ═══════════════════════════════════════════════════════════════════════════════
# ACTIVE LEARNING LOOP
# ═══════════════════════════════════════════════════════════════════════════════

def active_learning(df, fingerprints, selection_protocol,
                    epochs=150, lr=0.01, lr_decay=0.95,
                    kernel_factory=None):
    """Run the active learning simulation.

    Parameters
    ----------
    df : pd.DataFrame
        Must have columns: 'affinity', 'top_2p', 'top_5p'
    fingerprints : np.ndarray
        ECFP fingerprints, shape (N, 4096)
    selection_protocol : list of (method, batch_size) tuples
    kernel_factory : callable or None
        Returns a gpytorch kernel instance

    Returns
    -------
    cycle_results : list of dicts
    selected_indices : list
    all_predictions : list of np.ndarray
    gp_model, likelihood : trained model objects
    """
    df = df.copy()

    # Ensure top-k flags are clean integers
    for col in ['top_2p', 'top_5p']:
        df[col] = df[col].apply(
            lambda x: 1 if x in [True, 'TRUE', 'True', 'true', 1, '1'] else 0
        )

    total_2p = df['top_2p'].sum()
    total_5p = df['top_5p'].sum()
    print(f"Dataset: {len(df)} cpds | top_2p={total_2p} ({100*total_2p/len(df):.1f}%) "
          f"| top_5p={total_5p} ({100*total_5p/len(df):.1f}%)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    selected_df = pd.DataFrame(columns=df.columns)
    top_2p_count = 0
    top_5p_count = 0
    already_selected = []
    cycle_results = []
    all_predictions = []

    for cycle, (method, batch_size) in enumerate(selection_protocol):
        # ── Selection ──────────────────────────────────────────────────
        if method == "random":
            available = list(set(range(len(df))) - set(already_selected))
            new_idx = np.random.choice(available, size=batch_size, replace=False)
        elif method == "ucb":
            new_idx = ucb_selection(
                fingerprints, gp_model, likelihood, batch_size,
                alpha=1.0, beta=1.0, already_selected=already_selected
            )
        elif method == "explore":
            new_idx = ucb_selection(
                fingerprints, gp_model, likelihood, batch_size,
                alpha=0.0, beta=1.0, already_selected=already_selected
            )
        elif method == "exploit":
            new_idx = ucb_selection(
                fingerprints, gp_model, likelihood, batch_size,
                alpha=1.0, beta=0.0, already_selected=already_selected
            )
        else:
            raise ValueError(f"Unknown method: {method}")

        new_sel = df.iloc[new_idx]
        selected_df = pd.concat([selected_df, new_sel])
        top_2p_count += new_sel['top_2p'].sum()
        top_5p_count += new_sel['top_5p'].sum()
        already_selected.extend(list(new_idx))

        # ── Train ──────────────────────────────────────────────────────
        train_x = torch.tensor(
            np.array([np.array(fp) for fp in fingerprints])[selected_df.index]
        ).float().to(device)
        train_y = torch.tensor(
            selected_df['affinity'].values
        ).float().to(device)

        likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
        kernel = kernel_factory() if kernel_factory else None
        gp_model = GPRegressionModel(
            train_x, train_y, likelihood, kernel=kernel
        ).to(device)
        gp_model, likelihood, _ = train_gp_model(
            train_x, train_y, likelihood, gp_model, epochs, lr, lr_decay
        )

        # ── Evaluate ───────────────────────────────────────────────────
        gp_model.eval()
        likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            all_x = torch.tensor(fingerprints).float().to(device)
            predictions = likelihood(gp_model(all_x)).mean.cpu().numpy()

        all_predictions.append(predictions)

        r2, sp = calculate_metrics(
            gp_model, likelihood, all_x,
            torch.tensor(df['affinity'].values).float().to(device)
        )
        rmse = np.sqrt(np.mean((df['affinity'].values - predictions) ** 2))

        cycle_results.append({
            'cycle': cycle,
            'method': method,
            'top_2p': int(top_2p_count),
            'top_5p': int(top_5p_count),
            'top_2p_recall_cumulative': top_2p_count / total_2p if total_2p else 0,
            'top_5p_recall_cumulative': top_5p_count / total_5p if total_5p else 0,
            'r2': r2,
            'spearman': sp,
            'rmse': rmse,
            'compounds_acquired': len(selected_df),
        })

        print(f"Cycle {cycle} ({method:>8s}): R²={r2:.3f}  Spearman={sp:.3f}  "
              f"RMSE={rmse:.2f}  Acquired={len(selected_df)}  "
              f"Top2%={top_2p_count}/{total_2p}  Top5%={top_5p_count}/{total_5p}")

    return cycle_results, already_selected, all_predictions, gp_model, likelihood

# ═══════════════════════════════════════════════════════════════════════════════
# PLOTTING
# ═══════════════════════════════════════════════════════════════════════════════

def plot_results(results_df, protocol_name, kernel_name, out_dir):
    """Generate recall curves and model-quality plots."""
    x = results_df["compounds_acquired"]

    # ── Recall curves ─────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    colors = {"top_2p_recall_cumulative": "#e74c3c",
              "top_5p_recall_cumulative": "#f39c12"}
    labels = {"top_2p_recall_cumulative": "Top 2%",
              "top_5p_recall_cumulative": "Top 5%"}

    for ax, col in zip(axes, ["top_2p_recall_cumulative", "top_5p_recall_cumulative"]):
        ax.plot(x, results_df[col], "o-", color=colors[col], lw=2, ms=6)
        ax.axhline(1.0, color="gray", ls="--", alpha=0.4, label="Perfect recall")
        ax.set_xlabel("Compounds acquired")
        ax.set_ylabel("Recall")
        ax.set_title(labels[col])
        ax.legend()
        ax.set_ylim(0, 1.1)

    fig.suptitle(f"Protocol: {protocol_name}  |  Kernel: {kernel_name}",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/recall_curves.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → Saved {out_dir}/recall_curves.png")

    # ── Model quality ──────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.5))
    axes[0].plot(x, results_df["r2"], "o-", color="#2980b9", lw=2, ms=6)
    axes[0].set_xlabel("Compounds acquired")
    axes[0].set_ylabel("R²")
    axes[0].set_title("R² over cycles")

    axes[1].plot(x, results_df["spearman"], "o-", color="#27ae60", lw=2, ms=6)
    axes[1].set_xlabel("Compounds acquired")
    axes[1].set_ylabel("Spearman ρ")
    axes[1].set_title("Spearman over cycles")

    axes[2].plot(x, results_df["rmse"], "o-", color="#e74c3c", lw=2, ms=6)
    axes[2].set_xlabel("Compounds acquired")
    axes[2].set_ylabel("RMSE")
    axes[2].set_title("RMSE over cycles")

    plt.tight_layout()
    plt.savefig(f"{out_dir}/model_quality.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → Saved {out_dir}/model_quality.png")


def predict_pool(gp_model, likelihood, pool_smiles, lower_is_better, out_dir):
    """Predict activity for a list of unlabeled SMILES using the trained GP."""
    print(f"\nPredicting {len(pool_smiles)} unlabeled compounds...")
    pool_fp = smiles_to_ecfp8(pool_smiles)
    pool_tensor = torch.tensor(pool_fp).float()
    device = next(gp_model.parameters()).device
    pool_tensor = pool_tensor.to(device)

    gp_model.eval()
    likelihood.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        preds = likelihood(gp_model(pool_tensor))
        pred_mean = preds.mean.cpu().numpy()
        pred_std = preds.stddev.cpu().numpy()

    if lower_is_better:
        pred_value = -pred_mean  # undo negation
    else:
        pred_value = pred_mean

    out = pd.DataFrame({
        "SMILES": pool_smiles,
        "predicted_value": pred_value,
        "uncertainty": pred_std,
    })

    # Sort: best first
    ascending = lower_is_better
    out = out.sort_values("predicted_value", ascending=ascending).reset_index(drop=True)

    out_path = f"{out_dir}/pool_predictions.csv"
    out.to_csv(out_path, index=False)
    print(f"  → Saved {out_path} ({len(out)} compounds)")

    # Show top 20
    top_n = min(20, len(out))
    print(f"\nTop {top_n} predicted compounds:")
    print(out.head(top_n)[["SMILES", "predicted_value", "uncertainty"]].to_string(index=False))
    return out

# ═══════════════════════════════════════════════════════════════════════════════
# PROTOCOL BUILDERS
# ═══════════════════════════════════════════════════════════════════════════════

def build_protocol(name, init_size, batch_size, n_cycles):
    """Build a selection protocol from a name string."""
    protocols = {
        "random":           [("random", init_size)] + [("random", batch_size)] * n_cycles,
        "ucb-balanced":     [("random", init_size)] + [("ucb", batch_size)] * n_cycles,
        "ucb-alternate":    [("random", init_size)] + [
            ("explore" if i % 2 == 0 else "exploit", batch_size) for i in range(n_cycles)
        ],
        "ucb-sandwich":     [("random", init_size)] + (
            [("explore", batch_size)] * 2 +
            [("exploit", batch_size)] * 6 +
            [("explore", batch_size)] * 2
        ),
        "ucb-explore-heavy":[("random", init_size)] + (
            [("explore", batch_size)] * 7 +
            [("exploit", batch_size)] * 3
        ),
        "ucb-exploit-heavy":[("random", init_size)] + (
            [("explore", batch_size)] * 3 +
            [("exploit", batch_size)] * 7
        ),
        "ucb-gradual":      [("random", init_size)] + (
            [("explore", batch_size)] * 3 +
            [("ucb", batch_size)] * 4 +
            [("exploit", batch_size)] * 3
        ),
    }
    return protocols.get(name, protocols["ucb-balanced"])

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 72)
    print("  Active Learning for Drug Discovery")
    print("=" * 72)

    # ── 1. Load data ────────────────────────────────────────────────────────
    print(f"\n[1/6] Loading data: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    df = df.rename(columns={SMILES_COL: "SMILES"})
    df = df.dropna(subset=["SMILES", VALUE_COL]).reset_index(drop=True)
    print(f"      {len(df)} compounds loaded")

    # Handle direction
    if LOWER_IS_BETTER:
        df["affinity"] = -df[VALUE_COL].values
        print(f"      Negated '{VALUE_COL}' → affinity (lower=better → GP maximises)")
    else:
        df["affinity"] = df[VALUE_COL].values
        print(f"      Using '{VALUE_COL}' as affinity (higher=better)")

    # Clip extreme outliers
    LOW, HIGH = -15, 15
    before = len(df)
    df = df[(df["affinity"] >= LOW) & (df["affinity"] <= HIGH)].reset_index(drop=True)
    if before != len(df):
        print(f"      Clipped outliers: {before} → {len(df)} compounds")

    # Compute top-k flags
    for frac, name in [(0.02, "top_2p"), (0.05, "top_5p")]:
        n = max(1, int(frac * len(df)))
        idx = df["affinity"].nlargest(n).index
        df[name] = 0
        df.loc[idx, name] = 1

    # ── 2. Featurize ────────────────────────────────────────────────────────
    print(f"\n[2/6] Computing ECFP4 fingerprints...")
    fingerprints = smiles_to_ecfp8(df["SMILES"].tolist())
    print(f"      Shape: {fingerprints.shape}")

    # ── 3. Build protocol + kernel ──────────────────────────────────────────
    print(f"\n[3/6] Protocol: {PROTOCOL}  |  Kernel: {KERNEL}")
    protocol = build_protocol(PROTOCOL, INITIAL_SIZE, BATCH_SIZE, N_CYCLES)

    kernels = {
        "tanimoto": lambda: TanimotoKernel(),
        "rbf": lambda: gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel()),
        "matern": lambda: gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(nu=2.5)
        ),
    }
    kernel_fn = kernels.get(KERNEL, kernels["tanimoto"])

    # ── 4. Run AL ───────────────────────────────────────────────────────────
    print(f"\n[4/6] Running active learning ({len(protocol)} cycles)...")
    print("-" * 72)
    results, selected_idx, all_preds, gp_model, likelihood = active_learning(
        df, fingerprints, protocol,
        epochs=EPOCHS, lr=LR, lr_decay=LR_DECAY,
        kernel_factory=kernel_fn,
    )
    print("-" * 72)

    results_df = pd.DataFrame(results)

    # ── 5. Save + plot ──────────────────────────────────────────────────────
    print(f"\n[5/6] Saving results to {OUT_DIR}/")
    os.makedirs(OUT_DIR, exist_ok=True)

    results_df.to_csv(f"{OUT_DIR}/cycle_results.csv", index=False)

    # Selected compounds
    selected = df.iloc[selected_idx].copy()
    selected.to_csv(f"{OUT_DIR}/selected_compounds.csv", index=False)

    print(f"\nFinal results:")
    print(results_df[["cycle", "method", "compounds_acquired",
                       "r2", "spearman", "rmse",
                       "top_2p", "top_5p"]].to_string(index=False))

    plot_results(results_df, PROTOCOL, KERNEL, OUT_DIR)

    # ── 6. Predict on pool (optional) ───────────────────────────────────────
    if PREDICT_PATH and os.path.exists(PREDICT_PATH):
        print(f"\n[6/6] Predicting on unlabeled pool: {PREDICT_PATH}")
        pool_df = pd.read_csv(PREDICT_PATH)
        pool_df = pool_df.rename(columns={PREDICT_SMILES_COL: "SMILES"})
        pool_df = pool_df.dropna(subset=["SMILES"]).reset_index(drop=True)
        pool_smiles = pool_df["SMILES"].tolist()
        predict_pool(gp_model, likelihood, pool_smiles, LOWER_IS_BETTER, OUT_DIR)
    else:
        print(f"\n[6/6] No PREDICT_PATH set — skipping pool predictions.")

    print(f"\n{'=' * 72}")
    print(f"  ✅ Done! All results in: {OUT_DIR}/")
    print(f"{'=' * 72}")


if __name__ == "__main__":
    main()
