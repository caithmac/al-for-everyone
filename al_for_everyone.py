#!/usr/bin/env python
"""
Active Learning for Drug Discovery
===================================
Self-contained script for chemists — choose your model, kernel, and protocol.

QUICK START:
  python al_for_everyone.py --data my_data.csv --model gp

  # Or use a config file (recommended for repeatability):
  python al_for_everyone.py @config.txt

Full options: python al_for_everyone.py --help

Author: MJ / Satya — May 2026
"""
import os, sys, argparse, subprocess, warnings, numpy as np, pandas as pd
from tqdm import tqdm
warnings.filterwarnings("ignore")
from rdkit import RDLogger; RDLogger.DisableLog("rdApp.*")
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from scipy.stats import spearmanr

# ──────────────────────────────────────────────────────────────────────────────
# KERNELS
# ──────────────────────────────────────────────────────────────────────────────

class TanimotoKernel:
    """Tanimoto (Jaccard) similarity for binary fingerprint vectors."""
    @staticmethod
    def apply(x1, x2):
        import torch
        x1n = x1.pow(2).sum(-1, keepdim=True)
        x2n = x2.pow(2).sum(-1, keepdim=True)
        dot = torch.matmul(x1, x2.transpose(-1, -2))
        return dot / (x1n + x2n.transpose(-1, -2) - dot).clamp(min=1e-9)

def get_gp_kernel(name):
    """Return a gpytorch kernel by name."""
    import torch, gpytorch
    if name == "tanimoto":
        class TK(gpytorch.kernels.Kernel):
            def forward(self, x1, x2, diag=False, **params):
                if diag: return torch.ones_like(x1[:, 0])
                return TanimotoKernel.apply(x1, x2)
        return gpytorch.kernels.ScaleKernel(TK())
    elif name == "rbf":
        return gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
    elif name == "matern":
        return gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=2.5))
    raise ValueError(f"Unknown kernel: {name}")

# ──────────────────────────────────────────────────────────────────────────────
# GP MODEL
# ──────────────────────────────────────────────────────────────────────────────

def train_gp(train_x, train_y, kernel_name, epochs, lr, decay, device):
    """Train an Exact GP. Returns (model, likelihood) in eval mode."""
    import torch, gpytorch
    from gpytorch.distributions import MultivariateNormal

    class GPModel(gpytorch.models.ExactGP):
        def __init__(self, tx, ty, lik):
            super().__init__(tx, ty, lik)
            self.mean = gpytorch.means.ConstantMean()
            self.covar = get_gp_kernel(kernel_name)
        def forward(self, x):
            return MultivariateNormal(self.mean(x), self.covar(x).add_jitter(1e-6))

    tx = torch.tensor(train_x).float().to(device)
    ty = torch.tensor(train_y).float().to(device)
    lik = gpytorch.likelihoods.GaussianLikelihood().to(device)
    model = GPModel(tx, ty, lik).to(device)

    model.train(); lik.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=decay)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(lik, model)

    for _ in range(epochs):
        opt.zero_grad()
        loss = -mll(model(tx), ty)
        loss.backward(); opt.step(); sched.step()

    model.eval(); lik.eval()
    return model, lik

# ──────────────────────────────────────────────────────────────────────────────
# FINGERPRINTS
# ──────────────────────────────────────────────────────────────────────────────

def ecfp4(smiles_list):
    """ECFP4 fingerprints as numpy array (N x 4096, int8)."""
    fps = []
    for smi in tqdm(smiles_list, desc="  ECFP4"):
        mol = Chem.MolFromSmiles(smi)
        arr = np.zeros(4096, dtype=np.int8)
        if mol:
            AllChem.DataStructs.ConvertToNumpyArray(
                AllChem.GetMorganFingerprintAsBitVect(mol, 4, nBits=4096), arr)
        fps.append(arr)
    return np.vstack(fps)

def chemeleon_fingerprints(smiles_list):
    """CheMeleon foundation-model fingerprints via chemprop v2.2+."""
    import torch
    here = os.path.abspath(os.path.dirname(__file__))
    sys.path.insert(0, here)
    try:
        from chemeleon_fingerprint import CheMeleonFingerprint
    except ImportError as exc:
        raise ImportError("CheMeleon requires chemprop>=2.2.0; use the chemeleon environment") from exc
    fp_gen = CheMeleonFingerprint(device=torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"))
    bs = 1024
    return np.concatenate([fp_gen(smiles_list[i:i+bs])
                           for i in range(0, len(smiles_list), bs)]).astype(np.float32)

# ──────────────────────────────────────────────────────────────────────────────
# PROTOCOL BUILDER
# ──────────────────────────────────────────────────────────────────────────────

def build_protocol(name, init, batch, rounds):
    """Generate (method, batch_size) steps from a protocol name."""
    base = {
        "random":            ["random"] * (rounds + 1),
        "ucb-balanced":      ["random"] + ["ucb"] * rounds,
        "ucb-alternate":     ["random"] + [("explore" if i%2==0 else "exploit")
                                           for i in range(rounds)],
        "ucb-sandwich":      ["random"] + ["explore"]*2 + ["exploit"]*6 + ["explore"]*2,
        "ucb-explore-heavy": ["random"] + ["explore"]*7 + ["exploit"]*3,
        "ucb-exploit-heavy": ["random"] + ["explore"]*3 + ["exploit"]*7,
        "ucb-gradual":       ["random"] + ["explore"]*3 + ["ucb"]*4 + ["exploit"]*3,
    }
    ms = base.get(name, base["ucb-alternate"])
    return list(zip(ms, [init] + [batch] * (len(ms) - 1)))

# ──────────────────────────────────────────────────────────────────────────────
# ACTIVE LEARNING
# ──────────────────────────────────────────────────────────────────────────────

def run_al(fingerprints, target, top2_idx, top5_idx, steps, model_name, kernel,
           gp_epochs, gp_lr, gp_lr_decay, rf_trees, seed, device):
    """Run AL loop. Returns (records, selected_indices, final_model, likelihood)."""
    cnt2 = cnt5 = 0
    selected = []
    records = []
    model = lik = rf = None  # will be set after first training

    for rnd, (method, batch) in enumerate(steps):
        avail = [i for i in range(len(target)) if i not in selected]

        if method == "random":
            new_idx = list(np.random.RandomState(seed + rnd).choice(
                avail, batch, replace=False))

        else:
            # Use existing model for UCB prediction
            if model_name == "gp":
                import torch, gpytorch
                with torch.no_grad(), gpytorch.settings.fast_pred_var():
                    ax = torch.tensor(fingerprints[avail]).float().to(device)
                    preds = lik(model(ax))
                    mu = preds.mean.cpu().numpy()
                    sigma = preds.stddev.cpu().numpy()
            else:
                tp = np.array([t.predict(fingerprints[avail])
                               for t in rf.estimators_])
                mu = tp.mean(axis=0); sigma = tp.std(axis=0)

            a = 1.0 if method in ("ucb", "exploit") else 0.0
            b = 1.0 if method in ("ucb", "explore") else 0.0
            scores = a * mu + b * sigma
            new_idx = [avail[int(i)] for i in np.argsort(scores)[-batch:]]

        selected.extend(new_idx)

        # ── Train model on all selected ──
        sx = fingerprints[selected]
        sy = target.iloc[selected].values
        if model_name == "gp":
            model, lik = train_gp(sx, sy, kernel, gp_epochs,
                                  gp_lr, gp_lr_decay, device)
        else:
            rf = RandomForestRegressor(n_estimators=rf_trees, min_samples_leaf=2,
                                       n_jobs=-1, random_state=seed + rnd)
            rf.fit(sx, sy)

        # ── Track recall ──
        cnt2 += sum(1 for i in new_idx if i in top2_idx)
        cnt5 += sum(1 for i in new_idx if i in top5_idx)

        # ── Evaluate on remaining ──
        remaining = sorted(set(range(len(target))) - set(selected))
        if remaining:
            if model_name == "gp":
                import torch, gpytorch
                with torch.no_grad(), gpytorch.settings.fast_pred_var():
                    rfp = torch.tensor(fingerprints[remaining]).float().to(device)
                    rm = lik(model(rfp)).mean.cpu().numpy()
            else:
                rm = rf.predict(fingerprints[remaining])
            rt = target.iloc[remaining].values
            r2 = r2_score(rt, rm)
            sp, _ = spearmanr(rt, rm)
        else:
            r2 = sp = 0.0

        print(f"  R{rnd:2d} {method:>12s}: trained={len(selected):3d}  "
              f"R2={r2:.3f}  rho={sp:.3f}  "
              f"top2={cnt2}/{len(top2_idx)}  top5={cnt5}/{len(top5_idx)}")
        records.append({"round": rnd, "method": method, "trained": len(selected),
                        "r2": round(r2,3), "spearman": round(sp,3),
                        "top2": cnt2, "top5": cnt5})

    # Final model on all selected
    if model_name == "gp":
        final_model, final_lik = train_gp(
            fingerprints[selected], target.iloc[selected].values,
            kernel, gp_epochs, gp_lr, gp_lr_decay, device)
    else:
        final_model = RandomForestRegressor(n_estimators=rf_trees, min_samples_leaf=2,
                                            n_jobs=-1, random_state=seed)
        final_model.fit(fingerprints[selected], target.iloc[selected].values)
        final_lik = None

    return records, selected, final_model, final_lik

# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main():
    import torch
    p = argparse.ArgumentParser(
        description="Active Learning for Drug Discovery",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  # Config files (one flag per line):
  python al_for_everyone.py --config config.txt
  python al_for_everyone.py --data tyk2.csv --model gp --kernel tanimoto \\
    --protocol ucb-alternate --lower=True --library virtual.csv

Config files: one flag per line (e.g. --model gp on its own line).
Run: python al_for_everyone.py @config.txt""")
    p.add_argument("--data", default="my_compounds.csv")
    p.add_argument("--smiles_col", default="SMILES")
    p.add_argument("--val_col", default="calc_DDG_kcal")
    p.add_argument("--lower", default="True")
    p.add_argument("--library", default="")
    p.add_argument("--library_smi_col", default="smiles")
    p.add_argument("--top_n", type=int, default=1000)
    p.add_argument("--model", default="gp", choices=["gp","rf","chemeleon","consensus"])
    p.add_argument("--kernel", default="tanimoto", choices=["tanimoto","rbf","matern"])
    p.add_argument("--protocol", default="ucb-alternate",
                   choices=["random","ucb-balanced","ucb-alternate","ucb-sandwich",
                            "ucb-explore-heavy","ucb-exploit-heavy","ucb-gradual"])
    p.add_argument("--ucb_beta", type=float, default=2.0)
    p.add_argument("--init_size", type=int, default=60)
    p.add_argument("--batch_size", type=int, default=30)
    p.add_argument("--n_rounds", type=int, default=10)
    p.add_argument("--gp_epochs", type=int, default=150)
    p.add_argument("--gp_lr", type=float, default=0.01)
    p.add_argument("--gp_lr_decay", type=float, default=0.95)
    p.add_argument("--rf_trees", type=int, default=500)
    p.add_argument("--out", default="al_results")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--config", default="",
                   help="Config file path (one flag per line)")
    args = p.parse_args()

    # Config file support
    if args.config:
        with open(args.config) as f:
            lines = [l.strip() for l in f if l.strip() and not l.startswith('#')]
        import shlex
        args = p.parse_args(shlex.split(' '.join(lines)) + sys.argv[1:])

    lower_better = args.lower.strip().lower() in ("true","yes","1","t")

    # Reuse the existing single-model pipeline, then combine complete rankings.
    if args.model == "consensus":
        if not args.library:
            p.error("--model consensus requires --library")
        lib_size = len(pd.read_csv(args.library, usecols=[args.library_smi_col]))
        outputs = []
        for model_name in ("gp", "rf", "chemeleon"):
            model_out = os.path.join(args.out, model_name)
            cmd = [sys.executable, __file__]
            for key, value in vars(args).items():
                if key not in {"model", "out", "config", "top_n"} and value not in ("", None):
                    cmd.extend([f"--{key}", str(value)])
            cmd.extend(["--model", model_name, "--out", model_out,
                        "--top_n", str(lib_size)])
            subprocess.run(cmd, check=True)
            pred = pd.read_csv(os.path.join(model_out, "top_candidates.csv"))
            outputs.append(pred.rename(columns={"predicted": model_name}))

        ranked = outputs[0]
        for pred in outputs[1:]:
            ranked = ranked.merge(pred, on="SMILES")
        model_cols = ["gp", "rf", "chemeleon"]
        for col in model_cols:
            std = ranked[col].std()
            ranked[f"{col}_z"] = (ranked[col] - ranked[col].mean()) / (std or 1.0)
        ranked["consensus"] = ranked[[f"{c}_z" for c in model_cols]].mean(axis=1)
        ranked = ranked.sort_values("consensus", ascending=lower_better).reset_index(drop=True)
        os.makedirs(args.out, exist_ok=True)
        ranked.to_csv(os.path.join(args.out, "full_ranked_library.csv"), index=False)
        ranked.head(args.top_n).to_csv(os.path.join(args.out, "top_candidates.csv"), index=False)
        print(f"\nConsensus complete: {args.out}/top_candidates.csv")
        return

    os.makedirs(args.out, exist_ok=True)

    print(f"\n{'='*65}")
    print("  Active Learning for Drug Discovery")
    print(f"{'='*65}")
    print(f"  Data:     {args.data}")
    print(f"  Model:    {args.model.upper()}  |  Kernel: {args.kernel}")
    print(f"  Protocol: {args.protocol}  |  Budget: {args.init_size}+{args.n_rounds}x{args.batch_size}")
    if args.library:
        print(f"  Library:  {args.library}  ->  Top {args.top_n}")
    print(f"{'='*65}")

    # ── Load data ──
    df = pd.read_csv(args.data)[[args.smiles_col, args.val_col]].dropna()
    df.columns = ["SMILES", "value"]
    target = -df["value"] if lower_better else df["value"]
    sort_asc = lower_better
    print(f"\nLoaded: {len(df)} compounds  |  lower_is_better={lower_better}")

    n_top2 = max(1, int(0.02 * len(df)))
    n_top5 = max(1, int(0.05 * len(df)))
    top2_idx = set(target.nlargest(n_top2).index)
    top5_idx = set(target.nlargest(n_top5).index)
    print(f"Top-2%: {n_top2}  |  Top-5%: {n_top5}")

    # ── Fingerprints ──
    print("\nFingerprinting...")
    if args.model == "chemeleon":
        fingerprints = chemeleon_fingerprints(df["SMILES"].tolist())
        print(f"  CheMeleon: {fingerprints.shape}  OK")
    else:
        fingerprints = ecfp4(df["SMILES"].tolist())
        print(f"  ECFP4: {fingerprints.shape}  OK")

    # ── Protocol ──
    steps = build_protocol(args.protocol, args.init_size, args.batch_size, args.n_rounds)
    print(f"\nProtocol: {args.protocol}  |  {sum(b for _,b in steps)} total selections")

    # ── Run AL ──
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.model == "gp" else "cpu"
    records, selected, final_model, final_lik = run_al(
        fingerprints, target, top2_idx, top5_idx, steps,
        args.model, args.kernel, args.gp_epochs, args.gp_lr, args.gp_lr_decay,
        args.rf_trees, args.seed, device)

    # ── Honest test ──
    remaining = sorted(set(range(len(df))) - set(selected))
    if remaining:
        if args.model == "gp":
            import torch, gpytorch
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                rfp = torch.tensor(fingerprints[remaining]).float().to(device)
                rp = final_lik(final_model(rfp)).mean.cpu().numpy()
        else:
            rp = final_model.predict(fingerprints[remaining])
        if lower_better:
            rp = -rp
        rt = df["value"].iloc[remaining].values
        r2 = r2_score(rt, rp)
        sp, _ = spearmanr(rt, rp)
        print(f"\n{'='*65}")
        print(f"  Honest test - {len(remaining)} unseen compounds:")
        print(f"  R2={r2:.4f}  Spearman={sp:.4f}")
        pd.DataFrame({"SMILES": df["SMILES"].iloc[remaining].values,
                      "true": rt, "predicted": rp}).to_csv(
            f"{args.out}/honest_test.csv", index=False)
        print(f"  -> {args.out}/honest_test.csv")

    pd.DataFrame(records).to_csv(f"{args.out}/al_summary.csv", index=False)
    print(f"  -> {args.out}/al_summary.csv")

    # ── Library screening ──
    if args.library and os.path.exists(args.library):
        lib = pd.read_csv(args.library)
        lib_smi = lib[args.library_smi_col].tolist()
        print(f"\nScreening {len(lib):,} compounds...")

        if args.model == "chemeleon":
            fps = chemeleon_fingerprints(lib_smi)
        else:
            fps = ecfp4(lib_smi)

        if args.model == "gp":
            import torch, gpytorch
            gp_batch = 2048
            all_preds = []
            for i in range(0, len(fps), gp_batch):
                with torch.no_grad(), gpytorch.settings.fast_pred_var():
                    bfp = torch.tensor(fps[i:i+gp_batch]).float().to(device)
                    all_preds.append(final_lik(final_model(bfp)).mean.cpu().numpy())
            lpred = np.concatenate(all_preds)
        else:
            lpred = final_model.predict(fps)
        if lower_better:
            lpred = -lpred

        ranked = pd.DataFrame({"SMILES": lib_smi, "predicted": lpred})
        ranked = ranked.sort_values("predicted", ascending=sort_asc).reset_index(drop=True)
        ranked.head(args.top_n).to_csv(f"{args.out}/top_candidates.csv", index=False)
        print(f"  -> {args.out}/top_candidates.csv (top {args.top_n})")
        for i, row in ranked.head(10).iterrows():
            print(f"  {i+1:3d}.  {row['predicted']:7.3f}  {row['SMILES'][:60]}...")

    print(f"\n{'='*65}")
    print("  Done! Send top_candidates.csv to your testing lab.")
    print("=" * 65)

if __name__ == "__main__":
    main()
