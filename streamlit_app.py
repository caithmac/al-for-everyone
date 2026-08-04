"""Local UI for al_for_everyone.py."""
import io
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

import pandas as pd
import streamlit as st


st.set_page_config(page_title="Active Learning for Everyone", layout="wide")
st.title("Active Learning for Everyone")
st.caption("Train on measured compounds, run active learning, and rank an unlabeled library.")

labeled_file = st.file_uploader("Measured/training CSV", type="csv")
library_file = st.file_uploader("Unlabeled library CSV", type="csv")

if labeled_file and library_file:
    labeled = pd.read_csv(labeled_file)
    library = pd.read_csv(library_file)
    left, right = st.columns(2)
    with left:
        st.write(f"Training rows: {len(labeled):,}")
        train_smiles = st.selectbox("Training SMILES column", labeled.columns)
        target_col = st.selectbox("Measured target column", labeled.columns,
                                  index=min(1, len(labeled.columns) - 1))
    with right:
        st.write(f"Library rows: {len(library):,}")
        library_smiles = st.selectbox("Library SMILES column", library.columns)
        lower = st.toggle("Lower target values are better", value=True)

    st.subheader("Model and active-learning settings")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        model = st.selectbox("Model", ["consensus", "gp", "rf", "chemeleon"])
        kernel = st.selectbox("GP kernel", ["tanimoto", "rbf", "matern"])
    with c2:
        protocol = st.selectbox("Selection protocol", [
            "ucb-alternate", "ucb-balanced", "ucb-sandwich",
            "ucb-explore-heavy", "ucb-exploit-heavy", "ucb-gradual", "random",
        ])
        seed = st.number_input("Random seed", min_value=0, value=7)
    with c3:
        init_size = st.number_input("Initial random compounds", min_value=1, value=60)
        batch_size = st.number_input("Compounds per cycle", min_value=1, value=30)
        rounds = st.number_input("AL cycles", min_value=0, value=10)
    with c4:
        gp_epochs = st.number_input("GP training epochs", min_value=1, value=150)
        rf_trees = st.number_input("RF trees", min_value=1, value=500)
        top_n = st.number_input("Candidates to return", min_value=1, value=500)

    budget = int(init_size + rounds * batch_size)
    st.caption(f"Maximum requested training budget: {budget:,} compounds")
    invalid_target = train_smiles == target_col or not pd.api.types.is_numeric_dtype(labeled[target_col])

    if st.button("Run active learning", type="primary",
                 disabled=budget > len(labeled) or invalid_target):
        with tempfile.TemporaryDirectory(prefix="al_for_everyone_") as tmp:
            tmp_path = Path(tmp)
            train_path = tmp_path / "training.csv"
            library_path = tmp_path / "library.csv"
            output_path = tmp_path / "results"
            labeled.to_csv(train_path, index=False)
            library.to_csv(library_path, index=False)

            command = [
                sys.executable, str(Path(__file__).with_name("al_for_everyone.py")),
                "--data", str(train_path), "--smiles_col", train_smiles,
                "--val_col", target_col, "--lower", str(lower),
                "--library", str(library_path), "--library_smi_col", library_smiles,
                "--model", model, "--kernel", kernel, "--protocol", protocol,
                "--init_size", str(init_size), "--batch_size", str(batch_size),
                "--n_rounds", str(rounds), "--gp_epochs", str(gp_epochs),
                "--rf_trees", str(rf_trees), "--top_n", str(top_n),
                "--seed", str(seed), "--out", str(output_path),
            ]
            with st.spinner("Training models and screening the library..."):
                run = subprocess.run(command, capture_output=True, text=True)

            with st.expander("Run log", expanded=run.returncode != 0):
                st.code(run.stdout + run.stderr)
            if run.returncode:
                st.error("The run failed. See the log above.")
            else:
                result_file = output_path / "top_candidates.csv"
                results = pd.read_csv(result_file)
                st.success(f"Finished. Ranked {len(results):,} candidates.")
                st.dataframe(results, use_container_width=True)
                st.download_button("Download top candidates", result_file.read_bytes(),
                                   "top_candidates.csv", "text/csv")

                archive = io.BytesIO()
                with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zipped:
                    for path in output_path.rglob("*.csv"):
                        zipped.write(path, path.relative_to(output_path))
                st.download_button("Download all results", archive.getvalue(),
                                   "al_results.zip", "application/zip")
    elif budget > len(labeled):
        st.warning("Reduce the initial size, cycles, or batch size to fit the training data.")
    elif invalid_target:
        st.warning("Choose a numeric target column different from the SMILES column.")
else:
    st.info("Upload both CSV files to configure the run.")
