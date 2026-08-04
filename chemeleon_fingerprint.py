"""CheMeleon molecular fingerprints (MIT licensed; Jackson Burns, 2025)."""
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import torch
from chemprop import featurizers, nn
from chemprop.data import BatchMolGraph
from chemprop.models import MPNN
from chemprop.nn import RegressionFFN
from rdkit.Chem import Mol, MolFromSmiles


class CheMeleonFingerprint:
    def __init__(self, device: str | torch.device | None = None):
        self.featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
        checkpoint = Path.home() / ".chemprop" / "chemeleon_mp.pt"
        checkpoint.parent.mkdir(exist_ok=True)
        if not checkpoint.exists():
            urlretrieve(
                "https://zenodo.org/records/15460715/files/chemeleon_mp.pt",
                checkpoint,
            )
        state = torch.load(checkpoint, weights_only=True)
        message_passing = nn.BondMessagePassing(**state["hyper_parameters"])
        message_passing.load_state_dict(state["state_dict"])
        self.model = MPNN(
            message_passing=message_passing,
            agg=nn.MeanAggregation(),
            predictor=RegressionFFN(input_dim=message_passing.output_dim),
        )
        self.model.eval()
        if device is not None:
            self.model.to(device=device)

    def __call__(self, molecules: list[str | Mol]) -> np.ndarray:
        graph = BatchMolGraph([
            self.featurizer(MolFromSmiles(m) if isinstance(m, str) else m)
            for m in molecules
        ])
        graph.to(device=self.model.device)
        with torch.no_grad():
            return self.model.fingerprint(graph).numpy(force=True)
