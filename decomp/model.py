"""Products of one model under ROOT (docs/layout.md)."""
import json
from functools import cached_property
from pathlib import Path

import numpy as np
import pandas as pd

from . import tda

METHODS = ("graddot", "gradcos", "tracin", "trak_lam1", "trak_lam0.1", "trak_lam0.01", "trak_ens", "logra", "factgrass")


class Model:
    def __init__(self, root, name):
        self.root, self.name = Path(root), name
        self.dir = self.root / name

    def run(self, seed=0):
        return json.loads((self.dir / f"finetune/seed{seed}/run.json").read_text())

    @cached_property
    def final(self):
        return self.run()["total_steps"]

    def outputs(self):
        return pd.read_csv(self.dir / "outputs/outputs.csv")

    def labels(self, column):
        names, codes = np.unique(self.outputs()[column].to_numpy(), return_inverse=True)
        return codes, names

    def phi(self, seed, step):
        return np.load(self.dir / f"grads/catalogue/seed{seed}_step{step}/phi.npy", mmap_mode="r")

    def psi(self, seed, step):
        return np.load(self.dir / f"grads/outputs/seed{seed}_step{step}/psi.npy")

    def seg_ids(self):
        rows = self.dir / f"grads/catalogue/seed0_step{self.final}/rows.csv"
        return pd.read_csv(rows, dtype={"seg_id": str}).seg_id.to_numpy()

    def scores(self, method):
        f = self.dir / f"scores/{method}.npy"
        if not f.exists():
            f.parent.mkdir(parents=True, exist_ok=True)
            np.save(f, tda.compute(self, method).astype(np.float32))
        return np.load(f).astype(np.float64)

    def features(self):
        return dict(np.load(self.dir / "features/features.npz"))

    def analysis(self, name):
        return self.root / "analysis" / self.name / f"{name}.json"
