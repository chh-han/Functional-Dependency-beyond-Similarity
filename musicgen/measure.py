"""Mean per-token log-likelihood of every output under a deletion run."""
import argparse
import os

import pandas as pd
import torch

from musicgen.common import HF_NAMES, item_loss, load, outputs, run_dir


@torch.no_grad()
def values(model, proc, gen, codes):
    return [-item_loss(model, proc, codes, i, t).item() for i, t in enumerate(gen.prompt)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--model", choices=list(HF_NAMES), required=True)
    ap.add_argument("--deletion", required=True, help="<method>/<set>")
    ap.add_argument("--seed", type=int, required=True)
    a = ap.parse_args()

    model, proc, _ = load(run_dir(a.root, a.model, a.seed, a.deletion))
    method, name = a.deletion.split("/")
    gen, codes = outputs(a.root, a.model)
    out = os.path.join(a.root, a.model, "deletion", method, "measure")
    os.makedirs(out, exist_ok=True)
    pd.DataFrame(dict(set=name, seed=a.seed, gen_id=gen.gen_id, value=values(model, proc, gen, codes))).to_csv(
        os.path.join(out, f"{name}_seed{a.seed}.csv"), index=False)


if __name__ == "__main__":
    main()
